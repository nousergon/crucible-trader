"""The staging dry-run broker: simulated fills at the close, and no way to route.

`alpha-engine-config-I12507` (S-19). The properties:

* outside staging the switch is inert when off and REFUSED when on;
* in staging the routing switch is refused whatever its value, before anything opens;
* a simulated session plans through the prod `plan_orders` with the prod caps
  and fills every order in full at the panel close, deterministically;
* nothing simulated is filed as an execution measurement;
* the module cannot reach a broker: it imports no SDK and no connector.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import inspect
import json

import pandas as pd
import pytest
from crucible.execution import validate_execution_shortfall
from crucible.keys import data_panel_key
from crucible.store import LocalStore
from test_order_router import _book, _halt

from crucible_trader import simulated_broker as module
from crucible_trader.broker_session import OrderRefusedError
from crucible_trader.kill_switch import TradingHaltedError
from crucible_trader.order_router import (
    ROUTING_VAR,
    PriceUnavailableError,
    RiskCapExceededError,
    RouterConfig,
    RouterSettingsError,
    RoutingRefusedError,
    StagingRoutingRefusedError,
    assert_staging_never_routes,
    is_staging,
)
from crucible_trader.simulated_broker import (
    FILL_SOURCE,
    SIMULATED_FILLS_SCHEMA_VERSION,
    SIMULATED_FILLS_VAR,
    PanelCloseSource,
    SimulatedBrokerError,
    SimulatedFillRouter,
    SimulatedFillsRefusedError,
    StagingHoldControl,
    StagingSessionConfig,
    simulated_fills_key,
)

DAY = "2026-09-08"
CLOSES = {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0, "SPY": 400.0}
NOW = dt.datetime(2026, 9, 8, 20, 30, tzinfo=dt.UTC)
CAPS = {
    "CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD": "50000",
    "CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD": "100000",
}


def _staging(**extra: str) -> dict[str, str]:
    return {"NE_ENV": "staging", SIMULATED_FILLS_VAR: "on", **CAPS, **extra}


def _config(**overrides) -> StagingSessionConfig:
    config = StagingSessionConfig.from_env(_staging())
    assert config is not None
    return dataclasses.replace(config, **overrides)


def write_panel(store: LocalStore, day: str = DAY, closes: dict[str, float] | None = None) -> None:
    """The day's panel, plus a prior day whose closes must never be read."""
    closes = CLOSES if closes is None else closes
    prior = dt.date.fromisoformat(day) - dt.timedelta(days=1)
    rows = [
        {"trading_day": d, "ticker": t, "close_raw": c * scale, "volume_raw": 1e6}
        for d, scale in ((prior, 0.5), (dt.date.fromisoformat(day), 1.0))
        for t, c in closes.items()
    ]
    store.put_bytes(data_panel_key(day), pd.DataFrame(rows).to_parquet(index=False))


@pytest.fixture
def store(tmp_path) -> LocalStore:
    s = LocalStore(tmp_path)
    write_panel(s)
    return s


def _router(store, config=None) -> SimulatedFillRouter:
    return SimulatedFillRouter(config or _config(), store=store, trading_day=DAY, clock=lambda: NOW)


def _run(router: SimulatedFillRouter, book) -> None:
    for batch in router.batches(book):
        router.send(batch)


class TestTheSwitchOutsideStaging:
    @pytest.mark.parametrize(
        "env",
        [{}, {"NE_ENV": "prod"}, {SIMULATED_FILLS_VAR: "off"}, {SIMULATED_FILLS_VAR: " "}],
    )
    def test_off_outside_staging_is_the_prod_path(self, env) -> None:
        assert StagingSessionConfig.from_env(env) is None

    @pytest.mark.parametrize("env", [{}, {"NE_ENV": "prod"}, {"NE_ENV": "dev"}])
    def test_on_outside_staging_is_refused(self, env) -> None:
        with pytest.raises(SimulatedFillsRefusedError, match="never be filed as a prod"):
            StagingSessionConfig.from_env({**env, SIMULATED_FILLS_VAR: "on", **CAPS})

    @pytest.mark.parametrize("value", ["1", "true", "ON", "yes", "simulated"])
    def test_an_unrecognised_value_is_refused_not_guessed(self, value) -> None:
        for env in ({}, {"NE_ENV": "staging"}):
            with pytest.raises(RouterSettingsError, match="is not a value"):
                StagingSessionConfig.from_env({**env, SIMULATED_FILLS_VAR: value})

    def test_the_process_environment_is_read_when_no_env_is_given(self, monkeypatch) -> None:
        monkeypatch.delenv("NE_ENV", raising=False)
        monkeypatch.setenv(SIMULATED_FILLS_VAR, "on")
        with pytest.raises(SimulatedFillsRefusedError):
            StagingSessionConfig.from_env()


class TestStagingNeverRoutes:
    @pytest.mark.parametrize("value", ["ib_paper", "off", ""])
    def test_the_routing_switch_present_at_all_is_refused(self, value) -> None:
        for flag in ("on", "off"):
            with pytest.raises(StagingRoutingRefusedError, match="Staging never routes"):
                StagingSessionConfig.from_env(
                    _staging(**{SIMULATED_FILLS_VAR: flag, ROUTING_VAR: value})
                )

    @pytest.mark.parametrize("value", ["ib_paper", "off", ""])
    def test_no_entry_point_can_build_a_routing_config_in_staging(self, value) -> None:
        env = {
            "NE_ENV": " Staging ",
            ROUTING_VAR: value,
            "CRUCIBLE_TRADER_IB_ACCOUNT": "DU1",
            "CRUCIBLE_TRADER_IB_PORT": "4002",
            **CAPS,
        }
        with pytest.raises(StagingRoutingRefusedError):
            RouterConfig.from_env(env)

    def test_outside_staging_the_guard_reads_nothing_else(self) -> None:
        assert_staging_never_routes({ROUTING_VAR: "ib_paper"})
        assert_staging_never_routes({"NE_ENV": "prod", ROUTING_VAR: "ib_paper"})
        assert not is_staging({})
        assert is_staging({"NE_ENV": "STAGING"})

    def test_a_staging_session_without_the_switch_is_shadow(self) -> None:
        config = StagingSessionConfig.from_env({"NE_ENV": "staging"})
        assert config == StagingSessionConfig(simulated=False)

    def test_simulated_fills_carry_the_prod_caps(self) -> None:
        assert _config() == StagingSessionConfig(
            simulated=True, max_order_notional_usd=50_000.0, max_session_notional_usd=100_000.0
        )

    @pytest.mark.parametrize("var", sorted(CAPS))
    @pytest.mark.parametrize("value", [None, "0", "-1", "nan", "x"])
    def test_the_caps_are_required_and_positive(self, var, value) -> None:
        env = _staging()
        if value is None:
            del env[var]
        else:
            env[var] = value
        with pytest.raises(RouterSettingsError):
            StagingSessionConfig.from_env(env)

    @pytest.mark.parametrize("field", ["max_order_notional_usd", "max_session_notional_usd"])
    def test_the_caps_are_required_on_direct_construction(self, field) -> None:
        with pytest.raises(RouterSettingsError, match=field):
            _config(**{field: None})


class TestSimulatedFills:
    def test_a_book_fills_in_full_at_the_session_close_sells_first(self, store) -> None:
        router = _router(store)
        _run(router, _book({"AAA": 0.3, "BBB": 0.2, "CCC": 0.0}))

        fills = [(r.order.decision.symbol, r.filled_quantity, r.fill_price) for r in router.records]
        # 30% and 20% of the book's $100k at the day's close -- never the prior day's.
        assert fills == [("AAA", 300.0, 100.0), ("BBB", 400.0, 50.0)]
        assert {r.disposition for r in router.records} == {"simulated"}
        assert all(r.order.decision.decision_price == r.fill_price for r in router.records)

    def test_the_fills_are_deterministic(self, store) -> None:
        first, second = _router(store), _router(store)
        for router in (first, second):
            _run(router, _book())
        assert [r.to_document() for r in first.records] == [r.to_document() for r in second.records]

    def test_nothing_simulated_is_filed_as_an_execution_measurement(self, store) -> None:
        router = _router(store)
        _run(router, _book())

        shortfall = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=NOW)
        validate_execution_shortfall(shortfall)
        assert shortfall["outcome"] == "no_orders"
        assert "2 of 2 planned order(s) filled by simulation" in str(shortfall["outcome_reason"])
        assert "no order reached a broker" in str(shortfall["outcome_reason"])

    def test_the_simulated_fills_document_labels_every_fill(self, store) -> None:
        router = _router(store)
        _run(router, _book())

        document = json.loads(
            router.simulated_fills_payload(champion="m:arm:abc", trading_day=DAY, now=NOW)
        )
        assert document["schema_version"] == SIMULATED_FILLS_SCHEMA_VERSION
        assert document["environment"] == "staging"
        assert document["price_source"] == f"{data_panel_key(DAY)}#close_raw"
        assert document["slippage_bps"] == 0.0
        assert (document["n_planned"], document["n_filled"]) == (2, 2)
        assert {o["fill_source"] for o in document["orders"]} == {FILL_SOURCE}
        assert [o["shortfall_bps"] for o in document["orders"]] == [0.0, 0.0]
        assert simulated_fills_key(DAY) == f"trader/simulated_fills/{DAY}.json"

    def test_the_prod_caps_refuse_the_whole_plan(self, store) -> None:
        router = _router(store, _config(max_order_notional_usd=1_000.0))
        with pytest.raises(RiskCapExceededError, match="per-order cap"):
            router.batches(_book())
        assert router.records == ()

    def test_a_name_the_panel_does_not_close_refuses_the_whole_plan(self, store) -> None:
        write_panel(store, closes={"AAA": 100.0})
        with pytest.raises(PriceUnavailableError, match="BBB"):
            _router(store).batches(_book())

    def test_an_absent_panel_refuses(self, tmp_path) -> None:
        with pytest.raises(PriceUnavailableError, match="panel is absent"):
            _router(LocalStore(tmp_path)).batches(_book())

    def test_two_closes_for_one_name_refuse(self, store) -> None:
        rows = [
            {"trading_day": dt.date(2026, 9, 8), "ticker": "AAA", "close_raw": c} for c in (1, 2)
        ]
        store.put_bytes(data_panel_key(DAY), pd.DataFrame(rows).to_parquet(index=False))
        with pytest.raises(PriceUnavailableError, match="more than one"):
            PanelCloseSource(store, DAY)("AAA")

    def test_the_panel_is_read_once(self, store, monkeypatch) -> None:
        source = PanelCloseSource(store, DAY)
        assert source("AAA") == 100.0
        monkeypatch.setattr(store, "get_bytes", lambda key: pytest.fail("read twice"))
        assert source("BBB") == 50.0
        assert source("ZZZ") is None

    def test_a_book_for_another_day_is_refused(self, store) -> None:
        with pytest.raises(RoutingRefusedError, match="opened for"):
            _router(store).batches(_book(day="2026-09-09"))

    def test_an_engaged_kill_switch_blocks_every_simulated_fill(self, store) -> None:
        router = _router(store)
        (batch,) = router.batches(_book({"AAA": 0.3}))
        _halt(store, "freeze")
        with pytest.raises(TradingHaltedError):
            router.send(batch)
        assert router.records == ()

    def test_a_book_already_flat_plans_nothing(self, store) -> None:
        assert _router(store).batches(_book({"AAA": 0.0})) == []

    def test_simulated_fills_off_builds_no_simulated_router(self, store) -> None:
        with pytest.raises(SimulatedFillsRefusedError, match="off"):
            _router(store, StagingSessionConfig(simulated=False))

    def test_it_reaches_no_broker(self, store) -> None:
        router = _router(store)
        assert router.enabled is False
        assert router.simulated is True
        assert router.close() is None
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
        assert not {"load_sdk", "connect", "open_paper_router", "build_router"} & imported
        assert not {"ib_async", "ib_insync"} & roots


class TestTheStagingHoldControl:
    def test_it_changes_nothing_and_has_no_statement(self) -> None:
        control = StagingHoldControl()
        assert control.open_order_count() == 0
        assert control.orders_accepted_since(NOW) == []
        assert control.wait(1.0) is None
        with pytest.raises(OrderRefusedError):
            control.cancel_all()
        with pytest.raises(OrderRefusedError, match="AAA"):
            control.submit_close("AAA", 10)
        with pytest.raises(SimulatedBrokerError, match="no broker statement"):
            control.statement(DAY)
