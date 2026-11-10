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
import pathlib

import pandas as pd
import pytest
from crucible.execution import (
    FILL_SOURCE_SIMULATED_CLOSE,
    reduce_execution_window,
    validate_execution_shortfall,
)
from crucible.keys import data_panel_key, execution_shortfall_key
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
    assert_registry_forbids_routing,
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

    def test_the_fleet_registry_denies_staging_every_route(self) -> None:
        """nousergon-lib#499: `environments.staging.trader.routing_allowed` is []."""
        assert_registry_forbids_routing()

    def test_a_registry_that_would_let_staging_route_is_refused(self) -> None:
        class Allows:
            def assert_routing_allowed(self, route: str) -> None:
                return None

        names: list[str] = []

        def resolve(name: str) -> Allows:
            names.append(name)
            return Allows()

        with pytest.raises(StagingRoutingRefusedError, match="allows staging to route"):
            assert_registry_forbids_routing(resolve)
        assert names == ["staging"]

    def test_every_staging_session_asks_the_registry(self, monkeypatch) -> None:
        asked: list[None] = []
        monkeypatch.setattr(module, "assert_registry_forbids_routing", lambda: asked.append(None))
        for flag in ("on", "off"):
            StagingSessionConfig.from_env(_staging(**{SIMULATED_FILLS_VAR: flag}))
        assert len(asked) == 2

    def test_outside_staging_the_registry_is_never_read(self, monkeypatch) -> None:
        monkeypatch.setattr(
            module,
            "assert_registry_forbids_routing",
            lambda: pytest.fail("the prod path read the registry"),
        )
        for env in ({}, {"NE_ENV": "prod"}, {"NE_ENV": "prod", SIMULATED_FILLS_VAR: "off"}):
            assert StagingSessionConfig.from_env(env) is None

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

    def test_the_shortfall_is_computed_and_says_its_fills_were_simulated(self, store) -> None:
        """`alpha-engine-config-I12685` option (a): a first-class harness record."""
        router = _router(store)
        _run(router, _book())

        shortfall = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=NOW)
        validate_execution_shortfall(shortfall)
        assert shortfall["outcome"] == "computed"
        assert shortfall["fill_source"] == FILL_SOURCE == FILL_SOURCE_SIMULATED_CLOSE
        assert [o["shortfall_bps"] for o in shortfall["orders"]] == [0.0, 0.0]  # type: ignore[union-attr]

    def test_nothing_simulated_is_ever_pooled_as_a_measured_shortfall(self, store) -> None:
        """The harness's own reducer names the session and pools none of its orders."""
        router = _router(store)
        _run(router, _book())
        shortfall = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=NOW)
        store.put_bytes(execution_shortfall_key(DAY), json.dumps(shortfall).encode("utf-8"))

        window = reduce_execution_window(store, [DAY])

        assert window.simulated == (DAY,)
        assert window.n_filled == 0
        assert window.weighted_bps is None

    def test_a_session_that_filled_nothing_is_no_orders(self, store) -> None:
        router = _router(store)
        assert router.batches(_book({"AAA": 0.0})) == []

        shortfall = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=NOW)
        validate_execution_shortfall(shortfall)
        assert shortfall["outcome"] == "no_orders"
        assert "fill_source" not in shortfall
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
        router = _router(store, _config(max_session_notional_usd=1_000.0))
        with pytest.raises(RiskCapExceededError, match="per-session cap"):
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


class TestTheProdPathImportsNothingTheRunningReleaseLacks:
    """The box runs the release wheel `trader/release_pin` names, which may
    predate `simulated` and `fill_source` (crucible#445) and the registry's
    `trader.routing_allowed` (nousergon-lib#499). So no module the prod session
    imports may import those at module level; they are reached on the staging
    path only (`alpha-engine-config-I12685`)."""

    STAGING_ONLY = frozenset(
        {
            "FILL_SOURCE_SIMULATED_CLOSE",
            "FILL_SOURCE_BROKER",
            "STAGING_SERVED_SESSION_MODES",
            "served_session_modes",
            "RoutingNotAllowedError",
        }
    )

    def test_no_module_imports_a_staging_only_name_at_module_level(self) -> None:
        package = pathlib.Path(module.__file__).parent
        for path in sorted(package.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                    assert not names & self.STAGING_ONLY, (path.name, names)
                    assert node.module != "nousergon_lib.env", path.name
                    assert not (node.module == "nousergon_lib" and "env" in names), path.name
                elif isinstance(node, ast.Import):
                    assert "nousergon_lib.env" not in {a.name for a in node.names}, path.name


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


class TestTheSimulatedBookIsCarried:
    """alpha-engine-config-I13305: staging opens each session from the simulated
    book the last served session closed on, not from cash."""

    NEXT = "2026-09-09"

    def _close(self, router: SimulatedFillRouter) -> None:
        store_key = module.simulated_positions_key(router._day)  # noqa: SLF001
        router._store.put_bytes(  # noqa: SLF001
            store_key,
            router.simulated_positions_payload(trading_day=router._day, now=NOW),  # noqa: SLF001
        )

    def test_with_no_earlier_book_the_session_opens_all_cash(self, store) -> None:
        router = _router(store)
        assert router.opening == {}
        assert router.held_weights(100_000.0) == {"__CASH__": 1.0}

    def test_the_closing_book_is_the_opening_plus_the_fills(self, store) -> None:
        router = _router(store)
        _run(router, _book())
        assert router.closing_positions() == {"AAA": 300, "BBB": 400}
        document = json.loads(router.simulated_positions_payload(trading_day=DAY, now=NOW))
        assert document["schema_version"] == module.SIMULATED_POSITIONS_SCHEMA_VERSION
        assert document["carried_from"] is None
        assert document["positions"] == {"AAA": 300, "BBB": 400}

    def test_the_next_session_opens_from_it_and_trades_only_the_difference(self, store) -> None:
        first = _router(store)
        _run(first, _book())
        self._close(first)

        write_panel(store, day=self.NEXT)
        second = SimulatedFillRouter(
            _config(), store=store, trading_day=self.NEXT, clock=lambda: NOW
        )
        assert second.opening == {"AAA": 300, "BBB": 400}
        held = second.held_weights(100_000.0)
        assert held["AAA"] == pytest.approx(0.3) and held["BBB"] == pytest.approx(0.2)
        # Same target as yesterday: nothing to trade.
        assert second.batches(_book(day=self.NEXT)) == []
        # Trim AAA to 0.2: one sell of 100 shares, BBB untouched.
        (sells,) = second.batches(_book({"AAA": 0.2, "BBB": 0.2, "CCC": 0.0}, day=self.NEXT))
        assert [(o.decision.symbol, o.decision.side, o.decision.quantity) for o in sells] == [
            ("AAA", "sell", 100)
        ]
        _run(second, _book({"AAA": 0.2, "BBB": 0.2, "CCC": 0.0}, day=self.NEXT))
        document = json.loads(second.simulated_positions_payload(trading_day=self.NEXT, now=NOW))
        assert document["carried_from"] == DAY
        assert document["positions"] == {"AAA": 200, "BBB": 400}

    def test_a_book_filed_for_the_same_day_or_later_is_never_read(self, store) -> None:
        later = SimulatedFillRouter(
            _config(), store=store, trading_day=self.NEXT, clock=lambda: NOW
        )
        write_panel(store, day=self.NEXT)
        _run(later, _book(day=self.NEXT))
        self._close(later)
        same = _router(store)
        _run(same, _book())
        self._close(same)
        # A re-run of DAY opens from what DAY's first run opened from: nothing.
        assert _router(store).opening == {}

    def test_an_unreadable_book_is_refused(self, store) -> None:
        store.put_bytes(
            module.simulated_positions_key("2026-09-04"),
            json.dumps({"schema_version": "other", "positions": {}}).encode(),
        )
        with pytest.raises(module.SimulatedBrokerError, match="cannot be read"):
            _router(store)

    @pytest.mark.parametrize("shares", [-5, 1.5, True])
    def test_a_short_or_fractional_book_is_refused(self, store, shares) -> None:
        store.put_bytes(
            module.simulated_positions_key("2026-09-04"),
            json.dumps(
                {
                    "schema_version": module.SIMULATED_POSITIONS_SCHEMA_VERSION,
                    "positions": {"AAA": shares},
                }
            ).encode(),
        )
        with pytest.raises(module.SimulatedBrokerError, match="long only"):
            _router(store)

    def test_fills_that_would_leave_the_book_short_are_refused(self, store) -> None:
        router = _router(store)
        _run(router, _book())
        buy = router.records[0]
        sell = dataclasses.replace(
            buy,
            order=dataclasses.replace(
                buy.order,
                decision=dataclasses.replace(
                    buy.order.decision, side="sell", quantity=buy.order.decision.quantity * 2
                ),
            ),
        )
        router._records.append(sell)  # noqa: SLF001
        router._records.append(sell)  # noqa: SLF001
        with pytest.raises(module.SimulatedBrokerError, match="short"):
            router.closing_positions()


class TestCarriedSharesFollowSplits:
    """codex cfff90d3: a carried book is valued at the session's close_raw, which
    a split rebases, so the carried shares are rebased with it."""

    NEXT = "2026-09-09"

    def _carry(self, store, positions) -> None:
        store.put_bytes(
            module.simulated_positions_key(DAY),
            json.dumps(
                {
                    "schema_version": module.SIMULATED_POSITIONS_SCHEMA_VERSION,
                    "positions": positions,
                }
            ).encode(),
        )

    def _next(self, store, splits) -> SimulatedFillRouter:
        write_panel(store, day=self.NEXT, closes={**CLOSES, "AAA": 50.0})
        return SimulatedFillRouter(
            _config(), store=store, trading_day=self.NEXT, clock=lambda: NOW, splits=splits
        )

    def test_a_two_for_one_split_doubles_the_carried_shares(self, store) -> None:
        self._carry(store, {"AAA": 301, "BBB": 400})
        router = self._next(store, {self.NEXT: {"AAA": 2.0, "ZZZ": 3.0}}.get)
        assert router.opening == {"AAA": 602, "BBB": 400}
        # Valued at the post-split close, the book is what it was: no re-buy.
        assert router.held_weights(100_000.0)["AAA"] == pytest.approx(0.301)
        document = json.loads(router.simulated_positions_payload(trading_day=self.NEXT, now=NOW))
        assert document["splits"] == [
            {
                "trading_day": self.NEXT,
                "symbol": "AAA",
                "ratio": 2.0,
                "shares_before": 301,
                "shares_after": 602,
            }
        ]

    def test_a_reverse_split_drops_the_fractional_share(self, store) -> None:
        self._carry(store, {"AAA": 301})
        assert self._next(store, {self.NEXT: {"AAA": 0.1}}.get).opening == {"AAA": 30}

    def test_a_day_with_unknown_splits_refuses(self, store) -> None:
        self._carry(store, {"AAA": 300})
        with pytest.raises(SimulatedBrokerError, match="scan for 2026-09-09 is missing"):
            self._next(store, lambda _day: None)

    def test_a_carried_name_split_with_no_ratio_refuses(self, store) -> None:
        self._carry(store, {"AAA": 300})
        with pytest.raises(SimulatedBrokerError, match="no usable ratio"):
            self._next(store, {self.NEXT: {"AAA": None}}.get)

    def test_only_days_after_the_carried_book_through_the_session_are_read(self, store) -> None:
        self._carry(store, {"AAA": 300})
        asked: list[str] = []
        write_panel(store, day="2026-09-14", closes=CLOSES)
        SimulatedFillRouter(
            _config(),
            store=store,
            trading_day="2026-09-14",
            clock=lambda: NOW,
            splits=lambda day: asked.append(day) or {},
        )
        assert asked == ["2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"]

    def test_with_no_scans_configured_nothing_is_rebased_and_it_says_so(
        self, store, monkeypatch
    ) -> None:
        monkeypatch.delenv(module.SPLIT_SCANS_VAR, raising=False)
        self._carry(store, {"AAA": 300})
        router = self._next(store, None)
        assert router.opening == {"AAA": 300}
        document = json.loads(router.simulated_positions_payload(trading_day=self.NEXT, now=NOW))
        assert document["splits"] is None

    def test_the_scans_are_read_from_the_environment(self, monkeypatch) -> None:
        from test_corporate_action_feed import BASE, s3_with, scan

        doc = scan(day=self.NEXT, splits=[{"ticker": "BRK.B", "split_from": 1, "split_to": 3}])
        source = module.scan_split_source(BASE, s3_client=s3_with(doc, day=self.NEXT))
        assert source(self.NEXT) == {"BRK B": 3.0}
        missing = module.scan_split_source(BASE, s3_client=s3_with(doc, day="2026-09-10"))
        assert missing(self.NEXT) is None
        assert module.split_source_from_env({}) is None
        assert callable(module.split_source_from_env({module.SPLIT_SCANS_VAR: BASE}))
