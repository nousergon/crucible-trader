"""The pre-open plan: the day's order book before the open, no order, no router.

Brian 2026-11-09/10 (`alpha-engine-config-I12024`): the book must be built and
saved before 09:30 ET on every trading day.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import pytest
from conftest import PARAMS_YAML
from crucible.keys import manifest_key, strategy_slot_key
from crucible.release import TRADER_PIN_KEY
from crucible.serving import PredictionsFeed
from crucible.slots.strategy import BookUniverse
from crucible.store import LocalStore
from ib_fakes import FakeIB, FakeSdk, environ_for

import crucible_trader.order_book_plan as plan
from crucible_trader.broker_session import BrokerSettingsError
from crucible_trader.construction import TargetBook
from crucible_trader.contract import ContractUnavailable
from crucible_trader.kill_switch import KILL_SWITCH_KEY, halt_document
from crucible_trader.order_book import STAGE_PRE_OPEN, order_book_key
from crucible_trader.order_router import CASH_TICKER, CloseFallbackQuote
from crucible_trader.paper_smoke import RunningWheelMismatchError

SHA = "c" * 40
VERSION = f"0.1.0+g{'c' * 12}"
HEALTHY = {f"T{i}": 0.001 * i for i in range(10)}
DEGENERATE = {f"T{i}": 0.0 for i in range(6)}
#: Tuesday 2026-11-10, 08:40 ET (EST): before the open of a trading day whose
#: last closed session is Monday 2026-11-09.
PLAN_AT = dt.datetime(2026, 11, 10, 13, 40, tzinfo=dt.UTC)
DAY = "2026-11-09"


def _pin() -> bytes:
    return json.dumps(
        {
            "sha": SHA,
            "target": "trader",
            "pinned_at": "2026-11-05T22:00:00Z",
            "smoke_run_id": "01JG0000000000000000000000",
            "smoke_status": "ok",
            "smoke_manifest_key": manifest_key(
                "trader.smoke", "2026-11-05", discriminator=SHA[:12]
            ),
        }
    ).encode()


@pytest.fixture
def store(tmp_path) -> LocalStore:
    s = LocalStore(tmp_path)
    s.put_bytes(TRADER_PIN_KEY, _pin())
    s.put_bytes(strategy_slot_key("s"), PARAMS_YAML.encode("utf-8"))
    return s


def _contract(alphas=HEALTHY):
    feed = PredictionsFeed(
        slot="m",
        trading_day=DAY,
        champion="m:arm:abc",
        feature_version="v1",
        source_key="arms/m/x.json",
        predicted_alpha=alphas,
    )
    return lambda store, day: SimpleNamespace(feed=feed, arm_id=feed.champion, trading_day=day)


def _unavailable(store, day):
    raise ContractUnavailable(f"no feed for {day}")


class _Construct:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, store, **kwargs) -> TargetBook:
        self.calls.append(kwargs)
        tickers = ("AAA", "BBB", "SPY", "$CASH")
        return TargetBook(
            trading_day=kwargs["trading_day"],
            s_champion="s:arm:x",
            m_champion="m:arm:abc",
            u_champion=None,
            universe=BookUniverse(
                tickers=tickers, sectors=("t", "f", "idx", "cash"), benchmark_idx=2, cash_idx=3
            ),
            weights={"AAA": 0.6, "BBB": 0.3, "SPY": 0.0, "$CASH": 0.1},
            turnover_one_way_ratio=0.2,
            cost_bps=3.0,
            portfolio_notional_usd=1_000_000.0,
            evidence={},
            metric={},
            session_inputs_key="k",
            alpha={"AAA": 0.02, "BBB": 0.01},
            eligible={"AAA": True, "BBB": True},
        )


class _Holdings:
    def __init__(self) -> None:
        self.notional: list[float] = []

    def held_weights(self, notional: float) -> dict[str, float]:
        self.notional.append(notional)
        return {"AAA": 0.4, "$CASH": 0.6}


def _plan(store, *, contract=None, holdings=None, construct=None, version=VERSION):
    return plan.plan_order_book(
        store,
        trading_day=DAY,
        installed_version=version,
        holdings=holdings,
        clock=lambda: PLAN_AT,
        contract_resolver=contract or _contract(),
        construct=construct or _Construct(),
    )


def test_a_shadow_plan_writes_the_book_from_all_cash_before_the_open(store) -> None:
    construct = _Construct()
    outcome = _plan(store, construct=construct)
    assert outcome == plan.PlanOutcome(
        DAY,
        "written",
        f"{order_book_key(DAY)} written before the open, from all cash",
        order_book_key(DAY),
    )
    doc = json.loads(store.get_bytes(order_book_key(DAY)))
    assert doc["stage"] == STAGE_PRE_OPEN and doc["previous_book"] == "all_cash"
    assert doc["trading_day"] == DAY and doc["written_at_utc"] == "2026-11-10T13:40:00Z"
    assert construct.calls == [{"trading_day": DAY, "previous_weights": None, "now": PLAN_AT}]


def test_a_routed_plan_starts_from_the_positions_it_reads(store) -> None:
    holdings, construct = _Holdings(), _Construct()
    outcome = _plan(store, holdings=lambda: holdings, construct=construct)
    assert outcome.outcome == "written" and "the account's positions" in outcome.reason
    assert holdings.notional == [1_000_000.0]
    assert construct.calls[0]["previous_weights"] == {"AAA": 0.4, "$CASH": 0.6}
    assert json.loads(store.get_bytes(order_book_key(DAY)))["previous_book"] == "given"


def test_a_rerun_leaves_the_written_book_alone(store) -> None:
    _plan(store)
    before = store.get_bytes(order_book_key(DAY))
    outcome = _plan(store, holdings=lambda: _Holdings())
    assert outcome.outcome == "already_written" and outcome.key == order_book_key(DAY)
    assert store.get_bytes(order_book_key(DAY)) == before


def test_no_contract_yet_is_held_and_never_connects(store) -> None:
    opened: list[int] = []
    outcome = _plan(store, contract=_unavailable, holdings=lambda: opened.append(1))
    assert outcome.outcome == "held" and "no feed" in outcome.reason and outcome.key is None
    assert opened == [] and not store.exists(order_book_key(DAY))


def test_a_degenerate_feed_writes_no_book_and_no_kill_switch(store) -> None:
    outcome = _plan(store, contract=_contract(DEGENERATE))
    assert outcome.outcome == "held" and "hold_signal" in outcome.reason
    assert not store.exists(order_book_key(DAY)) and not store.exists(KILL_SWITCH_KEY)


def test_a_halted_day_writes_no_book(store) -> None:
    store.put_bytes(
        KILL_SWITCH_KEY,
        json.dumps(
            halt_document(
                engaged=True, mode="freeze", cause="drill", trading_day=DAY, since="x", run_id="r"
            )
        ).encode(),
    )
    outcome = _plan(store)
    assert outcome.outcome == "held" and not store.exists(order_book_key(DAY))


def test_a_wheel_the_pin_does_not_name_fails_the_plan(store) -> None:
    with pytest.raises(RunningWheelMismatchError):
        _plan(store, version="9.9.9")
    assert not store.exists(order_book_key(DAY))


# --- The plan's own client id --------------------------------------------------


def test_the_plan_connects_on_its_own_client_id(tmp_path) -> None:
    env = environ_for(tmp_path, CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID="9")
    assert plan.plan_gateway_address(env).client_id == 9
    del env["CRUCIBLE_TRADER_IB_MASTER_CLIENT_ID"]
    assert plan.plan_gateway_address(env).client_id == 9


@pytest.mark.parametrize(
    ("plan_id", "match"),
    [("7", "CRUCIBLE_TRADER_IB_CLIENT_ID"), ("8", "MASTER"), ("2", "v1 executor")],
)
def test_a_shared_client_id_is_refused(tmp_path, plan_id, match) -> None:
    env = environ_for(tmp_path, CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID=plan_id)
    with pytest.raises(BrokerSettingsError, match=match):
        plan.plan_gateway_address(env)


def test_no_plan_client_id_is_refused(tmp_path) -> None:
    with pytest.raises(BrokerSettingsError, match="CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID"):
        plan.plan_gateway_address(environ_for(tmp_path))


# --- Holdings, read-only ---------------------------------------------------------


def test_holdings_are_the_account_positions_at_the_quoted_prices(store) -> None:
    ib = FakeIB(positions={"AAA": 100})
    holdings = plan.IbHoldings(ib, FakeSdk(ib), store=store, trading_day=DAY, quote=lambda s: 50.0)
    assert holdings.held_weights(10_000.0) == {"AAA": 0.5, CASH_TICKER: 0.5}
    assert ib.order_calls == []


def test_holdings_default_to_the_routers_quote(store) -> None:
    ib = FakeIB()
    holdings = plan.IbHoldings(ib, FakeSdk(ib), store=store, trading_day=DAY)
    assert isinstance(holdings._quote, CloseFallbackQuote)  # noqa: SLF001


# --- The entry point ---------------------------------------------------------------


@dataclasses.dataclass
class _Recorded:
    calls: list[dict] = dataclasses.field(default_factory=list)
    holdings: list[object] = dataclasses.field(default_factory=list)

    def __call__(self, store, **kwargs) -> plan.PlanOutcome:
        self.calls.append(kwargs)
        if kwargs["holdings"] is not None:
            self.holdings.append(kwargs["holdings"]())
        return plan.PlanOutcome(kwargs["trading_day"], "written", "ok", "k")


def _main(env, *, at=PLAN_AT, sdk=None, argv=()):
    lines: list[str] = []
    code = plan.main(
        list(argv),
        environ=env,
        sdk_loader=sdk or FakeSdk(FakeIB()),
        version_reader=lambda: VERSION,
        clock=lambda: at,
        printer=lines.append,
    )
    return code, lines


def test_main_plans_a_shadow_day_without_touching_the_gateway(tmp_path, monkeypatch) -> None:
    recorded = _Recorded()
    monkeypatch.setattr(plan, "plan_order_book", recorded)
    ib = FakeIB()
    code, lines = _main(environ_for(tmp_path), sdk=FakeSdk(ib))
    assert code == 0 and lines == [f"order book plan {DAY}: written: ok"]
    assert recorded.calls[0]["trading_day"] == DAY and recorded.calls[0]["holdings"] is None
    assert ib.connected_with is None


def test_main_plans_a_routed_day_read_only_on_the_plan_client_id(tmp_path, monkeypatch) -> None:
    recorded = _Recorded()
    monkeypatch.setattr(plan, "plan_order_book", recorded)
    ib = FakeIB()
    env = environ_for(
        tmp_path,
        CRUCIBLE_TRADER_ORDER_ROUTING="ib_paper",
        CRUCIBLE_TRADER_IB_ACCOUNT="DU1",
        CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD="50000",
        CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD="500000",
        CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID="9",
    )
    code, _ = _main(env, sdk=FakeSdk(ib))
    assert code == 0 and isinstance(recorded.holdings[0], plan.IbHoldings)
    assert ib.connected_with is not None
    assert ib.connected_with["clientId"] == 9 and ib.connected_with["readonly"] is True
    assert ib.disconnected == 1 and ib.order_calls == []


@pytest.mark.parametrize(
    ("at", "says"),
    [
        (dt.datetime(2026, 11, 14, 13, 40, tzinfo=dt.UTC), "not an NYSE session"),  # Saturday
        (dt.datetime(2026, 11, 26, 13, 40, tzinfo=dt.UTC), "not an NYSE session"),  # Thanksgiving
        (dt.datetime(2026, 11, 10, 14, 30, tzinfo=dt.UTC), "not before the 09:30 open"),
        (dt.datetime(2026, 11, 10, 19, 0, tzinfo=dt.UTC), "not before the 09:30 open"),
    ],
)
def test_main_refuses_a_day_off_and_a_start_at_or_after_the_open(
    tmp_path, monkeypatch, at, says
) -> None:
    recorded = _Recorded()
    monkeypatch.setattr(plan, "plan_order_book", recorded)
    code, lines = _main(environ_for(tmp_path), at=at)
    assert code == 0 and says in lines[0] and recorded.calls == []


def test_main_plans_an_early_close_day(tmp_path, monkeypatch) -> None:
    """2026-11-27, the day after Thanksgiving, closes at 13:00 and is planned."""
    recorded = _Recorded()
    monkeypatch.setattr(plan, "plan_order_book", recorded)
    code, _ = _main(environ_for(tmp_path), at=dt.datetime(2026, 11, 27, 13, 40, tzinfo=dt.UTC))
    assert code == 0 and recorded.calls[0]["trading_day"] == "2026-11-25"


def test_main_skips_staging(tmp_path, monkeypatch) -> None:
    recorded = _Recorded()
    monkeypatch.setattr(plan, "plan_order_book", recorded)
    code, lines = _main(environ_for(tmp_path, NE_ENV="staging"))
    assert code == 0 and "staging" in lines[0] and recorded.calls == []


def test_main_takes_no_arguments(tmp_path) -> None:
    with pytest.raises(SystemExit, match="takes no arguments"):
        _main(environ_for(tmp_path), argv=["--run-mode", "live"])


def test_main_disconnects_when_the_plan_fails(tmp_path, monkeypatch) -> None:
    def failing(store, **kwargs):
        kwargs["holdings"]()
        raise RuntimeError("construction refused")

    monkeypatch.setattr(plan, "plan_order_book", failing)
    ib = FakeIB()
    env = environ_for(
        tmp_path,
        CRUCIBLE_TRADER_ORDER_ROUTING="ib_paper",
        CRUCIBLE_TRADER_IB_ACCOUNT="DU1",
        CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD="50000",
        CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD="500000",
        CRUCIBLE_TRADER_PLAN_IB_CLIENT_ID="9",
    )
    with pytest.raises(RuntimeError, match="construction refused"):
        _main(env, sdk=FakeSdk(ib))
    assert ib.disconnected == 1
