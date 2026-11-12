"""The hard-risk pre-check (v1 → v2 parity audit 2026-11-09, rows 3, 4, 7, 8).

Thresholds below are v1's live values, as fixtures: the trader declares none.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import AS_OF, PARAMS_YAML, TICKERS
from crucible.hard_risk import DRAWDOWN_EXIT, LOCKOUT, LOSS_FLOOR_EXIT, Holding
from crucible.keys import strategy_slot_key
from crucible.portfolio import PortfolioParams, solve_target_weights
from crucible.store import LocalStore
from ib_fakes import FakeIB, FakeSdk, ctx_for

import crucible_trader.daily_session as daily_session
from crucible_trader.construction import construct_target_book
from crucible_trader.execution_shortfall import decide as sizing_decision
from crucible_trader.hard_risk import (
    HARD_RISK_SCHEMA_VERSION,
    BrokerFacts,
    CarriedState,
    HardRiskError,
    HardRiskInputs,
    HardRiskPrecedenceError,
    Lockout,
    carried_state,
    decide,
    enforce_precedence,
    hard_risk_key,
    read_broker_facts,
)
from crucible_trader.order_router import PaperOrderRouter, RouterConfig
from crucible_trader.order_router import PlannedOrder as RouterOrder

V1_YAML = """\
  negative_alpha_exit_below: -0.02
  position_loss_floor_pct: -0.15
  hard_risk_reentry_lockout_sessions: 5
  drawdown_half_size_at: -0.03
  drawdown_quarter_size_at: -0.05
  drawdown_circuit_breaker_at: -0.08
  drawdown_forced_exits_at_half: 1
  drawdown_forced_exits_at_quarter: 2
"""

V1 = {
    "negative_alpha_exit_below": -0.02,
    "position_loss_floor_pct": -0.15,
    "hard_risk_reentry_lockout_sessions": 5,
    "drawdown_half_size_at": -0.03,
    "drawdown_quarter_size_at": -0.05,
    "drawdown_circuit_breaker_at": -0.08,
    "drawdown_forced_exits_at_half": 1,
    "drawdown_forced_exits_at_quarter": 2,
}

BASE = {
    "risk_aversion": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 1.0,
    "min_position_pct": 0.005,
    "covariance_shrinkage": "ledoit_wolf",
    "sigma_horizon_days": 1,
    "ewma_lambda_decay": 0.94,
    "vol_target_annual": None,
    "alpha_uncertainty_penalty": 0.0,
    "alpha_uncertainty_min_cv": 0.01,
    "max_pct_adv": None,
    "max_daily_turnover": 0.02,
    "large_move_turnover_flag": 0.35,
    "conviction_budget_gate_enabled": True,
    "conviction_ir_floor": 0.35,
    "conviction_ir_full": 0.75,
    "conviction_budget_min_multiple": 0.05,
    "conviction_gate_min_names": 3,
    "book_notional_usd": 1_000_000.0,
    "max_position_pct": None,
    "ineligible_held_exit": "discretionary",
}


def _params(**overrides) -> PortfolioParams:
    return PortfolioParams.from_mapping({**BASE, **V1, **overrides}, source="<fixture>")


NO_STATE = CarriedState(peak_nav_usd=None, lockouts={}, carried_from=None)


def _facts(nav: float = 1_000_000.0, **marks: float) -> BrokerFacts:
    """Each named holding at cost 100 marked at ``marks[name]``."""
    return BrokerFacts(
        holdings={t: Holding(shares=100, avg_cost=100.0, mark=m) for t, m in marks.items()},
        nav_usd=nav,
    )


def _order(symbol: str, side: str) -> RouterOrder:
    now = dt.datetime(2026, 11, 9, 19, 0, tzinfo=dt.UTC)
    return RouterOrder(
        trading_day="2026-11-09", decision=sizing_decision(symbol, side, 10, price=10.0, now=now)
    )


# ── decide ──────────────────────────────────────────────────────────────


class TestDecide:
    def test_a_16pct_loser_is_cut_and_locked_for_five_sessions(self) -> None:
        cut = decide(
            _params(),
            HardRiskInputs(_facts(AAA=84.0, BBB=99.0), NO_STATE),
            trading_day="2026-11-06",
            alpha={"AAA": 0.05, "BBB": 0.01},
        )
        assert cut.new_exits == {"AAA": LOSS_FLOOR_EXIT}
        assert cut.forced == {"AAA": LOSS_FLOOR_EXIT}
        assert cut.lockouts["AAA"] == Lockout("2026-11-06", LOSS_FLOOR_EXIT, "2026-11-13")
        assert cut.loss_from_cost["AAA"] == pytest.approx(-0.16)
        assert cut.buy_multiplier == 1.0 and cut.circuit_breaker is False
        assert cut.to_document()["schema_version"] == HARD_RISK_SCHEMA_VERSION

    def test_the_drawdown_tier_forces_the_lowest_conviction_names(self) -> None:
        carried = CarriedState(peak_nav_usd=1_000_000.0, lockouts={}, carried_from="2026-11-05")
        cut = decide(
            _params(),
            HardRiskInputs(_facts(nav=940_000.0, AAA=99.0, BBB=99.0, CCC=99.0), carried),
            trading_day="2026-11-06",
            alpha={"AAA": 0.01, "BBB": -0.01, "CCC": 0.03},
        )
        assert cut.drawdown is not None and cut.buy_multiplier == 0.25
        assert cut.new_exits == {"BBB": DRAWDOWN_EXIT, "AAA": DRAWDOWN_EXIT}
        assert cut.peak_nav_usd == 1_000_000.0

    def test_a_loss_floor_exit_counts_toward_the_tier(self) -> None:
        carried = CarriedState(peak_nav_usd=1_000_000.0, lockouts={}, carried_from="2026-11-05")
        cut = decide(
            _params(),
            HardRiskInputs(_facts(nav=960_000.0, AAA=80.0, BBB=99.0), carried),
            trading_day="2026-11-06",
            alpha={"AAA": 0.05, "BBB": -0.05},
        )
        assert cut.buy_multiplier == 0.5
        assert cut.new_exits == {"AAA": LOSS_FLOOR_EXIT}

    def test_at_the_breaker_buys_are_halted(self) -> None:
        carried = CarriedState(peak_nav_usd=1_000_000.0, lockouts={}, carried_from="2026-11-05")
        cut = decide(
            _params(),
            HardRiskInputs(_facts(nav=900_000.0), carried),
            trading_day="2026-11-06",
            alpha={},
        )
        assert cut.circuit_breaker is True and cut.buy_multiplier == 0.0

    def test_an_expired_lockout_is_dropped_and_a_live_one_is_forced(self) -> None:
        carried = CarriedState(
            peak_nav_usd=1_000_000.0,
            lockouts={
                "OLD": Lockout("2026-10-26", LOSS_FLOOR_EXIT, "2026-11-02"),
                "NEW": Lockout("2026-11-05", LOSS_FLOOR_EXIT, "2026-11-12"),
            },
            carried_from="2026-11-05",
        )
        cut = decide(
            _params(), HardRiskInputs(_facts(), carried), trading_day="2026-11-06", alpha={}
        )
        assert cut.forced == {"NEW": LOCKOUT}
        assert set(cut.lockouts) == {"NEW"}
        assert cut.carried_from == "2026-11-05"

    def test_lockout_off_still_forces_the_cut_today(self) -> None:
        cut = decide(
            _params(hard_risk_reentry_lockout_sessions=None),
            HardRiskInputs(_facts(AAA=80.0), NO_STATE),
            trading_day="2026-11-06",
            alpha={},
        )
        assert cut.forced == {"AAA": LOSS_FLOOR_EXIT} and cut.lockouts == {}

    def test_the_peak_is_tracked_with_the_ladder_off(self) -> None:
        params = _params(**{k: None for k in V1 if k.startswith("drawdown_")})
        carried = CarriedState(peak_nav_usd=1_200_000.0, lockouts={}, carried_from="x")
        cut = decide(params, HardRiskInputs(_facts(), carried), trading_day="2026-11-06", alpha={})
        assert cut.drawdown is None and cut.peak_nav_usd == 1_200_000.0
        assert cut.to_document()["drawdown"] is None

    def test_an_unevaluable_holding_is_a_hard_risk_error(self) -> None:
        facts = BrokerFacts(holdings={"AAA": Holding(1, None, 10.0)}, nav_usd=1.0)
        with pytest.raises(HardRiskError, match="cannot be evaluated"):
            decide(_params(), HardRiskInputs(facts, NO_STATE), trading_day="2026-11-06", alpha={})


# ── carried state ───────────────────────────────────────────────────────


class TestCarriedState:
    def _file(self, store, day: str, **overrides) -> None:
        document = {
            "schema_version": HARD_RISK_SCHEMA_VERSION,
            "peak_nav_usd": 1_000_000.0,
            "lockouts": {
                "AAA": {"exit_day": day, "reason": LOSS_FLOOR_EXIT, "through": "2099-01-01"}
            },
            **overrides,
        }
        store.put_bytes(hard_risk_key(day), json.dumps(document).encode())

    def test_nothing_filed_is_the_first_session(self, tmp_path) -> None:
        assert carried_state(LocalStore(tmp_path), "2026-11-06") == NO_STATE

    def test_the_newest_record_before_the_day_is_read_never_the_day_itself(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        self._file(store, "2026-11-04", peak_nav_usd=1.0)
        self._file(store, "2026-11-05", peak_nav_usd=2.0)
        self._file(store, "2026-11-06", peak_nav_usd=3.0)
        state = carried_state(store, "2026-11-06")
        assert state.carried_from == "2026-11-05" and state.peak_nav_usd == 2.0
        assert state.lockouts["AAA"].reason == LOSS_FLOOR_EXIT

    @pytest.mark.parametrize(
        "overrides",
        [{"schema_version": "x"}, {"lockouts": {"AAA": {}}}, {"peak_nav_usd": 0.0}],
    )
    def test_an_unreadable_record_raises_rather_than_forgetting_lockouts(
        self, tmp_path, overrides
    ) -> None:
        store = LocalStore(tmp_path)
        self._file(store, "2026-11-05", **overrides)
        with pytest.raises(HardRiskError):
            carried_state(store, "2026-11-06")


# ── precedence chokepoint ───────────────────────────────────────────────


def _cut(**kwargs) -> object:
    return decide(
        _params(), HardRiskInputs(_facts(**kwargs), NO_STATE), trading_day="2026-11-09", alpha={}
    )


class TestEnforcePrecedence:
    def test_a_buy_of_a_name_cut_today_refuses_the_whole_plan(self) -> None:
        cut = _cut(AAA=80.0)
        with pytest.raises(HardRiskPrecedenceError, match="AAA"):
            enforce_precedence([(_order("AAA", "sell"),), (_order("AAA", "buy"),)], cut)

    def test_sells_and_unrelated_buys_pass(self) -> None:
        cut = _cut(AAA=80.0)
        batches = [(_order("AAA", "sell"),), (_order("BBB", "buy"),)]
        assert [list(b) for b in enforce_precedence(batches, cut)] == [list(b) for b in batches]

    def test_the_breaker_drops_every_buy_and_keeps_the_sells(self) -> None:
        cut = decide(
            _params(),
            HardRiskInputs(
                _facts(nav=900_000.0, AAA=99.0),
                CarriedState(peak_nav_usd=1_000_000.0, lockouts={}, carried_from="x"),
            ),
            trading_day="2026-11-09",
            alpha={},
        )
        logged: list[str] = []
        sell = _order("AAA", "sell")
        kept = enforce_precedence([(sell,), (_order("BBB", "buy"),)], cut, log=logged.append)
        assert [list(b) for b in kept] == [[sell]]
        assert "BBB" in logged[0]

    def test_the_layer_off_passes_the_routers_batches_untouched(self) -> None:
        batches: list = []
        assert enforce_precedence(batches, None) is batches


# ── the broker read ─────────────────────────────────────────────────────


class CostedIB(FakeIB):
    def __init__(self, *, avg_costs: dict[str, float], **kwargs) -> None:
        super().__init__(**kwargs)
        self.avg_costs = avg_costs

    def positions(self):
        rows = super().positions()
        for row in rows:
            row.avgCost = self.avg_costs[row.contract.symbol]
        return rows


def test_read_broker_facts_values_cost_marks_and_nav() -> None:
    ib = CostedIB(
        positions={"AAA": 100, "BBB": 10}, cash=5_000.0, avg_costs={"AAA": 50.0, "BBB": 20.0}
    )
    facts = read_broker_facts(ib, trading_day="2026-11-09", quote={"AAA": 42.0, "BBB": 25.0}.get)
    assert facts.holdings["AAA"] == Holding(shares=100, avg_cost=50.0, mark=42.0)
    assert facts.nav_usd == pytest.approx(5_000.0 + 4_200.0 + 250.0)


def test_read_broker_facts_refuses_an_unmarked_holding() -> None:
    ib = CostedIB(positions={"AAA": 1}, avg_costs={"AAA": 1.0})
    with pytest.raises(HardRiskError, match="AAA"):
        read_broker_facts(ib, trading_day="2026-11-09", quote=lambda _s: None)


def test_the_paper_router_reads_facts_through_its_own_quote(tmp_path) -> None:
    ib = CostedIB(positions={"AAA": 10}, cash=100.0, avg_costs={"AAA": 10.0})
    config = RouterConfig(
        enabled=True,
        account="DU1",
        address=daily_session.GatewayAddress(host="127.0.0.1", port=4002, client_id=7),
        max_order_notional_usd=1.0,
        max_session_notional_usd=1.0,
    )
    router = PaperOrderRouter(
        config,
        client=ib,
        sdk=FakeSdk(ib),
        store=LocalStore(tmp_path),
        trading_day="2026-11-09",
        clock=ib.clock,
        quote=lambda _s: 8.0,
    )
    facts = router.hard_risk_facts()
    assert facts.nav_usd == pytest.approx(180.0)
    assert daily_session._hard_risk_facts(router) == facts  # noqa: SLF001
    assert daily_session._hard_risk_facts(object()) is None  # noqa: SLF001


# ── the session files the record and enforces precedence ────────────────


def test_session_cycle_files_the_record_before_any_order_and_refuses_a_forced_buy(
    tmp_path,
) -> None:
    from crucible.release import TRADER_PIN_KEY
    from test_session import HEALTHY, SHA, VERSION, IbBrokerControl, _contract, _pin_document
    from test_session import FakeIB as SessionIB  # the session tests' own fakes

    from crucible_trader.session import session_cycle

    store = LocalStore(tmp_path)
    store.put_bytes(TRADER_PIN_KEY, json.dumps(_pin_document(SHA)).encode())
    seen: dict = {}
    cut = _cut(AAA=80.0)

    def construct(store_, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            s_champion="s:arm:def",
            metric={"name": "portfolio_construction"},
            alpha_bench=None,
            hard_risk=cut,
        )

    sent: list = []
    router = SimpleNamespace(
        batches=lambda book: [(_order("AAA", "buy"),)], send=lambda batch: sent.append(batch)
    )
    ib = SessionIB()
    ctx = ctx_for(store, "trader.session")
    with pytest.raises(HardRiskPrecedenceError):
        session_cycle(
            ctx,
            broker=IbBrokerControl(ib, FakeSdk(ib)),
            router=router,
            installed_version=VERSION,
            previous_weights=None,
            clock=ib.clock,
            contract_resolver=_contract(HEALTHY),
            construct=construct,
            hard_risk_facts=_facts(AAA=80.0),
        )
    assert sent == []
    assert isinstance(seen["hard_risk"], HardRiskInputs)
    filed = json.loads(store.get_bytes(hard_risk_key("2026-09-08")))
    assert filed["forced"] == {"AAA": LOSS_FLOOR_EXIT}


# ── the trader's own construction applies the layer ─────────────────────


def _world_with_hard_risk(world) -> None:
    world.store.put_bytes(strategy_slot_key("s"), (PARAMS_YAML + V1_YAML).encode("utf-8"))
    world.record_session(world.arm_id)


def test_construct_target_book_forces_out_a_16pct_loser(world) -> None:
    _world_with_hard_risk(world)
    day = AS_OF.isoformat()
    now = dt.datetime(2026, 9, 11, 20, 30, tzinfo=dt.UTC)
    first = construct_target_book(world.store, trading_day=day, previous_weights=None, now=now)
    held = {t: w for t, w in first.weights.items() if w > 0.0 and t in TICKERS}
    loser = max(held, key=held.__getitem__)
    previous = {**held, "__CASH__": 1.0 - sum(held.values())}
    facts = BrokerFacts(
        holdings={
            t: Holding(shares=100, avg_cost=100.0, mark=84.0 if t == loser else 100.0) for t in held
        },
        nav_usd=1_000_000.0,
    )
    book = construct_target_book(
        world.store,
        trading_day=day,
        previous_weights=previous,
        now=now,
        hard_risk=HardRiskInputs(facts, NO_STATE),
    )
    assert book.weights[loser] == 0.0
    assert book.hard_risk is not None and book.hard_risk.new_exits == {loser: LOSS_FLOOR_EXIT}
    without = construct_target_book(
        world.store, trading_day=day, previous_weights=previous, now=now
    )
    assert without.weights[loser] > 0.0  # the turnover budget alone would have kept it


# ── replay: cut at -16%, not rebought for five sessions ─────────────────


def test_replay_a_16pct_loser_is_forced_out_and_not_rebought_for_five_sessions(
    tmp_path,
) -> None:
    """Seven sessions of the trader's hard-risk loop over the shared engine.

    AAA is held at 40% with the top alpha and marks 16% below cost on 11-06.
    Each session: read the carried state, decide on the broker facts, file the
    record, construct through `crucible.portfolio` with the forced mask, and
    put any BUY of a forced name through the precedence chokepoint. AAA is
    sold whole on 11-06 under a 2% budget, its mark recovers to cost the next
    day, and it still is not bought until 11-16 -- the sixth session after.
    """
    store = LocalStore(tmp_path)
    params = _params(conviction_budget_gate_enabled=False)
    tickers = ["AAA", "BBB", "CCC", "DDD", "SPY", "__CASH__"]
    alpha = np.array([0.08, 0.01, 0.01, 0.01, 0.0, -0.0001])
    rng = np.random.default_rng(3)
    panel = rng.normal(0.0, 0.01, size=(260, 6))
    panel[:, 5] = 0.0
    from crucible.portfolio import CostModel

    flat = CostModel(
        name="flat_bps_v0",
        placeholder=True,
        params={"half_spread_bps": 0.0, "commission_bps": 0.0, "slippage_bps": 5.0},
    )
    w_prev = np.array([0.40, 0.20, 0.20, 0.17, 0.0, 0.03])
    days = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"]
    days += ["2026-11-13", "2026-11-16"]
    aaa: list[float] = []
    for i, day in enumerate(days):
        marks = {t: 100.0 for t in tickers[:4] if w_prev[tickers.index(t)] > 0}
        if "AAA" in marks:
            marks["AAA"] = 84.0 if i == 0 else 100.0
        cut = decide(
            params,
            HardRiskInputs(_facts(**marks), carried_state(store, day)),
            trading_day=day,
            alpha=dict(zip(tickers, alpha, strict=True)),
        )
        store.put_bytes(hard_risk_key(day), json.dumps(cut.to_document()).encode())
        result = solve_target_weights(
            tickers,
            alpha,
            panel,
            w_prev,
            ["a", "b", "c", "d", "__benchmark__", "__cash__"],
            np.ones(6),
            np.ones(6, dtype=bool),
            4,
            5,
            params,
            flat,
            hard_exit=np.array([t in cut.forced for t in tickers]),
            buy_multiplier=cut.buy_multiplier,
        )
        if 1 <= i <= 5:
            # The optimizer is not the only gate: a BUY of AAA is refused outright.
            with pytest.raises(HardRiskPrecedenceError):
                enforce_precedence([(_order("AAA", "buy"),)], cut)
        w_prev = result.weights
        aaa.append(float(w_prev[0]))
    assert aaa[0] == pytest.approx(0.0, abs=1e-8)
    assert aaa[1:6] == pytest.approx([0.0] * 5, abs=1e-8)
    assert aaa[6] > 0.0
    last = json.loads(store.get_bytes(hard_risk_key(days[-1])))
    assert last["lockouts"] == {} and last["carried_from"] == days[-2]


def test_lockout_window_dataclass_round_trips() -> None:
    lock = Lockout("2026-11-06", LOSS_FLOOR_EXIT, "2026-11-13")
    assert dataclasses.asdict(lock) == lock.to_document()
