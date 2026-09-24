"""Shadow books per registered arm — `alpha-engine-config-I10653`.

The load-bearing assertion is `TestTheSameCostModelTheGradeCharges`: a shadow
book is advanced by `crucible.slots.strategy.construct_book` and charged by
the recipe's own `CostModel.cost_bps_for_trades`, the charge the S grade
subtracts. If this module ever priced a trade itself, those tests fail.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json

import numpy as np
import pytest
from crucible.execution import shadow_book_coverage, shadow_books_key, validate_shadow_books
from crucible.keys import TRADER_EVIDENCE_KEY
from crucible.models import TraderEvidenceDocument
from crucible.portfolio import CostModel, CostModelInputError, PortfolioParams
from crucible.slots.arms import write_register
from crucible.slots.strategy import (
    BookUniverse,
    ExitRuleSpec,
    SessionInputs,
    StrategyRecipe,
    construct_book,
)
from crucible.store import LocalStore
from nousergon_lib.arena.arms import ArmRegister

import crucible_trader.shadow_books as shadow_module
from crucible_trader.shadow_books import (
    ArmSessionInputs,
    ShadowBookFailure,
    advance_book,
    read_previous_books,
    run_shadow_books,
)

NOW = dt.datetime(2026, 9, 11, 21, 0, tzinfo=dt.UTC)
DAY1, DAY2 = "2026-09-10", "2026-09-11"

PARAMS: dict = {
    "risk_aversion": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "min_position_pct": 0.005,
    "covariance_shrinkage": "sample",
    "sigma_horizon_days": 1,
    "ewma_lambda_decay": 0.94,
    "vol_target_annual": None,
    "alpha_uncertainty_penalty": 0.0,
    "alpha_uncertainty_min_cv": 0.01,
    "max_pct_adv": 0.05,
    "max_daily_turnover": 0.20,
    "large_move_turnover_flag": 0.35,
    "conviction_budget_gate_enabled": True,
    "conviction_ir_floor": 0.35,
    "conviction_ir_full": 0.75,
    "conviction_budget_min_multiple": 0.05,
    "conviction_gate_min_names": 3,
    "book_notional_usd": 1_000_000.0,
}
FLAT = {
    "name": "flat_bps_v0",
    "placeholder": True,
    "params": {"half_spread_bps": 2.5, "commission_bps": 0.5, "slippage_bps": 10.0},
}
IMPACT = {
    "name": "sqrt_impact_v1",
    "placeholder": False,
    "params": {
        "half_spread_bps": 2.5,
        "impact_coef_bps": 10.0,
        "commission_bps": 0.5,
        "min_cost_bps": 0.0,
    },
}
NOTIONAL = 4_000_000.0


def _params() -> PortfolioParams:
    return PortfolioParams.from_mapping(PARAMS, source="<fixture>")


def _recipe(cost: dict, name: str = "shadow_fixture") -> StrategyRecipe:
    return StrategyRecipe(
        name=name,
        rules=(ExitRuleSpec(rule_id="profit_take", params={"profit_take_pct": 0.25}),),
        cost_model=CostModel(**cost),
    )


def _universe() -> BookUniverse:
    return BookUniverse(
        tickers=("AAA", "BBB", "SPY", "CASH"),
        sectors=("tech", "health", "__benchmark__", "__cash__"),
        benchmark_idx=2,
        cash_idx=3,
    )


def _session(day: str, *, with_adv: bool = True, i: int = 0) -> SessionInputs:
    rng = np.random.default_rng(11)
    panel = rng.normal(0.0, 0.01, size=(260, 4))
    panel[:, 3] = 0.0
    return SessionInputs(
        trading_day=day,
        alpha_hat=np.array([0.02, 0.01, 0.0, -1e-6]) * (1 + 0.05 * i),
        eligibility=np.ones(4, dtype=bool),
        stance_caps=np.array([0.12, 0.12, 1.0, 1.0]),
        realized_returns=np.array([0.003, -0.001, 0.001, 0.0]),
        benchmark_return=0.001,
        returns_panel=panel,
        adv_usd=np.array([4e7, 1.5e7, 1e10, 0.0]) if with_adv else None,
    )


def _inputs(
    cost: dict = IMPACT,
    *,
    day: str = DAY1,
    with_adv: bool = True,
    i: int = 0,
    name: str = "shadow_fixture",
) -> ArmSessionInputs:
    return ArmSessionInputs(
        recipe=_recipe(cost, name),
        params=_params(),
        universe=_universe(),
        session=_session(day, with_adv=with_adv, i=i),
        portfolio_notional=NOTIONAL,
    )


class TestTheSameCostModelTheGradeCharges:
    def test_the_book_is_built_by_the_grades_own_engine(self) -> None:
        assert shadow_module.construct_book is construct_book
        source = inspect.getsource(shadow_module)
        for pricing in (
            "cost_for_turnover",
            "bps_per_unit_turnover",
            "cost_bps_for_trades(",
            "TransactionCostModel",
        ):
            assert pricing not in source, f"the shadow module prices trades itself: {pricing}"

    @pytest.mark.parametrize("cost", [FLAT, IMPACT], ids=["flat", "sqrt_impact"])
    def test_the_charge_is_the_recipes_cost_model_on_the_realized_deltas(self, cost) -> None:
        inputs = _inputs(cost)
        entry = advance_book(arm_id="s:x:1", inputs=inputs, previous=None)
        w0 = np.array([0.0, 0.0, 0.0, 1.0])
        delta = np.array([entry["weights"][t] for t in inputs.universe.tickers]) - w0
        expected = CostModel(**cost).cost_bps_for_trades(
            weight_deltas=delta,
            adv_usd=inputs.session.adv_usd,
            portfolio_notional=NOTIONAL,
            benchmark_idx=2,
            cash_idx=3,
        )
        assert entry["cost_bps"] == pytest.approx(expected, rel=1e-12)
        grade_book = construct_book(
            recipe=inputs.recipe,
            params=inputs.params,
            universe=inputs.universe,
            sessions=[inputs.session],
            portfolio_notional=NOTIONAL,
            w_initial=w0,
        )
        assert entry["cost_bps"] == pytest.approx(grade_book.book.cost_bps[0], rel=1e-12)
        assert entry["gross_return_ratio"] == pytest.approx(grade_book.book.portfolio_returns[0])
        assert entry["cost_model"] == CostModel(**cost).record()

    def test_a_participation_model_with_no_adv_raises_rather_than_charging_flat(self) -> None:
        with pytest.raises(CostModelInputError):
            advance_book(arm_id="s:x:1", inputs=_inputs(IMPACT, with_adv=False), previous=None)


class TestTheBookAdvances:
    def test_a_second_session_advances_from_the_first_sessions_close(self) -> None:
        first = advance_book(arm_id="s:x:1", inputs=_inputs(), previous=None)
        second = advance_book(arm_id="s:x:1", inputs=_inputs(day=DAY2, i=1), previous=first)
        assert second["days_advanced"] == [DAY1, DAY2] and second["sessions"] == 2
        assert second["inception_trading_day"] == DAY1
        expected = (1 + first["net_return_ratio"]) * (1 + second["net_return_ratio"]) - 1
        assert second["cumulative_net_return_ratio"] == pytest.approx(expected)
        assert second["net_return_ratio"] == pytest.approx(
            second["gross_return_ratio"] - second["cost_bps"] / 1e4
        )

    def test_a_session_that_does_not_follow_is_refused(self) -> None:
        first = advance_book(arm_id="s:x:1", inputs=_inputs(day=DAY2), previous=None)
        with pytest.raises(ValueError, match="does not follow"):
            advance_book(arm_id="s:x:1", inputs=_inputs(day=DAY1), previous=first)

    def test_another_arms_or_a_failed_previous_state_is_refused(self) -> None:
        first = advance_book(arm_id="s:x:1", inputs=_inputs(), previous=None)
        with pytest.raises(ValueError, match="its own last ADVANCED state"):
            advance_book(arm_id="s:x:2", inputs=_inputs(day=DAY2), previous=first)

    def test_a_non_zero_held_name_outside_the_universe_is_refused(self) -> None:
        first = advance_book(arm_id="s:x:1", inputs=_inputs(), previous=None)
        first = {**first, "weights": {**first["weights"], "ZZZ": 0.1}}
        with pytest.raises(ValueError, match=r"holds \['ZZZ'\].*does not contain"):
            advance_book(arm_id="s:x:1", inputs=_inputs(day=DAY2), previous=first)

    def test_zero_weight_names_outside_and_new_names_inside_the_universe_advance(self) -> None:
        """Only the non-zero support must be in the universe (I10754): a name the
        book no longer holds may leave, and a name it never held enters at 0."""
        first = advance_book(arm_id="s:x:1", inputs=_inputs(), previous=None)
        carried = {t: w for t, w in first["weights"].items() if t != "BBB"}
        carried["GONE"] = 0.0
        second = advance_book(
            arm_id="s:x:1", inputs=_inputs(day=DAY2), previous={**first, "weights": carried}
        )
        assert set(second["weights"]) == set(_universe().tickers)
        w_prev = np.array([carried.get(t, 0.0) for t in _universe().tickers])
        direct = construct_book(
            recipe=_recipe(IMPACT),
            params=_params(),
            universe=_universe(),
            sessions=[_session(DAY2)],
            portfolio_notional=NOTIONAL,
            w_initial=w_prev,
        )
        assert tuple(second["weights"].values()) == direct.weights[0]

    def test_a_book_with_no_per_session_cost_is_refused(self, monkeypatch) -> None:
        real = shadow_module.construct_book

        def uncharged(**kwargs):
            built = real(**kwargs)
            import dataclasses

            return dataclasses.replace(built, book=dataclasses.replace(built.book, cost_bps=None))

        monkeypatch.setattr(shadow_module, "construct_book", uncharged)
        with pytest.raises(ValueError, match="no per-session cost"):
            advance_book(arm_id="s:x:1", inputs=_inputs(), previous=None)


def _registered(tmp_path, names: list[str]) -> tuple[LocalStore, list[str]]:
    """A store whose `s` register lists one active arm per name, ids as derived."""
    store = LocalStore(tmp_path)
    register = ArmRegister()
    ids: list[str] = []
    for name in names:
        register, record = register.register(
            slot="s",
            name=name,
            spec={"recipe": name},
            created_date="2026-09-01",
            filed_on="2026-09-01",
        )
        ids.append(record.arm_id)
    write_register(store, "s", register)
    return store, ids


class TestRunShadowBooks:
    def test_every_active_arm_gets_an_advanced_book_the_harness_reads_as_covered(
        self, tmp_path
    ) -> None:
        store, (arm_a, arm_b) = _registered(tmp_path, ["shadow_a", "shadow_b"])
        doc = run_shadow_books(
            store,
            trading_day=DAY1,
            previous_trading_day=None,
            inputs={arm_a: _inputs(IMPACT), arm_b: _inputs(FLAT)},
            now=NOW,
        )
        assert {b["arm_id"] for b in doc["books"]} == {arm_a, arm_b}
        validate_shadow_books(json.loads(store.get_bytes(shadow_books_key(DAY1))))
        assert shadow_book_coverage(store, DAY1, days_served=[DAY1]).met

        doc2 = run_shadow_books(
            store,
            trading_day=DAY2,
            previous_trading_day=DAY1,
            inputs={
                arm_a: _inputs(IMPACT, day=DAY2, i=1),
                arm_b: _inputs(FLAT, day=DAY2, i=1),
            },
            now=NOW,
        )
        assert all(b["sessions"] == 2 for b in doc2["books"])
        assert all(m["arm_id"] in (arm_a, arm_b) for m in doc2["metrics"])
        assert shadow_book_coverage(store, DAY2, days_served=[DAY1, DAY2]).met

    def test_an_arm_that_raises_is_a_recorded_failed_book_then_the_session_fails(
        self, tmp_path
    ) -> None:
        store, (arm_a, arm_b) = _registered(tmp_path, ["shadow_a", "shadow_b"])
        inputs = {arm_a: _inputs(FLAT), arm_b: _inputs(IMPACT, with_adv=False)}
        with pytest.raises(ShadowBookFailure, match="CostModelInputError"):
            run_shadow_books(
                store, trading_day=DAY1, previous_trading_day=None, inputs=inputs, now=NOW
            )
        written = validate_shadow_books(json.loads(store.get_bytes(shadow_books_key(DAY1))))
        by_arm = {b["arm_id"]: b for b in written["books"]}
        assert by_arm[arm_a]["status"] == "advanced"
        assert by_arm[arm_b]["status"] == "failed"
        assert not shadow_book_coverage(store, DAY1).met

    def test_an_active_arm_with_no_inputs_is_not_silently_dropped(self, tmp_path) -> None:
        store, (arm_a, _arm_b) = _registered(tmp_path, ["shadow_a", "shadow_b"])
        with pytest.raises(ShadowBookFailure, match="no session inputs"):
            run_shadow_books(
                store,
                trading_day=DAY1,
                previous_trading_day=None,
                inputs={arm_a: _inputs(FLAT)},
                now=NOW,
            )

    def test_inputs_for_an_unregistered_arm_are_refused(self, tmp_path) -> None:
        store, _ = _registered(tmp_path, ["shadow_a"])
        with pytest.raises(ValueError, match="does not list as active"):
            run_shadow_books(
                store,
                trading_day=DAY1,
                previous_trading_day=None,
                inputs={"s:stranger:000000000000": _inputs(FLAT)},
                now=NOW,
            )

    def test_previous_books_absent_or_unnamed_read_as_inception(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        assert read_previous_books(store, None) == {}
        assert read_previous_books(store, DAY1) == {}


class TestThePhase4GateGradesTheseBooks:
    """`run_shadow_books` is the producer; `shadow_books_cover_every_active_arm`
    is the consumer, joined to the trader's served days in `trader/evidence.json`.
    Read back through `crucible.gate.evaluate` as shipped in the pinned wheel
    (`alpha-engine-config-I9760`, `-I10653`)."""

    CLAUSE = "shadow_books_cover_every_active_arm"

    @staticmethod
    def _served(store: LocalStore, days: list[str]) -> None:
        document = TraderEvidenceDocument(
            schema_version="trader_evidence.v2",
            slot="m",
            champion="m:fixture_model:aaaaaaaaaaaa",
            trading_days=len(days),
            days_served=days,
            session_modes=dict.fromkeys(days, "shadow"),
            calendar_date=days[-1],
        )
        store.put_bytes(TRADER_EVIDENCE_KEY, json.dumps(document.model_dump()).encode("utf-8"))

    def _two_sessions(self, tmp_path) -> LocalStore:
        store, (arm_a, arm_b) = _registered(tmp_path, ["shadow_a", "shadow_b"])
        run_shadow_books(
            store,
            trading_day=DAY1,
            previous_trading_day=None,
            inputs={arm_a: _inputs(IMPACT), arm_b: _inputs(FLAT)},
            now=NOW,
        )
        run_shadow_books(
            store,
            trading_day=DAY2,
            previous_trading_day=DAY1,
            inputs={arm_a: _inputs(IMPACT, day=DAY2, i=1), arm_b: _inputs(FLAT, day=DAY2, i=1)},
            now=NOW,
        )
        return store

    def test_books_without_the_traders_served_days_are_unmet(self, tmp_path, phase4_clause):
        """Books alone prove nothing: the gate checks them against the days the
        trader says it served, and a trader that filed none is UNMET."""
        clause = phase4_clause(self._two_sessions(tmp_path), self.CLAUSE, DAY2)
        assert not clause.met and not clause.unmeasurable
        assert f"{TRADER_EVIDENCE_KEY} is absent" in clause.detail

    def test_every_arm_advanced_on_every_served_day_is_met(self, tmp_path, phase4_clause):
        store = self._two_sessions(tmp_path)
        self._served(store, [DAY1, DAY2])
        clause = phase4_clause(store, self.CLAUSE, DAY2)
        assert clause.met, clause.detail
        assert shadow_books_key(DAY2) in clause.evidence
