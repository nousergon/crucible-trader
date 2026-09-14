"""The trader's book is the grade's book — `alpha-engine-config-I10654`.

The load-bearing assertion is `TestTheTraderAndTheGradeConstructOneBook`: from
the same store, the trader's target book for a decision day and the book the
S grade constructs for that day once it settles are the same weights, the same
charge and the same engine evidence. The grade side is not re-derived here: it
is `resolve_strategy_sessions` + `construct_book`, the two calls
`crucible.slots.strategy.grade` makes, and `crucible`'s own
`tests/test_strategy_session_resolution.py` proves that pair reproduces a real
grading manifest.
"""

from __future__ import annotations

import datetime as dt
import inspect

import crucible.slots.inputs as harness_inputs
import crucible.slots.strategy as harness_strategy
import numpy as np
import pytest
from conftest import (
    AS_OF,
    D_PREV,
    IMPACT_COST,
    M_CHAMPION,
    NEXT,
    TICKERS,
    U_CHAMPION,
    put,
    remove,
    seat_champion,
)
from crucible.keys import champion_key, session_inputs_key, strategy_slot_key
from crucible.portfolio import (
    PORTFOLIO_METRIC_NAME,
    CostModelInputError,
    load_portfolio_params_from_store,
)
from crucible.slots.inputs import resolve_strategy_sessions
from crucible.slots.strategy import construct_book

import crucible_trader.construction as construction
from crucible_trader.construction import (
    _previous_weights,
    construct_target_book,
    held_tickers,
    initial_weights,
)
from crucible_trader.contract import ContractRefusal, ContractUnavailable

NOW = dt.datetime(2026, 9, 11, 20, 30, tzinfo=dt.UTC)
DAY = AS_OF.isoformat()


def _grade_book(world, *, w_initial=None):
    """The S grade's construction of DAY, once DAY has settled (panel at NEXT)."""
    recipe = next(
        arm.recipe
        for arm in harness_strategy.load_strategy_slot(store=world.store).registered
        if arm.arm_id == world.arm_id
    )
    params = load_portfolio_params_from_store("s", store=world.store)
    inputs = resolve_strategy_sessions(
        world.store,
        arm_id=world.arm_id,
        benchmark=recipe.benchmark,
        decision_days=[DAY],
        as_of=NEXT.isoformat(),
    )
    return construct_book(
        recipe=recipe,
        params=params,
        universe=inputs.universe,
        sessions=inputs.sessions,
        portfolio_notional=params.book_notional_usd,
        w_initial=initial_weights(inputs.universe) if w_initial is None else w_initial,
    )


class TestTheTraderAndTheGradeConstructOneBook:
    def test_the_trader_calls_the_grades_own_engine_and_resolver(self) -> None:
        assert construction.construct_book is harness_strategy.construct_book
        assert construction.resolve_strategy_sessions is harness_inputs.resolve_strategy_sessions
        assert "resolve_strategy_sessions(" in inspect.getsource(harness_strategy.grade), (
            "the grade no longer resolves its inputs through the function the trader calls"
        )
        source = inspect.getsource(construction)
        for private in ("solve_target_weights", "cost_bps_for_trades(", "_build_sessions"):
            assert private not in source, f"the trader re-implements {private}"

    def test_same_store_same_book(self, world) -> None:
        world.record_session(world.arm_id)
        traded = construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        graded = _grade_book(world)

        assert tuple(traded.weights) == traded.universe.tickers
        assert tuple(traded.weights.values()) == graded.weights[0]
        assert traded.cost_bps == graded.book.cost_bps[0]
        assert traded.turnover_one_way_ratio == graded.book.turnover[0]
        for field in ("engine", "cost_model", "params_digest", "arm_id", "sessions"):
            assert traded.evidence[field] == graded.evidence[field], field

    def test_same_book_from_a_held_position(self, world) -> None:
        world.record_session(world.arm_id)
        first = construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        held = {t: w for t, w in first.weights.items() if w != 0.0}
        held_vector = np.array([held.get(t, 0.0) for t in first.universe.tickers])

        again = construct_target_book(world.store, trading_day=DAY, previous_weights=held, now=NOW)
        graded = _grade_book(world, w_initial=held_vector)

        assert tuple(again.weights.values()) == graded.weights[0]
        assert again.cost_bps == graded.book.cost_bps[0]

    def test_the_evidence_names_the_engine_and_cost_model_on_a_metric_record(self, world) -> None:
        key = world.record_session(world.arm_id)
        book = construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        assert book.evidence["engine"] == "crucible.portfolio"
        assert book.evidence["cost_model"]["name"] == "flat_bps_v0"
        assert book.metric["name"] == PORTFOLIO_METRIC_NAME
        assert book.metric["last_updated_utc"] == "2026-09-11T20:30:00Z"
        assert (book.s_champion, book.m_champion, book.u_champion) == (
            world.arm_id,
            M_CHAMPION,
            U_CHAMPION,
        )
        assert book.session_inputs_key == key
        assert book.portfolio_notional_usd == 1_000_000.0

    def test_a_participation_model_with_adv_constructs_and_without_adv_raises(self, world) -> None:
        arm_id = world.file_recipe("impact_registry", cost=IMPACT_COST)
        seat_champion(world.store, "s", arm_id)
        world.record_session(arm_id)
        with pytest.raises(CostModelInputError):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        world.write_adv()
        book = construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        assert book.evidence["cost_model"]["name"] == "sqrt_impact_v1"


class TestNoAttestedChampionNoBook:
    def test_no_s_champion_is_a_hold(self, world) -> None:
        remove(world.store, champion_key("s"))
        with pytest.raises(ContractUnavailable, match="no champion pointer for slot 's'"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_an_s_champion_without_a_pass_attestation_is_refused(self, world) -> None:
        seat_champion(
            world.store,
            "s",
            world.arm_id,
            attestation={"kind": "pit_parity", "status": "PARTIAL", "reason": "3 sessions"},
        )
        with pytest.raises(ContractRefusal, match="UNVERIFIED"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_no_m_champion_is_a_hold(self, world) -> None:
        remove(world.store, champion_key("m"))
        with pytest.raises(ContractUnavailable, match="slot 'm'"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_an_unrecorded_session_is_a_hold(self, world) -> None:
        with pytest.raises(ContractUnavailable, match="recorded no construction inputs"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_a_session_recorded_under_another_m_champion_is_refused(self, world) -> None:
        world.record_session(world.arm_id)
        seat_champion(world.store, "m", "m:successor:cccccccccccc")
        with pytest.raises(ContractRefusal, match="different M champion"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_a_cut_from_a_demoted_u_champion_is_refused(self, world) -> None:
        world.record_session(world.arm_id)
        remove(world.store, champion_key("u"))
        with pytest.raises(ContractRefusal, match="no U champion is promoted now"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_a_cut_from_another_u_champion_is_refused(self, world) -> None:
        world.record_session(world.arm_id)
        seat_champion(world.store, "u", "u:other_cut:dddddddddddd")
        with pytest.raises(ContractRefusal, match="the U champion's cut is"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_a_recorded_absence_of_u_constructs_with_no_u_champion(self, world) -> None:
        remove(world.store, champion_key("u"))
        world.record_session(world.arm_id)
        book = construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        assert book.u_champion is None

    def test_a_champion_whose_recipe_is_not_filed_is_refused(self, world) -> None:
        seat_champion(world.store, "s", "s:unfiled:eeeeeeeeeeee")
        with pytest.raises(ContractRefusal, match="is not among the S recipes"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_an_unservable_slot_is_refused(self, world) -> None:
        world.file_recipe("stock_registry", registered_at="2026-12-01")
        with pytest.raises(ContractRefusal, match="unservable"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_a_missing_parameter_set_is_refused(self, world) -> None:
        remove(world.store, strategy_slot_key("s"))
        with pytest.raises(ContractRefusal, match="parameter set is unusable"):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)

    def test_a_held_name_with_no_price_is_a_hold_not_a_book(self, world) -> None:
        world.record_session(world.arm_id)
        with pytest.raises(ContractUnavailable, match=r"\['ZZZ'\].*no proxy is substituted"):
            construct_target_book(
                world.store, trading_day=DAY, previous_weights={"ZZZ": 0.1}, now=NOW
            )

    def test_a_universe_the_resolver_returned_without_a_held_name_is_refused(self, world) -> None:
        """Defence in depth: the resolver is specified to carry every held name."""
        world.record_session(world.arm_id)
        book = construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)
        with pytest.raises(ContractRefusal, match="passed to the resolver as held names"):
            _previous_weights(book.universe, {"ZZZ": 0.1}, trading_day=DAY)

    def test_a_session_document_filed_under_another_day_is_refused(self, world) -> None:
        from crucible.slots.inputs import ArmPredictionsContractError

        key = world.record_session(world.arm_id)
        document = __import__("json").loads(world.store.get_bytes(key))
        document["trading_day"] = "2026-09-10"
        put(world.store, session_inputs_key(world.arm_id, DAY), document)
        with pytest.raises(ArmPredictionsContractError):
            construct_target_book(world.store, trading_day=DAY, previous_weights=None, now=NOW)


class TestAHeldNameTheMChampionStopsPricingIsExited:
    """`alpha-engine-config-I10754`: the trader decides D_PREV from cash, carries
    that book into DAY, and on DAY the M champion no longer prices the name the
    book held most of. The trader must sell it, charged, and construct exactly
    the book the settled grade walk over [D_PREV, DAY] constructs for DAY."""

    def _carry(self, world):
        world.record_session(world.arm_id, D_PREV)
        first = construct_target_book(
            world.store, trading_day=D_PREV.isoformat(), previous_weights=None, now=NOW
        )
        held = {t: w for t, w in first.weights.items() if w != 0.0 and t in TICKERS}
        assert held, "the fixture's first book holds no name; nothing can leave the universe"
        dropped = max(held, key=held.get)
        world.drop_prediction(AS_OF, dropped)
        world.record_session(world.arm_id, AS_OF)
        return first, dropped

    def test_held_tickers_is_the_non_zero_non_sentinel_support(self) -> None:
        assert held_tickers(None, benchmark="SPY") == []
        weights = {"T01": 0.2, "T02": 0.0, "SPY": 0.1, "__CASH__": 0.7}
        assert held_tickers(weights, benchmark="SPY") == ["T01"]

    def test_the_dropped_name_is_exited_charged_and_equals_the_grade_walk(self, world) -> None:
        first, dropped = self._carry(world)
        book = construct_target_book(
            world.store, trading_day=DAY, previous_weights=first.weights, now=NOW
        )
        assert dropped in book.universe.tickers
        assert book.weights[dropped] < first.weights[dropped], "the dropped name was not sold"
        assert book.weights[dropped] == 0.0
        assert book.cost_bps > 0.0 and book.turnover_one_way_ratio > 0.0

        recipe = next(
            arm.recipe
            for arm in harness_strategy.load_strategy_slot(store=world.store).registered
            if arm.arm_id == world.arm_id
        )
        params = load_portfolio_params_from_store("s", store=world.store)
        walk = resolve_strategy_sessions(
            world.store,
            arm_id=world.arm_id,
            benchmark=recipe.benchmark,
            decision_days=[D_PREV.isoformat(), DAY],
            as_of=NEXT.isoformat(),
        )
        graded = construct_book(
            recipe=recipe,
            params=params,
            universe=walk.universe,
            sessions=walk.sessions,
            portfolio_notional=params.book_notional_usd,
            w_initial=initial_weights(walk.universe),
        )
        assert book.universe == walk.universe
        assert graded.weights[0] == tuple(first.weights.get(t, 0.0) for t in walk.universe.tickers)
        assert tuple(book.weights.values()) == graded.weights[1]
        assert book.cost_bps == graded.book.cost_bps[1]
        assert book.turnover_one_way_ratio == graded.book.turnover[1]
