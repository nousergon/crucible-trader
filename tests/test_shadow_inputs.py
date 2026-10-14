"""Each registered S arm's shadow-book session from the store — `alpha-engine-config-I10653`.

The inputs come from `crucible.slots.inputs.resolve_strategy_sessions`, the
resolution the S grade constructs on, so a shadow book's paper P&L is the same
object the grade scores. Every active arm comes back resolved or unresolved
with a reason, and an unresolved arm becomes a failed book naming it.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from conftest import AS_OF, D_PREV, NEXT, TICKERS, recipe_yaml, remove
from crucible.execution import (
    CONTROL_NON_BOOK_STATUS,
    shadow_book_coverage,
    shadow_books_key,
    validate_shadow_books,
)
from crucible.keys import strategy_arm_key, strategy_slot_key
from crucible.portfolio import load_portfolio_params_from_store
from crucible.slots.inputs import resolve_strategy_sessions
from crucible.slots.strategy import construct_book, load_strategy_slot

from crucible_trader.construction import initial_weights
from crucible_trader.shadow_books import ShadowBookFailure, run_shadow_books
from crucible_trader.shadow_inputs import resolve_shadow_inputs

NOW = dt.datetime(2026, 9, 14, 21, 0, tzinfo=dt.UTC)
DAY = AS_OF.isoformat()


def _resolve(world):
    return resolve_shadow_inputs(
        world.store, decision_day=DAY, as_of=NEXT.isoformat(), previous_trading_day=None
    )


class TestEveryActiveArmIsResolvedFromTheStore:
    def test_a_recorded_arm_resolves_and_its_book_is_the_grades_book(self, world) -> None:
        challenger = world.file_recipe("challenger")
        world.register([world.arm_id, challenger])
        world.record_session(world.arm_id)
        world.record_session(challenger)

        resolved = _resolve(world)
        assert set(resolved.active_arms) == {world.arm_id, challenger}
        assert resolved.unresolved == {}

        inputs = resolved.inputs[challenger]
        grade = construct_book(
            recipe=inputs.recipe,
            params=inputs.params,
            universe=inputs.universe,
            sessions=[inputs.session],
            portfolio_notional=inputs.portfolio_notional,
            w_initial=initial_weights(inputs.universe),
        )
        document = run_shadow_books(
            world.store,
            trading_day=DAY,
            previous_trading_day=None,
            inputs=resolved.inputs,
            unresolved=resolved.unresolved,
            now=NOW,
        )
        book = next(b for b in document["books"] if b["arm_id"] == challenger)
        assert book["status"] == "advanced"
        assert book["cost_bps"] == grade.book.cost_bps[0]
        assert book["gross_return_ratio"] == grade.book.portfolio_returns[0]
        assert inputs.portfolio_notional == 1_000_000.0

    def test_an_arm_with_no_recorded_session_is_a_failed_book_naming_why(self, world) -> None:
        challenger = world.file_recipe("challenger")
        world.register([world.arm_id, challenger])
        world.record_session(world.arm_id)

        resolved = _resolve(world)
        assert challenger in resolved.unresolved
        assert "recorded no construction inputs" in resolved.unresolved[challenger]
        with pytest.raises(ShadowBookFailure, match="recorded no construction inputs"):
            run_shadow_books(
                world.store,
                trading_day=DAY,
                previous_trading_day=None,
                inputs=resolved.inputs,
                unresolved=resolved.unresolved,
                now=NOW,
            )
        written = validate_shadow_books(json.loads(world.store.get_bytes(shadow_books_key(DAY))))
        assert {b["arm_id"]: b["status"] for b in written["books"]} == {
            world.arm_id: "advanced",
            challenger: "failed",
        }

    def test_an_active_arm_no_filed_recipe_registers_is_unresolved(self, world) -> None:
        challenger = world.file_recipe("challenger")
        world.register([world.arm_id, challenger])
        remove(world.store, strategy_arm_key("s", "challenger"))

        resolved = _resolve(world)
        assert "no filed recipe registers it" in resolved.unresolved[challenger]

    def test_an_unservable_slot_leaves_every_arm_unresolved(self, world) -> None:
        world.register([world.arm_id])
        world.store.put_bytes(
            strategy_arm_key("s", "stock_registry"),
            recipe_yaml("stock_registry", registered_at="2026-12-01").encode("utf-8"),
        )
        resolved = _resolve(world)
        assert "unservable" in resolved.unresolved[world.arm_id]

    def test_an_unusable_parameter_set_leaves_every_arm_unresolved(self, world) -> None:
        world.register([world.arm_id])
        remove(world.store, strategy_slot_key("s"))
        resolved = _resolve(world)
        assert "parameter set is unusable" in resolved.unresolved[world.arm_id]

    def test_an_empty_register_resolves_nothing_and_reads_no_parameters(self, world) -> None:
        remove(world.store, strategy_slot_key("s"))
        resolved = _resolve(world)
        assert resolved.active_arms == () and resolved.inputs == {} and resolved.unresolved == {}

    def test_a_reason_for_an_unregistered_arm_is_refused(self, world) -> None:
        world.register([world.arm_id])
        with pytest.raises(ValueError, match="does not list as active"):
            run_shadow_books(
                world.store,
                trading_day=DAY,
                previous_trading_day=None,
                inputs={},
                unresolved={"s:stranger:000000000000": "nope"},
                now=NOW,
            )


class TestControlsAreARuledNonBook:
    """`alpha-engine-config-I12021`, Brian's ruling 2026-10-05: an active control is
    listed with `control_scored_by_grade` and a reason — never dropped, never a
    failed book for lacking a recipe — and coverage needs CHALLENGER books only."""

    def _run(self, world, resolved):
        return run_shadow_books(
            world.store,
            trading_day=DAY,
            previous_trading_day=None,
            inputs=resolved.inputs,
            unresolved=resolved.unresolved,
            now=NOW,
        )

    def test_the_10_01_shape_lists_controls_and_advances_challengers(self, world) -> None:
        """Two harness controls beside two filed challengers, as on 2026-10-01."""
        challenger = world.file_recipe("challenger")
        controls = world.register([world.arm_id, challenger], controls=True)
        world.record_session(world.arm_id)
        world.record_session(challenger)

        resolved = _resolve(world)
        assert len(controls) == 2 and set(resolved.controls) == set(controls)
        assert set(resolved.active_arms) == {world.arm_id, challenger, *controls}
        assert set(resolved.inputs) == {world.arm_id, challenger}
        assert resolved.unresolved == {}

        document = self._run(world, resolved)
        status = {b["arm_id"]: b["status"] for b in document["books"]}
        assert status == {
            world.arm_id: "advanced",
            challenger: "advanced",
            **dict.fromkeys(controls, CONTROL_NON_BOOK_STATUS),
        }
        for book in document["books"]:
            if book["arm_id"] in controls:
                assert "control arm" in book["non_book_reason"]
                assert book["failure_reason"] is None
        assert {m["arm_id"] for m in document["metrics"]} == {world.arm_id, challenger}
        reading = shadow_book_coverage(world.store, DAY, days_served=[DAY])
        assert reading.met, reading.detail

    def test_every_challenger_failing_still_fails_the_run(self, world) -> None:
        challenger = world.file_recipe("challenger")
        controls = world.register([world.arm_id, challenger], controls=True)

        resolved = _resolve(world)
        assert set(resolved.unresolved) == {world.arm_id, challenger}
        with pytest.raises(ShadowBookFailure, match="2 of 4 shadow book"):
            self._run(world, resolved)
        written = validate_shadow_books(json.loads(world.store.get_bytes(shadow_books_key(DAY))))
        assert {b["arm_id"]: b["status"] for b in written["books"]} == {
            world.arm_id: "failed",
            challenger: "failed",
            **dict.fromkeys(controls, CONTROL_NON_BOOK_STATUS),
        }
        assert not shadow_book_coverage(world.store, DAY).met

    def test_inputs_for_a_control_are_refused(self, world) -> None:
        controls = world.register([world.arm_id], controls=True)
        with pytest.raises(ValueError, match="marks as control"):
            run_shadow_books(
                world.store,
                trading_day=DAY,
                previous_trading_day=None,
                inputs={},
                unresolved={controls[0]: "no filed recipe registers it"},
                now=NOW,
            )

    def test_a_controls_only_register_reads_no_parameters(self, world) -> None:
        controls = world.register([], controls=True)
        remove(world.store, strategy_slot_key("s"))
        resolved = _resolve(world)
        assert set(resolved.controls) == set(controls)
        assert resolved.inputs == {} and resolved.unresolved == {}


class TestAHeldNameTheMChampionStopsPricingIsExited:
    """`alpha-engine-config-I10754` for shadow books: the book advanced through
    D_PREV holds a name the M champion stops pricing on DAY. Its DAY session is
    resolved with that held name, the book sells it, charged, and matches the
    settled grade walk over [D_PREV, DAY] from cash."""

    def _advance_d_prev(self, world):
        world.register([world.arm_id])
        world.record_session(world.arm_id, D_PREV)
        first = resolve_shadow_inputs(
            world.store,
            decision_day=D_PREV.isoformat(),
            as_of=DAY,
            previous_trading_day=None,
        )
        document = run_shadow_books(
            world.store,
            trading_day=D_PREV.isoformat(),
            previous_trading_day=None,
            inputs=first.inputs,
            unresolved=first.unresolved,
            now=NOW,
        )
        (book,) = document["books"]
        held = {t: w for t, w in book["weights"].items() if w != 0.0 and t in TICKERS}
        assert held, "the fixture's first book holds no name; nothing can leave the universe"
        dropped = max(held, key=held.get)
        world.drop_prediction(AS_OF, dropped)
        world.record_session(world.arm_id, AS_OF)
        return book, dropped

    def test_the_dropped_name_is_exited_charged_and_equals_the_grade_walk(self, world) -> None:
        first, dropped = self._advance_d_prev(world)
        resolved = resolve_shadow_inputs(
            world.store,
            decision_day=DAY,
            as_of=NEXT.isoformat(),
            previous_trading_day=D_PREV.isoformat(),
        )
        assert dropped in resolved.inputs[world.arm_id].universe.tickers
        document = run_shadow_books(
            world.store,
            trading_day=DAY,
            previous_trading_day=D_PREV.isoformat(),
            inputs=resolved.inputs,
            unresolved=resolved.unresolved,
            now=NOW,
        )
        (book,) = document["books"]
        assert book["status"] == "advanced" and book["days_advanced"] == [D_PREV.isoformat(), DAY]
        assert book["weights"][dropped] < first["weights"][dropped], "the dropped name was not sold"
        assert book["weights"][dropped] == 0.0
        assert book["cost_bps"] > 0.0

        recipe = next(
            arm.recipe
            for arm in load_strategy_slot(store=world.store).registered
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
        assert tuple(book["weights"]) == walk.universe.tickers
        assert tuple(book["weights"].values()) == graded.weights[1]
        assert book["cost_bps"] == graded.book.cost_bps[1]
        assert book["gross_return_ratio"] == graded.book.portfolio_returns[1]

    def test_resolving_without_the_previous_book_is_a_recorded_failed_book(self, world) -> None:
        _first, dropped = self._advance_d_prev(world)
        resolved = resolve_shadow_inputs(
            world.store, decision_day=DAY, as_of=NEXT.isoformat(), previous_trading_day=None
        )
        assert dropped not in resolved.inputs[world.arm_id].universe.tickers
        with pytest.raises(ShadowBookFailure, match=f"holds \\['{dropped}'\\]"):
            run_shadow_books(
                world.store,
                trading_day=DAY,
                previous_trading_day=D_PREV.isoformat(),
                inputs=resolved.inputs,
                unresolved=resolved.unresolved,
                now=NOW,
            )
