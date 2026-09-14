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
from conftest import AS_OF, NEXT, recipe_yaml, remove
from crucible.execution import shadow_books_key, validate_shadow_books
from crucible.keys import strategy_arm_key, strategy_slot_key
from crucible.slots.strategy import construct_book

from crucible_trader.construction import initial_weights
from crucible_trader.shadow_books import ShadowBookFailure, run_shadow_books
from crucible_trader.shadow_inputs import resolve_shadow_inputs

NOW = dt.datetime(2026, 9, 14, 21, 0, tzinfo=dt.UTC)
DAY = AS_OF.isoformat()


def _resolve(world):
    return resolve_shadow_inputs(world.store, decision_day=DAY, as_of=NEXT.isoformat())


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
