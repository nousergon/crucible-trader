"""The after-close shadow-book run the executor box's timer starts (`alpha-engine-config-I11545`).

It only decides WHICH days, from the NYSE calendar: after the close of ``T``
the book for the session before ``T`` is advanced from the one before that, and
written where `shadow_books_cover_every_active_arm` reads it.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from conftest import AS_OF, D_PREV, NEXT
from crucible.execution import shadow_books_key

from crucible_trader.shadow_books import ShadowBookFailure
from crucible_trader.shadow_books_daily import main, sessions_for

#: Monday 2026-09-14, 17:45 ET -- after NEXT's close.
AFTER_CLOSE = dt.datetime(2026, 9, 14, 21, 45, tzinfo=dt.UTC)
#: Tuesday 2026-09-15, 11:00 ET -- the box timer's slot; NEXT is the last close.
NEXT_MORNING = dt.datetime(2026, 9, 15, 15, 0, tzinfo=dt.UTC)


def _env(world) -> dict[str, str]:
    return {"CRUCIBLE_TRADER_STORE_URI": f"file://{world.store.root}"}


class TestTheDays:
    def test_after_a_close_the_previous_sessions_book_advances_from_the_one_before(self) -> None:
        assert sessions_for(NEXT) == (NEXT.isoformat(), AS_OF.isoformat(), D_PREV.isoformat())

    def test_a_weekend_is_walked_back_over(self) -> None:
        """Monday's close advances Friday's book from Thursday's."""
        assert sessions_for(dt.date(2026, 9, 14))[1:] == ("2026-09-11", "2026-09-10")


class TestTheEntryPoint:
    def test_a_non_trading_day_is_skipped_before_the_store_is_opened(self) -> None:
        said: list[str] = []
        saturday = dt.datetime(2026, 9, 12, 21, 45, tzinfo=dt.UTC)

        assert main([], environ={}, clock=lambda: saturday, printer=said.append) == 0
        assert said[0].startswith("SKIP: 2026-09-12")

    def test_it_takes_no_arguments(self) -> None:
        with pytest.raises(SystemExit, match="takes no arguments"):
            main(["--genesis"], environ={})

    def test_every_active_arm_is_advanced_and_written_for_the_decision_day(self, world) -> None:
        world.register([world.arm_id])
        world.record_session(world.arm_id)
        said: list[str] = []

        code = main([], environ=_env(world), clock=lambda: AFTER_CLOSE, printer=said.append)

        document = json.loads(world.store.get_bytes(shadow_books_key(AS_OF.isoformat())))
        assert code == 0
        assert [b["arm_id"] for b in document["books"]] == [world.arm_id]
        assert document["books"][0]["status"] == "advanced"
        assert said[-1].startswith(f"shadow books {AS_OF.isoformat()} (as of {NEXT.isoformat()})")

    def test_the_morning_run_advances_the_same_books_as_an_after_close_run(self, world) -> None:
        """The box is stopped by the evening, so its timer fires the next morning.

        ``as_of`` is the LAST CLOSED session, not today: Tuesday morning's run
        advances Friday's book as of Monday, exactly what Monday's close would.
        """
        world.register([world.arm_id])
        world.record_session(world.arm_id)
        said: list[str] = []

        code = main([], environ=_env(world), clock=lambda: NEXT_MORNING, printer=said.append)

        assert code == 0
        assert world.store.exists(shadow_books_key(AS_OF.isoformat()))
        assert said[-1].startswith(f"shadow books {AS_OF.isoformat()} (as of {NEXT.isoformat()})")

    def test_a_failed_book_is_written_and_then_fails_the_run(self, world) -> None:
        world.register([world.arm_id])  # registered, but no session recorded

        with pytest.raises(ShadowBookFailure):
            main([], environ=_env(world), clock=lambda: AFTER_CLOSE)
        document = json.loads(world.store.get_bytes(shadow_books_key(AS_OF.isoformat())))
        assert document["books"][0]["status"] == "failed"
