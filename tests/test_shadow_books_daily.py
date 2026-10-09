"""The after-close shadow-book run the executor box's timer starts (`alpha-engine-config-I11545`).

It only decides WHICH days, from the NYSE calendar: after the close of ``T``
the book for the session before ``T`` is advanced from the one before that, and
written where `shadow_books_cover_every_active_arm` reads it.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import pytest
from conftest import AS_OF, D_PREV, M_CHAMPION, NEXT
from crucible.execution import shadow_books_key
from crucible.keys import TRADER_EVIDENCE_KEY
from crucible.models import TraderEvidenceDocument

import crucible_trader.broker_session as broker_session
import crucible_trader.order_router as order_router
import crucible_trader.paper_smoke as paper_smoke
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


#: Tuesday 2026-09-15, ~22:00 ET -- when the published phase-4 gate for 09-15 is
#: taken (`runs/gate/{day}/run.json` lands about then every trading day).
GATE_EVENING = dt.date(2026, 9, 15)


def _served(world, days: list[str]) -> None:
    """`trader/evidence.json` as the box's `trader.session` runs leave it: each
    morning's 10:45 ET session binds to the LAST CLOSED session, so by the
    evening of 09-15 the sessions of 09-14 and 09-15 have served 09-11 and 09-14.
    """
    document = TraderEvidenceDocument(
        schema_version="trader_evidence.v2",
        slot="m",
        champion=M_CHAMPION,
        trading_days=len(days),
        days_served=days,
        session_modes=dict.fromkeys(days, "shadow"),
        calendar_date=days[-1],
    )
    world.store.put_bytes(TRADER_EVIDENCE_KEY, json.dumps(document.model_dump()).encode("utf-8"))


class TestThePhase4GateReadsWhatTheBoxScheduleFiles:
    """C29 (2026-10-03 run), `alpha-engine-config-I10653` over the I11545 ruling.

    The round trip on the box's own clock, not on a convenient one: the 11:00 ET
    run of 09-15 advances 09-11's book as of 09-14, and the evening gate of
    09-15 reads it -- through `crucible.gate.evaluate(gate="phase4")` as shipped
    in the pinned wheel -- against the days the trader served. Before
    crucible's settlement-lag read the gate asked for `shadow_books/2026-09-15
    .json`, a document this schedule writes on 09-17, and the clause could never
    read MET.
    """

    CLAUSE = "shadow_books_cover_every_active_arm"

    def test_the_morning_run_is_read_met_by_that_evenings_gate(self, world, phase4_clause):
        world.register([world.arm_id])
        world.record_session(world.arm_id)
        _served(world, [AS_OF.isoformat(), NEXT.isoformat()])

        code = main([], environ=_env(world), clock=lambda: NEXT_MORNING, printer=lambda _: None)
        clause = phase4_clause(world.store, self.CLAUSE, GATE_EVENING.isoformat())

        assert code == 0
        assert clause.met and not clause.unmeasurable, clause.detail
        assert shadow_books_key(AS_OF.isoformat()) in clause.evidence
        assert "on all 1 served day(s)" in clause.detail

    def test_a_morning_the_run_did_not_happen_is_unmet_by_that_evening(self, world, phase4_clause):
        world.register([world.arm_id])
        _served(world, [AS_OF.isoformat(), NEXT.isoformat()])

        clause = phase4_clause(world.store, self.CLAUSE, GATE_EVENING.isoformat())

        assert not clause.met and not clause.unmeasurable
        assert shadow_books_key(AS_OF.isoformat()) in clause.evidence


class TestNoOrderPathIsReachable:
    """The shadow-book run is shadow-only by construction, proven at run time:
    every way this package opens a broker session or routes an order is made to
    raise, the broker SDK is made unimportable, and the run still advances and
    writes every book."""

    def test_the_run_completes_with_every_broker_and_order_entry_point_refusing(
        self, world, monkeypatch
    ) -> None:
        def refuse(*_args, **_kwargs):
            raise AssertionError("the shadow-book run reached a broker or order entry point")

        monkeypatch.setitem(sys.modules, "ib_async", None)  # any import raises
        for module in (broker_session, paper_smoke):
            monkeypatch.setattr(module, "connect", refuse)
            monkeypatch.setattr(module, "load_sdk", refuse)
        monkeypatch.setattr(order_router, "build_router", refuse)
        world.register([world.arm_id])
        world.record_session(world.arm_id)

        assert main([], environ=_env(world), clock=lambda: AFTER_CLOSE, printer=lambda _: None) == 0
        document = json.loads(world.store.get_bytes(shadow_books_key(AS_OF.isoformat())))
        assert [b["status"] for b in document["books"]] == ["advanced"]
