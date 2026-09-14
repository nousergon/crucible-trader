"""The fire drill: the real path, an artifact per drill, and not-done means not-done."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from crucible.models import MetricRecordRow
from crucible.store import LocalStore
from ib_fakes import DAY, T0, FakeClock, FakeIB, FakeSdk, ctx_for

from crucible_trader import fire_drill
from crucible_trader.fire_drill import (
    FIRE_DRILL_JOB,
    FIRE_DRILL_SCHEMA_VERSION,
    PLANTED_CHAMPION,
    DrillRefusedError,
    FireDrillFailedError,
    drill_cycle,
    evaluate_drills,
    fire_drill_key,
    planted_degenerate_feed,
    read_drills,
)
from crucible_trader.hold_book import HoldDecision
from crucible_trader.kill_switch import IbBrokerControl, read_state


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _run(store, ib, kind, *, announced=False, clock=None):
    ctx = ctx_for(store, FIRE_DRILL_JOB)
    document = drill_cycle(
        ctx,
        broker=IbBrokerControl(ib, FakeSdk(ib)),
        kind=kind,
        announced=announced,
        clock=clock or ib.clock,
    )
    return ctx, document


def _artifact(store, ctx):
    return json.loads(store.get_bytes(fire_drill_key(DAY.isoformat(), ctx.run_id)))


class TestRefusals:
    def test_an_unknown_kind_is_refused(self, store) -> None:
        with pytest.raises(DrillRefusedError, match="not in"):
            _run(store, FakeIB(), "kill_switch_pause")

    def test_outside_market_hours_nothing_is_fired(self, store) -> None:
        ib = FakeIB(positions={"AAA": 1})
        saturday = FakeClock(dt.datetime(2026, 9, 12, 14, 0, tzinfo=dt.UTC))
        with pytest.raises(DrillRefusedError, match="outside regular NYSE hours"):
            _run(store, ib, "kill_switch_flatten", clock=saturday)
        assert ib.order_calls == [] and ib.cancel_calls == 0 and read_state(store) is None


class TestDrills:
    def test_an_unannounced_flatten_drill_leaves_a_passing_artifact(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10, "BBB": 5})
        ctx, document = _run(store, ib, "kill_switch_flatten", announced=False)
        artifact = _artifact(store, ctx)
        assert artifact == document
        assert artifact["schema_version"] == FIRE_DRILL_SCHEMA_VERSION
        assert artifact["announced"] is False and artifact["passed"] is True
        assert artifact["fired_at"] == "2026-09-08T14:00:00.000000Z"
        assert artifact["settled_at"] == "2026-09-08T14:00:02.000000Z"
        assert artifact["state_before"]["positions"] == {"AAA": 10, "BBB": 5}
        assert artifact["state_after"]["positions"] == {}
        assert artifact["orders_accepted_after_fire"] == []
        assert artifact["hold_decision"] is None
        (metric,) = ctx.metrics
        MetricRecordRow.model_validate(metric)

    def test_a_drill_that_lets_an_order_through_is_filed_and_fails(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10})
        ib.foreign_on_sleep = ["XYZ"]
        ctx = ctx_for(store, FIRE_DRILL_JOB)
        with pytest.raises(FireDrillFailedError, match="accepted after the fire"):
            drill_cycle(
                ctx,
                broker=IbBrokerControl(ib, FakeSdk(ib)),
                kind="kill_switch_flatten",
                announced=True,
                clock=ib.clock,
            )
        artifact = _artifact(store, ctx)
        assert artifact["passed"] is False
        assert [o["symbol"] for o in artifact["orders_accepted_after_fire"]] == ["XYZ"]
        assert ctx.metrics[0]["status"] == "FAIL"

    def test_a_hold_book_drill_engages_the_same_hold_a_real_batch_would(self, store) -> None:
        ib = FakeIB(positions={"AAA": 3})
        ctx, document = _run(store, ib, "hold_book", announced=True)
        assert document["hold_decision"] == "hold_signal_degenerate" and document["passed"]
        assert read_state(store)["cause"] == "degenerate_predictions:hold_signal_degenerate"
        assert not store.exists("predictions/2026-09-08.json")

    def test_a_hold_gate_that_does_not_hold_fails_the_drill(self, store, monkeypatch) -> None:
        monkeypatch.setattr(
            fire_drill,
            "enforce_hold_book",
            lambda ctx, feed, broker, clock: (
                HoldDecision(False, "proceed_signal_healthy", {}),
                None,
            ),
        )
        with pytest.raises(FireDrillFailedError, match="did not hold"):
            _run(store, FakeIB(), "hold_book")


class TestPlantedFeed:
    def test_a_small_book_is_padded_to_a_judgeable_batch(self) -> None:
        feed = planted_degenerate_feed("2026-09-08", ["AAA"])
        assert len(feed.predicted_alpha) == 5 and set(feed.predicted_alpha.values()) == {0.0}
        assert feed.champion == PLANTED_CHAMPION

    def test_a_large_book_is_not_padded(self) -> None:
        feed = planted_degenerate_feed("2026-09-08", [f"S{i}" for i in range(7)])
        assert sorted(feed.predicted_alpha) == [f"S{i}" for i in range(7)]


def _doc(day="2026-09-08", *, announced=False, passed=True, orders=(), settle=2.0, bound=300.0):
    return {
        "schema_version": FIRE_DRILL_SCHEMA_VERSION,
        "kind": "kill_switch_flatten",
        "announced": announced,
        "trading_day": day,
        "fired_at": "t0",
        "settled_at": None if settle is None else "t1",
        "settle_seconds": settle,
        "bound_seconds": bound,
        "orders_accepted_after_fire": list(orders),
        "passed": passed,
    }


WINDOW = {"window_start": "2026-09-01", "window_end": "2026-09-30"}


class TestEvaluate:
    def test_undrilled_is_not_done(self) -> None:
        reading = evaluate_drills([], **WINDOW)
        assert not reading.done and "undrilled" in reading.reason

    def test_one_passed_drill_is_not_done(self) -> None:
        assert not evaluate_drills([("a", _doc())], **WINDOW).done

    def test_two_announced_drills_are_not_done(self) -> None:
        reading = evaluate_drills(
            [("a", _doc(announced=True)), ("b", _doc(announced=True))], **WINDOW
        )
        assert not reading.done and reading.n_passed == 2 and reading.n_unannounced_passed == 0

    def test_two_passed_one_unannounced_is_done(self) -> None:
        reading = evaluate_drills([("a", _doc(announced=True)), ("b", _doc())], **WINDOW)
        assert reading.done and reading.reason.startswith("2/2 drills passed")

    @pytest.mark.parametrize(
        "bad",
        [
            _doc(passed=False),
            _doc(orders=[{"order_id": 1}]),
            _doc(settle=None),
            _doc(settle=301.0),
            _doc(passed="true"),
        ],
    )
    def test_a_failed_or_self_contradicting_drill_does_not_count(self, bad) -> None:
        reading = evaluate_drills([("a", _doc(announced=True)), ("b", bad)], **WINDOW)
        assert not reading.done and reading.n_passed == 1

    def test_drills_outside_the_window_do_not_count(self) -> None:
        reading = evaluate_drills([("a", _doc("2026-08-01")), ("b", _doc("2026-10-01"))], **WINDOW)
        assert not reading.done and "undrilled" in reading.reason

    def test_a_malformed_artifact_raises(self) -> None:
        with pytest.raises(ValueError, match="not a fire_drill.v1"):
            evaluate_drills([("a", {"schema_version": FIRE_DRILL_SCHEMA_VERSION})], **WINDOW)


class TestReadDrills:
    def test_reads_every_artifact(self, store) -> None:
        store.put_bytes(fire_drill_key("2026-09-08", "r1"), json.dumps(_doc()).encode())
        assert [key for key, _ in read_drills(store)] == ["trader/fire_drills/2026-09-08/r1.json"]

    def test_an_unreadable_artifact_raises_rather_than_being_skipped(self, store) -> None:
        store.put_bytes(fire_drill_key("2026-09-08", "r1"), b"{")
        with pytest.raises(ValueError, match="not readable JSON"):
            read_drills(store)


def test_the_default_market_hours_predicate_is_the_calendars() -> None:
    assert fire_drill.is_market_hours(T0) is True
