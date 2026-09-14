"""The fire drill: the real path, an artifact per drill, and not-done means not-done."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest
from crucible.models import FireDrillDocument, MetricRecordRow
from crucible.store import LocalStore
from ib_fakes import DAY, T0, FakeClock, FakeIB, FakeSdk, ctx_for
from s3_fake import FakeSsm

from crucible_trader import fire_drill
from crucible_trader.drill_schedule import SealDoesNotOpenError, SealOpening, seal_schedule
from crucible_trader.fire_drill import (
    FIRE_DRILL_JOB,
    FIRE_DRILL_SCHEMA_VERSION,
    PLANTED_CHAMPION,
    DrillRefusedError,
    FireDrillFailedError,
    drill_cycle,
    fire_drill_key,
    planted_degenerate_feed,
)
from crucible_trader.hold_book import HoldDecision
from crucible_trader.kill_switch import IbBrokerControl, read_state


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _run(store, ib, kind, *, announced=True, seal=None, clock=None):
    ctx = ctx_for(store, FIRE_DRILL_JOB)
    document = drill_cycle(
        ctx,
        broker=IbBrokerControl(ib, FakeSdk(ib)),
        kind=kind,
        announced=announced,
        seal=seal,
        clock=clock or ib.clock,
    )
    return ctx, document


def _sealed(store, fire_instant="2026-09-08T14:00:00Z") -> SealOpening:
    """Seal a drill for ``fire_instant`` an hour before ``T0``, as the operator does."""
    ssm = FakeSsm()
    seal = seal_schedule(
        store,
        ssm,
        window_start="2026-09-08",
        window_end="2026-09-11",
        fire_instant=fire_instant,
        clock=lambda: T0 - dt.timedelta(hours=1),
    )
    (parameter,) = ssm.parameters.values()
    return SealOpening(
        seal.schedule_id, seal.window_start, seal.window_end, fire_instant, parameter["Value"]
    )


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
        seal = _sealed(store)
        ctx, document = _run(store, ib, "kill_switch_flatten", announced=False, seal=seal)
        artifact = _artifact(store, ctx)
        assert artifact == document
        assert artifact["schedule"] == dataclasses.asdict(seal)
        FireDrillDocument.model_validate(artifact)
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


class TestSealedDrills:
    def test_an_unannounced_drill_without_a_seal_is_refused(self, store) -> None:
        ib = FakeIB(positions={"AAA": 1})
        with pytest.raises(DrillRefusedError, match="sealed schedule"):
            _run(store, ib, "kill_switch_flatten", announced=False)
        assert ib.order_calls == [] and read_state(store) is None

    def test_an_announced_drill_handed_a_seal_is_refused(self, store) -> None:
        with pytest.raises(DrillRefusedError, match="announced drill was handed a seal"):
            _run(store, FakeIB(), "kill_switch_flatten", seal=_sealed(store))

    def test_a_seal_that_does_not_open_fires_nothing(self, store) -> None:
        ib = FakeIB(positions={"AAA": 1})
        seal = dataclasses.replace(_sealed(store), nonce="c" * 64)
        with pytest.raises(SealDoesNotOpenError):
            _run(store, ib, "kill_switch_flatten", announced=False, seal=seal)
        assert ib.order_calls == [] and ib.cancel_calls == 0 and read_state(store) is None

    def test_an_announced_artifact_reveals_no_schedule(self, store) -> None:
        _, document = _run(store, FakeIB(positions={"AAA": 1}), "kill_switch_freeze")
        assert document["schedule"] is None
        FireDrillDocument.model_validate(document)


class TestPlantedFeed:
    def test_a_small_book_is_padded_to_a_judgeable_batch(self) -> None:
        feed = planted_degenerate_feed("2026-09-08", ["AAA"])
        assert len(feed.predicted_alpha) == 5 and set(feed.predicted_alpha.values()) == {0.0}
        assert feed.champion == PLANTED_CHAMPION

    def test_a_large_book_is_not_padded(self) -> None:
        feed = planted_degenerate_feed("2026-09-08", [f"S{i}" for i in range(7)])
        assert sorted(feed.predicted_alpha) == [f"S{i}" for i in range(7)]


def test_the_default_market_hours_predicate_is_the_calendars() -> None:
    assert fire_drill.is_market_hours(T0) is True
