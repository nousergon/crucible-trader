"""The kill switch: halt first, then act, then MEASURE what the book did."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from crucible.models import MetricRecordRow
from crucible.store import LocalStore
from ib_fakes import DAY, T0, FakeClock, FakeIB, FakeSdk, ctx_for

from crucible_trader.kill_switch import (
    KILL_SWITCH_JOB,
    KILL_SWITCH_KEY,
    KILL_SWITCH_ORDER_REF,
    SETTLE_BOUND_S,
    AcceptedOrder,
    IbBrokerControl,
    KillSwitchNotSettledError,
    KillSwitchStateError,
    TradingHaltedError,
    assert_trading_permitted,
    fire,
    halt_document,
    kill_switch_event_key,
    read_state,
    record_outcome,
    release,
    utcnow,
)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _broker(ib):
    return IbBrokerControl(ib, FakeSdk(ib))


def _fire(store, ib, mode, **kw):
    ctx = ctx_for(store, KILL_SWITCH_JOB)
    return ctx, fire(ctx, _broker(ib), mode=mode, cause="operator", clock=ib.clock, **kw)


def _put_state(store, **fields):
    document = halt_document(
        engaged=True, mode="flatten", cause="c", trading_day="2026-09-08", since="s", run_id="r"
    )
    document.update(fields)
    store.put_bytes(KILL_SWITCH_KEY, json.dumps(document).encode())


class TestState:
    def test_never_written_is_none(self, store) -> None:
        assert read_state(store) is None
        assert_trading_permitted(store, "2026-09-08")

    def test_unreadable_is_an_error_never_off(self, store) -> None:
        store.put_bytes(KILL_SWITCH_KEY, b"{nope")
        with pytest.raises(KillSwitchStateError, match="not readable JSON"):
            assert_trading_permitted(store, "2026-09-08")

    @pytest.mark.parametrize(
        "fields",
        [{"extra": 1}, {"schema_version": "kill_switch.v0"}, {"engaged": "yes"}, {"mode": "pause"}],
    )
    def test_a_malformed_document_is_an_error(self, store, fields) -> None:
        _put_state(store, **fields)
        with pytest.raises(KillSwitchStateError, match="not a kill_switch.v1"):
            read_state(store)

    def test_a_non_object_is_an_error(self, store) -> None:
        store.put_bytes(KILL_SWITCH_KEY, b"[]")
        with pytest.raises(KillSwitchStateError):
            read_state(store)

    @pytest.mark.parametrize("mode", ["flatten", "freeze"])
    def test_flatten_and_freeze_persist_across_days_until_released(self, store, mode) -> None:
        _put_state(store, mode=mode, trading_day="2026-09-01")
        with pytest.raises(TradingHaltedError, match="A human releases it"):
            assert_trading_permitted(store, "2026-09-08")

    def test_a_hold_halts_its_own_session(self, store) -> None:
        _put_state(store, mode="hold")
        with pytest.raises(TradingHaltedError, match="lapses after session 2026-09-08"):
            assert_trading_permitted(store, "2026-09-08")

    def test_a_hold_from_an_earlier_session_has_lapsed(self, store) -> None:
        _put_state(store, mode="hold", trading_day="2026-09-04")
        assert_trading_permitted(store, "2026-09-08")

    def test_a_released_switch_permits_trading(self, store) -> None:
        _put_state(store, engaged=False, mode=None)
        assert_trading_permitted(store, "2026-09-08")


class TestFire:
    def test_flatten_writes_the_halt_before_any_broker_action(self, store) -> None:
        seen = []

        class Watching(FakeIB):
            def reqGlobalCancel(self):  # noqa: N802 - SDK spelling
                seen.append(read_state(store))
                super().reqGlobalCancel()

        ib = Watching(positions={"AAA": 10, "BBB": -3})
        fire(
            ctx_for(store, KILL_SWITCH_JOB), _broker(ib), mode="flatten", cause="x", clock=ib.clock
        )
        assert seen and seen[0]["engaged"] is True and seen[0]["mode"] == "flatten"

    def test_flatten_closes_every_position_and_settles_inside_the_bound(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10, "BBB": -3})
        pre_fire = ib.working_order("CCC", at=T0 - dt.timedelta(minutes=5))
        _, outcome = _fire(store, ib, "flatten")
        assert ib.book == {} and ib.cancel_calls == 1
        assert pre_fire.orderStatus.status == "Cancelled"
        orders = {
            (c.symbol, o.action, o.totalQuantity, o.orderRef, o.tif) for c, o in ib.order_calls
        }
        assert orders == {
            ("AAA", "SELL", 10, KILL_SWITCH_ORDER_REF, "DAY"),
            ("BBB", "BUY", 3, KILL_SWITCH_ORDER_REF, "DAY"),
        }
        assert outcome.passed and outcome.reached
        assert outcome.settle_seconds == 2.0 and outcome.bound_seconds == SETTLE_BOUND_S["flatten"]
        assert outcome.before.positions == {"AAA": 10, "BBB": -3} and outcome.after.positions == {}
        assert outcome.foreign_orders_after_fire == ()
        assert outcome.to_document()["orders_accepted_after_fire"] == []

    def test_freeze_cancels_places_nothing_and_keeps_positions(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10})
        ib.working_order("AAA", at=T0 - dt.timedelta(minutes=1))
        _, outcome = _fire(store, ib, "freeze")
        assert ib.order_calls == [] and ib.cancel_calls == 1 and ib.book == {"AAA": 10}
        assert outcome.passed and outcome.settle_seconds == 0.0

    def test_hold_leaves_working_orders_alone(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10})
        stop = ib.working_order("AAA", at=T0 - dt.timedelta(hours=1))
        _, outcome = _fire(store, ib, "hold")
        assert ib.cancel_calls == 0 and stop.orderStatus.status == "Submitted"
        assert outcome.passed and read_state(store)["mode"] == "hold"

    def test_an_order_accepted_after_the_fire_fails_it(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10})
        ib.foreign_on_sleep = ["XYZ"]
        _, outcome = _fire(store, ib, "flatten", bound_seconds=6)
        assert not outcome.passed
        assert [o.symbol for o in outcome.foreign_orders_after_fire] == ["XYZ"]
        assert "accepted after the fire instant" in outcome.reason()

    def test_a_book_that_never_goes_flat_is_not_reached(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10}, fills=False)
        _, outcome = _fire(store, ib, "flatten", bound_seconds=6)
        assert not outcome.reached and outcome.settled_at is None and outcome.settle_seconds is None
        assert "did not reach its target within 6s" in outcome.reason()

    def test_an_unknown_mode_and_an_empty_cause_are_refused(self, store) -> None:
        ib = FakeIB()
        ctx = ctx_for(store, KILL_SWITCH_JOB)
        with pytest.raises(ValueError, match="not in"):
            fire(ctx, _broker(ib), mode="pause", cause="x")
        with pytest.raises(ValueError, match="needs a cause"):
            fire(ctx, _broker(ib), mode="freeze", cause=" ")
        assert read_state(store) is None


class TestAcceptedOrders:
    def test_an_order_accepted_at_the_fire_instant_counts_as_after_it(self) -> None:
        """The boundary is inclusive: an order the broker accepted in the same
        instant the switch fired is not provably before it, so it fails the fire."""
        ib = FakeIB()
        ib.working_order("AAA", at=T0)
        assert [o.symbol for o in _broker(ib).orders_accepted_since(T0)] == ["AAA"]

    def test_orders_with_no_accepted_status_are_ignored(self) -> None:
        ib = FakeIB()
        ib.working_order("AAA", status="Inactive")
        assert _broker(ib).orders_accepted_since(T0 - dt.timedelta(days=1)) == []

    def test_a_naive_acceptance_time_is_refused(self) -> None:
        ib = FakeIB()
        ib.working_order("AAA", at=dt.datetime(2026, 9, 8, 10, 0))
        with pytest.raises(KillSwitchStateError, match="naive"):
            _broker(ib).orders_accepted_since(T0)

    def test_a_missing_order_ref_reads_as_empty(self) -> None:
        ib = FakeIB()
        ib.working_order("AAA", ref=None)
        (order,) = _broker(ib).orders_accepted_since(T0)
        assert order == AcceptedOrder(1, "AAA", "BUY", 1.0, "", "2026-09-08T14:00:00.000000Z")
        assert order.to_document()["order_ref"] == ""


class TestRecordOutcome:
    def test_a_passing_outcome_files_the_event_and_an_ok_metric(self, store) -> None:
        ib = FakeIB(positions={"AAA": 1})
        ctx, outcome = _fire(store, ib, "flatten")
        record_outcome(ctx, outcome)
        event = json.loads(store.get_bytes(kill_switch_event_key(DAY.isoformat(), ctx.run_id)))
        assert event["passed"] is True and event["state_after"]["positions"] == {}
        (metric,) = ctx.metrics
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "OK"

    def test_a_failing_outcome_is_filed_and_then_raised(self, store) -> None:
        ib = FakeIB(positions={"AAA": 1}, fills=False)
        ctx, outcome = _fire(store, ib, "flatten", bound_seconds=2)
        with pytest.raises(KillSwitchNotSettledError, match="did not reach"):
            record_outcome(ctx, outcome)
        assert store.exists(kill_switch_event_key(DAY.isoformat(), ctx.run_id))
        (metric,) = ctx.metrics
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "FAIL" and metric["value"] is None


class TestRelease:
    def test_release_disengages_and_records_what_it_released(self, store) -> None:
        ib = FakeIB()
        _fire(store, ib, "freeze")
        ctx = ctx_for(store, KILL_SWITCH_JOB)
        document = release(ctx, cause="drill over", clock=FakeClock())
        assert document["engaged"] is False and "was freeze" in document["cause"]
        assert_trading_permitted(store, "2026-09-08")

    def test_nothing_engaged_is_nothing_to_release(self, store) -> None:
        with pytest.raises(KillSwitchStateError, match="not engaged"):
            release(ctx_for(store, KILL_SWITCH_JOB), cause="x")

    def test_a_release_needs_a_cause(self, store) -> None:
        with pytest.raises(ValueError, match="needs a cause"):
            release(ctx_for(store, KILL_SWITCH_JOB), cause="")


def test_the_default_clock_is_utc_aware() -> None:
    assert utcnow().tzinfo is not None
