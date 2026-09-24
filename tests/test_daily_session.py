"""The daily session on the executor box: a shadow session is a served day.

`alpha-engine-config-I11545`, Brian's rulings of 2026-09-24. The properties:

* a shadow session (routing off) records the day as `shadow`, files a
  `no_orders` shortfall, and places, cancels and closes nothing;
* a held day (nothing published yet) records nothing and does not fail;
* a halted day is not a served day;
* a routed session records `live` and files the router's own shortfall, even
  when it fails after orders reached the broker;
* what it writes is what the phase-4 gate reads.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest
from crucible.execution import execution_shortfall_key
from crucible.keys import TRADER_EVIDENCE_KEY, manifest_key, predictions_key
from crucible.release import TRADER_PIN_KEY
from crucible.store import LocalStore
from ib_fakes import FakeIB, FakeSdk, ctx_for, environ_for
from test_contract import ARM, _feed, _put
from test_contract import _serveable as _contract_holds
from test_session import SHA, VERSION, _halt, _pin_document

from crucible_trader import daily_session as module
from crucible_trader.broker_session import OrderRefusedError
from crucible_trader.contract import ContractRefusal, ContractUnavailable
from crucible_trader.daily_session import (
    SESSION_JOB,
    ReadOnlyHoldControl,
    daily_session_cycle,
    main,
    run_daily_session,
)
from crucible_trader.execution_shortfall import build_no_orders_document
from crucible_trader.kill_switch import TradingHaltedError
from crucible_trader.order_router import RouterConfig, ShadowRouter, build_router

DAY = dt.date(2026, 9, 11)
#: Friday 2026-09-11, 10:15 ET: the session runs pre-close and binds to the
#: last CLOSED session, Thursday 09-10 -- so the fixtures serve 09-10.
NOW = dt.datetime(2026, 9, 11, 14, 15, tzinfo=dt.UTC)
BOUND = "2026-09-10"


def _serveable(store: LocalStore, *, trading_day: str) -> None:
    """The contract holds, with a feed broad enough that the hold-book
    safeguard reads it healthy (`hold_book.HOLD_BOOK_MIN_BATCH`)."""
    _contract_holds(store, trading_day=trading_day)
    feed = _feed(trading_day=trading_day)
    feed["predicted_alpha"] = {f"T{i}": 0.001 * (i + 1) for i in range(10)}
    _put(store, predictions_key(trading_day), feed)


def _construct(store_, **kwargs):
    return SimpleNamespace(s_champion="s:arm:def", metric={"name": "portfolio_construction"})


@pytest.fixture
def store(tmp_path) -> LocalStore:
    s = LocalStore(tmp_path)
    s.put_bytes(TRADER_PIN_KEY, json.dumps(_pin_document(SHA)).encode())
    return s


class LiveRouter:
    """A routed session's router: records orders it 'placed', optionally fails."""

    enabled = True

    def __init__(self, *, records=("placed",), fail: BaseException | None = None) -> None:
        self.records = tuple(records)
        self.fail = fail
        self.closed = 0
        self._client = FakeIB()

    def batches(self, book):
        return [["order"]]

    def send(self, batch):
        if self.fail is not None:
            raise self.fail

    def shortfall_document(self, *, champion, trading_day, now):
        return build_no_orders_document(
            trading_day=trading_day, champion=champion, reason="fixture: routed", now=now
        )

    def close(self):
        self.closed += 1


def _cycle(store, router, *, day=DAY, resolver=None):
    ib = FakeIB()
    ctx = ctx_for(store, SESSION_JOB, day=day)
    kwargs = {} if resolver is None else {"contract_resolver": resolver}
    result = daily_session_cycle(
        ctx,
        broker=ReadOnlyHoldControl(ib, FakeSdk(ib)),
        router=router,
        installed_version=VERSION,
        clock=ib.clock,
        construct=_construct,
        **kwargs,
    )
    return ctx, result, ib


class TestAShadowSessionIsAServedDay:
    def test_it_records_the_day_as_shadow_and_files_no_orders(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        ctx, result, ib = _cycle(store, ShadowRouter())

        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        shortfall = json.loads(store.get_bytes(execution_shortfall_key(DAY.isoformat())))
        assert evidence["session_modes"] == {DAY.isoformat(): "shadow"}
        assert evidence["champion"] == ARM
        assert shortfall["outcome"] == "no_orders"
        assert result["session_mode"] == "shadow"
        assert result["outcome"] == "served"
        assert result["shortfall_outcome"] == "no_orders"
        assert (result["trading_days"], result["shadow_days"]) == (1, 1)
        assert result["batches_sent"] == 0

    def test_both_artifacts_are_outputs_of_the_run(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        ctx, _, _ = _cycle(store, ShadowRouter())

        assert [o["key"] for o in ctx.outputs] == [
            TRADER_EVIDENCE_KEY,
            execution_shortfall_key(DAY.isoformat()),
        ]
        assert ctx.metrics[-1]["name"] == "trader_session_served"
        assert ctx.metrics[-1]["value"] == 1.0

    def test_nothing_touches_the_book(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _, _, ib = _cycle(store, ShadowRouter())

        assert ib.order_calls == [] and ib.cancel_calls == 0


class TestWhatIsNotAServedDay:
    def test_nothing_published_is_a_hold_that_records_nothing(self, store) -> None:
        def unavailable(store_, day):
            raise ContractUnavailable("no champion pointer for slot 'm'")

        ctx, result, _ = _cycle(store, ShadowRouter(), resolver=unavailable)

        assert result == {
            "trading_day": DAY.isoformat(),
            "session_mode": "shadow",
            "outcome": "held",
        }
        assert not store.exists(TRADER_EVIDENCE_KEY)
        assert not store.exists(execution_shortfall_key(DAY.isoformat()))
        assert ctx.metrics[-1]["value"] == 0.0
        assert "no champion pointer" in ctx.metrics[-1]["status_reason"]

    def test_a_refused_contract_fails_the_run_and_records_nothing(self, store) -> None:
        def refused(store_, day):
            raise ContractRefusal("the 'm' champion pointer exists and is unusable")

        with pytest.raises(ContractRefusal):
            _cycle(store, ShadowRouter(), resolver=refused)
        assert not store.exists(TRADER_EVIDENCE_KEY)

    @pytest.mark.parametrize("mode", ["freeze", "flatten"])
    def test_a_halted_day_is_not_served(self, store, mode) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _halt(store, mode, day=DAY.isoformat())

        with pytest.raises(TradingHaltedError):
            _cycle(store, ShadowRouter())
        assert not store.exists(TRADER_EVIDENCE_KEY)


class TestARoutedSession:
    def test_it_records_the_day_as_live_with_the_routers_shortfall(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _, result, _ = _cycle(store, LiveRouter())

        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        shortfall = json.loads(store.get_bytes(execution_shortfall_key(DAY.isoformat())))
        assert evidence["session_modes"] == {DAY.isoformat(): "live"}
        assert shortfall["outcome_reason"] == "fixture: routed"
        assert (result["session_mode"], result["shadow_days"]) == ("live", 0)

    def test_a_failure_after_orders_reached_the_broker_files_them_then_fails(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        with pytest.raises(RuntimeError, match="gateway dropped"):
            _cycle(store, LiveRouter(fail=RuntimeError("gateway dropped")))

        assert store.exists(execution_shortfall_key(DAY.isoformat()))
        assert not store.exists(TRADER_EVIDENCE_KEY)

    def test_a_failure_before_any_order_files_nothing(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        with pytest.raises(RuntimeError):
            _cycle(store, LiveRouter(records=(), fail=RuntimeError("refused")))

        assert not store.exists(execution_shortfall_key(DAY.isoformat()))


class TestTheHoldControlIsReadOnly:
    def test_it_refuses_to_cancel_or_close(self) -> None:
        ib = FakeIB()
        control = ReadOnlyHoldControl(ib, FakeSdk(ib))

        with pytest.raises(OrderRefusedError):
            control.cancel_all()
        with pytest.raises(OrderRefusedError):
            control.submit_close("AAPL", 10)
        assert ib.order_calls == [] and ib.cancel_calls == 0


class TestTheJob:
    def test_a_shadow_run_files_an_ok_manifest_and_disconnects(self, store) -> None:
        _serveable(store, trading_day=BOUND)
        ib = FakeIB()
        ctx = run_daily_session(
            store,
            connector=lambda: ib,
            sdk=FakeSdk(ib),
            router_config=RouterConfig(enabled=False),
            installed_version=VERSION,
            clock=lambda: NOW,
            run_mode="live",
            router_factory=lambda config, **kw: build_router(config, **kw),
        )

        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert ctx.trading_day.isoformat() == BOUND
        assert ib.disconnected == 1
        assert ib.order_calls == []

    def test_a_routed_run_reads_through_the_routers_connection_and_closes_it(
        self, store, monkeypatch
    ) -> None:
        _serveable(store, trading_day=BOUND)
        monkeypatch.setattr(module, "construct_target_book", _construct)
        router = LiveRouter()
        opened: list = []

        run_daily_session(
            store,
            connector=lambda: opened.append("second connection"),
            sdk=FakeSdk(router._client),
            router_config=RouterConfig(enabled=False),
            installed_version=VERSION,
            clock=lambda: NOW,
            run_mode="live",
            router_factory=lambda config, **kw: router,
        )

        assert opened == []
        assert router.closed == 1
        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        assert evidence["session_modes"] == {BOUND: "live"}


class TestTheEntryPoint:
    def test_a_non_trading_day_is_skipped_before_anything_opens(self, tmp_path) -> None:
        said: list[str] = []
        saturday = dt.datetime(2026, 9, 12, 14, 15, tzinfo=dt.UTC)

        assert main([], environ={}, clock=lambda: saturday, printer=said.append) == 0
        assert said and said[0].startswith("SKIP: 2026-09-12")

    def test_routing_stays_off_unless_the_environment_turns_it_on(self, store) -> None:
        """Nothing is published yet, so the session holds -- and it held as a
        SHADOW session: the router the entry point built was the shadow one."""
        ib = FakeIB()
        said: list[str] = []
        seen: list[RouterConfig] = []

        def factory(config, **kw):
            seen.append(config)
            return build_router(config, **kw)

        code = main(
            [],
            environ=environ_for(store.root),
            sdk_loader=FakeSdk(ib),
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
            router_factory=factory,
        )

        assert code == 0
        assert seen == [RouterConfig(enabled=False)]
        assert ib.connected_with["readonly"] is True
        assert said[-1].endswith("shadow session filed")
        assert not store.exists(TRADER_EVIDENCE_KEY)

    def test_a_routed_entry_point_says_so(self, store, monkeypatch) -> None:
        ib = FakeIB()
        said: list[str] = []
        env = environ_for(
            store.root,
            CRUCIBLE_TRADER_ORDER_ROUTING="ib_paper",
            CRUCIBLE_TRADER_IB_ACCOUNT="DU1",
            CRUCIBLE_TRADER_IB_PORT="4002",
            CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD="1000",
            CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD="5000",
        )

        main(
            [],
            environ=env,
            sdk_loader=FakeSdk(ib),
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
            router_factory=lambda config, **kw: LiveRouter(),
        )

        assert said[-1].endswith("routed session filed")
