"""The session composes the PR7 guards: a fired switch, an engaged hold and a
wheel the pin does not name each send no order."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from crucible.release import TRADER_PIN_KEY
from crucible.serving import PredictionsFeed
from crucible.store import LocalStore
from ib_fakes import FakeIB, FakeSdk, ctx_for

from crucible_trader.kill_switch import (
    KILL_SWITCH_KEY,
    IbBrokerControl,
    TradingHaltedError,
    halt_document,
    read_state,
)
from crucible_trader.paper_smoke import RunningWheelMismatchError
from crucible_trader.session import SESSION_SCHEMA_VERSION, session_cycle

SHA = "c" * 40
VERSION = f"0.1.0+g{'c' * 12}"
HEALTHY = {f"T{i}": 0.001 * i for i in range(10)}
DEGENERATE = {f"T{i}": 0.0 for i in range(6)}


class Router:
    def __init__(self, n_batches: int = 2, on_send=None) -> None:
        self.n_batches = n_batches
        self.on_send = on_send
        self.sent: list = []
        self.books: list = []

    def batches(self, book):
        self.books.append(book)
        return ([f"order-{i}"] for i in range(self.n_batches))

    def send(self, batch):
        self.sent.append(batch)
        if self.on_send is not None:
            self.on_send()


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path)
    s.put_bytes(
        TRADER_PIN_KEY,
        json.dumps({"sha": SHA, "target": "trader", "pinned_at": "2026-09-05T22:00:00Z"}).encode(),
    )
    return s


def _contract(alphas):
    feed = PredictionsFeed(
        slot="m",
        trading_day="2026-09-08",
        champion="m:arm:abc",
        feature_version="v1",
        source_key="arms/m/x.json",
        predicted_alpha=alphas,
    )
    return lambda store, day: SimpleNamespace(feed=feed, arm_id=feed.champion, trading_day=day)


def _run(store, ib, router, *, alphas=HEALTHY, version=VERSION, constructed=None):
    calls = [] if constructed is None else constructed

    def construct(store_, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(s_champion="s:arm:def", metric={"name": "portfolio_construction"})

    ctx = ctx_for(store, "trader.session")
    result = session_cycle(
        ctx,
        broker=IbBrokerControl(ib, FakeSdk(ib)),
        router=router,
        installed_version=version,
        previous_weights=None,
        clock=ib.clock,
        contract_resolver=_contract(alphas),
        construct=construct,
    )
    return ctx, result


def _halt(store, mode, day="2026-09-08"):
    store.put_bytes(
        KILL_SWITCH_KEY,
        json.dumps(
            halt_document(
                engaged=True, mode=mode, cause="drill", trading_day=day, since="x", run_id="r"
            )
        ).encode(),
    )


def test_a_clear_session_constructs_and_sends_every_batch(store) -> None:
    ib, router, constructed = FakeIB(), Router(), []
    ctx, result = _run(store, ib, router, constructed=constructed)
    assert router.sent == [["order-0"], ["order-1"]]
    assert result == {
        "schema_version": SESSION_SCHEMA_VERSION,
        "trading_day": "2026-09-08",
        "m_champion": "m:arm:abc",
        "s_champion": "s:arm:def",
        "hold_decision": "proceed_signal_healthy",
        "batches_sent": 2,
    }
    assert constructed[0]["trading_day"] == "2026-09-08" and constructed[0]["now"] == ib.clock()
    assert [m["name"] for m in ctx.metrics] == ["hold_book_engaged", "portfolio_construction"]


@pytest.mark.parametrize("mode", ["flatten", "freeze"])
def test_a_fired_kill_switch_sends_no_order(store, mode) -> None:
    _halt(store, mode, day="2026-09-01")  # persistent modes halt every later day too
    ib, router = FakeIB(), Router()
    with pytest.raises(TradingHaltedError, match=mode):
        _run(store, ib, router)
    assert router.sent == [] and ib.order_calls == []


def test_a_switch_fired_mid_session_stops_the_next_batch(store) -> None:
    router = Router(n_batches=3, on_send=lambda: _halt(store, "freeze"))
    with pytest.raises(TradingHaltedError):
        _run(store, FakeIB(), router)
    assert router.sent == [["order-0"]]


def test_an_engaged_hold_refuses_the_batch_and_constructs_nothing(store) -> None:
    ib, router, constructed = FakeIB(), Router(), []
    with pytest.raises(TradingHaltedError, match="hold"):
        _run(store, ib, router, alphas=DEGENERATE, constructed=constructed)
    assert read_state(store)["mode"] == "hold"
    assert router.sent == [] and constructed == [] and ib.order_calls == []


def test_a_hold_from_an_earlier_session_has_lapsed(store) -> None:
    _halt(store, "hold", day="2026-09-04")
    router = Router(n_batches=1)
    _run(store, FakeIB(), router)
    assert router.sent == [["order-0"]]


@pytest.mark.parametrize(("pin", "match"), [(None, "unset"), ("d" * 40, "running crucible")])
def test_a_wheel_disagreement_fails_the_run_before_anything(tmp_path, pin, match) -> None:
    store = LocalStore(tmp_path)
    if pin is not None:
        store.put_bytes(
            TRADER_PIN_KEY,
            json.dumps(
                {"sha": pin, "target": "trader", "pinned_at": "2026-09-05T22:00:00Z"}
            ).encode(),
        )
    ib, router, constructed = FakeIB(), Router(), []
    with pytest.raises(RunningWheelMismatchError, match=match):
        _run(store, ib, router, constructed=constructed)
    assert router.sent == [] and constructed == [] and ib.order_calls == []
