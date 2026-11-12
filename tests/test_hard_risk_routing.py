"""Hard-risk forced exits through the router (precedence: hard risk > timing).

A name the hard-risk layer forced out is sold whole, the way the gap stop
(crucible-trader#143) sells a gapped name: ahead of every other sell, and with
timed orders as the session's first batch, never held for an exit trigger. Its
standing protective stop (crucible-trader#140) is cancelled before the sell goes,
and the sell is clipped to the shares still held.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest
from crucible.hard_risk import LOSS_FLOOR_EXIT
from crucible.store import LocalStore
from test_entry_timing import PATHS, TimedIB, _run_day, _timed_router
from test_hard_risk import NO_STATE, _facts, _params
from test_nav_sizing_and_stops import _working_stop
from test_order_router import DAY, RoutingIB, _book, _config, _router

from crucible_trader.entry_timing import entry_watch_key
from crucible_trader.execution_shortfall import decide as sizing_decision
from crucible_trader.hard_risk import (
    CarriedState,
    HardRiskDecision,
    HardRiskInputs,
    HardRiskPrecedenceError,
    decide,
    enforce_precedence,
)
from crucible_trader.order_router import (
    HARD_RISK_EXIT_PREFIX,
    PlannedOrder,
    PriceUnavailableError,
    hard_risk_exits,
)

NOW = dt.datetime(2026, 9, 8, 13, 35, tzinfo=dt.UTC)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _decision(forced: dict[str, str], *, breaker: bool = False) -> HardRiskDecision:
    carried = (
        CarriedState(peak_nav_usd=1_000_000.0, lockouts={}, carried_from="x")
        if breaker
        else NO_STATE
    )
    base = decide(
        _params(),
        HardRiskInputs(_facts(nav=900_000.0 if breaker else 1_000_000.0), carried),
        trading_day=DAY,
        alpha={},
    )
    assert base.circuit_breaker is breaker
    return dataclasses.replace(base, forced=forced, new_exits=forced)


def _order(symbol: str, side: str, quantity: int = 10, price: float = 10.0) -> PlannedOrder:
    return PlannedOrder(
        trading_day=DAY, decision=sizing_decision(symbol, side, quantity, price=price, now=NOW)
    )


def _plan(batches) -> list[list[tuple[str, str, int]]]:
    return [
        [(o.decision.symbol, o.decision.side, int(o.decision.quantity)) for o in b] for b in batches
    ]


# -- the plan ------------------------------------------------------------------------


def test_a_forced_name_is_sold_whole_and_leads_the_sells(store) -> None:
    # CCC: 3000 held at $20 is $60k, over the $50k per-order cap, so the plan
    # alone would sell 2500 and leave 500. Forced out, all 3000 go, first.
    ib = RoutingIB(positions={"BBB": 100, "CCC": 3000})
    router = _router(store, ib, config=_config(gap_stop_pct=None))
    book = dataclasses.replace(
        _book({"AAA": 0.3, "BBB": 0.0, "CCC": 0.0}),
        hard_risk=_decision({"CCC": LOSS_FLOOR_EXIT}),
    )
    assert _plan(router.batches(book)) == [
        [("CCC", "sell", 3000), ("BBB", "sell", 100)],
        [("AAA", "buy", 300)],
    ]


def test_a_forced_name_the_plan_left_alone_gains_an_exit_and_a_full_sell_is_kept() -> None:
    # BBB's planned sell is already whole; CCC has no planned order at all.
    held_sell = _order("BBB", "sell", 100, 50.0)
    sells, buys, reasons = hard_risk_exits(
        _decision({"BBB": LOSS_FLOOR_EXIT, "CCC": "drawdown", "ZZZ": "lockout"}),
        [held_sell],
        [_order("AAA", "buy")],
        positions={"BBB": 100, "CCC": 250},
        quote={"CCC": 20.0}.get,
        clock=lambda: NOW,
    )
    assert sells[0] is held_sell
    assert [(o.decision.symbol, int(o.decision.quantity)) for o in sells] == [
        ("BBB", 100),
        ("CCC", 250),
    ]
    assert [o.decision.symbol for o in buys] == ["AAA"]
    # ZZZ is locked out but not held: nothing to sell, and no trigger for it.
    assert reasons == {
        "BBB": f"{HARD_RISK_EXIT_PREFIX}: {LOSS_FLOOR_EXIT}",
        "CCC": f"{HARD_RISK_EXIT_PREFIX}: drawdown",
    }


def test_a_gap_exit_stays_ahead_of_a_forced_exit_and_is_not_doubled() -> None:
    gap = _order("AAA", "sell", 100, 80.0)
    sells, _buys, reasons = hard_risk_exits(
        _decision({"AAA": LOSS_FLOOR_EXIT, "BBB": LOSS_FLOOR_EXIT}),
        [gap, _order("BBB", "sell", 10, 50.0), _order("CCC", "sell", 5, 20.0)],
        [],
        positions={"AAA": 100, "BBB": 40, "CCC": 5},
        quote=lambda _s: None,
        clock=lambda: NOW,
        skip=frozenset({"AAA"}),
    )
    assert [(o.decision.symbol, int(o.decision.quantity)) for o in sells] == [
        ("AAA", 100),
        ("BBB", 40),
        ("CCC", 5),
    ]
    assert set(reasons) == {"BBB"}


def test_with_nothing_forced_and_held_the_plan_is_unchanged() -> None:
    sells, buys = (_order("BBB", "sell"),), (_order("AAA", "buy"),)
    assert hard_risk_exits(
        _decision({"ZZZ": "lockout"}),
        sells,
        buys,
        positions={"BBB": 10},
        quote=lambda _s: None,
        clock=lambda: NOW,
    ) == (sells, buys, {})


def test_an_unpriced_forced_exit_refuses_the_book() -> None:
    with pytest.raises(PriceUnavailableError, match="forced out"):
        hard_risk_exits(
            _decision({"CCC": LOSS_FLOOR_EXIT}),
            [],
            [],
            positions={"CCC": 10},
            quote=lambda _s: None,
            clock=lambda: NOW,
        )


def test_a_buy_of_a_forced_name_is_refused_at_plan_time(store) -> None:
    router = _router(store, RoutingIB(), config=_config(gap_stop_pct=None))
    book = dataclasses.replace(_book(), hard_risk=_decision({"AAA": "lockout"}))
    with pytest.raises(HardRiskPrecedenceError, match="AAA"):
        router.batches(book)


def test_at_the_breaker_every_buy_is_dropped_and_the_sells_go(capsys) -> None:
    sell = _order("BBB", "sell")
    sells, buys, _ = hard_risk_exits(
        _decision({}, breaker=True),
        [sell],
        [_order("AAA", "buy")],
        positions={"BBB": 10},
        quote=lambda _s: None,
        clock=lambda: NOW,
    )
    assert (sells, buys) == ((sell,), ())
    assert "circuit breaker" in capsys.readouterr().err


# -- at the broker ---------------------------------------------------------------------


def test_a_forced_exit_cancels_the_names_standing_stop_before_it_is_sent(store) -> None:
    ib = RoutingIB(positions={"BBB": 500})
    stop = _working_stop(ib, "BBB", 500)
    router = _router(store, ib, config=_config(gap_stop_pct=None))
    book = dataclasses.replace(
        _book({"AAA": 0.0, "BBB": 0.25, "CCC": 0.0}),  # the plan would keep all 500
        hard_risk=_decision({"BBB": LOSS_FLOOR_EXIT}),
    )
    (sells,) = router.batches(book)
    status_when_sold: list[str] = []
    ib.on_place = lambda: status_when_sold.append(stop.orderStatus.status)
    router.send(sells)
    assert status_when_sold == ["Cancelled"]  # cancelled, confirmed, then sold
    assert ib.cancelled_orders == [stop.order.orderId]
    assert [
        (r.order.decision.symbol, r.order.action, int(r.order.decision.quantity))
        for r in router.records
    ] == [("BBB", "SELL", 500)]


def test_a_forced_exit_is_clipped_to_the_shares_still_held(store, capsys) -> None:
    ib = RoutingIB(positions={"BBB": 500})
    router = _router(store, ib, config=_config(gap_stop_pct=None))
    book = dataclasses.replace(
        _book({"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}), hard_risk=_decision({"BBB": LOSS_FLOOR_EXIT})
    )
    (sells,) = router.batches(book)
    ib.book["BBB"] = 200  # the old stop filled 300 between the plan and the send
    router.send(sells)
    assert [(r.order.decision.symbol, int(r.order.decision.quantity)) for r in router.records] == [
        ("BBB", 200)
    ]
    assert "the sell of 500 is clipped to 200" in capsys.readouterr().err


def test_with_timed_orders_a_forced_exit_goes_at_once_ahead_of_timed_sells(store) -> None:
    # DDD is flat all day and would wait for its 15:30 graduated exit; forced
    # out, it is the session's first batch at 09:35, its stop cancelled first.
    # EEE is a plain exit and still waits for a trigger of its own.
    paths = {**PATHS, "EEE": [(dt.time(9, 30), 10.0)]}
    ib = TimedIB(paths, positions={"DDD": 100, "EEE": 100})
    stop = _working_stop(ib, "DDD", 100)
    router = _timed_router(store, ib, gap_stop_pct=None)
    book = dataclasses.replace(
        _book({"AAA": 0.1, "DDD": 0.0, "EEE": 0.0}),
        hard_risk=_decision({"DDD": LOSS_FLOOR_EXIT}),
    )
    _run_day(router, book)
    placed = ib.placed()
    assert placed[0] == ("DDD", "SELL", 100, "09:35")
    # EEE waited for its own exit trigger, after the session start.
    (eee,) = [p for p in placed if p[0] == "EEE"]
    assert eee[1:3] == ("SELL", 100) and eee[3] > "09:35"
    assert stop.order.orderId in ib.cancelled_orders
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert (rows["DDD"]["status"], rows["DDD"]["trigger"]) == (
        "sent",
        f"{HARD_RISK_EXIT_PREFIX}: {LOSS_FLOOR_EXIT}",
    )
    assert "DDD" not in {symbol for symbol, *_ in ib.stops()}


# -- the chokepoint over a timed plan --------------------------------------------------


def test_a_timed_plan_is_checked_batch_by_batch_and_never_materialized() -> None:
    pulled: list[str] = []

    def timed():
        pulled.append("sell")
        yield (_order("AAA", "sell"),)
        pulled.append("buy")
        yield (_order("AAA", "buy"),)

    admitted = iter(enforce_precedence(timed(), _decision({"AAA": LOSS_FLOOR_EXIT})))
    assert pulled == []  # nothing runs before the session asks for a batch
    assert _plan([next(admitted)]) == [[("AAA", "sell", 10)]]
    assert pulled == ["sell"]
    with pytest.raises(HardRiskPrecedenceError, match="AAA"):
        next(admitted)


def test_at_the_breaker_a_timed_plan_loses_its_buys_and_keeps_its_sells() -> None:
    logged: list[str] = []

    def timed():
        yield (_order("BBB", "sell"),)
        yield (_order("AAA", "buy"),)

    admitted = list(enforce_precedence(timed(), _decision({}, breaker=True), log=logged.append))
    assert _plan(admitted) == [[("BBB", "sell", 10)]]
    assert "AAA" in logged[0]
