"""A live session sizes on the account's NAV and leaves every held long under a GTC stop.

The 2026-11-09 v1/v2 trading-parity audit, rows 5 and 1:

* the book is sized on ``min(book_notional_usd, NetLiquidation)``, the declared
  notional being the ceiling, and an unreadable NAV refuses the session;
* after the session's orders, every held long carries exactly one standing
  GTC ``STP`` SELL at the configured distance below the reference price, sized
  so that the stops never exceed the shares held, re-sized when the holding
  changes and left alone on a rerun.

Every test drives the recording fake gateway, so "placed nothing" is an
assertion on `order_calls`, not on the router's own say-so.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from crucible.store import LocalStore
from ib_fakes import FakeSdk
from test_order_router import (
    ACCOUNT,
    DAY,
    PRICES,
    RoutingIB,
    _book,
    _config,
    _env,
    _halt,
    _route,
    _router,
)

from crucible_trader.construction import SessionSizing
from crucible_trader.kill_switch import TradingHaltedError
from crucible_trader.order_router import (
    DEFAULT_STOP_DISTANCE_PCT,
    STOP_REF_PREFIX,
    NavUnavailableError,
    PriceUnavailableError,
    ProtectiveStopError,
    RiskCapExceededError,
    RouterConfig,
    RouterSettingsError,
    read_net_liquidation,
    stop_price_for,
    stop_ref,
)

YESTERDAY = "2026-09-04"


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _stops(ib: RoutingIB) -> list[tuple[str, str, int, float, str]]:
    """Every stop the router placed: (symbol, action, quantity, stop price, tif)."""
    return [
        (c.symbol, o.action, int(o.totalQuantity), o.auxPrice, o.tif)
        for c, o in ib.order_calls
        if getattr(o, "orderType", "") == "STP"
    ]


def _working_stop(ib: RoutingIB, symbol: str, quantity: int, *, day: str = YESTERDAY):
    """A protective stop already standing at the broker, as an earlier session left it."""
    order = FakeSdk.StopOrder("SELL", quantity, 1.0)
    order.orderRef = stop_ref(day, symbol, quantity, 1.0)
    trade = ib.placeOrder(SimpleNamespace(symbol=symbol), order)
    ib.order_calls.clear()
    return trade


def _working_sell(ib: RoutingIB, symbol: str, quantity: int, *, ref: str = "rebalance"):
    order = SimpleNamespace(action="SELL", totalQuantity=quantity, orderRef=ref, tif="DAY")
    trade = ib.placeOrder(SimpleNamespace(symbol=symbol), order)
    ib.order_calls.clear()
    return trade


# -- sizing on live NAV ----------------------------------------------------------------


def test_the_router_sizes_on_the_nav_with_the_declared_notional_as_ceiling(store) -> None:
    low = _router(store, RoutingIB(nav=850_000.0)).session_sizing(1_000_000.0)
    assert low == SessionSizing(ceiling_usd=1_000_000.0, nav_usd=850_000.0)
    assert low.notional_usd == 850_000.0
    high = _router(store, RoutingIB(nav=1_250_000.0)).session_sizing(1_000_000.0)
    assert high.notional_usd == 1_000_000.0


def test_an_unreadable_nav_refuses_the_session_before_any_order(store) -> None:
    ib = RoutingIB(nav=None)
    router = _router(store, ib)
    with pytest.raises(NavUnavailableError, match="found 0"):
        router.session_sizing(1_000_000.0)
    with pytest.raises(NavUnavailableError):
        router.batches(_book())
    assert ib.order_calls == []


def test_a_book_sized_past_the_nav_is_refused_before_any_order(store) -> None:
    ib = RoutingIB(nav=60_000.0)  # the fixture book is sized on $100k
    with pytest.raises(RiskCapExceededError, match="NetLiquidation"):
        _router(store, ib).batches(_book())
    assert ib.order_calls == []


def test_a_book_sized_within_the_nav_routes(store) -> None:
    ib = RoutingIB(nav=100_000.0)
    _route(_router(store, ib), _book())
    assert ib.book == {"AAA": 300, "BBB": 400}


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([("NetLiquidation", "USD", "1e6"), ("NetLiquidation", "USD", "1e6")], "found 2"),
        ([("NetLiquidation", "EUR", "1e6")], "found 0"),
        ([("NetLiquidation", "USD", "lots")], "not a number"),
        ([("NetLiquidation", "USD", None)], "not a number"),
        ([("NetLiquidation", "USD", "0")], "positive"),
        ([("NetLiquidation", "USD", "nan")], "positive"),
    ],
)
def test_the_nav_read_is_exactly_one_positive_usd_row(rows, match) -> None:
    client = SimpleNamespace(
        accountSummary=lambda: (
            [
                SimpleNamespace(account=ACCOUNT, tag=tag, currency=currency, value=value)
                for tag, currency, value in rows
            ]
            + [SimpleNamespace(account="DU2", tag="NetLiquidation", currency="USD", value="5")]
        )
    )
    with pytest.raises(NavUnavailableError, match=match):
        read_net_liquidation(client, ACCOUNT)


# -- the stop distance -------------------------------------------------------------------


def test_the_stop_distance_defaults_to_v1s_bracket_and_is_configurable() -> None:
    assert RouterConfig.from_env(_env()).stop_distance_pct == DEFAULT_STOP_DISTANCE_PCT == 0.20
    config = RouterConfig.from_env(_env(CRUCIBLE_TRADER_STOP_DISTANCE_PCT="0.15"))
    assert config.stop_distance_pct == 0.15


@pytest.mark.parametrize("value", ["0", "-0.1", "lots", "inf"])
def test_a_malformed_stop_distance_is_refused_from_the_environment(value) -> None:
    with pytest.raises(RouterSettingsError):
        RouterConfig.from_env(_env(CRUCIBLE_TRADER_STOP_DISTANCE_PCT=value))


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, float("nan")])
def test_a_stop_distance_outside_zero_one_is_refused(value) -> None:
    with pytest.raises(RouterSettingsError, match="stop_distance_pct"):
        _config(stop_distance_pct=value)


def test_the_stop_price_is_floored_to_the_cent_and_must_be_positive() -> None:
    assert stop_price_for(100.0, 0.20) == 80.0
    assert stop_price_for(12.345, 0.20) == 9.87
    with pytest.raises(PriceUnavailableError, match="not positive"):
        stop_price_for(0.01, 0.20)


# -- standing protective stops ---------------------------------------------------------


def test_every_held_long_gets_one_gtc_stop_after_the_sessions_orders(store) -> None:
    ib = RoutingIB(positions={"BBB": 500, "CCC": 100})
    router = _router(store, ib)
    _route(router, _book())
    results = router.protect_holdings()

    # AAA bought 300 at 100; BBB cut to 400 at 50; CCC sold whole.
    assert _stops(ib) == [("AAA", "SELL", 300, 80.0, "GTC"), ("BBB", "SELL", 400, 40.0, "GTC")]
    for _, order in ib.order_calls[-2:]:
        assert order.account == ACCOUNT and order.outsideRth is False and order.transmit is True
    assert [r["disposition"] for r in results] == ["placed", "placed"]
    assert results[0]["order_ref"] == f"{STOP_REF_PREFIX}|{DAY}|AAA|300|80.00"
    assert results[0]["reference_price"] == PRICES["AAA"]


def test_a_rerun_keeps_todays_stops_and_places_nothing(store) -> None:
    ib = RoutingIB(positions={"AAA": 300, "BBB": 400})
    router = _router(store, ib)
    _route(router, _book())
    router.protect_holdings()
    placed = len(ib.order_calls)

    again = _router(store, ib)
    _route(again, _book())
    results = again.protect_holdings()
    assert len(ib.order_calls) == placed and ib.cancelled_orders == []
    assert [r["disposition"] for r in results] == ["kept", "kept"]


def test_a_stop_is_replaced_when_the_holding_or_the_day_changes(store) -> None:
    ib = RoutingIB(positions={"AAA": 300, "BBB": 400, "ZZZ": 7})
    stale_day = _working_stop(ib, "AAA", 300)  # right size, an earlier day
    wrong_size = _working_stop(ib, "BBB", 250, day=DAY)
    gone = _working_stop(ib, "QQQ", 10)  # a name no longer held
    router = _router(store, ib)
    router._managed = frozenset({"AAA", "BBB"})  # noqa: SLF001 - as `batches` sets it
    results = {r["symbol"]: r for r in router.protect_holdings()}

    for trade in (stale_day, wrong_size, gone):
        assert trade.orderStatus.status == "Cancelled"
    assert _stops(ib) == [("AAA", "SELL", 300, 80.0, "GTC"), ("BBB", "SELL", 400, 40.0, "GTC")]
    assert results["QQQ"]["disposition"] == "cancelled"
    assert "ZZZ" not in results  # held, but not a name this trader manages


def test_two_stops_on_one_name_are_both_cancelled_and_one_placed(store) -> None:
    ib = RoutingIB(positions={"AAA": 300})
    first, second = _working_stop(ib, "AAA", 300, day=DAY), _working_stop(ib, "AAA", 300)
    router = _router(store, ib)
    router._managed = frozenset({"AAA"})  # noqa: SLF001
    router.protect_holdings()
    assert first.orderStatus.status == second.orderStatus.status == "Cancelled"
    assert _stops(ib) == [("AAA", "SELL", 300, 80.0, "GTC")]


def test_a_session_sell_cancels_the_names_stop_before_it_is_sent(store) -> None:
    ib = RoutingIB(positions={"BBB": 500})
    stop = _working_stop(ib, "BBB", 500)
    router = _router(store, ib)
    sells, _buys = router.batches(_book())
    status_when_sold: list[str] = []
    ib.on_place = lambda: status_when_sold.append(stop.orderStatus.status)
    router.send(sells)
    assert status_when_sold == ["Cancelled"]  # cancelled, confirmed, then sold
    assert ib.cancelled_orders == [stop.order.orderId]


def test_stops_never_exceed_the_shares_held_less_other_working_sells(store) -> None:
    ib = RoutingIB(positions={"AAA": 300, "BBB": 400})
    _working_sell(ib, "AAA", 100)
    partly = _working_sell(ib, "BBB", 500)
    partly.orderStatus.filled = 100.0  # 400 still working against 400 held
    router = _router(store, ib)
    router._managed = frozenset({"AAA", "BBB"})  # noqa: SLF001
    results = {r["symbol"]: r for r in router.protect_holdings()}
    assert _stops(ib) == [("AAA", "SELL", 200, 80.0, "GTC")]
    assert results["BBB"] == {"symbol": "BBB", "disposition": "cancelled", "cancelled": []}


class _StuckCancelIB(RoutingIB):
    def cancelOrder(self, order):  # noqa: N802 - SDK spelling
        self.cancelled_orders.append(order.orderId)  # acknowledged, never confirmed


def test_an_unconfirmed_cancel_places_no_replacement_and_pages(store) -> None:
    ib = _StuckCancelIB(positions={"AAA": 300})
    _working_stop(ib, "AAA", 250)
    router = _router(store, ib, fill_timeout_s=1.0)
    router._managed = frozenset({"AAA"})  # noqa: SLF001
    with pytest.raises(ProtectiveStopError, match="not confirmed cancelled"):
        router.protect_holdings()
    assert _stops(ib) == []


def test_a_name_that_cannot_be_protected_pages_after_the_others_are(store) -> None:
    ib = RoutingIB(positions={"AAA": 300, "BBB": 400, "CCC": 5})
    ib.unqualified.add("CCC")
    router = _router(store, ib, quote={"AAA": 100.0, "CCC": 20.0}.get)
    router._managed = frozenset({"AAA", "BBB", "CCC"})  # noqa: SLF001
    with pytest.raises(ProtectiveStopError, match="2 held name") as exc:
        router.protect_holdings()
    assert "BBB: no reference price" in str(exc.value) and "could not qualify CCC" in str(exc.value)
    assert _stops(ib) == [("AAA", "SELL", 300, 80.0, "GTC")]


def test_an_engaged_kill_switch_places_no_stop(store) -> None:
    ib = RoutingIB(positions={"AAA": 300})
    router = _router(store, ib)
    router._managed = frozenset({"AAA"})  # noqa: SLF001
    _halt(store, "freeze")
    with pytest.raises(TradingHaltedError):
        router.protect_holdings()
    assert ib.order_calls == []


def test_a_stop_is_never_mistaken_for_a_session_order(store) -> None:
    """A standing stop's ref must not be adopted as (or refused as) a session order."""
    ib = RoutingIB(positions={"AAA": 300})
    _working_stop(ib, "AAA", 300, day=DAY)
    router = _router(store, ib)
    assert router._known_refs() == {}  # noqa: SLF001
