"""Entry and exit timing: buys held for v1's triggers, sells for their mirror, nothing sent twice.

The router tests drive a whole simulated trading day on a fake clock: the fake
gateway's ``sleep`` advances the clock, its tickers read a scripted price path,
and every order call is recorded, so "waited until 15:30" is an assertion on
when the order reached the fake broker, not on the router's own say-so.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from crucible.keys import data_panel_key
from crucible.store import LocalStore
from ib_fakes import FakeClock, FakeSdk
from test_order_router import ACCOUNT, DAY, RoutingIB, _book, _config, _env, _halt

from crucible_trader.entry_timing import (
    ENTRY_WATCH_SCHEMA_VERSION,
    NYSE_ZONE,
    EntrySchedule,
    EntryTimingParams,
    EntryWatch,
    IbIntradayQuotes,
    IntradayQuote,
    WatchedEntry,
    entry_trigger,
    entry_watch_key,
    exit_trigger,
    panel_support_levels,
)
from crucible_trader.kill_switch import TradingHaltedError
from crucible_trader.order_intent import claim_order_intent, order_intent_document
from crucible_trader.order_router import (
    ENTRY_TIMING_VAR,
    PaperOrderRouter,
    PriceUnavailableError,
    RouterConfig,
    RouterSettingsError,
    RoutingRefusedError,
    UnknownOrderStateError,
    book_cash,
    order_key,
)

PARAMS = EntryTimingParams()
SESSION = dt.date(2026, 9, 8)  # a regular Tuesday session


def et(hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime(2026, 9, 8, hour, minute, tzinfo=NYSE_ZONE)


SCHEDULE = PARAMS.schedule(SESSION)


# -- parameters and schedule ----------------------------------------------------------


def test_the_defaults_are_what_v1_ran() -> None:
    assert PARAMS.document() == {
        "pullback_pct": 0.02,
        "vwap_discount_pct": 0.005,
        "support_pct": 0.01,
        "support_lookback_days": 20,
        "graduated_start_et": "15:30",
        "graduated_max_premium_pct": 0.01,
        "expiry_et": "15:55",
        "poll_interval_s": 60.0,
        "rally_pct": 0.02,
        "exit_graduated_max_discount_pct": 0.01,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pullback_pct", 0.0, "pullback_pct"),
        ("pullback_pct", 1.0, "pullback_pct"),
        ("graduated_max_premium_pct", -0.01, "graduated_max_premium_pct"),
        ("graduated_start_et", dt.time(15, 56), "inside the session"),
        ("expiry_et", dt.time(16, 0), "inside the session"),
        ("graduated_start_et", dt.time(9, 30), "inside the session"),
        ("vwap_discount_pct", 0.0, "vwap_discount_pct"),
        ("support_pct", 1.0, "support_pct"),
        ("support_lookback_days", 0, "support_lookback_days"),
        ("poll_interval_s", 0.0, "poll_interval_s"),
        ("poll_interval_s", float("nan"), "poll_interval_s"),
        ("rally_pct", 0.0, "rally_pct"),
        ("rally_pct", 1.0, "rally_pct"),
        ("exit_graduated_max_discount_pct", -0.01, "exit_graduated_max_discount_pct"),
        ("exit_graduated_max_discount_pct", 1.0, "exit_graduated_max_discount_pct"),
    ],
)
def test_a_parameter_outside_its_range_is_refused(field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(PARAMS, **{field: value})


def test_a_regular_day_runs_to_the_clock_times() -> None:
    assert SCHEDULE == EntrySchedule(
        opens_at=et(9, 30), graduated_at=et(15, 30), expires_at=et(15, 55), closes_at=et(16)
    )
    assert SCHEDULE.document()["expires_at"] == "2026-09-08T15:55:00-04:00"


def test_a_half_day_moves_every_instant_with_the_close() -> None:
    schedule = PARAMS.schedule(SESSION, close_et=lambda day: dt.time(13, 0))
    assert (schedule.graduated_at, schedule.expires_at, schedule.closes_at) == (
        et(12, 30),
        et(12, 55),
        et(13),
    )
    assert schedule.opens_at == et(9, 30)


def test_the_calendar_knows_the_day_after_thanksgiving_closes_early() -> None:
    assert PARAMS.schedule(dt.date(2026, 11, 27)).expires_at.time() == dt.time(12, 55)


# -- the triggers ---------------------------------------------------------------------


def _fires(last: float | None, high: float | None, at: dt.datetime, ref: float = 100.0):
    return entry_trigger(
        reference_price=ref,
        quote=IntradayQuote(last=last, high=high),
        now=at,
        schedule=SCHEDULE,
        params=PARAMS,
    )


def test_a_two_percent_pullback_from_the_high_releases_the_buy() -> None:
    assert _fires(98.0, 100.0, et(10)) == "pullback 2.00% from high 100.00"
    assert _fires(98.1, 100.0, et(10)) is None


def test_nothing_is_released_before_the_open_or_at_the_close() -> None:
    assert _fires(90.0, 100.0, et(9, 29)) is None
    assert _fires(90.0, 100.0, et(16)) is None


def test_the_graduated_window_takes_a_price_within_one_percent_of_the_sized_price() -> None:
    assert _fires(101.0, 101.5, et(15, 29)) is None
    assert _fires(101.0, 101.5, et(15, 30)).startswith("graduated_entry (+1.00%")
    assert _fires(101.5, 101.5, et(15, 30)) is None
    assert _fires(None, 101.5, et(15, 30)) is None


def test_the_expiry_releases_whatever_the_price_and_without_one() -> None:
    assert _fires(150.0, 150.0, et(15, 55)) == "time_expiry"
    assert _fires(None, None, et(15, 55)) == "time_expiry"


def test_a_pullback_is_named_before_the_expiry() -> None:
    assert _fires(90.0, 100.0, et(15, 58)).startswith("pullback")


def _exits(last: float | None, low: float | None, at: dt.datetime, ref: float = 100.0):
    return exit_trigger(
        reference_price=ref,
        quote=IntradayQuote(last=last, high=None, low=low),
        now=at,
        schedule=SCHEDULE,
        params=PARAMS,
    )


def test_a_two_percent_rally_off_the_low_releases_the_sell() -> None:
    assert _exits(102.0, 100.0, et(10)) == "rally 2.00% off low 100.00"
    assert _exits(101.9, 100.0, et(10)) is None
    assert _exits(None, 100.0, et(10)) is None
    assert _exits(102.0, None, et(10)) is None


def test_no_sell_is_released_before_the_open_or_at_the_close() -> None:
    assert _exits(110.0, 100.0, et(9, 29)) is None
    assert _exits(110.0, 100.0, et(16)) is None


def test_the_graduated_exit_takes_a_price_within_one_percent_below_the_sized_price() -> None:
    assert _exits(99.0, 99.0, et(15, 29)) is None
    assert _exits(99.0, 99.0, et(15, 30)) == (
        "graduated_exit (-1.00% vs sized price 100.00, limit -1.00%)"
    )
    assert _exits(105.0, 105.0, et(15, 30)).startswith("graduated_exit (+5.00%")
    assert _exits(98.9, 98.9, et(15, 30)) is None
    assert _exits(None, 98.9, et(15, 30)) is None


def test_the_expiry_releases_every_sell_whatever_the_price_and_without_one() -> None:
    assert _exits(50.0, 50.0, et(15, 55)) == "time_expiry"
    assert _exits(None, None, et(15, 55)) == "time_expiry"
    assert _exits(103.0, 100.0, et(15, 58)).startswith("rally")  # named before the expiry


def _quoted(
    quote: IntradayQuote, at: dt.datetime | None = None, *, support: float | None = None
) -> str | None:
    return entry_trigger(
        reference_price=100.0,
        quote=quote,
        now=at or et(10),
        schedule=SCHEDULE,
        params=PARAMS,
        support_level=support,
    )


def test_half_a_percent_under_the_vwap_releases_the_buy() -> None:
    assert _quoted(IntradayQuote(last=99.5, high=99.6, vwap=100.0)) == (
        "vwap_discount 0.50% below VWAP 100.00"
    )
    assert _quoted(IntradayQuote(last=99.6, high=99.6, vwap=100.0)) is None


def test_no_vwap_only_silences_the_vwap_trigger() -> None:
    assert _quoted(IntradayQuote(last=99.0, high=99.0, vwap=None)) is None
    assert _quoted(IntradayQuote(last=99.0, high=99.0, vwap=float("nan"))) is None
    assert _quoted(IntradayQuote(last=99.0, high=99.0), et(15, 55)) == "time_expiry"


def test_a_price_within_one_percent_above_support_releases_the_buy() -> None:
    quote = IntradayQuote(last=101.0, high=101.0, low=100.5)
    assert _quoted(quote, support=100.0) == "support_bounce 1.00% above support 100.00"
    assert _quoted(IntradayQuote(last=101.1, high=101.1, low=100.5), support=100.0) is None
    # Exactly at support is a bounce; no day low read is not a break (v1).
    assert _quoted(IntradayQuote(last=100.0, high=100.0), support=100.0) is not None


def test_a_support_the_day_has_broken_is_not_a_bounce() -> None:
    assert _quoted(IntradayQuote(last=100.5, high=100.5, low=99.9), support=100.0) is None
    assert _quoted(IntradayQuote(last=99.9, high=99.9, low=99.9), support=100.0) is None


def test_no_support_level_only_silences_the_support_trigger() -> None:
    assert _quoted(IntradayQuote(last=100.5, high=100.5, low=100.5), support=None) is None


def test_the_triggers_are_named_in_v1s_order() -> None:
    quote = IntradayQuote(last=98.0, high=100.0, low=98.0, vwap=99.0)
    assert _quoted(quote, support=97.5).startswith("pullback")
    quote = IntradayQuote(last=98.5, high=99.0, low=98.0, vwap=99.0)
    assert _quoted(quote, support=98.0).startswith("vwap_discount")


def test_an_unusable_price_is_no_price() -> None:
    assert _fires(float("nan"), 100.0, et(10)) is None
    assert _fires(-1.0, 100.0, et(10)) is None


# -- the watched entry and its document -----------------------------------------------


def _watched(symbol: str = "AAA") -> WatchedEntry:
    return WatchedEntry(
        symbol=symbol,
        order_key=order_key(DAY, symbol),
        quantity=10,
        reference_price=100.0,
        decided_at=et(9, 35),
    )


def test_the_high_is_never_below_a_last_price_seen() -> None:
    entry = _watched()
    assert entry.observe(IntradayQuote(last=101.0, high=None, market_data_type=3)) == (
        IntradayQuote(last=101.0, high=101.0, low=101.0)
    )
    assert entry.observe(IntradayQuote(last=None, high=None)) == IntradayQuote(
        None, 101.0, low=101.0
    )
    seen = entry.observe(IntradayQuote(last=99.0, high=100.5))
    assert seen == IntradayQuote(last=99.0, high=101.0, low=99.0)
    assert (entry.last, entry.market_data_type) == (99.0, 3)


def test_the_low_is_never_above_a_last_price_and_the_vwap_is_the_latest() -> None:
    entry = _watched()
    entry.observe(IntradayQuote(last=100.0, high=100.0, low=99.5, vwap=99.8))
    seen = entry.observe(IntradayQuote(last=99.0, high=None, low=None, vwap=None))
    assert seen == IntradayQuote(last=99.0, high=100.0, low=99.0, vwap=99.8)
    entry.observe(IntradayQuote(last=None, high=None, low=98.0, vwap=99.7))
    assert (entry.low, entry.vwap) == (98.0, 99.7)
    row = entry.document()
    assert (row["low"], row["vwap"], row["support_level"]) == (98.0, 99.7, None)


class _RefusingStore(LocalStore):
    def put_bytes(self, key, payload):
        raise PermissionError("AccessDenied")


def _watch(store) -> EntryWatch:
    return EntryWatch(
        store, trading_day=DAY, params=PARAMS, schedule=SCHEDULE, entries=[_watched()]
    )


def test_the_watch_document_names_every_waiting_buy(tmp_path) -> None:
    store = LocalStore(tmp_path)
    watch = _watch(store)
    watch.write(et(9, 35))
    document = json.loads(store.get_bytes(entry_watch_key(DAY)))
    assert document["schema_version"] == ENTRY_WATCH_SCHEMA_VERSION
    assert (document["state"], document["session_date"]) == ("watching", "2026-09-08")
    assert document["summary"] == {"waiting": 1}
    (row,) = document["entries"]
    assert (row["symbol"], row["status"], row["quantity"]) == ("AAA", "waiting", 10)
    assert row["decided_at_utc"] == "2026-09-08T13:35:00Z" and row["released_at_utc"] is None


def test_a_heartbeat_write_waits_for_the_heartbeat(tmp_path) -> None:
    store = LocalStore(tmp_path)
    watch = _watch(store)
    watch.write(et(9, 35), force=False)
    watch.entries[0].last = 1.0
    watch.write(et(9, 40), force=False)
    assert json.loads(store.get_bytes(watch.key))["entries"][0]["last"] is None
    watch.write(et(9, 46), force=False)
    assert json.loads(store.get_bytes(watch.key))["entries"][0]["last"] == 1.0


def test_a_watch_document_that_cannot_be_written_never_stops_the_watch(tmp_path, capsys) -> None:
    watch = _watch(_RefusingStore(tmp_path))
    watch.write(et(9, 35))
    assert watch.write_failures == 1
    assert "was not written (PermissionError" in capsys.readouterr().err


def test_a_watch_key_is_an_iso_day() -> None:
    with pytest.raises(ValueError):
        entry_watch_key("2026-9-8")


# -- the IB quote stream --------------------------------------------------------------


class _Ticker:
    def __init__(self, ib: TimedIB, symbol: str) -> None:
        self._ib, self._symbol = ib, symbol
        self.marketDataType = 3

    @property
    def last(self) -> float:
        return self._ib.price_at(self._symbol)

    @property
    def high(self) -> float:
        return self._ib.high_at(self._symbol)

    @property
    def low(self) -> float:
        return min(self._ib._seen(self._symbol))

    @property
    def vwap(self) -> float | None:
        now = self._ib.clock().astimezone(NYSE_ZONE).time()
        seen = [v for at, v in self._ib.vwaps.get(self._symbol, []) if at <= now]
        return seen[-1] if seen else None

    close = float("nan")


class TimedIB(RoutingIB):
    """A paper gateway on a scripted intraday price path, one entry per (ET time, price)."""

    def __init__(self, paths: dict[str, list[tuple[dt.time, float]]], **kwargs) -> None:
        self.paths = paths
        clock = kwargs.pop("clock", None) or FakeClock(et(9, 35).astimezone(dt.UTC))
        super().__init__(prices={s: p[0][1] for s, p in paths.items()}, clock=clock, **kwargs)
        self.connected = True
        self.subscribed: list[str] = []
        self.tick_lists: list[str] = []
        #: The session VWAP each name's stream carries over the day (absent: none streamed).
        self.vwaps: dict[str, list[tuple[dt.time, float]]] = {}

    def _seen(self, symbol: str) -> list[float]:
        now = self.clock().astimezone(NYSE_ZONE).time()
        return [price for at, price in self.paths[symbol] if at <= now] or [
            self.paths[symbol][0][1]
        ]

    def price_at(self, symbol: str) -> float:
        return self._seen(symbol)[-1]

    def high_at(self, symbol: str) -> float:
        return max(self._seen(symbol))

    def isConnected(self):  # noqa: N802 - SDK spelling
        return self.connected

    def reqMktData(self, contract, tick_list, snapshot, regulatory):  # noqa: N802
        self.subscribed.append(contract.symbol)
        self.tick_lists.append(tick_list)
        return _Ticker(self, contract.symbol)

    def sleep(self, seconds):
        self.prices = {s: self.price_at(s) for s in self.paths}
        super().sleep(seconds)

    def placed(self) -> list[tuple[str, str, int, str]]:
        """(symbol, action, shares, ET hh:mm) for every order, in order."""
        return [
            (
                c.symbol,
                o.action,
                int(o.totalQuantity),
                t.log[0].time.astimezone(NYSE_ZONE).strftime("%H:%M"),
            )
            for (c, o), t in zip(self.order_calls, self._trades, strict=False)
        ]


def test_the_quote_stream_reads_last_and_high_and_skips_an_unqualifiable_name() -> None:
    ib = TimedIB({"AAA": [(dt.time(9, 30), 100.0), (dt.time(9, 32), 103.0)]})
    ib.unqualified.add("ZZZ")
    quotes = IbIntradayQuotes(ib, FakeSdk(ib))
    quotes.subscribe(["AAA", "ZZZ"])
    assert ib.subscribed == ["AAA"] and ib.market_data_type == 3
    assert ib.tick_lists == ["233"]  # RTVolume: the stream carries the VWAP
    ib.vwaps["AAA"] = [(dt.time(9, 30), 101.5)]
    assert quotes("AAA") == IntradayQuote(
        last=103.0, high=103.0, market_data_type=3, low=100.0, vwap=101.5
    )
    assert quotes("ZZZ") == IntradayQuote(last=None, high=None)
    quotes.close()
    quotes.close()  # idempotent
    assert ib.cancelled_mkt == ["AAA"]


def test_a_failed_unsubscribe_never_blocks_teardown() -> None:
    ib = TimedIB({"AAA": [(dt.time(9, 30), 100.0)]})
    quotes = IbIntradayQuotes(ib, FakeSdk(ib))
    quotes.subscribe(["AAA"])

    def refuse(contract):
        raise ConnectionError("gone")

    ib.cancelMktData = refuse
    quotes.close()


def test_a_ticker_without_a_data_type_reads_as_unknown() -> None:
    ib = TimedIB({"AAA": [(dt.time(9, 30), 100.0)]})
    ib.reqMktData = lambda *a: SimpleNamespace(last=100.0, high=None)
    quotes = IbIntradayQuotes(ib, FakeSdk(ib))
    quotes.subscribe(["AAA"])
    assert quotes("AAA") == IntradayQuote(last=100.0, high=None, market_data_type=None)
    assert quotes("AAA").vwap is None and quotes("AAA").low is None


# -- the router, over a whole simulated day --------------------------------------------


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _timed_router(store, ib: TimedIB, **config) -> PaperOrderRouter:
    return PaperOrderRouter(
        _config(**{"entry_timing": PARAMS, **config}),
        client=ib,
        sdk=FakeSdk(ib),
        store=store,
        trading_day=DAY,
        clock=ib.clock,
    )


def _run_day(router: PaperOrderRouter, book) -> int:
    batches = 0
    for batch in router.batches(book):
        router.send(batch)
        batches += 1
    return batches


#: AAA pulls back 2% at 11:00; BBB drifts within 1% and goes at 15:30; CCC rallies
#: 5% and only the expiry takes it. DDD is held, flat all day, and sold at 15:30 (its
#: graduated exit).
PATHS = {
    "AAA": [(dt.time(9, 30), 100.0), (dt.time(10, 0), 101.0), (dt.time(11, 0), 98.9)],
    "BBB": [(dt.time(9, 30), 50.0), (dt.time(12, 0), 50.3)],
    "CCC": [(dt.time(9, 30), 20.0), (dt.time(12, 0), 21.0)],
    "DDD": [(dt.time(9, 30), 10.0)],
}
BOOK = {"AAA": 0.1, "BBB": 0.1, "CCC": 0.1, "DDD": 0.0}


def test_each_sell_and_each_buy_waits_for_its_trigger(store) -> None:
    ib = TimedIB(PATHS, positions={"DDD": 100})
    router = _timed_router(store, ib)
    batches = _run_day(router, _book(BOOK))

    assert ib.placed() == [
        ("AAA", "BUY", 100, "11:00"),
        ("DDD", "SELL", 100, "15:30"),  # the sells of a poll go before its buys
        ("BBB", "BUY", 200, "15:30"),
        ("CCC", "BUY", 500, "15:55"),
    ]
    assert batches == 4
    for _, order in ib.order_calls:
        assert order.orderRef.startswith(f"crucible-trader|{DAY}|")
    # The decision is the session-start one, so the shortfall measures the wait.
    ccc = next(r for r in router.records if r.order.decision.symbol == "CCC")
    assert ccc.order.decision.decision_price == 20.0
    assert ccc.fill_price == pytest.approx(21.0 * 1.01)

    document = json.loads(store.get_bytes(entry_watch_key(DAY)))
    assert document["state"] == "complete" and document["summary"] == {"sent": 4}
    assert document["schema_version"] == "entry_watch.v2"
    sides = {row["symbol"]: row["side"] for row in document["entries"]}
    assert sides == {"AAA": "buy", "BBB": "buy", "CCC": "buy", "DDD": "sell"}
    triggers = {row["symbol"]: row["trigger"] for row in document["entries"]}
    assert triggers["DDD"].startswith("graduated_exit (+0.00%")
    assert triggers["AAA"].startswith("pullback 2.08% from high 101.00")
    assert triggers["BBB"].startswith("graduated_entry (+0.60%")
    assert triggers["CCC"] == "time_expiry"
    assert sorted(ib.cancelled_mkt).count("AAA") >= 2  # the decision quote and the stream
    key, payload = router.entry_watch_payload(ib.clock())
    assert key == entry_watch_key(DAY) and json.loads(payload)["state"] == "complete"


def test_a_book_with_no_order_is_not_watched(store) -> None:
    ib = TimedIB(PATHS)
    router = _timed_router(store, ib)
    assert _run_day(router, _book({"DDD": 0.0})) == 0
    assert router.entry_watch_payload(ib.clock()) is None
    assert not store.exists(entry_watch_key(DAY))


def test_a_sell_is_released_on_a_two_percent_rally_off_the_low(store) -> None:
    path = [(dt.time(9, 30), 10.0), (dt.time(10, 0), 9.5), (dt.time(11, 0), 9.7)]
    ib = TimedIB({"DDD": path}, positions={"DDD": 100})
    router = _timed_router(store, ib)
    assert _run_day(router, _book({"DDD": 0.0})) == 1
    assert ib.placed() == [("DDD", "SELL", 100, "11:00")]
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert (row["side"], row["status"], row["low"]) == ("sell", "sent", 9.5)
    assert row["trigger"] == "rally 2.11% off low 9.50"
    assert (row["released_quantity"], row["release_price"]) == (100, 9.7)


def test_a_sell_that_only_falls_goes_at_the_expiry(store) -> None:
    """The backstop: a sell never rallies or comes back within 1%, and still goes by 15:55."""
    path = [(dt.time(9, 30), 10.0), (dt.time(10, 0), 9.8), (dt.time(13, 0), 9.5)]
    ib = TimedIB({"DDD": path}, positions={"DDD": 100})
    router = _timed_router(store, ib)
    _run_day(router, _book({"DDD": 0.0}))
    assert ib.placed() == [("DDD", "SELL", 100, "15:55")]
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert row["trigger"] == "time_expiry"


def test_a_sell_within_one_percent_of_its_sized_price_goes_at_the_graduated_start(store) -> None:
    path = [(dt.time(9, 30), 10.0), (dt.time(10, 0), 9.8), (dt.time(13, 0), 9.92)]
    ib = TimedIB({"DDD": path}, positions={"DDD": 100})
    router = _timed_router(store, ib)
    _run_day(router, _book({"DDD": 0.0}))
    # 9.8 -> 9.92 is a 1.2% rally (not 2%); at 15:30 9.92 is 0.8% under the sized 10.00.
    assert ib.placed() == [("DDD", "SELL", 100, "15:30")]
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert row["trigger"].startswith("graduated_exit (-0.80% vs sized price 10.00")


def test_a_book_of_buys_only_opens_with_no_batch(store) -> None:
    ib = TimedIB(PATHS)
    router = _timed_router(store, ib)
    assert _run_day(router, _book({"CCC": 0.1})) == 1
    assert ib.placed() == [("CCC", "BUY", 500, "15:55")]


def test_the_kill_switch_stops_the_waiting_orders(store) -> None:
    ib = TimedIB(PATHS, positions={"DDD": 100})
    router = _timed_router(store, ib)
    batches = iter(router.batches(_book(BOOK)))
    router.send(next(batches))  # AAA at its 11:00 pullback
    _halt(store, "freeze")
    with pytest.raises(TradingHaltedError):
        next(batches)
    assert [c.symbol for c, _ in ib.order_calls] == ["AAA"]  # the waiting sell too
    assert json.loads(store.get_bytes(entry_watch_key(DAY)))["state"] == "stopped"


def test_a_dropped_gateway_stops_the_watch_and_sends_nothing_more(store) -> None:
    ib = TimedIB(PATHS)
    router = _timed_router(store, ib)
    batches = iter(router.batches(_book(BOOK)))
    ib.connected = False
    with pytest.raises(RoutingRefusedError, match="3 order"):
        next(batches)
    assert ib.order_calls == []
    router.close()
    assert ib.disconnected == 1


def test_a_buy_still_waiting_at_the_close_is_not_sent(store) -> None:
    clock = FakeClock(et(16, 0).astimezone(dt.UTC))
    ib = TimedIB(PATHS, clock=clock)
    router = _timed_router(store, ib)
    assert _run_day(router, _book({"AAA": 0.1})) == 0
    assert ib.order_calls == []
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert (row["status"], row["trigger"]) == ("not_sent", "market closed before a trigger")


def test_a_buy_whose_price_ran_past_the_per_order_cap_is_clipped_to_it(store, capsys) -> None:
    ib = TimedIB(PATHS)
    router = _timed_router(store, ib, max_order_notional_usd=10_000.0)
    _run_day(router, _book({"CCC": 0.1}))
    # Sized at $20 to $10,000 (500 shares); at $21 the cap holds 476.
    assert ib.placed() == [("CCC", "BUY", 476, "15:55")]
    (record,) = router.records
    assert record.order.decision.decision_price == 20.0 and record.order.decision.quantity == 476
    assert "clipped to 476" in capsys.readouterr().err


def test_a_buy_the_session_cap_has_no_room_for_is_not_sent(store) -> None:
    paths = {**PATHS, "EEE": [(dt.time(9, 30), 10.0), (dt.time(12, 0), 2_500.0)]}
    ib = TimedIB(paths, positions={"DDD": 100})
    router = _timed_router(store, ib, max_session_notional_usd=2_000.0)
    _run_day(router, _book({"EEE": 0.01, "DDD": 0.0}))
    # Planned at the cap: DDD's $1,000 sell and EEE's 100 shares at $10. At the
    # expiry EEE is $2,500 and the $1,000 left buys no share.
    assert [s for s, *_ in ib.placed()] == ["DDD"]
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert rows["EEE"]["status"] == "not_sent" and "caps leave no room" in rows["EEE"]["trigger"]


def test_a_rerun_adopts_a_buy_the_broker_knows_and_watches_the_rest(store) -> None:
    ib = TimedIB(PATHS)
    first = _timed_router(store, ib)
    batches = iter(first.batches(_book(BOOK)))
    router_batch = next(batches)  # AAA at its 11:00 pullback
    first.send(router_batch)
    assert [c.symbol for c, _ in ib.order_calls] == ["AAA"]
    ib.completed.extend(ib._trades)  # the broker's completed-order record

    rerun = _timed_router(store, ib)
    _run_day(rerun, _book(BOOK))
    assert [c.symbol for c, _ in ib.order_calls] == ["AAA", "BBB", "CCC"]  # AAA once
    adopted = next(r for r in rerun.records if r.order.decision.symbol == "AAA")
    assert adopted.disposition == "adopted"


def test_a_buy_with_an_intent_and_no_broker_order_is_never_sent(store) -> None:
    ib = TimedIB(PATHS)
    claim_order_intent(
        store,
        order_intent_document(
            order_key=order_key(DAY, "BBB"),
            order_ref=f"{order_key(DAY, 'BBB')}|buy|200|50.0|2026-09-08T13:35:00.000000Z",
            trading_day=DAY,
            symbol="BBB",
            side="buy",
            quantity=200,
            decision_price=50.0,
            account=ACCOUNT,
            intended_at=et(9, 35),
        ),
    )
    router = _timed_router(store, ib)
    with pytest.raises(UnknownOrderStateError, match="BBB"):
        _run_day(router, _book(BOOK))
    assert [c.symbol for c, _ in ib.order_calls] == ["AAA", "CCC"]
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert rows["BBB"]["status"] == "not_sent" and rows["BBB"]["trigger"].startswith("unknown")


def test_a_rerun_adopts_a_sell_the_broker_is_working_and_never_resends_it(store) -> None:
    path = [(dt.time(9, 30), 10.0), (dt.time(10, 0), 9.5), (dt.time(11, 0), 9.7)]
    ib = TimedIB({"DDD": path}, positions={"DDD": 100}, stall=True)  # the sell never fills
    first = _timed_router(store, ib)
    batches = iter(first.batches(_book({"DDD": 0.0})))
    first.send(next(batches))  # the 11:00 rally; the session then dies
    assert ib.placed() == [("DDD", "SELL", 100, "11:00")]

    rerun = _timed_router(store, ib)
    assert _run_day(rerun, _book({"DDD": 0.0})) == 1
    assert len(ib.order_calls) == 1  # adopted, not re-sent
    (record,) = rerun.records
    assert record.disposition == "adopted" and record.order.decision.side == "sell"
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert (row["status"], row["trigger"]) == ("sent", "adopted: already at the broker")


def test_a_rerun_after_a_filled_sell_plans_no_sell(store) -> None:
    path = [(dt.time(9, 30), 10.0), (dt.time(10, 0), 9.5), (dt.time(11, 0), 9.7)]
    ib = TimedIB({"DDD": path}, positions={"DDD": 100})
    first = _timed_router(store, ib)
    batches = iter(first.batches(_book({"DDD": 0.0})))
    first.send(next(batches))  # filled: the account no longer holds DDD
    rerun = _timed_router(store, ib)
    assert _run_day(rerun, _book({"DDD": 0.0})) == 0
    assert len(ib.order_calls) == 1


def test_a_sell_with_an_intent_and_no_broker_order_is_never_sent(store) -> None:
    ib = TimedIB(PATHS, positions={"DDD": 100})
    claim_order_intent(
        store,
        order_intent_document(
            order_key=order_key(DAY, "DDD"),
            order_ref=f"{order_key(DAY, 'DDD')}|sell|100|10.0|2026-09-08T13:35:00.000000Z",
            trading_day=DAY,
            symbol="DDD",
            side="sell",
            quantity=100,
            decision_price=10.0,
            account=ACCOUNT,
            intended_at=et(9, 35),
        ),
    )
    router = _timed_router(store, ib)
    with pytest.raises(UnknownOrderStateError, match="DDD"):
        _run_day(router, _book({"AAA": 0.1, "DDD": 0.0}))
    assert ib.placed() == [("AAA", "BUY", 100, "11:00")]
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert rows["DDD"]["status"] == "not_sent" and rows["DDD"]["trigger"].startswith("unknown")


# -- the book's cash: no buy on proceeds a pending sell has not freed ------------------

#: AAA pulls back at 11:00; DDD is flat all day, so its sell goes at 15:30.
CASH_PATHS = {"AAA": PATHS["AAA"], "CCC": PATHS["CCC"], "DDD": [(dt.time(9, 30), 10.0)]}
#: Caps wide enough that only the cash binds.
WIDE = {"max_order_notional_usd": 200_000.0, "max_session_notional_usd": 300_000.0}


def test_a_buy_waits_for_the_sell_whose_proceeds_it_needs(store, capsys) -> None:
    # $95,000 of DDD held in a $100,000 book: $5,000 of cash. AAA needs $9,890 at 11:00.
    ib = TimedIB(CASH_PATHS, positions={"DDD": 9_500})
    router = _timed_router(store, ib, **WIDE)
    _run_day(router, _book({"AAA": 0.1, "DDD": 0.0}))
    assert ib.placed() == [("DDD", "SELL", 9_500, "15:30"), ("AAA", "BUY", 100, "15:30")]
    assert "AAA pullback 2.08% from high 101.00; held for sell proceeds: $5,000.00" in (
        capsys.readouterr().err
    )
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert rows["AAA"]["status"] == "sent" and rows["AAA"]["trigger"].startswith("pullback")


def test_a_buy_the_book_has_cash_for_does_not_wait_for_a_sell(store) -> None:
    ib = TimedIB(CASH_PATHS, positions={"DDD": 100})
    router = _timed_router(store, ib, **WIDE)
    _run_day(router, _book({"AAA": 0.1, "DDD": 0.0}))
    assert ib.placed() == [("AAA", "BUY", 100, "11:00"), ("DDD", "SELL", 100, "15:30")]


def test_with_no_sell_left_a_buy_is_clipped_to_the_cash(store, capsys) -> None:
    # DDD carried unchanged ($95,000): $5,000 of cash. CCC sized at $20 to 250
    # shares; at the 15:55 expiry it is $21, and $5,000 buys 238.
    ib = TimedIB(CASH_PATHS, positions={"DDD": 9_500})
    router = _timed_router(store, ib, **WIDE)
    _run_day(router, _book({"CCC": 0.05, "DDD": 0.95}))
    assert ib.placed() == [("CCC", "BUY", 238, "15:55")]
    assert "exceeds the book's cash at its trigger; clipped to 238" in capsys.readouterr().err


def test_a_sell_that_has_not_filled_frees_no_cash(store) -> None:
    # A $100,000 book wholly in DDD: no cash until the 100-share sell fills, and it never does.
    ib = TimedIB(CASH_PATHS, positions={"DDD": 10_000}, stall=True)
    router = _timed_router(store, ib, **WIDE)
    _run_day(router, _book({"CCC": 0.01, "DDD": 0.99}))
    assert ib.placed() == [("DDD", "SELL", 100, "15:30")]
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert rows["CCC"]["status"] == "not_sent"
    assert rows["CCC"]["trigger"] == "time_expiry; the book's cash ($0.00) buys no share at 21.00"


def test_the_book_cash_is_the_notional_less_the_held_book() -> None:
    prices = {"AAA": 100.0, "DDD": 10.0}
    assert book_cash(_book(), {"AAA": 300, "DDD": 0}, quote=prices.get) == 70_000.0
    assert book_cash(_book(), {"AAA": 2_000}, quote=prices.get) == 0.0  # never negative
    with pytest.raises(PriceUnavailableError, match="ZZZ"):
        book_cash(_book(), {"ZZZ": 1}, quote=prices.get)


# -- configuration --------------------------------------------------------------------


def test_routing_holds_buys_for_v1s_triggers_by_default() -> None:
    assert RouterConfig.from_env(_env()).entry_timing == EntryTimingParams()
    assert RouterConfig.from_env(_env(**{ENTRY_TIMING_VAR: "v1_triggers"})).entry_timing == PARAMS


def test_immediate_entry_timing_sends_every_buy_at_the_start() -> None:
    assert RouterConfig.from_env(_env(**{ENTRY_TIMING_VAR: "immediate"})).entry_timing is None


def test_an_unrecognised_entry_timing_is_refused() -> None:
    with pytest.raises(RouterSettingsError, match=ENTRY_TIMING_VAR):
        RouterConfig.from_env(_env(**{ENTRY_TIMING_VAR: "later"}))


# -- the support level from the price panel -------------------------------------------


def _panel(store, rows: dict[str, list[float]], *, day: str = DAY) -> None:
    """A panel for ``day`` whose rows for each name end on ``day``, one low per session."""
    end = dt.date.fromisoformat(day)
    records = []
    for ticker, lows in rows.items():
        for back, low in enumerate(reversed(lows)):
            records.append(
                {
                    "trading_day": end - dt.timedelta(days=back),
                    "ticker": ticker,
                    "low_raw": low,
                }
            )
    # A row after the decision day is never part of its history.
    records.append({"trading_day": end + dt.timedelta(days=1), "ticker": "AAA", "low_raw": 1.0})
    frame = pd.DataFrame(records)
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    store.put_bytes(data_panel_key(day), buffer.getvalue())


def test_the_support_level_is_the_low_of_the_last_twenty_sessions(store, capsys) -> None:
    _panel(
        store,
        {
            "AAA": [5.0, *[100.0 + i for i in range(19)], 99.0],  # 21 rows; the 5.0 is too old
            "BBB": [50.0] * 19,  # one session short: no level (v1)
            "CCC": [float("nan"), -1.0, *([20.0] * 18)],  # unusable lows skipped
            "DDD": [float("nan")] * 20,  # nothing usable
        },
    )
    levels = panel_support_levels(store, DAY, ["AAA", "BBB", "CCC", "DDD", "EEE"], lookback_days=20)
    assert levels == {"AAA": 99.0, "CCC": 20.0}
    assert "for BBB, DDD, EEE" in capsys.readouterr().err


def test_no_panel_leaves_no_support_level_and_says_so(store, capsys) -> None:
    assert panel_support_levels(store, DAY, ["AAA"], lookback_days=20) == {}
    assert "the support-bounce trigger will not fire today" in capsys.readouterr().err
    assert panel_support_levels(store, DAY, [], lookback_days=20) == {}


def test_a_buy_is_released_at_its_vwap_discount(store) -> None:
    ib = TimedIB({"AAA": [(dt.time(9, 30), 100.0), (dt.time(10, 30), 100.5)]})
    # 100.0 at its VWAP until 10:30, when 100.5 is 0.59% under it.
    ib.vwaps["AAA"] = [(dt.time(9, 30), 100.0), (dt.time(10, 30), 101.1)]
    router = _timed_router(store, ib)
    _run_day(router, _book({"AAA": 0.1}))
    assert ib.placed() == [("AAA", "BUY", 100, "10:30")]
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert row["trigger"].startswith("vwap_discount") and row["vwap"] == 101.1


def test_a_buy_is_released_on_a_bounce_off_its_support(store) -> None:
    _panel(store, {"AAA": [96.0] * 20})
    path = [(dt.time(9, 30), 100.0), (dt.time(10, 0), 99.0), (dt.time(13, 0), 96.9)]
    ib = TimedIB({"AAA": path})
    ib.vwaps["AAA"] = [(dt.time(9, 30), 96.0)]  # never at a discount to the VWAP
    router = _timed_router(store, ib, entry_timing=dataclasses.replace(PARAMS, pullback_pct=0.5))
    _run_day(router, _book({"AAA": 0.1}))
    assert ib.placed() == [("AAA", "BUY", 100, "13:00")]
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert row["support_level"] == 96.0
    assert row["trigger"] == "support_bounce 0.94% above support 96.00"


def test_a_stream_with_no_vwap_is_logged_and_the_buy_still_goes(store, capsys) -> None:
    ib = TimedIB({"CCC": [(dt.time(9, 30), 20.0), (dt.time(12, 0), 21.0)]})
    router = _timed_router(store, ib)
    _run_day(router, _book({"CCC": 0.1}))
    assert ib.placed() == [("CCC", "BUY", 500, "15:55")]
    assert "no VWAP was streamed for CCC" in capsys.readouterr().err
