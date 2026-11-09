"""Entry timing: sells at the start, buys held for v1's triggers, nothing sent twice.

The router tests drive a whole simulated trading day on a fake clock: the fake
gateway's ``sleep`` advances the clock, its tickers read a scripted price path,
and every order call is recorded, so "waited until 15:30" is an assertion on
when the order reached the fake broker, not on the router's own say-so.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import pytest
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
)
from crucible_trader.kill_switch import TradingHaltedError
from crucible_trader.order_intent import claim_order_intent, order_intent_document
from crucible_trader.order_router import (
    ENTRY_TIMING_VAR,
    PaperOrderRouter,
    RouterConfig,
    RouterSettingsError,
    RoutingRefusedError,
    UnknownOrderStateError,
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
        "graduated_start_et": "15:30",
        "graduated_max_premium_pct": 0.01,
        "expiry_et": "15:55",
        "poll_interval_s": 60.0,
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
        ("poll_interval_s", 0.0, "poll_interval_s"),
        ("poll_interval_s", float("nan"), "poll_interval_s"),
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
        IntradayQuote(last=101.0, high=101.0)
    )
    assert entry.observe(IntradayQuote(last=None, high=None)) == IntradayQuote(None, 101.0)
    seen = entry.observe(IntradayQuote(last=99.0, high=100.5))
    assert seen == IntradayQuote(last=99.0, high=101.0)
    assert (entry.last, entry.market_data_type) == (99.0, 3)


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

    close = float("nan")


class TimedIB(RoutingIB):
    """A paper gateway on a scripted intraday price path, one entry per (ET time, price)."""

    def __init__(self, paths: dict[str, list[tuple[dt.time, float]]], **kwargs) -> None:
        self.paths = paths
        clock = kwargs.pop("clock", None) or FakeClock(et(9, 35).astimezone(dt.UTC))
        super().__init__(prices={s: p[0][1] for s, p in paths.items()}, clock=clock, **kwargs)
        self.connected = True
        self.subscribed: list[str] = []

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
    assert quotes("AAA") == IntradayQuote(last=103.0, high=103.0, market_data_type=3)
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


# -- the router, over a whole simulated day --------------------------------------------


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _timed_router(store, ib: TimedIB, **config) -> PaperOrderRouter:
    return PaperOrderRouter(
        _config(entry_timing=PARAMS, **config),
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
#: 5% and only the expiry takes it. DDD is held and sold at the start.
PATHS = {
    "AAA": [(dt.time(9, 30), 100.0), (dt.time(10, 0), 101.0), (dt.time(11, 0), 98.9)],
    "BBB": [(dt.time(9, 30), 50.0), (dt.time(12, 0), 50.3)],
    "CCC": [(dt.time(9, 30), 20.0), (dt.time(12, 0), 21.0)],
    "DDD": [(dt.time(9, 30), 10.0)],
}
BOOK = {"AAA": 0.1, "BBB": 0.1, "CCC": 0.1, "DDD": 0.0}


def test_sells_go_at_the_start_and_each_buy_waits_for_its_trigger(store) -> None:
    ib = TimedIB(PATHS, positions={"DDD": 100})
    router = _timed_router(store, ib)
    batches = _run_day(router, _book(BOOK))

    assert ib.placed() == [
        ("DDD", "SELL", 100, "09:35"),
        ("AAA", "BUY", 100, "11:00"),
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
    assert document["state"] == "complete" and document["summary"] == {"sent": 3}
    triggers = {row["symbol"]: row["trigger"] for row in document["entries"]}
    assert triggers["AAA"].startswith("pullback 2.08% from high 101.00")
    assert triggers["BBB"].startswith("graduated_entry (+0.60%")
    assert triggers["CCC"] == "time_expiry"
    assert sorted(ib.cancelled_mkt).count("AAA") >= 2  # the decision quote and the stream
    key, payload = router.entry_watch_payload(ib.clock())
    assert key == entry_watch_key(DAY) and json.loads(payload)["state"] == "complete"


def test_a_book_of_sells_only_is_not_watched(store) -> None:
    ib = TimedIB(PATHS, positions={"DDD": 100})
    router = _timed_router(store, ib)
    assert _run_day(router, _book({"DDD": 0.0})) == 1
    assert router.entry_watch_payload(ib.clock()) is None
    assert not store.exists(entry_watch_key(DAY))


def test_a_book_of_buys_only_opens_with_no_batch(store) -> None:
    ib = TimedIB(PATHS)
    router = _timed_router(store, ib)
    assert _run_day(router, _book({"CCC": 0.1})) == 1
    assert ib.placed() == [("CCC", "BUY", 500, "15:55")]


def test_the_kill_switch_stops_the_waiting_buys(store) -> None:
    ib = TimedIB(PATHS, positions={"DDD": 100})
    router = _timed_router(store, ib)
    batches = iter(router.batches(_book(BOOK)))
    router.send(next(batches))  # the sells
    _halt(store, "freeze")
    with pytest.raises(TradingHaltedError):
        next(batches)
    assert [c.symbol for c, _ in ib.order_calls] == ["DDD"]
    assert json.loads(store.get_bytes(entry_watch_key(DAY)))["state"] == "stopped"


def test_a_dropped_gateway_stops_the_watch_and_sends_nothing_more(store) -> None:
    ib = TimedIB(PATHS)
    router = _timed_router(store, ib)
    batches = iter(router.batches(_book(BOOK)))
    ib.connected = False
    with pytest.raises(RoutingRefusedError, match="3 buy"):
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
    (row,) = json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]
    assert row["status"] == "not_sent" and "caps leave no room" in row["trigger"]


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


# -- configuration --------------------------------------------------------------------


def test_routing_holds_buys_for_v1s_triggers_by_default() -> None:
    assert RouterConfig.from_env(_env()).entry_timing == EntryTimingParams()
    assert RouterConfig.from_env(_env(**{ENTRY_TIMING_VAR: "v1_triggers"})).entry_timing == PARAMS


def test_immediate_entry_timing_sends_every_buy_at_the_start() -> None:
    assert RouterConfig.from_env(_env(**{ENTRY_TIMING_VAR: "immediate"})).entry_timing is None


def test_an_unrecognised_entry_timing_is_refused() -> None:
    with pytest.raises(RouterSettingsError, match=ENTRY_TIMING_VAR):
        RouterConfig.from_env(_env(**{ENTRY_TIMING_VAR: "later"}))
