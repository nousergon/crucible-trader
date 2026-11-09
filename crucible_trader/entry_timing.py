"""Entry timing: buys wait for v1's intraday triggers instead of going out at the open.

Brian 2026-11-09: "in V1 ... we wait for right time to trade rather than buying
and selling everything right at the market open ... it wouldn't trade until
like right before market close unless the stock price went down a certain
percentage." Until this module the v2 session sent every order within about two
minutes of its 09:35 start.

**What waits and what does not.** Sells and reduces go out at the session start,
exactly as before: an exit is a risk decision and v1's daemon sent its urgent
EXIT/REDUCE orders at the open too (`crucible-executor` `executor/daemon.py`,
Phase 0). Buys and increases are held and released by :func:`entry_trigger`,
which ports `crucible-executor` `executor/entry_triggers.py`:

1. **pullback** -- the last price is at least :attr:`EntryTimingParams.pullback_pct`
   below the intraday high;
2. **time expiry** -- unconditional at :attr:`EntryTimingParams.expiry_et`;
3. **graduated** -- from :attr:`EntryTimingParams.graduated_start_et`, the last
   price is at most :attr:`EntryTimingParams.graduated_max_premium_pct` above the
   price the buy was sized at (v1's "morning price").

v1 had two more, VWAP discount and support bounce. Both read inputs v1's planner
computed (the previous day's VWAP, a 20-day support level) that the v2 book does
not carry, so they are not ported; the three above are the ones Brian named.

**The defaults are what v1 RAN, not what its code said.** v1's code default for
the graduated start was 14:00; its live configuration set 15:30, so a buy that
neither pulls back nor expires waits until the last half hour. Every other value
matches v1's code and live configuration alike: 2% pullback, +1% graduated
premium, 15:55 expiry, a 60 s poll.

**Early closes.** The clock times are for a 16:00 close. On an NYSE half day
each is moved by the same amount the close is (13:00 close: graduated 12:30,
expiry 12:55), so a buy can never be scheduled after the market has shut. A buy
still waiting when the market closes is not sent, and is recorded as such.

**Visible while it waits.** The watch is written to
``trader/entry_watch/{day}.json`` (:data:`ENTRY_WATCH_SCHEMA_VERSION`) at the
start, on every trigger, and on a heartbeat: every waiting buy, the price it was
sized at, the last price and high seen, and what released it. Like the order
book, a write that fails never stops the orders; it is logged and counted.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import sys
from collections.abc import Callable, Iterable
from typing import Any, Literal
from zoneinfo import ZoneInfo

from crucible.store import Store
from krepis.trading_calendar import session_close_et

__all__ = [
    "ENTRY_WATCH_PREFIX",
    "ENTRY_WATCH_SCHEMA_VERSION",
    "EntrySchedule",
    "EntryTimingParams",
    "EntryWatch",
    "IbIntradayQuotes",
    "IntradayQuote",
    "WatchedEntry",
    "entry_trigger",
    "entry_watch_key",
]

log = logging.getLogger(__name__)

ENTRY_WATCH_SCHEMA_VERSION = "entry_watch.v1"
ENTRY_WATCH_PREFIX = "trader/entry_watch/"

NYSE_ZONE = ZoneInfo("America/New_York")
REGULAR_OPEN_ET = dt.time(9, 30)
REGULAR_CLOSE_ET = dt.time(16, 0)

#: IB market-data type 3: live when the account subscribes, delayed when not
#: (the router's own decision prices use the same, `order_router.MARKET_DATA_TYPE`).
MARKET_DATA_TYPE = 3

#: How often the watch document is rewritten while nothing triggers.
HEARTBEAT_S = 600.0

EntryStatus = Literal["waiting", "triggered", "sent", "not_sent"]


@dataclasses.dataclass(frozen=True)
class EntryTimingParams:
    """v1's entry-trigger parameters, under v1's own names. Defaults are what v1 ran."""

    #: v1 `intraday_pullback_pct`: a buy is released this far below the intraday high.
    pullback_pct: float = 0.02
    #: v1 `intraday_graduated_start_time` (its live configuration: "15:30").
    graduated_start_et: dt.time = dt.time(15, 30)
    #: v1 `intraday_graduated_max_premium_pct`.
    graduated_max_premium_pct: float = 0.01
    #: v1 `intraday_expiry_time`: every buy still waiting goes out at market.
    expiry_et: dt.time = dt.time(15, 55)
    #: v1 `intraday.poll_interval_sec`.
    poll_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if not (0.0 < self.pullback_pct < 1.0):
            raise ValueError(f"pullback_pct {self.pullback_pct!r} must be in (0, 1)")
        if not (0.0 <= self.graduated_max_premium_pct < 1.0):
            raise ValueError(
                f"graduated_max_premium_pct {self.graduated_max_premium_pct!r} must be in [0, 1)"
            )
        if not (REGULAR_OPEN_ET < self.graduated_start_et <= self.expiry_et < REGULAR_CLOSE_ET):
            raise ValueError(
                f"the graduated start {self.graduated_start_et} and expiry {self.expiry_et} must "
                f"fall inside the session, start first ({REGULAR_OPEN_ET} < start <= expiry < "
                f"{REGULAR_CLOSE_ET})"
            )
        if not (math.isfinite(self.poll_interval_s) and self.poll_interval_s > 0):
            raise ValueError(f"poll_interval_s {self.poll_interval_s!r} must be positive")

    def document(self) -> dict[str, object]:
        """The parameters in force, as recorded on the watch document."""
        return {
            "pullback_pct": self.pullback_pct,
            "graduated_start_et": self.graduated_start_et.strftime("%H:%M"),
            "graduated_max_premium_pct": self.graduated_max_premium_pct,
            "expiry_et": self.expiry_et.strftime("%H:%M"),
            "poll_interval_s": self.poll_interval_s,
        }

    def schedule(
        self, session_date: dt.date, *, close_et: Callable[[dt.date], dt.time] = session_close_et
    ) -> EntrySchedule:
        """The day's instants, shifted with the close on an early-close day."""
        close = close_et(session_date)
        regular = dt.datetime.combine(session_date, REGULAR_CLOSE_ET, NYSE_ZONE)
        actual = dt.datetime.combine(session_date, close, NYSE_ZONE)
        shift = actual - regular

        def at(clock: dt.time) -> dt.datetime:
            return dt.datetime.combine(session_date, clock, NYSE_ZONE) + shift

        return EntrySchedule(
            opens_at=dt.datetime.combine(session_date, REGULAR_OPEN_ET, NYSE_ZONE),
            graduated_at=at(self.graduated_start_et),
            expires_at=at(self.expiry_et),
            closes_at=actual,
        )


@dataclasses.dataclass(frozen=True)
class EntrySchedule:
    """One session's entry instants, timezone-aware."""

    opens_at: dt.datetime
    graduated_at: dt.datetime
    expires_at: dt.datetime
    closes_at: dt.datetime

    def document(self) -> dict[str, str]:
        return {
            name: getattr(self, name).isoformat()
            for name in ("opens_at", "graduated_at", "expires_at", "closes_at")
        }


@dataclasses.dataclass(frozen=True)
class IntradayQuote:
    """What the watch reads for one name: the last trade and the session high."""

    last: float | None
    high: float | None
    market_data_type: int | None = None


def _valid(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def entry_trigger(
    *,
    reference_price: float,
    quote: IntradayQuote,
    now: dt.datetime,
    schedule: EntrySchedule,
    params: EntryTimingParams,
) -> str | None:
    """Why a waiting buy is released now, or ``None`` to keep waiting.

    v1's order and v1's arithmetic (`EntryTriggerEngine.should_enter`). Nothing is
    released before the open or at/after the close; expiry needs no price, the
    other two need a last price, and the pullback a high as well.
    """
    if now < schedule.opens_at or now >= schedule.closes_at:
        return None
    last = _valid(quote.last)
    high = _valid(quote.high)
    if last is not None and high is not None:
        pullback = (high - last) / high
        if pullback >= params.pullback_pct:
            return f"pullback {pullback:.2%} from high {high:.2f}"
    if now >= schedule.expires_at:
        return "time_expiry"
    if now >= schedule.graduated_at and last is not None:
        premium = (last - reference_price) / reference_price
        if premium <= params.graduated_max_premium_pct:
            return (
                f"graduated_entry ({premium:+.2%} vs sized price {reference_price:.2f}, "
                f"limit {params.graduated_max_premium_pct:+.2%})"
            )
    return None


def entry_watch_key(trading_day: str) -> str:
    dt.date.fromisoformat(trading_day)  # raises on anything but an ISO day
    return f"{ENTRY_WATCH_PREFIX}{trading_day}.json"


def _stamp(at: dt.datetime) -> str:
    return at.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclasses.dataclass
class WatchedEntry:
    """One buy held for its trigger, and what has been seen of it."""

    symbol: str
    order_key: str
    quantity: int
    reference_price: float
    decided_at: dt.datetime
    status: EntryStatus = "waiting"
    trigger: str | None = None
    released_at: dt.datetime | None = None
    released_quantity: int | None = None
    release_price: float | None = None
    last: float | None = None
    high: float | None = None
    market_data_type: int | None = None

    def observe(self, quote: IntradayQuote) -> IntradayQuote:
        """Fold a quote in; the high is never lower than any last price seen."""
        last = _valid(quote.last)
        if last is not None:
            self.last = last
        for value in (_valid(quote.high), last):
            if value is not None:
                self.high = value if self.high is None else max(self.high, value)
        if quote.market_data_type is not None:
            self.market_data_type = quote.market_data_type
        return IntradayQuote(last=self.last if last is not None else None, high=self.high)

    def document(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "order_key": self.order_key,
            "side": "buy",
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "decided_at_utc": _stamp(self.decided_at),
            "status": self.status,
            "trigger": self.trigger,
            "released_at_utc": None if self.released_at is None else _stamp(self.released_at),
            "released_quantity": self.released_quantity,
            "release_price": self.release_price,
            "last": self.last,
            "high": self.high,
            "market_data_type": self.market_data_type,
        }


class EntryWatch:
    """The day's waiting buys and the document that shows them. Writes never raise."""

    def __init__(
        self,
        store: Store,
        *,
        trading_day: str,
        params: EntryTimingParams,
        schedule: EntrySchedule,
        entries: Iterable[WatchedEntry],
    ) -> None:
        self._store = store
        self._day = trading_day
        self.params = params
        self.schedule = schedule
        self.entries = list(entries)
        self.state = "watching"
        self.write_failures = 0
        self._written_at: dt.datetime | None = None
        self.polled_at: dt.datetime | None = None

    @property
    def key(self) -> str:
        return entry_watch_key(self._day)

    def waiting(self) -> list[WatchedEntry]:
        return [e for e in self.entries if e.status == "waiting"]

    def document(self, now: dt.datetime) -> dict[str, object]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return {
            "schema_version": ENTRY_WATCH_SCHEMA_VERSION,
            "trading_day": self._day,
            "session_date": self.schedule.opens_at.date().isoformat(),
            "state": self.state,
            "params": self.params.document(),
            "schedule": self.schedule.document(),
            "updated_at_utc": _stamp(now),
            "last_polled_at_utc": None if self.polled_at is None else _stamp(self.polled_at),
            "summary": dict(sorted(counts.items())),
            "entries": [e.document() for e in self.entries],
        }

    def payload(self, now: dt.datetime) -> bytes:
        return json.dumps(self.document(now), indent=2, sort_keys=True).encode("utf-8")

    def write(self, now: dt.datetime, *, force: bool = True) -> None:
        """Rewrite the document; without ``force``, only once a heartbeat has passed."""
        if (
            not force
            and self._written_at is not None
            and (now - self._written_at).total_seconds() < HEARTBEAT_S
        ):
            return
        try:
            self._store.put_bytes(self.key, self.payload(now))
        except Exception as exc:  # noqa: BLE001 - a missing record must never stop the orders
            self.write_failures += 1
            log.exception("entry watch write failed")
            print(
                f"entry_timing: the watch document {self.key} was not written "
                f"({type(exc).__name__}: {exc}); the waiting buys are still watched",
                file=sys.stderr,
            )
        self._written_at = now
        waiting = ", ".join(f"{e.symbol}@{e.last}" for e in self.waiting()) or "none"
        print(f"entry_timing: {self.state}; waiting: {waiting}", file=sys.stderr)


class IbIntradayQuotes:
    """Streaming IB quotes for the waiting buys (v1 `PriceMonitor`, on `ib_async`).

    One `reqMktData` subscription per name, opened once and read on every poll:
    the client's ``sleep`` between polls is what lets the SDK apply the ticks.
    """

    def __init__(self, client: Any, sdk: Any) -> None:  # pyright: ignore[reportExplicitAny]  # untyped SDK objects
        self._client = client
        self._sdk = sdk
        self._tickers: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]  # SDK Ticker
        self._contracts: list[Any] = []  # pyright: ignore[reportExplicitAny]  # SDK Contract

    def subscribe(self, symbols: Iterable[str]) -> None:
        self._client.reqMarketDataType(MARKET_DATA_TYPE)
        for symbol in symbols:
            contract = self._sdk.Stock(symbol, "SMART", "USD")
            qualified = self._client.qualifyContracts(contract)
            if not qualified or qualified[0] is None:
                # Watched without a price: only the expiry can release it, and the
                # order path refuses an unqualifiable contract loudly at that point.
                print(f"entry_timing: {symbol} did not qualify; no quote", file=sys.stderr)
                continue
            self._contracts.append(contract)
            self._tickers[symbol] = self._client.reqMktData(contract, "", False, False)

    def __call__(self, symbol: str) -> IntradayQuote:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return IntradayQuote(last=None, high=None)
        kind = getattr(ticker, "marketDataType", None)
        return IntradayQuote(
            last=_valid(getattr(ticker, "last", None)),
            high=_valid(getattr(ticker, "high", None)),
            market_data_type=kind if isinstance(kind, int) else None,
        )

    def close(self) -> None:
        contracts, self._contracts, self._tickers = self._contracts, [], {}
        for contract in contracts:
            try:
                self._client.cancelMktData(contract)
            except Exception:  # noqa: BLE001 - teardown never blocks the session's end
                log.exception("cancelMktData failed during teardown")
