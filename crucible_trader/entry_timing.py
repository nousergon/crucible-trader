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
2. **VWAP discount** -- the last price is at least
   :attr:`EntryTimingParams.vwap_discount_pct` below the VWAP;
3. **support bounce** -- the last price is at or above the support level and
   within :attr:`EntryTimingParams.support_pct` of it, and the day's low has not
   broken it;
4. **time expiry** -- unconditional at :attr:`EntryTimingParams.expiry_et`;
5. **graduated** -- from :attr:`EntryTimingParams.graduated_start_et`, the last
   price is at most :attr:`EntryTimingParams.graduated_max_premium_pct` above the
   price the buy was sized at (v1's "morning price").

All five are v1's, in v1's order; v1's live ``disabled_triggers`` was empty, so
none is left out. Two inputs come from different places than in v1:

* **The VWAP is today's, from IB's stream** (generic tick 233, RTVolume, which
  ``ib_async`` folds into ``Ticker.vwap``). v1 compared against the PREVIOUS
  day's VWAP its planner loaded from ArcticDB; the v2 panel carries OHLCV and
  no VWAP. Today's VWAP is the conventional benchmark for "buying below the
  average price paid today".
* **The support level is the N-day low of the price panel** the session already
  reads (:func:`panel_support_levels`): the minimum positive ``low_raw`` over the
  last :attr:`EntryTimingParams.support_lookback_days` sessions up to the book's
  decision day, and none at all for a name with fewer -- v1's
  ``_compute_support_level`` exactly.

**An input that is not there fails safe.** A name with no VWAP on its stream,
or no support level in the panel, simply never fires that trigger; the other
four still can. Each is logged, and the watch document carries ``vwap`` and
``support_level`` per name, so a trigger that could not fire is visible.

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
import io
import json
import logging
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from crucible.keys import data_panel_key
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
    "panel_support_levels",
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

#: IB generic tick 233 (RTVolume): ``ib_async`` derives ``Ticker.vwap`` from it.
VWAP_GENERIC_TICKS = "233"

#: How often the watch document is rewritten while nothing triggers.
HEARTBEAT_S = 600.0

EntryStatus = Literal["waiting", "triggered", "sent", "not_sent"]


@dataclasses.dataclass(frozen=True)
class EntryTimingParams:
    """v1's entry-trigger parameters, under v1's own names. Defaults are what v1 ran."""

    #: v1 `intraday_pullback_pct`: a buy is released this far below the intraday high.
    pullback_pct: float = 0.02
    #: v1 `intraday_vwap_discount_pct`: released this far below the VWAP.
    vwap_discount_pct: float = 0.005
    #: v1 `intraday_support_pct`: released within this distance above the support level.
    support_pct: float = 0.01
    #: v1 `intraday_support_lookback_days`: the support level is the low of this many sessions.
    support_lookback_days: int = 20
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
        for name in ("vwap_discount_pct", "support_pct"):
            value = getattr(self, name)
            if not (0.0 < value < 1.0):
                raise ValueError(f"{name} {value!r} must be in (0, 1)")
        if self.support_lookback_days < 1:
            raise ValueError(
                f"support_lookback_days {self.support_lookback_days!r} must be at least 1"
            )
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
            "vwap_discount_pct": self.vwap_discount_pct,
            "support_pct": self.support_pct,
            "support_lookback_days": self.support_lookback_days,
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
    """What the watch reads for one name: the last trade, the session high and low, the VWAP."""

    last: float | None
    high: float | None
    market_data_type: int | None = None
    low: float | None = None
    vwap: float | None = None


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
    support_level: float | None = None,
) -> str | None:
    """Why a waiting buy is released now, or ``None`` to keep waiting.

    v1's order and v1's arithmetic (`EntryTriggerEngine.should_enter`). Nothing is
    released before the open or at/after the close; expiry needs no price, every
    other trigger needs a last price, and each its own input as well (the high,
    the VWAP, the support level). A missing input only keeps its trigger quiet.
    """
    if now < schedule.opens_at or now >= schedule.closes_at:
        return None
    last = _valid(quote.last)
    high = _valid(quote.high)
    if last is not None and high is not None:
        pullback = (high - last) / high
        if pullback >= params.pullback_pct:
            return f"pullback {pullback:.2%} from high {high:.2f}"
    vwap = _valid(quote.vwap)
    if last is not None and vwap is not None:
        discount = (vwap - last) / vwap
        if discount >= params.vwap_discount_pct:
            return f"vwap_discount {discount:.2%} below VWAP {vwap:.2f}"
    support = _valid(support_level)
    low = _valid(quote.low)
    # v1: a day low under the support level means support broke; no bounce.
    if last is not None and support is not None and (low is None or low >= support):
        distance = (last - support) / support
        if 0.0 <= distance <= params.support_pct:
            return f"support_bounce {distance:.2%} above support {support:.2f}"
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
    support_level: float | None = None
    status: EntryStatus = "waiting"
    trigger: str | None = None
    released_at: dt.datetime | None = None
    released_quantity: int | None = None
    release_price: float | None = None
    last: float | None = None
    high: float | None = None
    low: float | None = None
    vwap: float | None = None
    market_data_type: int | None = None

    def observe(self, quote: IntradayQuote) -> IntradayQuote:
        """Fold a quote in: the high never below, the low never above, any last price seen.

        The VWAP is the stream's latest; a poll without one keeps the last seen.
        """
        last = _valid(quote.last)
        if last is not None:
            self.last = last
        for value in (_valid(quote.high), last):
            if value is not None:
                self.high = value if self.high is None else max(self.high, value)
        for value in (_valid(quote.low), last):
            if value is not None:
                self.low = value if self.low is None else min(self.low, value)
        vwap = _valid(quote.vwap)
        if vwap is not None:
            self.vwap = vwap
        if quote.market_data_type is not None:
            self.market_data_type = quote.market_data_type
        return IntradayQuote(
            last=self.last if last is not None else None,
            high=self.high,
            low=self.low,
            vwap=self.vwap,
        )

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
            "low": self.low,
            "vwap": self.vwap,
            "support_level": self.support_level,
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
    Generic tick :data:`VWAP_GENERIC_TICKS` (RTVolume) is requested so the
    ticker carries the session VWAP; a name whose stream never carries one only
    loses the VWAP trigger.
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
            self._tickers[symbol] = self._client.reqMktData(
                contract, VWAP_GENERIC_TICKS, False, False
            )

    def __contains__(self, symbol: object) -> bool:
        """Whether ``symbol`` is streamed now (subscribed and not yet closed)."""
        return symbol in self._tickers

    def __call__(self, symbol: str) -> IntradayQuote:
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return IntradayQuote(last=None, high=None)
        kind = getattr(ticker, "marketDataType", None)
        return IntradayQuote(
            last=_valid(getattr(ticker, "last", None)),
            high=_valid(getattr(ticker, "high", None)),
            market_data_type=kind if isinstance(kind, int) else None,
            low=_valid(getattr(ticker, "low", None)),
            vwap=_valid(getattr(ticker, "vwap", None)),
        )

    def close(self) -> None:
        contracts, self._contracts, self._tickers = self._contracts, [], {}
        for contract in contracts:
            try:
                self._client.cancelMktData(contract)
            except Exception:  # noqa: BLE001 - teardown never blocks the session's end
                log.exception("cancelMktData failed during teardown")


def panel_support_levels(
    store: Store, trading_day: str, symbols: Sequence[str], *, lookback_days: int
) -> dict[str, float]:
    """Each name's support level: the lowest ``low_raw`` of its last ``lookback_days`` sessions.

    Read from the price panel compiled for ``trading_day`` (the book's decision
    day), whose rows run up to and including that day -- the history v1's
    ``_compute_support_level`` took the low of. A name with fewer than
    ``lookback_days`` rows, or no positive low among them, has no support level
    (v1 likewise). Never raises: a panel that cannot be read leaves every name
    without one, logged, and the support trigger then simply never fires.
    """
    wanted = set(symbols)
    if not wanted:
        return {}
    key = data_panel_key(trading_day)
    try:
        import pandas as pd  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]  # no pandas-stubs in the lock; only when a buy is watched

        frame = pd.read_parquet(  # pyright: ignore[reportUnknownMemberType]  # pandas ships no complete stubs here
            io.BytesIO(store.get_bytes(key)), columns=["trading_day", "ticker", "low_raw"]
        )
        records = cast(
            "list[dict[str, object]]",
            frame.to_dict(orient="records"),  # pyright: ignore[reportUnknownMemberType]  # pandas ships no complete stubs here
        )
    except Exception as exc:  # noqa: BLE001 - a missing input silences one trigger, never the session
        log.warning("support levels unavailable from %s: %r", key, exc)
        print(
            f"entry_timing: no support levels ({key}: {type(exc).__name__}: {exc}); "
            "the support-bounce trigger will not fire today",
            file=sys.stderr,
        )
        return {}
    history: dict[str, list[tuple[str, object]]] = {}
    for row in records:
        ticker, day = str(row["ticker"]), str(row["trading_day"])[:10]
        if ticker in wanted and day <= trading_day:
            history.setdefault(ticker, []).append((day, row["low_raw"]))
    levels: dict[str, float] = {}
    for ticker, rows in history.items():
        if len(rows) < lookback_days:
            continue
        lows = [v for _, low in sorted(rows)[-lookback_days:] if (v := _valid(low)) is not None]
        if lows:
            levels[ticker] = min(lows)
    missing = sorted(wanted - set(levels))
    if missing:
        print(
            f"entry_timing: no {lookback_days}-session support level for {', '.join(missing)}; "
            "the support-bounce trigger will not fire for them",
            file=sys.stderr,
        )
    return levels
