"""The day's order book: every name the champion considered, and its target.

Brian 2026-11-05: the console must show the daily order book the old
dashboard's Order Book Rationale page showed, and every day's book must be
kept. Until this module the session's target book existed only in memory and
as the `portfolio_construction` metric on the run manifest: the orders it
produced (`execution_shortfall`) and the broker's holdings (`broker_statements`)
were saved, but not the book that explains them, including the names that
were considered and NOT bought.

One document per trading day at ``trader/order_books/{day}.json``
(`order_book.v1`). One row per name in the session's universe, the benchmark
and cash sentinels included and labelled, with:

* ``alpha`` -- the forecast the construction engine was handed;
* ``eligible`` -- whether the engine could hold the name at all;
* ``previous_weight`` / ``target_weight`` -- the book held into the session
  and the book the engine built, and ``target_notional_usd``;
* ``alpha_rank`` -- 1 is the highest alpha among eligible names;
* ``book_position`` -- the row's place, largest target first (positions,
  then the rest by alpha rank), and the weights again in percent;
* ``action`` -- what the book says about the name (see :func:`_action`);
* ``held_sessions`` -- for a name the book holds, how many consecutive
  sessions' books (this one included) have held it; 0 for a name it does not
  hold, ``null`` on the two sentinels. It is the minimum hold's clock
  (`crucible.portfolio.PortfolioParams.min_hold_days`,
  `alpha-engine-config-I12024`): tomorrow's session reads it straight off this
  book (:func:`held_sessions_from_order_books`).

**Never raises into the session.** A book that cannot be written must not
stop the day's orders -- a no-trade day is a worse outcome than a missing
record -- so a failed write is a ``FAIL`` metric naming the reason, and the
session goes on. The written book is put under the same COMPLIANCE retention
as the trader's other books (`crucible_trader.books_retention`), so it is
kept five years.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Mapping
from typing import Any

from crucible.store import Store
from krepis.trading_calendar import previous_trading_day

from crucible_trader.books_retention import lock_books_records
from crucible_trader.construction import TargetBook

log = logging.getLogger(__name__)

__all__ = [
    "HELD_SESSIONS_WALK_LIMIT",
    "ORDER_BOOKS_PREFIX",
    "ORDER_BOOK_METRIC",
    "ORDER_BOOK_SCHEMA_VERSION",
    "held_sessions_from_order_books",
    "order_book_document",
    "order_book_key",
    "record_order_book",
]

ORDER_BOOK_SCHEMA_VERSION = "order_book.v1"
ORDER_BOOKS_PREFIX = "trader/order_books/"
ORDER_BOOK_METRIC = "trader_order_book_written"

#: Weights at or below this are no position (float noise from the engine).
_ZERO = 1e-9

#: The most books :func:`held_sessions_from_order_books` reads walking back
#: through books that predate the ``held_sessions`` row field. A name held
#: longer than this is reported as held this many sessions: every minimum hold
#: worth ruling is far shorter, so the bound changes no decision and caps the
#: reads a session start can make.
HELD_SESSIONS_WALK_LIMIT = 30

#: The two sentinel rows' ``action`` values; neither is a held name.
_SENTINEL_ACTIONS = frozenset({"cash", "benchmark"})


def order_book_key(trading_day: str) -> str:
    dt.date.fromisoformat(trading_day)  # raises on anything but an ISO day
    return f"{ORDER_BOOKS_PREFIX}{trading_day}.json"


def _action(*, previous: float, target: float, eligible: bool, sentinel: str | None) -> str:
    """One word for what the book does with the name.

    ``enter`` / ``exit`` / ``increase`` / ``reduce`` / ``hold`` when the name
    is or was in the book; ``not_selected`` for an eligible name the engine
    gave no weight; ``ineligible`` for one it could not hold; ``cash`` and
    ``benchmark`` for the two sentinels.
    """
    if sentinel is not None:
        return sentinel
    was, will = previous > _ZERO, target > _ZERO
    if was and not will:
        return "exit"
    if will and not was:
        return "enter"
    if was and will:
        if target > previous + _ZERO:
            return "increase"
        if target < previous - _ZERO:
            return "reduce"
        return "hold"
    return "not_selected" if eligible else "ineligible"


def order_book_document(
    book: TargetBook,
    *,
    previous_weights: dict[str, float] | None,
    written_at_utc: str,
    held_sessions: Mapping[str, int] | None = None,
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # JSON document
    """The `order_book.v1` document for ``book``.

    ``held_sessions`` is the session's INPUT clock
    (:func:`held_sessions_from_order_books`: sessions each name had been held
    into before this one). A name this book holds is stamped one more than that,
    since this book holds it too; a name it does not hold is stamped 0.
    """
    universe = book.universe
    cash = universe.tickers[universe.cash_idx]
    benchmark = universe.tickers[universe.benchmark_idx]
    alpha = book.alpha or {}
    eligible = book.eligible or {}
    # No previous book is an all-cash book (construction's w_initial), so the
    # cash row's previous weight is 1.0, not 0 (codex ac64b346aa382999).
    previous = previous_weights if previous_weights is not None else {cash: 1.0}
    ages = held_sessions or {}

    ranked = sorted(
        (t for t in universe.tickers if eligible.get(t, False) and t not in {cash, benchmark}),
        key=lambda t: (-alpha.get(t, float("-inf")), t),
    )
    rank = {t: i + 1 for i, t in enumerate(ranked)}

    rows: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]  # JSON rows
    for ticker, sector in zip(universe.tickers, universe.sectors, strict=True):
        sentinel = "cash" if ticker == cash else "benchmark" if ticker == benchmark else None
        target = float(book.weights.get(ticker, 0.0))
        prior = float(previous.get(ticker, 0.0))
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "alpha": alpha.get(ticker),
                "alpha_rank": rank.get(ticker),
                "eligible": eligible.get(ticker),
                "previous_weight": prior,
                "target_weight": target,
                "weight_change": target - prior,
                "target_notional_usd": round(target * book.portfolio_notional_usd, 2),
                # The same weights in percent, for a reader (the console page).
                "previous_weight_pct": round(prior * 100.0, 3),
                "target_weight_pct": round(target * 100.0, 3),
                "action": _action(
                    previous=prior,
                    target=target,
                    eligible=bool(eligible.get(ticker, False)),
                    sentinel=sentinel,
                ),
                "held_sessions": (
                    None
                    if sentinel is not None
                    else int(ages.get(ticker, 0)) + 1
                    if target > _ZERO
                    else 0
                ),
            }
        )
    rows.sort(key=lambda r: (-r["target_weight"], r["alpha_rank"] or 10**9, r["ticker"]))
    for position, row in enumerate(rows, start=1):
        row["book_position"] = position  # 1 is the largest target
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    return {
        "schema_version": ORDER_BOOK_SCHEMA_VERSION,
        "trading_day": book.trading_day,
        "s_champion": book.s_champion,
        "m_champion": book.m_champion,
        "u_champion": book.u_champion,
        "portfolio_notional_usd": book.portfolio_notional_usd,
        # A live session's NAV-based sizing (broker NetLiquidation, capped at
        # the declared notional); None when sized on the declared notional.
        "sizing": None if book.sizing is None else book.sizing.record(),
        "turnover_one_way_ratio": book.turnover_one_way_ratio,
        "cost_bps": book.cost_bps,
        "session_inputs_key": book.session_inputs_key,
        "previous_book": "given" if previous_weights is not None else "all_cash",
        "written_at_utc": written_at_utc,
        "summary": {
            "names": len(rows),
            "positions": sum(
                1 for r in rows if r["target_weight"] > _ZERO and r["action"] != "cash"
            ),
            "cash_weight": float(book.weights.get(cash, 0.0)),
            "actions": dict(sorted(counts.items())),
        },
        "rows": rows,
    }


def record_order_book(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    book: TargetBook,
    *,
    previous_weights: dict[str, float] | None,
    now: dt.datetime,
    held_sessions: Mapping[str, int] | None = None,
) -> str | None:
    """Write and lock the day's order book, and record one metric. Never raises.

    ``held_sessions`` is the clock the session constructed under (see
    :func:`order_book_document`). Returns the key written, or ``None`` when the
    write failed.
    """
    stamp = now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = None
    try:
        key = order_book_key(book.trading_day)
        document = order_book_document(
            book,
            previous_weights=previous_weights,
            written_at_utc=stamp,
            held_sessions=held_sessions,
        )
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        ctx.record_output(key, payload, ORDER_BOOK_SCHEMA_VERSION)
        _, refused = lock_books_records(
            ctx.store, [key], trading_day=dt.date.fromisoformat(book.trading_day)
        )
        status, reason = "OK", f"{len(document['rows'])} names written to {key}"
        if refused:
            status, reason = "FAIL", f"written but not locked: {refused[key]}"
    except Exception as exc:  # noqa: BLE001 - a missing record must never stop the orders
        log.exception("order book write failed")
        status, reason, key = "FAIL", f"order book not written: {type(exc).__name__}: {exc}", None
    try:
        ctx.record_metric(
            {
                "name": ORDER_BOOK_METRIC,
                "module": "crucible_trader.order_book",
                "metric_type": "operational",
                "value": 1.0 if key else 0.0,
                "unit": "count",
                "n_floor": 0,
                "status": status,
                "status_reason": reason[:1000],
                "source_path": ORDER_BOOKS_PREFIX,
                "last_updated_utc": stamp,
            }
        )
    except Exception:  # noqa: BLE001
        log.exception("order book metric not recorded")
    return key


def _read_book(store: Store, trading_day: str) -> list[dict[str, Any]] | None:  # pyright: ignore[reportExplicitAny]  # JSON rows
    """``trading_day``'s order-book rows, or ``None`` when there is no usable book.

    A book that is absent, unreadable, or filed for another day ends a holding
    run (:func:`held_sessions_from_order_books`); it never stops the session.
    The cost is a shorter clock -- the hold protects a name for fewer sessions,
    the behaviour before the hold existed -- never a refused day.
    """
    key = order_book_key(trading_day)
    try:
        if not store.exists(key):
            return None
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        if document.get("trading_day") != trading_day or not isinstance(document["rows"], list):
            raise ValueError(f"not the {trading_day} order book")
        return [r for r in document["rows"] if isinstance(r, dict)]
    except Exception as exc:  # noqa: BLE001 - a holding run ends; the session goes on
        log.warning("order book %s unreadable, ending the holding run: %s", key, exc)
        return None


def _held_names(rows: list[dict[str, Any]]) -> set[str]:  # pyright: ignore[reportExplicitAny]
    """The names a book holds: a positive target, sentinels excluded."""
    return {
        str(r["ticker"])
        for r in rows
        if r.get("action") not in _SENTINEL_ACTIONS
        and isinstance(r.get("target_weight"), (int, float))
        and r["target_weight"] > _ZERO
    }


def _stamped(rows: list[dict[str, Any]], held: set[str]) -> dict[str, int] | None:  # pyright: ignore[reportExplicitAny]
    """The book's own ``held_sessions`` stamps for its held names, or ``None``
    when any held name lacks a whole count >= 1 (a book written before the
    field existed)."""
    stamps: dict[str, int] = {}
    for row in rows:
        ticker = str(row["ticker"])
        if ticker not in held:
            continue
        count = row.get("held_sessions")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return None
        stamps[ticker] = count
    return stamps


def held_sessions_from_order_books(
    store: Store, trading_day: str, *, max_books: int = HELD_SESSIONS_WALK_LIMIT
) -> dict[str, int]:
    """By ticker, how many consecutive sessions' books held each name into ``trading_day``.

    The minimum hold's clock (`alpha-engine-config-I12024`;
    `crucible.slots.strategy.construct_book`'s ``held_sessions``): 1 for a name
    the previous session's book entered, and so on back. Only names the
    previous session's book holds (``target_weight > 0``) appear.

    One read in steady state: the previous session's book stamps each row with
    its own count. A book written before that field existed is counted by
    walking back one trading session at a time, at most ``max_books`` books; a
    stamped book met on the way finishes the count. A session with no usable
    book is a GAP and ends every run at it -- a held day with no book, or a
    record that cannot be read, is not evidence the name was held.
    """
    day = previous_trading_day(dt.date.fromisoformat(trading_day))
    rows = _read_book(store, day.isoformat())
    if rows is None:
        return {}
    run = _held_names(rows)
    stamps = _stamped(rows, run)
    if stamps is not None:
        return stamps
    counts = dict.fromkeys(run, 1)
    books = 1
    while run and books < max_books:
        day = previous_trading_day(day)
        rows = _read_book(store, day.isoformat())
        if rows is None:
            break
        books += 1
        run &= _held_names(rows)
        stamps = _stamped(rows, run)
        if stamps is not None:
            for ticker in run:
                counts[ticker] += stamps[ticker]
            break
        for ticker in run:
            counts[ticker] += 1
    return counts
