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
* ``action`` -- what the book says about the name (see :func:`_action`).

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
from typing import Any

from crucible_trader.books_retention import lock_books_records
from crucible_trader.construction import TargetBook

log = logging.getLogger(__name__)

__all__ = [
    "ORDER_BOOKS_PREFIX",
    "ORDER_BOOK_METRIC",
    "ORDER_BOOK_SCHEMA_VERSION",
    "order_book_document",
    "order_book_key",
    "record_order_book",
]

ORDER_BOOK_SCHEMA_VERSION = "order_book.v1"
ORDER_BOOKS_PREFIX = "trader/order_books/"
ORDER_BOOK_METRIC = "trader_order_book_written"

#: Weights at or below this are no position (float noise from the engine).
_ZERO = 1e-9


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
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # JSON document
    universe = book.universe
    cash = universe.tickers[universe.cash_idx]
    benchmark = universe.tickers[universe.benchmark_idx]
    alpha = book.alpha or {}
    eligible = book.eligible or {}
    previous = previous_weights or {}

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
) -> str | None:
    """Write and lock the day's order book, and record one metric. Never raises.

    Returns the key written, or ``None`` when the write failed.
    """
    stamp = now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = None
    try:
        key = order_book_key(book.trading_day)
        document = order_book_document(
            book, previous_weights=previous_weights, written_at_utc=stamp
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
