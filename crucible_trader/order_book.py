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

**The book is built before the open** (Brian 2026-11-09/10,
`alpha-engine-config-I12024`). `crucible_trader.order_book_plan` runs at
08:40 ET on every trading day and writes the day's book at
``trader/order_books/{day}.json`` with ``stage: pre_open``. It places no order
and builds no router. The session still builds its own book from what it sees
at its start, and then:

* **No book at the day's key yet** (the plan did not run): the session's book
  is written there, ``stage: session``, exactly as before the plan existed.
* **A book is there and the session's book matches it**: nothing is written.
  The pre-open book IS the traded book, and the metric names it.
* **A book is there and the session's book differs** (the account moved, or a
  price did, between 08:40 and the start): the session's book is written at
  ``trader/order_books/as_traded/{day}.json``, ``stage: as_traded``, naming the
  pre-open version it differs from and the names whose target moved.

**Why a second key and not a second version of the first.** The day's key is
under COMPLIANCE retention from the moment it is written, so a version written
before the open can never be removed or shortened. A later write to the same
key would still succeed (it adds a version), but every reader of the day's key
would then see the session's book and not the one announced before the open,
and the pre-open version would only be reachable by version id. So the day's
key is written ONCE, by a create-only conditional write (``If-None-Match``),
whichever of the plan and the session gets there first, and the as-traded book
has a key of its own. Both are locked and both stay readable by key.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from crucible.store import ETAG_ABSENT, PointerConflictError, Store

from crucible_trader.books_retention import lock_books_records
from crucible_trader.construction import TargetBook

log = logging.getLogger(__name__)

__all__ = [
    "AS_TRADED_PREFIX",
    "ORDER_BOOKS_PREFIX",
    "ORDER_BOOK_METRIC",
    "ORDER_BOOK_SCHEMA_VERSION",
    "STAGE_AS_TRADED",
    "STAGE_PRE_OPEN",
    "STAGE_SESSION",
    "as_traded_key",
    "changed_targets",
    "order_book_document",
    "order_book_key",
    "record_order_book",
    "write_pre_open_order_book",
]

ORDER_BOOK_SCHEMA_VERSION = "order_book.v1"
ORDER_BOOKS_PREFIX = "trader/order_books/"
#: The session's book when it differs from the pre-open one. Under
#: :data:`ORDER_BOOKS_PREFIX`, so the same retention and the same write grant
#: cover it.
AS_TRADED_PREFIX = f"{ORDER_BOOKS_PREFIX}as_traded/"
ORDER_BOOK_METRIC = "trader_order_book_written"

#: Who wrote a book: the pre-open plan, the session at the day's key (no
#: pre-open book existed), or the session at the as-traded key.
STAGE_PRE_OPEN = "pre_open"
STAGE_SESSION = "session"
STAGE_AS_TRADED = "as_traded"

#: Fields that say when and by which stage a book was written. Everything else
#: in the document is the decision, so two books with equal decisions are the
#: same book.
_PROVENANCE_FIELDS = frozenset({"written_at_utc", "stage", "pre_open_book"})

#: Weights at or below this are no position (float noise from the engine).
_ZERO = 1e-9


def order_book_key(trading_day: str) -> str:
    dt.date.fromisoformat(trading_day)  # raises on anything but an ISO day
    return f"{ORDER_BOOKS_PREFIX}{trading_day}.json"


def as_traded_key(trading_day: str) -> str:
    dt.date.fromisoformat(trading_day)
    return f"{AS_TRADED_PREFIX}{trading_day}.json"


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
    stage: str = STAGE_SESSION,
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # JSON document
    universe = book.universe
    cash = universe.tickers[universe.cash_idx]
    benchmark = universe.tickers[universe.benchmark_idx]
    alpha = book.alpha or {}
    eligible = book.eligible or {}
    # No previous book is an all-cash book (construction's w_initial), so the
    # cash row's previous weight is 1.0, not 0 (codex ac64b346aa382999).
    previous = previous_weights if previous_weights is not None else {cash: 1.0}

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
        "stage": stage,
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


def _decision(document: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # JSON document
    return {k: v for k, v in document.items() if k not in _PROVENANCE_FIELDS}


def changed_targets(
    pre_open: dict[str, Any],  # pyright: ignore[reportExplicitAny]  # JSON document
    session: dict[str, Any],  # pyright: ignore[reportExplicitAny]  # JSON document
) -> list[str]:
    """The tickers whose target weight differs between two books, sorted.

    A name in only one of the two books counts as changed.
    """
    before = {r["ticker"]: r["target_weight"] for r in pre_open.get("rows", [])}
    after = {r["ticker"]: r["target_weight"] for r in session.get("rows", [])}
    return sorted(
        t
        for t in set(before) | set(after)
        if t not in before or t not in after or abs(before[t] - after[t]) > _ZERO
    )


def _stamp(now: dt.datetime) -> str:
    return now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(document: dict[str, Any]) -> bytes:  # pyright: ignore[reportExplicitAny]  # JSON document
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8")


def _lock(store: Store, key: str, trading_day: str) -> str | None:
    """Lock one book; the refusal reason, or None when locked (or nothing to lock)."""
    _, refused = lock_books_records(store, [key], trading_day=dt.date.fromisoformat(trading_day))
    return refused.get(key) or refused.get("*")


def write_pre_open_order_book(
    store: Store,
    book: TargetBook,
    *,
    previous_weights: dict[str, float] | None,
    now: dt.datetime,
) -> tuple[str, bool]:
    """Write the day's book before the open, create-only, and lock it. RAISES.

    Returns ``(key, written)``: ``written`` is False when a book was already at
    the day's key (a rerun of the plan, or a session that ran first), which is
    left exactly as it is. Unlike :func:`record_order_book` this raises: the
    plan has no orders to protect, and a book it could not write or lock is its
    job's failure.
    """
    key = order_book_key(book.trading_day)
    document = order_book_document(
        book, previous_weights=previous_weights, written_at_utc=_stamp(now), stage=STAGE_PRE_OPEN
    )
    try:
        store.compare_and_swap(key, ETAG_ABSENT, _payload(document))
    except PointerConflictError:
        return key, False
    refused = _lock(store, key, book.trading_day)
    if refused is not None:
        raise RuntimeError(f"pre-open order book {key} written but not locked: {refused}")
    return key, True


def _record_session_book(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    book: TargetBook,
    *,
    previous_weights: dict[str, float] | None,
    stamp: str,
) -> tuple[str, str, str]:
    """``(key, status, reason)`` for the session's book. Raises on a store error."""
    key = order_book_key(book.trading_day)
    document = order_book_document(book, previous_weights=previous_weights, written_at_utc=stamp)
    try:
        ctx.record_output_cas(key, ETAG_ABSENT, _payload(document), ORDER_BOOK_SCHEMA_VERSION)
    except PointerConflictError:
        pass
    else:
        reason = f"{len(document['rows'])} names written to {key} (no pre-open book)"
        refused = _lock(ctx.store, key, book.trading_day)
        if refused is not None:
            return key, "FAIL", f"written but not locked: {refused}"
        return key, "OK", reason
    existing = json.loads(ctx.store.get_bytes(key))
    seen = f"{existing.get('stage', 'unstaged')} book written {existing.get('written_at_utc')}"
    if _decision(existing) == _decision(document):
        return key, "OK", f"{key} reused: the session's book matches the {seen}"
    changed = changed_targets(existing, document)
    traded = as_traded_key(book.trading_day)
    document["stage"] = STAGE_AS_TRADED
    document["pre_open_book"] = {
        "key": key,
        "stage": existing.get("stage"),
        "written_at_utc": existing.get("written_at_utc"),
        "changed_targets": changed,
    }
    ctx.record_output(traded, _payload(document), ORDER_BOOK_SCHEMA_VERSION)
    reason = (
        f"as-traded book written to {traded}: it differs from the {seen} at {key} "
        f"on {len(changed)} target(s)"
    )
    refused = _lock(ctx.store, traded, book.trading_day)
    if refused is not None:
        return traded, "FAIL", f"written but not locked: {refused}"
    return traded, "OK", reason


def record_order_book(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    book: TargetBook,
    *,
    previous_weights: dict[str, float] | None,
    now: dt.datetime,
) -> str | None:
    """Keep the session's order book, and record one metric. Never raises.

    Writes the book at the day's key when no book is there, reuses the pre-open
    book when the session's matches it, and otherwise writes the as-traded book
    at its own key (module docstring). Returns the key that holds the book the
    session traded, or ``None`` when that could not be kept.
    """
    stamp = _stamp(now)
    key = None
    try:
        key, status, reason = _record_session_book(
            ctx, book, previous_weights=previous_weights, stamp=stamp
        )
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
