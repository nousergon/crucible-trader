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
* ``sector_label`` -- the name's GICS sector for a reader (see
  :func:`sector_labels`), ``null`` when the feature layer does not label it
  and on the two sentinels. A DISPLAY field: ``sector`` stays the engine's own
  `universe.sectors` value, which today is ``__unsectored__`` by design (no
  sector cap: Brian's ruling, `alpha-engine-config-I13578`);
* ``previous_shares`` / ``target_shares`` / ``share_price_usd`` -- the shares
  the broker held into the session, the whole shares the book targets after
  the day's trades, and the decision price both were sized at (see
  :class:`ShareBasis`). ``null`` on a session that sized no orders (shadow)
  and on the two sentinels.

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
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.features.compute import read_features
from crucible.keys import features_key
from crucible.store import Store

from crucible_trader.books_retention import lock_books_records
from crucible_trader.construction import TargetBook

log = logging.getLogger(__name__)

__all__ = [
    "ORDER_BOOKS_PREFIX",
    "ORDER_BOOK_METRIC",
    "ORDER_BOOK_SCHEMA_VERSION",
    "SECTOR_LABEL_COLUMN",
    "ShareBasis",
    "order_book_document",
    "order_book_key",
    "record_order_book",
    "sector_labels",
]

ORDER_BOOK_SCHEMA_VERSION = "order_book.v1"
ORDER_BOOKS_PREFIX = "trader/order_books/"
ORDER_BOOK_METRIC = "trader_order_book_written"

#: Weights at or below this are no position (float noise from the engine).
_ZERO = 1e-9

#: The compiled feature column a row's ``sector_label`` is read from
#: (`crucible.features.registry` `sector_raw`: the GICS sector of the newest
#: constituents snapshot fetched before the session). The same feature artifact
#: the session's construction already reads for ADV, so no new read grant.
SECTOR_LABEL_COLUMN = "sector_raw"

#: The float slack in the target-share floor -- the SAME value as
#: `crucible_trader.order_router.plan_orders`' ``_SHARE_EPSILON`` (asserted by
#: test), so a row's ``target_shares`` is the target the plan sized against.
_SHARE_EPSILON = 1e-6


@dataclass(frozen=True)
class ShareBasis:
    """What the session sized its orders on, for the book's share columns.

    ``positions`` is the broker's shares by name held into the session and
    ``prices`` the decision price the plan used for each name it priced
    (`crucible_trader.order_router.plan_orders`). A record, never an input to
    sizing: the router hands this over after planning and nothing reads it back.
    """

    positions: Mapping[str, int]
    prices: Mapping[str, float]

    def target_shares(self, symbol: str, weight: float, notional_usd: float) -> int | None:
        """``floor(weight * notional / price)``, as the plan sizes it; None unpriced."""
        if weight <= _ZERO:
            return 0
        price = self.prices.get(symbol)
        if price is None or not (math.isfinite(price) and price > 0):
            return None
        return math.floor(weight * notional_usd / price + _SHARE_EPSILON)


def sector_labels(
    store: Store, trading_day: str, names: Iterable[str], *, feature_version: str | None = None
) -> tuple[dict[str, str], str | None]:
    """Each of ``names``' GICS sector on ``trading_day``, and the key it was read from.

    Read from the compiled feature layer's :data:`SECTOR_LABEL_COLUMN`; a name it
    leaves null or blank is absent from the mapping. ``({}, None)`` when the
    day's feature artifact does not exist or carries no sector column. May
    raise on an unreadable artifact; :func:`record_order_book` catches it.
    """
    key = features_key(feature_version or DEFAULT_FEATURE_VERSION, trading_day)
    if not store.exists(key):
        return {}, None
    frame = read_features(store.get_bytes(key))
    if SECTOR_LABEL_COLUMN not in frame.columns:
        return {}, None
    wanted = set(names)
    labels: dict[str, str] = {}
    tickers = cast("Iterable[object]", frame["ticker"])
    values = cast("Iterable[object]", frame[SECTOR_LABEL_COLUMN])
    for ticker, label in zip(tickers, values, strict=True):
        name = str(ticker)
        if name in wanted and isinstance(label, str) and label.strip():
            labels[name] = label.strip()
    return labels, key


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
    sectors: Mapping[str, str] | None = None,
    sectors_source: str | None = None,
    shares: ShareBasis | None = None,
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # JSON document
    """The `order_book.v1` document for ``book``.

    ``sectors`` (:func:`sector_labels`, read from ``sectors_source``) fills each
    row's ``sector_label``; ``shares`` (:class:`ShareBasis`) fills its share
    columns. Either absent leaves those keys ``null``; neither changes a weight.
    """
    universe = book.universe
    cash = universe.tickers[universe.cash_idx]
    benchmark = universe.tickers[universe.benchmark_idx]
    alpha = book.alpha or {}
    eligible = book.eligible or {}
    # No previous book is an all-cash book (construction's w_initial), so the
    # cash row's previous weight is 1.0, not 0 (codex ac64b346aa382999).
    previous = previous_weights if previous_weights is not None else {cash: 1.0}
    labels = sectors or {}

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
        basis = None if sentinel is not None else shares
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
                "sector_label": None if sentinel is not None else labels.get(ticker),
                "previous_shares": None if basis is None else int(basis.positions.get(ticker, 0)),
                "target_shares": (
                    None
                    if basis is None
                    else basis.target_shares(ticker, target, book.portfolio_notional_usd)
                ),
                "share_price_usd": None if basis is None else basis.prices.get(ticker),
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
        # Where each row's `sector_label` came from (a feature artifact key), and
        # whether its share columns are the session's own plan.
        "sector_label_source": sectors_source,
        "share_basis": None if shares is None else "router_plan",
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


def _display_shares(share_basis: Callable[[], ShareBasis | None] | None) -> ShareBasis | None:
    """The router's share basis, or None. Never raises."""
    if share_basis is None:
        return None
    try:
        return share_basis()
    except Exception:  # noqa: BLE001 - a display column must never stop the record
        log.exception("order book share basis unavailable")
        return None


def _display_sectors(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    book: TargetBook,
    feature_version: str | None,
) -> tuple[dict[str, str], str | None]:
    """:func:`sector_labels` for the book's universe, or ``({}, None)``. Never raises."""
    try:
        return sector_labels(
            ctx.store, book.trading_day, book.universe.tickers, feature_version=feature_version
        )
    except Exception:  # noqa: BLE001 - a display column must never stop the record
        log.exception("order book sector labels unavailable")
        return {}, None


def record_order_book(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    book: TargetBook,
    *,
    previous_weights: dict[str, float] | None,
    now: dt.datetime,
    share_basis: Callable[[], ShareBasis | None] | None = None,
    feature_version: str | None = None,
) -> str | None:
    """Write and lock the day's order book, and record one metric. Never raises.

    ``share_basis`` is the router's (`PaperOrderRouter.share_basis`), called
    once here; ``feature_version`` the version the session constructed under
    (None: the default). A share basis or a sector read that fails leaves those
    columns ``null`` and the book is written anyway -- the display columns are
    never a reason to lose the record.

    Returns the key written, or ``None`` when the write failed.
    """
    stamp = now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    shares = _display_shares(share_basis)
    sectors, sectors_source = _display_sectors(ctx, book, feature_version)
    key = None
    try:
        key = order_book_key(book.trading_day)
        document = order_book_document(
            book,
            previous_weights=previous_weights,
            written_at_utc=stamp,
            sectors=sectors,
            sectors_source=sectors_source,
            shares=shares,
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
