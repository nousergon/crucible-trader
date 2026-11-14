"""The monthly rebuild-from-raw check: back office 1h, the last of its checks.

`alpha-engine-config-I13129` (epic `-I12892`; nous-ergon-ops
`plans/2026-11-02-back-office-implementation.md` § Control standard: "the
close, the performance statement and the fee bill, rebuilt from stored inputs,
must match what was published"). The daily rebuild of the ledger's own views
lives in :func:`crucible_trader.ledger.run_checks`; this module does the same
for the three MONTHLY documents.

Once a month is closed and its statement filed, the first reconcile run that
finds no rebuild for it:

* rebuilds the **close** with :func:`~crucible_trader.month_end_close.close_month`
  from what the store holds -- the tenant's journals, the IB statement and the
  ledger view for the month's last trading day, the previous month's close --
  and the settled closes;
* rebuilds the **statement** and the **fee bill** with
  :func:`~crucible_trader.performance_statement.build_statement` from the
  REBUILT close (not the published one, so the chain is raw end to end), the
  previous month's published statement and the benchmark closes;
* compares each to the published document field by field, and files the
  result at :func:`rebuild_key`, create-only, so a month is rebuilt once.

Any difference is a BREAK: it is listed in the document's ``breaks`` and turns
the :data:`REBUILD_METRIC` MetricRecord FAIL.

**Failure isolation.** :func:`verify_month_rebuild` rides `trader.reconcile`
as step 5g and NEVER raises: it changes nothing the job decides or pages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from crucible.store import ETAG_ABSENT, PointerConflictError, Store

from crucible_trader.benchmark_closes import with_benchmark_closes
from crucible_trader.exception_record import PAPER_FUND_TENANT
from crucible_trader.month_end_close import (
    LEDGER_CLOSE_PREFIX,
    CloseSource,
    close_key,
    close_month,
    panel_close_source,
    previous_month,
    stored_close_inputs,
)
from crucible_trader.performance_statement import (
    FEE_SCHEDULES,
    SIMULATED_FEE_SCHEDULE,
    benchmark_returns,
    build_statement,
    statement_key,
)

REBUILD_SCHEMA_VERSION = "month_rebuild.v1"
REBUILD_METRIC = "month_rebuild"
_FEE_BILL = "simulated_fee_bill"


def _schedule_named_by(published: Mapping[str, object]) -> Mapping[str, object]:
    """The registered schedule the published bill names, so a bill filed before
    a schedule change still rebuilds under its own terms. An unknown or absent
    id rebuilds under today's schedule, and the bill's ``schedule`` then differs:
    a bill naming terms nobody registered is a break, never a match."""
    bill = cast("Mapping[str, object]", published.get(_FEE_BILL) or {})
    named = cast("Mapping[str, object]", bill.get("schedule") or {})
    return FEE_SCHEDULES.get(str(named.get("schedule_id")), SIMULATED_FEE_SCHEDULE)


def rebuild_key(tenant: str, month: str, book: str | None = None) -> str:
    """The month's rebuild result: written once, never rewritten."""
    return f"{close_key(tenant, month, book)[: -len('.json')]}/rebuild.json"


def _read_json(store: Store, key: str) -> dict[str, Any] | None:  # pyright: ignore[reportExplicitAny]
    if not store.exists(key):
        return None
    return cast("dict[str, Any]", json.loads(store.get_bytes(key).decode("utf-8")))  # pyright: ignore[reportExplicitAny]


def _normal(document: Mapping[str, object]) -> dict[str, object]:
    """What the document reads back as once published (tuples become lists)."""
    return cast("dict[str, object]", json.loads(json.dumps(document, sort_keys=True)))


def _compare(
    key: str, published: Mapping[str, object], rebuilt: Mapping[str, object]
) -> dict[str, object]:
    a, b = _normal(published), _normal(rebuilt)
    # A field absent on one side differs from an explicit null on the other (codex 165efe5b).
    differs = sorted(k for k in set(a) | set(b) if k not in a or k not in b or a[k] != b[k])
    return {"key": key, "matches": not differs, "differs": differs}


def rebuild_month(
    store: Store,
    *,
    tenant: str,
    account: str,
    month: str,
    day: str,
    closes: CloseSource,
    now: str,
    book: str | None = None,
) -> dict[str, object] | None:
    """Rebuild ``month``'s close, statement and fee bill in ``book`` (``None``:
    the live book) from stored inputs and compare each to what was published.
    Reads only, through the close's own reader
    (:func:`~crucible_trader.month_end_close.stored_close_inputs`), so a
    prior-period book's last month is rebuilt against the same bridge. ``None``
    while the close or the statement is not published yet."""
    published_close = _read_json(store, close_key(tenant, month, book))
    published_statement = _read_json(store, statement_key(tenant, month, book))
    if published_close is None or published_statement is None:
        return None
    close = close_month(
        stored_close_inputs(
            store,
            tenant=tenant,
            account=account,
            month=month,
            day=day,
            closes=closes,
            now=now,
            book=book,
        )
    )
    documents: dict[str, dict[str, object]] = {
        "close": _compare(close_key(tenant, month, book), published_close, close)
    }
    statement_at = statement_key(tenant, month, book)
    try:
        statement = build_statement(
            close,
            _read_json(store, statement_key(tenant, previous_month(month), book)),
            benchmark_returns(
                with_benchmark_closes(store, closes),
                str(close["start_trading_day"]),
                str(close["last_trading_day"]),
            ),
            _schedule_named_by(published_statement),
        )
    except Exception as exc:  # noqa: BLE001 - a statement that no longer builds is the break
        why = f"does not rebuild: {type(exc).__name__}: {exc}"[:300]
        documents["statement"] = {"key": statement_at, "matches": False, "differs": [why]}
        documents["fee_bill"] = {"key": statement_at, "matches": False, "differs": [why]}
    else:
        documents["statement"] = _compare(
            statement_at,
            {k: v for k, v in published_statement.items() if k != _FEE_BILL},
            {k: v for k, v in statement.items() if k != _FEE_BILL},
        )
        documents["fee_bill"] = _compare(
            statement_at,
            cast("Mapping[str, object]", published_statement.get(_FEE_BILL) or {}),
            cast("Mapping[str, object]", statement[_FEE_BILL]),
        )
    breaks = [
        {
            "check": "rebuild_from_raw",
            "document": name,
            "detail": f"{month} {name} differs from its rebuild in {result['differs']}"[:400],
        }
        for name, result in documents.items()
        if not result["matches"]
    ]
    return {
        **({} if book is None else {"book": book}),
        "schema_version": REBUILD_SCHEMA_VERSION,
        "tenant": tenant,
        "account": account,
        "month": month,
        "rebuilt_on": day,
        "generated_at": now,
        "matches": not breaks,
        "documents": documents,
        "breaks": breaks,
    }


def verify_month_rebuild(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    *,
    day: str,
    account: str,
    now: str,
    tenant: str = PAPER_FUND_TENANT,
    closes: CloseSource | None = None,
    book: str | None = None,
) -> dict[str, object] | None:
    """Step 5g of `trader.reconcile`: rebuild the month before ``day``'s month
    once its close and statement are published, unless it is already rebuilt.
    NEVER raises; returns the rebuild document, or ``None``. ``book``: a
    declared prior-period book to rebuild instead of the live one."""
    month = previous_month(day[:7])
    key = None
    try:
        store = ctx.store
        key = rebuild_key(tenant, month, book)
        if store.exists(key):
            return None
        document = rebuild_month(
            store,
            tenant=tenant,
            account=account,
            month=month,
            day=day,
            closes=closes or panel_close_source(store),
            now=now,
            book=book,
        )
        if document is None:
            return None
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        try:
            ctx.record_output_cas(key, ETAG_ABSENT, payload, REBUILD_SCHEMA_VERSION)
        except PointerConflictError:
            return None  # a concurrent run rebuilt it first; a rebuild is never rewritten
    except Exception as exc:  # noqa: BLE001 - must never change what the job decides or pages
        _metric(
            ctx,
            now,
            key,
            "FAIL",
            None,
            f"{month} rebuild NOT run on {day}: {type(exc).__name__}: {exc}",
        )
        return None
    breaks = cast("list[Mapping[str, object]]", document["breaks"])
    _metric(
        ctx,
        now,
        key,
        "OK" if not breaks else "FAIL",
        float(len(breaks)),
        f"{month} close, statement and fee bill rebuilt from raw on {day}: "
        + ("all match" if not breaks else "; ".join(str(b["detail"]) for b in breaks)),
    )
    return document


def _metric(
    ctx: Any,  # pyright: ignore[reportExplicitAny]
    now: str,
    key: str | None,
    status: str,
    value: float | None,
    reason: str,
) -> None:
    try:
        ctx.record_metric(
            {
                "name": REBUILD_METRIC,
                "module": "crucible_trader.month_rebuild",
                "metric_type": "operational",
                "value": value,
                "unit": "count" if value is not None else None,
                "n_floor": 0,
                "status": status,
                "status_reason": reason[:400],
                "source_path": key or LEDGER_CLOSE_PREFIX,
                "last_updated_utc": now,
            }
        )
    except Exception:  # noqa: BLE001, S110 - a metric must not raise into the trader either
        pass


__all__ = [
    "REBUILD_METRIC",
    "REBUILD_SCHEMA_VERSION",
    "rebuild_key",
    "rebuild_month",
    "verify_month_rebuild",
]
