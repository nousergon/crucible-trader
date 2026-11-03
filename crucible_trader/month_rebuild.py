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

import datetime as dt
import json
from collections.abc import Mapping
from typing import Any, cast

from crucible.keys import trader_broker_statement_key  # pyright: ignore[reportMissingTypeStubs]
from crucible.store import ETAG_ABSENT, PointerConflictError, Store  # pyright: ignore[reportMissingTypeStubs]

from crucible_trader.broker_statement import statement_from_document
from crucible_trader.exception_record import PAPER_FUND_TENANT
from crucible_trader.ledger import ledger_reconciliation_key, load_journals
from crucible_trader.month_end_close import (
    LEDGER_CLOSE_PREFIX,
    CloseInputs,
    CloseSource,
    close_key,
    close_month,
    month_trading_days,
    panel_close_source,
    previous_month,
)
from crucible_trader.performance_statement import (
    benchmark_returns,
    build_statement,
    statement_key,
)

REBUILD_SCHEMA_VERSION = "month_rebuild.v1"
REBUILD_METRIC = "month_rebuild"
_FEE_BILL = "simulated_fee_bill"


def rebuild_key(tenant: str, month: str) -> str:
    """The month's rebuild result: written once, never rewritten."""
    return f"{close_key(tenant, month)[: -len('.json')]}/rebuild.json"


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
    differs = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
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
) -> dict[str, object] | None:
    """Rebuild ``month``'s close, statement and fee bill from stored inputs and
    compare each to what was published. Reads only. ``None`` while the close or
    the statement is not published yet."""
    published_close = _read_json(store, close_key(tenant, month))
    published_statement = _read_json(store, statement_key(tenant, month))
    if published_close is None or published_statement is None:
        return None
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    last_day = month_trading_days(month)[-1]
    ib = _read_json(store, trader_broker_statement_key(last_day))
    close = close_month(
        CloseInputs(
            tenant=tenant,
            account=account,
            month=month,
            run_day=day,
            journals=load_journals(store, tenant, before=nxt),
            statement=statement_from_document(ib) if ib else None,
            view=_read_json(store, ledger_reconciliation_key(tenant, last_day)),
            prior=_read_json(store, close_key(tenant, previous_month(month))),
            closes=closes,
            detected_at=now,
        )
    )
    documents: dict[str, dict[str, object]] = {
        "close": _compare(close_key(tenant, month), published_close, close)
    }
    statement_at = statement_key(tenant, month)
    try:
        statement = build_statement(
            close,
            _read_json(store, statement_key(tenant, previous_month(month))),
            benchmark_returns(
                closes, str(close["start_trading_day"]), str(close["last_trading_day"])
            ),
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
) -> dict[str, object] | None:
    """Step 5g of `trader.reconcile`: rebuild the month before ``day``'s month
    once its close and statement are published, unless it is already rebuilt.
    NEVER raises; returns the rebuild document, or ``None``."""
    month = previous_month(day[:7])
    key = None
    try:
        store = ctx.store
        key = rebuild_key(tenant, month)
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
