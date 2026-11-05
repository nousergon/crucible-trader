"""WORM retention on the v2 back-office books (alpha-engine-config-I13191).

The ledger, the broker statements it ties to, the daily reconciliations and
the exception records, proposals and scores its adjustments cite are an investment
adviser's books and records: SEC Rule 204-2 keeps them for five years from
the end of the fiscal year of the last entry, the first two easily
accessible (declared in alpha-engine-config ``ARTIFACT_REGISTRY.yaml ::
records_retention``, alpha-engine-config-I13128). Versioning and "the trader
role has no delete" hold that only against the trader; this step makes each
written version undeletable by anyone, root included, until that date.

**COMPLIANCE mode, not GOVERNANCE.** The books are append-only by policy
(portfolio-accounting-policy Rule 3: a correction is a new entry, never an
edit), so no legitimate operation ever needs to remove a version early, and
GOVERNANCE would leave exactly that removal path open to any principal holding
``s3:BypassGovernanceRetention`` -- which every ``s3:*`` admin does. That path
is the one WORM storage exists to close. The cost is that a version written by
mistake is kept, and billed, until its date; these are kilobyte JSON files.

**Never raises, never changes a payload.** It runs after every write of the
cycle (step 5h of :func:`crucible_trader.reconciliation_job.reconcile_cycle`),
as a separate ``PutObjectRetention`` on each key's current version, rather
than as lock parameters on the write itself (``Store.put_bytes``'s
``object_lock_mode``): a write that carried the lock would fail outright if the
grant were missing or S3 refused the lock, and the book of record must never
go unwritten because its lock could not be applied. A refused lock is a
``FAIL`` metric on the run's manifest naming each key, and the step moves on.
The cost is a window of seconds in which a fresh version is unlocked, during
which the only identity that can write these prefixes holds no delete.

Retention only ever extends: COMPLIANCE refuses a shorter date, and a re-run
asks for the same date it asked for the first time (the date is derived from
the record's trading year, not from the clock), so a re-run is idempotent.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable
from typing import Any

from crucible.store import S3Store, Store

log = logging.getLogger(__name__)

#: The books prefixes, relative to the store root. A key outside them is never
#: locked: a bucket-wide lock would also freeze documents that are
#: legitimately rewritten and are not records (evidence, pins, kill switch).
BOOKS_PREFIXES: tuple[str, ...] = (
    "trader/ledger/",
    "trader/broker_statements/",
    "trader/reconciliation/",
    "trader/exceptions/",
    "trader/exception_proposals/",
    "trader/exception_scores/",
    # The day's order book (`crucible_trader.order_book`): every name the
    # champion considered and its target. Locked by the session that writes
    # it, not by step 5h (alpha-engine-config order-book page, 2026-11-05).
    "trader/order_books/",
)

#: The reviewers' decisions (1c/1d): the approvals the ledger's clearing
#: adjustments cite, written by the decision writer's own role, never by the
#: trader, so they are locked by that writer as it records each one
#: (:mod:`crucible_trader.exception_decision_writer`), not by step 5h.
DECISIONS_PREFIX = "trader/exception_decisions/"

RETENTION_MODE = "COMPLIANCE"

#: Rule 204-2(e)(1): five years from the end of the fiscal year during which
#: the last entry was made. The adviser's fiscal year is the calendar year.
RETENTION_YEARS = 5

BOOKS_RETENTION_METRIC = "books_worm_retention"


def is_books_key(key: str) -> bool:
    return key.startswith(BOOKS_PREFIXES)


def retain_until(trading_day: dt.date) -> dt.datetime:
    """Midnight UTC after the last day of ``trading_day``'s fiscal year plus
    :data:`RETENTION_YEARS` (a 2026 record is kept through 2031-12-31)."""
    return dt.datetime(trading_day.year + RETENTION_YEARS + 1, 1, 1, tzinfo=dt.UTC)


def lock_books_records(
    store: Store,
    keys: Iterable[str],
    *,
    trading_day: dt.date,
    prefixes: tuple[str, ...] = BOOKS_PREFIXES,
) -> tuple[list[str], dict[str, str]]:
    """Apply COMPLIANCE retention to each key in ``keys`` under ``prefixes``
    (the trader's books by default).

    Returns ``(locked, refused)``: the keys locked, and ``{key: reason}`` for
    each one S3 refused. A store with no Object Lock concept (``LocalStore``)
    locks nothing and refuses nothing. Never raises.
    """
    locked: list[str] = []
    refused: dict[str, str] = {}
    try:
        if not isinstance(store, S3Store):
            return locked, refused
        until = retain_until(trading_day)
        for key in dict.fromkeys(k for k in keys if k.startswith(prefixes)):
            try:
                store.client.put_object_retention(
                    Bucket=store.bucket,
                    Key=store._s3_key(key),  # pyright: ignore[reportPrivateUsage]  # the store's own key mapping
                    Retention={"Mode": RETENTION_MODE, "RetainUntilDate": until},
                )
            except Exception as exc:  # noqa: BLE001 - a refused lock must never stop the trader
                refused[key] = f"{type(exc).__name__}: {exc}"[:300]
            else:
                locked.append(key)
    except Exception as exc:  # noqa: BLE001 - nor may anything around it
        refused["*"] = f"{type(exc).__name__}: {exc}"[:300]
    return locked, refused


def lock_cycle_books(ctx: Any, *, now: str) -> None:  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    """Lock every books record this run wrote (``ctx.outputs``) and record the
    outcome as one metric. Never raises."""
    try:
        keys = [str(output["key"]) for output in ctx.outputs]
        locked, refused = lock_books_records(ctx.store, keys, trading_day=ctx.trading_day)
        if not locked and not refused:
            return
        if refused:
            log.warning("books WORM retention refused for %d key(s): %s", len(refused), refused)
            status = "FAIL"
            reason = f"{len(refused)} books record(s) not locked: " + "; ".join(
                f"{k}: {v}" for k, v in sorted(refused.items())
            )
        else:
            status = "OK"
            reason = (
                f"{len(locked)} books record(s) under {RETENTION_MODE} retention until "
                f"{retain_until(ctx.trading_day).date().isoformat()}"
            )
        ctx.record_metric(
            {
                "name": BOOKS_RETENTION_METRIC,
                "module": "crucible_trader.books_retention",
                "metric_type": "operational",
                "value": float(len(locked)),
                "unit": "count",
                "n_floor": 0,
                "status": status,
                "status_reason": reason[:1000],
                "source_path": BOOKS_PREFIXES[0],
                "last_updated_utc": now,
            }
        )
    except Exception:  # noqa: BLE001 - the lock step must not raise into the trader
        log.exception("books WORM retention step failed")
