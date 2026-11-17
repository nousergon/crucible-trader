"""Ledger views: the master ledger laid out for people to read (aec#13753).

Brian, 2026-11-16 17:27Z: "having the master ledgers and access to the master
ledgers to review are important pieces of this project". The console's
/page/books reads one views document per tenant per trading day, at
``trader/ledger/views/{tenant}/{day}.json``, holding three arrays:

* ``accounts`` -- the trial balance by account after the day closes: every
  ledger account's balance in cents (and shares, for the accounts that carry
  them), with ``trial_balance_cents`` beside it, which must be 0.
* ``postings`` -- every posting of the day's entries with the account's
  running balance after that entry, the entry's kind and event, and the store
  key of the document it was booked from (``source_key``).
* ``tieout`` -- cash and each instrument, ledger against broker: what ties,
  what broke today (with its exception ids) and what is still in suspense
  from an earlier day (with the gap entry and its exception ids).

**Derived, never of record.** Every balance is a replay of the stored
journals through :class:`crucible_trader.ledger.Ledger` (the one balance
engine) and every tie-out figure is read from the day's reconciliation view;
nothing is computed that the ledger does not already hold. So the documents
are written with a plain ``put_bytes``, not ``record_output``: they are not
books records, step 5h does not WORM-lock them, and a later run rewrites them.

**Never raises.** :func:`publish_ledger_views` is step 5d' of
``trader.reconcile``. A failure logs and returns ``None``; the job's decision,
page and metrics are exactly what they were without it. Each run also
backfills every earlier journal day that has no views document yet.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import cast

from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import Store

from crucible_trader.ledger import (
    PAPER_FUND_TENANT,
    Entry,
    Ledger,
    journal_key,
    ledger_reconciliation_key,
    load_journals,
)
from crucible_trader.order_intent_reconciliation import order_intent_reconciliation_key

log = logging.getLogger(__name__)

LEDGER_VIEWS_PREFIX = "trader/ledger/views/"
LEDGER_VIEWS_SCHEMA_VERSION = "ledger_views.v1"
CASH_ITEM = "USD"
_AFTER_EVERY_DAY = "9999-12-31"
_GAP_ITEM_PREFIX = "gap/"


def ledger_views_key(tenant: str, trading_day: str) -> str:
    journal_key(tenant, trading_day)  # the journal's tenant and date checks
    return f"{LEDGER_VIEWS_PREFIX}{tenant}/{trading_day}.json"


def source_key(entry: Entry) -> str:
    """The store key of the document an entry was booked from: the day's IB
    statement for statement-sourced entries, the day's order-intent
    reconciliation (which lists the IB execution ids) for fills and
    commissions, the day's broker reconciliation for the rest, and the
    journal itself when nothing else is named."""
    source, day = entry.evidence.get("source"), entry.trading_day
    if source == "ib_statement":
        return trader_broker_statement_key(day)
    if source in ("ib_execution", "ib_commission_report"):
        return order_intent_reconciliation_key(day)
    if source is not None:
        return trader_reconciliation_key(day)
    return journal_key(entry.tenant, day)


def _rows(value: object) -> list[Mapping[str, object]]:
    return cast("list[Mapping[str, object]]", value or [])


def _ids(value: object) -> list[str]:
    return [str(i) for i in cast("list[object]", value or [])]


def _int(value: object) -> int:
    return int(cast("int", value or 0))


def _replay(
    journals: Mapping[str, Sequence[Entry]], tenant: str, day: str
) -> tuple[Ledger, list[dict[str, object]], dict[tuple[str, str], int], set[tuple[str, str]]]:
    ledger = Ledger()
    postings: list[dict[str, object]] = []
    counts: dict[tuple[str, str], int] = {}
    carries_shares: set[tuple[str, str]] = set()
    for d in sorted(x for x in journals if x <= day):
        for entry in journals[d]:
            ledger.append(entry)
            for p in entry.postings:
                key = (entry.account, p.account)
                counts[key] = counts.get(key, 0) + 1
                if p.quantity is not None:
                    carries_shares.add(key)
                if d != day:
                    continue
                postings.append(
                    {
                        "tenant": tenant,
                        "trading_day": d,
                        "entry_id": entry.entry_id,
                        "kind": entry.kind,
                        "event_ref": entry.event_ref,
                        "broker_account": entry.account,
                        "ledger_account": p.account,
                        "amount_cents": p.amount_cents,
                        "quantity": p.quantity,
                        "running_balance_cents": ledger.cents(tenant, entry.account, p.account),
                        "running_quantity": (
                            None
                            if p.quantity is None
                            else ledger.shares(tenant, entry.account, p.account)
                        ),
                        "source": entry.evidence.get("source"),
                        "source_key": source_key(entry),
                        "journal_key": journal_key(tenant, d),
                        "reason": entry.evidence.get("reason"),
                        "decision_id": entry.evidence.get("decision_id"),
                        "exception_ids": _ids(entry.evidence.get("exception_ids")),
                    }
                )
    return ledger, postings, counts, carries_shares


def _accounts(
    ledger: Ledger,
    counts: Mapping[tuple[str, str], int],
    carries_shares: set[tuple[str, str]],
    tenant: str,
    day: str,
) -> list[dict[str, object]]:
    """``quantity`` is ``None`` for an account no posting has ever given shares."""
    return [
        {
            "tenant": tenant,
            "trading_day": day,
            "broker_account": broker_account,
            "ledger_account": ledger_account,
            "balance_cents": ledger.cents(tenant, broker_account, ledger_account),
            "quantity": (
                ledger.shares(tenant, broker_account, ledger_account)
                if (broker_account, ledger_account) in carries_shares
                else None
            ),
            "postings_to_date": n,
        }
        for (broker_account, ledger_account), n in sorted(counts.items())
    ]


def _open_gaps(
    reconciliation: Mapping[str, object], journals: Mapping[str, Sequence[Entry]]
) -> dict[str, list[dict[str, object]]]:
    """Gaps still in suspense, by item, with the entry that booked each."""
    by_id = {e.entry_id: e for entries in journals.values() for e in entries}
    checks = cast("Mapping[str, object]", reconciliation.get("checks") or {})
    gaps: dict[str, list[dict[str, object]]] = {}
    for row in _rows(checks.get("open_breaks")):
        entry = by_id.get(str(row.get("entry_id")))
        item = (
            entry.event_ref.removeprefix(_GAP_ITEM_PREFIX)
            if entry is not None
            else CASH_ITEM
            if row.get("kind") == "cash_gap"
            else "?"
        )
        gaps.setdefault(item, []).append(
            {
                "day": row.get("day"),
                "age": row.get("age"),
                "entry_id": row.get("entry_id"),
                "exception_ids": [] if entry is None else _ids(entry.evidence.get("exception_ids")),
            }
        )
    return gaps


def _tieout(
    reconciliation: Mapping[str, object],
    journals: Mapping[str, Sequence[Entry]],
    tenant: str,
    day: str,
) -> list[dict[str, object]]:
    breaks = {str(b.get("instrument")): b for b in _rows(reconciliation.get("breaks"))}
    gaps = _open_gaps(reconciliation, journals)
    suspense = cast("Mapping[str, object]", reconciliation.get("suspense") or {})
    suspense_shares = cast("Mapping[str, object]", suspense.get("shares") or {})
    ours = cast("Mapping[str, object]", reconciliation.get("ledger_positions") or {})
    theirs = cast("Mapping[str, object]", reconciliation.get("broker_positions") or {})
    rows: list[dict[str, object]] = []
    for item in [CASH_ITEM, *sorted(set(ours) | set(theirs) | (set(gaps) - {CASH_ITEM}))]:
        if item == CASH_ITEM:
            unit = "cents"
            ledger, broker = (
                reconciliation.get("ledger_cash_cents"),
                reconciliation.get("broker_cash_cents"),
            )
            in_suspense = _int(suspense.get("cash_cents"))
        else:
            unit, ledger, broker = "shares", ours.get(item, 0), theirs.get(item, 0)
            in_suspense = _int(suspense_shares.get(item))
        today = breaks.get(item)
        if today is not None:
            # The ledger as it stood before today's gap entry moved it to the broker.
            ledger = today.get("ledger")
        open_gaps = gaps.get(item, [])
        status = (
            "break_today"
            if today is not None
            else "in_suspense"
            if in_suspense or open_gaps
            else "ties"
        )
        rows.append(
            {
                "tenant": tenant,
                "trading_day": day,
                "item": item,
                "unit": unit,
                "ledger": ledger,
                "broker": broker,
                "delta": _int(broker) - _int(ledger),
                "in_suspense": in_suspense,
                "status": status,
                "tier": None if today is None else today.get("tier"),
                "exception_ids": list(
                    dict.fromkeys(
                        [
                            *([] if today is None else _ids(today.get("exception_ids"))),
                            *(i for gap in open_gaps for i in _ids(gap["exception_ids"])),
                        ]
                    )
                ),
                "open_since": min((str(g["day"]) for g in open_gaps), default=None),
                "suspense_entry_ids": [str(g["entry_id"]) for g in open_gaps],
            }
        )
    return rows


def day_views(
    journals: Mapping[str, Sequence[Entry]],
    reconciliation: Mapping[str, object],
    *,
    tenant: str,
    day: str,
) -> dict[str, object]:
    """The views document of ``tenant`` on ``day``: ``journals`` holds every
    journal of the tenant (days after ``day`` are ignored) and
    ``reconciliation`` that day's ledger reconciliation view."""
    ledger, postings, counts, carries_shares = _replay(journals, tenant, day)
    return {
        "schema_version": LEDGER_VIEWS_SCHEMA_VERSION,
        "tenant": tenant,
        "trading_day": day,
        "trial_balance_cents": ledger.trial_balance(),
        "clean": reconciliation.get("clean"),
        "sources": {
            "journal": journal_key(tenant, day),
            "ledger_reconciliation": ledger_reconciliation_key(tenant, day),
            "broker_statement": trader_broker_statement_key(day),
        },
        "accounts": _accounts(ledger, counts, carries_shares, tenant, day),
        "postings": postings,
        "tieout": _tieout(reconciliation, journals, tenant, day),
    }


def publish_ledger_views(
    store: Store, *, day: str, tenant: str = PAPER_FUND_TENANT
) -> list[str] | None:
    """Step 5d': write ``day``'s views document and every earlier missing one.
    Returns the keys written; NEVER raises (``None`` on failure)."""
    try:
        journals = load_journals(store, tenant, before=_AFTER_EVERY_DAY)
        present = set(store.list_keys(f"{LEDGER_VIEWS_PREFIX}{tenant}/"))
        written: list[str] = []
        for d in sorted(x for x in journals if x <= day):
            key = ledger_views_key(tenant, d)
            view_key = ledger_reconciliation_key(tenant, d)
            if (d != day and key in present) or not store.exists(view_key):
                continue
            reconciliation = json.loads(store.get_bytes(view_key).decode("utf-8"))
            document = day_views(journals, reconciliation, tenant=tenant, day=d)
            store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))
            written.append(key)
    except Exception:  # noqa: BLE001 - a view must never change what the job decides or pages
        log.warning("ledger views for %s NOT published", day, exc_info=True)
        return None
    return written
