"""The opening-cost evidence pack (`alpha-engine-config-I14137`, part 1 of -I13787).

The ledger's ``opening_balance`` entry books every position at 0 cents, so the
securities account of a name that has since been sold holds the whole sale
proceeds. Fixing that changes the books, so a human sees the per-name
arithmetic first. This module COMPUTES that arithmetic and files it. It posts
nothing: no journal is written, and nothing calls it from the daily job.

``as_of`` is the ledger day the pack is struck on (balances, quantities).
The statement the ledger opened from is the opening entry's own
``evidence.statement_day``, and that day prices the names:

* a Flex statement for it under ``<flex>/YYYY-MM/YYYYMMDD-*.xml`` gives each
  name's ``OpenPosition`` SUMMARY ``costBasisMoney`` (``ib_flex_cost_basis``);
* otherwise shares x the settled close on that day (``settled_close``, the
  plan's Rule 7 valuation).

Every figure is whole cents via :func:`crucible_trader.ledger.to_cents`.
Refuse, never degrade: a name with no price, an ambiguous Flex file, a Flex
statement that disagrees with the stored one, or an existing pack all raise
:class:`OpeningCostError`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import cast
from xml.etree import ElementTree as ET

from crucible.keys import trader_broker_statement_key
from crucible.store import ETAG_ABSENT, PointerConflictError, Store, open_store

from crucible_trader.broker_statement import (
    BrokerStatement,
    BrokerStatementError,
    statement_from_document,
)
from crucible_trader.ledger import (
    LEDGER_RECONCILIATION_PREFIX,
    OPENING,
    REALIZED_GAINS,
    SECURITIES,
    Entry,
    LedgerError,
    Posting,
    _panel_closes,  # pyright: ignore[reportPrivateUsage]
    ledger_from,
    load_journals,
    to_cents,
)
from crucible_trader.reconciliation_job import DEFAULT_FLEX_URI, FLEX_URI_ENV

OPENING_COST_SCHEMA_VERSION = "opening_cost_pack.v1"
SOURCE_FLEX = "ib_flex_cost_basis"
SOURCE_CLOSE = "settled_close"


class OpeningCostError(ValueError):
    """The pack cannot be built, or filed, without guessing."""


def opening_cost_key(tenant: str, as_of: str) -> str:
    """Inside the trader's existing ``WriteItsOwnLedgerReconciliationView`` grant."""
    return f"{LEDGER_RECONCILIATION_PREFIX}{tenant}/opening_cost/{as_of}.json"


def _next_day(day: str) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()


def _opening_statement_day(entries: Sequence[tuple[str, Entry]], tenant: str, account: str) -> str:
    for day, entry in entries:
        if entry.kind == "opening_balance" and entry.tenant == tenant and entry.account == account:
            return str(entry.evidence.get("statement_day") or day)
    raise OpeningCostError(f"{tenant}/{account} has no opening_balance entry up to the as-of day")


def _statement(store: Store, day: str, account: str) -> BrokerStatement:
    key = trader_broker_statement_key(day)
    try:
        statement = statement_from_document(json.loads(store.get_bytes(key)))
    except (BrokerStatementError, KeyError, ValueError, OSError) as exc:
        raise OpeningCostError(f"opening statement {key} is unreadable: {exc}") from exc
    if statement.account != account:
        raise OpeningCostError(f"opening statement {key} is for a different account")
    return statement


def _read_flex(text: str, key: str, statement: BrokerStatement) -> dict[str, int]:
    """``costBasisMoney`` in cents per SUMMARY ``OpenPosition`` of the Flex file.

    Reads only what the pack needs, and refuses a file that is not exactly the
    stored statement's account, day and positions. (The reconciliation job owns
    the full Flex parser; this module never reaches into it.)"""
    body = text.lstrip("\ufeff \t\r\n")
    # IB's documents declare no DTD; one that does is not one (entity expansion).
    if "<!DOCTYPE" in body.upper() or "<!ENTITY" in body.upper():
        raise OpeningCostError(f"Flex file {key} carries a DTD or entity declaration")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise OpeningCostError(f"Flex file {key} is not well-formed: {exc}") from exc
    stamp = statement.trading_day.replace("-", "")
    statements = root.findall("FlexStatements/FlexStatement")
    if (
        root.tag != "FlexQueryResponse"
        or len(statements) != 1
        or statements[0].get("accountId") != statement.account
        or (statements[0].get("fromDate"), statements[0].get("toDate")) != (stamp, stamp)
    ):
        raise OpeningCostError(
            f"Flex file {key} is not one FlexStatement for {statement.account} on {stamp}"
        )
    quantity: dict[str, int] = {}
    cost: dict[str, int] = {}
    for row in statements[0].findall("OpenPositions/OpenPosition"):
        if row.get("levelOfDetail", "SUMMARY") != "SUMMARY":
            continue  # lot rows restate the summary row
        symbol = row.get("symbol", "")
        try:
            quantity[symbol] = int(Decimal((row.get("position") or "").replace(",", "")))
            cost[symbol] = to_cents(Decimal((row.get("costBasisMoney") or "").replace(",", "")))
        except (LedgerError, ArithmeticError) as exc:
            raise OpeningCostError(
                f"Flex file {key}: {symbol} position or cost unreadable"
            ) from exc
    if quantity != dict(statement.positions):
        raise OpeningCostError(
            f"Flex file {key} positions disagree with the stored {statement.trading_day} statement"
        )
    return cost


def _flex_cost_basis(flex: Store, statement: BrokerStatement) -> tuple[str, dict[str, int]] | None:
    """The Flex file for the statement day and its cost basis per name in cents,
    or ``None`` when no Flex file exists for that day."""
    stamp = statement.trading_day.replace("-", "")
    month = statement.trading_day[:7] + "/"
    keys = sorted(
        k
        for k in flex.list_keys(month)
        if k.rsplit("/", 1)[-1].startswith(f"{stamp}-") and k.endswith(".xml")
    )
    if not keys:
        return None
    if len(keys) > 1:
        raise OpeningCostError(
            f"{len(keys)} Flex files match {stamp}-*.xml: {keys}; refusing to pick"
        )
    cost = _read_flex(flex.get_bytes(keys[0]).decode("utf-8"), keys[0], statement)
    return keys[0], cost


def _balanced(postings: Sequence[Posting], what: str) -> None:
    total = sum(p.amount_cents for p in postings)
    if total != 0:
        raise OpeningCostError(f"{what} does not balance: {total} cents")


def _entry(postings: Sequence[Posting], what: str) -> dict[str, object]:
    _balanced(postings, what)
    return {"kind": "adjustment", "postings": [p.to_document() for p in postings]}


def opening_cost_pack(
    store: Store,
    tenant: str,
    account: str,
    as_of: str,
    *,
    flex_store: Store | None = None,
) -> dict[str, object]:
    """The proposed opening-cost correction for ``tenant``/``account`` on ``as_of``.

    ``flex_store`` holds the Flex statements (default: the reconciliation job's
    Flex location). Read-only toward everything: nothing is written."""
    journals = load_journals(store, tenant, before=_next_day(as_of))
    ledger = ledger_from(journals)
    statement_day = _opening_statement_day(
        [(d, e) for d, es in sorted(journals.items()) for e in es], tenant, account
    )
    statement = _statement(store, statement_day, account)
    tickers = sorted(statement.positions)

    flex = _flex_cost_basis(
        flex_store
        if flex_store is not None
        else open_store(os.environ.get(FLEX_URI_ENV, DEFAULT_FLEX_URI), dry_run=True),
        statement,
    )
    lines: list[dict[str, object]] = []
    if flex is not None:
        flex_key, basis = flex
        for t in tickers:
            lines.append(
                {
                    "ticker": t,
                    "shares": statement.positions[t],
                    "cost_cents": basis[t],
                    "source": SOURCE_FLEX,
                    "source_record": flex_key,
                }
            )
    else:
        closes = _panel_closes(store)(statement_day, tickers)
        absent = [t for t in tickers if t not in closes]
        if absent:
            raise OpeningCostError(f"no settled close on {statement_day} for {', '.join(absent)}")
        for t in tickers:
            lines.append(
                {
                    "ticker": t,
                    "shares": statement.positions[t],
                    "cost_cents": to_cents(
                        Decimal(statement.positions[t]) * Decimal(repr(closes[t]))
                    ),
                    "source": SOURCE_CLOSE,
                    "source_record": statement_day,
                }
            )

    cost = {cast("str", ln["ticker"]): cast("int", ln["cost_cents"]) for ln in lines}
    first = [Posting(SECURITIES + t, c) for t, c in cost.items()]
    first.append(Posting(OPENING, -sum(cost.values())))

    second: list[Posting] = []
    for t, c in cost.items():
        if ledger.shares(tenant, account, SECURITIES + t) == 0:
            left = ledger.cents(tenant, account, SECURITIES + t) + c
            second += [Posting(SECURITIES + t, -left), Posting(REALIZED_GAINS + t, left)]

    accounts = sorted({p.account for p in first + second})
    before = {a: ledger.cents(tenant, account, a) for a in accounts}
    after = dict(before)
    for p in first + second:
        after[p.account] += p.amount_cents
    # Both entries are proven balanced by `_entry` below, and a ledger's own trial
    # balance is zero by construction, so this is 0 unless one of those is broken.
    trial_balance = ledger.trial_balance() + sum(p.amount_cents for p in first + second)
    return {
        "schema_version": OPENING_COST_SCHEMA_VERSION,
        "tenant": tenant,
        "account": account,
        "as_of": as_of,
        "statement_day": statement_day,
        "lines": lines,
        "entry_1": _entry(first, "entry_1"),
        "entry_2": _entry(second, "entry_2"),
        "before": before,
        "after": after,
        "trial_balance_after_cents": trial_balance,
    }


def write_pack(store: Store, pack: Mapping[str, object]) -> str:
    """File ``pack`` create-only; refuse if one already exists for the day."""
    key = opening_cost_key(cast("str", pack["tenant"]), cast("str", pack["as_of"]))
    payload = json.dumps(pack, indent=2, sort_keys=True).encode("utf-8")
    try:
        store.compare_and_swap(key, ETAG_ABSENT, payload)
    except PointerConflictError as exc:
        raise OpeningCostError(f"{key} already exists; a filed pack is never overwritten") from exc
    return key


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Callable[[str | None, bool], Store] = lambda uri, write: open_store(
        uri, dry_run=not write
    ),
    flex_opener: Callable[[str], Store] = lambda uri: open_store(uri, dry_run=True),
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.opening_cost",
        description="Print the opening-cost evidence pack; --write files it create-only.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--as-of", required=True, help="ledger day, YYYY-MM-DD")
    parser.add_argument("--flex-uri", default=None, help="Flex statements location")
    parser.add_argument("--write", action="store_true", help="file the pack, create-only")
    args = parser.parse_args(argv)
    write = cast("bool", args.write)
    store = opener(cast("str | None", args.store), write)
    flex_uri = cast("str | None", args.flex_uri) or os.environ.get(FLEX_URI_ENV, DEFAULT_FLEX_URI)
    pack = opening_cost_pack(
        store,
        cast("str", args.tenant),
        cast("str", args.account),
        cast("str", args.as_of),
        flex_store=flex_opener(flex_uri),
    )
    sys.stdout.write(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    if write:
        sys.stderr.write(f"wrote {write_pack(store, pack)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
