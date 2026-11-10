"""Backfill one missing reconcile day from an IB Daily Activity Statement.

`alpha-engine-config-I13232` (parent `-I13193`, epic `-I12892`). The October
close cannot finish from statement files alone: `back_office_dry_run` books
only days with a stored `trader/reconciliation/{day}.json`, and the close's
completeness check wants the day's journal, ledger view, IB statement AND
reconciliation. A filed statement supplies one of the four. This module supplies
all four the only honest way: it runs the REAL reconciler
(:func:`crucible_trader.reconciliation_job.run_reconciliation`, the job body
`trader.reconcile` runs every evening) for that day, with the broker's side read
from the activity statement IB itself issued for it.

Three properties, enforced rather than requested:

1. **The broker side is IB's own document.** :func:`parse_activity_statement`
   reads the statement's ``Open Positions`` (stocks, whole shares) and its
   ``Cash Report`` ``Ending Cash`` row in USD into a ``broker_statement.v2``
   with ``source: ib_paper``. It refuses a non-``DU`` account, any non-USD
   row, a statement whose period is not exactly the day, and anything it
   cannot read as one position per symbol. Nothing is reconstructed from fills
   or a ledger (`broker_statement.py` property 1). The accrual is recorded as
   ``None`` (not recorded): a field kept only for measurement is never guessed.
2. **A dry run by default, and the dry run IS the apply's plan.** The cycle
   runs against `crucible.store.capturing` over the real store: real reads,
   every write recorded and none performed. ``--apply`` writes exactly the
   captured objects, CREATE-ONLY: it refuses before the first write if any of
   them already exists, if the day already has a reconciliation (it is not a
   missing day), or inside 09:00-16:30 ET on a weekday, when the live
   reconcile and the trader own the store. Nothing is overwritten, so there is
   nothing to back up; the refusal is the backup.
3. **Failure-isolated from the trader.** Nothing in the trading path imports
   this module; it is an operator command. A paging or void reconcile is an
   expected OUTCOME of a backfill (the day may have real breaks) and is
   reported in the summary, never raised; the run manifest records it as the
   live job would.

The writes the live job makes that are not the day's own are left out of the
apply and listed: the back office's ``current.json`` queue pointer belongs to
the latest live run, and a backfill of an earlier day must not move it.

    python -m crucible_trader.statement_backfill --store URI --day 2026-10-14 \\
        --statement DU_20261014.csv [--fills PATH] [--apply]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import pathlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from zoneinfo import ZoneInfo

from crucible.keys import trader_reconciliation_key
from crucible.store import (
    ETAG_ABSENT,
    PointerConflictError,
    Store,
    capture_ledger_of,
    capturing,
    open_store,
)

from crucible_trader.back_office_dry_run import dry_run, load_fills, write_out
from crucible_trader.books_retention import lock_books_records
from crucible_trader.broker_statement import (
    PAPER_ACCOUNT_PREFIX,
    BrokerStatement,
    BrokerStatementError,
)
from crucible_trader.exception_record import PAPER_FUND_TENANT
from crucible_trader.ledger import journal_key, ledger_reconciliation_key
from crucible_trader.reconcile_entrypoint import live_corporate_action_source
from crucible_trader.reconciliation import Fill
from crucible_trader.reconciliation_job import run_reconciliation

__all__ = [
    "BackfillRefused",
    "FiledStatementSource",
    "apply_writes",
    "backfill_day",
    "main",
    "parse_activity_statement",
    "rehearse_close",
]

SUMMARY_SCHEMA_VERSION = "statement_backfill.v1"

#: Captured writes a backfill never applies: pointers owned by the LATEST live
#: run. Matched on the key's last path segment.
LIVE_POINTERS = frozenset({"current.json"})

#: The window the live trader and reconcile own (America/New_York, weekdays).
_ET = ZoneInfo("America/New_York")
_WINDOW = (dt.time(9, 0), dt.time(16, 30))

_USD = "USD"
_BASE_SUMMARY = "Base Currency Summary"


class BackfillRefused(RuntimeError):
    """A precondition of the backfill does not hold; nothing was written."""


# -- 1. the statement -----------------------------------------------------------


def _rows(text: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(text.lstrip("﻿"))) if row]


def _sections(text: str) -> dict[str, list[dict[str, str]]]:
    """``{section: [data row as {header: value}]}``. A section may re-declare
    its header part-way (IB does, per asset class); each Data row is read
    against the header in force above it."""
    out: dict[str, list[dict[str, str]]] = {}
    headers: dict[str, list[str]] = {}
    for row in _rows(text):
        if len(row) < 2:
            continue
        section, kind, values = row[0].strip(), row[1].strip(), row[2:]
        if kind == "Header":
            headers[section] = [v.strip() for v in values]
        elif kind == "Data":
            if section not in headers:
                raise BrokerStatementError(f"{section!r} has a Data row before its Header")
            out.setdefault(section, []).append(
                {h: v.strip() for h, v in zip(headers[section], values, strict=False)}
            )
    return out


def _number(raw: str, what: str) -> float:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        raise BrokerStatementError(f"{what}: {raw!r} is not a number") from None


def _field(rows: Sequence[Mapping[str, str]], name: str, section: str) -> str:
    values = [r.get("Field Value", "") for r in rows if r.get("Field Name") == name]
    if len(values) != 1 or not values[0]:
        raise BrokerStatementError(
            f"{section} must carry exactly one {name!r} (found {len(values)})"
        )
    return values[0]


def parse_activity_statement(text: str, *, trading_day: str) -> BrokerStatement:
    """IB's Daily Activity Statement CSV for ``trading_day`` as a broker statement.

    Raises :class:`BrokerStatementError` on anything that is not exactly one
    paper account's USD stock positions and USD ending cash for that one day.
    """
    sections = _sections(text)
    period = _field(sections.get("Statement", []), "Period", "Statement")
    try:
        day = dt.datetime.strptime(period, "%B %d, %Y").date().isoformat()
    except ValueError:
        raise BrokerStatementError(
            f"statement period {period!r} is not one day; a backfill reads a DAILY statement"
        ) from None
    if day != trading_day:
        raise BrokerStatementError(f"statement period is {day}, not {trading_day}")

    info = sections.get("Account Information", [])
    account = _field(info, "Account", "Account Information")
    if not account.startswith(PAPER_ACCOUNT_PREFIX):
        raise BrokerStatementError(
            f"account {account!r} is not an IB paper account ({PAPER_ACCOUNT_PREFIX}...); "
            "the paper -> live crossing is a reserved action"
        )
    base = _field(info, "Base Currency", "Account Information")
    if base != _USD:
        raise BrokerStatementError(f"base currency is {base!r}, not {_USD}")

    positions: dict[str, int] = {}
    for row in sections.get("Open Positions", []):
        if row.get("DataDiscriminator", "Summary") != "Summary":
            continue  # lot rows restate the summary row
        symbol, currency = row.get("Symbol", ""), row.get("Currency", "")
        if currency != _USD:
            raise BrokerStatementError(f"position {symbol!r} is in {currency!r}, not {_USD}")
        if row.get("Asset Category") != "Stocks":
            raise BrokerStatementError(
                f"position {symbol!r} is {row.get('Asset Category')!r}; only Stocks reconcile"
            )
        quantity = _number(row.get("Quantity", ""), f"position {symbol!r} quantity")
        if not quantity.is_integer():
            raise BrokerStatementError(
                f"position {symbol!r} quantity {quantity} is not whole shares"
            )
        if symbol in positions:
            raise BrokerStatementError(f"position {symbol!r} appears twice")
        positions[symbol] = int(quantity)

    cash: list[float] = []
    for row in sections.get("Cash Report", []):
        currency = row.get("Currency", "")
        if currency not in (_USD, _BASE_SUMMARY):
            raise BrokerStatementError(f"cash report row in {currency!r}, not {_USD}")
        if row.get("Currency Summary") == "Ending Cash" and currency == _USD:
            cash.append(_number(row.get("Total", ""), "Ending Cash (USD)"))
    if len(cash) != 1:
        raise BrokerStatementError(
            f"cash report must carry exactly one USD Ending Cash row (found {len(cash)})"
        )

    return BrokerStatement(
        account=account,
        trading_day=trading_day,
        positions=positions,
        cash_usd=cash[0],
        source="ib_paper",
        accrued_cash_usd=None,
    )


class FiledStatementSource:
    """A :class:`~crucible_trader.broker_statement.StatementSourceProtocol` that
    answers one day, from a statement already parsed, and refuses any other."""

    def __init__(self, statement: BrokerStatement) -> None:
        self.statement = statement

    def read_statement(self, trading_day: str) -> BrokerStatement:
        if trading_day != self.statement.trading_day:
            raise BrokerStatementError(
                f"asked for {trading_day}; this backfill holds {self.statement.trading_day} only"
            )
        return self.statement


# -- 2. the cycle, captured --------------------------------------------------------


def backfill_day(
    backend: Store,
    statement: BrokerStatement,
    *,
    fills: Sequence[Fill] = (),
    corporate_actions: Any = None,  # pyright: ignore[reportExplicitAny]
    runner: Callable[..., Any] = run_reconciliation,  # pyright: ignore[reportExplicitAny]
) -> tuple[Store, dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Run the real reconcile for ``statement``'s day against a capturing view
    of ``backend``. Returns that view (holding the captured objects) and the
    summary. Writes nothing to ``backend``.

    A ``backend`` that is already capturing is used as the view itself, so
    several days and then the month's close can run in ONE capture and see
    each other's writes (:func:`rehearse_close`). Wrapping it again would not:
    `crucible.store.capturing` starts a fresh overlay over the same backend,
    so the inner capture's writes are invisible to the outer one."""
    day = statement.trading_day
    if backend.exists(trader_reconciliation_key(day)):
        raise BackfillRefused(
            f"{trader_reconciliation_key(day)} already exists: {day} is not a missing day"
        )
    shared = capture_ledger_of(backend)
    view = backend if shared is not None else capturing(backend)
    start = len(shared.writes) if shared is not None else 0
    outcome, error = "ok", None
    try:
        runner(
            view,
            FiledStatementSource(statement),
            trading_day=dt.date.fromisoformat(day),
            fills=tuple(fills),
            cash_flows=(),
            corporate_actions=corporate_actions
            if corporate_actions is not None
            else live_corporate_action_source(),
            run_mode="replay",
        )
    except Exception as exc:  # noqa: BLE001 - a paging day is an outcome, not a crash
        outcome, error = "failed", f"{type(exc).__name__}: {exc}"
    ledger = capture_ledger_of(view)
    writes = ledger.writes[start:] if ledger is not None else ()
    keys = list(dict.fromkeys(w.key for w in writes))
    summary: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "trading_day": day,
        "account": statement.account,
        "positions": len(statement.positions),
        "cash_usd": statement.cash_usd,
        "fills": len(fills),
        "reconcile": outcome,
        "error": error,
        "keys_to_write": [k for k in keys if k.rsplit("/", 1)[-1] not in LIVE_POINTERS],
        "keys_not_applied": [k for k in keys if k.rsplit("/", 1)[-1] in LIVE_POINTERS],
        "locked_writes": sorted({w.key for w in writes if w.object_lock_mode}),
        "has_reconciliation": trader_reconciliation_key(day) in keys,
        # The day's journal and its ledger view: a cycle whose ledger posting
        # was refused wrote a reconciliation but no books (codex 29e6af2958a0c547).
        # A journal an earlier, stopped apply already wrote is not re-written:
        # the ledger verified it byte for byte before writing the view
        # (`ledger.post_daily_ledger`), so the view stands for both (codex 40108050).
        "has_ledger": (
            journal_key(PAPER_FUND_TENANT, day) in keys
            or view.exists(journal_key(PAPER_FUND_TENANT, day))
        )
        and ledger_reconciliation_key(PAPER_FUND_TENANT, day) in keys,
    }
    return view, summary


def rehearse_close(
    backend: Store,
    statements: Sequence[BrokerStatement],
    *,
    month: str,
    now: str,
    fills: Mapping[str, Sequence[Fill]] | None = None,
    corporate_actions: Any = None,  # pyright: ignore[reportExplicitAny]
) -> tuple[Store, dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Backfill every statement's day and run the month's back-office dry run,
    all in one capture of ``backend``. This is the proof the close needs
    BEFORE ``--apply`` makes the books permanent: the close reads the
    backfilled days exactly as it would after the apply. Writes nothing to
    ``backend``.

    Each day is backfilled in date order BETWEEN the stored days the dry run
    books, not all first: its reconcile posts the day's journal, and a stored
    earlier day with no journal yet would then be refused as inserted behind
    it (codex d411bf68)."""
    twice = sorted(
        {
            s.trading_day
            for s in statements
            if [t.trading_day for t in statements].count(s.trading_day) > 1
        }
    )
    if twice:
        raise BackfillRefused(f"one statement per day: {twice} given more than once")
    view = capturing(backend)
    by_day = fills or {}
    days: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]

    def backfill(statement: BrokerStatement) -> Callable[[], None]:
        def run() -> None:
            days.append(
                backfill_day(
                    view,
                    statement,
                    fills=by_day.get(statement.trading_day, ()),
                    corporate_actions=corporate_actions,
                )[1]
            )

        return run

    summary = dry_run(
        view,
        month=month,
        now=now,
        fills=by_day,
        backfill={s.trading_day: backfill(s) for s in statements},
    )
    summary["backfill"] = days
    return view, summary


# -- 3. the apply -----------------------------------------------------------------


def _in_window(now: dt.datetime) -> bool:
    local = now.astimezone(_ET)
    return local.weekday() < 5 and _WINDOW[0] <= local.time() < _WINDOW[1]


def apply_writes(
    backend: Store,
    view: Store,
    summary: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    now: dt.datetime,
) -> list[str]:
    """Copy the captured objects to ``backend``, create-only, then put the
    books among them under WORM retention exactly as the live cycle's step 5h
    does (:func:`crucible_trader.books_retention.lock_books_records`, same
    prefixes, same date from the later of the day's trading year and the
    apply's year). Every refusal is checked
    before the first write. A refused lock never unwinds the writes; it is
    recorded in ``summary["lock_refused"]`` (``summary["locked"]`` lists the
    keys locked).

    **A refusal part-way still locks what it wrote, and a re-run resumes it**
    (codex 40108050). A writer landing between the scan and a put stops the
    apply with earlier keys already created; those books are locked in a
    ``finally`` (``summary["written"]`` lists them), never left without
    retention. A key that already holds exactly the captured bytes is the same
    write, not a conflict: it is skipped, and locked again. Writes go in three
    tiers: the broker statement and the ledger (which replay byte for byte),
    then the day's reconciliation -- the record that makes the day "not
    missing" to :func:`backfill_day` -- then everything stamped with its run's
    time (exceptions, proposals, score, manifest, claim). An apply stopped in
    the first tier re-runs and completes, skipping what it already wrote; one
    stopped after the reconciliation has the day's books written and locked,
    and is refused by name rather than overwritten."""
    if _in_window(now):
        raise BackfillRefused(
            f"it is {now.astimezone(_ET):%H:%M} ET on a weekday, inside 09:00-16:30 ET; "
            "re-run --apply outside market hours"
        )
    if not summary["has_reconciliation"]:
        raise BackfillRefused(
            f"the cycle wrote no reconciliation ({summary['error']}); nothing to apply"
        )
    if not summary.get("has_ledger"):
        raise BackfillRefused(
            f"the cycle wrote no journal and ledger view for {summary['trading_day']} "
            f"({summary['error']}); nothing to apply"
        )
    if summary["locked_writes"]:
        raise BackfillRefused(f"Object Lock writes cannot be replayed: {summary['locked_writes']}")
    keys = cast("list[str]", summary["keys_to_write"])
    same = [k for k in keys if backend.exists(k) and _same(backend, view, k)]
    existing = [k for k in keys if k not in same and backend.exists(k)]
    if existing:
        raise BackfillRefused(f"create-only: {len(existing)} key(s) already exist: {existing}")
    sentinel = trader_reconciliation_key(str(summary["trading_day"]))
    order = sorted(
        (k for k in keys if k not in same),
        key=lambda k: 0 if k.startswith(_REPLAYABLE) else 1 if k == sentinel else 2,
    )
    written: list[str] = list(same)
    try:
        for key in order:
            # Create-only at the write itself: the scan above refuses before any
            # write, but a writer landing between the scan and this put must not
            # be overwritten (codex 9b1df526e391319d).
            try:
                backend.compare_and_swap(key, ETAG_ABSENT, view.get_bytes(key))
            except PointerConflictError:
                if not _same(backend, view, key):
                    raise BackfillRefused(
                        f"create-only: {key} was written during the apply; "
                        f"{len(written)} key(s) already written and locked: {written}"
                    ) from None
            written.append(key)
    finally:
        locked, refused = lock_books_records(
            backend,
            written,
            trading_day=dt.date.fromisoformat(str(summary["trading_day"])),
            written_on=now.astimezone(dt.UTC).date(),
        )
        summary["written"] = written
        summary["locked"], summary["lock_refused"] = locked, refused
    return keys


#: The books a re-run of the same statement reproduces byte for byte.
_REPLAYABLE: tuple[str, ...] = ("trader/broker_statements/", "trader/ledger/")


def _same(backend: Store, view: Store, key: str) -> bool:
    """Whether ``backend`` already holds exactly the captured bytes at ``key``."""
    try:
        return backend.get_bytes(key) == view.get_bytes(key)
    except KeyError:  # gone between the check and the read: not the same write
        return False


# -- CLI ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Callable[[str | None], Store] = lambda uri: open_store(uri),
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.statement_backfill",
        description="Backfill one missing reconcile day from an IB activity statement. "
        "Dry run unless --apply.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument(
        "--day",
        required=True,
        action="append",
        help="YYYY-MM-DD, the missing trading day (repeat with --close, one per --statement)",
    )
    parser.add_argument(
        "--statement",
        required=True,
        action="append",
        type=pathlib.Path,
        help="IB activity statement CSV, in --day order",
    )
    parser.add_argument("--fills", default=None, type=pathlib.Path, help="fills as JSON lines")
    parser.add_argument(
        "--apply", action="store_true", help="write the captured objects, create-only"
    )
    parser.add_argument(
        "--close",
        default=None,
        metavar="YYYY-MM",
        help="dry-run this month's close over the backfilled days (needs --out; no --apply)",
    )
    parser.add_argument("--out", default=None, type=pathlib.Path, help="local directory")
    parser.add_argument("--now", default=None, help="stamp for every close record")
    args = parser.parse_args(argv)
    days = cast("list[str]", args.day)
    paths = cast("list[pathlib.Path]", args.statement)
    if len(days) != len(paths):
        parser.error("give one --statement per --day")
    if args.close is not None:
        return _rehearse(parser, args, days, paths, opener)
    if len(days) != 1:
        parser.error("more than one --day needs --close")
    day = days[0]
    try:
        statement = parse_activity_statement(paths[0].read_text(encoding="utf-8"), trading_day=day)
        fills = load_fills(cast("pathlib.Path | None", args.fills)).get(day, [])
        backend = opener(cast("str | None", args.store))
        view, summary = backfill_day(backend, statement, fills=fills)
        summary["mode"] = "apply" if args.apply else "dry-run"
        if args.apply:
            summary["written"] = apply_writes(backend, view, summary, now=clock())
    except (BackfillRefused, BrokerStatementError, OSError, ValueError) as exc:
        sys.stderr.write(f"NO-GO: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if summary.get("lock_refused"):
        sys.stderr.write(
            f"{day}: written, but {len(summary['lock_refused'])} books record(s) are NOT under "
            "WORM retention (see lock_refused)\n"
        )
        return 2
    sys.stderr.write(
        f"{day}: reconcile {summary['reconcile']}; {len(summary['keys_to_write'])} key(s) "
        + ("written" if args.apply else "would be written (dry run; re-run with --apply)")
        + "\n"
    )
    return 0


def _rehearse(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    days: Sequence[str],
    paths: Sequence[pathlib.Path],
    opener: Callable[[str | None], Store],
) -> int:
    if args.apply:
        parser.error("--close is a dry run; --apply one --day at a time")
    out = cast("pathlib.Path | None", args.out)
    if out is None:
        parser.error("--close needs --out")
    if out.exists() and any(out.iterdir()):
        parser.error(f"--out {out} is not empty; a dry run never mixes with an earlier one")
    month = cast("str", args.close)
    now = cast("str | None", args.now) or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        statements = [
            parse_activity_statement(p.read_text(encoding="utf-8"), trading_day=d)
            for d, p in zip(days, paths, strict=True)
        ]
        fills = load_fills(cast("pathlib.Path | None", args.fills))
        view, summary = rehearse_close(
            opener(cast("str | None", args.store)), statements, month=month, now=now, fills=fills
        )
    except (BackfillRefused, BrokerStatementError, OSError, ValueError) as exc:
        sys.stderr.write(f"NO-GO: {exc}\n")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    write_out(view, summary, out)
    close = cast("Mapping[str, Any] | None", summary["close"])  # pyright: ignore[reportExplicitAny]
    missing = [
        e for e in (close or {}).get("exceptions", []) if e["class"] == "close_input_missing"
    ]
    sys.stderr.write(
        f"{month}: backfilled {', '.join(sorted(days))} (dry run); "
        f"close {close['status'] if close else 'not attempted'}, "
        f"{len(missing)} close_input_missing line(s); wrote {out}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
