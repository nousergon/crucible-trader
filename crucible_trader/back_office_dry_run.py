"""The back office's dry run of one month, offline: back office stage 2.

`alpha-engine-config-I13124` (epic `-I12892`; nous-ergon-ops
`plans/2026-11-02-back-office-implementation.md` stage 2, the October dry run).
One command runs the whole loop over a month that has already happened, and
NOTHING it does reaches the store:

1. **export** every stored break of the month as `exception.v1` records
   (:func:`crucible_trader.exception_export.export_exceptions`, one day at a
   time). A day whose stored documents cannot be exported is not skipped and
   does not stop the run: it is listed in ``export_failures``, a break to
   explain in its own right. So is every trading day of the month with no
   stored reconciliation at all (``days_without_reconciliation``);
2. **propose** over every record with the shadow proposer
   (:mod:`crucible_trader.exception_proposal`), filed as day files exactly as
   step 5c files them;
3. **book** each stored day into the v2 ledger
   (:func:`crucible_trader.ledger.post_daily_ledger`) from its stored broker
   statement and reconciliation, anchored on the previous stored statement.
   v2 kept no fills before the cutover, so ``--fills`` takes them as JSON lines
   (``trading_day``, ``ticker``, ``shares``, ``cash_usd``, optional
   ``commission_usd`` and ``exec_id``); without it every position change is a
   ledger gap booked to suspense, which the close then refuses on -- the plan's
   "any day that ledger cannot reproduce is itself a break to explain";
4. **close** the month (:func:`crucible_trader.month_end_close.close_prior_month`)
   as the first run of the next month would, and, if it closes, file its
   statement and fee bill (:func:`crucible_trader.performance_statement.file_statement`).
   A refused close files proposals for its lines (`alpha-engine-config-I13121`);
5. **score** every proposal against every decision the store already holds
   (:func:`crucible_trader.exception_decision.score`): the queue the reviewer
   would see.

**No write reaches the store.** It is opened with ``dry_run=True``
(`crucible.store.capturing`): every write is recorded in memory and served
back to the run's own reads, and the capturing store has no code path to the
backend's write. At the end every captured object is copied to
``OUT/store/<key>``, a local store tree, and ``OUT/summary.json`` says what
happened. Reviewer decisions can then be recorded against that tree without
touching S3 and the score rebuilt from it:

    python -m crucible_trader.back_office_dry_run --store URI --month 2026-10 \\
        --out DIR [--fills PATH] [--now YYYY-MM-DDTHH:MM:SSZ]
    python -m crucible_trader.exception_decision record --store DIR/store ...
    python -m crucible_trader.exception_decision score --store DIR/store
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from crucible.calendar import is_trading_day  # pyright: ignore[reportMissingTypeStubs]
from crucible.keys import (  # pyright: ignore[reportMissingTypeStubs]
    trader_broker_statement_key,
    trader_reconciliation_key,
)
from crucible.runner import RunContext  # pyright: ignore[reportMissingTypeStubs]
from crucible.store import (  # pyright: ignore[reportMissingTypeStubs]
    Store,
    capture_ledger_of,
    open_store,
)

from crucible_trader.broker_statement import BrokerStatement, statement_from_document
from crucible_trader.exception_decision import load_decisions, load_proposals, score
from crucible_trader.exception_export import export_exceptions
from crucible_trader.exception_proposal import (
    exception_proposals_key,
    proposal_day_document,
    propose,
)
from crucible_trader.ledger import post_daily_ledger
from crucible_trader.month_end_close import CloseSource, close_prior_month, month_trading_days
from crucible_trader.performance_statement import file_statement
from crucible_trader.reconciliation import CorporateActionSet, Fill
from crucible_trader.reconciliation_job import RECONCILE_JOB, RECONCILIATION_PREFIX

__all__ = ["dry_run", "load_fills", "main"]

SUMMARY_SCHEMA_VERSION = "back_office_dry_run.v1"

#: The ledger reads only a set's ``actions``; v2 stored none before the cutover.
_NO_ACTIONS = CorporateActionSet(resolved=frozenset())


def load_fills(path: pathlib.Path | None) -> dict[str, list[Fill]]:
    """``--fills``: one JSON object per line, grouped by ``trading_day``."""
    out: dict[str, list[Fill]] = {}
    if path is None:
        return out
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))  # pyright: ignore[reportExplicitAny]
        try:
            day = dt.date.fromisoformat(str(row["trading_day"])).isoformat()
            fill = Fill(
                str(row["ticker"]),
                int(row["shares"]),
                float(row["cash_usd"]),
                commission_usd=None
                if row.get("commission_usd") is None
                else float(row["commission_usd"]),
                exec_id=None if row.get("exec_id") is None else str(row["exec_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{n}: not a fill ({type(exc).__name__}: {exc})") from None
        out.setdefault(day, []).append(fill)
    return out


def _next_trading_day(day: str) -> str:
    current = dt.date.fromisoformat(day) + dt.timedelta(days=1)
    while not is_trading_day(current):
        current += dt.timedelta(days=1)
    return current.isoformat()


def _ctx(store: Store, day: str, now: str) -> RunContext:
    return RunContext(
        run_id="01DRYRUN000000000000000000",
        job=RECONCILE_JOB,
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=dt.datetime.fromisoformat(now.replace("Z", "+00:00")),
    )


def _statement(store: Store, day: str) -> BrokerStatement:
    raw = cast("dict[str, Any]", json.loads(store.get_bytes(trader_broker_statement_key(day))))  # pyright: ignore[reportExplicitAny]
    return statement_from_document(raw)


def dry_run(
    store: Store,
    *,
    month: str,
    now: str,
    fills: Mapping[str, Sequence[Fill]] | None = None,
    closes: CloseSource | None = None,
) -> dict[str, object]:
    """Run the month's loop against ``store``, which must be a capturing
    (dry-run) store, and return the summary. Raises only for a store that
    could write, or a month with nothing stored."""
    capture = capture_ledger_of(store)
    if capture is None:
        raise ValueError("the dry run needs a capturing store: open it with dry_run=True")
    trading_days = month_trading_days(month)
    stored = sorted(
        day
        for key in store.list_keys(RECONCILIATION_PREFIX)
        if key.endswith(".json")
        and (day := key.removeprefix(RECONCILIATION_PREFIX).removesuffix(".json")) in trading_days
    )
    if not stored:
        raise ValueError(f"no stored reconciliation for any trading day of {month}")
    fills = fills or {}
    unused_fills = sorted(set(fills) - set(stored))

    # 1 + 2: export and propose, one day at a time.
    records: list[dict[str, object]] = []
    export_failures: list[dict[str, str]] = []
    for day in stored:
        try:
            day_records = export_exceptions(store, since=day, until=day)
        except Exception as exc:  # noqa: BLE001 - a day that cannot export is a finding
            export_failures.append({"trading_day": day, "error": f"{exc}"[:400]})
            continue
        records.extend(day_records)
        if day_records:
            proposals = [propose(r, proposed_at=now) for r in day_records]
            store.put_bytes(
                exception_proposals_key(day),
                json.dumps(
                    proposal_day_document(day, proposals), indent=2, sort_keys=True
                ).encode(),
            )

    # 3: book every stored day into the ledger, oldest first.
    ledger: list[dict[str, object]] = []
    anchor: BrokerStatement | None = None
    for day in stored:
        ctx = _ctx(store, day, now)
        broker = _statement(store, day)
        reconciliation = cast(
            "dict[str, object]", json.loads(store.get_bytes(trader_reconciliation_key(day)))
        )
        posting = post_daily_ledger(
            ctx,
            broker=broker,
            anchor=anchor,
            fills=fills.get(day, ()),
            cash_flows=(),
            corporate_actions=_NO_ACTIONS,
            reconciliation=reconciliation,
            now=now,
        )
        ledger.extend(
            {"trading_day": day, "status": m["status"], "reason": m["status_reason"]}
            for m in cast("list[dict[str, object]]", ctx.metrics)
        )
        if posting is not None:
            anchor = broker

    # 4: close the month as the next month's first run would.
    run_day = _next_trading_day(trading_days[-1])
    ctx = _ctx(store, run_day, now)
    account = _statement(store, stored[-1]).account
    close = close_prior_month(
        ctx,
        day=run_day,
        account=account,
        now=now,
        closes=closes,
        after_close=lambda c: file_statement(ctx, c, now=now, closes=closes),
    )

    # 5: the queue the reviewer would see.
    queue = score(
        load_proposals(store, until=run_day),
        load_decisions(store),
        as_of=run_day,
        generated_at=now,
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "month": month,
        "generated_at": now,
        "close_run_day": run_day,
        "account": account,
        "days_with_reconciliation": stored,
        "days_without_reconciliation": [d for d in trading_days if d not in stored],
        "export_failures": export_failures,
        "fills_on_days_without_reconciliation": unused_fills,
        "records": len(records),
        "records_by_class": dict(sorted(Counter(str(r["class"]) for r in records).items())),
        "ledger": ledger,
        "close": None
        if close is None
        else {
            "status": close.get("status"),
            "exceptions": [
                {"class": e["class"], "instrument": e["instrument"], "detail": e["detail"]}
                for e in cast("list[Mapping[str, object]]", close.get("exceptions", []))
            ],
        },
        "close_metrics": [
            {"name": m["name"], "status": m["status"], "reason": m["status_reason"]}
            for m in cast("list[dict[str, object]]", ctx.metrics)
        ],
        "score_totals": queue["totals"],
        "keys_written": sorted(capture.keys),
    }


def write_out(store: Store, summary: Mapping[str, object], out: pathlib.Path) -> None:
    """Copy every captured object to ``out/store/<key>`` and write the summary
    and the score beside it."""
    tree = out / "store"
    for key in cast("list[str]", summary["keys_written"]):
        path = tree / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(store.get_bytes(key))
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Callable[[str | None], Store] = lambda uri: open_store(uri, dry_run=True),
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.back_office_dry_run",
        description="Export, propose, book, close and score one month. Writes only to --out.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="local directory")
    parser.add_argument("--fills", default=None, type=pathlib.Path, help="fills as JSON lines")
    parser.add_argument("--now", default=None, help="stamp for every record (default: now)")
    args = parser.parse_args(argv)
    out = cast("pathlib.Path", args.out)
    if out.exists() and any(out.iterdir()):
        parser.error(f"--out {out} is not empty; a dry run never mixes with an earlier one")
    store = opener(cast("str | None", args.store))
    summary = dry_run(
        store,
        month=cast("str", args.month),
        now=cast("str | None", args.now) or _now(),
        fills=load_fills(cast("pathlib.Path | None", args.fills)),
    )
    out.mkdir(parents=True, exist_ok=True)
    write_out(store, summary, out)
    close = cast("Mapping[str, object] | None", summary["close"])
    sys.stderr.write(
        f"{summary['month']}: {summary['records']} records, "
        f"{len(cast('list[object]', summary['export_failures']))} export failure(s), "
        f"close {close['status'] if close else 'not attempted'}; "
        f"queue {cast('Mapping[str, object]', summary['score_totals'])['queue']}; "
        f"wrote {out}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
