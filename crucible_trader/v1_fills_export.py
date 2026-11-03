"""v1's fills for one month, as the JSON lines the back office's dry run reads.

`alpha-engine-config-I13124` (back office stage 2, the October dry run). v1
traded the same IB paper account through October and v2 stored no fills before
the cutover, so the dry run's ledger booked every v1 trade as a gap to suspense
and the close refused on them. v1 kept its executions in its own SQLite trade
log, backed up nightly to ``s3://alpha-engine-research/trades/trades_latest.db``
(and ``trades_<date>.db``). This module reads that file, never writes it, and
writes one :class:`crucible_trader.reconciliation.Fill` per line in the shape
:func:`crucible_trader.back_office_dry_run.load_fills` reads:

    aws s3 cp s3://alpha-engine-research/trades/trades_latest.db DB
    python -m crucible_trader.v1_fills_export --db DB --month 2026-10 --out FILLS.jsonl
    python -m crucible_trader.back_office_dry_run ... --fills FILLS.jsonl

The mapping, each line of it measured against October's stored broker
statements:

- ``trading_day`` is the row's ``date``, the SESSION the order filled in. Not
  its ``trading_day`` column, which is v1's knowledge axis (the last closed
  session when it decided) and a day early.
- ``shares`` is ``filled_shares``, signed by ``action`` (``ENTER`` buys;
  ``REDUCE`` and ``EXIT`` sell). ``shares`` is what v1 intended: eight October
  rows differ, and only ``filled_shares`` ties to the broker.
- ``cash_usd`` is the signed consideration at ``fill_price`` net of
  ``commission_usd``, as :class:`Fill` defines it.
- ``exec_id`` is ``v1:<trade_id>``: v1 kept IB's order id, not the execution
  id, and the ledger addresses each fill by this field.

A row v1 recorded as filled but cannot be read as a fill (an unknown
``action``, no ``fill_price``) stops the export naming the row, rather than
being dropped: a missing fill is a suspense line the close would then blame on
the broker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sqlite3
import sys
from collections.abc import Sequence
from decimal import Decimal

__all__ = ["V1_DB_KEY", "export_fills", "main", "read_fills"]

#: Where v1's executor backs its trade log up each evening.
V1_DB_KEY = "s3://alpha-engine-research/trades/trades_latest.db"

_SIGN = {"ENTER": 1, "REDUCE": -1, "EXIT": -1}

_QUERY = """
SELECT trade_id, date, ticker, action, filled_shares, fill_price, commission_usd
FROM trades
WHERE date >= ? AND date < ? AND status = 'Filled'
ORDER BY date, fill_time, trade_id
"""


def _month_bounds(month: str) -> tuple[str, str]:
    first = dt.date.fromisoformat(f"{month}-01")
    after = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return first.isoformat(), after.isoformat()


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001")))


def read_fills(db: pathlib.Path, month: str) -> list[dict[str, object]]:
    """Every fill v1 recorded in ``month`` (``YYYY-MM``), oldest first."""
    first, after = _month_bounds(month)
    connection = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(_QUERY, (first, after)).fetchall()
    finally:
        connection.close()
    fills: list[dict[str, object]] = []
    for trade_id, day, ticker, action, filled, price, commission in rows:
        if action not in _SIGN or price is None or not filled:
            raise ValueError(
                f"v1 trade {trade_id} ({day} {ticker} {action}) is filled but not a fill: "
                f"filled_shares={filled!r}, fill_price={price!r}"
            )
        shares = _SIGN[action] * int(filled)
        fee = Decimal(repr(float(commission or 0.0)))
        fills.append(
            {
                "trading_day": dt.date.fromisoformat(str(day)).isoformat(),
                "ticker": str(ticker),
                "shares": shares,
                "cash_usd": _money(-shares * Decimal(repr(float(price))) - fee),
                "commission_usd": None if commission is None else _money(fee),
                "exec_id": f"v1:{trade_id}",
            }
        )
    return fills


def export_fills(db: pathlib.Path, month: str, out: pathlib.Path) -> int:
    """Write ``month``'s fills to ``out`` as JSON lines; return how many."""
    fills = read_fills(db, month)
    out.write_text("".join(json.dumps(f, sort_keys=True) + "\n" for f in fills), encoding="utf-8")
    return len(fills)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.v1_fills_export",
        description=f"v1's fills for one month as --fills JSON lines (v1 log: {V1_DB_KEY}).",
    )
    parser.add_argument("--db", required=True, type=pathlib.Path, help="v1 trades.db, local")
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="JSON lines to write")
    args = parser.parse_args(argv)
    db, out, month = args.db, args.out, str(args.month)
    if not db.is_file():
        parser.error(f"--db {db} is not a file")
    if out.exists():
        parser.error(f"--out {out} exists; an export never overwrites one")
    count = export_fills(db, month, out)
    sys.stderr.write(f"{month}: {count} v1 fill(s) from {db}; wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
