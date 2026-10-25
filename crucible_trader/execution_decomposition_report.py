"""Read-only report: the shortfall decomposition over stored v1 sessions (CR-06).

`execution_decomposition` (crucible-trader#52) decomposes one order's lifecycle
into delay, execution, fees and opportunity, but nothing ran it over a real
session. This module runs it over the `execution_shortfall.v1` documents the
order router already writes, one per trading day, and prints one report. It
writes nothing: the store is opened with :func:`crucible.store.read_only`, so a
write is structurally refused.

What v1 can and cannot supply is stated, never filled in:

* **Decision, side, quantity, fill price, filled quantity:** carried by v1, so
  the filled half (`filled_vs_decision_usd`) is real.
* **Arrival price, commission, terminal mark:** v1 records none of them, so
  every order with a fill names ``arrival_price`` and ``commission_usd`` in
  ``unavailable``, an unfilled remainder names ``terminal_mark``, and the order,
  session and multi-day totals are ``None``. That is the decomposition's own
  rule, and it is why an unknown never renders as a zero cost.
* **Executions:** v1 keeps one aggregate fill price per order, not broker
  executions, so each filled order becomes ONE piece whose id is
  ``<order_id>:v1-aggregate``. It is labelled as an aggregate, not a broker
  ``exec_id``.
* **Terminal state:** v1 does not record it. ``filled`` and ``no_fill`` follow
  from the quantities; a strict partial fill is reported as ``partial``.

A session with no orders, or one the router did not compute, is listed under
``sessions_without_orders`` with its v1 reason and kept out of the day-clustered
summary: it has no notional to weight, and counting it as a zero-cost day would
be the computed zero v1 itself refuses to write.

Measurement only. Nothing here touches routing, sizing, orders or the v1
document. The remaining CR-06 work (alpha-engine-config-I12566) is capturing
the arrival quote and per-execution commissions in `order_router`, and
versioning the harness contract to carry them. When that lands, this report's
``unavailable`` list shrinks with no change here.

Usage::

    python -m crucible_trader.execution_decomposition_report \\
        --store s3://<bucket>/crucible --start 2026-10-19 --end 2026-10-23
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Mapping, Sequence
from typing import cast

from crucible.execution import execution_shortfall_key  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs
from crucible.store import Store, open_store, read_only  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs

from crucible_trader.execution_decomposition import (
    ExecutionPiece,
    OrderOutcome,
    day_clustered_summary,
    session_decomposition,
)
from crucible_trader.execution_shortfall import SizingDecision

REPORT_SCHEMA = "execution_decomposition_report.v0"
V1_AGGREGATE_SUFFIX = ":v1-aggregate"

__all__ = [
    "REPORT_SCHEMA",
    "build_report",
    "main",
    "outcomes_from_v1",
]


def _parse_utc(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _terminal_state(filled: float, intended: float) -> str:
    if filled <= 0:
        return "no_fill"
    if filled >= intended:
        return "filled"
    return "partial"


def outcomes_from_v1(document: Mapping[str, object]) -> list[OrderOutcome]:
    """Each v1 order as an :class:`OrderOutcome`, with nothing invented.

    Arrival, commission and terminal mark are left ``None`` because v1 does not
    observe them; :func:`decompose` then names them as unavailable."""
    outcomes: list[OrderOutcome] = []
    for raw in cast("Sequence[Mapping[str, object]]", document.get("orders") or []):
        order_id = str(raw["order_id"])
        intended = float(cast("float", raw["quantity"]))
        filled = float(cast("float", raw.get("filled_quantity") or 0.0))
        fill_price = raw.get("fill_price")
        side = str(raw["side"])
        if side not in ("buy", "sell"):
            raise ValueError(f"order {order_id}: side {side!r} is not buy or sell")
        decision = SizingDecision(
            symbol=str(raw["symbol"]),
            side=side,
            quantity=intended,
            decision_price=float(cast("float", raw["decision_price"])),
            decided_at=_parse_utc(str(raw["decided_at_utc"])),
        )
        pieces: tuple[ExecutionPiece, ...] = ()
        if filled > 0 and fill_price is not None:
            pieces = (
                ExecutionPiece(
                    exec_id=order_id + V1_AGGREGATE_SUFFIX,
                    quantity=filled,
                    price=float(cast("float", fill_price)),
                ),
            )
        filled_seen = filled if pieces else 0.0
        outcomes.append(
            OrderOutcome(
                order_id=order_id,
                decision=decision,
                terminal_state=_terminal_state(filled_seen, intended),
                executions=pieces,
            )
        )
    return outcomes


def _days(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise ValueError(f"--end {end} is before --start {start}")
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def build_report(store: Store, start: dt.date, end: dt.date) -> dict[str, object]:
    """The decomposition over every stored v1 session in ``[start, end]``."""
    sessions: dict[str, dict[str, object]] = {}
    without_orders: list[dict[str, object]] = []
    missing: list[str] = []
    for day in _days(start, end):
        key = execution_shortfall_key(day.isoformat())
        if not store.exists(key):
            missing.append(day.isoformat())
            continue
        document = cast("dict[str, object]", json.loads(store.get_bytes(key)))
        if document.get("outcome") != "computed" or not document.get("orders"):
            without_orders.append(
                {
                    "trading_day": day.isoformat(),
                    "outcome": document.get("outcome"),
                    "outcome_reason": document.get("outcome_reason"),
                }
            )
            continue
        sessions[day.isoformat()] = session_decomposition(outcomes_from_v1(document))
    return {
        "schema": REPORT_SCHEMA,
        "source_schema": "execution_shortfall.v1",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "read_only": True,
        "sessions": sessions,
        "sessions_without_orders": without_orders,
        "days_without_document": missing,
        "summary": day_clustered_summary(sessions),
        "notes": [
            "v1 records no arrival price, commission or terminal mark; those "
            "components are unavailable and every total that needs them is null",
            "each filled order is one aggregate execution piece "
            f"('<order_id>{V1_AGGREGATE_SUFFIX}'), not a broker exec_id",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.execution_decomposition_report",
        description="Read-only shortfall decomposition over stored v1 sessions.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--end", required=True, type=dt.date.fromisoformat)
    args = parser.parse_args(argv)
    store = read_only(
        open_store(cast("str | None", args.store)),
        reason="execution_decomposition_report is read-only",
    )
    report = build_report(store, cast("dt.date", args.start), cast("dt.date", args.end))
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
