"""The monthly performance statement and simulated fee bill: back office 1f.

`alpha-engine-config-I13123` (epic `-I12892`; nous-ergon-ops
`plans/2026-11-02-back-office-implementation.md` stage 1f). Produced from a
FINISHED month-end close (:mod:`crucible_trader.month_end_close`, 1e) and
nothing else but two benchmark closes, so the statement and the bill rebuild
from stored inputs (plan, Control standard: reproducibility).

**Return.** Time-weighted, gross of the simulated fees and net of them. The
paper fund has no external flow after its opening (the ledger's chart of
accounts has none, and the close records ``external_flows_cents: 0``), so the
month's time-weighted return is exactly ``NAV_end / NAV_start - 1``: daily
valuation and geometric linking give the same number when no flow splits the
month. A close that ever carries a flow is REFUSED here rather than reported
with the wrong formula; daily linking lands with the first flow.

**Benchmarks.** SPY and QQQ, each its settled-close PRICE return over the same
two days (the close's start and last trading day, from the day's price panel,
or, for a benchmark the panel does not carry, from the trader's own capture of
the Polygon close: :mod:`crucible_trader.benchmark_closes`).
The fund's return includes its dividends, so the comparison is total return
against price return; the statement says so in ``benchmark_basis``. A benchmark
the panel does not carry is ``null`` with a reason, never estimated.

**Simulated fee bill (shadow only).** Never booked to the ledger and never sent
to anyone. The fee schedule is :data:`SIMULATED_FEE_SCHEDULE`; the bill runs on
a shadow NET NAV chain carried statement to statement:

* adviser fee = the closed month's AUM (net NAV before fees) x annual rate / 12,
  prorated by calendar days when the book opened inside the month after its
  first trading day (from the opening day through month end, inclusive): a
  bill covers the period the statement measures, never a whole month the book
  did not hold
  (``partial_period: calendar_days``; found by the October dry run,
  alpha-engine-config-I13125, where the v2 book opens on 2026-10-09);
* performance fee = rate x the net NAV after the adviser fee ABOVE the
  high-water mark, crystallised monthly; the mark only rises.

The high-water mark lives in the statement chain, not in the ledger: the bill
is a shadow and the ledger is the book of record, which must not hold a fee
nobody owes.

**Failure isolation.** :func:`file_statement` rides step 5f of
`trader.reconcile` and NEVER raises; it reports through one MetricRecord
(:data:`STATEMENT_METRIC`).
"""

from __future__ import annotations

import calendar
import json
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, cast

from crucible.store import ETAG_ABSENT, PointerConflictError

from crucible_trader.benchmark_closes import with_benchmark_closes
from crucible_trader.month_end_close import (
    CloseSource,
    close_key,
    panel_close_source,
    previous_month,
)

STATEMENT_SCHEMA_VERSION = "performance_statement.v1"
LEDGER_STATEMENTS_PREFIX = "trader/ledger/statements/"
STATEMENT_METRIC = "performance_statement"
BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ")

#: Every fee schedule a bill may name, by ``schedule_id``. Placeholder rates
#: until the fund's terms are ruled. A schedule is never edited: a change is a
#: new id here, so a bill always names the terms it was computed under, and the
#: monthly rebuild (:mod:`crucible_trader.month_rebuild`) recomputes a bill
#: under the schedule it names rather than under today's default. A schedule
#: without ``partial_period`` bills a whole month whenever the book opened.
FEE_SCHEDULES: Mapping[str, Mapping[str, object]] = {
    "simulated.2026-11-03": {
        "schedule_id": "simulated.2026-11-03",
        "adviser_fee_annual_bps": 100,
        "performance_fee_bps": 2000,
        "high_water_mark": True,
        "crystallisation": "monthly",
        "hurdle": None,
    },
    "simulated.2026-11-04": {
        "schedule_id": "simulated.2026-11-04",
        "adviser_fee_annual_bps": 100,
        "performance_fee_bps": 2000,
        "high_water_mark": True,
        "crystallisation": "monthly",
        "hurdle": None,
        "partial_period": "calendar_days",
    },
}

#: The schedule a new bill is computed under.
SIMULATED_FEE_SCHEDULE: Mapping[str, object] = FEE_SCHEDULES["simulated.2026-11-04"]


class StatementError(ValueError):
    """A statement that cannot be produced from what it was given."""


def statement_key(tenant: str, month: str) -> str:
    close_key(tenant, month)  # validates both
    return f"{LEDGER_STATEMENTS_PREFIX}{tenant}/{month}.json"


def _cents(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _ratio(value: Decimal) -> float:
    return float(value.quantize(Decimal("1e-10"), rounding=ROUND_HALF_UP))


def benchmark_returns(
    closes: CloseSource, start_day: str, end_day: str
) -> dict[str, dict[str, object]]:
    start = closes(start_day, BENCHMARKS)
    end = closes(end_day, BENCHMARKS)
    out: dict[str, dict[str, object]] = {}
    for symbol in BENCHMARKS:
        a, b = start.get(symbol), end.get(symbol)
        if a is None or b is None or a <= 0:
            out[symbol] = {
                "return": None,
                "reason": f"no settled close for {symbol} on "
                + " and ".join(d for d, c in ((start_day, a), (end_day, b)) if c is None or c <= 0),
            }
            continue
        out[symbol] = {
            "start_close_usd": a,
            "end_close_usd": b,
            "return": _ratio(Decimal(repr(b)) / Decimal(repr(a)) - 1),
        }
    return out


def build_statement(
    close: Mapping[str, object],
    prior: Mapping[str, object] | None,
    benchmarks: Mapping[str, Mapping[str, object]],
    schedule: Mapping[str, object] = SIMULATED_FEE_SCHEDULE,
) -> dict[str, object]:
    """The statement and bill for ``close``'s month. Pure. Raises
    :class:`StatementError` on an unfinished close, a flow, or a broken chain."""
    if close.get("status") != "closed":
        raise StatementError("a statement is produced from a finished close only")
    month, tenant = str(close["month"]), str(close["tenant"])
    if close.get("external_flows_cents") != 0:
        raise StatementError(f"{month} carries an external flow; daily linking is required")
    nav_start, nav_end = (
        int(cast("int", close["nav_start_cents"])),
        int(cast("int", close["nav_end_cents"])),
    )
    if nav_start <= 0:
        raise StatementError(f"{month} starts from NAV {nav_start} cents; no return is defined")
    if prior is None:
        if not close.get("opened_in_month"):
            raise StatementError(f"{previous_month(month)}'s statement is missing")
        net_start = hwm_start = nav_start
    else:
        if prior.get("month") != previous_month(month) or prior.get("tenant") != tenant:
            raise StatementError(f"the prior statement is not {previous_month(month)}'s")
        net_start = int(cast("int", prior["shadow_net_nav_end_cents"]))
        hwm_start = int(cast("int", prior["high_water_mark_cents"]))

    gross = Decimal(nav_end - nav_start) / Decimal(nav_start)
    aum = _cents(Decimal(net_start) * (1 + gross))
    days_in_month = calendar.monthrange(int(month[:4]), int(month[5:7]))[1]
    billed_days = days_in_month
    if (
        schedule.get("partial_period") == "calendar_days"
        and close.get("opened_in_month")
        and close.get("start_trading_day") != close.get("first_trading_day")
        # An opening valued at the prior month's last statement is open from
        # this month's first day: the whole month is billed (aec#13237).
        and str(close.get("start_trading_day"))[:7] == month
    ):
        billed_days = days_in_month - int(str(close["start_trading_day"])[8:10]) + 1
    adviser = _cents(
        Decimal(aum)
        * Decimal(cast("int", schedule["adviser_fee_annual_bps"]))
        / 10_000
        / 12
        * billed_days
        / days_in_month
    )
    after_adviser = aum - adviser
    performance = _cents(
        Decimal(max(0, after_adviser - hwm_start))
        * Decimal(cast("int", schedule["performance_fee_bps"]))
        / 10_000
    )
    net_end = after_adviser - performance
    hwm_end = max(hwm_start, net_end)
    net = Decimal(net_end - net_start) / Decimal(net_start)
    excess = {
        s: None if b.get("return") is None else _ratio(gross - Decimal(repr(b["return"])))
        for s, b in benchmarks.items()
    }
    return {
        "schema_version": STATEMENT_SCHEMA_VERSION,
        "tenant": tenant,
        "account": close["account"],
        "month": month,
        "start_trading_day": close["start_trading_day"],
        "last_trading_day": close["last_trading_day"],
        "shadow": True,
        "nav_start_cents": nav_start,
        "nav_end_cents": nav_end,
        "time_weighted_return_gross": _ratio(gross),
        "time_weighted_return_net": _ratio(net),
        "method": "nav ratio; exact time-weighted return with no external flow in the month",
        "pnl": close["pnl"],
        "benchmarks": dict(benchmarks),
        "benchmark_basis": "fund total return (dividends included) vs benchmark price return",
        "excess_return_gross": excess,
        "simulated_fee_bill": {
            "booked": False,
            "schedule": dict(schedule),
            "aum_cents": aum,
            "adviser_fee_cents": adviser,
            "performance_fee_cents": performance,
            "total_cents": adviser + performance,
            "high_water_mark_start_cents": hwm_start,
            **(
                {"billed_days": billed_days, "days_in_month": days_in_month}
                if "partial_period" in schedule
                else {}
            ),
        },
        "shadow_net_nav_start_cents": net_start,
        "shadow_net_nav_end_cents": net_end,
        "high_water_mark_cents": hwm_end,
        "inputs": {
            "close": close_key(tenant, month),
            "prior_statement": statement_key(tenant, previous_month(month)) if prior else None,
        },
    }


def file_statement(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    close: Mapping[str, object],
    *,
    now: str,
    closes: CloseSource | None = None,
) -> dict[str, object] | None:
    """Produce and file the statement for a finished ``close`` unless it is
    already filed. NEVER raises; returns the statement, or ``None``."""
    key = None
    try:
        tenant, month = str(close["tenant"]), str(close["month"])
        key = statement_key(tenant, month)
        store = ctx.store
        if store.exists(key):
            return None
        prior_key = statement_key(tenant, previous_month(month))
        prior = (
            cast("dict[str, Any]", json.loads(store.get_bytes(prior_key).decode("utf-8")))  # pyright: ignore[reportExplicitAny]
            if store.exists(prior_key)
            else None
        )
        source = with_benchmark_closes(store, closes or panel_close_source(store))
        statement = build_statement(
            close,
            prior,
            benchmark_returns(
                source, str(close["start_trading_day"]), str(close["last_trading_day"])
            ),
        )
        payload = json.dumps(statement, indent=2, sort_keys=True).encode("utf-8")
        try:
            ctx.record_output_cas(key, ETAG_ABSENT, payload, STATEMENT_SCHEMA_VERSION)
        except PointerConflictError:
            return None  # a concurrent run filed it first; a statement is never rewritten
    except Exception as exc:  # noqa: BLE001 - must never change what the job decides or pages
        _metric(ctx, now, key, "FAIL", None, f"statement NOT filed: {type(exc).__name__}: {exc}")
        return None
    bill = cast("Mapping[str, object]", statement["simulated_fee_bill"])
    # A benchmark with no settled close is stated on the statement's face
    # (``return: None`` and a reason) and must not read OK on the metric: an
    # unbenchmarked statement is one a reader would otherwise take as complete
    # (alpha-engine-config-I13125, found by a planted break).
    benchmarks = cast("Mapping[str, Mapping[str, object]]", statement["benchmarks"])
    missing = [f"{s}: {b['reason']}" for s, b in sorted(benchmarks.items()) if b["return"] is None]
    _metric(
        ctx,
        now,
        key,
        "WATCH" if missing else "OK",
        float(cast("float", statement["time_weighted_return_gross"])),
        f"{month} statement filed: gross {statement['time_weighted_return_gross']}, net "
        f"{statement['time_weighted_return_net']}, simulated bill {bill['total_cents']} cents"
        + (f"; benchmark missing: {'; '.join(missing)}" if missing else ""),
    )
    return statement


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
                "name": STATEMENT_METRIC,
                "module": "crucible_trader.performance_statement",
                "metric_type": "operational",
                "value": value,
                "unit": "ratio" if value is not None else None,
                "n_floor": 0,
                "status": status,
                "status_reason": reason[:400],
                "source_path": key or LEDGER_STATEMENTS_PREFIX,
                "last_updated_utc": now,
            }
        )
    except Exception:  # noqa: BLE001, S110 - a metric must not raise into the trader either
        pass


__all__ = [
    "BENCHMARKS",
    "LEDGER_STATEMENTS_PREFIX",
    "SIMULATED_FEE_SCHEDULE",
    "STATEMENT_METRIC",
    "STATEMENT_SCHEMA_VERSION",
    "StatementError",
    "benchmark_returns",
    "build_statement",
    "file_statement",
    "statement_key",
]
