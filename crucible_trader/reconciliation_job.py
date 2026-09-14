"""The daily broker-reconciliation job: I/O around the pure reconciler.

`alpha-engine-config-I10651` deliverables 1, 3, 4 and 5; `-I10413` wiring.

One cycle, in order, inside `crucible.runner.run_job` so that whatever happens
the harness's single manifest writer files `runs/trader.reconcile/{day}/run.json`
(plan §4.2: manifest or it did not happen):

1. read today's statement from the broker (:class:`StatementSourceProtocol`);
2. read the anchor -- the latest stored broker statement BEFORE today -- and
   record it as an input, content-hashed;
3. reconcile (:func:`crucible_trader.reconciliation.reconcile`);
4. run the control arm over the same inputs
   (:func:`crucible_trader.reconciliation_control.run_control_arm`);
5. write today's statement (tomorrow's anchor) and the reconciliation result as
   outputs, and one MetricRecord per headline number;
6. raise :class:`ReconciliationVoidError` when a plant was missed, else
   :class:`ReconciliationDiscrepancyError` when any finding exists. Either makes
   the manifest `status: failed`, which is the page condition (§4.6), and the
   reason names every finding's instrument, delta and side.

**Registered in the harness contract** (crucible-PR297): `trader.reconcile` is
in `crucible.models.JOB_VALUES`, both keys are owned by `crucible.keys`
(`trader_reconciliation_key`, `trader_broker_statement_key`), and the
reconciliation key is on `crucible.manifest.MONEY_PATH_PREDICATES`, so the
harness's single manifest writer attaches a `money_path_link` from this run's
OUTPUTS (`alpha-engine-config-I10414`). Nothing here declares a key shape.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence

from crucible.calendar import is_trading_day
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.runner import RunContext, run_job
from crucible.store import Store

from crucible_trader.broker_statement import (
    BROKER_STATEMENT_SCHEMA_VERSION,
    BrokerStatement,
    StatementSourceProtocol,
    statement_from_document,
)
from crucible_trader.reconciliation import (
    RECONCILIATION_SCHEMA_VERSION,
    CashFlow,
    CorporateActionSet,
    Fill,
    ReconciliationInputs,
    ReconciliationResult,
    reconcile,
)
from crucible_trader.reconciliation_control import (
    ControlVerdict,
    Reconciler,
    ReconciliationVoidError,
    run_control_arm,
)

RECONCILE_JOB = "trader.reconcile"
METRIC_MODULE = "crucible_trader.reconciliation"

#: The listing prefix of every stored statement, derived from the harness's key
#: function (any valid day) so the key shape has one owner.
BROKER_STATEMENT_PREFIX = trader_broker_statement_key("2000-01-03").rsplit("/", 1)[0] + "/"


class ReconciliationDiscrepancyError(RuntimeError):
    """The broker disagrees with the expected book. A page, by manifest status."""


def previous_trading_day(day: dt.date) -> dt.date:
    """The session before ``day`` on the NYSE calendar the harness uses."""
    candidate = day - dt.timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= dt.timedelta(days=1)
    return candidate


def latest_anchor(store: Store, trading_day: str) -> tuple[str, bytes, BrokerStatement] | None:
    """The newest stored statement strictly before ``trading_day``, or ``None``.

    ``None`` is a real absence (no statement was ever filed), and the reconciler
    turns it into a `no_anchor` finding unless genesis was declared. A statement
    that exists and does not parse raises: it is never skipped in favour of an
    older one, which would silently widen the window being reconciled.
    """
    days = sorted(
        key.removeprefix(BROKER_STATEMENT_PREFIX).removesuffix(".json")
        for key in store.list_keys(BROKER_STATEMENT_PREFIX)
        if key.endswith(".json")
    )
    earlier = [d for d in days if d < trading_day]
    if not earlier:
        return None
    key = trader_broker_statement_key(earlier[-1])
    payload = store.get_bytes(key)
    return key, payload, statement_from_document(json.loads(payload.decode("utf-8")))


def reconcile_cycle(
    ctx: RunContext,
    *,
    source: StatementSourceProtocol,
    fills: Sequence[Fill],
    cash_flows: Sequence[CashFlow],
    corporate_actions: CorporateActionSet,
    genesis: bool = False,
    reconciler: Reconciler = reconcile,
) -> ReconciliationResult:
    """The job body. See the module docstring for the order and the raises."""
    day = ctx.trading_day.isoformat()
    broker = source.read_statement(day)
    anchor_entry = latest_anchor(ctx.store, day)
    anchor = None
    if anchor_entry is not None:
        anchor_key, anchor_payload, anchor = anchor_entry
        ctx.record_input(anchor_key, anchor_payload, BROKER_STATEMENT_SCHEMA_VERSION)

    inputs = ReconciliationInputs(
        trading_day=day,
        broker=broker,
        anchor=anchor,
        fills=tuple(fills),
        cash_flows=tuple(cash_flows),
        corporate_actions=corporate_actions,
        previous_trading_day=previous_trading_day(ctx.trading_day).isoformat(),
        genesis=genesis,
    )
    result = reconciler(inputs)
    verdict = run_control_arm(inputs, reconciler=reconciler)

    ctx.record_output(
        trader_broker_statement_key(day),
        json.dumps(broker.to_document(), indent=2, sort_keys=True).encode("utf-8"),
        BROKER_STATEMENT_SCHEMA_VERSION,
    )
    document = {
        **result.to_document(),
        "control_arm": verdict.to_document(),
        "void": not verdict.passed,
    }
    ctx.record_output(
        trader_reconciliation_key(day),
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        RECONCILIATION_SCHEMA_VERSION,
    )
    ctx.record_rows(
        rows_in=len(broker.positions) + len(inputs.fills), rows_out=len(result.discrepancies)
    )
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for metric in metric_records(
        result, verdict, now=now, source_path=trader_reconciliation_key(day)
    ):
        ctx.record_metric(metric)

    if not verdict.passed:
        raise ReconciliationVoidError(verdict.reason())
    if not result.clean:
        raise ReconciliationDiscrepancyError(page_text(result))
    return result


def page_text(result: ReconciliationResult) -> str:
    """Every finding, headline first, each naming instrument, delta and side."""
    return (
        f"broker reconciliation {result.trading_day} ({result.account}): "
        f"{len(result.discrepancies)} discrepancies. "
        + " | ".join(d.page_line() for d in result.discrepancies)
    )


def metric_records(
    result: ReconciliationResult, verdict: ControlVerdict, *, now: str, source_path: str
) -> list[dict]:
    """MetricRecord rows (`crucible.models.MetricRecordRow`). Status comes from
    the FINDINGS: a 1.000 rate beside any finding is FAIL, and no comparable
    rows is `N/A-LOW-N`, never OK."""
    status = "OK" if result.clean else "FAIL"
    headline = result.discrepancies[0].page_line() if result.discrepancies else None

    def row(
        name: str, value: float | None, unit: str, n_floor: int, row_status: str, reason: str
    ) -> dict:
        return {
            "name": name,
            "module": METRIC_MODULE,
            "metric_type": "operational",
            "value": value,
            "unit": unit if value is not None else None,
            "n_floor": n_floor,
            "status": row_status,
            "status_reason": reason,
            "source_path": source_path,
            "last_updated_utc": now,
        }

    rate = result.match_rate
    if rate is None:
        rate_status = "N/A-LOW-N"
        rate_reason = (
            f"zero comparable positions on {result.trading_day}, so there is no match rate "
            "to report; an empty comparison is not a pass"
            + (f" (and {headline})" if headline else "")
        )
    else:
        rate_status = status
        rate_reason = f"{result.n_matched}/{result.n_comparable} positions matched" + (
            f"; failed on {headline}" if headline else " and no other finding"
        )
    count_delta = (
        None
        if result.expected_cash_usd is None
        else float(result.n_positions_broker - result.n_positions_expected)
    )
    return [
        row("broker_reconciliation_match_rate", rate, "ratio", 1, rate_status, rate_reason),
        row(
            "broker_reconciliation_discrepancies",
            float(len(result.discrepancies)),
            "count",
            0,
            status,
            headline or "no discrepancy between the expected book and the broker",
        ),
        row(
            "broker_reconciliation_position_count_delta",
            count_delta,
            "count",
            1,
            status if count_delta is not None else "N/A-MISSING-INPUT",
            f"broker {result.n_positions_broker} vs expected {result.n_positions_expected}"
            if count_delta is not None
            else "no anchor statement, so there is no expected position count",
        ),
        row(
            "broker_reconciliation_cash_residual_usd",
            result.cash_residual_usd,
            "usd",
            1,
            status if result.cash_residual_usd is not None else "N/A-MISSING-INPUT",
            f"broker cash minus expected cash on {result.trading_day}"
            if result.cash_residual_usd is not None
            else "no anchor statement, so there is no expected cash",
        ),
        row(
            "broker_reconciliation_control_detected",
            verdict.n_detected / len(verdict.outcomes),
            "ratio",
            len(verdict.outcomes),
            "OK" if verdict.passed else "FAIL",
            "every planted discrepancy detected and classified"
            if verdict.passed
            else verdict.reason(),
        ),
    ]


def run_reconciliation(
    store: Store,
    source: StatementSourceProtocol,
    *,
    trading_day: dt.date,
    fills: Sequence[Fill],
    cash_flows: Sequence[CashFlow],
    corporate_actions: CorporateActionSet,
    genesis: bool = False,
    run_mode: str | None = None,
    body: Callable[..., ReconciliationResult] = reconcile_cycle,
) -> RunContext:
    """Run one cycle as job :data:`RECONCILE_JOB` under the harness runner."""
    return run_job(
        RECONCILE_JOB,
        lambda ctx: body(
            ctx,
            source=source,
            fills=fills,
            cash_flows=cash_flows,
            corporate_actions=corporate_actions,
            genesis=genesis,
        ),
        store=store,
        trading_day=trading_day,
        run_mode=run_mode,
    )
