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
5a. when the caller hands it the broker's order records, reconcile the day's
   durable order intents against them
   (:mod:`crucible_trader.order_intent_reconciliation`) and write that document
   as an output with its own MetricRecord (see :func:`reconcile_intents_step`);
5b. file every break found by 3, 4 and 5a as an `exception.v1` record
   (:mod:`crucible_trader.exception_record`, `alpha-engine-config-I12892`
   phase 0) -- best effort, and never a change to what step 6 decides or pages
   (see :func:`file_exception_records`);
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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from crucible.calendar import is_trading_day
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.runner import RunContext, run_job
from crucible.store import Store

from crucible_trader.broker_statement import (
    BROKER_STATEMENT_SCHEMA_VERSION,
    BrokerStatement,
    StatementSourceProtocol,
    statement_from_document,
    statement_schema_version,
)
from crucible_trader.dividend_entitlements import (
    OWED_BOOTSTRAP_FROM,
    load_declared_entitlements,
    owed_dividends,
)
from crucible_trader.exception_record import (
    EXCEPTION_SCHEMA_VERSION,
    exception_day_document,
    exceptions_key,
    records_from_broker_reconciliation,
    records_from_order_intent_reconciliation,
)
from crucible_trader.order_intent_reconciliation import (
    RECONCILIATION_SCHEMA_VERSION as INTENT_RECONCILIATION_SCHEMA_VERSION,
)
from crucible_trader.order_intent_reconciliation import (
    BrokerExecution,
    BrokerOrder,
    intents_for_day,
    order_intent_reconciliation_key,
    reconcile_order_intents,
)
from crucible_trader.order_router import ORDER_REF_PREFIX
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

#: The listing prefix of every stored reconciliation result, derived the same way.
RECONCILIATION_PREFIX = trader_reconciliation_key("2000-01-03").rsplit("/", 1)[0] + "/"


#: A corporate-action source that needs to know what it is resolving: called
#: with the run context, the trading day and the instruments the reconcile will
#: look at (anchor, broker and fills), after the statements are read.
CorporateActionProvider = Callable[[RunContext, str, frozenset[str]], CorporateActionSet]


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


def stored_results(store: Store, trading_day: str) -> dict[str, tuple[str, bytes]]:
    """Stored reconciliation results from :data:`OWED_BOOTSTRAP_FROM` up to
    (not including) ``trading_day``, as ``day -> (key, payload)``. A result that
    exists and does not read raises; it is never skipped."""
    out: dict[str, tuple[str, bytes]] = {}
    for key in store.list_keys(RECONCILIATION_PREFIX):
        if not key.endswith(".json"):
            continue
        day = key.removeprefix(RECONCILIATION_PREFIX).removesuffix(".json")
        if OWED_BOOTSTRAP_FROM <= day < trading_day:
            out[day] = (key, store.get_bytes(key))
    return out


def reconcile_cycle(
    ctx: RunContext,
    *,
    source: StatementSourceProtocol,
    fills: Sequence[Fill],
    cash_flows: Sequence[CashFlow],
    corporate_actions: CorporateActionSet | CorporateActionProvider,
    genesis: bool = False,
    reconciler: Reconciler = reconcile,
    order_records: tuple[Sequence[BrokerOrder], Sequence[BrokerExecution]] | None = None,
) -> ReconciliationResult:
    """The job body. See the module docstring for the order and the raises.

    ``corporate_actions`` is either the day's set, or a
    :data:`CorporateActionProvider` asked for it once the statements are read.
    ``order_records`` is the broker's orders and executions
    (:func:`~crucible_trader.order_intent_reconciliation.broker_records_from_client`);
    when given, step 5a runs.
    """
    day = ctx.trading_day.isoformat()
    broker = source.read_statement(day)
    anchor_entry = latest_anchor(ctx.store, day)
    anchor = None
    if anchor_entry is not None:
        anchor_key, anchor_payload, anchor = anchor_entry
        ctx.record_input(anchor_key, anchor_payload, statement_schema_version(anchor_payload))
    if not isinstance(corporate_actions, CorporateActionSet):
        universe = (
            frozenset(broker.positions)
            | frozenset(anchor.positions if anchor is not None else ())
            | frozenset(f.ticker for f in fills)
        )
        corporate_actions = corporate_actions(ctx, day, universe)

    documents: dict[str, dict[str, object]] = {}
    for prior_day, (key, payload) in stored_results(ctx.store, day).items():
        ctx.record_input(key, payload, RECONCILIATION_SCHEMA_VERSION)
        documents[prior_day] = json.loads(payload.decode("utf-8"))
    owed = owed_dividends(documents, today=day, declared=load_declared_entitlements())

    inputs = ReconciliationInputs(
        trading_day=day,
        broker=broker,
        anchor=anchor,
        fills=tuple(fills),
        cash_flows=tuple(cash_flows),
        corporate_actions=corporate_actions,
        previous_trading_day=previous_trading_day(ctx.trading_day).isoformat(),
        genesis=genesis,
        owed_dividends=owed,
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
    intent_document: dict[str, Any] | None = None  # pyright: ignore[reportExplicitAny]  # raw JSON document
    try:
        if order_records is not None:
            orders, executions = order_records
            intent_document = reconcile_intents_step(ctx, orders=orders, executions=executions)
    finally:
        # In a `finally` so a step-5a raise still files the broker breaks, and
        # `file_exception_records` never raises, so the raise (or step 6's
        # page) is exactly what it would have been without it.
        file_exception_records(ctx, document, intent_document, now=now)

    if not verdict.passed:
        raise ReconciliationVoidError(verdict.reason())
    if not result.clean:
        raise ReconciliationDiscrepancyError(page_text(result))
    return result


def reconcile_intents_step(
    ctx: RunContext,
    *,
    orders: Sequence[BrokerOrder],
    executions: Sequence[BrokerExecution],
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    """Step 5a: the day's durable order intents against the broker. Read-only.

    `alpha-engine-config-I12558` (audit FIN-03): the intent reconciler had no
    production caller, so an intent whose submit was lost was discoverable by
    nobody. This step reads every intent for the run's trading day, classifies
    it against the broker's own records, and files the
    ``order_intent_reconciliation`` document at
    :func:`~crucible_trader.order_intent_reconciliation.order_intent_reconciliation_key`
    plus one MetricRecord, ``order_intent_reconciliation_breaks``, whose status
    is FAIL while any intent is in a break class. Every row carries its class
    and age.

    It places, cancels and re-sends nothing: what to do about an ``unknown``
    intent stays a human decision. It does not change this job's manifest
    status either -- a break is reported in the document and the metric, and
    whether it should page (fail the run, which the cutover gate reads) is a
    separate ruling. An intent that cannot be read DOES raise, because a
    reconciliation that skipped it would report clean over it.
    """
    day = ctx.trading_day.isoformat()
    now = dt.datetime.now(dt.UTC)
    document = reconcile_order_intents(
        trading_day=day,
        intents=intents_for_day(ctx.store, day),
        orders=orders,
        executions=executions,
        key_prefix=ORDER_REF_PREFIX,
        now=now,
    )
    key = order_intent_reconciliation_key(day)
    ctx.record_output(
        key,
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        INTENT_RECONCILIATION_SCHEMA_VERSION,
    )
    breaks = document["breaks"]
    by_key = {row["order_key"]: row for row in document["rows"]}
    ctx.record_metric(
        {
            "name": "order_intent_reconciliation_breaks",
            "module": "crucible_trader.order_intent_reconciliation",
            "metric_type": "operational",
            "value": float(len(breaks)),
            "unit": "count",
            "n_floor": 0,
            "status": "OK" if document["clean"] else "FAIL",
            "status_reason": (
                f"{document['intents']} intents on {day}, every one terminal or working "
                "at the broker and none overfilled, mismatched or corrupt"
                if document["clean"]
                else f"{len(breaks)} order-intent breaks on {day}: "
                + " | ".join(f"{k} {by_key[k]['class']}" for k in breaks)
            ),
            "source_path": key,
            "last_updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return document


#: The MetricRecord that says whether step 5b filed the day's breaks.
EXCEPTION_RECORDS_METRIC = "exception_records_filed"


def file_exception_records(
    ctx: RunContext,
    broker_document: Mapping[str, object],
    intent_document: Mapping[str, object] | None,
    *,
    now: str,
) -> None:
    """Step 5b: file the day's breaks as `exception.v1` records. Never raises.

    `alpha-engine-config-I12892` phase 0: a break was a page and nothing else,
    so no later proposal or decision had an item to point at. This step builds
    one record per break from the documents steps 3-5a already wrote (the
    broker findings, a void control arm, and every order-intent row in a break
    class) and files them together at
    :func:`~crucible_trader.exception_record.exceptions_key`, plus one
    ``exception_records_filed`` MetricRecord. A day with no break files nothing
    and adds no metric, so a clean run is byte-for-byte what it was.

    It reads the detectors' documents and decides nothing. A failure to build or
    write the records is CAUGHT and reported as that metric with status FAIL --
    because the alternative is worse: a raise here would replace step 6's page
    (whose reason names every finding) with a storage error, or fail a run the
    detectors passed. The page is the control; this record rides beside it.
    """
    day = ctx.trading_day.isoformat()
    key = exceptions_key(day)
    count = 0
    try:
        records = records_from_broker_reconciliation(broker_document, detected_at=now)
        if intent_document is not None:
            records += records_from_order_intent_reconciliation(
                intent_document, void=broker_document.get("void") is True
            )
        count = len(records)
        if records:
            ctx.record_output(
                key,
                json.dumps(exception_day_document(day, records), indent=2, sort_keys=True).encode(
                    "utf-8"
                ),
                EXCEPTION_SCHEMA_VERSION,
            )
    except Exception as exc:  # noqa: BLE001 - see docstring: must never mask the page
        status, value = "FAIL", None
        reason = (
            f"exception records for {day} were NOT filed ({count} breaks): "
            + (f"{type(exc).__name__}: {exc}")[:400]
        )
    else:
        if not records:
            return
        status, value = "OK", float(count)
        reason = f"{count} exception records filed for {day}"
    ctx.record_metric(
        {
            "name": EXCEPTION_RECORDS_METRIC,
            "module": "crucible_trader.exception_record",
            "metric_type": "operational",
            "value": value,
            "unit": "count" if value is not None else None,
            "n_floor": 0,
            "status": status,
            "status_reason": reason,
            "source_path": key,
            "last_updated_utc": now,
        }
    )


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
