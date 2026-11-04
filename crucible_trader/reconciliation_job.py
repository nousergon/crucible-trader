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
5c. run the shadow proposer (:mod:`crucible_trader.exception_proposal`) over
   the records 5b just filed and file its proposals at
   `trader/exception_proposals/{day}.json` (`alpha-engine-config-I13119`,
   back office 1b) -- same contract as 5b: any failure is logged and
   swallowed, and never changes what step 6 decides or pages (see
   :func:`file_exception_proposals`);
5d. book the day into the v2 double-entry ledger, the book of record, and
   reconcile the LEDGER to the broker (:func:`crucible_trader.ledger.post_daily_ledger`,
   `alpha-engine-config-I13128`) -- never raises, and never a change to what
   step 6 decides or pages;
5e. file the back office's score and queue
   (:func:`crucible_trader.exception_decision.score`) at
   `trader/exception_scores/{day}.json` and `current.json`: every stored proposal against every
   reviewer decision, agreement per class and ruleset version, and the
   undecided queue oldest first with its age (`alpha-engine-config-I13120`,
   back office 1c) -- same contract as 5c (see :func:`file_exception_score`).
   A queue item older than 3 trading days makes its ``exception_queue_overdue``
   row FAIL, which the crucible ``alerts.sweep`` pages on
   (`alpha-engine-config-I13129`); it never changes step 6's verdict.
   It runs after 5f, so the proposals a refused close files are in the same
   day's queue (`alpha-engine-config-I13121`);
5f. close the previous month when it is not closed yet
   (:func:`crucible_trader.month_end_close.close_prior_month`, back office 1e,
   `alpha-engine-config-I13122`): positions, cash and P&L tied out to IB and
   the settled closes, the period locked, every unexplained line an
   `exception.v1` record with a shadow proposal for the review queue, and a
   refused close retried next run; then file the
   month's performance statement and simulated fee bill from that close
   (:func:`crucible_trader.performance_statement.file_statement`, 1f,
   `alpha-engine-config-I13123`). Never raises; a no-op once the month is closed
   and its statement filed;
5g. once that close and statement are published, rebuild the close, the
   statement and the fee bill from stored inputs and compare each to what was
   published, once per month
   (:func:`crucible_trader.month_rebuild.verify_month_rebuild`, back office 1h,
   `alpha-engine-config-I13129`). A difference is a break and a FAIL metric;
   it never raises;
6. raise :class:`ReconciliationVoidError` when a plant was missed, else
   :class:`ReconciliationDiscrepancyError` when any finding PAGES (see
   :func:`paging_decision`). Either makes the manifest `status: failed`, which
   is the page condition (§4.6), and the reason names every finding's
   instrument, delta and side.

**Cash tiers** (Brian, 2026-11-02, AI House thread; extends the 5-cent band of
crucible-trader#66). A cash gap of 5 cents or less never becomes a finding: the
reconciler rounds to IB and records it as `cash_rounded_to_broker_usd`. A
`cash_delta` finding of at most :data:`CASH_PAGE_THRESHOLD_USD` is still a
finding -- written to the result, filed as an `exception.v1` record, `clean` is
false and every metric row says `WATCH` -- but on its own it does not page.
Anything larger pages, as does any other class of finding, and so does the
month's running total of unpaged cash (rounding plus low-band gaps, absolute)
once it passes :data:`MONTHLY_UNPAGED_CASH_CAP_USD`, so many small gaps cannot
add up unseen.

**Registered in the harness contract** (crucible-PR297): `trader.reconcile` is
in `crucible.models.JOB_VALUES`, both keys are owned by `crucible.keys`
(`trader_reconciliation_key`, `trader_broker_statement_key`), and the
reconciliation key is on `crucible.manifest.MONEY_PATH_PREDICATES`, so the
harness's single manifest writer attaches a `money_path_link` from this run's
OUTPUTS (`alpha-engine-config-I10414`). Nothing here declares a key shape.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

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
from crucible_trader.exception_decision import (
    SCORE_CURRENT_KEY,
    SCORE_SCHEMA_VERSION,
    SCORES_PREFIX,
    exception_score_key,
    load_decisions,
    load_proposals,
    score,
)
from crucible_trader.exception_proposal import (
    PROPOSAL_SCHEMA_VERSION,
    PROPOSALS_PREFIX,
    exception_proposals_key,
    proposal_day_document,
    propose_day,
)
from crucible_trader.exception_record import (
    EXCEPTION_SCHEMA_VERSION,
    PAPER_FUND_TENANT,
    exception_day_document,
    exceptions_key,
    records_from_broker_reconciliation,
    records_from_order_intent_reconciliation,
)
from crucible_trader.ledger import load_ledger, post_daily_ledger
from crucible_trader.month_end_close import close_prior_month
from crucible_trader.month_rebuild import verify_month_rebuild
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
from crucible_trader.performance_statement import file_statement
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

_LOG = logging.getLogger(__name__)

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


#: The largest `cash_delta` (absolute USD) that is filed and queued but does not
#: page on its own. Above it, a cash gap pages.
CASH_PAGE_THRESHOLD_USD = 1.00

#: The month-to-date total (absolute USD) of cash the job has NOT paged on --
#: every `cash_rounded_to_broker_usd` plus every low-band `cash_delta` -- above
#: which today pages anyway. An estimate set 2026-11-02; Brian may change it.
MONTHLY_UNPAGED_CASH_CAP_USD = 5.00


def _low_band_cash(discrepancy: Mapping[str, object]) -> float | None:
    """The absolute delta of a `cash_delta` at or under the page threshold, else
    ``None`` (any other class, a missing delta, or a gap that pages)."""
    if discrepancy.get("klass") != "cash_delta":
        return None
    delta = discrepancy.get("delta")
    if not isinstance(delta, int | float) or abs(delta) > CASH_PAGE_THRESHOLD_USD:
        return None
    return abs(float(delta))


def unpaged_cash_usd(document: Mapping[str, object]) -> float:
    """Cash one stored result did not page on: the absolute rounding absorbed
    plus every low-band `cash_delta`. A pre-tier result counts the same way, so
    the cap only ever errs toward paging."""
    rounded = document.get("cash_rounded_to_broker_usd")
    total = abs(float(rounded)) if isinstance(rounded, int | float) else 0.0
    discrepancies = document.get("discrepancies")
    rows = cast("list[object]", discrepancies) if isinstance(discrepancies, list) else []
    for d in rows:
        low = _low_band_cash(cast("Mapping[str, object]", d)) if isinstance(d, Mapping) else None
        total += low or 0.0
    return round(total, 2)


@dataclasses.dataclass(frozen=True)
class PagingDecision:
    """Whether step 6 pages on today's findings, and the month's unpaged cash."""

    pages: bool
    unpaged_cash_mtd_usd: float
    reason: str


def paging_decision(
    result: ReconciliationResult, prior_documents: Mapping[str, Mapping[str, object]]
) -> PagingDecision:
    """Today pages when any finding is not a low-band `cash_delta`, or when the
    month's unpaged cash, today included, is over the cap. ``prior_documents``
    is day -> stored result; only days in today's calendar month count."""
    month = result.trading_day[:7]
    mtd = round(
        sum(unpaged_cash_usd(d) for day, d in prior_documents.items() if day[:7] == month)
        + unpaged_cash_usd(result.to_document()),
        2,
    )
    paging = [d for d in result.discrepancies if _low_band_cash(d.to_document()) is None]
    if paging:
        return PagingDecision(True, mtd, f"{len(paging)} finding(s) above the no-page band")
    if mtd > MONTHLY_UNPAGED_CASH_CAP_USD:
        return PagingDecision(
            True,
            mtd,
            f"unpaged cash this month is {mtd:.2f} USD, over the "
            f"{MONTHLY_UNPAGED_CASH_CAP_USD:.2f} USD cap",
        )
    if result.discrepancies:
        return PagingDecision(
            False,
            mtd,
            f"cash gap within {CASH_PAGE_THRESHOLD_USD:.2f} USD: filed and queued, not paged",
        )
    return PagingDecision(False, mtd, "no finding")


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
    ledger_positions, ledger_source = ledger_leg(ctx.store, day, broker.account)

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
        ledger_positions=ledger_positions,
        ledger_source=ledger_source,
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
    decision = paging_decision(result, documents)
    for metric in metric_records(
        result, verdict, now=now, source_path=trader_reconciliation_key(day), decision=decision
    ):
        ctx.record_metric(metric)
    intent_document: dict[str, Any] | None = None  # pyright: ignore[reportExplicitAny]  # raw JSON document
    try:
        if order_records is not None:
            orders, executions = order_records
            intent_document = reconcile_intents_step(ctx, orders=orders, executions=executions)
    finally:
        # In a `finally` so a step-5a raise still files the broker breaks, and
        # neither step 5b nor 5c raises, so the raise (or step 6's page) is
        # exactly what it would have been without them.
        filed = file_exception_records(ctx, document, intent_document, now=now)
        file_exception_proposals(ctx, filed, now=now)
        # Step 5d: the ledger is the book of record. Like 5b and 5c it never
        # raises, so the decision and page below are exactly what they were.
        post_daily_ledger(
            ctx,
            broker=broker,
            anchor=anchor,
            fills=inputs.fills,
            cash_flows=inputs.cash_flows,
            corporate_actions=corporate_actions,
            reconciliation=document,
            now=now,
        )
        # Step 5f: after 5d, so the month's last journal is in the ledger.
        close_prior_month(
            ctx,
            day=day,
            account=broker.account,
            now=now,
            after_close=lambda close: file_statement(ctx, close, now=now),
        )
        # Step 5g: once a month's close and statement are published, rebuild
        # them and the fee bill from stored inputs (1h). Never raises.
        verify_month_rebuild(ctx, day=day, account=broker.account, now=now)
        # Step 5e runs LAST: after 5c and 5f, so today's proposals and a
        # refused close's proposals are both in the queue it files.
        file_exception_score(ctx, now=now)

    if not verdict.passed:
        raise ReconciliationVoidError(verdict.reason())
    if decision.pages:
        raise ReconciliationDiscrepancyError(f"{page_text(result)} [{decision.reason}]")
    return result


def ledger_leg(store: Store, day: str, account: str) -> tuple[dict[str, int] | None, str]:
    """The ledger's share counts before ``day``: the third leg of the
    corporate-action check. NEVER raises: an unreadable ledger is an
    unavailable leg (recorded with its reason), and step 5d reports the ledger's
    own failure, so this read cannot change what the job decides or pages."""
    try:
        ledger = load_ledger(store, PAPER_FUND_TENANT, before=day)
        if not ledger.has_account(PAPER_FUND_TENANT, account):
            return None, f"unavailable: no {PAPER_FUND_TENANT} ledger for {account} before {day}"
        return (
            dict(ledger.positions(PAPER_FUND_TENANT, account)),
            f"ledger journals ({PAPER_FUND_TENANT}) before {day}",
        )
    except Exception as exc:  # noqa: BLE001 - a missing leg is recorded, never a raise
        return None, f"unavailable: {type(exc).__name__}: {exc}"[:300]


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
) -> dict[str, object] | None:
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

    Returns the day document it filed, for step 5c, or ``None`` when it filed
    nothing (a clean day, or a failure).
    """
    day = ctx.trading_day.isoformat()
    key = exceptions_key(day)
    count = 0
    filed: dict[str, object] | None = None
    try:
        records = records_from_broker_reconciliation(broker_document, detected_at=now)
        if intent_document is not None:
            records += records_from_order_intent_reconciliation(
                intent_document, void=broker_document.get("void") is True
            )
        count = len(records)
        if records:
            day_document = exception_day_document(day, records)
            ctx.record_output(
                key,
                json.dumps(day_document, indent=2, sort_keys=True).encode("utf-8"),
                EXCEPTION_SCHEMA_VERSION,
            )
            filed = day_document
    except Exception as exc:  # noqa: BLE001 - see docstring: must never mask the page
        status, value = "FAIL", None
        reason = (
            f"exception records for {day} were NOT filed ({count} breaks): "
            + (f"{type(exc).__name__}: {exc}")[:400]
        )
    else:
        if not records:
            return None
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
    return filed


#: The MetricRecord that says whether step 5c filed the day's proposals.
EXCEPTION_PROPOSALS_METRIC = "exception_proposals_filed"


def file_exception_proposals(
    ctx: RunContext, exception_document: Mapping[str, object] | None, *, now: str
) -> None:
    """Step 5c: file the shadow proposer's proposals for the day's records.
    Never raises.

    `alpha-engine-config-I13119` (back office 1b): the proposer
    (:func:`~crucible_trader.exception_proposal.propose_day`) ran only by hand,
    so no day had proposals for a reviewer to decide. This step proposes over
    exactly the day document step 5b filed and files the result at
    :func:`~crucible_trader.exception_proposal.exception_proposals_key`, plus
    one ``exception_proposals_filed`` MetricRecord. When 5b filed nothing there
    is nothing to propose, and this step adds nothing.

    The proposals are shadow: nothing reads them to act. So ANY failure here --
    in the proposer, the write or the metric -- is logged and swallowed, and the
    job's outputs, decision, raise and page text are what they would have been
    without this step. A failed proposer costs the day's proposals, never the
    reconciliation.
    """
    if exception_document is None:
        return
    day = ctx.trading_day.isoformat()
    key = PROPOSALS_PREFIX  # the metric's source path until the day's key is built
    try:
        key = exception_proposals_key(day)
        proposals = propose_day(exception_document, proposed_at=now)
        ctx.record_output(
            key,
            json.dumps(proposal_day_document(day, proposals), indent=2, sort_keys=True).encode(
                "utf-8"
            ),
            PROPOSAL_SCHEMA_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring: shadow, must never reach the job
        _LOG.warning("exception proposals for %s were NOT filed", day, exc_info=True)
        status, value = "FAIL", None
        reason = (
            f"exception proposals for {day} were NOT filed: "
            + (f"{type(exc).__name__}: {exc}")[:400]
        )
    else:
        status, value = "OK", float(len(proposals))
        reason = f"{len(proposals)} shadow exception proposals filed for {day}"
    try:
        ctx.record_metric(
            {
                "name": EXCEPTION_PROPOSALS_METRIC,
                "module": "crucible_trader.exception_proposal",
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
    except Exception:  # noqa: BLE001 - see docstring
        _LOG.warning("exception proposals metric for %s was NOT recorded", day, exc_info=True)


#: The MetricRecord that says whether step 5e filed the day's score.
EXCEPTION_SCORE_METRIC = "exception_score_filed"

#: The MetricRecord that says whether any queue item is past the age limit
#: (`alpha-engine-config-I13129`: "anything older than 3 trading days pages").
#: FAIL while one is; the crucible ``alerts.sweep`` reads it from this job's
#: manifest and pages (``crucible.alerts.evaluate_overdue_breaks``). It is a
#: row of its own, never a raise, so a reviewer backlog never becomes a failed
#: reconcile: "the broker disagrees" and "a person has not decided yet" stay
#: two different pages.
EXCEPTION_OVERDUE_METRIC = "exception_queue_overdue"

#: How many overdue items the overdue row's reason names, oldest first.
_OVERDUE_NAMED = 5


def file_exception_score(ctx: RunContext, *, now: str) -> None:
    """Step 5e: file the back office's score and queue for the day. Never raises.

    `alpha-engine-config-I13120` (back office 1c): the reviewer's decisions are
    written by the reviewer's own identity
    (:func:`~crucible_trader.exception_decision.record_decision`), never by the
    trader, which only proposes. This step READS every stored proposal up to
    today and every decision, and files the derived
    :func:`~crucible_trader.exception_decision.score` -- agreement per class
    and ruleset version, and the queue of undecided and deferred proposals
    oldest first with their age in trading days -- at
    :func:`~crucible_trader.exception_decision.exception_score_key` and, unless
    a later day already filed one, at
    :data:`~crucible_trader.exception_decision.SCORE_CURRENT_KEY` (the review
    page's read), plus one
    ``exception_score_filed`` MetricRecord whose status is WATCH while any
    queue item is past the age limit. When the score is filed it also records
    ``exception_queue_overdue`` (:data:`EXCEPTION_OVERDUE_METRIC`): the
    overdue count, FAIL while it is above zero, naming the oldest items. That
    row is the page signal (`alpha-engine-config-I13129`); the WATCH stays as
    it was, because the sweep reads it from a trader pinned before this row.

    It runs every day, breaks or not, because a queue item ages on a clean day
    too. Shadow like 5c: ANY failure is logged and swallowed and reported as
    that metric with status FAIL, and the job's outputs, decision, raise and
    page text are what they would have been without this step.
    """
    day = ctx.trading_day.isoformat()
    key = SCORES_PREFIX  # the metric's source path until the day's key is built
    overdue_row: dict[str, object] | None = None
    try:
        key = exception_score_key(day)
        document = score(
            load_proposals(ctx.store, until=day),
            load_decisions(ctx.store),
            as_of=day,
            generated_at=now,
        )
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        ctx.record_output(key, payload, SCORE_SCHEMA_VERSION)
        current: dict[str, object] = (
            json.loads(ctx.store.get_bytes(SCORE_CURRENT_KEY).decode("utf-8"))
            if ctx.store.exists(SCORE_CURRENT_KEY)
            else {}
        )
        if str(current.get("trading_day", "")) <= day:
            ctx.record_output(SCORE_CURRENT_KEY, payload, SCORE_SCHEMA_VERSION)
        totals = cast("Mapping[str, object]", document["totals"])
        overdue = cast("int", totals["overdue"])
        status, value = ("WATCH" if overdue else "OK"), float(cast("int", totals["queue"]))
        reason = (
            f"back-office queue for {day}: {totals['queue']} open, {overdue} overdue, "
            f"agreement {totals['agreement']} over {totals['agree']} agreed and "
            f"{totals['correct']} corrected"
        )
        overdue_row = _overdue_metric(document, day=day, key=key, now=now)
    except Exception as exc:  # noqa: BLE001 - see docstring: shadow, must never reach the job
        _LOG.warning("exception score for %s was NOT filed", day, exc_info=True)
        status, value = "FAIL", None
        reason = (
            f"exception score for {day} was NOT filed: " + (f"{type(exc).__name__}: {exc}")[:400]
        )
    try:
        ctx.record_metric(
            {
                "name": EXCEPTION_SCORE_METRIC,
                "module": "crucible_trader.exception_decision",
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
    except Exception:  # noqa: BLE001 - see docstring
        _LOG.warning("exception score metric for %s was NOT recorded", day, exc_info=True)
    if overdue_row is not None:
        try:
            ctx.record_metric(overdue_row)
        except Exception:  # noqa: BLE001 - see docstring
            _LOG.warning("exception overdue metric for %s was NOT recorded", day, exc_info=True)


def _overdue_metric(
    document: Mapping[str, object], *, day: str, key: str, now: str
) -> dict[str, object]:
    """The ``exception_queue_overdue`` row for a filed score (see step 5e)."""
    queue = cast("list[Mapping[str, object]]", document["queue"])
    overdue = [q for q in queue if q["overdue"] is True]
    limit = document["age_limit_trading_days"]
    if overdue:
        named = "; ".join(
            f"{q['proposal_id']} ({q['trading_day']}, {q['age_trading_days']} trading days)"
            for q in overdue[:_OVERDUE_NAMED]
        )
        more = len(overdue) - _OVERDUE_NAMED
        reason = (
            f"{len(overdue)} back-office queue items on {day} are older than {limit} "
            f"trading days with no decision, oldest first: {named}"
            + (f"; and {more} more" if more > 0 else "")
        )
    else:
        reason = f"no back-office queue item on {day} is older than {limit} trading days"
    return {
        "name": EXCEPTION_OVERDUE_METRIC,
        "module": "crucible_trader.exception_decision",
        "metric_type": "operational",
        "value": float(len(overdue)),
        "unit": "count",
        "n_floor": 0,
        "status": "FAIL" if overdue else "OK",
        "status_reason": reason,
        "source_path": key,
        "last_updated_utc": now,
    }


def page_text(result: ReconciliationResult) -> str:
    """Every finding, headline first, each naming instrument, delta and side."""
    return (
        f"broker reconciliation {result.trading_day} ({result.account}): "
        f"{len(result.discrepancies)} discrepancies. "
        + " | ".join(d.page_line() for d in result.discrepancies)
    )


def metric_records(
    result: ReconciliationResult,
    verdict: ControlVerdict,
    *,
    now: str,
    source_path: str,
    decision: PagingDecision,
) -> list[dict[str, object]]:
    """MetricRecord rows (`crucible.models.MetricRecordRow`). Status comes from
    the FINDINGS: a 1.000 rate beside any finding is FAIL (or `WATCH` when the
    only findings are low-band cash gaps that do not page), and no comparable
    rows is `N/A-LOW-N`, never OK. ``decision`` is :func:`paging_decision`'s."""
    status = "OK" if result.clean else ("FAIL" if decision.pages else "WATCH")
    headline = result.discrepancies[0].page_line() if result.discrepancies else None

    def row(
        name: str, value: float | None, unit: str, n_floor: int, row_status: str, reason: str
    ) -> dict[str, object]:
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
            "broker_reconciliation_unpaged_cash_mtd_usd",
            decision.unpaged_cash_mtd_usd,
            "usd",
            1,
            "OK" if decision.unpaged_cash_mtd_usd <= MONTHLY_UNPAGED_CASH_CAP_USD else "FAIL",
            f"cash not paged on in {result.trading_day[:7]} (rounding plus gaps up to "
            f"{CASH_PAGE_THRESHOLD_USD:.2f} USD), cap {MONTHLY_UNPAGED_CASH_CAP_USD:.2f} USD",
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
