"""The live-environment seam feeding `reconcile_cycle`'s broker-derived arguments.

`alpha-engine-config-I11071`: `trader.reconcile` was registered, implemented,
gate-graded and granted, and nothing dispatched it -- `commands.py` wired no
`reconcile` subcommand. This module is the missing entry point, in the same
shape `broker_session.py`'s own docstring already names as the seam: an
already-connected IB session, never a constructed one.

**Two of `reconcile_cycle`'s three broker-derived arguments are wired for
real; the third is a stated, fail-loud gap, not a fabrication:**

* ``source`` -- :class:`~crucible_trader.broker_statement.IbPaperStatementSource`
  over a :class:`~crucible_trader.broker_session.ReadOnlyBroker`. Already
  existed (`paper_smoke.py` uses the identical pair); reused, not reimplemented.
* ``fills`` -- :func:`fills_from_ib`: today's executions from an explicit
  `reqExecutions()` on a session connected with the gateway's **Master API
  client ID**, mapped to :class:`~crucible_trader.reconciliation.Fill`.
  `alpha-engine-config-I12376` (Brian's option (b)): the paper account is
  shared with the v1 executor, and IB returns executions to a session for its
  own client id only -- so the reconcile used to see none of v1's fills and
  graded each as a `share_delta`/`cash_delta`. A master session receives every
  API client's executions, so the reconcile is now of the WHOLE account, and
  the guard and its tolerances are unchanged. If the gateway is not actually
  configured with that master id, v1's fills are simply absent and the guard
  fails loud on them, as before; it can never pass on a missing fill. **Not exercised against a real
  IB Gateway in this change** -- there is no live box in this session (plan
  §9.5 paper->live is reserved) -- so it is tested here only against a fake
  shaped to the documented `ib_async` `Fill`/`Execution`/`CommissionReport`
  fields. A shape mismatch on the real SDK is a defect this module's own
  narrow `_READ` surface (mirroring `ReadOnlyBroker`) is built to surface
  loudly rather than swallow.

  **Commissions come from the session's fill store, matched by execId**
  (`alpha-engine-config-I12376`, 2026-10-21 run: a `cash_delta` of exactly
  -1.00 on a day whose one v1 fill, DUOL 39 @ 142.44, carried a 1.00
  commission). In `ib_async` 2.1.0 the connect's own startup sync already
  requests the day's executions, so the session store (`fills()`) holds them
  and its `commissionReport` callback updates THOSE `Fill` objects in place.
  An explicit `reqExecutions()` then answers each already-known execution
  with a NEW `Fill` carrying a default `CommissionReport()` -- `execId=""`,
  `commission=0.0`, never `None` -- that nothing ever fills in. Reading the
  commission off that answer costed every fill at zero, silently. The answer
  still names the executions (it asks the gateway afresh); each one's cost is
  read from the store, and only from a report whose `execId` is that
  execution's, waiting a bounded moment for one still in flight.
* ``cash_flows`` and ``corporate_actions`` -- **no live source exists.** IB's
  API gives no direct feed of declared non-trade cash movements (interest,
  fees) or corporate actions on held instruments; both need a source this
  package does not have (a FlexQuery report, or a market-data corporate-action
  feed). Rather than fabricate one, this module supplies the safe defaults the
  reconciler's own contract already defines for "no source": an empty
  ``cash_flows`` tuple, and :func:`live_corporate_actions`'s empty
  ``resolved`` set. Per `reconciliation.py`'s own docstring, an empty
  ``resolved`` makes every held instrument a `corporate_action_check_incomplete`
  finding -- the reconciler's own designed fail-loud behaviour for "I cannot
  positively say nothing happened", never a false clean. A real cash flow
  shows up as an unexplained `cash_delta` finding for the same reason: an
  undeclared movement is a discrepancy by design, not a hole.

  **Corporate actions now have a source** (`alpha-engine-config-I11076`, option
  A, Brian's 2026-10-22 ruling: Polygon for corporate actions, IB for holdings
  and cash). :func:`live_corporate_action_source` reads nousergon-data's daily
  Polygon scan (:mod:`crucible_trader.corporate_action_feed`). When that scan is
  absent, unreadable or not configured, it falls back to exactly the state
  :func:`live_corporate_actions` describes, so the fail-loud default holds.
  ``cash_flows`` (interest, fees) still has no source.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Protocol

from crucible_trader.broker_session import ReadOnlyBroker
from crucible_trader.broker_statement import IbPaperStatementSource, StatementSourceProtocol
from crucible_trader.corporate_action_feed import scan_provider, scans_uri_from_env
from crucible_trader.reconciliation import CorporateActionSet, Fill

#: IB execution sides this module knows how to sign. Anything else is a
#: defect surfaced loudly, never coerced.
_SIDE_SIGN = {"BOT": 1, "SLD": -1}

#: How long, in total, :func:`fills_from_ib` lets the session's event loop run
#: for commission reports still in flight before it gives up loudly, and the
#: step it waits in. IB sends each report just after its execution; this is
#: headroom, not a tuning knob.
COMMISSION_REPORT_WAIT_S = 10.0
COMMISSION_REPORT_POLL_S = 0.5

#: The TWS API's "no value" double (`ibapi.common.UNSET_DOUBLE`); a report
#: carrying it, or any non-finite number, has no commission to cost.
_UNSET_DOUBLE = 1.7976931348623157e308


class UnrecognizedFillSideError(RuntimeError):
    """An IB execution reported a side this module cannot sign. Never guessed."""


class MissingCommissionReportError(RuntimeError):
    """An IB fill carries no commission report yet. A fill cannot be costed
    without one, and costing it as zero would be a silent, wrong number on
    the money path."""


class _Execution(Protocol):
    execId: str  # noqa: N815 - SDK spelling
    time: dt.datetime
    side: str
    shares: float
    #: This execution's own price. `avgPrice` (the order's cumulative average,
    #: beside `cumQty`) is deliberately not read: see `fills_from_ib`.
    price: float


class _CommissionReport(Protocol):
    execId: str  # noqa: N815 - SDK spelling
    commission: float | None


class _IbFill(Protocol):  # the subset of `ib_async.Fill` read here
    contract: Any
    execution: _Execution
    commissionReport: _CommissionReport | None  # noqa: N815 - SDK spelling


class _FillReader(Protocol):
    def reqExecutions(self) -> list[_IbFill]: ...  # noqa: N802 - SDK spelling

    def fills(self) -> list[_IbFill]: ...

    def sleep(self, seconds: float) -> Any: ...


def live_statement_source(client: Any) -> StatementSourceProtocol:
    """The connected session's book, read the same way `paper_smoke` reads it:
    through the read-only view, never the order-capable client directly."""
    return IbPaperStatementSource(ReadOnlyBroker(client))


def _commission(report: _CommissionReport | None, exec_id: str) -> float | None:
    """The commission ``report`` attaches to execution ``exec_id``, or None
    when it attaches none: no report, a report for another execution (the
    SDK's default `CommissionReport()` has ``execId == ""``), or no number."""
    if report is None or getattr(report, "execId", None) != exec_id:
        return None
    value = report.commission
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value >= _UNSET_DOUBLE:
        return None
    return value


def _store_commissions(client: _FillReader, exec_ids: set[str]) -> dict[str, float]:
    found: dict[str, float] = {}
    for record in client.fills():
        exec_id = record.execution.execId
        if exec_id in exec_ids:
            value = _commission(record.commissionReport, exec_id)
            if value is not None:
                found[exec_id] = value
    return found


def fills_from_ib(
    client: _FillReader,
    trading_day: str,
    *,
    report_wait_s: float = COMMISSION_REPORT_WAIT_S,
    poll_s: float = COMMISSION_REPORT_POLL_S,
) -> tuple[Fill, ...]:
    """Today's executions on the account, as reconciliation `Fill`s.

    The executions are read with `reqExecutions()`, which asks the gateway
    afresh rather than trusting the session's startup cache; on a
    Master-API-client-ID session it returns every API client's executions
    (see the module docstring). Filtered to ``trading_day`` by the
    execution's own timestamp -- the gateway may return more than today's --
    and de-duplicated by ``execId``.

    Each execution's commission is the one the session's fill store
    (`fills()`) has attached to that same ``execId`` -- or the answer's own
    report, if it carries that execId -- and never the SDK's unattached
    default (see the module docstring). Reports still in flight are waited
    for, letting the session's event loop run in ``poll_s`` steps for at most
    ``report_wait_s`` in total. A side this module does not recognise, or an
    execution still without its own commission report after that, raises
    rather than being priced as a guess or a zero.
    """
    executions: dict[str, _IbFill] = {}
    for record in client.reqExecutions():
        execution = record.execution
        if execution.time.date().isoformat() != trading_day:
            continue
        if execution.side not in _SIDE_SIGN:
            raise UnrecognizedFillSideError(
                f"{record.contract.symbol!r} execution side {execution.side!r} is not one of "
                f"{sorted(_SIDE_SIGN)}"
            )
        executions.setdefault(execution.execId, record)

    commissions = {
        exec_id: value
        for exec_id, record in executions.items()
        if (value := _commission(record.commissionReport, exec_id)) is not None
    }
    waited = 0.0
    while True:
        pending = set(executions) - set(commissions)
        if pending:
            commissions.update(_store_commissions(client, pending))
            pending = set(executions) - set(commissions)
        if not pending or waited >= report_wait_s:
            break
        client.sleep(poll_s)
        waited += poll_s

    fills: list[Fill] = []
    for exec_id, record in executions.items():
        execution = record.execution
        commission = commissions.get(exec_id)
        if commission is None:
            raise MissingCommissionReportError(
                f"{record.contract.symbol!r} fill {exec_id!r} at {execution.time} has no "
                f"commission report of its own after {waited:g}s; a fill cannot be costed "
                "without one"
            )
        sign = _SIDE_SIGN[execution.side]
        # `execution.price`, never `execution.avgPrice`: IB's `avgPrice` is the
        # ORDER's cumulative average up to this execution (it travels with
        # `cumQty`), so `shares * avgPrice` summed over an order's partial
        # executions is not what the broker debited. The 2026-10-26 run (24
        # executions across 10 orders) read +0.55 broker_over on exactly that,
        # while the same orders at their final average reconciled to the cent.
        notional = float(execution.shares) * float(execution.price)
        fills.append(
            Fill(
                ticker=record.contract.symbol,
                shares=sign * int(execution.shares),
                # A buy (sign=+1) pays notional plus commission (cash_usd < 0);
                # a sell (sign=-1) receives notional minus commission.
                cash_usd=-sign * notional - commission,
            )
        )
    return tuple(fills)


def live_corporate_actions() -> CorporateActionSet:
    """No live corporate-action data source is wired (see module docstring).
    Empty ``resolved`` is the reconciler's own documented "cannot positively
    say nothing happened" state, not a stand-in for a real feed."""
    return CorporateActionSet(resolved=frozenset())


def live_corporate_action_source(env: Any = None) -> Any:
    """The live corporate-action provider: the daily Polygon scan named by
    ``CRUCIBLE_TRADER_CORPORATE_ACTION_SCANS_URI``. Unset, missing or unreadable
    resolves nothing, the same state as :func:`live_corporate_actions`."""
    return scan_provider(scans_uri_from_env(env))
