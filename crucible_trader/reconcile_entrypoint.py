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
* ``fills`` -- :func:`fills_from_ib`, new here: today's executions from the
  connected session's own `fills()` read, mapped to
  :class:`~crucible_trader.reconciliation.Fill`. **Not exercised against a real
  IB Gateway in this change** -- there is no live box in this session (plan
  §9.5 paper->live is reserved) -- so it is tested here only against a fake
  shaped to the documented `ib_async` `Fill`/`Execution`/`CommissionReport`
  fields. A shape mismatch on the real SDK is a defect this module's own
  narrow `_READ` surface (mirroring `ReadOnlyBroker`) is built to surface
  loudly rather than swallow.
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

  Tracked follow-up for a real source: `alpha-engine-config-I11076`.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

from crucible_trader.broker_session import ReadOnlyBroker
from crucible_trader.broker_statement import IbPaperStatementSource, StatementSourceProtocol
from crucible_trader.reconciliation import CorporateActionSet, Fill

#: IB execution sides this module knows how to sign. Anything else is a
#: defect surfaced loudly, never coerced.
_SIDE_SIGN = {"BOT": 1, "SLD": -1}


class UnrecognizedFillSideError(RuntimeError):
    """An IB execution reported a side this module cannot sign. Never guessed."""


class MissingCommissionReportError(RuntimeError):
    """An IB fill carries no commission report yet. A fill cannot be costed
    without one, and costing it as zero would be a silent, wrong number on
    the money path."""


class _Execution(Protocol):
    time: dt.datetime
    side: str
    shares: float
    avgPrice: float  # noqa: N815 - SDK spelling


class _CommissionReport(Protocol):
    commission: float | None


class _IbFill(Protocol):  # the subset of `ib_async.Fill` read here
    contract: Any
    execution: _Execution
    commissionReport: _CommissionReport | None  # noqa: N815 - SDK spelling


class _FillReader(Protocol):
    def fills(self) -> list[_IbFill]: ...


def live_statement_source(client: Any) -> StatementSourceProtocol:
    """The connected session's book, read the same way `paper_smoke` reads it:
    through the read-only view, never the order-capable client directly."""
    return IbPaperStatementSource(ReadOnlyBroker(client))


def fills_from_ib(client: _FillReader, trading_day: str) -> tuple[Fill, ...]:
    """Today's executions on the connected session, as reconciliation `Fill`s.

    Filtered to ``trading_day`` by the execution's own timestamp -- IB's
    `fills()` returns the session's whole cache, not just today's. A side
    this module does not recognise, or a fill with no commission report yet,
    raises rather than being priced as a guess or a zero.
    """
    fills: list[Fill] = []
    for record in client.fills():
        execution = record.execution
        if execution.time.date().isoformat() != trading_day:
            continue
        sign = _SIDE_SIGN.get(execution.side)
        if sign is None:
            raise UnrecognizedFillSideError(
                f"{record.contract.symbol!r} execution side {execution.side!r} is not one of "
                f"{sorted(_SIDE_SIGN)}"
            )
        report = record.commissionReport
        if report is None or report.commission is None:
            raise MissingCommissionReportError(
                f"{record.contract.symbol!r} fill at {execution.time} has no commission report; "
                "a fill cannot be costed without one"
            )
        notional = float(execution.shares) * float(execution.avgPrice)
        commission = float(report.commission)
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
