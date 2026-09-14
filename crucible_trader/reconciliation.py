"""The broker reconciler: an expected book against the broker's statement.

`alpha-engine-config-I10651`, lifted from `crucible-executor`'s
`executor/reconciliation_audit.py` (anchored parity, same-day split rebase) and
`executor/eod_reconcile.py` (its caller) -- WITHOUT their defect class. Plan
§9.5, amended 2026-09-10: the v1 reconciler produced a **false 1.000 match rate**
behind an incomplete split check, and a ledger-replay backfill reported **20
positions against a real 7**. Both are a checker that cannot produce a negative
result. Each is closed here by construction:

**The false 1.000.** v1 computed ``rate = 1.0 if not universe``, skipped a
malformed split row with a warning, and graded an unresolved split check
``OK_UNVERIFIED`` beside a 1.000. Here:

* the match rate is ``None`` over zero comparable rows, and its metric renders
  ``N/A-LOW-N`` with a reason -- never a pass;
* the corporate-action check must positively RESOLVE every instrument on either
  side; an unresolved instrument is a finding, and a finding fails the run;
* an action kind the reconciler does not handle (spinoff, merger, stock
  dividend, ...) is a finding -- never a skipped row;
* a split that does not produce whole shares is a finding, not a rounding;
* the run's status is derived from the FINDINGS, not from the match rate, so a
  1.000 position rate beside a cash break or a count break reads failed.

**The 20-vs-7.** v1's cold-start fallback replayed the whole trades ledger when
no prior broker snapshot existed. Here there is no replay path at all: the
expected book is ANCHORED on the prior day's :class:`BrokerStatement` (the
broker's own word, which `broker_statement` refuses to build from a ledger) plus
the day's fills and actions. With no anchor the reconciliation is a finding
unless the operator declared genesis explicitly. And the POSITION COUNT is
compared as its own check: a count disagreement is the headline finding
(:data:`HEADLINE_CLASS`), listed first, never a footnote under a rate.

Pure: no store, no broker, no clock. :mod:`crucible_trader.reconciliation_job`
does the I/O and :mod:`crucible_trader.reconciliation_control` replays this
function with planted discrepancies.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from crucible_trader.broker_statement import BrokerStatement

RECONCILIATION_SCHEMA_VERSION = "broker_reconciliation.v1"

#: Cash agreement tolerance. One cent: IB reports cash to the cent, and any
#: larger unexplained movement (an undeclared interest posting, a fee) is a
#: discrepancy to be explained by a declared cash flow, not absorbed.
CASH_TOLERANCE_USD = 0.01

#: The corporate-action kinds this reconciler can apply. Anything else that is
#: RESOLVED for a held instrument is an `unhandled_corporate_action` finding.
HANDLED_ACTION_KINDS: tuple[str, ...] = ("split", "cash_dividend")

#: Every finding class, closed. Order is severity for rendering: the count
#: comparison is the headline (plan §9.5 amendment).
DiscrepancyClass = Literal[
    "position_count",
    "share_delta",
    "cash_delta",
    "corporate_action",
    "unhandled_corporate_action",
    "corporate_action_check_incomplete",
    "no_anchor",
    "stale_anchor",
]
DISCREPANCY_CLASSES: tuple[str, ...] = DiscrepancyClass.__args__  # type: ignore[attr-defined]
HEADLINE_CLASS = "position_count"

#: The instrument name a cash finding carries.
CASH_INSTRUMENT = "USD"


class ReconciliationInputError(ValueError):
    """The inputs are not a reconciliation that can be run at all."""


@dataclasses.dataclass(frozen=True)
class Fill:
    """One execution the trader recorded. ``shares`` signed (+buy), ``cash_usd``
    signed (a buy is negative), commission included."""

    ticker: str
    shares: int
    cash_usd: float


@dataclasses.dataclass(frozen=True)
class CashFlow:
    """A declared non-trade cash movement (interest, fee). Dividends are NOT
    declared here: they are derived from `cash_dividend` actions."""

    description: str
    cash_usd: float


@dataclasses.dataclass(frozen=True)
class CorporateAction:
    """A same-day action on one instrument.

    ``split``: ``ratio`` is new shares per old (2.0 for 2-for-1, 0.1 for 1-for-10).
    ``cash_dividend``: ``amount_per_share`` in USD, paid on the anchored holding.
    Any other ``kind`` is carried so it can be REFUSED as a finding.
    """

    ticker: str
    kind: str
    ratio: float | None = None
    amount_per_share: float | None = None


@dataclasses.dataclass(frozen=True)
class CorporateActionSet:
    """The day's actions AND the instruments whose action status was established.

    ``resolved`` is the positive half v1 lacked: an instrument absent from
    ``actions`` is "no action" only if it is in ``resolved``. A source whose
    query failed returns an empty ``resolved``, and every held instrument
    becomes a `corporate_action_check_incomplete` finding.
    """

    resolved: frozenset[str]
    actions: tuple[CorporateAction, ...] = ()


@dataclasses.dataclass(frozen=True)
class ReconciliationInputs:
    trading_day: str
    broker: BrokerStatement
    anchor: BrokerStatement | None
    fills: tuple[Fill, ...] = ()
    cash_flows: tuple[CashFlow, ...] = ()
    corporate_actions: CorporateActionSet = CorporateActionSet(resolved=frozenset())
    previous_trading_day: str | None = None
    #: Declared by an operator, never inferred from a missing anchor.
    genesis: bool = False

    def __post_init__(self) -> None:
        if self.broker.trading_day != self.trading_day:
            raise ReconciliationInputError(
                f"broker statement is for {self.broker.trading_day}, not {self.trading_day}"
            )
        if self.genesis and self.anchor is not None:
            raise ReconciliationInputError(
                "genesis was declared over an existing anchor statement; genesis means "
                "there is no prior broker statement, and declaring it over one would "
                "discard the comparison"
            )
        if self.anchor is not None and self.anchor.account != self.broker.account:
            raise ReconciliationInputError(
                f"anchor account {self.anchor.account!r} is not broker account "
                f"{self.broker.account!r}"
            )
        if self.anchor is not None and self.anchor.trading_day >= self.trading_day:
            raise ReconciliationInputError(
                f"anchor statement {self.anchor.trading_day} is not before {self.trading_day}"
            )


@dataclasses.dataclass(frozen=True)
class Discrepancy:
    klass: str
    instrument: str
    expected: float | None
    actual: float | None
    delta: float | None
    side: str
    detail: str

    def page_line(self) -> str:
        return (
            f"{self.klass}: {self.instrument} expected={_fmt(self.expected)} "
            f"actual={_fmt(self.actual)} delta={_fmt(self.delta)} side={self.side} "
            f"-- {self.detail}"
        )

    def to_document(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReconciliationResult:
    trading_day: str
    account: str
    genesis: bool
    n_positions_expected: int
    n_positions_broker: int
    n_comparable: int
    n_matched: int
    expected_positions: Mapping[str, int]
    broker_positions: Mapping[str, int]
    expected_cash_usd: float | None
    broker_cash_usd: float
    cash_residual_usd: float | None
    discrepancies: tuple[Discrepancy, ...]

    @property
    def match_rate(self) -> float | None:
        """``None`` over zero comparable rows. There is no 1.0 for nothing."""
        if self.n_comparable == 0:
            return None
        return self.n_matched / self.n_comparable

    @property
    def clean(self) -> bool:
        return not self.discrepancies

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "trading_day": self.trading_day,
            "account": self.account,
            "genesis": self.genesis,
            "clean": self.clean,
            "headline": self.discrepancies[0].page_line() if self.discrepancies else None,
            "n_positions_expected": self.n_positions_expected,
            "n_positions_broker": self.n_positions_broker,
            "n_comparable": self.n_comparable,
            "n_matched": self.n_matched,
            "match_rate": self.match_rate,
            "expected_positions": dict(self.expected_positions),
            "broker_positions": dict(self.broker_positions),
            "expected_cash_usd": self.expected_cash_usd,
            "broker_cash_usd": self.broker_cash_usd,
            "cash_residual_usd": self.cash_residual_usd,
            "discrepancies": [d.to_document() for d in self.discrepancies],
        }


def reconcile(inputs: ReconciliationInputs) -> ReconciliationResult:
    """Reconcile ``inputs``. Every negative the module docstring names is a
    :class:`Discrepancy` in the result; nothing is logged-and-skipped."""
    broker = dict(inputs.broker.positions)
    findings: list[Discrepancy] = []

    if inputs.anchor is None:
        if not inputs.genesis:
            findings.append(
                Discrepancy(
                    "no_anchor",
                    inputs.broker.account,
                    None,
                    None,
                    None,
                    "trader",
                    "no prior broker statement to anchor the expected book on, and genesis "
                    "was not declared. v1 fell back to replaying the trades ledger here, "
                    "which is the path that reported 20 positions against a real 7; this "
                    "reconciler has no such path.",
                )
            )
        return _result(inputs, broker, {}, None, findings)

    if (
        inputs.previous_trading_day is not None
        and inputs.anchor.trading_day != inputs.previous_trading_day
    ):
        findings.append(
            Discrepancy(
                "stale_anchor",
                inputs.broker.account,
                None,
                None,
                None,
                "trader",
                f"anchor statement is {inputs.anchor.trading_day}, not the previous "
                f"session {inputs.previous_trading_day}; the days between were never "
                "reconciled, so today's comparison would absorb their changes",
            )
        )

    anchor = dict(inputs.anchor.positions)
    fills = _net_fills(inputs.fills)
    universe = sorted(set(anchor) | set(broker) | set(fills))
    actions = _actions_by_ticker(inputs.corporate_actions.actions)

    unresolved = [t for t in universe if t not in inputs.corporate_actions.resolved]
    for ticker in unresolved:
        findings.append(
            Discrepancy(
                "corporate_action_check_incomplete",
                ticker,
                None,
                None,
                None,
                "trader",
                "corporate-action status was not established for this instrument; its "
                "parity cannot be verified and is not reported as a match",
            )
        )

    expected: dict[str, int] = {}
    ratio_of: dict[str, float] = {}
    for ticker in universe:
        ratio = 1.0
        for action in actions.get(ticker, ()):
            if action.kind not in HANDLED_ACTION_KINDS:
                findings.append(
                    Discrepancy(
                        "unhandled_corporate_action",
                        ticker,
                        None,
                        None,
                        None,
                        "trader",
                        f"corporate action kind {action.kind!r} is not one this reconciler "
                        f"applies ({', '.join(HANDLED_ACTION_KINDS)}); the reconciliation "
                        "fails rather than skipping the row",
                    )
                )
            elif action.kind == "split":
                if action.ratio is None or not math.isfinite(action.ratio) or action.ratio <= 0:
                    findings.append(
                        Discrepancy(
                            "unhandled_corporate_action",
                            ticker,
                            None,
                            None,
                            None,
                            "trader",
                            f"split ratio {action.ratio!r} is not a positive finite number "
                            "(v1 skipped such a row; it is refused here)",
                        )
                    )
                else:
                    ratio *= action.ratio
        rebased = anchor.get(ticker, 0) * ratio
        if not float(rebased).is_integer():
            findings.append(
                Discrepancy(
                    "unhandled_corporate_action",
                    ticker,
                    rebased,
                    float(broker.get(ticker, 0)),
                    None,
                    "trader",
                    f"split ratio {ratio!r} on {anchor.get(ticker, 0)} shares is not whole "
                    "shares (cash in lieu is not modelled); not rounded into agreement",
                )
            )
        ratio_of[ticker] = ratio
        value = int(round(rebased)) + fills.get(ticker, 0)
        if value != 0:
            expected[ticker] = value

    for ticker in universe:
        want, have = expected.get(ticker, 0), broker.get(ticker, 0)
        if want == have:
            continue
        delta = have - want
        unapplied = anchor.get(ticker, 0) + fills.get(ticker, 0)
        if ratio_of[ticker] != 1.0 and have == unapplied:
            findings.append(
                Discrepancy(
                    "corporate_action",
                    ticker,
                    float(want),
                    float(have),
                    float(delta),
                    "broker",
                    f"broker shares equal the pre-action holding; the same-day split "
                    f"(ratio {ratio_of[ticker]!r}) is not applied on the broker side",
                )
            )
        else:
            findings.append(
                Discrepancy(
                    "share_delta",
                    ticker,
                    float(want),
                    float(have),
                    float(delta),
                    "broker_over" if delta > 0 else "broker_under",
                    "broker share count differs from anchor + fills + applied actions",
                )
            )

    n_expected, n_broker = len(expected), len(broker)
    if n_expected != n_broker:
        findings.append(
            Discrepancy(
                HEADLINE_CLASS,
                inputs.broker.account,
                float(n_expected),
                float(n_broker),
                float(n_broker - n_expected),
                "broker_over" if n_broker > n_expected else "broker_under",
                f"the expected book holds {n_expected} positions and the broker holds {n_broker}",
            )
        )

    dividends = {
        ticker: sum(
            (a.amount_per_share or 0.0) * anchor.get(ticker, 0)
            for a in actions.get(ticker, ())
            if a.kind == "cash_dividend"
        )
        for ticker in universe
    }
    expected_cash = (
        inputs.anchor.cash_usd
        + sum(f.cash_usd for f in inputs.fills)
        + sum(c.cash_usd for c in inputs.cash_flows)
        + sum(dividends.values())
    )
    residual = inputs.broker.cash_usd - expected_cash
    if abs(residual) > CASH_TOLERANCE_USD:
        unpaid = [
            t
            for t, amount in dividends.items()
            if amount and abs(residual + amount) <= CASH_TOLERANCE_USD
        ]
        if unpaid:
            findings.append(
                Discrepancy(
                    "corporate_action",
                    unpaid[0],
                    round(expected_cash, 2),
                    inputs.broker.cash_usd,
                    round(residual, 2),
                    "broker",
                    f"cash is short exactly the {unpaid[0]} dividend "
                    f"({dividends[unpaid[0]]:.2f}); it is not applied on the broker side",
                )
            )
        else:
            findings.append(
                Discrepancy(
                    "cash_delta",
                    CASH_INSTRUMENT,
                    round(expected_cash, 2),
                    inputs.broker.cash_usd,
                    round(residual, 2),
                    "broker_over" if residual > 0 else "broker_under",
                    "broker cash differs from anchor + fills + declared flows + dividends",
                )
            )

    resolved_rows = [t for t in universe if t in inputs.corporate_actions.resolved]
    return _result(
        inputs,
        broker,
        expected,
        expected_cash,
        findings,
        n_comparable=len(resolved_rows),
        # Matches over RESOLVED rows only: an unresolved instrument that happens
        # to agree is not a verified match (v1's `OK_UNVERIFIED` beside 1.000).
        n_matched=sum(1 for t in resolved_rows if expected.get(t, 0) == broker.get(t, 0)),
    )


def _result(
    inputs: ReconciliationInputs,
    broker: Mapping[str, int],
    expected: Mapping[str, int],
    expected_cash: float | None,
    findings: Sequence[Discrepancy],
    *,
    n_comparable: int = 0,
    n_matched: int = 0,
) -> ReconciliationResult:
    order = {klass: i for i, klass in enumerate(DISCREPANCY_CLASSES)}
    return ReconciliationResult(
        trading_day=inputs.trading_day,
        account=inputs.broker.account,
        genesis=inputs.genesis,
        n_positions_expected=len(expected),
        n_positions_broker=len(broker),
        n_comparable=n_comparable,
        n_matched=n_matched,
        expected_positions=dict(sorted(expected.items())),
        broker_positions=dict(sorted(broker.items())),
        expected_cash_usd=None if expected_cash is None else round(expected_cash, 2),
        broker_cash_usd=inputs.broker.cash_usd,
        cash_residual_usd=(
            None if expected_cash is None else round(inputs.broker.cash_usd - expected_cash, 2)
        ),
        discrepancies=tuple(sorted(findings, key=lambda d: (order[d.klass], d.instrument))),
    )


def _net_fills(fills: Iterable[Fill]) -> dict[str, int]:
    net: dict[str, int] = {}
    for fill in fills:
        net[fill.ticker] = net.get(fill.ticker, 0) + fill.shares
    return net


def _actions_by_ticker(
    actions: Iterable[CorporateAction],
) -> dict[str, list[CorporateAction]]:
    grouped: dict[str, list[CorporateAction]] = {}
    for action in actions:
        grouped.setdefault(action.ticker, []).append(action)
    return grouped


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"
