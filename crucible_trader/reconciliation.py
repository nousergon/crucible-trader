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
import datetime as dt
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from crucible_trader.broker_statement import BrokerStatement

RECONCILIATION_SCHEMA_VERSION = "broker_reconciliation.v1"

#: Cash agreement tolerance, on the CENT GRID (Brian, 2026-10-28: "only track
#: down to cents and when there are small cents-wide differences round to the
#: IB reporting"). IB `TotalCashValue` is the ground truth for cash and is
#: reported to the cent, so the expected side is rounded to that grid
#: (`_cash_cents_off`) and a gap of up to this many cents is ROUNDED TO IB: the
#: result takes IB's figure as the expected cash and records the absorbed
#: amount as `cash_rounded_to_broker_usd`, so every adjustment stays visible.
#: Anything larger (an undeclared interest posting, a fee, the 10-26 $0.55
#: avgPrice break) is a `cash_delta` to be explained, never absorbed.
#:
#: Why a band and not one cent: the anchor and the broker cash are both
#: cent-rounded IB figures while fills are costed at IB's six-decimal
#: commissions, so the raw residual carries rounding noise of about a cent:
#: 10-23 +0.01 passed and 10-27 -0.01 failed with every position matched
#: (alpha-engine-config-I12376).
CASH_TOLERANCE_USD = 0.05
_CASH_TOLERANCE_CENTS = round(CASH_TOLERANCE_USD * 100)


def _cash_cents_off(broker_cash: float, expected_cash: float) -> int:
    """How many whole cents the broker's cash is off the expected cash, once the
    expected side is rounded to the cent grid the broker reports on."""
    return round(broker_cash * 100) - round(expected_cash * 100)


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
    ``cash_dividend``: ``amount_per_share`` in USD, paid on ``entitled_shares``:
    the holding at the record date, which is what the broker pays on. ``None``
    falls back to the anchored holding, which is only right when the record
    date is the anchor's day. A dividend can be paid on an instrument no longer
    held, so its cash counts whether or not the ticker is in today's book.
    Any other ``kind`` is carried so it can be REFUSED as a finding.
    """

    ticker: str
    kind: str
    ratio: float | None = None
    amount_per_share: float | None = None
    entitled_shares: int | None = None
    #: How ``entitled_shares`` was established, and whether it is an estimate.
    entitlement_source: str = "stored broker statement"
    estimated: bool = False


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


#: How long a dividend the broker has not applied may stay owed before it is a
#: finding of its own. IB's paper account has credited a paid dividend days
#: after its pay date; it has never taken this long.
OWED_DIVIDEND_MAX_DAYS = 10

#: The most owed dividends a single day's cash is matched against. The match
#: tries every combination, so this bounds it (2**12 sums).
OWED_DIVIDEND_MATCH_LIMIT = 12


@dataclasses.dataclass(frozen=True)
class OwedDividend:
    """A dividend paid on an earlier pay date that the broker had not applied.

    The anchor is the broker's own statement, so a dividend the broker credits
    a day or more after its pay date arrives as cash nobody declared. Without
    this record that credit is an unexplained `cash_delta` -- and the pay-date
    shortfall that preceded it was already a finding, so one dividend paid late
    would fail two sessions. A credit is matched to an owed dividend only when
    the cash agrees to the cent band; anything else stays a finding.

    ``entitlement_source`` says how the entitled shares were established and
    ``estimated`` marks a derived entitlement (principle: a missing input is
    estimated, and marked estimated).
    """

    ticker: str
    pay_date: str
    amount_usd: float
    entitlement_source: str = "stored broker statement"
    estimated: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.ticker, self.pay_date)

    def to_document(self) -> dict[str, object]:
        return dataclasses.asdict(self)


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
    #: Dividends paid on an EARLIER pay date that the broker had not applied
    #: (:class:`OwedDividend`); a credit today is matched against these.
    owed_dividends: tuple[OwedDividend, ...] = ()

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
    #: Cents absorbed by rounding to IB's reported cash (signed, broker minus
    #: the unrounded expected; 0.0 when they agreed to the cent). None when the
    #: gap was a finding or there was no expected cash.
    cash_rounded_to_broker_usd: float | None = None
    #: Owed dividends the broker credited today, matched to the cash.
    dividend_credits: tuple[OwedDividend, ...] = ()
    #: Dividends paying today that the broker did not apply: owed from today.
    dividends_unpaid: tuple[OwedDividend, ...] = ()

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
            "cash_rounded_to_broker_usd": self.cash_rounded_to_broker_usd,
            "discrepancies": [d.to_document() for d in self.discrepancies],
            "dividend_credits": [d.to_document() for d in self.dividend_credits],
            "dividends_unpaid": [d.to_document() for d in self.dividends_unpaid],
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

    paid_on = sorted(
        set(universe)
        | {a.ticker for a in inputs.corporate_actions.actions if a.kind == "cash_dividend"}
    )
    dividends = {
        ticker: sum(
            (a.amount_per_share or 0.0)
            * (a.entitled_shares if a.entitled_shares is not None else anchor.get(ticker, 0))
            for a in actions.get(ticker, ())
            if a.kind == "cash_dividend"
        )
        for ticker in paid_on
    }
    expected_cash = (
        inputs.anchor.cash_usd
        + sum(f.cash_usd for f in inputs.fills)
        + sum(c.cash_usd for c in inputs.cash_flows)
        + sum(dividends.values())
    )
    credits = _match_owed_credits(
        inputs.broker.cash_usd, expected_cash, dividends, inputs.owed_dividends
    )
    expected_cash += sum(c.amount_usd for c in credits)
    credited = {c.key for c in credits}
    for owed in inputs.owed_dividends:
        if owed.key in credited:
            continue
        days = (_date(inputs.trading_day) - _date(owed.pay_date)).days
        if days > OWED_DIVIDEND_MAX_DAYS:
            findings.append(
                Discrepancy(
                    "corporate_action",
                    owed.ticker,
                    owed.amount_usd,
                    None,
                    -owed.amount_usd,
                    "broker",
                    f"the {owed.ticker} dividend paid {owed.pay_date} "
                    f"({owed.amount_usd:.2f}) is still not credited after {days} days",
                )
            )
    residual = inputs.broker.cash_usd - expected_cash
    unpaid_today: list[OwedDividend] = []
    if abs(_cash_cents_off(inputs.broker.cash_usd, expected_cash)) > _CASH_TOLERANCE_CENTS:
        unpaid = [
            t
            for t, amount in dividends.items()
            if amount
            and abs(_cash_cents_off(inputs.broker.cash_usd, expected_cash - amount))
            <= _CASH_TOLERANCE_CENTS
        ]
        if unpaid:
            unpaid_today.append(
                OwedDividend(
                    unpaid[0],
                    inputs.trading_day,
                    round(dividends[unpaid[0]], 2),
                    *_entitlement(actions.get(unpaid[0], ())),
                )
            )
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
        rounded_to_broker = None
    else:
        # A cents-wide gap: IB's reported cash is the truth, so the expected
        # side is rounded to it and the absorbed amount is recorded.
        rounded_to_broker = round(residual, 2) + 0.0  # + 0.0: never -0.0

    resolved_rows = [t for t in universe if t in inputs.corporate_actions.resolved]
    return _result(
        inputs,
        broker,
        expected,
        expected_cash,
        findings,
        cash_rounded_to_broker_usd=rounded_to_broker,
        n_comparable=len(resolved_rows),
        # Matches over RESOLVED rows only: an unresolved instrument that happens
        # to agree is not a verified match (v1's `OK_UNVERIFIED` beside 1.000).
        n_matched=sum(1 for t in resolved_rows if expected.get(t, 0) == broker.get(t, 0)),
        dividend_credits=credits,
        dividends_unpaid=tuple(unpaid_today),
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
    cash_rounded_to_broker_usd: float | None = None,
    dividend_credits: tuple[OwedDividend, ...] = (),
    dividends_unpaid: tuple[OwedDividend, ...] = (),
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
        # Rounded to IB: the expected cash IS the broker's reported figure. The
        # raw residual below stays, because the control arm plants against it.
        expected_cash_usd=(
            None
            if expected_cash is None
            else inputs.broker.cash_usd
            if cash_rounded_to_broker_usd is not None
            else round(expected_cash, 2)
        ),
        broker_cash_usd=inputs.broker.cash_usd,
        cash_residual_usd=(
            None if expected_cash is None else round(inputs.broker.cash_usd - expected_cash, 2)
        ),
        cash_rounded_to_broker_usd=None if expected_cash is None else cash_rounded_to_broker_usd,
        discrepancies=tuple(sorted(findings, key=lambda d: (order[d.klass], d.instrument))),
        dividend_credits=dividend_credits,
        dividends_unpaid=dividends_unpaid,
    )


def _date(day: str) -> dt.date:
    return dt.date.fromisoformat(day)


def _entitlement(actions: Iterable[CorporateAction]) -> tuple[str, bool]:
    """How a ticker's paying dividend established its entitled shares. Only
    called for a ticker that has a paying dividend."""
    action = next(a for a in actions if a.kind == "cash_dividend")
    return action.entitlement_source, action.estimated


def _match_owed_credits(
    broker_cash: float,
    expected_cash: float,
    dividends: Mapping[str, float],
    owed: Sequence[OwedDividend],
) -> tuple[OwedDividend, ...]:
    """The smallest set of ``owed`` dividends whose credit explains today's cash.

    Tried in order of size, starting from none, so a day whose cash already
    agrees credits nothing and a day short exactly one of today's dividends is
    still that finding. A set matches when the cash agrees within the cent band
    with it credited -- either outright, or short exactly one of today's
    dividends (credited late and paying late on the same day). No set matching
    leaves the gap to the findings below, unchanged: a credit that does not
    agree with what was owed is never absorbed.
    """
    candidates = list(owed)[:OWED_DIVIDEND_MATCH_LIMIT]
    today = [amount for amount in dividends.values() if amount]
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            with_credit = expected_cash + sum(c.amount_usd for c in subset)
            if abs(_cash_cents_off(broker_cash, with_credit)) <= _CASH_TOLERANCE_CENTS or any(
                abs(_cash_cents_off(broker_cash, with_credit - amount)) <= _CASH_TOLERANCE_CENTS
                for amount in today
            ):
                return tuple(subset)
    return ()


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
