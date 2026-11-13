"""The v2 double-entry ledger: the trader's book of record.

`alpha-engine-config-I13128` (back office 1g), per nous-ergon-ops
`policies/portfolio-accounting-policy.md` Rules 1-4 and 8. Until this module the
v2 reconciler's expected side was IB's prior statement plus the day's fills, so
the broker's figures were compared with themselves. This module keeps our own
book and reconciles THAT to the broker.

**Shape.** A journal of :class:`Entry` records, each a set of :class:`Posting`
s whose ``amount_cents`` sum to exactly zero (USD, integer cents: Brian's
"cash to whole cents"; never a float). A posting on a securities account also
carries ``quantity`` (shares). Every entry carries ``tenant`` (multi-tenant from
day one; the Nous Ergon paper fund is :data:`PAPER_FUND_TENANT`) and the broker
``account``.

**One economic event, one entry** (Rule 4). Each fill is a ``fill`` entry
(security against cash at the execution price) and its commission is its own
``commission`` entry. A cash dividend is a ``dividend_receivable`` entry and a
``dividend_settlement`` entry. A split is a ``split`` entry restating quantity
at unchanged cents. A declared cash flow (interest, fee) is its own entry.
Entry ids are event-addressed (:func:`entry_id`), so a re-run cannot book an
event twice.

**Ground truth** (Brian 2026-10-22, Rule 5): IB for shares and cash; Polygon
(the reconciler's corporate-action set) for splits and dividends. After the
day's events are booked, the ledger is compared with the broker's statement.
Every difference is a BREAK, and **every cash gap gets a journal entry**
(Brian 2026-11-02): a ``cash_gap`` entry moves the difference between cash and
the named suspense account :data:`SUSPENSE_CASH`; a share difference is a
``position_gap`` entry against ``suspense:position_gap:<T>``. Each gap entry
cites the break's exception ids (the reconciler's own ``exception.v1`` ids for
the same instrument, plus the ledger's break id), ``decision_id`` (``None``
until a 1c decision posts the clearing adjustment) and an ``expires`` day. The
suspense balances are reported every day and the ledger is not ``clean`` while
any is non-zero: a gap is booked, never absorbed (Rule 8).

**Clearing a decided break** (`alpha-engine-config-I13160`). A gap leaves
suspense only by an ``adjustment`` entry, and only on a reviewer's
``exception_decision.v1`` record (1c): when the standing decision on an
exception the gap cites is ``agree`` or ``correct``, the day's posting books ONE
adjustment per gap that reverses its suspense posting against
:data:`RECONCILIATION_ADJUSTMENTS` (cash, whole cents) or
``equity:share_adjustments:<T>`` (shares), citing the gap's exception ids, the
decision id and who decided. A ``defer``, a superseded or a conflicted decision
clears nothing. The trader makes no call here: the approver is the reviewer
whose decision is cited (maker-checker), and a decision takes effect on the
first trading day AFTER the day it was made, so a re-run of any day books the
same bytes. A gap is cleared once: any adjustment already citing one of its ids
settles it, exactly as the 1h aging check and the close (1e) read it.

**Cash tiers** (Brian 2026-10-28 and 2026-11-02, the same bands as the
reconciler's, read from :mod:`crucible_trader.reconciliation`). The ledger's own
cash difference to IB is judged in whole cents:

- **at most** ``CASH_TOLERANCE_USD`` (5 cents): rounded to IB. It is NOT a break
  and goes to no suspense; it is booked as its own ``cash_rounding`` entry
  against the named account :data:`CASH_ROUNDING`, reported in the day's view
  (``cash_rounded_to_broker_cents``, and the account's running balance) and as
  its own line in the close's attribution, so every absorbed cent stays visible;
- **over 5 cents, up to** ``CASH_PAGE_THRESHOLD_USD`` ($1.00): a ``cash_gap`` in
  suspense with ``tier: review``; the metric says ``WATCH`` and the break goes to
  the review queue, aging as above;
- **over $1.00**: the same ``cash_gap`` with ``tier: page``, and the metric
  says ``FAIL``.

Shares have no band: any share difference is a ``position_gap``.

A decision on the month-end close's own line for a gap still in suspense
(``close_suspense_cash`` / ``close_suspense_shares``) clears that gap the same
way: :func:`close_suspense_line_ids` derives the ids the close gives a gap's
lines, so a close-line decision resolves to exactly the gaps that line
summarises. The adjustment still cites the GAP's exception ids (so it settles
the gap once, whichever line was decided) and names the decided close line in
``decided_exception_id``.

**Append-only** (Rule 3). A day's journal is written once, create-if-absent
(:func:`post_daily_ledger`); a re-run that produces the same bytes is a no-op
and one that produces different bytes is refused and reported. A correction is
a new ``adjustment`` entry in a later day's journal.

**Trial balance on every append**: :meth:`Ledger.append` refuses an unbalanced
entry, a duplicate id, and re-checks that the whole ledger sums to zero.

**Opening balance.** The first day a tenant's account has no journal, an
``opening_balance`` entry books the cash and share counts of the anchor (IB's
statement for the previous session; on a declared genesis day, today's).
Securities open at zero cents: cost basis is not established at opening, and
valuation is shares x settled close (Rule 7), which lands with the close (1e).
The securities accounts' cents are trade consideration (net cash paid in), not
cost basis.

**Known deltas, stated rather than hidden.** The corporate-action feed gives a
dividend on its PAY date, so the receivable is recognised on the pay date, not
the ex-date (Rule 4 wants the ex-date); the cash settles when the reconciler
says it arrived (``dividends_unpaid`` / ``dividend_credits``). Valuation entries
(Rule 7) are not booked here.

**Accrued interest** (`alpha-engine-config-I13202`). IB's NetLiquidation
includes interest accrued and not yet posted to cash (``accrued_cash_usd``,
``AccruedCash``); GIPS needs it accrued and Rule 1 wants it as its own entry.
Each day whose statement records an accrual, the ledger moves
:data:`INTEREST_RECEIVABLE` to IB's figure: a move away from zero is an
``interest_accrual`` entry against :data:`INTEREST_INCOME` (IB's figure is
negative for debit interest), a move toward zero is an ``interest_settlement``
entry into cash (IB posting the interest; the posted amount itself has
no source yet, so any difference from IB's cash is that day's ordinary cash
gap). The accrual the book OPENED with is opening equity, not income: the first
time the receivable is booked, the opening statement's accrual is its own
``interest_accrual`` entry against :data:`OPENING`. When the opening statement
did not record an accrual (``broker_statement.v1``), how much of the first
recorded balance predates the opening is a genuine data gap: the balance is
booked once to :data:`INTEREST_FIRST_RECORDED`, its own account and its own
line in the close, never folded into the month's interest income. A recorded
zero is a baseline like any other (a zero entry). A statement that records no
accrual books nothing; the receivable stands.

**Failure isolation.** :func:`post_daily_ledger` NEVER raises: the trader's
reconcile job decides and pages exactly as before; this step rides beside it
and reports through one MetricRecord (``FAIL`` when it could not post).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, cast

from crucible.calendar import is_trading_day
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key
from crucible.store import (
    ETAG_ABSENT,
    PointerConflictError,
    Store,
)

from crucible_trader.broker_statement import BrokerStatement, statement_from_document
from crucible_trader.exception_decision import effective_decisions, load_decisions
from crucible_trader.exception_record import (
    PAPER_FUND_TENANT,
    SOURCE_MONTH_END_CLOSE,
    exception_id,
    records_from_broker_reconciliation,
)
from crucible_trader.reconciliation import (
    CASH_PAGE_THRESHOLD_USD,
    CASH_TOLERANCE_USD,
    CashFlow,
    CorporateActionSet,
    Fill,
)

LEDGER_SCHEMA_VERSION = "ledger_journal.v1"
LEDGER_RECONCILIATION_SCHEMA_VERSION = "ledger_reconciliation.v1"

TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

#: Journals are create-only (append-only by grant as well as by code); the
#: reconciliation view is a derived document and may be rewritten by a re-run.
LEDGER_JOURNAL_PREFIX = "trader/ledger/journal/"
LEDGER_RECONCILIATION_PREFIX = "trader/ledger/reconciliation/"

CASH = "assets:cash:USD"
COMMISSIONS = "expenses:commissions"
DECLARED_FLOWS = "income:declared_cash_flows"
OPENING = "equity:opening_balance"
SUSPENSE_CASH = "suspense:cash_gap"
SECURITIES = "assets:securities:"
RECEIVABLE = "assets:dividends_receivable:"
DIVIDEND_INCOME = "income:dividends:"
SUSPENSE_POSITION = "suspense:position_gap:"
#: Where a decided cash gap's suspense goes (alpha-engine-config-I13160): the
#: broker's cash is ground truth, so a cleared difference is P&L, its own line
#: in the close's attribution, never folded into another account.
RECONCILIATION_ADJUSTMENTS = "income:reconciliation_adjustments"
#: Interest accrued and not yet posted to cash, held at IB's ``AccruedCash``.
INTEREST_RECEIVABLE = "assets:interest_receivable:USD"
INTEREST_INCOME = "income:interest"
#: The first recorded accrual when the opening's accrual was not recorded: the
#: part earned before the book opened is unknown (alpha-engine-config-I13202).
INTEREST_FIRST_RECORDED = "income:interest_first_recorded"
#: Where a decided share gap's suspense goes: the securities account already
#: holds IB's count, so the offsetting shares leave suspense to here.
SHARE_ADJUSTMENTS = "equity:share_adjustments:"
#: A sale's realized gain (credit, so negative) or loss (debit, positive), per
#: instrument, BEFORE commission (the commission is its own entry), at IB's own
#: figure: the commission report's ``realizedPNL`` plus that commission. The
#: securities account is relieved at the cost IB used, so proceeds less cost is
#: exactly this posting (Brian 2026-11-13: track the gain or loss on each sale).
REALIZED_GAINS = "income:realized_gains:"
#: Where a cash difference inside the 5-cent band goes (Brian 2026-10-28: "round
#: to the IB reporting"): a NAMED plug (policy Rule 2), its balance reported.
CASH_ROUNDING = "income:cash_rounded_to_broker"
_FIXED_ACCOUNTS = frozenset(
    {
        CASH,
        COMMISSIONS,
        DECLARED_FLOWS,
        OPENING,
        SUSPENSE_CASH,
        RECONCILIATION_ADJUSTMENTS,
        CASH_ROUNDING,
        INTEREST_RECEIVABLE,
        INTEREST_INCOME,
        INTEREST_FIRST_RECORDED,
    }
)
#: The cash tiers in whole cents: rounded to IB at or under the first; a gap
#: for review up to the second; a gap that pages above it.
CASH_ROUNDING_BAND_CENTS = round(CASH_TOLERANCE_USD * 100)
CASH_PAGE_THRESHOLD_CENTS = round(CASH_PAGE_THRESHOLD_USD * 100)
_PER_INSTRUMENT = (
    SECURITIES,
    RECEIVABLE,
    DIVIDEND_INCOME,
    SUSPENSE_POSITION,
    SHARE_ADJUSTMENTS,
    REALIZED_GAINS,
)
#: Only these accounts hold shares.
_QUANTITY_ACCOUNTS = (SECURITIES, SUSPENSE_POSITION, SHARE_ADJUSTMENTS)

#: The verdicts that settle a break; ``defer`` leaves it in suspense.
CLEARING_VERDICTS = frozenset({"agree", "correct"})
_GAP_KINDS = ("cash_gap", "position_gap")
#: The month-end close's classes for a gap still in suspense at the close.
CLOSE_SUSPENSE_CASH = "close_suspense_cash"
CLOSE_SUSPENSE_SHARES = "close_suspense_shares"

ENTRY_KINDS = frozenset(
    {
        "opening_balance",
        "fill",
        "commission",
        "declared_cash_flow",
        "interest_accrual",
        "interest_settlement",
        "dividend_receivable",
        "dividend_settlement",
        "split",
        "cash_rounding",
        "cash_gap",
        "position_gap",
        "adjustment",
    }
)

#: Trading days a gap entry's suspense may stand before it is overdue (the
#: plan's break-aging limit: older than 3 trading days pages, 1c/1d).
GAP_EXPIRY_TRADING_DAYS = 3

LEDGER_METRIC = "ledger_book_of_record"


class LedgerError(ValueError):
    """An entry or journal the ledger refuses. Raised inside the pure core;
    :func:`post_daily_ledger` catches it and reports, never re-raises."""


def to_cents(usd: float | Decimal | str) -> int:
    """USD to integer cents, half-up on the decimal string (never binary)."""
    value = Decimal(repr(usd)) if isinstance(usd, float) else Decimal(usd)
    if not value.is_finite():
        raise LedgerError(f"amount {usd!r} is not finite")
    return int((value * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _realized_cents(fill: Fill) -> int | None:
    """A sale's realized gain (positive) or loss before commission, in cents,
    from IB's net ``realizedPNL`` plus the commission it netted; None for a buy,
    or when IB states no realized figure or no commission."""
    if fill.shares >= 0 or fill.realized_pnl_usd is None or fill.commission_usd is None:
        return None
    return to_cents(Decimal(repr(fill.realized_pnl_usd)) + Decimal(repr(fill.commission_usd)))


def _valid_account(name: str) -> bool:
    if name in _FIXED_ACCOUNTS:
        return True
    return any(name.startswith(p) and len(name) > len(p) for p in _PER_INSTRUMENT)


@dataclasses.dataclass(frozen=True)
class Posting:
    account: str
    amount_cents: int
    quantity: int | None = None

    def __post_init__(self) -> None:
        if not _valid_account(self.account):
            raise LedgerError(f"account {self.account!r} is not in the chart of accounts")
        if type(self.amount_cents) is not int:
            raise LedgerError(
                f"{self.account}: amount_cents must be an int, not {self.amount_cents!r}"
            )
        if self.quantity is not None:
            if type(self.quantity) is not int:
                raise LedgerError(f"{self.account}: quantity must be an int")
            if not self.account.startswith(_QUANTITY_ACCOUNTS):
                raise LedgerError(f"{self.account}: only securities accounts carry shares")

    def to_document(self) -> dict[str, object]:
        return {
            "account": self.account,
            "amount_cents": self.amount_cents,
            "quantity": self.quantity,
        }


def entry_id(tenant: str, account: str, trading_day: str, kind: str, event_ref: str) -> str:
    """Event-addressed: the same event always has the same id."""
    payload = json.dumps([tenant, account, trading_day, kind, event_ref], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclasses.dataclass(frozen=True)
class Entry:
    tenant: str
    account: str
    trading_day: str
    kind: str
    event_ref: str
    postings: tuple[Posting, ...]
    evidence: Mapping[str, object] = dataclasses.field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        if not TENANT_RE.fullmatch(self.tenant):
            raise LedgerError(f"tenant {self.tenant!r} does not match {TENANT_RE.pattern}")
        if self.kind not in ENTRY_KINDS:
            raise LedgerError(f"entry kind {self.kind!r} is not one of {sorted(ENTRY_KINDS)}")
        if not self.postings:
            raise LedgerError(f"{self.kind} {self.event_ref}: an entry needs a posting")
        total = sum(p.amount_cents for p in self.postings)
        if total != 0:
            raise LedgerError(
                f"{self.kind} {self.event_ref} on {self.trading_day} does not balance: "
                f"postings sum to {total} cents"
            )
        if self.kind == "adjustment" and not (
            self.evidence.get("exception_ids") and self.evidence.get("decision_id")
        ):
            raise LedgerError("an adjustment must cite its exception ids and decision id")

    @property
    def entry_id(self) -> str:
        return entry_id(self.tenant, self.account, self.trading_day, self.kind, self.event_ref)

    def to_document(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "tenant": self.tenant,
            "account": self.account,
            "trading_day": self.trading_day,
            "kind": self.kind,
            "event_ref": self.event_ref,
            "postings": [p.to_document() for p in self.postings],
            "evidence": dict(self.evidence),
        }


def entry_from_document(document: Mapping[str, Any]) -> Entry:  # pyright: ignore[reportExplicitAny]
    entry = Entry(
        tenant=document["tenant"],
        account=document["account"],
        trading_day=document["trading_day"],
        kind=document["kind"],
        event_ref=document["event_ref"],
        postings=tuple(
            Posting(p["account"], p["amount_cents"], p.get("quantity"))
            for p in document["postings"]
        ),
        evidence=document.get("evidence") or {},
    )
    if document.get("entry_id") != entry.entry_id:
        raise LedgerError(f"stored entry id {document.get('entry_id')!r} is not its event's id")
    return entry


class Ledger:
    """An append-only, always-balanced set of entries."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._ids: set[str] = set()
        self._cents: dict[str, int] = {}
        self._shares: dict[str, int] = {}

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def append(self, entry: Entry) -> None:
        """Append ``entry``; the trial balance is re-checked on every append."""
        if entry.entry_id in self._ids:
            raise LedgerError(
                f"{entry.kind} {entry.event_ref}: entry {entry.entry_id} already booked"
            )
        # Entry checks its own balance too; the append never trusts that alone.
        # A balanced entry onto a zero trial balance keeps it zero.
        entry_sum, balance = sum(p.amount_cents for p in entry.postings), self.trial_balance()
        if entry_sum != 0 or balance != 0:
            raise LedgerError(
                f"refusing {entry.entry_id}: entry sums to {entry_sum} cents onto a trial "
                f"balance of {balance}"
            )
        for p in entry.postings:
            key = _balance_key(entry, p.account)
            self._cents[key] = self._cents.get(key, 0) + p.amount_cents
            if p.quantity is not None:
                self._shares[key] = self._shares.get(key, 0) + p.quantity
        self._entries.append(entry)
        self._ids.add(entry.entry_id)

    def trial_balance(self) -> int:
        return sum(self._cents.values())

    def has_account(self, tenant: str, account: str) -> bool:
        return any(e.tenant == tenant and e.account == account for e in self._entries)

    def cents(self, tenant: str, account: str, ledger_account: str) -> int:
        return self._cents.get(f"{tenant}|{account}|{ledger_account}", 0)

    def positions(self, tenant: str, account: str) -> dict[str, int]:
        prefix = f"{tenant}|{account}|{SECURITIES}"
        return {k[len(prefix) :]: v for k, v in self._shares.items() if k.startswith(prefix) and v}

    def balances(self, tenant: str, account: str, prefix: str) -> dict[str, int]:
        head = f"{tenant}|{account}|"
        return {
            k[len(head) :]: v for k, v in self._cents.items() if k.startswith(head + prefix) and v
        }

    def suspense(self, tenant: str, account: str) -> dict[str, object]:
        head = f"{tenant}|{account}|"
        return {
            "cash_cents": self.cents(tenant, account, SUSPENSE_CASH),
            "shares": {
                k[len(head + SUSPENSE_POSITION) :]: v
                for k, v in self._shares.items()
                if k.startswith(head + SUSPENSE_POSITION) and v
            },
        }


def _balance_key(entry: Entry, ledger_account: str) -> str:
    return f"{entry.tenant}|{entry.account}|{ledger_account}"


def _later_trading_day(day: str, n: int) -> str:
    current = dt.date.fromisoformat(day)
    while n > 0:
        current += dt.timedelta(days=1)
        if is_trading_day(current):
            n -= 1
    return current.isoformat()


@dataclasses.dataclass(frozen=True)
class DayInputs:
    """Everything one day's posting reads. ``reconciliation`` is the stored
    ``broker_reconciliation.v1`` document (with ``control_arm``/``void``)."""

    tenant: str
    broker: BrokerStatement
    anchor: BrokerStatement | None
    fills: tuple[Fill, ...]
    cash_flows: tuple[CashFlow, ...]
    corporate_actions: CorporateActionSet
    reconciliation: Mapping[str, object]
    #: Every stored ``exception_decision.v1`` record (validated); the ones that
    #: clear a gap are chosen by :func:`clearing_decisions`.
    decisions: tuple[Mapping[str, object], ...] = ()
    #: The statement the account's ledger was opened from, when that was an
    #: earlier day: its accrual is the opening's. ``None`` when unreadable.
    opening: BrokerStatement | None = None


@dataclasses.dataclass(frozen=True)
class DayPosting:
    entries: tuple[Entry, ...]
    breaks: tuple[dict[str, object], ...]
    document: dict[str, object]


def book_day(ledger: Ledger, inputs: DayInputs) -> DayPosting:
    """Book one day's events into ``ledger`` (mutating it), compare with the
    broker, book a gap entry per break, and return the day's entries and the
    reconciliation view. Pure: no I/O. Raises :class:`LedgerError`."""
    tenant, broker = inputs.tenant, inputs.broker
    day, account = broker.trading_day, broker.account
    new: list[Entry] = []

    def book(kind: str, ref: str, postings: Iterable[Posting], **evidence: object) -> None:
        entry = Entry(tenant, account, day, kind, ref, tuple(postings), evidence)
        ledger.append(entry)
        new.append(entry)

    opened_from_today = False
    opened_with = inputs.opening
    if not ledger.has_account(tenant, account):
        opening = (
            inputs.anchor
            if inputs.anchor is not None and inputs.anchor.account == account
            else None
        )
        source = opening if opening is not None else broker
        # Opened from TODAY's statement (genesis, no anchor): today's events are
        # already inside it, so booking them again would count them twice.
        opened_from_today = opening is None
        opened_with = source
        cash = to_cents(source.cash_usd)
        book(
            "opening_balance",
            f"statement/{source.trading_day}",
            [Posting(CASH, cash), Posting(OPENING, -cash)]
            + [
                Posting(SECURITIES + t, 0, int(q)) for t, q in sorted(source.positions.items()) if q
            ],
            source="ib_statement",
            statement_day=source.trading_day,
            cost_basis="not established at opening; valued at shares x settled close (Rule 7)",
            includes_this_days_events=opened_from_today,
        )

    prior = ledger.positions(tenant, account)
    actions = () if opened_from_today else inputs.corporate_actions.actions
    fills = () if opened_from_today else inputs.fills
    flows = () if opened_from_today else inputs.cash_flows
    for action in actions:
        if action.kind != "split" or action.ratio is None or action.ratio <= 0:
            continue
        held = prior.get(action.ticker, 0)
        rebased = held * action.ratio
        if not float(rebased).is_integer() or int(rebased) == held:
            continue  # cash in lieu is not modelled; the gap check books it
        book(
            "split",
            f"split/{action.ticker}/{action.ratio!r}",
            [Posting(SECURITIES + action.ticker, 0, int(rebased) - held)],
            source="polygon_corporate_actions",
            ratio=action.ratio,
        )

    for index, fill in enumerate(fills):
        ref = fill.exec_id or f"fill/{index}/{fill.ticker}/{fill.shares}/{fill.cash_usd!r}"
        if fill.commission_usd is None:
            consideration, separated = to_cents(fill.cash_usd), False
        else:
            consideration = to_cents(
                Decimal(repr(fill.cash_usd)) + Decimal(repr(fill.commission_usd))
            )
            separated = True
        realized = _realized_cents(fill)
        if realized is None:
            postings = [
                Posting(CASH, consideration),
                Posting(SECURITIES + fill.ticker, -consideration, fill.shares),
            ]
            evidence: dict[str, object] = (
                {"realized": "not stated by IB for this sale"} if fill.shares < 0 else {}
            )
        else:
            # One entry per sale (Rule 4): cash at proceeds, the holding
            # relieved at IB's cost, the difference to realized gains.
            postings = [
                Posting(CASH, consideration),
                Posting(SECURITIES + fill.ticker, -(consideration - realized), fill.shares),
                Posting(REALIZED_GAINS + fill.ticker, -realized),
            ]
            evidence = {
                "realized_source": "ib_commission_report.realizedPNL + commission",
                "realized_cents": realized,
                "ib_realized_pnl_net_cents": to_cents(fill.realized_pnl_usd),  # pyright: ignore[reportArgumentType]
            }
        book(
            "fill",
            ref,
            postings,
            source="ib_execution",
            commission_separated=separated,
            **evidence,
        )
        if fill.commission_usd is not None and (commission := to_cents(fill.commission_usd)):
            book(
                "commission",
                ref,
                [Posting(COMMISSIONS, commission), Posting(CASH, -commission)],
                source="ib_commission_report",
            )

    for index, flow in enumerate(flows):
        cents = to_cents(flow.cash_usd)
        book(
            "declared_cash_flow",
            f"flow/{index}/{flow.description}",
            [Posting(CASH, cents), Posting(DECLARED_FLOWS, -cents)],
            description=flow.description,
        )

    _book_interest(ledger, inputs, opened_with, book)

    unpaid = {
        (str(u["ticker"]), str(u["pay_date"]))
        for u in _rows(inputs.reconciliation.get("dividends_unpaid"))
    }
    # Two dividends on one ticker and day (an ordinary and a special) are two
    # events: the first keeps the original ref, a later one takes a sequence
    # suffix, so the second is not refused as a duplicate (codex d3c058f085a32ed0).
    dividends_seen: dict[str, int] = {}
    for action in actions:
        if action.kind != "cash_dividend" or action.amount_per_share is None:
            continue
        shares = (
            action.entitled_shares
            if action.entitled_shares is not None
            else prior.get(action.ticker, 0)
        )
        cents = to_cents(Decimal(repr(action.amount_per_share)) * shares)
        if not cents:
            continue
        seq = dividends_seen.get(action.ticker, 0) + 1
        dividends_seen[action.ticker] = seq
        ref = f"dividend/{action.ticker}/{day}" + (f"/{seq}" if seq > 1 else "")
        book(
            "dividend_receivable",
            ref,
            [
                Posting(RECEIVABLE + action.ticker, cents),
                Posting(DIVIDEND_INCOME + action.ticker, -cents),
            ],
            source="polygon_corporate_actions",
            entitled_shares=shares,
            entitlement_source=action.entitlement_source,
            estimated=action.estimated,
            recognised_on="pay_date (the feed carries no ex-date receivable)",
        )
        if (action.ticker, day) not in unpaid:
            book(
                "dividend_settlement",
                ref,
                [Posting(CASH, cents), Posting(RECEIVABLE + action.ticker, -cents)],
                source="ib_statement",
            )
    credits = [] if opened_from_today else _rows(inputs.reconciliation.get("dividend_credits"))
    for credit in credits:
        cents = to_cents(float(credit["amount_usd"]))  # pyright: ignore[reportArgumentType]
        ticker = str(credit["ticker"])
        book(
            "dividend_settlement",
            f"dividend/{ticker}/{credit['pay_date']}",
            [Posting(CASH, cents), Posting(RECEIVABLE + ticker, -cents)],
            source="ib_statement",
            late_credit_of_pay_date=credit["pay_date"],
        )

    breaks = _book_gaps(ledger, inputs, book)
    cleared = _book_clearances(ledger, inputs, book)
    after = ledger.positions(tenant, account)
    suspense = ledger.suspense(tenant, account)
    clean = not breaks and suspense["cash_cents"] == 0 and not suspense["shares"]
    document: dict[str, object] = {
        "schema_version": LEDGER_RECONCILIATION_SCHEMA_VERSION,
        "tenant": tenant,
        "account": account,
        "trading_day": day,
        "trial_balance_cents": ledger.trial_balance(),
        "entries_today": len(new),
        "entry_ids": [e.entry_id for e in new],
        "ledger_cash_cents": ledger.cents(tenant, account, CASH),
        "broker_cash_cents": to_cents(broker.cash_usd),
        "ledger_positions": dict(sorted(after.items())),
        "broker_positions": {t: int(q) for t, q in sorted(broker.positions.items()) if q},
        "breaks": list(breaks),
        "suspense": suspense,
        # Inside the 5-cent band: rounded to IB, its own entry, never a break.
        "cash_rounded_to_broker_cents": sum(
            p.amount_cents
            for e in new
            if e.kind == "cash_rounding"
            for p in e.postings
            if p.account == CASH
        ),
        "cash_rounding_balance_cents": -ledger.cents(tenant, account, CASH_ROUNDING),
        "dividends_receivable_cents": ledger.balances(tenant, account, RECEIVABLE),
        "interest_receivable_cents": ledger.cents(tenant, account, INTEREST_RECEIVABLE),
        "broker_accrued_interest_cents": (
            None if broker.accrued_cash_usd is None else to_cents(broker.accrued_cash_usd)
        ),
        # Per instrument, a gain positive and a loss negative (the account
        # holds the credit), before commission: today's sales and to date.
        "realized_gains_today_cents": _realized_today(new),
        "realized_gains_cents": {
            name[len(REALIZED_GAINS) :]: -c
            for name, c in sorted(ledger.balances(tenant, account, REALIZED_GAINS).items())
        },
        "cleared_today": cleared,
        "clean": clean,
        "known_deltas": [
            "dividend receivable recognised on the pay date, not the ex-date",
            "no daily valuation entry (Rule 7) yet",
            "interest posted to cash is inferred from IB's accrual moving toward zero; the posted "
            "amount has no source, so a difference is that day's cash gap",
        ],
    }
    return DayPosting(tuple(new), tuple(breaks), document)


def _book_interest(
    ledger: Ledger,
    inputs: DayInputs,
    opened_with: BrokerStatement | None,
    book: Any,  # pyright: ignore[reportExplicitAny]
) -> None:
    """Hold :data:`INTEREST_RECEIVABLE` at IB's recorded accrual (module doc)."""
    broker = inputs.broker
    if broker.accrued_cash_usd is None:
        return
    tenant, account, day = inputs.tenant, broker.account, broker.trading_day
    target = to_cents(broker.accrued_cash_usd)
    established = any(
        e.tenant == tenant
        and e.account == account
        and e.kind in ("interest_accrual", "interest_settlement")
        for e in ledger.entries
    )
    if not established:
        if opened_with is None or opened_with.accrued_cash_usd is None:
            # A recorded zero is a baseline too: booked (at zero) so the
            # account is established and a later balance accrues as income.
            book(
                "interest_accrual",
                f"accrual/first-recorded/{day}",
                [
                    Posting(INTEREST_RECEIVABLE, target),
                    Posting(INTEREST_FIRST_RECORDED, -target),
                ],
                source="ib_statement",
                opening_accrual="not recorded by the opening statement",
                period="unknown: some of it may predate the opening",
            )
            return
        at_opening = to_cents(opened_with.accrued_cash_usd)
        book(
            "interest_accrual",
            f"accrual/opening/{opened_with.trading_day}",
            [Posting(INTEREST_RECEIVABLE, at_opening), Posting(OPENING, -at_opening)],
            source="ib_statement",
            statement_day=opened_with.trading_day,
            basis="accrued before the book opened: opening equity, not income",
        )
    # The previous statement's accrual, when the receivable never reached it:
    # days the ledger posted before it booked interest (`alpha-engine-config-I13234`).
    # Interest IB accrued by then is that period's income. Skipping it nets it
    # into today's change, which on a posting day understates the posting by
    # exactly that growth (2026-11-04: $6.77 of an $8.62 page-tier cash gap).
    anchor = inputs.anchor
    if anchor is not None and anchor.accrued_cash_usd is not None and anchor.trading_day < day:
        behind = to_cents(anchor.accrued_cash_usd) - ledger.cents(
            tenant, account, INTEREST_RECEIVABLE
        )
        if behind > 0:
            book(
                "interest_accrual",
                f"accrual/{anchor.trading_day}",
                [Posting(INTEREST_RECEIVABLE, behind), Posting(INTEREST_INCOME, -behind)],
                source="ib_statement",
                statement_day=anchor.trading_day,
                accrued_cents=to_cents(anchor.accrued_cash_usd),
                booked_late="the previous statement's accrual, not booked on its own day",
            )
    held = ledger.cents(tenant, account, INTEREST_RECEIVABLE)
    # A move TOWARD zero is a posting to cash; a move away from zero accrues.
    # IB's AccruedCash is negative for debit interest, so a falling negative
    # balance is more interest owed, not a settlement; a move across zero
    # settles the old balance and accrues the new one.
    settled = (
        held - max(target, 0)
        if held > 0 and target < held
        else held - min(target, 0)
        if held < 0 and target > held
        else 0
    )
    if settled:
        book(
            "interest_settlement",
            f"accrual/{day}",
            [Posting(CASH, settled), Posting(INTEREST_RECEIVABLE, -settled)],
            source="ib_statement",
            accrued_cents=target,
            inferred="a move of IB's accrual toward zero is a posting to cash",
        )
    change = target - held + settled
    if change:
        book(
            "interest_accrual",
            f"accrual/{day}",
            [Posting(INTEREST_RECEIVABLE, change), Posting(INTEREST_INCOME, -change)],
            source="ib_statement",
            accrued_cents=target,
        )


def _realized_today(entries: Sequence[Entry]) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in entries:
        for p in entry.postings:
            if p.account.startswith(REALIZED_GAINS):
                ticker = p.account[len(REALIZED_GAINS) :]
                out[ticker] = out.get(ticker, 0) - p.amount_cents
    return dict(sorted(out.items()))


def _rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, Mapping)]  # pyright: ignore[reportUnknownVariableType]


def _book_gaps(ledger: Ledger, inputs: DayInputs, book: Any) -> list[dict[str, object]]:  # pyright: ignore[reportExplicitAny]
    """Compare with the broker (ground truth for shares and cash) and book one
    gap entry per break, against named suspense. Returns the breaks."""
    tenant, broker = inputs.tenant, inputs.broker
    day, account = broker.trading_day, broker.account
    cited: dict[str, list[str]] = {}
    for record in records_from_broker_reconciliation(
        inputs.reconciliation, detected_at=day, tenant=tenant
    ):
        if record.instrument is not None:
            cited.setdefault(record.instrument, []).append(record.exception_id)
    expires = _later_trading_day(day, GAP_EXPIRY_TRADING_DAYS)
    breaks: list[dict[str, object]] = []

    held = ledger.positions(tenant, account)
    for ticker in sorted(set(held) | {t for t, q in broker.positions.items() if q}):
        ours, theirs = held.get(ticker, 0), int(broker.positions.get(ticker, 0))
        if ours == theirs:
            continue
        ledger_break = exception_id("ledger", day, ("position_gap", account, ticker), tenant=tenant)
        ids = [*cited.get(ticker, []), ledger_break]
        breaks.append(
            {
                "class": "position_gap",
                "instrument": ticker,
                "ledger": ours,
                "broker": theirs,
                "delta": theirs - ours,
                "exception_ids": ids,
            }
        )
        book(
            "position_gap",
            f"gap/{ticker}",
            [
                Posting(SECURITIES + ticker, 0, theirs - ours),
                Posting(SUSPENSE_POSITION + ticker, 0, ours - theirs),
            ],
            source="ib_statement",
            reason="broker share count differs from the ledger; in suspense pending a decision",
            exception_ids=ids,
            decision_id=None,
            expires=expires,
        )

    ours_cash, theirs_cash = ledger.cents(tenant, account, CASH), to_cents(broker.cash_usd)
    delta = theirs_cash - ours_cash
    if delta and abs(delta) <= CASH_ROUNDING_BAND_CENTS:
        book(
            "cash_rounding",
            "rounding/USD",
            [Posting(CASH, delta), Posting(CASH_ROUNDING, -delta)],
            source="ib_statement",
            reason=(
                f"broker cash differs from the ledger by {delta} cent(s), inside the "
                f"{CASH_ROUNDING_BAND_CENTS}-cent band: rounded to IB, not a break"
            ),
            ledger_cents=ours_cash,
            broker_cents=theirs_cash,
        )
    elif delta:
        ledger_break = exception_id("ledger", day, ("cash_gap", account, "USD"), tenant=tenant)
        ids = [*cited.get("USD", []), ledger_break]
        breaks.append(
            {
                "class": "cash_gap",
                "instrument": "USD",
                "ledger": ours_cash,
                "broker": theirs_cash,
                "delta": delta,
                "tier": "review" if abs(delta) <= CASH_PAGE_THRESHOLD_CENTS else "page",
                "exception_ids": ids,
            }
        )
        book(
            "cash_gap",
            "gap/USD",
            [Posting(CASH, delta), Posting(SUSPENSE_CASH, -delta)],
            source="ib_statement",
            reason="broker cash differs from the ledger; booked to suspense pending a decision",
            exception_ids=ids,
            decision_id=None,
            expires=expires,
        )
    return breaks


def clearing_decisions(
    decisions: Iterable[Mapping[str, object]], tenant: str, *, before: str
) -> dict[str, Mapping[str, object]]:
    """Exception id -> the standing ``agree``/``correct`` decision on it, for
    ``tenant``, among decisions made before the day ``before``. A superseded,
    conflicted or deferred decision clears nothing."""
    made = [
        d
        for d in decisions
        if d.get("tenant") == tenant and str(d.get("decided_at", ""))[:10] < before
    ]
    out: dict[str, Mapping[str, object]] = {}
    standing = effective_decisions(made).values()
    for decision in sorted(
        (d for state, d in standing if state == "decided" and d is not None),
        key=lambda d: (str(d["decided_at"]), str(d["decision_id"])),
    ):
        if decision["verdict"] in CLEARING_VERDICTS:
            out.setdefault(str(decision["exception_id"]), decision)
    return out


def _book_clearances(ledger: Ledger, inputs: DayInputs, book: Any) -> list[dict[str, object]]:  # pyright: ignore[reportExplicitAny]
    """Book one ``adjustment`` per open gap a reviewer has decided, moving its
    suspense to :data:`RECONCILIATION_ADJUSTMENTS` / :data:`SHARE_ADJUSTMENTS`.
    Returns what was cleared today."""
    tenant, account, day = inputs.tenant, inputs.broker.account, inputs.broker.trading_day
    decided = clearing_decisions(inputs.decisions, tenant, before=day)
    if not decided:
        return []
    close_days = sorted(
        {
            str(d["trading_day"])
            for d in decided.values()
            if d.get("source") == SOURCE_MONTH_END_CLOSE
        }
    )
    mine = [e for e in ledger.entries if e.tenant == tenant and e.account == account]
    settled = {str(i) for e in mine if e.kind == "adjustment" for i in _cited(e)}
    out: list[dict[str, object]] = []
    for gap in mine:
        ids = _cited(gap)
        if gap.kind not in _GAP_KINDS or settled.intersection(ids):
            continue
        # The gap's own exceptions first, then any month-end close line that
        # reported it still in suspense (a close covers gaps booked by its day).
        lines = [
            i
            for close_day in close_days
            if gap.trading_day <= close_day
            for i in close_suspense_line_ids(gap, close_day)
        ]
        decision = next((decided[i] for i in (*ids, *lines) if i in decided), None)
        if decision is None:
            continue
        postings: list[Posting] = []
        for p in gap.postings:
            if p.account == SUSPENSE_CASH and p.amount_cents:
                postings += [
                    Posting(SUSPENSE_CASH, -p.amount_cents),
                    Posting(RECONCILIATION_ADJUSTMENTS, p.amount_cents),
                ]
            elif p.account.startswith(SUSPENSE_POSITION) and p.quantity:
                ticker = p.account[len(SUSPENSE_POSITION) :]
                postings += [
                    Posting(p.account, 0, -p.quantity),
                    Posting(SHARE_ADJUSTMENTS + ticker, 0, p.quantity),
                ]
        decision_id = str(decision["decision_id"])
        book(
            "adjustment",
            f"clear/{gap.entry_id}",
            postings,
            source="exception_decision",
            reason=f"clears {gap.kind} {gap.entry_id} booked {gap.trading_day} on a decision",
            exception_ids=ids,
            decision_id=decision_id,
            decided_exception_id=decision["exception_id"],
            verdict=decision["verdict"],
            decided=decision["decided"],
            decided_by=decision["decided_by"],
            gap_entry_id=gap.entry_id,
        )
        settled.update(ids)
        out.append(
            {"gap_entry_id": gap.entry_id, "gap_day": gap.trading_day, "decision_id": decision_id}
        )
    return out


def close_suspense_line_ids(gap: Entry, close_day: str) -> list[str]:
    """The exception ids the month-end close whose last trading day is
    ``close_day`` gives ``gap``'s suspense postings while the gap is open: one
    per posting, exactly as :func:`crucible_trader.month_end_close.close_month`
    files them (class, then account, month and ``<instrument>@<gap day>``)."""
    out: list[str] = []
    for p in gap.postings:
        if p.account == SUSPENSE_CASH:
            klass, instrument = CLOSE_SUSPENSE_CASH, "USD"
        elif p.account.startswith(SUSPENSE_POSITION) and p.quantity:
            klass, instrument = CLOSE_SUSPENSE_SHARES, p.account[len(SUSPENSE_POSITION) :]
        else:
            continue
        identity = (klass, gap.account, close_day[:7], f"{instrument}@{gap.trading_day}")
        out.append(exception_id(SOURCE_MONTH_END_CLOSE, close_day, identity, tenant=gap.tenant))
    return out


def _cited(entry: Entry) -> list[str]:
    return [str(i) for i in cast("list[object]", entry.evidence.get("exception_ids") or [])]


# --------------------------------------------------------------------------- I/O


def journal_key(tenant: str, trading_day: str) -> str:
    _require(tenant, trading_day)
    return f"{LEDGER_JOURNAL_PREFIX}{tenant}/{trading_day}.json"


def ledger_reconciliation_key(tenant: str, trading_day: str) -> str:
    _require(tenant, trading_day)
    return f"{LEDGER_RECONCILIATION_PREFIX}{tenant}/{trading_day}.json"


def _require(tenant: str, trading_day: str) -> None:
    if not TENANT_RE.fullmatch(tenant):
        raise LedgerError(f"tenant {tenant!r} does not match {TENANT_RE.pattern}")
    dt.date.fromisoformat(trading_day)


def journal_document(tenant: str, trading_day: str, entries: Sequence[Entry]) -> bytes:
    document = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "tenant": tenant,
        "trading_day": trading_day,
        "entries": [e.to_document() for e in entries],
    }
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8")


def load_journals(store: Store, tenant: str, *, before: str) -> dict[str, tuple[Entry, ...]]:
    """Every stored journal of ``tenant`` dated before ``before``, by day."""
    prefix = f"{LEDGER_JOURNAL_PREFIX}{tenant}/"
    days = sorted(
        k[len(prefix) : -len(".json")]
        for k in store.list_keys(prefix)
        if k.endswith(".json") and k[len(prefix) : -len(".json")] < before
    )
    journals: dict[str, tuple[Entry, ...]] = {}
    for day in days:
        document = json.loads(store.get_bytes(journal_key(tenant, day)).decode("utf-8"))
        if document.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerError(f"journal {day} is {document.get('schema_version')!r}")
        journals[day] = tuple(entry_from_document(raw) for raw in document["entries"])
    return journals


def _later_journal_days(store: Store, tenant: str, day: str) -> list[str]:
    """Stored journal days of ``tenant`` dated after ``day``, ascending."""
    prefix = f"{LEDGER_JOURNAL_PREFIX}{tenant}/"
    return sorted(
        d
        for k in store.list_keys(prefix)
        if k.endswith(".json") and (d := k[len(prefix) : -len(".json")]) > day
    )


def ledger_from(journals: Mapping[str, Sequence[Entry]]) -> Ledger:
    """Replay ``journals`` in day order into a new ledger."""
    ledger = Ledger()
    for day in sorted(journals):
        for entry in journals[day]:
            ledger.append(entry)
    return ledger


def load_ledger(store: Store, tenant: str, *, before: str) -> Ledger:
    """Replay every stored journal of ``tenant`` dated before ``before``."""
    return ledger_from(load_journals(store, tenant, before=before))


def post_daily_ledger(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    *,
    broker: BrokerStatement,
    anchor: BrokerStatement | None,
    fills: Sequence[Fill],
    cash_flows: Sequence[CashFlow],
    corporate_actions: CorporateActionSet,
    reconciliation: Mapping[str, object],
    now: str,
    tenant: str = PAPER_FUND_TENANT,
    closes: SettledCloses | None = None,
) -> DayPosting | None:
    """Step 5d of `trader.reconcile`: book the day, reconcile the LEDGER to the
    broker, write the journal (create-only) and the reconciliation view, and
    record one MetricRecord. NEVER raises; returns ``None`` on failure.

    ``closes`` defaults to the stored price panels (``close_raw``)."""
    day = broker.trading_day
    key = None
    try:
        key = journal_key(tenant, day)
        locks = read_period_locks(ctx.store, tenant)
        if day[:7] in locks:
            raise LedgerError(
                f"{day[:7]} is closed; a late fix is a prior-period adjustment dated in the "
                "open month"
            )
        journals = load_journals(ctx.store, tenant, before=day)
        if not ctx.store.exists(key) and (later := _later_journal_days(ctx.store, tenant, day)):
            raise LedgerError(
                f"journals after {day} already exist ({later[0]}); a day is never inserted "
                "behind a later journal -- post a correction instead"
            )
        ledger = ledger_from(journals)
        decisions, unreadable = _read_decisions(ctx.store)
        opening = _opening_statement(ctx.store, journals, tenant, broker.account)
        posting = book_day(
            ledger,
            DayInputs(
                tenant,
                broker,
                anchor,
                tuple(fills),
                tuple(cash_flows),
                corporate_actions,
                reconciliation,
                decisions,
                opening,
            ),
        )
        if unreadable is not None:
            posting.document["decisions_unreadable"] = unreadable
        checks = run_checks(
            ctx.store,
            tenant,
            broker.account,
            day=day,
            journals={**journals, day: posting.entries},
            locks=locks,
            view=posting.document,
            closes=closes if closes is not None else _panel_closes(ctx.store),
        )
        posting.document["checks"] = checks
        payload = journal_document(tenant, day, posting.entries)
        try:
            ctx.record_output_cas(key, ETAG_ABSENT, payload, LEDGER_SCHEMA_VERSION)
        except PointerConflictError:
            if ctx.store.get_bytes(key) != payload:
                raise LedgerError(
                    f"the {day} journal is already written with different entries; a journal "
                    "is never rewritten -- post a correction instead"
                ) from None
        view_key = ledger_reconciliation_key(tenant, day)
        ctx.record_output(
            view_key,
            json.dumps(posting.document, indent=2, sort_keys=True).encode("utf-8"),
            LEDGER_RECONCILIATION_SCHEMA_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - must never change what the job decides or pages
        _metric(
            ctx,
            now,
            key,
            "FAIL",
            None,
            f"ledger for {day} NOT posted: {type(exc).__name__}: {exc}"[:400],
        )
        return None
    n = len(posting.breaks)
    failed = cast("list[dict[str, object]]", posting.document["checks"]["breaks"])
    if failed:
        _metric(
            ctx,
            now,
            view_key,
            "FAIL",
            float(len(failed)),
            f"ledger checks failed on {day}: "
            + "; ".join(f"{b['check']} {b['detail']}" for b in failed)[:360],
        )
    elif paging := [b for b in posting.breaks if b.get("tier") == "page"]:
        _metric(
            ctx,
            now,
            view_key,
            "FAIL",
            float(n),
            (
                f"ledger cash gap over {CASH_PAGE_THRESHOLD_USD:.2f} USD on {day}: "
                + "; ".join(f"{b['delta']} cents" for b in paging)
                + f"; booked to suspense {posting.document['suspense']}"
            )[:400],
        )
    elif posting.document["clean"]:
        _metric(
            ctx, now, view_key, "OK", 0.0, f"ledger ties to the broker on {day}; trial balance 0"
        )
    else:
        _metric(
            ctx,
            now,
            view_key,
            "WATCH",
            float(n),
            (
                f"{n} ledger break(s) on {day} booked to suspense; "
                f"suspense {posting.document['suspense']}"
                + (
                    f"; decisions unreadable, nothing cleared: {unreadable}"
                    if unreadable is not None
                    else ""
                )
            )[:400],
        )
    return posting


def _opening_statement(
    store: Store, journals: Mapping[str, Sequence[Entry]], tenant: str, account: str
) -> BrokerStatement | None:
    """The stored statement the account's ledger opened from, or ``None`` (no
    opening yet, or unreadable: the accrual is then treated as unrecorded). Read
    only while the interest receivable is not yet booked."""
    if any(
        e.kind in ("interest_accrual", "interest_settlement")
        for entries in journals.values()
        for e in entries
        if e.tenant == tenant and e.account == account
    ):
        return None
    for day in sorted(journals):
        for e in journals[day]:
            if e.kind == "opening_balance" and e.tenant == tenant and e.account == account:
                statement_day = str(e.evidence.get("statement_day") or e.trading_day)
                try:
                    raw = store.get_bytes(trader_broker_statement_key(statement_day))
                    statement = statement_from_document(json.loads(raw))
                except Exception:  # noqa: BLE001 - an unreadable opening is "not recorded"
                    return None
                return statement if statement.account == account else None
    return None


def _read_decisions(store: Store) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    """Every stored decision, or none and why: an unreadable decision store
    means no break is cleared today (it stays in suspense and ages), never a
    day the ledger fails to post."""
    try:
        return tuple(load_decisions(store)), None
    except Exception as exc:  # noqa: BLE001 - clearing waits; the day still posts
        return (), f"{type(exc).__name__}: {exc}"[:300]


def _metric(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    now: str,
    key: str | None,
    status: str,
    value: float | None,
    reason: str,
) -> None:
    try:
        ctx.record_metric(
            {
                "name": LEDGER_METRIC,
                "module": "crucible_trader.ledger",
                "metric_type": "operational",
                "value": value,
                "unit": "count" if value is not None else None,
                "n_floor": 0,
                "status": status,
                "status_reason": reason,
                "source_path": key or LEDGER_JOURNAL_PREFIX,
                "last_updated_utc": now,
            }
        )
    except Exception:  # noqa: BLE001, S110 - a metric must not raise into the trader either
        pass


# ------------------------------------------------------------ checks (1h)

#: ``(trading_day, tickers) -> {ticker: settled close in USD}``; a ticker with
#: no close is absent (the shape of ``month_end_close.CloseSource``).
SettledCloses = Callable[[str, Sequence[str]], Mapping[str, float]]


def _panel_closes(store: Store) -> SettledCloses:
    """Settled closes from the stored price panels. The import (pandas) and the
    read both happen inside the call, which `run_checks` guards, so neither
    can stop a journal from posting."""

    def closes(day: str, tickers: Sequence[str]) -> Mapping[str, float]:
        from crucible_trader.month_end_close import panel_close_source  # noqa: PLC0415 - cycle, pandas

        return panel_close_source(store)(day, tickers)

    return closes


# alpha-engine-config-I13129: roll-forward, completeness, rebuild-from-raw,
# break aging and period lock over the ledger. Each failure is a BREAK in the
# day's view and turns the ledger metric FAIL; none of them raises.

LEDGER_PERIODS_PREFIX = "trader/ledger/periods/"
PERIOD_LOCK_SCHEMA_VERSION = "ledger_period_lock.v1"


def period_lock_key(tenant: str, month: str) -> str:
    _require(tenant, f"{month}-01")
    return f"{LEDGER_PERIODS_PREFIX}{tenant}/{month}.json"


def journal_digest(tenant: str, day: str, entries: Sequence[Entry]) -> str:
    return hashlib.sha256(journal_document(tenant, day, entries)).hexdigest()


def period_lock_document(
    tenant: str, month: str, journals: Mapping[str, Sequence[Entry]]
) -> dict[str, object]:
    """What the month-end close (1e) writes to lock ``month``: the digest of
    every journal in it. A closed month never changes."""
    return {
        "schema_version": PERIOD_LOCK_SCHEMA_VERSION,
        "tenant": tenant,
        "month": month,
        "journals": {
            d: journal_digest(tenant, d, journals[d]) for d in sorted(journals) if d[:7] == month
        },
    }


def read_period_locks(store: Store, tenant: str) -> dict[str, dict[str, str]]:
    """Every locked month of ``tenant`` -> its journal digests."""
    prefix = f"{LEDGER_PERIODS_PREFIX}{tenant}/"
    locks: dict[str, dict[str, str]] = {}
    for key in sorted(store.list_keys(prefix)):
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        if document.get("schema_version") != PERIOD_LOCK_SCHEMA_VERSION:
            raise LedgerError(f"period lock {key} is {document.get('schema_version')!r}")
        locks[str(document["month"])] = dict(document["journals"])
    return locks


def trading_days_between(first: str, last: str) -> list[str]:
    """Trading days from ``first`` to ``last``, both included."""
    days: list[str] = []
    current, end = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    while current <= end:
        if is_trading_day(current):
            days.append(current.isoformat())
        current += dt.timedelta(days=1)
    return days


def roll_forward(
    opening: Ledger,
    entries: Sequence[Entry],
    tenant: str,
    account: str,
    *,
    closing_cash_cents: object,
    closing_positions: object,
) -> dict[str, object]:
    """Opening + activity = the PUBLISHED closing, for cash and per position,
    with every activity line traced to its entry id."""
    lines = [
        (e.entry_id, p)
        for e in entries
        if e.tenant == tenant and e.account == account
        for p in e.postings
    ]
    cash_open = opening.cents(tenant, account, CASH)
    cash_activity = sum(p.amount_cents for _, p in lines if p.account == CASH)
    shares = dict(opening.positions(tenant, account))
    for _, p in lines:
        if p.account.startswith(SECURITIES) and p.quantity is not None:
            ticker = p.account[len(SECURITIES) :]
            shares[ticker] = shares.get(ticker, 0) + p.quantity
    rolled = {t: q for t, q in sorted(shares.items()) if q}
    return {
        "cash_opening_cents": cash_open,
        "cash_activity_cents": cash_activity,
        "cash_closing_cents": closing_cash_cents,
        "activity_lines": len(lines),
        "traced_entries": sorted({entry_id for entry_id, _ in lines}),
        "ties": cash_open + cash_activity == closing_cash_cents and rolled == closing_positions,
    }


def run_checks(
    store: Store,
    tenant: str,
    account: str,
    *,
    day: str,
    journals: Mapping[str, Sequence[Entry]],
    locks: Mapping[str, Mapping[str, str]],
    view: Mapping[str, object],
    closes: SettledCloses | None = None,
) -> dict[str, object]:
    """The 1h checks for ``day`` (whose entries are already in ``journals``).

    ``closes`` (``(trading_day, tickers) -> {ticker: close}``) adds the daily
    settled-close completeness check; ``None`` skips it."""
    breaks: list[dict[str, object]] = []
    days = sorted(journals)
    month_start = f"{day[:7]}-01"
    # The previous trading day the ledger covers, and the names it held at
    # that day's end (alpha-engine-config-I13129 settled-close completeness).
    span = trading_days_between(days[0], day) if days else [day]
    prev = span[-2] if len(span) > 1 else None
    held_after: dict[str, list[str]] = {}

    # Roll-forward, today, and rebuild-from-raw: each earlier day of the open
    # month rebuilt from its journals must equal the view published for it.
    replay = Ledger()
    rolled: dict[str, object] = {}
    for d in days:
        if d == day:
            rolled = roll_forward(
                replay,
                journals[d],
                tenant,
                account,
                closing_cash_cents=view.get("ledger_cash_cents"),
                closing_positions=view.get("ledger_positions"),
            )
            if not rolled["ties"]:
                breaks.append({"check": "roll_forward", "day": d, "detail": "does not tie"})
        for entry in journals[d]:
            replay.append(entry)
        if prev is not None and d <= prev:
            held_after[d] = sorted(replay.positions(tenant, account))
        if month_start <= d < day:
            key = ledger_reconciliation_key(tenant, d)
            if not store.exists(key):
                continue  # a missing view is counted by completeness below
            stored = json.loads(store.get_bytes(key).decode("utf-8"))
            rebuilt = {
                "ledger_cash_cents": replay.cents(tenant, account, CASH),
                "ledger_positions": dict(sorted(replay.positions(tenant, account).items())),
                "trial_balance_cents": replay.trial_balance(),
            }
            published = {k: stored.get(k) for k in rebuilt}
            if stored.get("account") == account and published != rebuilt:
                breaks.append(
                    {"check": "rebuild_from_raw", "day": d, "detail": "view differs from journals"}
                )

    # Completeness: from the open month's start (or the first journal), every
    # trading day has a journal, a published view and an IB statement.
    first = max(month_start, days[0]) if days else day
    # ...and every name held at the end of each open-month trading day up to
    # the PREVIOUS one has its settled close. The day's own price panel is
    # compiled after this job runs (data/<day>/panel.parquet lands ~02:20Z the
    # next morning), so each run checks through the day before, and re-checks
    # every earlier day so an unresolved gap stays a break (aec#13237); the
    # first trading day of a month leaves the last day of the prior month to
    # the month-end close (`close_input_missing`).
    closes_day: str | None = None
    close_gaps: dict[str, list[str]] = {}
    if closes is not None and prev is not None and prev >= first:
        closes_day = prev
        held: list[str] = []
        for d in trading_days_between(days[0], prev):
            held = held_after.get(d, held)
            if d < first or not held:
                continue
            have: set[str]
            try:
                have = set(closes(d, held))
            except Exception:  # noqa: BLE001 - an unreadable panel is a missing close, never a raise
                have = set()
            close_gaps[d] = [f"close:{t}" for t in held if t not in have]
    missing: list[dict[str, object]] = []
    for d in trading_days_between(first, day):
        lacking = [
            name
            for name, present in (
                ("journal", d in journals),
                ("view", d == day or store.exists(ledger_reconciliation_key(tenant, d))),
                ("ib_statement", store.exists(trader_broker_statement_key(d))),
                ("reconciliation", d == day or store.exists(trader_reconciliation_key(d))),
            )
            if not present
        ] + close_gaps.get(d, [])
        if lacking:
            missing.append({"day": d, "missing": lacking})
            breaks.append({"check": "completeness", "day": d, "detail": f"missing {lacking}"})

    # Break aging: every gap not cleared by an adjustment citing it, oldest
    # first; older than GAP_EXPIRY_TRADING_DAYS trading days is overdue.
    cleared = {
        str(i)
        for d in days
        for e in journals[d]
        if e.kind == "adjustment"
        for i in cast("list[object]", e.evidence.get("exception_ids") or [])
    }
    open_breaks: list[dict[str, object]] = []
    for d in days:
        for e in journals[d]:
            if e.kind not in ("cash_gap", "position_gap") or e.account != account:
                continue
            ids = [str(i) for i in cast("list[object]", e.evidence.get("exception_ids") or [])]
            if cleared.intersection(ids):
                continue
            age = len(trading_days_between(d, day)) - 1
            open_breaks.append({"day": d, "entry_id": e.entry_id, "kind": e.kind, "age": age})
            if age > GAP_EXPIRY_TRADING_DAYS:
                breaks.append(
                    {"check": "break_aging", "day": d, "detail": f"{e.kind} open {age} days"}
                )

    # Period lock: a closed month's journals still hash to what was locked.
    for month, digests in sorted(locks.items()):
        current = {d: journal_digest(tenant, d, journals[d]) for d in days if d[:7] == month}
        if current != dict(digests):
            breaks.append({"check": "period_lock", "day": month, "detail": "closed month changed"})

    return {
        "roll_forward": rolled,
        "completeness": {
            "from": first,
            "through": day,
            "missing": missing,
            "closes_day": closes_day,
        },
        "open_breaks": sorted(open_breaks, key=lambda b: (b["day"], b["entry_id"])),
        "locked_months": sorted(locks),
        "breaks": breaks,
    }
