"""The month-end close of the paper fund: back office 1e.

`alpha-engine-config-I13122` (epic `-I12892`; nous-ergon-ops
`plans/2026-11-02-back-office-implementation.md` stage 1e), per
`policies/portfolio-accounting-policy.md`. It reads the v2 ledger (1g, the book
of record) and closes one month for one tenant's account:

* **positions** -- the ledger's month-end share counts, each valued at its
  settled close (``close_raw`` in the day's price panel, the Polygon close the
  rest of v2 marks to; Rule 7) and tied out to IB's statement for the month's
  last trading day;
* **cash** -- the ledger's month-end cash tied out to the same statement;
* **P&L** -- realised (average cost), the change in unrealised, dividends,
  accrued interest (its first recorded balance on its own line when the
  opening's accrual was not recorded), declared cash flows, commissions,
  suspense movements, the reconciliation adjustments that clear decided
  breaks and the cash rounded to IB inside the 5-cent band, which must sum
  to the change in NAV to the cent (the attribution check);
* **fees accrued** -- the commissions booked in the month. The simulated
  adviser and performance fees are 1f's bill, produced from this close
  (:mod:`crucible_trader.performance_statement`); they are never booked.

**Unexplained lines become exception records.** Each line the close cannot
explain is an ``exception.v1`` record from
:data:`~crucible_trader.exception_record.SOURCE_MONTH_END_CLOSE`, and the close
REFUSES TO FINISH while any exists (plan, Control standard: "the close refuses
to finish on any unexplained line"). A refused attempt is filed at
:func:`close_attempt_key` with its records and the close is tried again on the
next reconcile run. Each refused attempt also files the shadow proposer's
proposals for its records at
:func:`~crucible_trader.exception_proposal.close_proposals_key`, so the lines
wait in the same review queue as the daily breaks (`alpha-engine-config-I13121`).

**Period lock** (Rule 3). A finished close writes the month's
:func:`~crucible_trader.ledger.period_lock_document` (create-only), after which
step 5d refuses any journal dated in the month and the 1h checks turn a changed
closed journal into a break. A late fix is an ``adjustment`` dated in the open
month that cites the break's exception ids; a month-end suspense balance is
explained exactly when such an adjustment exists.

**Cost basis.** The ledger opens positions at zero cents (cost basis is not
established at opening, `crucible_trader.ledger`). The close carries a cost
basis of its own: a position held at the opening is costed at its settled
close on the opening statement's day, a buy adds its consideration, a sell
removes average cost. Each finished close stores its per-position cost basis,
and the next month starts from it, so months close in order: a month whose
previous month is in the ledger and not closed is refused.

**Prior-period books and the bridge** (`alpha-engine-config-I13232`). Every
step takes ``book`` (``None``: the live book, keys unchanged): a declared
prior-period book (:data:`~crucible_trader.ledger.PRIOR_PERIOD_BOOKS`) closes
from its own journals, views and prior close, and files its close, attempts and
period lock under its own root. The close of the LAST month such a book holds
carries a ``bridge``: the book's month-end balances against what the live book
OPENED with (:func:`~crucible_trader.ledger.opening_balances`) -- cash and each
accrued receivable to the cent, each position to the share, and the opening's
statement day against the month's last trading day. This is the roll-forward
across the cutover: the prior period's closing balance must be the next
period's opening balance. Every difference is itemised in the bridge and is an
unexplained line (``close_bridge_cash`` / ``close_bridge_shares``, or
``close_input_missing`` when the live book has no opening yet), so the book's
last month does not close, and is not period-locked, until the two books are
one continuous record.

**Failure isolation.** :func:`close_prior_month` is step 5f of
`trader.reconcile` and NEVER raises: it reports through one MetricRecord
(:data:`CLOSE_METRIC`) and changes nothing the job decides or pages.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any, cast

from crucible.keys import trader_broker_statement_key
from crucible.store import ETAG_ABSENT, PointerConflictError, Store

from crucible_trader.broker_statement import BrokerStatement, statement_from_document
from crucible_trader.exception_proposal import (
    PROPOSAL_SCHEMA_VERSION,
    close_proposals_key,
    proposal_day_document,
    propose_records,
)
from crucible_trader.exception_record import (
    PAPER_FUND_TENANT,
    SOURCE_MONTH_END_CLOSE,
    ExceptionRecord,
)
from crucible_trader.ledger import (
    CASH,
    CASH_ROUNDING,
    CLOSE_SUSPENSE_CASH,
    CLOSE_SUSPENSE_SHARES,
    COMMISSIONS,
    DECLARED_FLOWS,
    DIVIDEND_INCOME,
    GAP_EXPIRY_TRADING_DAYS,
    INTEREST_FIRST_RECORDED,
    INTEREST_INCOME,
    INTEREST_RECEIVABLE,
    OPENING,
    REALIZED_GAINS,
    RECEIVABLE,
    RECONCILIATION_ADJUSTMENTS,
    SECURITIES,
    SUSPENSE_CASH,
    SUSPENSE_POSITION,
    Entry,
    Ledger,
    LedgerError,
    book_final_month,
    book_root,
    journal_key,
    journal_prefix,
    ledger_from,
    ledger_reconciliation_key,
    load_journals,
    opening_balances,
    period_lock_document,
    period_lock_key,
    to_cents,
    trading_days_between,
)

CLOSE_SCHEMA_VERSION = "month_end_close.v1"
LEDGER_CLOSE_PREFIX = "trader/ledger/close/"
CLOSE_METRIC = "month_end_close"

#: ``(trading_day, tickers) -> {ticker: settled close in USD}``; a ticker with
#: no close is absent. Raises nothing for a missing panel: returns ``{}``.
CloseSource = Callable[[str, Sequence[str]], Mapping[str, float]]


class CloseError(ValueError):
    """A close that cannot even be attempted (bad month, wrong prior close)."""


def close_prefix(book: str | None = None) -> str:
    """:data:`LEDGER_CLOSE_PREFIX` for the live book, else the book's own."""
    return LEDGER_CLOSE_PREFIX if book is None else f"{book_root(book)}close/"


def close_key(tenant: str, month: str, book: str | None = None) -> str:
    """The finished close: written once, never rewritten."""
    _month_start(month)
    period_lock_key(tenant, month, book)  # validates the tenant and the book
    return f"{close_prefix(book)}{tenant}/{month}.json"


def close_attempt_key(tenant: str, month: str, run_day: str, book: str | None = None) -> str:
    """One refused attempt, as filed by the run on ``run_day``."""
    dt.date.fromisoformat(run_day)
    return f"{close_prefix(book)}{tenant}/{month}/attempts/{run_day}.json"


def previous_month(month: str) -> str:
    first = _month_start(month)
    return (first - dt.timedelta(days=1)).strftime("%Y-%m")


def month_trading_days(month: str) -> list[str]:
    first = _month_start(month)
    nxt = (first + dt.timedelta(days=32)).replace(day=1)
    return trading_days_between(first.isoformat(), (nxt - dt.timedelta(days=1)).isoformat())


def _month_start(month: str) -> dt.date:
    try:
        return dt.date.fromisoformat(f"{month}-01")
    except ValueError:
        raise CloseError(f"month {month!r} is not YYYY-MM") from None


@dataclasses.dataclass(frozen=True)
class CloseInputs:
    """Everything one close reads. ``journals`` holds every journal of the
    tenant up to the run day (later days are where prior-period adjustments
    live); ``view`` is the ledger's published view for the month's last
    trading day; ``prior`` the previous month's finished close, if any.

    ``book`` is the prior-period book being closed (``None``: the live book).
    ``bridge_to`` is what the live book opened with
    (:func:`~crucible_trader.ledger.opening_balances`), read when ``month`` is
    ``book``'s last (:func:`bridge_required`); ``None`` there means the live
    book has no opening yet, itself an unexplained line."""

    tenant: str
    account: str
    month: str
    run_day: str
    journals: Mapping[str, Sequence[Entry]]
    statement: BrokerStatement | None
    view: Mapping[str, object] | None
    prior: Mapping[str, object] | None
    closes: CloseSource
    detected_at: str
    book: str | None = None
    bridge_to: tuple[Ledger, Sequence[tuple[str, Entry]]] | None = None


@dataclasses.dataclass
class _Lot:
    shares: int = 0
    cost_cents: int = 0


def close_month(inputs: CloseInputs) -> dict[str, object]:
    """The close document for ``inputs.month``: ``status`` is ``closed`` when
    every line is explained, else ``refused`` with one exception record per
    unexplained line. Pure: no I/O."""
    tenant, account, month = inputs.tenant, inputs.account, inputs.month
    days = month_trading_days(month)
    first_day, last_day = days[0], days[-1]
    records: list[ExceptionRecord] = []
    statement_key = trader_broker_statement_key(last_day)
    view_key = ledger_reconciliation_key(tenant, last_day, inputs.book)

    def unexplained(
        klass: str,
        instrument: str | None,
        detail: str,
        *,
        amount: float | None = None,
        expected: float | None = None,
        actual: float | None = None,
        document_key: str = view_key,
        ref: str | None = None,
        actual_record: str | None = None,
    ) -> None:
        # ``ref`` tells apart lines that share a class and instrument -- two
        # failed checks of one name on different days, or the three missing
        # inputs -- which would otherwise share one exception id, and so one
        # proposal and one decision (alpha-engine-config-I13121).
        records.append(
            ExceptionRecord(
                source=SOURCE_MONTH_END_CLOSE,
                klass=klass,
                trading_day=last_day,
                account=account,
                instrument=instrument,
                side=None,
                amount=amount,
                expected_value=expected,
                expected_record=view_key if expected is not None else None,
                actual_value=actual,
                actual_record=(actual_record or statement_key) if actual is not None else None,
                detail=f"{month} close (run {inputs.run_day}): {detail}"[:500],
                document_key=document_key,
                void=False,
                detected_at=inputs.detected_at,
                identity=(account, month, instrument, *((ref,) if ref is not None else ())),
                tenant=tenant,
            )
        )

    ours = {
        d: tuple(e for e in entries if e.tenant == tenant and e.account == account)
        for d, entries in inputs.journals.items()
    }
    before = {d: v for d, v in ours.items() if d < first_day}
    within = {d: v for d, v in ours.items() if first_day <= d <= last_day}
    through = {**before, **within}
    opening = [e for d in sorted(within) for e in within[d] if e.kind == "opening_balance"]

    # ---- where the month starts: the prior close, or the opening in-month.
    lots: dict[str, _Lot] = {}
    marks_start: dict[str, float] = {}
    unpriced = False
    if opening:
        start_day = str(opening[0].evidence.get("statement_day") or opening[0].trading_day)
        start_cash = sum(p.amount_cents for p in opening[0].postings if p.account == CASH)
        held = {
            p.account[len(SECURITIES) :]: int(p.quantity or 0)
            for p in opening[0].postings
            if p.account.startswith(SECURITIES) and p.quantity
        }
        marks_start = dict(inputs.closes(start_day, sorted(held)))
        for ticker, shares in held.items():
            if ticker not in marks_start:
                unpriced = True
                unexplained(
                    "close_input_missing",
                    ticker,
                    f"no settled close for {ticker} on the opening day {start_day}",
                )
            lots[ticker] = _Lot(shares, _value(shares, marks_start.get(ticker, 0.0)))
        nav_start = start_cash + sum(lot.cost_cents for lot in lots.values())
    elif inputs.prior is not None:
        prior = inputs.prior
        if prior.get("month") != previous_month(month) or prior.get("status") != "closed":
            raise CloseError(f"the prior close is not {previous_month(month)}'s finished close")
        start_day = str(prior["last_trading_day"])
        carried = cast("Mapping[str, Mapping[str, int | float]]", prior["positions"])
        for ticker, row in carried.items():
            lots[ticker] = _Lot(int(row["shares"]), int(row["cost_basis_cents"]))
            marks_start[ticker] = float(row["close_usd"])
        nav_start = int(cast("int", prior["nav_end_cents"]))
    elif before:
        raise CloseError(
            f"{previous_month(month)} is in the ledger and not closed; months close in order"
        )
    else:
        return {"status": "nothing_to_close", "month": month, "tenant": tenant}
    unrealised_start = sum(
        _value(lot.shares, marks_start[t]) - lot.cost_cents
        for t, lot in lots.items()
        if t in marks_start
    )

    # ---- completeness: every trading day of the month the fund was open has
    # a journal, and the month-end view and statement exist.
    open_from = opening[0].trading_day if opening else first_day
    missing_days = [d for d in days if d >= open_from and d not in within]
    if missing_days:
        unexplained(
            "close_input_missing",
            None,
            f"no ledger journal for {missing_days}",
            ref="journals",
        )
    statement = inputs.statement
    if statement is None or statement.account != account:
        unexplained(
            "close_input_missing",
            None,
            f"no IB statement for {account} on {last_day}",
            document_key=statement_key,
            ref="statement",
        )
        statement = None
    if inputs.view is None:
        unexplained("close_input_missing", None, f"no ledger view for {last_day}", ref="view")
    else:
        for b in _breaks(inputs.view):
            if b.get("check") == "break_aging":
                continue  # suspense is judged below, counting later adjustments
            klass = (
                "close_input_missing"
                if b.get("check") == "completeness"
                else ("close_checks_failed")
            )
            unexplained(
                klass,
                str(b.get("check")),
                f"ledger check {b.get('check')} on {b.get('day')}: {b.get('detail')}",
                ref=str(b.get("day")),
            )

    # ---- walk the month: average cost, realised P&L, income and expense.
    realised = dividends = flows = commissions = suspense = adjustments = rounding = 0
    interest = interest_first_recorded = opening_accrual = 0
    for d in sorted(within):
        for entry in within[d]:
            if entry.kind == "opening_balance":
                continue
            for p in entry.postings:
                if p.account == COMMISSIONS:
                    commissions += p.amount_cents
                elif p.account == DECLARED_FLOWS:
                    flows -= p.amount_cents
                elif p.account == INTEREST_INCOME:
                    interest -= p.amount_cents
                elif p.account == INTEREST_FIRST_RECORDED:
                    interest_first_recorded -= p.amount_cents
                elif p.account == OPENING and entry.kind == "interest_accrual":
                    # the opening's accrued interest, booked as its own entry:
                    # opening equity, so it is part of the starting NAV. Any
                    # other credit to opening equity is an external flow the
                    # attribution has no line for, and refuses.
                    opening_accrual -= p.amount_cents
                elif p.account.startswith(DIVIDEND_INCOME):
                    dividends -= p.amount_cents
                elif p.account == SUSPENSE_CASH or p.account.startswith(SUSPENSE_POSITION):
                    suspense -= p.amount_cents
                elif p.account == RECONCILIATION_ADJUSTMENTS:
                    adjustments -= p.amount_cents
                elif p.account == CASH_ROUNDING:
                    rounding -= p.amount_cents
                elif p.account.startswith(REALIZED_GAINS):
                    # IB's realized figure, booked beside a holding relieved at
                    # IB's cost: with the securities leg below it sums to the
                    # proceeds, so the close's average-cost realised is unchanged.
                    realised -= p.amount_cents
                elif p.account.startswith(SECURITIES):
                    realised += _move(lots, p.account[len(SECURITIES) :], p, entry.kind)
    # A position that went to zero realises whatever cost it still carries.
    realised -= sum(lot.cost_cents for lot in lots.values() if not lot.shares)
    lots = {t: lot for t, lot in lots.items() if lot.shares}
    nav_start += opening_accrual

    # ---- month end: value, tie out to IB, suspense.
    ledger = ledger_from(through)
    ledger_positions = ledger.positions(tenant, account)
    # The lots replay the same postings as the ledger, so their share counts
    # are the ledger's; a cost that drifts shows in the attribution check.
    marks_end = dict(inputs.closes(last_day, sorted(lots))) if lots else {}
    positions: dict[str, dict[str, object]] = {}
    market_value = 0
    for ticker, lot in sorted(lots.items()):
        close = marks_end.get(ticker)
        if close is None:
            unexplained(
                "close_input_missing", ticker, f"no settled close for {ticker} on {last_day}"
            )
        value = _value(lot.shares, close) if close is not None else None
        market_value += value or 0
        positions[ticker] = {
            "shares": lot.shares,
            "close_usd": close,
            "market_value_cents": value,
            "cost_basis_cents": lot.cost_cents,
            "unrealised_cents": None if value is None else value - lot.cost_cents,
        }
    cash = ledger.cents(tenant, account, CASH)
    receivables = sum(ledger.balances(tenant, account, RECEIVABLE).values())
    interest_receivable = ledger.cents(tenant, account, INTEREST_RECEIVABLE)
    ib_cash = to_cents(statement.cash_usd) if statement is not None else None
    if statement is not None:
        if ib_cash != cash:
            unexplained(
                "close_cash_delta",
                "USD",
                f"ledger cash {cash} cents, IB {ib_cash} cents",
                amount=(cast("int", ib_cash) - cash) / 100,
                expected=cash / 100,
                actual=cast("int", ib_cash) / 100,
            )
        for ticker in sorted(set(ledger_positions) | set(statement.positions)):
            mine, theirs = ledger_positions.get(ticker, 0), int(statement.positions.get(ticker, 0))
            if mine != theirs:
                unexplained(
                    "close_share_delta",
                    ticker,
                    f"ledger {mine} shares, IB {theirs}",
                    amount=float(theirs - mine),
                    expected=float(mine),
                    actual=float(theirs),
                )

    cleared = {
        str(i)
        for d in ours
        for e in ours[d]
        if e.kind == "adjustment"
        for i in cast("list[object]", e.evidence.get("exception_ids") or [])
    }
    cleared_after: list[str] = []
    for d in sorted(through):
        for e in through[d]:
            if e.kind not in ("cash_gap", "position_gap"):
                continue
            ids = [str(i) for i in cast("list[object]", e.evidence.get("exception_ids") or [])]
            if cleared.intersection(ids):
                cleared_after.append(e.entry_id)
                continue
            for p in e.postings:
                if p.account == SUSPENSE_CASH:
                    unexplained(
                        CLOSE_SUSPENSE_CASH,
                        f"USD@{d}",
                        f"cash gap {e.entry_id} of {-p.amount_cents} cents booked {d} is not "
                        "cleared by any adjustment",
                        amount=-p.amount_cents / 100,
                    )
                elif p.account.startswith(SUSPENSE_POSITION) and p.quantity:
                    ticker = p.account[len(SUSPENSE_POSITION) :]
                    unexplained(
                        CLOSE_SUSPENSE_SHARES,
                        f"{ticker}@{d}",
                        f"share gap {e.entry_id} of {-p.quantity} {ticker} booked {d} is not "
                        "cleared by any adjustment",
                        amount=float(-p.quantity),
                    )

    nav_end = cash + receivables + interest_receivable + market_value
    unrealised_end = market_value - sum(lot.cost_cents for lot in lots.values())
    attribution = {
        "realised_cents": realised,
        "unrealised_change_cents": unrealised_end - unrealised_start,
        "dividends_cents": dividends,
        "interest_cents": interest,
        "interest_first_recorded_cents": interest_first_recorded,
        "declared_cash_flows_cents": flows,
        "commissions_cents": -commissions,
        "suspense_cents": suspense,
        "reconciliation_adjustments_cents": adjustments,
        "cash_rounded_to_broker_cents": rounding,
    }
    total = sum(attribution.values())
    priced = not unpriced and all(row["close_usd"] is not None for row in positions.values())
    if priced and total != nav_end - nav_start:
        unexplained(
            "close_attribution",
            None,
            f"attribution {total} cents, NAV change {nav_end - nav_start} cents",
            amount=(nav_end - nav_start - total) / 100,
        )
    bridge = _bridge(inputs, ledger, last_day, unexplained) if bridge_required(inputs) else None
    exceptions = [r.to_document() for r in records]
    return {
        **({} if inputs.book is None else {"book": inputs.book, "bridge": bridge}),
        "schema_version": CLOSE_SCHEMA_VERSION,
        "status": "refused" if exceptions else "closed",
        "tenant": tenant,
        "account": account,
        "month": month,
        "first_trading_day": first_day,
        "last_trading_day": last_day,
        "start_trading_day": start_day,
        "opened_in_month": bool(opening),
        "external_flows_cents": 0,
        "nav_start_cents": nav_start,
        "nav_end_cents": nav_end,
        "cash_cents": cash,
        "ib_cash_cents": ib_cash,
        "ib_accrued_interest_usd": statement.accrued_cash_usd if statement else None,
        "receivables_cents": receivables,
        "interest_receivable_cents": interest_receivable,
        "market_value_cents": market_value,
        "positions": positions,
        "pnl": {**attribution, "total_cents": total},
        "fees_accrued": {"commissions_cents": commissions},
        "trial_balance_cents": ledger.trial_balance(),
        "suspense_month_end": ledger.suspense(tenant, account),
        "suspense_cleared_after_month_end": cleared_after,
        "inputs": {
            "ib_statement": statement_key,
            "ledger_view": view_key,
            "prior_close": close_key(tenant, previous_month(month), inputs.book)
            if inputs.prior
            else None,
            "journal_days": sorted(within),
        },
        "exceptions": exceptions,
    }


def bridge_required(inputs: CloseInputs) -> bool:
    """Whether this close must bridge into the live book: it closes the last
    month of a prior-period book."""
    return inputs.book is not None and inputs.month == book_final_month(inputs.book)


def _balances(ledger: Ledger, tenant: str, account: str) -> dict[str, int]:
    """The balances a bridge compares: cash and each receivable in cents, each
    position in shares (``shares:<T>``). A securities account's cents are trade
    consideration, not a balance the next book carries (it opens them at zero),
    so they are not compared; suspense is the close's own lines."""
    out = {
        "cash_cents": ledger.cents(tenant, account, CASH),
        "interest_receivable_cents": ledger.cents(tenant, account, INTEREST_RECEIVABLE),
        **{
            f"dividends_receivable_cents:{name[len(RECEIVABLE) :]}": cents
            for name, cents in ledger.balances(tenant, account, RECEIVABLE).items()
        },
        **{f"shares:{t}": q for t, q in ledger.positions(tenant, account).items()},
    }
    return {k: v for k, v in out.items() if v or k == "cash_cents"}


def _bridge(
    inputs: CloseInputs,
    ledger: Ledger,
    last_day: str,
    unexplained: Callable[..., None],
) -> dict[str, object]:
    """The roll-forward across the cutover (`alpha-engine-config-I13232`): this
    book's balances at the end of ``last_day`` against the live book's opening.
    Files one unexplained line per difference and returns the itemised bridge."""
    tenant, account = inputs.tenant, inputs.account
    if inputs.bridge_to is None:
        unexplained(
            "close_input_missing",
            None,
            f"the live book holds no opening balance for {account} to bridge {inputs.month} into",
            ref="bridge",
        )
        return {"ties": False, "opening": None, "lines": [], "differences": []}
    opening, cited = inputs.bridge_to
    entry = next(e for _, e in cited if e.kind == "opening_balance")
    opening_day = str(entry.evidence.get("statement_day") or entry.trading_day)
    opening_key = journal_key(tenant, next(d for d, e in cited if e is entry))
    closing, opened = _balances(ledger, tenant, account), _balances(opening, tenant, account)
    lines: list[dict[str, object]] = []
    for item in sorted(set(closing) | set(opened)):
        mine, theirs = closing.get(item, 0), opened.get(item, 0)
        lines.append(
            {
                "item": item,
                "closing": mine,
                "opening": theirs,
                "difference": theirs - mine,
                "unit": "shares" if item.startswith("shares:") else "cents",
            }
        )
    differences = [line for line in lines if line["difference"]]
    if opening_day != last_day:
        unexplained(
            "close_checks_failed",
            "bridge",
            f"the live book opened from IB's {opening_day} statement, not this book's last "
            f"trading day {last_day}",
            document_key=opening_key,
            ref="statement_day",
        )
    for line in differences:
        item = str(line["item"])
        mine, theirs = int(cast("int", line["closing"])), int(cast("int", line["opening"]))
        what = item.split(":", 1)
        if what[0] == "shares":
            unexplained(
                "close_bridge_shares",
                what[1],
                f"{last_day} closes with {mine} {what[1]}; the live book opened with {theirs}",
                amount=float(theirs - mine),
                expected=float(mine),
                actual=float(theirs),
                actual_record=opening_key,
                ref="bridge",
            )
        else:
            name = item.replace("_cents", "")  # cash, interest_receivable, dividends_receivable:T
            unexplained(
                "close_bridge_cash",
                name,
                f"{last_day} closes with {name} {mine} cents; the live book opened with {theirs}",
                amount=(theirs - mine) / 100,
                expected=mine / 100,
                actual=theirs / 100,
                actual_record=opening_key,
                ref="bridge",
            )
    return {
        "ties": not differences and opening_day == last_day,
        "closing_day": last_day,
        "opening": {
            "statement_day": opening_day,
            "journal": opening_key,
            "entries": [e.entry_id for _, e in cited],
        },
        "lines": lines,
        "differences": differences,
    }


def _value(shares: int, close: float) -> int:
    return to_cents(Decimal(repr(close)) * shares)


def _move(lots: dict[str, _Lot], ticker: str, p: Any, kind: str) -> int:  # pyright: ignore[reportExplicitAny]
    """Apply one securities posting to the lots; return the realised cents."""
    lot = lots.setdefault(ticker, _Lot())
    quantity = int(p.quantity or 0)
    if kind == "fill" and quantity < 0 and lot.shares > 0:
        sold = min(-quantity, lot.shares)
        removed = (lot.cost_cents * sold) // lot.shares
        proceeds = -p.amount_cents * sold // -quantity
        lot.cost_cents -= removed
        lot.shares -= sold
        rest = -quantity - sold  # a sale past zero opens a short at its own price
        lot.cost_cents += (p.amount_cents + proceeds) if rest else 0
        lot.shares -= rest
        return proceeds - removed
    if kind == "fill" and quantity > 0 and lot.shares < 0:
        # Covering a short realises the credit received for the covered shares
        # less what buying them back cost (codex 416432a9d79c6d39).
        covered = min(quantity, -lot.shares)
        credit = (-lot.cost_cents * covered) // -lot.shares
        paid = p.amount_cents * covered // quantity
        lot.cost_cents += credit
        lot.shares += covered
        rest = quantity - covered  # a buy past zero opens a long at its own price
        lot.cost_cents += (p.amount_cents - paid) if rest else 0
        lot.shares += rest
        return credit - paid
    lot.shares += quantity
    lot.cost_cents += p.amount_cents
    return 0


def _breaks(view: Mapping[str, object]) -> list[Mapping[str, object]]:
    checks: object = view.get("checks")
    if not isinstance(checks, Mapping):
        return []
    breaks: object = cast("Mapping[str, object]", checks).get("breaks")
    return cast("list[Mapping[str, object]]", breaks or [])


# --------------------------------------------------------------------------- I/O


def panel_close_source(store: Store) -> CloseSource:
    """Settled closes from the day's price panel (``close_raw``)."""
    from crucible_trader.simulated_broker import PanelCloseSource  # noqa: PLC0415 - pandas

    def closes(day: str, tickers: Sequence[str]) -> Mapping[str, float]:
        source = PanelCloseSource(store, day)
        try:
            return {t: c for t in tickers if (c := source(t)) is not None}
        except Exception:  # noqa: BLE001 - an absent or bad panel is a missing close
            return {}

    return closes


def _read_json(store: Store, key: str) -> dict[str, Any] | None:  # pyright: ignore[reportExplicitAny]
    if not store.exists(key):
        return None
    return cast("dict[str, Any]", json.loads(store.get_bytes(key).decode("utf-8")))  # pyright: ignore[reportExplicitAny]


def _month_to_close(store: Store, tenant: str, day: str, book: str | None = None) -> str:
    """The oldest month before ``day``'s that has journals and no finished
    close, else the month before ``day``'s. Months close in order, so a month
    still refused when the calendar moves on is retried before the next one
    (codex 9270d56e) instead of the next one refusing on it forever."""
    prefix = journal_prefix(tenant, book)
    months = sorted(
        {
            k[len(prefix) : len(prefix) + 7]
            for k in store.list_keys(prefix)
            if k.endswith(".json") and k[len(prefix) : len(prefix) + 7] < day[:7]
        }
    )
    for candidate in months:
        if not store.exists(close_key(tenant, candidate, book)):
            return candidate
    return previous_month(day[:7])


def close_prior_month(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    *,
    day: str,
    account: str,
    now: str,
    tenant: str = PAPER_FUND_TENANT,
    closes: CloseSource | None = None,
    after_close: Callable[[Mapping[str, object]], object] | None = None,
    book: str | None = None,
) -> dict[str, object] | None:
    """Step 5f of `trader.reconcile`: close the oldest unclosed month before
    ``day``'s month (normally the month before it), then hand the finished close to ``after_close``
    (1f's statement). NEVER raises; returns the close document, or ``None``
    when there was nothing to do or it failed. ``book``: a declared
    prior-period book to close instead of the live one."""
    month = previous_month(day[:7])
    key = None
    try:
        store = ctx.store
        month = _month_to_close(store, tenant, day, book)
        key = close_key(tenant, month, book)
        finished = _read_json(store, key)
        if finished is None:
            finished = _attempt(ctx, store, tenant, account, month, day, now, closes, book)
            if finished is None:
                return None
            key = cast("str", finished.pop("_key"))
            if finished["status"] != "closed":
                n = len(cast("list[object]", finished["exceptions"]))
                # Lateness runs from the end of the month being CLOSED, not the
                # run day's month: an older month retried later is already
                # overdue (codex 3aac006d90588adc).
                due_from = (_month_start(month) + dt.timedelta(days=31)).replace(day=1)
                late = (
                    len(trading_days_between(due_from.isoformat(), day)) > GAP_EXPIRY_TRADING_DAYS
                )
                # One review queue for every book: a prior-period close's lines
                # wait beside the live ones (owner decision, aec#13232).
                filed = _file_proposals(ctx, finished, tenant=tenant, month=month, day=day, now=now)
                _metric(
                    ctx,
                    now,
                    key,
                    "FAIL" if late else "WATCH",
                    float(n),
                    f"{month} close refused on {day}: {n} unexplained line(s), {filed}; "
                    + "; ".join(
                        str(e["class"])
                        for e in cast("list[Mapping[str, object]]", finished["exceptions"])
                    )[:300],
                )
                return finished
            _metric(ctx, now, key, "OK", 0.0, f"{month} closed on {day}; period locked")
        if after_close is not None:
            after_close(finished)
        return finished
    except Exception as exc:  # noqa: BLE001 - must never change what the job decides or pages
        _metric(
            ctx,
            now,
            key,
            "FAIL",
            None,
            f"{month} close NOT attempted on {day}: {type(exc).__name__}: {exc}"[:400],
        )
        return None


def _attempt(
    ctx: Any,  # pyright: ignore[reportExplicitAny]
    store: Store,
    tenant: str,
    account: str,
    month: str,
    day: str,
    now: str,
    closes: CloseSource | None,
    book: str | None = None,
) -> dict[str, object] | None:
    inputs = stored_close_inputs(
        store,
        tenant=tenant,
        account=account,
        month=month,
        day=day,
        closes=closes or panel_close_source(store),
        now=now,
        book=book,
    )
    document = close_month(inputs)
    if document["status"] == "nothing_to_close":
        return None
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    if document["status"] != "closed":
        attempt = close_attempt_key(tenant, month, day, book)
        ctx.record_output(attempt, payload, CLOSE_SCHEMA_VERSION)
        return {**document, "_key": attempt}
    # Lock first: the lock is derived from immutable journals, so a re-run
    # after a failed close write rewrites the same bytes.
    in_month = {d: v for d, v in inputs.journals.items() if d[:7] == month}
    lock = json.dumps(
        period_lock_document(tenant, month, in_month), indent=2, sort_keys=True
    ).encode("utf-8")
    _create_once(ctx, period_lock_key(tenant, month, book), lock, "ledger_period_lock.v1")
    key = close_key(tenant, month, book)
    _create_once(ctx, key, payload, CLOSE_SCHEMA_VERSION)
    return {**document, "_key": key}


def stored_close_inputs(
    store: Store,
    *,
    tenant: str,
    account: str,
    month: str,
    day: str,
    closes: CloseSource,
    now: str,
    book: str | None = None,
) -> CloseInputs:
    """Everything :func:`close_month` reads for ``month`` as the run on ``day``
    finds it in ``book``, and, for a prior-period book's last month, what the
    live book opened with. The close (5f) and its rebuild (5g) both read
    through here, so a rebuild reads exactly what the close read."""
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    last_day = month_trading_days(month)[-1]
    statement_doc = _read_json(store, trader_broker_statement_key(last_day))
    inputs = CloseInputs(
        tenant=tenant,
        account=account,
        month=month,
        run_day=day,
        journals=load_journals(store, tenant, before=nxt, book=book),
        statement=statement_from_document(statement_doc) if statement_doc else None,
        view=_read_json(store, ledger_reconciliation_key(tenant, last_day, book)),
        prior=_read_json(store, close_key(tenant, previous_month(month), book)),
        closes=closes,
        detected_at=now,
        book=book,
    )
    if bridge_required(inputs):
        inputs = dataclasses.replace(
            inputs, bridge_to=opening_balances(store, tenant, account, book=None)
        )
    return inputs


def _file_proposals(
    ctx: Any,  # pyright: ignore[reportExplicitAny]
    refused: Mapping[str, object],
    *,
    tenant: str,
    month: str,
    day: str,
    now: str,
) -> str:
    """File the shadow proposals for a refused close's unexplained lines at
    :func:`~crucible_trader.exception_proposal.close_proposals_key`, so they
    join the review queue step 5e builds (`alpha-engine-config-I13121`).
    NEVER raises: a failure costs the proposals, never the close or its
    metric, and is named in the metric's reason. Returns that phrase."""
    try:
        records = cast("list[Mapping[str, object]]", refused["exceptions"])
        proposals = propose_records(records, proposed_at=now)
        document = proposal_day_document(cast("str", refused["last_trading_day"]), proposals)
        ctx.record_output(
            close_proposals_key(tenant, month, day),
            json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
            PROPOSAL_SCHEMA_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - shadow output must not change the close
        return f"proposals NOT filed ({type(exc).__name__}: {exc})"[:120]
    return f"{len(proposals)} proposal(s) filed for review"


def _create_once(ctx: Any, key: str, payload: bytes, schema: str) -> None:  # pyright: ignore[reportExplicitAny]
    try:
        ctx.record_output_cas(key, ETAG_ABSENT, payload, schema)
    except PointerConflictError:
        if ctx.store.get_bytes(key) != payload:
            raise LedgerError(f"{key} is already written with different bytes") from None


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
                "name": CLOSE_METRIC,
                "module": "crucible_trader.month_end_close",
                "metric_type": "operational",
                "value": value,
                "unit": "count" if value is not None else None,
                "n_floor": 0,
                "status": status,
                "status_reason": reason,
                "source_path": key or LEDGER_CLOSE_PREFIX,
                "last_updated_utc": now,
            }
        )
    except Exception:  # noqa: BLE001, S110 - a metric must not raise into the trader either
        pass


__all__ = [
    "CLOSE_METRIC",
    "CLOSE_SCHEMA_VERSION",
    "LEDGER_CLOSE_PREFIX",
    "CloseError",
    "CloseInputs",
    "close_attempt_key",
    "close_key",
    "close_month",
    "close_prior_month",
    "month_trading_days",
    "panel_close_source",
    "previous_month",
]
