"""The sign-of-life order: at least one order on every trading day (alpha-engine-config-I14646).

Brian, 2026-11-27: *"We should place a minimum of one trade each market open
day, even if the trade is small. I want to see a sign of life!"* The v2 session
placed no order on 2026-11-23 or 2026-11-27: the conviction gate cut the day's
budget to 1% of the book, no wider than the 1% no-trade band, so every stepped
trade was banded away. Any future mix of gate, ladder, band and holds could zero
a day again, so this rule guarantees the order structurally. Brian's rule
holds throughout: risk controls scale participation down, never to zero, and
no risk check is weakened to make this work.

**When it fires.** After the target book is built and before it is routed
(`crucible_trader.session.session_cycle`): if the router's own plan for the
book (``plan``: `order_router.plan_orders` over the account's shares and
prices, sending nothing) has no order -- every change under one whole share or
inside the band the router holds -- the session adds exactly one order. A book
that already plans an order is returned unchanged.

**Which order.** The largest single step toward the aim, from what the
construction recorded: a DEFERRED ENTRY (``aim_adjustment.entries_deferred``,
a name whose stepped weight fell below ``min_position_pct``; its step is the
aim weight) or a BANDED CHANGE (``rebalance_band.held``; its step is the engine's
own target minus the held weight). Candidates are taken largest step first, and
the first that qualifies AND that the router then plans as a whole-share order
of that name is the order; a step the router would size to no order is passed
over for the next. Its size is the smallest of the full step,
``max(rebalance_band_pct, $1,000)`` of the book, and -- for a buy -- what the
hard-risk ladder allows (``buy_multiplier`` times the room under the per-name
cap, as `crucible.portfolio._name_caps` throttles it) and the cash above the
sleeve.

**The order is judged as the router will send it.** The router's order of the
name is taken only when its side is the candidate's own (a sell step the
router plans as a buy, because the held shares sit under the book's weight, is
passed over), when -- for a buy -- that very order passes every buy check
below at its own size, and when it trades no more than the rule sized.

**What it never does.** It never buys a name the hard-risk layer forced out or
locked out, a name the cut made ineligible, or a name with alpha below zero.
It never takes a name past ``max_position_pct`` and never steps past the
engine's own target. The book's turnover budget, and its recorded turnover,
are raised by exactly that one order's planned notional (after the router's
share floor and per-order cap), and the order is refused if the book would
otherwise exceed it.

**When nothing qualifies** it places nothing and says so: the
``sign_of_life_order`` metric is recorded ``unmeasurable`` with the reason, and
the close alert names it (`crucible_trader.box.market_alert`). When it fires,
the metric records the order (ticker, side, notional, reason).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

from crucible.keys import manifest_key

from crucible_trader.construction import TargetBook
from crucible_trader.order_router import PlannedOrder, RoutingRefusedError

__all__ = [
    "MIN_NOTIONAL_USD",
    "SIGN_OF_LIFE_METRIC",
    "Candidate",
    "Planner",
    "SignOfLifeOrder",
    "SignOfLifeOutcome",
    "apply_sign_of_life",
    "metric_record",
]

SIGN_OF_LIFE_METRIC = "sign_of_life_order"
METRIC_MODULE = "crucible_trader.sign_of_life"
_SESSION_JOB = "trader.session"

#: The smallest sign-of-life order, in dollars, whatever the band.
MIN_NOTIONAL_USD = 1_000.0

#: A weight change at or below this is no trade (the engine's float slack).
_DUST = 1e-6

Side = Literal["buy", "sell"]
Source = Literal["deferred_entry", "banded_change"]

#: The router's plan for a book, sending nothing (`PaperOrderRouter.plan`).
Planner = Callable[[TargetBook], Sequence[PlannedOrder]]


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One step toward the aim the construction recorded and did not trade."""

    ticker: str
    source: Source
    #: The signed weight change that would close the step in full.
    step: float


@dataclasses.dataclass(frozen=True)
class SignOfLifeOrder:
    """The one order the rule added."""

    ticker: str
    side: Side
    #: The book weight the router's order trades: its notional over the book.
    weight: float
    #: The router's planned order: its whole shares, and their notional at its price.
    shares: int
    notional_usd: float
    source: Source
    full_step: float
    reason: str

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SignOfLifeOutcome:
    """``fired``, ``not_needed`` or ``unmeasurable`` (nothing qualified), and why."""

    status: Literal["fired", "not_needed", "unmeasurable"]
    reason: str
    planned_orders: int
    order: SignOfLifeOrder | None = None

    def record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "planned_orders": self.planned_orders,
            "order": None if self.order is None else self.order.record(),
        }


def _sentinels(book: TargetBook) -> frozenset[str]:
    tickers = book.universe.tickers
    return frozenset({tickers[book.universe.benchmark_idx], tickers[book.universe.cash_idx]})


def _cash_ticker(book: TargetBook) -> str:
    return book.universe.tickers[book.universe.cash_idx]


def _section(book: TargetBook, part: str, key: str) -> object:
    record = book.optimizer or {}
    block = record.get(part)
    return cast(Mapping[str, object], block).get(key) if isinstance(block, Mapping) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _band(book: TargetBook) -> float:
    band = _number(_section(book, "settings", "rebalance_band_pct"))
    return band if band is not None and band > 0 else 0.0


def _candidates(book: TargetBook) -> list[Candidate]:
    """Every recorded step toward the aim, largest first (ties by ticker)."""
    sentinels = _sentinels(book)
    found: dict[str, Candidate] = {}
    deferred = _section(book, "solved", "aim_adjustment")
    names = (
        cast(Mapping[str, object], deferred).get("entries_deferred")
        if isinstance(deferred, Mapping)
        else None
    )
    aim = book.aim_weights or {}
    for ticker in _list(names):
        step = _number(aim.get(str(ticker)))
        if str(ticker) not in sentinels and step is not None and step > _DUST:
            found[str(ticker)] = Candidate(str(ticker), "deferred_entry", step)
    banded = _section(book, "solved", "rebalance_band")
    held = cast(Mapping[str, object], banded).get("held") if isinstance(banded, Mapping) else None
    for row in _list(held):
        if not isinstance(row, Mapping):
            continue
        entry = cast(Mapping[str, object], row)
        ticker, step = str(entry.get("ticker")), _number(entry.get("delta"))
        if ticker in sentinels or step is None or abs(step) <= _DUST:
            continue
        if ticker not in found or abs(step) > abs(found[ticker].step):
            found[ticker] = Candidate(ticker, "banded_change", step)
    return sorted(found.values(), key=lambda c: (-abs(c.step), c.ticker))


def _size(book: TargetBook, candidate: Candidate) -> tuple[float, str | None]:
    """The weight the order trades (positive), or ``(0, why it does not qualify)``."""
    ticker = candidate.ticker
    if ticker not in book.weights:
        return 0.0, "not in the book's universe"
    notional = book.portfolio_notional_usd
    size = min(abs(candidate.step), max(_band(book), MIN_NOTIONAL_USD / notional))
    if candidate.step < 0:
        return min(size, book.weights[ticker]), None
    limit, why = _buy_limit(book, ticker)
    return (0.0, why) if why is not None else (min(size, limit), None)


def _buy_limit(book: TargetBook, ticker: str) -> tuple[float, str | None]:
    """The most of ``ticker`` (a weight) the rule may buy, or ``(0, why no buy qualifies)``."""
    current = book.weights[ticker]
    decision = book.hard_risk
    if decision is not None and ticker in decision.forced:
        return 0.0, f"hard risk forced it out ({decision.forced[ticker]})"
    if not (book.eligible or {}).get(ticker, False):
        return 0.0, "ineligible"
    alpha = (book.alpha or {}).get(ticker)
    if alpha is None or not math.isfinite(alpha) or alpha < 0:
        return 0.0, f"alpha {alpha!r} is below zero or unknown"
    cap = _number(_section(book, "settings", "max_position_pct"))
    multiplier = 1.0 if decision is None else decision.buy_multiplier
    room = multiplier * max((1.0 if cap is None else cap) - current, 0.0)
    if room <= _DUST:
        return 0.0, f"no room under the cap at buy multiplier {multiplier:g}"
    sleeve = _number(_section(book, "settings", "cash_sleeve_pct")) or 0.0
    cash = book.weights.get(_cash_ticker(book), 0.0) - sleeve
    if cash <= _DUST:
        return 0.0, "no cash above the sleeve"
    return min(room, cash), None


def _stepped(book: TargetBook, candidate: Candidate, size: float) -> TargetBook:
    """``book`` with ``candidate`` stepped by ``size`` out of (or into) cash, and unbanded."""
    signed = math.copysign(size, candidate.step)
    weights = dict(book.weights)
    weights[candidate.ticker] += signed
    weights[_cash_ticker(book)] -= signed
    return dataclasses.replace(book, weights=weights, unbanded=book.unbanded | {candidate.ticker})


def _charged(book: TargetBook, weight: float) -> TargetBook:
    """``book`` with its turnover and its budget both raised by ``weight``, the order taken."""
    budget = book.turnover_budget
    return dataclasses.replace(
        book,
        turnover_one_way_ratio=book.turnover_one_way_ratio + weight,
        turnover_budget=None if budget is None else budget + weight,
    )


def _accepted(
    book: TargetBook, candidate: Candidate, size: float, order: PlannedOrder
) -> tuple[float, str | None]:
    """The book weight the router's ``order`` trades, and why the rule refuses it (``None``: taken).

    Judged on the order the router will really send, never on the step asked
    for: a routed BUY passes every buy check again at its own size, whatever
    the candidate's side; its side must be the candidate's; and it may trade
    no more than the rule sized.
    """
    side = order.decision.side
    wanted: Side = "buy" if candidate.step > 0 else "sell"
    price = order.decision.decision_price
    weight = int(order.decision.quantity) * price / book.portfolio_notional_usd
    limit = size
    if side == "buy":
        room, why = _buy_limit(book, candidate.ticker)
        if why is not None:
            return weight, f"the router plans a buy of it, refused: {why}"
        limit = min(limit, room)
    if side != wanted:
        return weight, f"the router plans a {side} of it, not the {wanted} its step asks for"
    if weight > limit + _DUST:
        return weight, (
            f"the router's {side} is {weight:.4%} of the book, over the {limit:.4%} the rule sized"
        )
    return weight, None


def _routed(plan: Planner, book: TargetBook, ticker: str) -> tuple[PlannedOrder | None, str | None]:
    """The router's order of ``ticker`` for ``book``, or ``(None, why it plans none)``."""
    try:
        orders = plan(book)
    except RoutingRefusedError as error:
        return None, f"the router refused it ({error})"
    for order in orders:
        if order.decision.symbol == ticker:
            return order, None
    return None, "the router plans no whole-share order for it"


def apply_sign_of_life(book: TargetBook, plan: Planner) -> tuple[TargetBook, SignOfLifeOutcome]:
    """``book`` with one sign-of-life order added when the router plans none, and what was decided.

    ``plan`` is the router's own plan (`PaperOrderRouter.plan`): the book plans
    an order when it returns one, and a candidate is the order only when the
    router plans a whole-share order of that name for the stepped book.
    """
    planned = len(plan(book))
    if planned:
        return book, SignOfLifeOutcome("not_needed", f"the plan has {planned} order(s)", planned)
    candidates = _candidates(book)
    if not candidates:
        return book, SignOfLifeOutcome(
            "unmeasurable",
            "the plan has no order and the construction recorded no deferred entry or banded "
            "change to step toward the aim",
            0,
        )
    budget = book.turnover_budget
    if budget is not None and book.turnover_one_way_ratio > budget + _DUST:
        # The order may raise the budget by itself only: a book already past
        # its budget gets nothing on top.
        return book, SignOfLifeOutcome(
            "unmeasurable",
            f"the plan has no order but the book's turnover {book.turnover_one_way_ratio:.4%} "
            f"is already over its budget {budget:.4%}",
            0,
        )
    refused: list[str] = []
    for candidate in candidates:
        size, why = _size(book, candidate)
        if why is None and size <= _DUST:
            why = "nothing left to trade"
        stepped, routed, weight = book, None, 0.0
        if why is None:
            stepped = _stepped(book, candidate, size)
            routed, why = _routed(plan, stepped, candidate.ticker)
        if routed is not None:
            weight, why = _accepted(book, candidate, size, routed)
        if routed is None or why is not None:
            refused.append(f"{candidate.ticker} {why}")
            continue
        side = routed.decision.side
        shares = int(routed.decision.quantity)
        order = SignOfLifeOrder(
            ticker=candidate.ticker,
            side=side,
            weight=weight,
            shares=shares,
            notional_usd=shares * routed.decision.decision_price,
            source=candidate.source,
            full_step=candidate.step,
            reason=(
                f"the plan had no order: {side} {candidate.ticker}, the largest step toward the "
                f"aim ({candidate.source.replace('_', ' ')}, full step {candidate.step:+.4%})"
                + (f"; passed over: {'; '.join(refused)}" if refused else "")
            ),
        )
        # The budget and the recorded turnover rise by the order as the router
        # sized it (share floor, per-order cap), not by the step asked for.
        return _charged(stepped, weight), SignOfLifeOutcome("fired", order.reason, 0, order)
    return book, SignOfLifeOutcome(
        "unmeasurable", "the plan has no order and no candidate qualified: " + "; ".join(refused), 0
    )


def metric_record(
    outcome: SignOfLifeOutcome, *, trading_day: str, now_utc: str
) -> dict[str, object]:
    """The ``sign_of_life_order`` MetricRecord: ``OK`` (not needed, fired) or ``unmeasurable``."""
    record: dict[str, object] = {
        "name": SIGN_OF_LIFE_METRIC,
        "module": METRIC_MODULE,
        "metric_type": "operational",
        "n_floor": 1,
        "status": "unmeasurable" if outcome.status == "unmeasurable" else "OK",
        "status_reason": f"{outcome.status}: {outcome.reason}",
        "source_path": manifest_key(_SESSION_JOB, trading_day),
        "last_updated_utc": now_utc,
        SIGN_OF_LIFE_METRIC: outcome.record(),
    }
    if outcome.status != "unmeasurable":
        record["value"] = 1.0 if outcome.order is not None else 0.0
        record["unit"] = "order"
    return record
