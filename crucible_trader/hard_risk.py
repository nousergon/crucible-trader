"""The hard-risk pre-check: loss floor, drawdown ladder and re-entry lockout, from the broker.

v1 → v2 trading-rule parity audit (2026-11-09), rows 3, 4, 7 and 8. The rules
themselves are `crucible.hard_risk` -- pure functions of facts and the S slot's
`PortfolioParams` -- so a replay decides exactly as the trader did. This module
supplies the two things only the trader has:

* **the broker's facts**: each held name's average cost and mark, and the
  account's NAV (:func:`read_broker_facts`);
* **the trader's state across sessions**: the running peak NAV and the names
  still locked out of re-entry, carried in one immutable document per served
  session at :func:`hard_risk_key` and read back by :func:`carried_state` --
  the newest document strictly BEFORE the session, so a re-run of a session
  decides from the same state its first run did.

**Precedence, highest first: hard risk > eligibility > optimizer > timing.**
Every name this layer cuts or locks out reaches the construction as a FORCED
exit (`SessionInputs.hard_exit`), which `crucible.portfolio` pins to zero above
the U cut, above ``ineligible_held_exit: discretionary`` and inside the
mandatory turnover floor -- so the sale is never budget-limited. The drawdown
rung reaches it as `SessionInputs.buy_multiplier`. Then
:func:`enforce_precedence`, the session's one chokepoint before any order is
sent, refuses a plan that BUYs a name with a same-day hard exit or lockout, and
drops every BUY while the circuit breaker is engaged.

**A rule whose parameter is null is off**, and the whole layer is off for a
session with no broker (shadow, staging): there is no average cost or NAV to
evaluate it on, and none is invented.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Protocol, cast

from crucible.hard_risk import (
    DRAWDOWN_EXIT,
    LOCKOUT,
    LOSS_FLOOR_EXIT,
    DrawdownState,
    Holding,
    drawdown_state,
    lockout_through,
    loss_floor_breaches,
    select_drawdown_exits,
)
from crucible.portfolio import PortfolioParams
from crucible.store import Store

from crucible_trader.broker_statement import IbPaperStatementSource

__all__ = [
    "HARD_RISK_PREFIX",
    "HARD_RISK_SCHEMA_VERSION",
    "BrokerFacts",
    "CarriedState",
    "HardRiskDecision",
    "HardRiskError",
    "HardRiskInputs",
    "HardRiskPrecedenceError",
    "Lockout",
    "carried_state",
    "decide",
    "enforce_precedence",
    "hard_risk_key",
    "read_broker_facts",
]

HARD_RISK_SCHEMA_VERSION = "trader_hard_risk.v1"
HARD_RISK_PREFIX = "trader/hard_risk/"

#: How the NAV in :class:`BrokerFacts` was measured, recorded on every document.
NAV_SOURCE_CASH_PLUS_MARKS = "ib_total_cash_plus_marked_positions"


class HardRiskError(RuntimeError):
    """The hard-risk state cannot be read, or the broker's facts cannot be evaluated."""


class HardRiskPrecedenceError(HardRiskError):
    """A plan would BUY a name the hard-risk layer cut or locked out today."""


def hard_risk_key(trading_day: str) -> str:
    """The session's hard-risk record: peak NAV, rung, exits and live lockouts."""
    return f"{HARD_RISK_PREFIX}{trading_day}.json"


@dataclasses.dataclass(frozen=True)
class BrokerFacts:
    """What the broker says at decision time, for the hard-risk rules."""

    holdings: Mapping[str, Holding]
    nav_usd: float
    nav_source: str = NAV_SOURCE_CASH_PLUS_MARKS


@dataclasses.dataclass(frozen=True)
class Lockout:
    """A name a hard-risk exit cut, and the last session it may not be bought."""

    exit_day: str
    reason: str
    through: str

    def to_document(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CarriedState:
    """The trader's hard-risk state as of the newest session before this one."""

    peak_nav_usd: float | None
    lockouts: Mapping[str, Lockout]
    carried_from: str | None


@dataclasses.dataclass(frozen=True)
class HardRiskInputs:
    """What the session hands construction: the broker's facts and the carried state."""

    facts: BrokerFacts
    carried: CarriedState


@dataclasses.dataclass(frozen=True)
class HardRiskDecision:
    """This session's hard-risk outcome, recorded before any order is sent."""

    trading_day: str
    #: Every name pinned to zero this session, by reason: today's exits
    #: (loss floor, drawdown) and every live lockout.
    forced: Mapping[str, str]
    #: The subset of ``forced`` cut TODAY (each starts a lockout).
    new_exits: Mapping[str, str]
    lockouts: Mapping[str, Lockout]
    loss_from_cost: Mapping[str, float]
    peak_nav_usd: float
    nav_usd: float
    nav_source: str
    drawdown: DrawdownState | None
    carried_from: str | None

    @property
    def buy_multiplier(self) -> float:
        return 1.0 if self.drawdown is None else self.drawdown.multiplier

    @property
    def circuit_breaker(self) -> bool:
        return self.drawdown is not None and self.drawdown.circuit_breaker

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": HARD_RISK_SCHEMA_VERSION,
            "trading_day": self.trading_day,
            "forced": dict(sorted(self.forced.items())),
            "new_exits": dict(sorted(self.new_exits.items())),
            "lockouts": {t: lock.to_document() for t, lock in sorted(self.lockouts.items())},
            "loss_from_cost": dict(sorted(self.loss_from_cost.items())),
            "peak_nav_usd": self.peak_nav_usd,
            "nav_usd": self.nav_usd,
            "nav_source": self.nav_source,
            "drawdown": None if self.drawdown is None else self.drawdown.to_document(),
            "buy_multiplier": self.buy_multiplier,
            "circuit_breaker": self.circuit_breaker,
            "carried_from": self.carried_from,
        }


def carried_state(store: Store, trading_day: str) -> CarriedState:
    """The newest hard-risk record filed strictly before ``trading_day``.

    No record at all is the first session: no peak, no lockout. A record that
    cannot be read RAISES: a lockout list that silently came back empty would
    let the optimizer rebuy every name the layer just cut.
    """
    days = sorted(
        key[len(HARD_RISK_PREFIX) : -len(".json")]
        for key in store.list_keys(HARD_RISK_PREFIX)
        if key.endswith(".json")
    )
    earlier = [d for d in days if d < trading_day]
    if not earlier:
        return CarriedState(peak_nav_usd=None, lockouts={}, carried_from=None)
    day = earlier[-1]
    key = hard_risk_key(day)
    try:
        document = cast(dict[str, object], json.loads(store.get_bytes(key)))
        if document.get("schema_version") != HARD_RISK_SCHEMA_VERSION:
            raise ValueError(f"schema_version {document.get('schema_version')!r}")
        peak = float(cast(float, document["peak_nav_usd"]))
        raw = cast(dict[str, dict[str, str]], document["lockouts"])
        lockouts = {
            str(t): Lockout(
                exit_day=str(v["exit_day"]), reason=str(v["reason"]), through=str(v["through"])
            )
            for t, v in raw.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HardRiskError(
            f"{key} cannot be read as {HARD_RISK_SCHEMA_VERSION} ({exc}); the peak NAV and "
            "the re-entry lockouts it carries are not assumed away"
        ) from exc
    if not (math.isfinite(peak) and peak > 0):
        raise HardRiskError(f"{key} carries peak NAV {peak!r}, which is not a positive number")
    return CarriedState(peak_nav_usd=peak, lockouts=lockouts, carried_from=day)


def decide(
    params: PortfolioParams,
    inputs: HardRiskInputs,
    *,
    trading_day: str,
    alpha: Mapping[str, float],
) -> HardRiskDecision:
    """This session's forced exits, lockouts and drawdown rung. Pure; writes nothing.

    Order: the loss floor first, then the drawdown tier's forced exits (names
    already cut by the floor count toward the tier's total, as in v1), then
    the carried lockouts that still cover the session, then today's exits
    added as new lockouts. A name cut today is forced today whether or not
    the lockout is on: same-day precedence is not a parameter.
    """
    facts, carried = inputs.facts, inputs.carried
    day = dt.date.fromisoformat(trading_day)
    try:
        floor = loss_floor_breaches(facts.holdings, params)
        rung = drawdown_state(facts.nav_usd, carried.peak_nav_usd, params)
    except ValueError as exc:
        raise HardRiskError(
            f"the hard-risk rules cannot be evaluated on {trading_day}: {exc}"
        ) from exc
    new_exits: dict[str, str] = {t: LOSS_FLOOR_EXIT for t in floor}
    if rung is not None and rung.forced_exit_count:
        for ticker in select_drawdown_exits(
            facts.holdings, alpha, rung.forced_exit_count, already_exiting=new_exits
        ):
            new_exits[ticker] = DRAWDOWN_EXIT
    lockouts = {
        t: lock
        for t, lock in carried.lockouts.items()
        if day <= dt.date.fromisoformat(lock.through)
    }
    sessions = params.hard_risk_reentry_lockout_sessions
    if sessions is not None:
        for ticker, reason in new_exits.items():
            lockouts[ticker] = Lockout(
                exit_day=trading_day,
                reason=reason,
                through=lockout_through(day, sessions).isoformat(),
            )
    forced = {t: LOCKOUT for t in lockouts} | new_exits
    peak = (
        rung.peak_nav_usd
        if rung is not None
        else max(facts.nav_usd, carried.peak_nav_usd or facts.nav_usd)
    )
    return HardRiskDecision(
        trading_day=trading_day,
        forced=forced,
        new_exits=new_exits,
        lockouts=lockouts,
        loss_from_cost={
            t: loss for t, h in facts.holdings.items() if (loss := h.loss_from_cost) is not None
        },
        peak_nav_usd=float(peak),
        nav_usd=float(facts.nav_usd),
        nav_source=facts.nav_source,
        drawdown=rung,
        carried_from=carried.carried_from,
    )


def enforce_precedence(
    batches: Iterable[Sequence[object]],
    decision: HardRiskDecision | None,
    *,
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
) -> Iterable[Sequence[object]]:
    """The session's one chokepoint between the plan and the broker.

    A BUY of a name the hard-risk layer forced out today (cut or locked out)
    RAISES :class:`HardRiskPrecedenceError` before it is sent: the
    construction pins such a name to zero, so a BUY of it is a defect, not a
    market condition (v1's SMCI case, 2026-10-15). While the circuit breaker
    is engaged every BUY is DROPPED rather than refused, so the forced SELLS
    still go: the construction already caps every name at its held weight
    there, and a BUY can only come from a quote that moved between valuing the
    book and planning it.

    A plan handed over whole (a list or tuple of batches) is checked whole, so
    nothing is sent when any batch offends. A timed plan (a generator,
    `order_router.PaperOrderRouter` with entry timing) is never materialized --
    each step of it runs only after the batch before it was sent -- so each
    batch is checked as it is released; the live router already refuses such
    a BUY at plan time (`order_router.hard_risk_exits`), before any order.

    An order is read by its ``action`` (``BUY``/``SELL``) and
    ``decision.symbol``, the `order_router.PlannedOrder` surface.
    """
    if decision is None:
        return batches  # the layer is off: the router's batches, untouched
    if isinstance(batches, (list, tuple)):
        materialized = [tuple(batch) for batch in batches]
        for batch in materialized:
            _refuse_forced_buys(batch, decision)
        return [kept for batch in materialized if (kept := _admit(batch, decision, log))]
    return _admit_each(batches, decision, log)


def _admit_each(
    batches: Iterable[Sequence[object]],
    decision: HardRiskDecision,
    log: Callable[[str], None],
) -> Iterator[Sequence[object]]:
    for batch in batches:
        released = tuple(batch)
        _refuse_forced_buys(released, decision)
        kept = _admit(released, decision, log)
        if kept:
            yield kept


def _refuse_forced_buys(batch: Sequence[object], decision: HardRiskDecision) -> None:
    offending = sorted(
        _symbol(order) for order in batch if _is_buy(order) and _symbol(order) in decision.forced
    )
    if offending:
        raise HardRiskPrecedenceError(
            f"the plan for {decision.trading_day} BUYs {offending}, which the hard-risk layer "
            f"forced out today ({ {t: decision.forced[t] for t in offending} }). Hard risk "
            "outranks eligibility, the optimizer and timing; nothing is sent."
        )


def _admit(
    batch: Sequence[object], decision: HardRiskDecision, log: Callable[[str], None]
) -> tuple[object, ...]:
    """``batch`` as it may go: every BUY dropped while the circuit breaker is engaged."""
    if not decision.circuit_breaker:
        return tuple(batch)
    dropped = sorted(_symbol(o) for o in batch if _is_buy(o))
    if dropped:
        log(
            f"hard_risk: circuit breaker engaged on {decision.trading_day} (drawdown "
            f"{decision.drawdown.drawdown if decision.drawdown else float('nan'):.2%}); "
            f"BUY order(s) dropped: {dropped}"
        )
    return tuple(o for o in batch if not _is_buy(o))


def _is_buy(order: object) -> bool:
    return str(getattr(order, "action", "")).upper() == "BUY"


def _symbol(order: object) -> str:
    return str(cast(_Order, order).decision.symbol)


class _Decision(Protocol):
    symbol: str


class _Order(Protocol):
    decision: _Decision


class _Contract(Protocol):
    symbol: str
    secType: str  # noqa: N815 - SDK spelling


class _PositionRow(Protocol):
    account: str
    contract: _Contract
    position: float
    avgCost: float  # noqa: N815 - SDK spelling


class _IbClient(Protocol):  # the subset of `ib_insync.IB` read here
    def managedAccounts(self) -> list[str]: ...  # noqa: N802 - SDK spelling
    def positions(self) -> Iterable[_PositionRow]: ...
    def accountSummary(self) -> Iterable[object]: ...  # noqa: N802 - SDK spelling


def read_broker_facts(
    client: _IbClient, *, trading_day: str, quote: Callable[[str], float | None]
) -> BrokerFacts:
    """Average cost and mark per held name, and NAV, from a connected IB paper session.

    Positions and cash come from `IbPaperStatementSource` -- every refusal it
    owns (paper account only, stock only, whole shares, one cash row) applies
    here too. Average cost is IB's per-share ``avgCost`` on the same position
    row. The mark is the router's own decision-price source, so the floor is
    judged on the price the exit will be planned at. NAV is cash plus the
    marked positions.
    """
    statement = IbPaperStatementSource(client).read_statement(trading_day)
    avg_cost: dict[str, float] = {}
    for row in client.positions():
        if row.account == statement.account and row.contract.symbol in statement.positions:
            avg_cost[str(row.contract.symbol)] = float(row.avgCost)
    holdings: dict[str, Holding] = {}
    for symbol, shares in statement.positions.items():
        mark = quote(symbol)
        holdings[symbol] = Holding(shares=shares, avg_cost=avg_cost.get(symbol), mark=mark)
    if any(h.mark is None for h in holdings.values()):
        missing = sorted(s for s, h in holdings.items() if h.mark is None)
        raise HardRiskError(f"no mark for held {missing}; NAV is not valued on a partial book")
    nav = statement.cash_usd + sum(h.shares * float(h.mark or 0.0) for h in holdings.values())
    return BrokerFacts(holdings=holdings, nav_usd=nav)
