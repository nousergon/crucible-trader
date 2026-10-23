"""Implementation-shortfall decomposition: delay, execution, fees, opportunity.

`alpha-engine-config-I12566` (audit CR-06); SYSTEM_OPTIMIZED §6.1. The
`execution_shortfall.v1` row (:mod:`crucible_trader.execution_shortfall`)
measures one thing on purpose: the FILLED quantity's price against the
decision price. It names its own gap — the opportunity cost of an unfilled
remainder is not in the figure, there is no arrival observation, and no fee.
This module is the decomposition that closes those gaps, as a pure function
of what one order's lifecycle observed. It changes no routing, sizing or
order behaviour, and it does not alter the v1 document; carrying these
fields in the harness contract is a versioned change on the consumer side.

**Conventions, declared.** For an order with side sign ``s`` (+1 buy, -1
sell), intended quantity ``Q``, filled ``q``, unfilled ``u = Q - q``,
decision price ``P_d``, arrival price ``P_a`` (observed when the order
reached the market), fill VWAP ``P_f``, fees ``F`` (USD, positive is a
cost) and terminal mark ``P_m`` for the unfilled remainder::

    delay        = s * (P_a - P_d) * q     the market moved before we arrived
    execution    = s * (P_f - P_a) * q     what working the order cost
    fees         = F
    opportunity  = s * (P_m - P_d) * u     what not trading the rest cost
    total        = delay + execution + fees + opportunity

Positive is a cost. ``delay + execution`` equals the v1 filled-quantity
shortfall exactly, so the decomposition reconciles to the row already
published. The denominator for bps is the INTENDED notional ``Q * P_d`` —
the full decision, filled or not — so a no-fill order is not a zero-weight
order.

**Unknown is named, never zero.** A component whose input was not observed
is ``None`` and listed in ``unavailable``; the total is then ``None`` and
the order is ``complete: False``. A session with any incomplete order cannot
render a complete total. That is the closes-when's "unknown fees/arrival/mark
cannot render green", made structural.

**Fees are counted once.** :func:`tie_out_fees` compares the fees each order
carried here against the cash-fee ledger's per-order fees, so a fee present
in both is reconciled rather than double-counted, and one present in only
one of them is a named discrepancy.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Mapping, Sequence

from crucible_trader.execution_shortfall import SizingDecision

__all__ = [
    "CONVENTIONS",
    "TERMINAL_STATES",
    "ArrivalObservation",
    "ExecutionPiece",
    "OrderOutcome",
    "TerminalMark",
    "day_clustered_summary",
    "decompose",
    "session_decomposition",
    "tie_out_fees",
]

#: How an order ended at the broker. ``partial`` is a terminal partial fill
#: (the remainder expired or was cancelled after some quantity filled).
TERMINAL_STATES = ("filled", "partial", "cancelled", "rejected", "no_fill")

#: The declared conventions every document built here carries.
CONVENTIONS = {
    "sign": "positive_is_cost",
    "denominator": "intended_quantity_x_decision_price",
    "delay": "side * (arrival_price - decision_price) * filled_quantity",
    "execution": "side * (fill_vwap - arrival_price) * filled_quantity",
    "fees": "commission_usd, positive is a cost",
    "opportunity": "side * (terminal_mark - decision_price) * unfilled_quantity",
    "total": "delay + execution + fees + opportunity",
}

_COMPONENTS = ("delay", "execution", "fees", "opportunity")

# Tolerance for the reconciliation identities, in USD. The components are
# products of floats; anything larger than this is a defect, not rounding.
_RECONCILE_TOL_USD = 1e-6


@dataclasses.dataclass(frozen=True)
class ArrivalObservation:
    """The market price observed when the order reached the market."""

    price: float
    observed_at: dt.datetime

    def __post_init__(self) -> None:
        if not self.price > 0:
            raise ValueError(f"arrival price must be positive, got {self.price}")


@dataclasses.dataclass(frozen=True)
class ExecutionPiece:
    """One broker execution: a quantity at a price. Stable id from the broker."""

    exec_id: str
    quantity: float
    price: float

    def __post_init__(self) -> None:
        if not self.exec_id:
            raise ValueError("an execution names its broker exec_id")
        if not self.quantity > 0 or not self.price > 0:
            raise ValueError(
                f"execution {self.exec_id}: quantity and price must be positive, got "
                f"{self.quantity} @ {self.price}"
            )


@dataclasses.dataclass(frozen=True)
class TerminalMark:
    """The price the unfilled remainder is evaluated at, and what it is."""

    price: float
    basis: str  # e.g. "session_close", "next_open"
    horizon: str  # e.g. "same_session"

    def __post_init__(self) -> None:
        if not self.price > 0:
            raise ValueError(f"terminal mark must be positive, got {self.price}")
        if not self.basis or not self.horizon:
            raise ValueError("a terminal mark declares its basis and horizon")


@dataclasses.dataclass(frozen=True)
class OrderOutcome:
    """Everything one order's lifecycle observed, from decision to terminal state."""

    order_id: str
    decision: SizingDecision
    terminal_state: str
    executions: tuple[ExecutionPiece, ...] = ()
    arrival: ArrivalObservation | None = None
    commission_usd: float | None = None
    mark: TerminalMark | None = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("an order outcome names its order_id")
        if self.terminal_state not in TERMINAL_STATES:
            raise ValueError(
                f"order {self.order_id}: terminal_state is one of {TERMINAL_STATES}, "
                f"not {self.terminal_state!r}"
            )
        ids = [e.exec_id for e in self.executions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"order {self.order_id}: an execution id appears twice")
        filled = self.filled_quantity
        intended = self.decision.quantity
        if filled > intended + 1e-9:
            raise ValueError(f"order {self.order_id}: filled {filled} of {intended}")
        state = self.terminal_state
        consistent = {
            "filled": math.isclose(filled, intended),
            "partial": 0 < filled < intended and not math.isclose(filled, intended),
            "cancelled": filled < intended and not math.isclose(filled, intended),
            "rejected": filled == 0,
            "no_fill": filled == 0,
        }[state]
        if not consistent:
            raise ValueError(
                f"order {self.order_id}: terminal_state {state!r} contradicts filled "
                f"{filled} of {intended}"
            )
        if self.commission_usd is not None and self.commission_usd < 0:
            raise ValueError(f"order {self.order_id}: commission cannot be negative")

    @property
    def filled_quantity(self) -> float:
        return float(sum(e.quantity for e in self.executions))

    @property
    def unfilled_quantity(self) -> float:
        return max(float(self.decision.quantity) - self.filled_quantity, 0.0)

    @property
    def fill_vwap(self) -> float | None:
        filled = self.filled_quantity
        if not filled:
            return None
        return sum(e.quantity * e.price for e in self.executions) / filled


def decompose(outcome: OrderOutcome) -> dict[str, object]:
    """One order's shortfall components under :data:`CONVENTIONS`."""
    d = outcome.decision
    sign = 1.0 if d.side == "buy" else -1.0
    q = outcome.filled_quantity
    u = outcome.unfilled_quantity
    vwap = outcome.fill_vwap
    intended_notional = float(d.quantity) * float(d.decision_price)

    unavailable: list[str] = []
    components: dict[str, float | None] = {}

    # Filled half. With nothing filled, both are exactly zero whatever was
    # observed: no quantity moved, so no price could have cost anything.
    filled_vs_decision = sign * (vwap - d.decision_price) * q if vwap is not None else 0.0
    if q == 0:
        components["delay"] = 0.0
        components["execution"] = 0.0
    elif outcome.arrival is None:
        components["delay"] = None
        components["execution"] = None
        unavailable += ["arrival_price"]
    else:
        arrival = outcome.arrival.price
        components["delay"] = sign * (arrival - d.decision_price) * q
        components["execution"] = sign * (vwap - arrival) * q  # type: ignore[operator]

    # Fees: a rejected order or one that never filled pays nothing only if
    # the broker said so; an unreported commission is unknown, not free.
    if outcome.commission_usd is None:
        components["fees"] = None
        unavailable.append("commission_usd")
    else:
        components["fees"] = float(outcome.commission_usd)

    # Unfilled half.
    if u == 0:
        components["opportunity"] = 0.0
    elif outcome.mark is None:
        components["opportunity"] = None
        unavailable.append("terminal_mark")
    else:
        components["opportunity"] = sign * (outcome.mark.price - d.decision_price) * u

    complete = not unavailable
    total = sum(components[c] for c in _COMPONENTS) if complete else None  # type: ignore[misc]
    if complete:
        _assert_reconciles(outcome.order_id, components, filled_vs_decision)

    def _bps(value: float | None) -> float | None:
        return None if value is None else value / intended_notional * 1e4

    return {
        "order_id": outcome.order_id,
        "symbol": d.symbol,
        "side": d.side,
        "terminal_state": outcome.terminal_state,
        "intended_quantity": float(d.quantity),
        "filled_quantity": q,
        "unfilled_quantity": u,
        "decision_price": float(d.decision_price),
        "arrival_price": None if outcome.arrival is None else outcome.arrival.price,
        "fill_vwap": vwap,
        "terminal_mark": (
            None
            if outcome.mark is None
            else {
                "price": outcome.mark.price,
                "basis": outcome.mark.basis,
                "horizon": outcome.mark.horizon,
            }
        ),
        "exec_ids": [e.exec_id for e in outcome.executions],
        "intended_notional_usd": intended_notional,
        "filled_vs_decision_usd": filled_vs_decision,
        "components_usd": components,
        "components_bps": {c: _bps(v) for c, v in components.items()},
        "total_usd": total,
        "total_bps": _bps(total),
        "complete": complete,
        "unavailable": unavailable,
    }


def _assert_reconciles(
    order_id: str, components: Mapping[str, float | None], filled_vs_decision: float
) -> None:
    """``delay + execution`` must equal the v1 filled-quantity shortfall."""
    filled_half = float(components["delay"]) + float(components["execution"])  # type: ignore[arg-type]
    if abs(filled_half - filled_vs_decision) > _RECONCILE_TOL_USD:
        raise ArithmeticError(
            f"order {order_id}: delay + execution = {filled_half} but the filled quantity's "
            f"shortfall against the decision price is {filled_vs_decision}"
        )


def session_decomposition(outcomes: Sequence[OrderOutcome]) -> dict[str, object]:
    """Every order's decomposition and the session's gross and net totals.

    ``gross`` excludes fees, ``net`` includes them. Both are ``None`` when any
    order is incomplete — a session total over the orders that happened to be
    measurable would be the survivors' total, not the session's.
    """
    orders = [decompose(o) for o in outcomes]
    ids = [o["order_id"] for o in orders]
    if len(ids) != len(set(ids)):
        raise ValueError("an order id appears twice in one session")
    intended = sum(float(o["intended_notional_usd"]) for o in orders)  # type: ignore[arg-type]
    complete = all(o["complete"] for o in orders)
    # Quantity identity, per order: every intended share is filled or unfilled.
    for o in orders:
        if not math.isclose(
            float(o["filled_quantity"]) + float(o["unfilled_quantity"]),  # type: ignore[arg-type]
            float(o["intended_quantity"]),  # type: ignore[arg-type]
        ):
            raise ArithmeticError(f"order {o['order_id']}: filled + unfilled != intended")
    by_component: dict[str, float | None] = {}
    for c in _COMPONENTS:
        values = [o["components_usd"][c] for o in orders]  # type: ignore[index]
        by_component[c] = None if any(v is None for v in values) else float(sum(values))
    gross = net = None
    if complete:
        gross = sum(float(by_component[c]) for c in ("delay", "execution", "opportunity"))  # type: ignore[arg-type]
        net = gross + float(by_component["fees"])  # type: ignore[arg-type]
    return {
        "conventions": dict(CONVENTIONS),
        "orders": orders,
        "n_orders": len(orders),
        "intended_notional_usd": intended,
        "components_usd": by_component,
        "gross_usd": gross,
        "net_usd": net,
        "gross_bps": None if gross is None or not intended else gross / intended * 1e4,
        "net_bps": None if net is None or not intended else net / intended * 1e4,
        "complete": complete,
        "unavailable": sorted({u for o in orders for u in o["unavailable"]}),  # type: ignore[attr-defined]
    }


def day_clustered_summary(sessions: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Mean net shortfall in bps across sessions, with a DAY-clustered standard error.

    Orders inside one session share that day's market, so they are not
    independent draws; treating them as such overstates precision. The
    cluster is the session: one observation per day (its notional-weighted
    net bps), and the standard error is over days. Any incomplete session
    makes the summary incomplete rather than silently dropping that day.
    """
    days = sorted(sessions)
    incomplete = [d for d in days if not sessions[d]["complete"]]
    if incomplete:
        return {
            "n_days": len(days),
            "complete": False,
            "incomplete_days": incomplete,
            "mean_net_bps": None,
            "se_net_bps": None,
            "mean_gross_bps": None,
        }
    net = [float(sessions[d]["net_bps"]) for d in days]  # type: ignore[arg-type]
    gross = [float(sessions[d]["gross_bps"]) for d in days]  # type: ignore[arg-type]
    n = len(net)
    mean = sum(net) / n if n else None
    se = None
    if n >= 2:
        var = sum((x - mean) ** 2 for x in net) / (n - 1)  # type: ignore[operator]
        se = math.sqrt(var / n)
    return {
        "n_days": n,
        "complete": True,
        "incomplete_days": [],
        "mean_net_bps": mean,
        "se_net_bps": se,
        "mean_gross_bps": sum(gross) / n if n else None,
    }


def tie_out_fees(
    session: Mapping[str, object], ledger_fees_by_order: Mapping[str, float]
) -> dict[str, object]:
    """Reconcile the fees in ``session`` against the cash-fee ledger, per order.

    Returns the orders whose fee differs, the orders present on only one
    side, and whether the two agree. Agreement means each fee appears in the
    shortfall AND the ledger with the same amount, so a consumer summing net
    shortfall and the cash ledger together knows to count it once.
    """
    ours = {
        str(o["order_id"]): o["components_usd"]["fees"]  # type: ignore[index]
        for o in session["orders"]  # type: ignore[attr-defined]
    }
    mismatched = {
        oid: {"shortfall": ours[oid], "ledger": float(ledger_fees_by_order[oid])}
        for oid in sorted(set(ours) & set(ledger_fees_by_order))
        if ours[oid] is None
        or abs(float(ours[oid]) - float(ledger_fees_by_order[oid])) > _RECONCILE_TOL_USD
    }
    only_shortfall = sorted(set(ours) - set(ledger_fees_by_order))
    only_ledger = sorted(set(ledger_fees_by_order) - set(ours))
    return {
        "agrees": not mismatched and not only_shortfall and not only_ledger,
        "mismatched": mismatched,
        "only_in_shortfall": only_shortfall,
        "only_in_ledger": only_ledger,
    }
