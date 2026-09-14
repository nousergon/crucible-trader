"""Per-order implementation shortfall against the DECISION price.

`alpha-engine-config-I10652`; plan §4.5 (the fifth attribution row), §9.5
(execution quality; paper→live entry condition 2).

The trader is the only system that can measure this row, and it is the only
row that measures the trader at all. What is written here is the
`execution_shortfall.v1` document the harness declares
(`crucible.execution`), validated through the harness's own validator before
it is written — the schema, the key and the cross-field rules are the
consumer's, imported, never restated.

**The decision price is captured when the sizing decision is TAKEN.** A
:class:`SizingDecision` is the only thing an order can be built from, and it
is constructed with the price observed at that instant and the instant itself.
Order submission accepts no price: by the time an order is submitted, the
arrival price is a different number, and measuring against it would silently
excuse every cent the market moved between deciding and submitting — exactly
the delay implementation shortfall exists to charge. The basis is recorded in
the artifact (`decision_price_basis: sizing_decision`), not implied here.

**Three outcomes, never collapsed.**

    computed        orders placed, every filled order's shortfall measured
    no_orders       a fact: nothing was traded this session
    not_computed    orders placed and shortfall NOT measured -- a producer
                    failure. The document is written (so the attribution row
                    renders RED rather than absent) and the caller then exits
                    non-zero.

**Scope, declared:** the figure covers the filled quantity's execution price
against the decision price. The opportunity cost of an unfilled remainder is
not in the figure; unfilled orders are carried with a null shortfall so the
gap is visible. SOTA is the full Perold shortfall including that residual
marked to the session close; the delta is that it needs a close mark the
trader does not hold at fill time.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Sequence
from typing import Literal

from crucible.execution import (
    DECISION_PRICE_BASIS,
    EXECUTION_METRIC_NAME,
    EXECUTION_SHORTFALL_SCHEMA_VERSION,
    execution_shortfall_key,
    validate_execution_shortfall,
)
from crucible.store import Store

__all__ = [
    "SHORTFALL_BAND",
    "ShortfallBand",
    "Fill",
    "SizingDecision",
    "build_no_orders_document",
    "build_not_computed_document",
    "build_shortfall_document",
    "decide",
    "write_shortfall_document",
]

MODULE = "crucible_trader.execution_shortfall"

Side = Literal["buy", "sell"]


@dataclasses.dataclass(frozen=True)
class ShortfallBand:
    """The declared band the fifth row is graded against. Lower is better.

    ``placeholder`` is a required field, not an inference, for the reason
    `crucible.portfolio.CostModel.placeholder` is: a band nobody calibrated
    must say so on its face, or the row reads as graded against a tolerance
    that was measured.
    """

    baseline_bps: float
    upper_bps: float
    placeholder: bool

    def __post_init__(self) -> None:
        if not self.upper_bps > self.baseline_bps:
            raise ValueError(
                f"band upper_bps {self.upper_bps} must exceed baseline_bps {self.baseline_bps}"
            )

    def record(self) -> dict[str, float | bool]:
        return {
            "baseline_bps": float(self.baseline_bps),
            "upper_bps": float(self.upper_bps),
            "placeholder": bool(self.placeholder),
        }


#: Declared in phase 4 so phase 6 entry condition 2 ("with a declared band") is
#: readable rather than retroactively unmeetable. Baseline 0 bps is a fill at the
#: decision price. The 25 bps red line is a PLACEHOLDER: roughly the round-trip
#: charge a flat cost model with a 2.5 bps half-spread and 10 bps slippage
#: assumes, doubled — the level at which paper execution would be costing more
#: than twice what the grade charges. Replace it with a band calibrated from the
#: first full phase-4 window of measured shortfall, and flip ``placeholder``.
SHORTFALL_BAND = ShortfallBand(baseline_bps=0.0, upper_bps=25.0, placeholder=True)


def _utc_stamp(moment: dt.datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(
            f"{moment!r} is a naive datetime; a decision instant with no zone cannot be "
            "ordered against a submission instant from another clock"
        )
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclasses.dataclass(frozen=True)
class SizingDecision:
    """One sized order, frozen at the instant the decision was taken.

    Build it with :func:`decide`, from the sizing step. Nothing downstream can
    change ``decision_price``: submission takes a decision and a clock, never
    a price.
    """

    symbol: str
    side: Side
    quantity: float
    decision_price: float
    decided_at: dt.datetime

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("a sizing decision names its symbol")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if not self.quantity > 0:
            raise ValueError(f"{self.symbol}: quantity must be positive, got {self.quantity}")
        if not self.decision_price > 0:
            raise ValueError(
                f"{self.symbol}: decision price must be positive, got {self.decision_price}; "
                "a sizing decision taken without a price has nothing to measure shortfall from"
            )
        _utc_stamp(self.decided_at)


def decide(
    symbol: str, side: Side, quantity: float, *, price: float, now: dt.datetime
) -> SizingDecision:
    """Record a sizing decision with the price observed AT the decision instant."""
    return SizingDecision(
        symbol=symbol, side=side, quantity=quantity, decision_price=price, decided_at=now
    )


@dataclasses.dataclass(frozen=True)
class Fill:
    """What became of one decision: when it was submitted, and what filled at what price."""

    order_id: str
    decision: SizingDecision
    submitted_at: dt.datetime
    filled_quantity: float
    fill_price: float | None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("a fill names its order_id")
        if self.submitted_at < self.decision.decided_at:
            raise ValueError(
                f"order {self.order_id}: submitted at {self.submitted_at.isoformat()} before "
                f"its sizing decision at {self.decision.decided_at.isoformat()}"
            )
        if self.filled_quantity < 0 or self.filled_quantity > self.decision.quantity:
            raise ValueError(
                f"order {self.order_id}: filled {self.filled_quantity} of {self.decision.quantity}"
            )
        if (self.filled_quantity > 0) != (self.fill_price is not None):
            raise ValueError(
                f"order {self.order_id}: a fill price is present iff quantity filled "
                f"(filled {self.filled_quantity}, price {self.fill_price})"
            )
        if self.fill_price is not None and not self.fill_price > 0:
            raise ValueError(f"order {self.order_id}: fill price must be positive")

    def record(self) -> dict[str, object]:
        d = self.decision
        base: dict[str, object] = {
            "order_id": self.order_id,
            "symbol": d.symbol,
            "side": d.side,
            "quantity": float(d.quantity),
            "decided_at_utc": _utc_stamp(d.decided_at),
            "decision_price": float(d.decision_price),
            "submitted_at_utc": _utc_stamp(self.submitted_at),
            "filled_quantity": float(self.filled_quantity),
        }
        if self.fill_price is None:
            return {
                **base,
                "fill_price": None,
                "notional_usd": 0.0,
                "decision_notional_usd": 0.0,
                "shortfall_bps": None,
                "shortfall_usd": 0.0,
            }
        sign = 1.0 if d.side == "buy" else -1.0
        move = self.fill_price - d.decision_price
        return {
            **base,
            "fill_price": float(self.fill_price),
            "notional_usd": float(self.filled_quantity * self.fill_price),
            "decision_notional_usd": float(self.filled_quantity * d.decision_price),
            "shortfall_bps": float(sign * move / d.decision_price * 1e4),
            "shortfall_usd": float(sign * move * self.filled_quantity),
        }


def _metric(
    *,
    trading_day: str,
    value: float | None,
    n_filled: int,
    band: ShortfallBand,
    now: dt.datetime,
) -> dict[str, object]:
    """The session's MetricRecord. Session-level status only: the WEEK is graded
    by `crucible.report` over its own window, so this row states what one day
    measured and makes no claim about the band's verdict."""
    return {
        "name": EXECUTION_METRIC_NAME,
        "module": MODULE,
        "metric_type": "ratio",
        "value": value,
        "unit": "bps" if value is not None else None,
        "n_samples": n_filled,
        "n_floor": 1,
        "target": band.baseline_bps,
        "red_line": band.upper_bps,
        "status": "N/A-LOW-N" if value is None else "OK",
        "status_reason": (
            f"{trading_day}: notional-weighted implementation shortfall over {n_filled} "
            f"filled order(s) against the sizing-decision price"
            if value is not None
            else f"{trading_day}: orders placed, none filled — no execution price to measure"
        ),
        "source_path": execution_shortfall_key(trading_day),
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _envelope(
    *, trading_day: str, champion: str, band: ShortfallBand, now: dt.datetime
) -> dict[str, object]:
    return {
        "schema_version": EXECUTION_SHORTFALL_SCHEMA_VERSION,
        "trading_day": trading_day,
        "calendar_date": now.astimezone(dt.UTC).date().isoformat(),
        "champion": champion,
        "decision_price_basis": DECISION_PRICE_BASIS,
        "shortfall_scope": "filled_quantity_vs_decision_price",
        "band": band.record(),
    }


def build_shortfall_document(
    *,
    trading_day: str,
    champion: str,
    fills: Sequence[Fill],
    now: dt.datetime,
    band: ShortfallBand = SHORTFALL_BAND,
) -> dict[str, object]:
    """The `computed` document for a session in which orders were placed."""
    if not fills:
        raise ValueError(
            f"{trading_day}: no fills to measure. A session with no orders is "
            "`build_no_orders_document` — a fact — never a computed zero"
        )
    orders = [f.record() for f in fills]
    notional = sum(float(o["notional_usd"]) for o in orders)  # type: ignore[arg-type]
    decision_notional = sum(float(o["decision_notional_usd"]) for o in orders)  # type: ignore[arg-type]
    shortfall = sum(float(o["shortfall_usd"]) for o in orders)  # type: ignore[arg-type]
    n_filled = sum(1 for f in fills if f.filled_quantity > 0)
    # Weighted by DECISION notional, so the figure over identical orders equals
    # each order's own bps (whose denominator is the decision price).
    weighted = shortfall / decision_notional * 1e4 if decision_notional > 0 else None
    document = {
        **_envelope(trading_day=trading_day, champion=champion, band=band, now=now),
        "outcome": "computed",
        "outcome_reason": None,
        "orders": orders,
        "summary": {
            "n_orders": len(orders),
            "n_filled": n_filled,
            "notional_usd": notional,
            "decision_notional_usd": decision_notional,
            "shortfall_usd": shortfall,
            "shortfall_bps_notional_weighted": weighted,
        },
        "metrics": [
            _metric(trading_day=trading_day, value=weighted, n_filled=n_filled, band=band, now=now)
        ],
    }
    return validate_execution_shortfall(document, origin=execution_shortfall_key(trading_day))


def build_no_orders_document(
    *,
    trading_day: str,
    champion: str,
    reason: str,
    now: dt.datetime,
    band: ShortfallBand = SHORTFALL_BAND,
) -> dict[str, object]:
    """A session the trader ran and in which it placed nothing. Emitted, not skipped:
    a zero-order session that writes nothing is graded as a miss."""
    document = {
        **_envelope(trading_day=trading_day, champion=champion, band=band, now=now),
        "outcome": "no_orders",
        "outcome_reason": reason,
        "orders": [],
        "summary": {
            "n_orders": 0,
            "n_filled": 0,
            "notional_usd": 0.0,
            "decision_notional_usd": 0.0,
            "shortfall_usd": 0.0,
            "shortfall_bps_notional_weighted": None,
        },
        "metrics": [],
    }
    return validate_execution_shortfall(document, origin=execution_shortfall_key(trading_day))


def build_not_computed_document(
    *,
    trading_day: str,
    champion: str,
    reason: str,
    fills: Sequence[Fill] = (),
    now: dt.datetime,
    band: ShortfallBand = SHORTFALL_BAND,
) -> dict[str, object]:
    """Orders were placed and shortfall could not be measured — a PRODUCER FAILURE.

    Written so the attribution row renders RED naming ``reason``; the caller
    must still exit non-zero after writing it.
    """
    document = {
        **_envelope(trading_day=trading_day, champion=champion, band=band, now=now),
        "outcome": "not_computed",
        "outcome_reason": reason,
        "orders": [f.record() for f in fills],
        "summary": None,
        "metrics": [],
    }
    return validate_execution_shortfall(document, origin=execution_shortfall_key(trading_day))


def write_shortfall_document(store: Store, document: dict[str, object]) -> str:
    """Re-validate and write under the harness-declared key. Returns the key."""
    key = execution_shortfall_key(str(document["trading_day"]))
    validate_execution_shortfall(document, origin=key)
    store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))
    return key
