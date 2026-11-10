"""The catastrophic gap stop: a held name priced 15% or more below its last close is sold whole.

v1 → v2 trading-rule parity audit (2026-11-09), row 2. v1's rule is
`crucible-executor` `executor/intraday_exit_manager.py:check_catastrophic_gap`:
EXIT a held name in full when its price is at least ``catastrophic_gap_stop_pct``
(0.15, `strategies/config.py`) below its ``gap_reference_price`` -- the most
recent close when the session was planned. It was the only intraday exit v1
kept under the optimizer, and it fired whatever the book wanted.

Here it runs when the session plans its orders (`order_router.PaperOrderRouter.
batches`, 09:35 ET): every held name is priced, and each one that breaches is
planned as a full SELL and any BUY of it is dropped. With timed orders (#150:
sells wait for an intraday exit trigger) a gap exit does NOT wait: it is the
session's first batch, sent at once; with immediate timing it leads the opening
sells. So it goes out through the same kill-switch-checked path as every other
sell, before any buy, and it overrides the book: a gap exit is hard risk, which
outranks eligibility, the optimizer and timing.

**The reference is the book day's close** from the price panel compiled for it
(`simulated_broker.PanelCloseSource`) -- the close the session was planned on,
exactly v1's ``gap_reference_price``. **The price must be live.** A name the
router could only price from that same close (`order_router.CloseFallbackQuote`)
has no measurable gap and is recorded ``unpriced``, never read as a 0% drop.

**Every check is recorded**, fired or not, at :func:`gap_stop_key` before any
order is sent -- the counterfactual drop of the names it watched and did not
stop is the cohort v1 never kept (config#846). An exit carries the reason
:data:`GAP_STOP_EXIT`, v1's own string, so the hard-risk layer's re-entry
lockout can read it as one more hard-exit reason.

**Declared scope.** It is evaluated once, when the session plans its orders. A name that
crosses the threshold later in the day is not re-checked here; the broker-side
stop is the intraday backstop.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence

from crucible.store import Store

from crucible_trader.execution_shortfall import SizingDecision, decide

__all__ = [
    "DEFAULT_GAP_STOP_PCT",
    "GAP_STOP_EXIT",
    "GAP_STOP_PREFIX",
    "GAP_STOP_SCHEMA_VERSION",
    "GapCheck",
    "apply_gap_exits",
    "check_gaps",
    "exit_decision",
    "gap_stop_document",
    "gap_stop_key",
    "write_gap_stop_document",
]

#: v1's reason string (`intraday_exit_manager.check_catastrophic_gap`).
GAP_STOP_EXIT = "catastrophic_gap_stop"

#: v1's `catastrophic_gap_stop_pct` (`strategies/config.py`).
DEFAULT_GAP_STOP_PCT = 0.15

GAP_STOP_SCHEMA_VERSION = "trader_gap_stop.v1"
GAP_STOP_PREFIX = "trader/gap_stop/"


def gap_stop_key(trading_day: str) -> str:
    dt.date.fromisoformat(trading_day)  # raises on anything but an ISO day
    return f"{GAP_STOP_PREFIX}{trading_day}.json"


@dataclasses.dataclass(frozen=True)
class GapCheck:
    """One held name, checked against its reference close."""

    symbol: str
    shares: int
    reference_close: float | None
    price: float | None
    #: ``(reference - price) / reference``; None when either side is missing.
    drop: float | None
    fired: bool
    #: ``fired``, ``held``, ``unpriced`` (no live price) or ``no_reference``.
    outcome: str

    def to_document(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _valid(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) and value > 0 else None


def check_gaps(
    positions: Mapping[str, int],
    *,
    reference: Callable[[str], float | None],
    live_price: Callable[[str], float | None],
    threshold: float,
) -> tuple[GapCheck, ...]:
    """Check every long position. Pure: sends and writes nothing.

    ``live_price`` returns None for a name with no live price; that name is
    ``unpriced`` and does not fire. ``reference`` returning None leaves the
    name ``no_reference`` (v1 likewise skipped a name with no valid reference).
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"gap stop threshold {threshold!r} must be in (0, 1)")
    checks: list[GapCheck] = []
    for symbol, shares in sorted(positions.items()):
        if shares <= 0:
            continue
        ref = _valid(reference(symbol))
        if ref is None:
            checks.append(GapCheck(symbol, shares, None, None, None, False, "no_reference"))
            continue
        price = _valid(live_price(symbol))
        if price is None:
            checks.append(GapCheck(symbol, shares, ref, None, None, False, "unpriced"))
            continue
        drop = (ref - price) / ref
        fired = drop >= threshold
        checks.append(
            GapCheck(symbol, shares, ref, price, drop, fired, "fired" if fired else "held")
        )
    return tuple(checks)


def gap_stop_document(
    trading_day: str,
    checks: Iterable[GapCheck],
    *,
    threshold: float,
    reference_source: str,
    now: dt.datetime,
) -> dict[str, object]:
    """The session's gap-stop record. ``exits`` is keyed by ticker, with its reason."""
    rows = list(checks)
    return {
        "schema_version": GAP_STOP_SCHEMA_VERSION,
        "trading_day": trading_day,
        "threshold": threshold,
        "reference_source": reference_source,
        "checked_at_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exits": {c.symbol: {"reason": GAP_STOP_EXIT, "drop": c.drop} for c in rows if c.fired},
        "checks": [c.to_document() for c in rows],
    }


def write_gap_stop_document(store: Store, trading_day: str, document: Mapping[str, object]) -> None:
    """Write the record. Never raises: a missing record must never stop an exit."""
    key = gap_stop_key(trading_day)
    try:
        store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - the exit still goes; the miss is printed
        print(
            f"gap_stop: the record {key} was not written ({type(exc).__name__}: {exc}); "
            "the exits it names are still sent",
            file=sys.stderr,
        )


def apply_gap_exits[O](
    sells: Sequence[O],
    buys: Sequence[O],
    checks: Iterable[GapCheck],
    *,
    symbol_of: Callable[[O], str],
    full_exit: Callable[[GapCheck], O],
) -> tuple[tuple[O, ...], tuple[O, ...]]:
    """``sells`` with every fired name sold whole, and ``buys`` without them.

    A planned partial sell of a fired name is replaced by the full exit; a
    fired name the book kept unchanged gains one; a planned BUY of it is dropped.
    """
    fired = {c.symbol: c for c in checks if c.fired}
    if not fired:
        return tuple(sells), tuple(buys)
    kept = [o for o in sells if symbol_of(o) not in fired]
    exits = [full_exit(fired[s]) for s in sorted(fired)]
    return (*exits, *kept), tuple(o for o in buys if symbol_of(o) not in fired)


def exit_decision(check: GapCheck, *, now: dt.datetime) -> SizingDecision:
    """The full-exit sizing decision for a fired check, at the price that fired it."""
    assert check.price is not None  # a fired check always carries its price
    return decide(check.symbol, "sell", check.shares, price=check.price, now=now)
