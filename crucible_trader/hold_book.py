"""The hold-book safeguard: a degenerate prediction batch holds the book automatically.

`alpha-engine-config-I10650` deliverable 1; plan §9.5. **Lifted from the v1
executor** (`crucible-executor/executor/main.py::_should_hold_book` and
`::_served_alpha_diagnostics`, frozen reference, not edited), keeping its
numbers and its decision names:

* `HOLD_BOOK_ALPHA_MODAL_FRACTION = 0.90` -- a batch whose most common value
  (rounded to 1e-9) covers >= 90% of names is degenerate -> `hold_signal_degenerate`;
* `_HOLD_BOOK_MIN_BATCH = 5` -- fewer finite alphas cannot be judged ->
  `hold_signal_undeterminable`;
* a non-finite alpha is a failure of the tradable signal ->
  `hold_tradable_signal_failed`.

**What is NOT lifted, and why** (the delta from "as-is"): v1 consulted the
predictor's `output_distribution_gate` verdict first and ran the scale-free test
only when that gate flagged. The v2 predictions feed (`predictions_feed.v1`)
carries no gate verdict -- the champion contract (`crucible.champion`) refuses a
champion whose run was not `ok` upstream of this -- so the scale-free test runs
on every batch. It needs no floor on `alpha_stdev`, which v1 had deliberately
stopped owning (alpha-engine-config-I10179/I10184), so running it
unconditionally re-introduces no threshold.

A hold engages the kill switch in `hold` mode (`crucible_trader.kill_switch`):
the halt document is written, the session sends no order for this trading day,
working orders (stops) are left alone -- the v1 hold's safety argument.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Callable, Mapping
from typing import Any

from crucible.keys import predictions_key
from crucible.runner import RunContext
from crucible.serving import PredictionsFeed

from crucible_trader.kill_switch import BrokerControl, KillSwitchOutcome, fire

HOLD_BOOK_ALPHA_MODAL_FRACTION = 0.90
HOLD_BOOK_MIN_BATCH = 5
METRIC_MODULE = "crucible_trader.hold_book"
HOLD_CAUSE_PREFIX = "degenerate_predictions"


@dataclasses.dataclass(frozen=True)
class HoldDecision:
    hold: bool
    decision: str
    diagnostics: Mapping[str, Any]


def served_alpha_diagnostics(predicted_alpha: Mapping[str, float]) -> tuple[list[float], dict]:
    """Finite served alphas plus their scale-free shape stats (v1, lifted)."""
    alphas: list[float] = []
    n_nonfinite = 0
    for value in predicted_alpha.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            n_nonfinite += 1
        elif math.isfinite(value):
            alphas.append(float(value))
        else:
            n_nonfinite += 1
    diag: dict[str, Any] = {"n_alpha": len(alphas), "n_nonfinite": n_nonfinite}
    if alphas:
        diag["alpha_stdev"] = round(statistics.pstdev(alphas), 6)
        rounded = [round(a, 9) for a in alphas]
        diag["alpha_modal_fraction"] = round(
            max(rounded.count(v) for v in set(rounded)) / len(rounded), 4
        )
        diag["n_unique_alpha"] = len(set(rounded))
    return alphas, diag


def should_hold_book(
    feed: PredictionsFeed,
    *,
    modal_fraction_ceiling: float = HOLD_BOOK_ALPHA_MODAL_FRACTION,
    min_batch: int = HOLD_BOOK_MIN_BATCH,
) -> HoldDecision:
    alphas, diag = served_alpha_diagnostics(feed.predicted_alpha)
    diag["alpha_modal_fraction_ceiling"] = modal_fraction_ceiling
    if diag["n_nonfinite"]:
        return HoldDecision(True, "hold_tradable_signal_failed", diag)
    if len(alphas) < min_batch:
        return HoldDecision(True, "hold_signal_undeterminable", diag)
    if diag["alpha_modal_fraction"] >= modal_fraction_ceiling:
        return HoldDecision(True, "hold_signal_degenerate", diag)
    return HoldDecision(False, "proceed_signal_healthy", diag)


def enforce_hold_book(
    ctx: RunContext,
    feed: PredictionsFeed,
    broker: BrokerControl,
    *,
    clock: Callable[[], Any] | None = None,
) -> tuple[HoldDecision, KillSwitchOutcome | None]:
    """Decide, and when the decision is a hold, engage it. The session calls this
    on every feed before construction; a hold then makes
    `kill_switch.assert_trading_permitted` refuse every order for the day."""
    decision = should_hold_book(feed)
    ctx.record_metric(
        {
            "name": "hold_book_engaged",
            "module": METRIC_MODULE,
            "metric_type": "operational",
            "value": 1.0 if decision.hold else 0.0,
            "unit": "flag",
            "n_floor": 1,
            "status": "OK",
            "status_reason": f"{decision.decision} on {feed.trading_day} ({feed.champion})",
            "source_path": predictions_key(feed.trading_day),
            "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if not decision.hold:
        return decision, None
    kwargs = {} if clock is None else {"clock": clock}
    outcome = fire(
        ctx,
        broker,
        mode="hold",
        cause=f"{HOLD_CAUSE_PREFIX}:{decision.decision}",
        **kwargs,
    )
    return decision, outcome


__all__ = [
    "HOLD_BOOK_ALPHA_MODAL_FRACTION",
    "HOLD_BOOK_MIN_BATCH",
    "HoldDecision",
    "enforce_hold_book",
    "served_alpha_diagnostics",
    "should_hold_book",
]
