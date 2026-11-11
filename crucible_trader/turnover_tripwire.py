"""The rolling turnover tripwire: a churn alarm that names its driver.

v1 → v2 trading-rule parity audit (2026-11-09), row 16
(`crucible-executor` `executor/turnover_tripwire.py:check_turnover_tripwire`).
v2 recorded each session's one-way turnover against its budget
(`daily_session._record_turnover`), but only the staging soak gate graded it,
and nothing looked at a WINDOW. Churn by many small moves -- every day under
its own budget, the week's sum abnormal -- was therefore invisible in prod,
and the 11-03 to 11-05 rebuild-from-cash days (97% one-way) were found by a
person.

**Two bands, checked on every served session, both from recorded facts:**

* **daily** -- the session's one-way turnover above its own budget (the
  discretionary budget plus the forced exits it explains,
  :func:`crucible_trader.construction.session_turnover_budget`) times
  ``daily_multiple``. v1 multiplied the configured cap; v2's budget already
  forgives forced exits, so a breach here is unexplained turnover, never a hard
  exit the rules demanded.
* **rolling** -- the sum over the last ``rolling_sessions`` served sessions
  (this one included, the earlier ones read from their `trader.session`
  manifests) above ``rolling_sum_band``.

**The block carries its driver** (v1 `alpha-engine-config-I9315`): the window
is split into forced and discretionary turnover and the dominant cause is named
out of a closed set, with ``unattributed`` for a combination it does not know
-- a new failure mode is reported as new, never rounded to the nearest known
one. The attribution runs on every session, breach or not, so a condition can
be watched building.

**Surface.** One MetricRecord on the session's own manifest,
``trader_turnover_tripwire``: status ``BREACH`` on either band, ``OK``
otherwise, ``unmeasurable`` when the session recorded no turnover or the
parameters are not declared. The values are strategy edge and live in the
private strategy tree's `strategy/slots/s.yaml` (``turnover_tripwire:``
block); this repository carries none.

**Posture.** Secondary observability on the session's primary path: it never
raises. A failure is RECORDED as an ``unmeasurable`` metric naming the error,
so a dead tripwire is itself visible rather than silent.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import yaml
from crucible.calendar import previous_trading_day
from crucible.keys import manifest_key, strategy_slot_key
from crucible.store import Store

__all__ = [
    "METRIC_NAME",
    "TRIPWIRE_BLOCK",
    "TripwireConfig",
    "TripwireConfigError",
    "check",
    "driver_row",
    "load_config",
    "metric_record",
]

#: The metric this module files on every served `trader.session` run.
METRIC_NAME = "trader_turnover_tripwire"

#: The top-level block of `strategy/slots/s.yaml` the parameters live in.
TRIPWIRE_BLOCK = "turnover_tripwire"

_FIELDS: tuple[str, ...] = ("daily_multiple", "rolling_sessions", "rolling_sum_band")

#: The manifest job whose prior runs make up the rolling window.
_SESSION_JOB = "trader.session"

#: How many trading days back the window search may reach
#: for a session that served nothing (a held day files no turnover). Bounded
#: so a long outage cannot make the read unbounded.
_SEARCH_SESSIONS = 30


class TripwireConfigError(ValueError):
    """The tripwire block is present but malformed."""


@dataclasses.dataclass(frozen=True)
class TripwireConfig:
    """The tripwire's three parameters. Every one is required; none has a default."""

    daily_multiple: float
    rolling_sessions: int
    rolling_sum_band: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.daily_multiple) and self.daily_multiple >= 1.0):
            raise TripwireConfigError(
                f"daily_multiple must be >= 1; got {self.daily_multiple}. Below 1 the band "
                "pages on a session that stayed inside its own budget."
            )
        if self.rolling_sessions < 2:
            raise TripwireConfigError(
                f"rolling_sessions must be >= 2; got {self.rolling_sessions}. One session is "
                "the daily band."
            )
        if not (math.isfinite(self.rolling_sum_band) and self.rolling_sum_band > 0.0):
            raise TripwireConfigError(
                f"rolling_sum_band must be positive; got {self.rolling_sum_band}"
            )

    @classmethod
    def from_mapping(cls, payload: object, *, source: str) -> TripwireConfig:
        if not isinstance(payload, Mapping):
            raise TripwireConfigError(f"{source}: {TRIPWIRE_BLOCK} must be a mapping")
        block = cast(Mapping[str, object], payload)
        missing = sorted(set(_FIELDS) - set(block))
        unknown = sorted(set(block) - set(_FIELDS))
        if missing or unknown:
            raise TripwireConfigError(
                f"{source}: {TRIPWIRE_BLOCK} is missing {missing} / declares unknown {unknown}; "
                f"it takes exactly {list(_FIELDS)}"
            )
        rolling = block["rolling_sessions"]
        if isinstance(rolling, bool) or not isinstance(rolling, int):
            raise TripwireConfigError(f"{source}: rolling_sessions must be an integer")
        numbers: dict[str, float] = {}
        for name in ("daily_multiple", "rolling_sum_band"):
            raw = block[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TripwireConfigError(f"{source}: {name} must be a number; got {raw!r}")
            numbers[name] = float(raw)
        return cls(
            daily_multiple=numbers["daily_multiple"],
            rolling_sessions=rolling,
            rolling_sum_band=numbers["rolling_sum_band"],
        )


def load_config(store: Store) -> TripwireConfig | None:
    """The tripwire parameters from the S slot's parameter file, or None when undeclared.

    Undeclared is not a default: :func:`check` records it as ``unmeasurable``,
    named, every session, so an absent block is never read as a quiet week.
    """
    key = strategy_slot_key("s")
    payload = yaml.safe_load(store.get_bytes(key).decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TripwireConfigError(f"{key}: expected a mapping")
    block = cast(Mapping[str, object], payload).get(TRIPWIRE_BLOCK)
    if block is None:
        return None
    return TripwireConfig.from_mapping(block, source=key)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def driver_row(trading_day: str, result: Mapping[str, object]) -> dict[str, object]:
    """One session's turnover reduced to the facts that EXPLAIN it.

    ``result`` is the session document (`crucible_trader.session`), whose
    ``turnover_drivers`` carry the construction's own diagnostics. A field the
    session did not carry is ``None`` -- unknown -- never zero.
    """
    raw = result.get("turnover_drivers")
    drivers: Mapping[str, object] = (
        cast(Mapping[str, object], raw) if isinstance(raw, Mapping) else {}
    )
    return {
        "date": trading_day,
        "turnover_one_way": _number(result.get("turnover_one_way")),
        "budget": _number(result.get("turnover_budget")),
        "mandatory_floor": _number(drivers.get("turnover_mandatory_floor")),
        "budget_discretionary": _number(drivers.get("turnover_budget_discretionary")),
        "conviction_ir_xs": _number(drivers.get("conviction_ir_xs")),
        "conviction_gate_applied": bool(drivers.get("conviction_gate_applied")),
        "conviction_gate_reason": (
            str(drivers["conviction_gate_reason"])
            if drivers.get("conviction_gate_reason") is not None
            else None
        ),
        "binding": bool(drivers.get("turnover_constraint_binding")),
    }


def _prior_rows(store: Store, trading_day: str, wanted: int) -> list[dict[str, object]]:
    """The ``wanted`` most recent served sessions strictly before ``trading_day``.

    Read from each `trader.session` manifest: this tripwire's own row when the
    session filed one, else the run's recorded ``turnover`` (value and budget
    only -- the drivers are then unknown). A session that recorded neither (held,
    failed, or before turnover was recorded) is skipped, so the window spans
    served sessions, not calendar days.
    """
    rows: list[dict[str, object]] = []
    cursor = dt.date.fromisoformat(trading_day)
    for _ in range(_SEARCH_SESSIONS):
        if len(rows) >= wanted:
            break
        cursor = previous_trading_day(cursor)
        day = cursor.isoformat()
        key = manifest_key(_SESSION_JOB, day)
        if not store.exists(key):
            continue
        manifest = json.loads(store.get_bytes(key))
        row = _row_from_manifest(day, manifest)
        if row is not None:
            rows.append(row)
    return rows


def _row_from_manifest(day: str, manifest: Mapping[str, object]) -> dict[str, object] | None:
    metrics = manifest.get("metrics")
    for metric in cast(Sequence[object], metrics) if isinstance(metrics, list) else ():
        if not isinstance(metric, Mapping):
            continue
        record = cast(Mapping[str, object], metric)
        block = record.get(TRIPWIRE_BLOCK)
        if record.get("name") == METRIC_NAME and isinstance(block, Mapping):
            today = cast(Mapping[str, object], block).get("today")
            if (
                isinstance(today, Mapping)
                and _number(cast(Mapping[str, object], today).get("turnover_one_way")) is not None
            ):
                return dict(cast(Mapping[str, object], today))
    turnover = manifest.get("turnover")
    if isinstance(turnover, Mapping):
        value = _number(cast(Mapping[str, object], turnover).get("value"))
        if value is not None:
            return driver_row(
                day,
                {
                    "turnover_one_way": value,
                    "turnover_budget": cast(Mapping[str, object], turnover).get("budget"),
                },
            )
    return None


def _attribute(window: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Split the window into forced and discretionary turnover; name the driver.

    v1's closed set (`turnover_tripwire._attribute`), ordered most-specific
    first. ``forced`` is each session's mandatory floor, capped at what it
    traded; a session whose drivers are unknown contributes to neither split
    and is counted in ``n_unattributed_sessions``.
    """

    def num(row: Mapping[str, object], key: str) -> float | None:
        return _number(row.get(key))

    total = sum(num(r, "turnover_one_way") or 0.0 for r in window)
    known = [r for r in window if num(r, "mandatory_floor") is not None]
    forced = sum(
        min(num(r, "mandatory_floor") or 0.0, num(r, "turnover_one_way") or 0.0) for r in known
    )
    known_total = sum(num(r, "turnover_one_way") or 0.0 for r in known)
    discretionary = max(known_total - forced, 0.0)
    n_binding = sum(1 for r in known if r.get("binding"))
    n_gated = sum(1 for r in known if r.get("conviction_gate_applied"))
    irs = sorted(v for r in known if (v := num(r, "conviction_ir_xs")) is not None)
    median_ir = irs[len(irs) // 2] if irs else None
    half = max(1, len(known) // 2)
    if not known:
        driver = "unattributed"
        detail = (
            f"{total:.1%} one-way over {len(window)} session(s), and none of them recorded "
            "its construction drivers, so forced and discretionary turnover cannot be told "
            "apart. Read each session's order book (`trader/order_books/{day}.json`)."
        )
    elif forced > 0.6 * known_total and known_total > 0:
        driver = "forced_exits"
        detail = (
            f"{forced:.1%} of {known_total:.1%} was MANDATORY turnover -- ineligibility pins, "
            "cap trims, the cash sleeve or hard-risk exits. The optimizer is not choosing to "
            "churn; rules are ejecting positions."
        )
    elif n_gated >= half:
        driver = "conviction_gate_throttling"
        detail = (
            f"the conviction gate throttled the discretionary budget on {n_gated} of "
            f"{len(known)} sessions, so this turnover is what SURVIVED the throttle; the "
            "residue is mandatory turnover or the gate floor is too high."
        )
    elif median_ir is not None and median_ir < 0.5:
        driver = "predictor_conviction_collapse"
        detail = (
            f"the alpha cross-section's information ratio is only {median_ir:.3f} (median over "
            f"{len(irs)} sessions): the names are not statistically distinguishable, so the "
            "target reshuffles on noise. An UPSTREAM condition in the M champion."
        )
    elif n_binding >= half:
        driver = "budget_saturation"
        detail = (
            f"the daily budget bound the solve on {n_binding} of {len(known)} sessions with "
            "conviction passing the gate: the target moves at least as fast as the budget lets "
            "the book travel, so the walk never converges."
        )
    else:
        driver = "unattributed"
        detail = (
            f"no single driver dominates: {forced:.1%} forced of {known_total:.1%}, budget "
            f"binding on {n_binding}/{len(known)}, conviction gate on {n_gated}. Treat it as a "
            "new failure mode."
        )
    return {
        "driver": driver,
        "detail": detail,
        "forced_sum": round(forced, 6),
        "discretionary_sum": round(discretionary, 6),
        "n_binding": n_binding,
        "n_conviction_gated": n_gated,
        "n_unattributed_sessions": len(window) - len(known),
        "median_conviction_ir": median_ir,
        "per_day": [
            {
                "date": r.get("date"),
                "turnover": round(num(r, "turnover_one_way") or 0.0, 4),
                "forced": None
                if num(r, "mandatory_floor") is None
                else round(num(r, "mandatory_floor") or 0.0, 4),
            }
            for r in window
        ],
    }


def check(
    store: Store,
    *,
    trading_day: str,
    result: Mapping[str, object],
    config_loader: Callable[[Store], TripwireConfig | None] = load_config,
) -> dict[str, object]:
    """Both bands and the attribution for this session. Never raises.

    Returns the block :func:`metric_record` files: ``status`` is one of
    ``ok``, ``breach_daily``, ``breach_rolling``, ``undeclared``,
    ``no_turnover`` or ``error``.
    """
    try:
        config = config_loader(store)
        if config is None:
            return {
                "status": "undeclared",
                "reason": (
                    f"`strategy/slots/s.yaml` declares no `{TRIPWIRE_BLOCK}` block, so the churn "
                    "alarm has no band to read against"
                ),
            }
        today = driver_row(trading_day, result)
        value = _number(today["turnover_one_way"])
        if value is None:
            return {
                "status": "no_turnover",
                "reason": "the session recorded no finite turnover_one_way",
            }
        budget = _number(today["budget"])
        daily_band = None if budget is None else budget * config.daily_multiple
        window = [today, *_prior_rows(store, trading_day, config.rolling_sessions - 1)]
        rolling_sum = float(sum(_number(r["turnover_one_way"]) or 0.0 for r in window))
        daily_breach = daily_band is not None and value > daily_band
        rolling_breach = rolling_sum > config.rolling_sum_band
        # Daily outranks rolling: unexplained turnover above the session's own
        # budget is the more urgent finding.
        status = "breach_daily" if daily_breach else "breach_rolling" if rolling_breach else "ok"
        return {
            "status": status,
            "today": today,
            "turnover_one_way": round(value, 6),
            "daily_band": None if daily_band is None else round(daily_band, 6),
            "daily_breach": daily_breach,
            "rolling_sessions": config.rolling_sessions,
            "n_sessions_used": len(window),
            "rolling_sum": round(rolling_sum, 6),
            "rolling_band": config.rolling_sum_band,
            "rolling_breach": rolling_breach,
            "attribution": _attribute(window),
        }
    except Exception as exc:  # noqa: BLE001 - secondary observability; recorded, not swallowed
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


_METRIC_STATUS = {
    "ok": "OK",
    "breach_daily": "BREACH",
    "breach_rolling": "BREACH",
    "undeclared": "unmeasurable",
    "no_turnover": "unmeasurable",
    "error": "unmeasurable",
}


def metric_record(
    block: Mapping[str, object], *, trading_day: str, now_utc: str, module: str
) -> dict[str, object]:
    """The `trader_turnover_tripwire` MetricRecord for ``block``."""
    status = str(block["status"])
    if status == "ok" or status.startswith("breach"):
        attribution = cast(Mapping[str, object], block["attribution"])
        reason = (
            f"{status}: today {block['turnover_one_way']:.1%} one-way (daily band "
            f"{'none' if block['daily_band'] is None else format(block['daily_band'], '.1%')}), "
            f"{block['rolling_sum']:.1%} over {block['n_sessions_used']} session(s) (band "
            f"{block['rolling_band']:.0%}). DRIVER {attribution['driver']}: "
            f"{attribution['detail']}"
        )
        value: float | None = float(cast(float, block["rolling_sum"]))
    else:
        reason = f"{status}: {block.get('reason')}"
        value = None
    record: dict[str, object] = {
        "name": METRIC_NAME,
        "module": module,
        "metric_type": "operational",
        "n_floor": 1,
        "status": _METRIC_STATUS[status],
        "status_reason": reason,
        "source_path": manifest_key(_SESSION_JOB, trading_day),
        "last_updated_utc": now_utc,
        TRIPWIRE_BLOCK: dict(block),
    }
    if value is not None:
        record["value"] = value
        record["unit"] = "ratio"
    return record
