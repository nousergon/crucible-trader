"""Shadow books per registered arm, on paper — the realized half of the loop.

`alpha-engine-config-I10653`; plan §10.6 row 6: *"The trader keeps a simulated
book per registered challenger (fills at close, same cost model), so realized
paper P&L is per arm."*

**One construction engine, one charge.** A shadow book is advanced by
`crucible.slots.strategy.construct_book` — the function S-slot grading builds
every book with — over exactly one session, from the book's own previous
closing weights. The cost is therefore the arm recipe's own
`crucible.portfolio.CostModel`, priced by `CostModel.cost_bps_for_trades` on
the realized weight deltas, which is the charge the S grade subtracts. Nothing
in this module prices a trade. A shadow book built by a second engine, or
charged by a second cost function, would measure a different object from the
grade it is supposed to be compared with — the whole failure this deliverable
exists to remove (`alpha-engine-config-I10654`: one engine for the grade and
the trade).

**A participation-aware model handed no ADV RAISES** (`-I10503` / `-I10513`),
and that raise is not caught into a flat charge here. It becomes a FAILED
book, recorded in the artifact beside every other arm's, and the session exits
non-zero after the document is written.

**No active arm is silently excluded** (`-I10636`). The active set is read
from the harness's register (`crucible.slots.arms.read_register`); an active
arm the caller supplied no inputs for is a failed book naming that, never a
shorter list.

**Evidence, not a promotion input.** Nothing here moves a champion pointer.
Whether realized edge ever does is a champion-challenger-policy change,
out of scope by decision (I10653 deliverable 5).

**Data plumbing is the caller's.** This module takes each arm's
`SessionInputs` already assembled (alpha from the arm's predictions, the
price panel, ADV); assembling them from the store is the trader session's job
and lands with the `crucible.portfolio` adoption (`-I10654`).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from crucible.execution import (
    SHADOW_BOOK_METRIC_NAME,
    SHADOW_BOOKS_SCHEMA_VERSION,
    SHADOW_FILL_BASIS,
    shadow_books_key,
    validate_shadow_books,
)
from crucible.portfolio import PortfolioParams
from crucible.slots.arms import read_register
from crucible.slots.strategy import BookUniverse, SessionInputs, StrategyRecipe, construct_book
from crucible.store import Store

__all__ = [
    "ArmSessionInputs",
    "ShadowBookFailure",
    "advance_book",
    "build_shadow_books_document",
    "read_previous_books",
    "run_shadow_books",
]

MODULE = "crucible_trader.shadow_books"


class ShadowBookFailure(RuntimeError):
    """One or more active arms' books failed to advance. Raised AFTER the
    document recording each failure is written, so the finding is durable and
    the process still exits non-zero."""


@dataclasses.dataclass(frozen=True)
class ArmSessionInputs:
    """Everything one arm's book needs to advance one session."""

    recipe: StrategyRecipe
    params: PortfolioParams
    universe: BookUniverse
    session: SessionInputs
    portfolio_notional: float


def _initial_weights(universe: BookUniverse) -> np.ndarray:
    """A book's first session starts all cash: every position is a trade, so the
    inception session is charged for building the book, as a real one would be."""
    weights = np.zeros(len(universe.tickers), dtype=np.float64)
    weights[universe.cash_idx] = 1.0
    return weights


def advance_book(
    *,
    arm_id: str,
    inputs: ArmSessionInputs,
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Advance one arm's book by one session and return its `shadow_books.v1` entry.

    ``previous`` is this arm's advanced entry from the last session's document,
    or None at inception. Raises whatever `construct_book` raises — including
    `CostModelInputError` for a participation-aware model with no ADV.
    """
    universe = inputs.universe
    trading_day = inputs.session.trading_day
    if previous is None:
        w_prev = _initial_weights(universe)
        inception = trading_day
        days: list[str] = []
        cumulative = 0.0
    else:
        if previous["arm_id"] != arm_id or previous["status"] != "advanced":
            raise ValueError(
                f"{arm_id}: previous entry is {previous['arm_id']!r}/{previous['status']!r}; "
                "a book advances only from its own last ADVANCED state"
            )
        held = previous["weights"]
        unseen = sorted(t for t, w in held.items() if w != 0.0 and t not in universe.tickers)
        if unseen:
            raise ValueError(
                f"{arm_id}: the book holds {unseen}, which the session's universe "
                f"{sorted(universe.tickers)} does not contain. Held names are passed to the "
                "resolver (`resolve_shadow_inputs`) so a name the M champion stopped pricing "
                "enters as a forced exit; a universe without it could neither size nor "
                "charge that exit (alpha-engine-config-I10754)"
            )
        w_prev = np.array([float(held.get(t, 0.0)) for t in universe.tickers], dtype=np.float64)
        inception = str(previous["inception_trading_day"])
        days = list(previous["days_advanced"])
        cumulative = float(previous["cumulative_net_return_ratio"])
        if days and trading_day <= days[-1]:
            raise ValueError(
                f"{arm_id}: session {trading_day} does not follow the book's last "
                f"session {days[-1]}"
            )

    constructed = construct_book(
        recipe=inputs.recipe,
        params=inputs.params,
        universe=universe,
        sessions=[inputs.session],
        portfolio_notional=inputs.portfolio_notional,
        w_initial=w_prev,
    )
    book = constructed.book
    gross = float(book.portfolio_returns[0])
    cost_bps = float(book.cost_bps[0]) if book.cost_bps is not None else None
    if cost_bps is None:
        raise ValueError(
            f"{arm_id}: construct_book returned no per-session cost; a shadow book with no "
            "charge is not net of the grade's cost model"
        )
    net = gross - cost_bps / 1e4
    cumulative = (1.0 + cumulative) * (1.0 + net) - 1.0
    weights = constructed.weights[0]
    return {
        "arm_id": arm_id,
        "status": "advanced",
        "failure_reason": None,
        "cost_model": inputs.recipe.cost_model.record(),
        "inception_trading_day": inception,
        "days_advanced": [*days, trading_day],
        "sessions": len(days) + 1,
        "portfolio_notional_usd": float(inputs.portfolio_notional),
        "gross_return_ratio": gross,
        "cost_bps": cost_bps,
        "net_return_ratio": net,
        "cumulative_net_return_ratio": cumulative,
        "turnover_one_way_ratio": float(book.turnover[0]),
        "weights": {t: float(w) for t, w in zip(universe.tickers, weights, strict=True)},
    }


def _failed(arm_id: str, reason: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "status": "failed",
        "failure_reason": reason,
        "cost_model": None,
        "inception_trading_day": None,
        "days_advanced": [],
        "sessions": 0,
        "portfolio_notional_usd": None,
        "gross_return_ratio": None,
        "cost_bps": None,
        "net_return_ratio": None,
        "cumulative_net_return_ratio": None,
        "turnover_one_way_ratio": None,
        "weights": {},
    }


def _metric(book: Mapping[str, Any], trading_day: str, now: dt.datetime) -> dict[str, Any]:
    return {
        "name": SHADOW_BOOK_METRIC_NAME,
        "module": MODULE,
        "metric_type": "ratio",
        "value": book["cumulative_net_return_ratio"],
        "unit": "ratio",
        "n_samples": book["sessions"],
        "n_floor": 1,
        "status": "OK",
        "status_reason": (
            f"{book['arm_id']}: cumulative paper net return over {book['sessions']} session(s) "
            f"since {book['inception_trading_day']}, fills at close, net of "
            f"{book['cost_model']['name']} — realized edge, never collapsed with the "
            "backtested verdict"
        ),
        "source_path": shadow_books_key(trading_day),
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arm_id": book["arm_id"],
    }


def build_shadow_books_document(
    *,
    trading_day: str,
    slot: str,
    active_arms: list[str],
    books: list[dict[str, Any]],
    now: dt.datetime,
) -> dict[str, Any]:
    """Assemble and validate the session's `shadow_books.v1` document."""
    document = {
        "schema_version": SHADOW_BOOKS_SCHEMA_VERSION,
        "trading_day": trading_day,
        "calendar_date": now.astimezone(dt.UTC).date().isoformat(),
        "fill_basis": SHADOW_FILL_BASIS,
        "slot": slot,
        "active_arms": list(active_arms),
        "books": books,
        "metrics": [_metric(b, trading_day, now) for b in books if b["status"] == "advanced"],
    }
    return validate_shadow_books(document, origin=shadow_books_key(trading_day))


def read_previous_books(store: Store, previous_trading_day: str | None) -> dict[str, dict]:
    """Each arm's ADVANCED entry from the previous session's document, by arm id.

    Absent previous document (first session) → empty: every book incepts. A
    present one is validated; a corrupt one raises.
    """
    if previous_trading_day is None:
        return {}
    key = shadow_books_key(previous_trading_day)
    if not store.exists(key):
        return {}
    document = validate_shadow_books(json.loads(store.get_bytes(key).decode("utf-8")), origin=key)
    return {b["arm_id"]: b for b in document["books"] if b["status"] == "advanced"}


def run_shadow_books(
    store: Store,
    *,
    trading_day: str,
    previous_trading_day: str | None,
    inputs: Mapping[str, ArmSessionInputs],
    now: dt.datetime,
    slot: str = "s",
    unresolved: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Advance a book for every ACTIVE arm in ``slot``'s register, write, then refuse failures.

    Every active arm gets an entry: advanced, or failed with its reason. Inputs
    for an arm the register does not list as active are refused — a book for
    an unregistered arm is not a challenger's evidence.

    ``unresolved`` carries, per arm, why its inputs could not be assembled
    (`crucible_trader.shadow_inputs.resolve_shadow_inputs`); that reason becomes
    the failed book's `failure_reason` instead of a generic absence.
    """
    unresolved = dict(unresolved or {})
    active = list(read_register(store, slot).active_arms())
    stray = sorted((set(inputs) | set(unresolved)) - set(active))
    if stray:
        raise ValueError(
            f"inputs supplied for {stray}, which the {slot!r} register does not list as "
            "active; shadow books are kept for registered arms only"
        )
    previous = read_previous_books(store, previous_trading_day)
    books: list[dict[str, Any]] = []
    for arm_id in active:
        arm_inputs = inputs.get(arm_id)
        if arm_inputs is None:
            books.append(
                _failed(
                    arm_id,
                    unresolved.get(arm_id)
                    or f"no session inputs were assembled for {arm_id} on {trading_day}",
                )
            )
            continue
        try:
            books.append(
                advance_book(arm_id=arm_id, inputs=arm_inputs, previous=previous.get(arm_id))
            )
        except Exception as exc:
            # Failure mode swallowed HERE: one arm's construction raising (e.g.
            # CostModelInputError, no ADV). Recording surface: the book's
            # `failure_reason` in shadow_books.v1, and ShadowBookFailure raised
            # after the document is written — so one arm cannot hide the others'
            # books and the session still exits non-zero.
            books.append(_failed(arm_id, f"{type(exc).__name__}: {exc}"))
    document = build_shadow_books_document(
        trading_day=trading_day, slot=slot, active_arms=active, books=books, now=now
    )
    store.put_bytes(
        shadow_books_key(trading_day),
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
    )
    failed = [b for b in books if b["status"] == "failed"]
    if failed:
        raise ShadowBookFailure(
            f"{len(failed)} of {len(books)} shadow book(s) failed on {trading_day}: "
            + "; ".join(f"{b['arm_id']}: {b['failure_reason']}" for b in failed)
        )
    return document
