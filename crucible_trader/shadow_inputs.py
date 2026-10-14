"""Each registered S arm's shadow-book session, resolved from the store.

The remaining deliverable of `alpha-engine-config-I10653`: `run_shadow_books`
takes each arm's `ArmSessionInputs` already assembled, and this module
assembles them the way the S grade does — through
`crucible.slots.inputs.resolve_strategy_sessions`, over the arm's own recorded
`session_inputs.v1` document for the decision day, joined onto the session it
was held through. So a shadow book is constructed on the grade's inputs by the
grade's engine, and the paper P&L it reports is the same object the grade
scores (`alpha-engine-config-I10654`).

**Fills at close.** The decision day's book is held through its successor
session; advancing it needs the panel compiled for that successor. Run it
after the successor's close, with ``as_of`` = that successor.

**Held names are exited, not refused** (`alpha-engine-config-I10754`). Each
arm's book advanced through ``previous_trading_day`` is read back
(`read_previous_books`, the read `run_shadow_books` advances from) and its
non-zero, non-sentinel names are passed to the resolver as ``held_tickers``, so
a name that arm's session did not price enters its universe as a forced exit
charged by the recipe's cost model — the book the settled grade walk builds.

**No active arm is silently dropped** (`alpha-engine-config-I10636`). An arm
whose inputs cannot be resolved comes back in ``unresolved`` with the reason,
and `run_shadow_books` records it as a failed book naming that reason.

**Controls are not resolved** (`alpha-engine-config-I12021`). A control arm
(`ArmRecord.control`) has no filed recipe and no construction inputs by
design: it is scored by the grade's seeded selection. It is listed in
``active_arms`` and ``controls`` and in neither ``inputs`` nor ``unresolved``;
`run_shadow_books` records it with the ruled non-book status.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from crucible.portfolio import load_portfolio_params_from_store
from crucible.slots.arms import read_register
from crucible.slots.inputs import SlotUnservableError, resolve_strategy_sessions
from crucible.slots.strategy import RegisteredStrategyArm, load_strategy_slot
from crucible.store import Store

from crucible_trader.construction import held_tickers
from crucible_trader.shadow_books import ArmSessionInputs, read_previous_books

__all__ = ["ResolvedShadowInputs", "resolve_shadow_inputs"]

SLOT = "s"


@dataclasses.dataclass(frozen=True)
class ResolvedShadowInputs:
    """Every active arm, either resolved or unresolved with its reason — never absent."""

    decision_day: str
    as_of: str
    active_arms: tuple[str, ...]
    controls: tuple[str, ...]
    inputs: Mapping[str, ArmSessionInputs]
    unresolved: Mapping[str, str]


def _registered_by_id(store: Store, today: str) -> tuple[dict[str, RegisteredStrategyArm], str]:
    try:
        loaded = load_strategy_slot(store=store, register=read_register(store, SLOT), today=today)
    except SlotUnservableError as exc:
        return {}, f"the S slot is unservable: every filed recipe is refused ({exc})"
    refused = ", ".join(f"{r.arm}: {r.reason}" for r in loaded.refused) or "none"
    return {arm.arm_id: arm for arm in loaded.registered}, f"refused recipes: {refused}"


def resolve_shadow_inputs(
    store: Store,
    *,
    decision_day: str,
    as_of: str,
    previous_trading_day: str | None,
    feature_version: str | None = None,
) -> ResolvedShadowInputs:
    """Resolve ``decision_day``'s session for every ACTIVE arm in the S register.

    ``previous_trading_day`` is the session whose `shadow_books.v1` document the
    books advance from — the same value handed to `run_shadow_books` — or None
    at inception. It is required so a caller cannot silently omit the held names.
    """
    register = read_register(store, SLOT)
    active = tuple(register.active_arms())
    controls = tuple(arm_id for arm_id in active if register.state(arm_id).record.control)
    challengers = tuple(arm_id for arm_id in active if arm_id not in controls)
    previous = read_previous_books(store, previous_trading_day) if challengers else {}
    registered, load_note = _registered_by_id(store, decision_day)
    inputs: dict[str, ArmSessionInputs] = {}
    unresolved: dict[str, str] = {}
    params = None
    params_error: str | None = None
    if challengers:
        try:
            params = load_portfolio_params_from_store(SLOT, store=store)
        except Exception as exc:
            # Failure mode swallowed: the slot's parameter set is unusable, which
            # fails EVERY arm's book. Recording surface: each active arm's
            # `unresolved` reason -> a failed book in shadow_books.v1, and
            # ShadowBookFailure raised by run_shadow_books after writing.
            params_error = f"{type(exc).__name__}: {exc}"
    for arm_id in challengers:
        arm = registered.get(arm_id)
        if arm is None:
            unresolved[arm_id] = (
                f"{arm_id} is active in the S register but no filed recipe registers it on "
                f"{decision_day} ({load_note})"
            )
            continue
        if params is None:
            unresolved[arm_id] = f"the S portfolio parameter set is unusable: {params_error}"
            continue
        try:
            resolved = resolve_strategy_sessions(
                store,
                arm_id=arm_id,
                benchmark=arm.recipe.benchmark,
                decision_days=[decision_day],
                as_of=as_of,
                feature_version=feature_version,
                held_tickers=held_tickers(
                    previous[arm_id]["weights"] if arm_id in previous else None,
                    benchmark=arm.recipe.benchmark,
                ),
            )
        except Exception as exc:
            # Failure mode swallowed: one arm's inputs are unresolvable (no
            # recorded session, no settled successor, a misfiled document).
            # Recording surface: the arm's `unresolved` reason -> its failed book
            # in shadow_books.v1, and ShadowBookFailure after the write — so one
            # arm cannot hide the others' books and the run still exits non-zero.
            unresolved[arm_id] = f"{type(exc).__name__}: {exc}"
            continue
        (session,) = resolved.sessions
        inputs[arm_id] = ArmSessionInputs(
            recipe=arm.recipe,
            params=params,
            universe=resolved.universe,
            session=session,
            portfolio_notional=params.book_notional_usd,
        )
    return ResolvedShadowInputs(
        decision_day=decision_day,
        as_of=as_of,
        active_arms=active,
        controls=controls,
        inputs=inputs,
        unresolved=unresolved,
    )
