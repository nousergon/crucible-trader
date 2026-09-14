"""The trader's book, constructed by the grade's own engine on the grade's own inputs.

`alpha-engine-config-I10654`; plan §10.6 row 2: *"One portfolio-construction
engine shared by grading and trading (`crucible.portfolio`) ... The S-slot grade
and the trader call the same function on the same inputs."*

**The function** is `crucible.slots.strategy.construct_book` — the one route by
which the S grade builds a book. **The inputs** are resolved by
`crucible.slots.inputs.resolve_strategy_sessions`, the function
`crucible.slots.strategy.grade` itself resolves every arm's inputs with: the
`session_inputs.v1` document `experiment.run --slot s` recorded for the S
champion on the decision day (the M champion's predictions as alpha, the U
champion's cut as eligibility, the stance caps), the price panel compiled for
that day, the book universe, and the day's `dollar_volume_20d_raw` as ADV.
Nothing in this module sizes, prices a trade,
estimates a covariance or joins an input. A second implementation of any of
those here would make the traded book and the graded book two different
objects, which is the failure this module exists to remove.

**It refuses to construct with no attested champion.** The S champion is read
through `crucible_trader.contract.read_slot_champion`, which calls the harness's
`read_champion` — absent is :class:`ContractUnavailable` (hold), a failed
producing run or an attestation that is not `PASS` is :class:`ContractRefusal`
(page now). The M champion is read the same way, and the recorded session must
name that attested M champion's predictions as its alpha and the current U
champion's cut (or its recorded absence) as its eligibility — a session
recorded under a champion that has since been demoted is refused, not traded.

**A participation-aware cost model handed no ADV RAISES** inside the engine
(`alpha-engine-config-I10503`). That raise propagates: no ADV at trade time is a
refusal to size, never a cheaper charge.

**A held name the M champion stopped pricing is EXITED, not refused**
(`alpha-engine-config-I10754`, crucible-PR298). The book held into the session
is handed to the resolver as ``held_tickers`` (its non-zero, non-sentinel
names); each enters the universe ineligible with alpha 0, so the engine's
ineligibility pin sells it and the recipe's cost model charges the sale — the
same book the settled grade walk constructs for that day. A held name the price
panel has no return for raises `MissingArtifactError` in the resolver (a hold:
an exit with no price is not a fillable trade).

**What this returns** is the target book and the `portfolio_construction.v1`
evidence the engine produced beside it, plus that evidence as a `MetricRecord`
(`crucible.portfolio.portfolio_metric_record`) — the record the grading
manifest carries, so "the grade and the trade used the same engine and cost
model" is a document comparison.

The decision is taken at the decision day's close over that day's panel and is
held through the next session; its realized return does not exist yet and is
left NaN by the resolver, never zero. Only the weights and the engine's charge
are read off it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Mapping
from typing import Any

import numpy as np
from crucible.keys import arm_predictions_key, champion_key, shadow_key
from crucible.portfolio import PortfolioParamsError, load_portfolio_params_from_store
from crucible.portfolio import portfolio_metric_record as _portfolio_metric_record
from crucible.slots.arms import read_register
from crucible.slots.cycle import MissingArtifactError
from crucible.slots.inputs import SlotUnservableError, resolve_strategy_sessions
from crucible.slots.strategy import (
    CASH_TICKER,
    BookUniverse,
    RegisteredStrategyArm,
    ResolvedSession,
    construct_book,
    load_strategy_slot,
)
from crucible.store import Store

from crucible_trader.contract import ContractRefusal, ContractUnavailable, read_slot_champion

__all__ = [
    "STRATEGY_SLOT",
    "TargetBook",
    "construct_target_book",
    "held_tickers",
    "initial_weights",
    "resolve_strategy_arm",
]

#: The slot whose champion decides how the book is constructed.
STRATEGY_SLOT = "s"


@dataclasses.dataclass(frozen=True)
class TargetBook:
    """The book the trader will hold through the next session, and its evidence."""

    trading_day: str
    s_champion: str
    m_champion: str
    u_champion: str | None
    universe: BookUniverse
    weights: dict[str, float]
    turnover_one_way_ratio: float
    cost_bps: float
    portfolio_notional_usd: float
    evidence: dict[str, Any]
    metric: dict[str, Any]
    session_inputs_key: str


def initial_weights(universe: BookUniverse) -> np.ndarray:
    """All cash: the inception book, where every position is a trade and charged as one."""
    weights = np.zeros(len(universe.tickers), dtype=np.float64)
    weights[universe.cash_idx] = 1.0
    return weights


def resolve_strategy_arm(store: Store, *, arm_id: str, trading_day: str) -> RegisteredStrategyArm:
    """The registered S recipe for ``arm_id``, loaded the way the S cycle job loads it.

    `load_strategy_slot` over the recipes synced into the store, with the slot's
    register and the day, so `registered_at` resolves exactly as it did for the
    grade. An arm the slot refuses or does not file is a :class:`ContractRefusal`:
    a pointer to a recipe the harness cannot construct is not a book to trade.
    """
    try:
        loaded = load_strategy_slot(
            store=store, register=read_register(store, STRATEGY_SLOT), today=trading_day
        )
    except SlotUnservableError as exc:
        raise ContractRefusal(
            f"the S slot is unservable on {trading_day}: every filed recipe is refused "
            f"({exc}). The champion {arm_id!r} has no recipe to construct with."
        ) from exc
    for arm in loaded.registered:
        if arm.arm_id == arm_id:
            return arm
    refused = sorted(r.arm for r in loaded.refused)
    raise ContractRefusal(
        f"arm {arm_id!r} is not among the S recipes the store registers on {trading_day} "
        f"(registered: {sorted(a.arm_id for a in loaded.registered)}; refused: {refused}). "
        "Its cost model and benchmark are declared by that recipe, and constructing without "
        "it would size under parameters no recipe names."
    )


def _assert_session_names_the_attested_champions(
    session: ResolvedSession, *, trading_day: str, m_champion: str, u_champion: str | None
) -> None:
    expected_alpha = arm_predictions_key(m_champion, trading_day)
    if session.alpha_source != expected_alpha:
        raise ContractRefusal(
            f"the session recorded for {trading_day} took its alpha from "
            f"{session.alpha_source!r}, but the attested M champion's predictions are "
            f"{expected_alpha!r}. The recorded inputs were resolved under a different M "
            "champion, and trading them would size on a forecast no pointer authorises."
        )
    if u_champion is None:
        if not session.eligibility_source.startswith("absent"):
            raise ContractRefusal(
                f"the session recorded for {trading_day} applied the cut "
                f"{session.eligibility_source!r}, but no U champion is promoted now. A cut "
                "from a pointer that no longer exists is not this session's eligibility."
            )
        return
    expected_cut = shadow_key(u_champion, trading_day)
    if session.eligibility_source != expected_cut:
        raise ContractRefusal(
            f"the session recorded for {trading_day} took its eligibility from "
            f"{session.eligibility_source!r}, but the U champion's cut is {expected_cut!r}."
        )


def held_tickers(previous: Mapping[str, float] | None, *, benchmark: str) -> list[str]:
    """The book's non-zero support, sentinels (benchmark, cash) excluded — exactly
    what `resolve_strategy_sessions(held_tickers=...)` is specified to take: a
    zero-weight name would add a column the grade's universe does not have."""
    if previous is None:
        return []
    return sorted(t for t, w in previous.items() if w != 0.0 and t not in {benchmark, CASH_TICKER})


def _previous_weights(
    universe: BookUniverse, previous: Mapping[str, float] | None, *, trading_day: str
) -> np.ndarray:
    if previous is None:
        return initial_weights(universe)
    unseen = sorted(t for t, w in previous.items() if t not in universe.tickers and w != 0.0)
    if unseen:
        raise ContractRefusal(
            f"the book held into {trading_day} carries {unseen}, which the session's universe "
            "does not contain although they were passed to the resolver as held names. The "
            "engine cannot see a name it is not given, so it could neither size nor price "
            "the trade that exits it; constructing anyway would report a book that is not "
            "the one held."
        )
    return np.array([float(previous.get(t, 0.0)) for t in universe.tickers], dtype=np.float64)


def construct_target_book(
    store: Store,
    *,
    trading_day: str,
    previous_weights: Mapping[str, float] | None,
    now: dt.datetime,
    feature_version: str | None = None,
) -> TargetBook:
    """Construct the book for ``trading_day`` through `construct_book`, or refuse.

    ``previous_weights`` is the book held into the session by ticker (None at
    inception: all cash). Raises :class:`ContractUnavailable` when a half the
    construction needs has not been published yet (no champion, no recorded
    session, no panel), :class:`ContractRefusal` when something exists and must
    not be traded, and lets the engine's own refusals (e.g.
    `CostModelInputError`) propagate unchanged.
    """
    s_pointer = read_slot_champion(store, STRATEGY_SLOT)
    m_pointer = read_slot_champion(store, "m")
    u_champion = read_slot_champion(store, "u").arm_id if store.exists(champion_key("u")) else None
    arm = resolve_strategy_arm(store, arm_id=s_pointer.arm_id, trading_day=trading_day)
    try:
        params = load_portfolio_params_from_store(STRATEGY_SLOT, store=store)
    except PortfolioParamsError as exc:
        raise ContractRefusal(f"the S slot's portfolio parameter set is unusable: {exc}") from exc

    try:
        inputs = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=[trading_day],
            as_of=trading_day,
            feature_version=feature_version,
            unsettled_last=True,
            held_tickers=held_tickers(previous_weights, benchmark=arm.recipe.benchmark),
        )
    except MissingArtifactError as exc:
        raise ContractUnavailable(
            f"the construction inputs for {trading_day} are not published: {exc}"
        ) from exc
    (session,) = inputs.resolved
    _assert_session_names_the_attested_champions(
        session, trading_day=trading_day, m_champion=m_pointer.arm_id, u_champion=u_champion
    )

    constructed = construct_book(
        recipe=arm.recipe,
        params=params,
        universe=inputs.universe,
        sessions=inputs.sessions,
        portfolio_notional=params.book_notional_usd,
        w_initial=_previous_weights(inputs.universe, previous_weights, trading_day=trading_day),
    )
    (weights,) = constructed.weights
    book = constructed.book
    assert book.cost_bps is not None  # construct_book always prices; a None is a harness defect
    return TargetBook(
        trading_day=trading_day,
        s_champion=s_pointer.arm_id,
        m_champion=m_pointer.arm_id,
        u_champion=u_champion,
        universe=inputs.universe,
        weights={t: float(w) for t, w in zip(inputs.universe.tickers, weights, strict=True)},
        turnover_one_way_ratio=float(book.turnover[0]),
        cost_bps=float(book.cost_bps[0]),
        portfolio_notional_usd=float(params.book_notional_usd),
        evidence=constructed.evidence,
        metric=_portfolio_metric_record(
            constructed.evidence,
            now_utc=now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        session_inputs_key=inputs.session_inputs_keys[0],
    )
