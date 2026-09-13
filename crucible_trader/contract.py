"""The two-document contract, resolved once per session, and its refusals.

The trader reads exactly two documents from the v2 store:

    champions/{slot}/current.json      the arm it may serve
    predictions/{trading_day}.json     that arm's cross-section for the session

Nothing here re-implements a refusal. `crucible.champion.read_champion` already
re-derives the producing run's manifest `status` on every read and raises
`ChampionUnusableError` when it is not `ok`, and already refuses an S-slot
champion whose `pit_parity` attestation is not `PASS`.
`crucible.serving.read_predictions_feed` already validates the feed against
`predictions_feed.v1` and refuses one misfiled under another day's key. This
module CALLS those readers and translates what they raise into the two actions
the trader can take.

**Two outcomes, never collapsed.**

    ContractUnavailable   the harness has not published yet.
                          HOLD the book; page on the absence deadline.
    ContractRefusal       something exists and must not be served.
                          PAGE NOW; trade nothing; exit non-zero.

A trader that collapsed them would treat a refused champion as a quiet day. That
is the single most expensive confusion available in this system, which is why the
distinction is a type here and not a flag.

There is no third outcome and no fallback: never yesterday's feed, never another
slot, never a partial read.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

from crucible.champion import ChampionPointer, ChampionUnusableError, read_champion
from crucible.explain import verify_money_path_chain
from crucible.keys import TRADER_EVIDENCE_KEY
from crucible.models import TraderEvidenceDocument
from crucible.serving import PredictionsFeed, PredictionsFeedContractError, read_predictions_feed
from crucible.store import Store

#: The schema the evidence document declares. Resolved from the model rather
#: than retyped, so the producer cannot name a version the consumer's schema
#: does not describe.
EVIDENCE_SCHEMA_VERSION = "trader_evidence.v1"


class ContractUnavailable(RuntimeError):
    """The harness has published nothing to serve yet.

    HOLD. This is a legitimate state before the day's feed lands, and it becomes
    a page only when the absence deadline passes -- a deadline the harness's own
    `components.yaml` owns, not this module.
    """


class ContractRefusal(RuntimeError):
    """Something exists in the store and the trader must not serve it.

    PAGE NOW and trade nothing. Distinct from :class:`ContractUnavailable` on
    purpose: an absent pointer is a quiet day, an unusable one is an incident,
    and a trader that could not tell them apart would sit quietly through the
    second.
    """


@dataclasses.dataclass(frozen=True)
class ResolvedContract:
    """Both halves of the contract, agreeing with each other.

    Frozen: what a session serves is decided once, before the first order. A
    contract that could be re-resolved mid-session could size half a book on one
    champion and half on another and report one of them.
    """

    trading_day: str
    champion: ChampionPointer
    feed: PredictionsFeed

    @property
    def arm_id(self) -> str:
        return self.champion.arm_id


def resolve(store: Store, trading_day: str, *, slot: str = "m") -> ResolvedContract:
    """Read both documents for ``trading_day`` and refuse anything unserveable.

    Raises :class:`ContractUnavailable` when either half is simply absent, and
    :class:`ContractRefusal` when either half exists and fails a check --
    including the two the harness's readers make on the trader's behalf (the
    producing manifest is not `ok`; an attested slot's attestation is not PASS)
    and the two this function makes because only a consumer of BOTH documents
    can: that the feed names the champion the pointer names, and that the
    money-path history in this store is intact.
    """
    champion = _read_champion(store, slot)
    feed = _read_feed(store, trading_day)

    if feed.champion != champion.arm_id:
        raise ContractRefusal(
            f"the feed for {trading_day} was produced by arm {feed.champion!r} but the "
            f"{slot!r} champion pointer names {champion.arm_id!r}. The two halves of the "
            "contract disagree about what is being served, and sizing on the feed anyway "
            "would trade a cross-section no pointer authorises. Neither document is "
            "wrong on its own -- which is exactly why only a reader of both can catch it."
        )

    _assert_money_path_intact(store)

    return ResolvedContract(trading_day=trading_day, champion=champion, feed=feed)


def _read_champion(store: Store, slot: str) -> ChampionPointer:
    try:
        return read_champion(store, slot)
    except ChampionUnusableError as exc:
        # PAGE NOW. A pointer exists and must not be served: its producing run
        # failed, its manifest is missing, or its attestation is not PASS.
        raise ContractRefusal(
            f"the {slot!r} champion pointer exists and is unusable: {exc}"
        ) from exc
    except KeyError as exc:
        # HOLD. No champion has been promoted for this slot yet, which is a
        # different fact from "a champion I must not serve" and is why these are
        # two exception types rather than one with a reason string.
        raise ContractUnavailable(
            f"no champion pointer for slot {slot!r}. Nothing has been promoted yet, so "
            "there is nothing to serve -- hold the book and page on the absence "
            "deadline rather than treating an empty slot as a decision to trade nothing."
        ) from exc


def _read_feed(store: Store, trading_day: str) -> PredictionsFeed:
    try:
        return read_predictions_feed(store, trading_day)
    except PredictionsFeedContractError as exc:
        # PAGE NOW. A feed exists for this session and is malformed or misfiled.
        raise ContractRefusal(f"the feed for {trading_day} is unserveable: {exc}") from exc
    except KeyError as exc:
        # HOLD. The harness has not published this session's feed yet.
        raise ContractUnavailable(
            f"no predictions feed for trading day {trading_day}. The harness has not "
            "published this session yet -- hold, and page once the absence deadline "
            "passes. Serving yesterday's feed is never the answer: it would size today "
            "on a cross-section fitted for a day that has already settled."
        ) from exc


def _assert_money_path_intact(store: Store) -> None:
    """The money-path hash chain in this store verifies.

    Called through `crucible.explain.verify_money_path_chain` rather than
    re-derived: the chain's three checks (contiguous indices, each link matching
    the digest of its predecessor AS THE STORE HOLDS IT NOW, no unlinked
    money-path manifest) are the harness's, and a second implementation here
    would be a second opinion about whether the history was edited.

    A broken chain refuses the SESSION, not just the record: a history that
    cannot be trusted makes every reconciliation and every NAV figure downstream
    of it unattributable, and trading on top of one is how an unnoticed
    restatement becomes a position.
    """
    verification = verify_money_path_chain(store)
    if verification.status != "ok":
        raise ContractRefusal(
            f"the money-path chain in this store does not verify: {verification.reason}. "
            "The trader does not open a session over a history it cannot attest."
        )


def record_session(
    store: Store,
    contract: ResolvedContract,
    *,
    calendar_date: str | None = None,
) -> TraderEvidenceDocument:
    """Add ``contract``'s trading day to the consumer-evidence document.

    This is the artifact the phase-4 gate clause `trader_one_week_on_v2_champion`
    reads (`crucible.keys.TRADER_EVIDENCE_KEY`). The harness may not reach into
    the trader, so this record is the trader's own account of what it served --
    which is why the document's own schema makes the count uninflatable:
    `trading_days` must equal `len(days_served)`, the days must be real, unique
    and strictly increasing.

    **A promotion ENDS a run of days.** When the champion has changed since the
    last record, the count restarts at this session rather than carrying the old
    arm's days forward: "a week on the v2 champion" is a week on ONE champion,
    and a rolling total across promotions would let five one-day champions read
    as a week.

    Idempotent within a day: recording the same trading day twice leaves the
    document unchanged, so a retried session does not inflate the count.
    """
    day = contract.trading_day
    previous = _read_evidence(store)

    if previous is not None and previous.champion == contract.arm_id:
        days = list(previous.days_served)
        if day not in days:
            days.append(day)
            days.sort()
    else:
        days = [day]

    document = TraderEvidenceDocument(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        slot=contract.champion.slot,
        champion=contract.arm_id,
        trading_days=len(days),
        days_served=days,
        calendar_date=calendar_date or dt.date.today().isoformat(),
    )
    store.put_bytes(
        TRADER_EVIDENCE_KEY,
        json.dumps(document.model_dump(), indent=2, sort_keys=True).encode("utf-8"),
    )
    return document


def _read_evidence(store: Store) -> TraderEvidenceDocument | None:
    """The evidence document already in the store, or `None` when there is none.

    `None` here is the ONE place this module returns it, and it is a real
    absence rather than a swallowed failure: a trader on its first session has
    filed nothing, and that is a different fact from a document it could not
    read. A document that EXISTS and does not validate raises -- overwriting it
    with a fresh count would silently reset the evidence the gate grades, which
    is the one write this module must never make by accident.
    """
    try:
        payload = store.get_bytes(TRADER_EVIDENCE_KEY)
    except KeyError:
        return None
    return TraderEvidenceDocument.model_validate(json.loads(payload.decode("utf-8")))
