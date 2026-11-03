"""`exception_decision.v1`: the reviewer's call on one proposal, and the score.

`alpha-engine-config-I13120` (back office 1c). Stage 1b filed a shadow
:mod:`~crucible_trader.exception_proposal` for every break; nothing recorded
what the reviewer made of it, so no agreement could be measured. This module
adds the other half of the loop:

* **The decision record.** One document per call, written create-only at
  :func:`decision_key` and never rewritten. A verdict is one of
  :data:`VERDICTS`:

  - ``agree``: the proposal's nature and action stand;
  - ``correct``: the reviewer names the right nature and action (from the
    proposer's own closed sets) and a note saying why -- a scored
    disagreement, and the correction is the label;
  - ``defer``: no call yet, with a note; the item stays in the queue.

  A changed mind is a NEW decision that names the one it replaces in
  ``supersedes``; history is never edited (maker-checker, the back-office
  plan's control standard). The approver is ``decided_by``, which may never be
  the proposer: whoever proposes cannot approve.

* **The score** (:func:`score`): the effective decision per proposal, and per
  ``(tenant, source, class, ruleset_version)`` the counts, the agreement rate
  ``agree / (agree + correct)`` and the current run of consecutive agreements.
  A deferred or undecided proposal is in the queue, oldest first, with its age
  in trading days; past :data:`AGE_LIMIT_TRADING_DAYS` it is overdue.
  Agreement is per ruleset version because a proposal id is: a rule change
  starts a fresh record.

**No class ever auto-applies** (:data:`AUTO_APPLY` is ``False`` and every score
says so). The score is evidence for a later promotion gate, never the gate.

Who writes what. The decision is written by the REVIEWER's identity through
``record`` below (today the operator CLI; a console write path is a separate
ruling). The score is a derived document: the daily reconcile job files it as
step 5e (:func:`crucible_trader.reconciliation_job.file_exception_score`), and
``score`` below recomputes it read-only from the stored records, so anyone can
rebuild it.

    python -m crucible_trader.exception_decision record --store URI \\
        --proposal-id ID --verdict agree|correct|defer --by NAME \\
        [--nature N --action A] [--note TEXT] [--supersedes DECISION_ID]
    python -m crucible_trader.exception_decision score --store URI \\
        [--as-of YYYY-MM-DD] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from crucible.calendar import is_trading_day  # pyright: ignore[reportMissingTypeStubs]
from crucible.store import (  # pyright: ignore[reportMissingTypeStubs]
    ETAG_ABSENT,
    PointerConflictError,
    Store,
    open_store,
    read_only,
)

from crucible_trader.exception_proposal import (
    ACTIONS,
    NATURES,
    PROPOSAL_SCHEMA_VERSION,
    PROPOSALS_PREFIX,
    PROPOSER,
)

__all__ = [
    "AGE_LIMIT_TRADING_DAYS",
    "AUTO_APPLY",
    "DECISIONS_PREFIX",
    "DECISION_SCHEMA_VERSION",
    "SCORES_PREFIX",
    "SCORE_CURRENT_KEY",
    "SCORE_SCHEMA_VERSION",
    "VERDICTS",
    "DecisionError",
    "decide",
    "decision_key",
    "exception_score_key",
    "load_decisions",
    "load_proposals",
    "main",
    "record_decision",
    "score",
    "validate_decision",
]

DECISION_SCHEMA_VERSION = "exception_decision.v1"
SCORE_SCHEMA_VERSION = "exception_score.v1"

#: Decisions sit beside the proposals they decide, outside `crucible.keys` for
#: the same reason (a key there moves the trader pin). One object per decision,
#: `{DECISIONS_PREFIX}{trading_day}/{decision_id}.json`, keyed by the BREAK's
#: trading day so every key binds to a session (plan §4.12).
DECISIONS_PREFIX = "trader/exception_decisions/"

#: The daily score, `{SCORES_PREFIX}{trading_day}.json`; derived, so a same-day
#: re-run may rewrite it.
SCORES_PREFIX = "trader/exception_scores/"

#: The newest score, for a reader that wants today's queue and not a dated
#: series (the console's review page, alpha-engine-config-I13121). Undated, so
#: legal under plan §4.12; a re-run of an EARLIER day never moves it back.
SCORE_CURRENT_KEY = f"{SCORES_PREFIX}current.json"

VERDICTS: tuple[str, ...] = ("agree", "correct", "defer")

#: A queue item older than this many trading days is overdue (back-office plan,
#: control standard "Break aging").
AGE_LIMIT_TRADING_DAYS = 3

#: Nothing auto-applies in the back-office plan; every score states it.
AUTO_APPLY = False

_REVIEWER_RE = re.compile(r"^[a-z0-9][a-z0-9._@-]{0,63}$")
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_NOTE_MAX = 2000

#: The proposal fields a decision copies, so it stands alone as evidence.
_PROPOSAL_FIELDS = (
    "proposal_id",
    "exception_id",
    "tenant",
    "trading_day",
    "source",
    "class",
    "ruleset_version",
    "rule_id",
)

_DECISION_KEYS = frozenset(
    {
        "schema_version",
        "decision_id",
        *_PROPOSAL_FIELDS,
        "proposed",
        "verdict",
        "decided",
        "note",
        "decided_by",
        "decided_at",
        "supersedes",
    }
)


class DecisionError(ValueError):
    """A decision that cannot be recorded or read as `exception_decision.v1`."""


def _day(value: object) -> str:
    if not isinstance(value, str):
        raise DecisionError(f"trading day {value!r} is not a string")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        raise DecisionError(f"trading day {value!r} is not an ISO date") from None
    if parsed.isoformat() != value or not is_trading_day(parsed):
        raise DecisionError(f"{value} is not an NYSE trading day")
    return value


def decision_key(trading_day: str, decision_id: str) -> str:
    """The store key of one decision about a break on ``trading_day``."""
    if not _ID_RE.fullmatch(decision_id):
        raise DecisionError(f"decision id {decision_id!r} is not 32 lowercase hex")
    return f"{DECISIONS_PREFIX}{_day(trading_day)}/{decision_id}.json"


def exception_score_key(trading_day: str) -> str:
    """The store key of the score filed by ``trading_day``'s reconcile run."""
    return f"{SCORES_PREFIX}{_day(trading_day)}.json"


def _decision_id(document: Mapping[str, object]) -> str:
    body = {k: v for k, v in document.items() if k != "decision_id"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _check_proposal(proposal: Mapping[str, object]) -> None:
    if proposal.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise DecisionError(f"not an {PROPOSAL_SCHEMA_VERSION} document")
    for field in (*_PROPOSAL_FIELDS, "nature", "action"):
        if not isinstance(proposal.get(field), str) or not proposal[field]:
            raise DecisionError(f"proposal field {field!r} is missing or empty")


def decide(
    proposal: Mapping[str, object],
    *,
    verdict: str,
    decided_by: str,
    decided_at: str,
    note: str = "",
    nature: str | None = None,
    action: str | None = None,
    supersedes: str | None = None,
) -> dict[str, object]:
    """The `exception_decision.v1` document for one call on ``proposal``.

    Refuses: an unknown verdict; a reviewer who is the proposer or not a plain
    name; a ``correct`` without a nature, action and note, or one that repeats
    the proposal; an ``agree`` or ``defer`` that names a nature or action; a
    ``defer`` without a note; a malformed ``supersedes``.
    """
    _check_proposal(proposal)
    if verdict not in VERDICTS:
        raise DecisionError(f"verdict {verdict!r} is not one of {VERDICTS}")
    if decided_by == PROPOSER or not _REVIEWER_RE.fullmatch(decided_by):
        raise DecisionError(
            f"reviewer {decided_by!r} must be a plain name ({_REVIEWER_RE.pattern}) "
            "and never the proposer"
        )
    if not _TS_RE.fullmatch(decided_at):
        raise DecisionError(f"decided_at {decided_at!r} is not YYYY-MM-DDTHH:MM:SSZ")
    note = note.strip()
    if len(note) > _NOTE_MAX:
        raise DecisionError(f"note is longer than {_NOTE_MAX} characters")
    if supersedes is not None and not _ID_RE.fullmatch(supersedes):
        raise DecisionError(f"supersedes {supersedes!r} is not a decision id")
    proposed = {"nature": proposal["nature"], "action": proposal["action"]}
    if verdict == "correct":
        if nature not in NATURES or action not in ACTIONS:
            raise DecisionError(
                f"a correction names a nature from {NATURES} and an action from {ACTIONS}"
            )
        if {"nature": nature, "action": action} == proposed:
            raise DecisionError("a correction that repeats the proposal is an agreement")
        if not note:
            raise DecisionError("a correction carries a note saying why")
        decided: dict[str, object] | None = {"nature": nature, "action": action}
    else:
        if nature is not None or action is not None:
            raise DecisionError(f"a {verdict} names no nature or action")
        if verdict == "defer" and not note:
            raise DecisionError("a deferral carries a note saying what it waits for")
        decided = dict(proposed) if verdict == "agree" else None
    document: dict[str, object] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        **{field: proposal[field] for field in _PROPOSAL_FIELDS},
        "proposed": proposed,
        "verdict": verdict,
        "decided": decided,
        "note": note,
        "decided_by": decided_by,
        "decided_at": decided_at,
        "supersedes": supersedes,
    }
    document["decision_id"] = _decision_id(document)
    return document


def validate_decision(document: Mapping[str, object]) -> None:
    """Raise :class:`DecisionError` unless ``document`` is a well-formed,
    self-consistent `exception_decision.v1` record."""
    if frozenset(document) != _DECISION_KEYS:
        raise DecisionError(
            f"decision fields differ: missing {sorted(_DECISION_KEYS - set(document))}, "
            f"extra {sorted(set(document) - _DECISION_KEYS)}"
        )
    if document["schema_version"] != DECISION_SCHEMA_VERSION:
        raise DecisionError(f"not an {DECISION_SCHEMA_VERSION} document")
    proposed = document["proposed"]
    if not isinstance(proposed, dict):
        raise DecisionError("proposed is not an object")
    proposed = cast("dict[str, object]", proposed)
    decided = cast("dict[str, object] | None", document["decided"])
    rebuilt = decide(
        {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            **{field: document[field] for field in _PROPOSAL_FIELDS},
            "nature": proposed.get("nature"),
            "action": proposed.get("action"),
        },
        verdict=cast("str", document["verdict"]),
        decided_by=cast("str", document["decided_by"]),
        decided_at=cast("str", document["decided_at"]),
        note=cast("str", document["note"]),
        nature=cast("str", decided["nature"])
        if document["verdict"] == "correct" and isinstance(decided, dict)
        else None,
        action=cast("str", decided["action"])
        if document["verdict"] == "correct" and isinstance(decided, dict)
        else None,
        supersedes=cast("str | None", document["supersedes"]),
    )
    if rebuilt != dict(document):
        raise DecisionError(
            f"decision {document['decision_id']!r} does not rebuild from its fields"
        )


def _json_object(store: Store, key: str) -> dict[str, object]:
    raw: object = json.loads(store.get_bytes(key).decode("utf-8"))
    if not isinstance(raw, dict):
        raise DecisionError(f"{key} is not a JSON object")
    return cast("dict[str, object]", raw)


def load_proposals(store: Store, *, until: str | None = None) -> list[dict[str, object]]:
    """Every stored proposal filed up to ``until`` inclusive, each once.

    Two kinds of file sit under the prefix: the day files of step 5c
    (``{day}.json``) and a refused month-end close's files
    (:func:`~crucible_trader.exception_proposal.close_proposals_key`,
    ``month_end_close/{tenant}/{month}/{run_day}.json``,
    `alpha-engine-config-I13121`). Both end in the day they were written, which
    is what ``until`` compares. A close refused on several days files the same
    proposal ids each time; the first filing is the one kept.
    """
    out: list[dict[str, object]] = []
    seen: set[object] = set()
    for key in sorted(store.list_keys(PROPOSALS_PREFIX)):
        filed_on = key.removesuffix(".json").rsplit("/", 1)[-1]
        if not key.endswith(".json") or (until is not None and filed_on > until):
            continue
        document = _json_object(store, key)
        for proposal in cast("list[dict[str, object]]", document.get("proposals", [])):
            if proposal.get("proposal_id") not in seen:
                seen.add(proposal.get("proposal_id"))
                out.append(proposal)
    return out


def load_decisions(store: Store, *, prefix: str = DECISIONS_PREFIX) -> list[dict[str, object]]:
    """Every stored decision under ``prefix``, each validated."""
    out: list[dict[str, object]] = []
    for key in sorted(store.list_keys(prefix)):
        if not key.endswith(".json"):
            continue
        document = _json_object(store, key)
        try:
            validate_decision(document)
        except DecisionError as exc:
            raise DecisionError(f"{key}: {exc}") from exc
        out.append(document)
    return out


def _effective(
    decisions: Iterable[Mapping[str, object]],
) -> dict[str, tuple[str, Mapping[str, object] | None]]:
    """Per proposal id: ``("decided", decision)`` for the one decision nothing
    supersedes, or ``("conflicted", None)`` when two stand side by side."""
    by_proposal: dict[str, list[Mapping[str, object]]] = {}
    for decision in decisions:
        by_proposal.setdefault(cast("str", decision["proposal_id"]), []).append(decision)
    out: dict[str, tuple[str, Mapping[str, object] | None]] = {}
    for proposal_id, chain in by_proposal.items():
        superseded = {d["supersedes"] for d in chain}
        standing = [d for d in chain if d["decision_id"] not in superseded]
        out[proposal_id] = ("decided", standing[0]) if len(standing) == 1 else ("conflicted", None)
    return out


def record_decision(store: Store, document: Mapping[str, object]) -> str:
    """Write one decision create-only and return its key.

    Refuses a decision on a proposal that already has a standing decision
    unless it supersedes exactly that one, and one that supersedes a decision
    that does not stand. Writing the same decision twice is a no-op.
    """
    validate_decision(document)
    day = cast("str", document["trading_day"])
    key = decision_key(day, cast("str", document["decision_id"]))
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if store.exists(key):
        if store.get_bytes(key) != payload:
            raise DecisionError(f"{key} already holds a different decision")
        return key
    prior = [
        d
        for d in load_decisions(store, prefix=f"{DECISIONS_PREFIX}{day}/")
        if d["proposal_id"] == document["proposal_id"]
    ]
    state = _effective(prior).get(cast("str", document["proposal_id"]))
    standing = state[1]["decision_id"] if state is not None and state[1] is not None else None
    if state is not None and state[0] == "conflicted":
        raise DecisionError("this proposal has conflicting decisions; settle them by hand first")
    if document["supersedes"] != standing:
        raise DecisionError(
            f"this decision must supersede the standing decision {standing!r}, "
            f"not {document['supersedes']!r}"
        )
    try:
        store.compare_and_swap(key, ETAG_ABSENT, payload)
    except PointerConflictError:
        if store.get_bytes(key) != payload:
            raise DecisionError(f"{key} was written concurrently with different bytes") from None
    return key


def _age(day: str, as_of: str) -> int:
    current, end, age = dt.date.fromisoformat(day), dt.date.fromisoformat(as_of), 0
    while current < end:
        current += dt.timedelta(days=1)
        age += 1 if is_trading_day(current) else 0
    return age


def score(
    proposals: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
    *,
    as_of: str,
    generated_at: str,
) -> dict[str, object]:
    """The `exception_score.v1` document over every proposal up to ``as_of``.

    Decisions about proposals not among ``proposals`` are counted as
    ``orphan_decisions`` and score nothing.
    """
    _day(as_of)
    effective = _effective(decisions)
    known = {cast("str", p["proposal_id"]) for p in proposals}
    groups: dict[tuple[str, str, str, str], dict[str, object]] = {}
    queue: list[dict[str, object]] = []
    calls: dict[tuple[str, str, str, str], list[tuple[str, str]]] = {}
    for proposal in sorted(proposals, key=lambda p: (str(p["trading_day"]), str(p["proposal_id"]))):
        if str(proposal["trading_day"]) > as_of:
            continue
        group_key = (
            cast("str", proposal["tenant"]),
            cast("str", proposal["source"]),
            cast("str", proposal["class"]),
            cast("str", proposal["ruleset_version"]),
        )
        group = groups.setdefault(
            group_key,
            {
                "tenant": group_key[0],
                "source": group_key[1],
                "class": group_key[2],
                "ruleset_version": group_key[3],
                "nature": proposal["nature"],
                "promotable": False,
                "proposals": 0,
                **dict.fromkeys(("agree", "correct", "defer", "undecided", "conflicted"), 0),
                "consecutive_agree": 0,
                "auto_apply": AUTO_APPLY,
            },
        )
        group["proposals"] = cast("int", group["proposals"]) + 1
        group["promotable"] = group["promotable"] or proposal["promotable"] is True
        status, decision = effective.get(cast("str", proposal["proposal_id"]), ("undecided", None))
        verdict = cast("str", decision["verdict"]) if decision is not None else status
        group[verdict] = cast("int", group[verdict]) + 1
        if decision is not None and verdict in ("agree", "correct"):
            calls.setdefault(group_key, []).append((cast("str", decision["decided_at"]), verdict))
            continue
        age = _age(cast("str", proposal["trading_day"]), as_of)
        queue.append(
            {
                **{f: proposal[f] for f in _PROPOSAL_FIELDS},
                "nature": proposal["nature"],
                "action": proposal["action"],
                "rationale": proposal.get("rationale"),
                "held": proposal.get("held"),
                "status": "deferred" if verdict == "defer" else verdict,
                "note": decision["note"] if decision is not None else None,
                "standing_decision_id": decision["decision_id"] if decision is not None else None,
                "age_trading_days": age,
                "overdue": age > AGE_LIMIT_TRADING_DAYS,
            }
        )
    classes: list[dict[str, object]] = []
    for group_key in sorted(groups):
        group = groups[group_key]
        # The current run of agreements, in the order the calls were made: a
        # correction ends it, a deferral neither extends nor ends it.
        streak = 0
        for _at, verdict in sorted(calls.get(group_key, [])):
            streak = streak + 1 if verdict == "agree" else 0
        group["consecutive_agree"] = streak
        agree, correct = cast("int", group["agree"]), cast("int", group["correct"])
        group["agreement"] = round(agree / (agree + correct), 4) if agree + correct else None
        classes.append(group)
    agree = sum(cast("int", g["agree"]) for g in classes)
    correct = sum(cast("int", g["correct"]) for g in classes)
    queue.sort(key=lambda q: (-cast("int", q["age_trading_days"]), str(q["proposal_id"])))
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "trading_day": as_of,
        "generated_at": generated_at,
        "auto_apply": AUTO_APPLY,
        "age_limit_trading_days": AGE_LIMIT_TRADING_DAYS,
        "totals": {
            "proposals": sum(cast("int", g["proposals"]) for g in classes),
            "agree": agree,
            "correct": correct,
            "defer": sum(cast("int", g["defer"]) for g in classes),
            "undecided": sum(cast("int", g["undecided"]) for g in classes),
            "conflicted": sum(cast("int", g["conflicted"]) for g in classes),
            "agreement": round(agree / (agree + correct), 4) if agree + correct else None,
            "queue": len(queue),
            "overdue": sum(1 for q in queue if q["overdue"]),
            "oldest_age_trading_days": max(
                (cast("int", q["age_trading_days"]) for q in queue), default=None
            ),
            "orphan_decisions": sum(1 for d in decisions if d["proposal_id"] not in known),
        },
        "classes": classes,
        "queue": queue,
    }


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _find_proposal(store: Store, proposal_id: str) -> dict[str, object]:
    for proposal in load_proposals(store):
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    raise DecisionError(f"no stored proposal has id {proposal_id!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.exception_decision",
        description="Record a reviewer's decision on a proposal, or print the score.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="write one decision, create-only")
    rec.add_argument("--proposal-id", required=True)
    rec.add_argument("--verdict", required=True, choices=VERDICTS)
    rec.add_argument("--by", required=True, help="the reviewer (never the proposer)")
    rec.add_argument("--note", default="")
    rec.add_argument("--nature", default=None, choices=NATURES)
    rec.add_argument("--action", default=None, choices=ACTIONS)
    rec.add_argument("--supersedes", default=None)
    sco = sub.add_parser("score", help="recompute the score from stored records; read-only")
    sco.add_argument("--as-of", default=None, type=dt.date.fromisoformat)
    sco.add_argument("--out", default=None, type=pathlib.Path)
    args = parser.parse_args(argv)
    store = open_store(cast("str | None", args.store))
    try:
        if args.command == "record":
            document = decide(
                _find_proposal(store, cast("str", args.proposal_id)),
                verdict=cast("str", args.verdict),
                decided_by=cast("str", args.by),
                decided_at=_now(),
                note=cast("str", args.note),
                nature=cast("str | None", args.nature),
                action=cast("str | None", args.action),
                supersedes=cast("str | None", args.supersedes),
            )
            key = record_decision(store, document)
            sys.stdout.write(json.dumps({"key": key, **document}, sort_keys=True) + "\n")
            return 0
        reader = read_only(store, reason="exception_decision score is read-only")
        as_of_date = cast("dt.date | None", args.as_of) or dt.datetime.now(dt.UTC).date()
        as_of = as_of_date.isoformat()
        while not is_trading_day(dt.date.fromisoformat(as_of)):
            as_of = (dt.date.fromisoformat(as_of) - dt.timedelta(days=1)).isoformat()
        document = score(
            load_proposals(reader, until=as_of),
            load_decisions(reader),
            as_of=as_of,
            generated_at=_now(),
        )
    except DecisionError as exc:
        sys.stderr.write(f"refused: {exc}\n")
        return 2
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    out = cast("pathlib.Path | None", args.out)
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
