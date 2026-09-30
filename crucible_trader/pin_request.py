"""Whether the QUEUED trader pin needs this box's paper smoke today
(alpha-engine-config-I11545).

Brian's ruling: anyone may queue a sha at any time; the next clean post-close
evening pins it, and the pin is gated on the trader's own passing paper smoke
for that sha (`crucible.release.pin_trader`). The harness may not reach into
the trader, so the harness cannot run that smoke. This module is the trader's
half: read the request and the live pin, and say whether a smoke is owed.

    crucible_trader.pin_request.main([])   (called by the script; no `__main__`
                                            block, the coverage floor has no
                                            exclusions)

prints ONE line, ``<state> <sha|-> <reason>``, for
`scripts/trader_pin_request_smoke.sh` to act on:

* ``none``  -- nothing was ever queued.
* ``noop``  -- the pin already names the requested sha (done, or cancelled by
  requesting the current pin). Decided BEFORE ``stale`` for that reason.
* ``stale`` -- the pin no longer names the request's ``from_sha``: somebody
  moved it after asking. The harness refuses to apply it; smoking it would
  only produce evidence for a pin nobody will make.
* ``fresh`` -- the request is the one pending change: smoke ``sha``.

The classification is the harness's own (`crucible.release.pending_trader_pin`)
restated here over plain reads, because the crucible this repository pins
predates it; the next pin bump replaces :data:`REQUEST_KEY` with
`crucible.keys.TRADER_PIN_REQUEST_KEY`.

**An unreadable request is an error, never "no request".** Presence is decided
by LISTING the exact key, which the trader identity can be granted narrowly
(`s3:ListBucket` on `s3:prefix` = the key) and which, refused, raises. A
refusal exits :data:`UNREADABLE_EXIT` naming the grant, so a missing grant
fails the box unit loudly instead of reading as a quiet day forever.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from crucible.documents import load_document_bytes
from crucible.release import TRADER_PIN_KEY, read_pointer
from crucible.store import open_store

#: `crucible.keys.TRADER_PIN_REQUEST_KEY`, as a literal only until this
#: repository's crucible pin carries the constant.
REQUEST_KEY = "trader/pin_request.json"
SCHEMA_VERSION = "trader_pin_request.v1"
STORE_VAR = "CRUCIBLE_TRADER_STORE_URI"

#: Exit status when the request or the pin cannot be read. Distinct from 1 (a
#: defect) and 2 (usage), so the unit's journal says which.
UNREADABLE_EXIT = 3

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def read_trader_pin(store: Any) -> str | None:
    """The sha `trader/release_pin` names, or ``None`` when it has never been set.

    Absence is decided by LISTING the exact key, the same way the request's is,
    because `read_pointer` decides it with a HEAD, and a HEAD on an absent key
    is a 403 for an identity whose `s3:ListBucket` is prefix-conditioned.
    Measured on the executor box 2026-09-30: the pin had never been set, and
    every smoke failed `HeadObject ... Forbidden` before it could say so. A
    refused listing still raises; it is never read as "unset".
    """
    if TRADER_PIN_KEY not in set(store.list_keys(TRADER_PIN_KEY)):
        return None
    sha, _ = read_pointer(store, TRADER_PIN_KEY)
    return sha


class PinRequestUnreadableError(RuntimeError):
    """The request could not be read or does not conform. Never "none"."""


@dataclass(frozen=True)
class Decision:
    state: str
    sha: str | None
    reason: str

    def line(self) -> str:
        return f"{self.state} {self.sha or '-'} {self.reason}"


def _error_code(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", "")) or None
    return None


def _read_request(store: Any) -> dict[str, Any] | None:
    try:
        if REQUEST_KEY not in set(store.list_keys(REQUEST_KEY)):
            return None
        payload = store.get_bytes(REQUEST_KEY)
    except Exception as exc:
        code = _error_code(exc)
        hint = (
            f" The trader identity needs s3:ListBucket (s3:prefix {REQUEST_KEY!r}) and "
            f"s3:GetObject on {REQUEST_KEY!r} in the store."
            if code in ("AccessDenied", "403")
            else ""
        )
        raise PinRequestUnreadableError(
            f"cannot read {REQUEST_KEY}: {type(exc).__name__}: {exc}.{hint} This is not "
            "'no request'."
        ) from exc
    try:
        document = load_document_bytes(REQUEST_KEY, payload)
    except ValueError as exc:
        raise PinRequestUnreadableError(f"{REQUEST_KEY} is not a JSON document: {exc}") from exc
    problems = []
    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {document.get('schema_version')!r}")
    if not isinstance(document.get("sha"), str) or not _SHA_RE.match(document["sha"]):
        problems.append(f"sha {document.get('sha')!r}")
    from_sha = document.get("from_sha")
    if from_sha is not None and (not isinstance(from_sha, str) or not _SHA_RE.match(from_sha)):
        problems.append(f"from_sha {from_sha!r}")
    if problems:
        raise PinRequestUnreadableError(
            f"{REQUEST_KEY} does not conform to {SCHEMA_VERSION}: {', '.join(problems)}"
        )
    return document


def decide(store: Any) -> Decision:
    """Classify the queued request against the live pin."""
    request = _read_request(store)
    if request is None:
        return Decision("none", None, "no trader pin has been requested")
    sha, from_sha = request["sha"], request.get("from_sha")
    try:
        pin_sha = read_trader_pin(store)
    except Exception as exc:
        hint = (
            f" The trader identity needs s3:ListBucket (s3:prefix {TRADER_PIN_KEY!r}) and "
            f"s3:GetObject on {TRADER_PIN_KEY!r} in the store."
            if _error_code(exc) in ("AccessDenied", "403")
            else ""
        )
        raise PinRequestUnreadableError(
            f"cannot read {TRADER_PIN_KEY}: {type(exc).__name__}: {exc}.{hint}"
        ) from exc
    asked = f"requested by {request.get('requested_by')} at {request.get('requested_at')}"
    if pin_sha == sha:
        return Decision("noop", sha, f"the trader is already pinned to it ({asked})")
    if pin_sha != from_sha:
        return Decision(
            "stale",
            sha,
            f"the pin moved to {pin_sha or '(unset)'} after the request from "
            f"{from_sha or '(unset)'} ({asked}); the harness will not apply it",
        )
    return Decision("fresh", sha, f"pending from {from_sha or '(unset)'} ({asked})")


def main(argv: list[str] | None = None) -> int:
    if argv:
        print(f"usage: pin_request takes no arguments (got {argv})", file=sys.stderr)
        return 2
    uri = os.environ.get(STORE_VAR, "").strip()
    if not uri:
        print(f"{STORE_VAR} is unset; the trader has no default store", file=sys.stderr)
        return 2
    try:
        decision = decide(open_store(uri))
    except PinRequestUnreadableError as exc:
        print(f"pin_request: {exc}", file=sys.stderr)
        return UNREADABLE_EXIT
    print(decision.line())
    return 0
