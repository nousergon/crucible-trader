"""The sealed fire-drill schedule: an unannounced drill is committed to before it fires.

`alpha-engine-config-I10761`. `announced: false` on a drill document used to be
a flag whoever fired the drill set. Now an unannounced drill fires only under a
SEAL, and the harness counts it as unannounced only when the seal verifies
(`crucible.gate._verify_drill_seal`):

    seal     (operator identity, before the window)   :func:`seal_schedule`
        a nonce -> SSM SecureString `NONCE_PARAMETER_ROOT{schedule_id}`
        `fire_drill_schedule.v1` holding sha256(fire_instant || nonce)
            -> `trader/fire_drills/schedule/{window_start}_{window_end}/{schedule_id}.json`
    fire     (the drill, at the sealed instant)       :func:`open_seal`
        the operator hands the nonce to the drill; the drill checks it opens the
        commitment and that it is firing at the sealed instant, THEN fires, and
        its `fire_drill.v1` document reveals instant and nonce

**Why the nonce is an SSM SecureString, not a store key.** The trader's session
identity must not read it before the fire: with it, a session's ~23,400 seconds
enumerate against the commitment in milliseconds. A SecureString is encrypted
under KMS, access to it is one IAM resource per parameter that CloudTrail
records on every read by default (S3 object reads need paid data events), and it
sits outside the store prefixes the trader is granted -- so a later widening of
the trader's store grants cannot expose it. The operated stack explicitly DENIES
the trader this path (and the seal prefix's write), rather than relying on the
absence of a grant, because `AmazonSSMManagedInstanceCore` grants
`ssm:GetParameter` on `*` and is one attachment from any box:
nous-ergon-ops `tests/test_crucible_v2_trader_drill_seal_denied.py` evaluates
that. The two literals are restated there and here; this module's test asserts
this side.

**The seal time is the store's, never the sealer's.** The schedule document has
no written-at field; the gate reads the object's S3 `LastModified`. So the seal
is created with a conditional create (`compare_and_swap` from absent), never
overwritten: an overwrite would move that time to after the fire and void it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import secrets
import uuid
from collections.abc import Callable
from typing import Any

from crucible.keys import trader_fire_drill_schedule_key
from crucible.models import (
    FIRE_DRILL_COMMITMENT_SCHEME,
    FIRE_DRILL_SEAL_TOLERANCE_SECONDS,
    FireDrillScheduleDocument,
    FireDrillScheduleReveal,
    fire_drill_commitment,
)
from crucible.store import ETAG_ABSENT, Store
from krepis.trading_calendar import is_market_hours, is_trading_day
from pydantic import ValidationError

#: The SSM parameter path each nonce is written under; the operated stack's
#: TraderRole Deny names the same root (see the module docstring).
NONCE_PARAMETER_ROOT = "/crucible-v2/trader/fire-drill-nonce/"

#: Where `fire-drill run --unannounced` reads the nonce the operator hands it.
#: An environment variable, not an argument: an argv lands in shell history and
#: in the process table before the fire, which is exactly when it must not be
#: readable.
NONCE_ENV_VAR = "CRUCIBLE_TRADER_DRILL_NONCE"

SCHEDULE_SCHEMA_VERSION = "fire_drill_schedule.v1"


class SealRefusedError(RuntimeError):
    """Nothing was sealed: the window or the instant cannot seal a drill."""


class SealDoesNotOpenError(RuntimeError):
    """The seal a drill names does not open, or it is not the sealed instant.
    Raised before anything fires."""


def nonce_parameter_name(schedule_id: str) -> str:
    return f"{NONCE_PARAMETER_ROOT}{schedule_id}"


def _instant(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


@dataclasses.dataclass(frozen=True)
class Seal:
    """What the sealer may print: everything except the nonce."""

    schedule_id: str
    key: str
    parameter: str
    window_start: str
    window_end: str
    fire_instant: str


def seal_schedule(
    store: Store,
    ssm: Any,
    *,
    window_start: str,
    window_end: str,
    fire_instant: str,
    clock: Callable[[], dt.datetime],
    token: Callable[[], str] = lambda: secrets.token_hex(32),
    new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> Seal:
    """Seal one drill for ``fire_instant`` inside ``[window_start, window_end]``.

    Refuses (and writes nothing) unless both window bounds are NYSE sessions, the
    instant is a canonical `YYYY-MM-DDTHH:MM:SSZ` inside the window's regular
    hours, and it is still in the future. The nonce is written first: a seal
    whose nonce write failed must not exist, while an orphan nonce with no seal
    opens nothing.
    """
    schedule_id, nonce = new_id(), token()
    try:
        FireDrillScheduleReveal.model_validate(
            {
                "schedule_id": schedule_id,
                "window_start": window_start,
                "window_end": window_end,
                "fire_instant": fire_instant,
                "nonce": nonce,
            }
        )
    except ValidationError as exc:
        raise SealRefusedError(
            f"cannot seal: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}"
        ) from exc
    for bound in (window_start, window_end):
        if not is_trading_day(dt.date.fromisoformat(bound)):
            raise SealRefusedError(f"window bound {bound} is not an NYSE trading day")
    at = _instant(fire_instant)
    if not window_start <= fire_instant[:10] <= window_end:
        raise SealRefusedError(f"{fire_instant} is outside the window {window_start}..{window_end}")
    if not is_market_hours(at):
        raise SealRefusedError(
            f"{fire_instant} is outside regular NYSE hours; a drill fired there would be refused"
        )
    if at <= clock():
        raise SealRefusedError(
            f"{fire_instant} is not in the future; a seal written at or after its fire "
            "seals nothing"
        )
    document = FireDrillScheduleDocument.model_validate(
        {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "schedule_id": schedule_id,
            "window_start": window_start,
            "window_end": window_end,
            "commitment_scheme": FIRE_DRILL_COMMITMENT_SCHEME,
            "commitment": fire_drill_commitment(fire_instant, nonce),
        }
    ).model_dump()
    key = trader_fire_drill_schedule_key(window_start, window_end, schedule_id)
    parameter = nonce_parameter_name(schedule_id)
    ssm.put_parameter(
        Name=parameter,
        Value=nonce,
        Type="SecureString",
        Overwrite=False,
        Description=f"fire-drill seal nonce for {key}",
    )
    store.compare_and_swap(
        key, ETAG_ABSENT, json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    )
    return Seal(schedule_id, key, parameter, window_start, window_end, fire_instant)


@dataclasses.dataclass(frozen=True)
class SealOpening:
    """What the operator hands an unannounced drill at fire time."""

    schedule_id: str
    window_start: str
    window_end: str
    fire_instant: str
    nonce: str


def open_seal(store: Store, opening: SealOpening, *, now: dt.datetime) -> dict[str, Any]:
    """Verify ``opening`` against its sealed schedule; return the reveal the drill
    document carries. Raises :class:`SealDoesNotOpenError` -- before any broker
    call -- when the seal is absent, the nonce does not open the commitment, or
    ``now`` is not within the sealed instant's tolerance."""
    try:
        reveal = FireDrillScheduleReveal.model_validate(dataclasses.asdict(opening))
    except ValidationError as exc:
        raise SealDoesNotOpenError(
            f"not a seal opening: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}"
        ) from exc
    key = trader_fire_drill_schedule_key(reveal.window_start, reveal.window_end, reveal.schedule_id)
    try:
        raw = store.get_bytes(key)
    except KeyError as exc:
        raise SealDoesNotOpenError(f"no sealed schedule at {key}") from exc
    sealed = FireDrillScheduleDocument.model_validate_json(raw)
    if fire_drill_commitment(reveal.fire_instant, reveal.nonce) != sealed.commitment:
        raise SealDoesNotOpenError(
            f"the nonce and instant {reveal.fire_instant} do not open the commitment at {key}"
        )
    late = (now - _instant(reveal.fire_instant)).total_seconds()
    if late < 0 or late > FIRE_DRILL_SEAL_TOLERANCE_SECONDS:
        raise SealDoesNotOpenError(
            f"now is {late:+.0f}s from the sealed instant {reveal.fire_instant} "
            f"(0..{FIRE_DRILL_SEAL_TOLERANCE_SECONDS}s admitted); firing now is not the sealed fire"
        )
    return reveal.model_dump()
