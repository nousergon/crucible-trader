"""The reviewer's decision writer: one click on the console records one decision.

`alpha-engine-config-I13121` (back office 1d), option (b) of Brian's ruling of
2026-11-03: the console stays read-only, and a reviewer's call is written by
this small handler, deployed as its own Lambda with its own narrow role, behind
the same Cloudflare Access login as the console.

``GET`` serves the reviewer's form (:mod:`crucible_trader.exception_decision_form`)
behind the same identity check: the queue of the approver's tenants, one small
form per proposal, which posts JSON back to this same URL. It only reads.

What it does, per ``POST``:

1. **Who is asking.** The request must carry Cloudflare Access's signed
   assertion (``Cf-Access-Jwt-Assertion``). The handler verifies it itself --
   RS256 against the team's published keys, the application audience, the
   issuer and the expiry -- so a request that reaches the function by any other
   route than the Access-gated hostname is refused. The verified ``email`` is
   mapped to a plain reviewer name through an approver list the deployment
   supplies (:envvar:`DECISION_APPROVERS`); a form field never names the
   reviewer. A service token (no ``email``) is never an approver: people only.
2. **For whom.** Every approver lists the tenants whose books they may decide.
   The request names its tenant, and the proposal must be that tenant's
   (multi-tenant from day one, like the records it decides).
3. **What it writes.** It finds the stored proposal and calls
   :func:`crucible_trader.exception_decision.decide` and then
   :func:`~crucible_trader.exception_decision.record_decision`, which is
   create-only and refuses a decision that does not supersede the standing
   one. Nothing here re-implements a rule of the decision record.

Whoever proposes cannot approve: ``decide`` refuses the proposer as reviewer,
the approver list holds people only, and the deployment's IAM role is the only
identity meant to write decisions (the role and its grant live in the
operations repository, not here).

Configuration (all required; a missing one refuses every request with 503):

* ``ACCESS_TEAM_DOMAIN`` -- the Access team domain, e.g. ``team.cloudflareaccess.com``;
* ``ACCESS_AUD`` -- the Access application's audience tag;
* ``ALLOWED_ORIGIN`` -- the console's origin; a browser request from any other
  origin is refused (a cross-site form cannot set the JSON content type, and a
  script on another origin is refused here);
* ``DECISION_APPROVERS`` -- JSON ``{email: {"reviewer": name, "tenants": [..]}}``;
* ``CRUCIBLE_STORE`` -- the store URI.

Every request leaves one structured log line: the outcome, the reviewer, the
tenant, the proposal and, when written, the decision id and key. The decision
object itself is the evidence; the log is the trail of refusals.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import hmac
import json
import math
import secrets
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from crucible.store import Store, open_store, read_only

from crucible_trader.exception_decision import (
    DecisionError,
    decide,
    load_decisions,
    load_proposals,
    record_decision,
)
from crucible_trader.exception_decision_form import (
    content_security_policy,
    queue_items,
    render_queue_page,
)

__all__ = [
    "AccessDenied",
    "AccessKeys",
    "Approver",
    "KeysUnavailable",
    "WriterConfig",
    "WriterConfigError",
    "handle",
    "handler",
    "load_config",
    "verify_access_jwt",
]

#: Clock skew tolerated on ``exp``, ``nbf`` and ``iat``, in seconds.
LEEWAY_SECONDS = 30

#: The request fields; anything else is refused rather than ignored.
_REQUIRED = ("proposal_id", "tenant", "verdict")
_OPTIONAL = ("note", "nature", "action", "supersedes")

#: EMSA-PKCS1-v1_5 DigestInfo prefix for SHA-256 (RFC 8017 §9.2, note 1).
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


class AccessDenied(Exception):
    """The request carries no valid Access identity, or one that may not decide."""


class KeysUnavailable(Exception):
    """The Access signing keys could not be fetched or read."""


class WriterConfigError(ValueError):
    """The deployment's configuration is missing or malformed."""


@dataclass(frozen=True)
class Approver:
    """A person allowed to decide, and for which tenants."""

    reviewer: str
    tenants: frozenset[str]


@dataclass(frozen=True)
class WriterConfig:
    team_domain: str
    audience: str
    allowed_origin: str
    approvers: Mapping[str, Approver]
    store_uri: str

    @property
    def issuer(self) -> str:
        return f"https://{self.team_domain}"

    @property
    def certs_url(self) -> str:
        return f"{self.issuer}/cdn-cgi/access/certs"


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise WriterConfigError(f"{name} is not set")
    return value


def load_config(env: Mapping[str, str]) -> WriterConfig:
    """The writer's configuration from ``env``; raises on anything missing."""
    try:
        raw: object = json.loads(_required(env, "DECISION_APPROVERS"))
    except json.JSONDecodeError:
        raise WriterConfigError("DECISION_APPROVERS is not JSON") from None
    if not isinstance(raw, dict) or not raw:
        raise WriterConfigError("DECISION_APPROVERS is not a non-empty object")
    approvers: dict[str, Approver] = {}
    for email, entry in cast("dict[object, object]", raw).items():
        if not isinstance(email, str) or not isinstance(entry, dict):
            raise WriterConfigError("each approver is an email mapped to an object")
        fields = cast("dict[str, object]", entry)
        reviewer, tenants = fields.get("reviewer"), fields.get("tenants")
        if not isinstance(reviewer, str) or not isinstance(tenants, list) or not tenants:
            raise WriterConfigError(f"approver {email!r} needs a reviewer and tenants")
        names = cast("list[object]", tenants)
        if not all(isinstance(t, str) and t for t in names):
            raise WriterConfigError(f"approver {email!r} has a malformed tenant")
        approvers[email.strip().lower()] = Approver(reviewer, frozenset(cast("list[str]", names)))
    return WriterConfig(
        team_domain=_required(env, "ACCESS_TEAM_DOMAIN"),
        audience=_required(env, "ACCESS_AUD"),
        allowed_origin=_required(env, "ALLOWED_ORIGIN"),
        approvers=approvers,
        store_uri=_required(env, "CRUCIBLE_STORE"),
    )


def _b64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _json_object(data: bytes) -> dict[str, object]:
    value: object = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    return cast("dict[str, object]", value)


def _rs256_valid(key: tuple[int, int], signing_input: bytes, signature: bytes) -> bool:
    """RSASSA-PKCS1-v1_5 with SHA-256 (RFC 8017 §8.2.2), by the book."""
    modulus, exponent = key
    size = (modulus.bit_length() + 7) // 8
    padding = size - 3 - len(_SHA256_DIGEST_INFO) - 32
    if len(signature) != size or padding < 8:
        return False
    recovered = pow(int.from_bytes(signature, "big"), exponent, modulus).to_bytes(size, "big")
    expected = (
        b"\x00\x01"
        + b"\xff" * padding
        + b"\x00"
        + _SHA256_DIGEST_INFO
        + hashlib.sha256(signing_input).digest()
    )
    return hmac.compare_digest(recovered, expected)


def _number(claims: Mapping[str, object], name: str) -> float | None:
    value = claims.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise AccessDenied(f"claim {name} is not a number")
    return float(value)


def verify_access_jwt(
    token: str,
    *,
    keys: Callable[[str], tuple[int, int] | None],
    audience: str,
    issuer: str,
    now: float,
) -> dict[str, object]:
    """The claims of a valid Access application token; raises :class:`AccessDenied`.

    ``keys`` maps a key id to its RSA ``(modulus, exponent)``, or ``None``.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise AccessDenied("the assertion is not a JWT")
    try:
        header = _json_object(_b64url(parts[0]))
        claims = _json_object(_b64url(parts[1]))
        signature = _b64url(parts[2])
    except (ValueError, binascii.Error):
        raise AccessDenied("the assertion is malformed") from None
    kid = header.get("kid")
    if header.get("alg") != "RS256" or not isinstance(kid, str):
        raise AccessDenied("the assertion is not RS256 with a key id")
    key = keys(kid)
    if key is None:
        raise AccessDenied("the assertion is signed by an unknown key")
    if not _rs256_valid(key, f"{parts[0]}.{parts[1]}".encode("ascii"), signature):
        raise AccessDenied("the assertion's signature is invalid")
    if claims.get("iss") != issuer:
        raise AccessDenied("the assertion is from another issuer")
    aud = claims.get("aud")
    audiences = cast("list[object]", aud) if isinstance(aud, list) else [aud]
    if audience not in audiences:
        raise AccessDenied("the assertion is for another application")
    expires, not_before, issued = (_number(claims, n) for n in ("exp", "nbf", "iat"))
    if expires is None or now >= expires + LEEWAY_SECONDS:
        raise AccessDenied("the assertion has expired")
    if (not_before is not None and now + LEEWAY_SECONDS < not_before) or (
        issued is not None and now + LEEWAY_SECONDS < issued
    ):
        raise AccessDenied("the assertion is not valid yet")
    email = claims.get("email")
    if not isinstance(email, str) or not email or "common_name" in claims:
        raise AccessDenied("the assertion names no person (service tokens never decide)")
    return claims


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - https from config
        return cast("bytes", response.read())


class AccessKeys:
    """The Access team's signing keys, cached; refetched hourly or on an unknown
    key id (at most every ``min_refetch`` seconds), so a key rotation is picked
    up without a deploy and a forged key id cannot hammer the endpoint."""

    def __init__(
        self,
        certs_url: str,
        *,
        fetch: Callable[[str], bytes] = _http_get,
        clock: Callable[[], float] = time.time,
        ttl: float = 3600.0,
        min_refetch: float = 30.0,
    ) -> None:
        self._url, self._fetch, self._clock = certs_url, fetch, clock
        self._ttl, self._min_refetch = ttl, min_refetch
        self._keys: dict[str, tuple[int, int]] = {}
        self._fetched_at = -math.inf

    def __call__(self, kid: str) -> tuple[int, int] | None:
        now = self._clock()
        age = now - self._fetched_at
        if age > self._ttl or (kid not in self._keys and age > self._min_refetch):
            self._refresh(now)
        return self._keys.get(kid)

    def _refresh(self, now: float) -> None:
        try:
            document = _json_object(self._fetch(self._url))
            keys: dict[str, tuple[int, int]] = {}
            for jwk in cast("list[dict[str, object]]", document.get("keys", [])):
                kid = jwk.get("kid")
                if jwk.get("kty") == "RSA" and isinstance(kid, str):
                    keys[kid] = (
                        int.from_bytes(_b64url(cast("str", jwk["n"])), "big"),
                        int.from_bytes(_b64url(cast("str", jwk["e"])), "big"),
                    )
        except (OSError, ValueError, KeyError, TypeError, AttributeError, binascii.Error) as exc:
            raise KeysUnavailable(f"cannot read the Access signing keys: {exc}") from None
        self._keys, self._fetched_at = keys, now


def _respond(status: int, body: Mapping[str, object]) -> dict[str, object]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body, sort_keys=True),
    }


def _log(status: int, outcome: str, **fields: object) -> None:
    line = {"event": "exception_decision_write", "status": status, "outcome": outcome, **fields}
    print(json.dumps(line, sort_keys=True, default=str), flush=True)


def _refuse(status: int, reason: str, **fields: object) -> dict[str, object]:
    _log(status, "refused", reason=reason, **fields)
    return _respond(status, {"error": reason})


def _parse_body(event: Mapping[str, object]) -> dict[str, str | None]:
    raw = event.get("body") or ""
    if not isinstance(raw, str):
        raise ValueError("the body is not text")
    text = base64.b64decode(raw).decode("utf-8") if event.get("isBase64Encoded") else raw
    body = _json_object(text.encode("utf-8"))
    unknown = sorted(set(body) - {*_REQUIRED, *_OPTIONAL})
    if unknown:
        raise ValueError(f"unknown fields {unknown}")
    out: dict[str, str | None] = {}
    for name in (*_REQUIRED, *_OPTIONAL):
        value = body.get(name)
        if name in _REQUIRED and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} is required")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} is not a string")
        out[name] = value
    return out


def _now_stamp(now: float) -> str:
    return dt.datetime.fromtimestamp(now, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _approver(
    headers: Mapping[str, str],
    *,
    config: WriterConfig,
    keys: Callable[[str], tuple[int, int] | None],
    now: float,
) -> Approver | dict[str, object]:
    """The approver the verified Access identity maps to, or the refusal."""
    try:
        claims = verify_access_jwt(
            headers.get("cf-access-jwt-assertion", ""),
            keys=keys,
            audience=config.audience,
            issuer=config.issuer,
            now=now,
        )
    except KeysUnavailable as exc:
        return _refuse(503, str(exc))
    except AccessDenied as exc:
        return _refuse(401, str(exc))
    approver = config.approvers.get(cast("str", claims["email"]).strip().lower())
    if approver is None:
        return _refuse(403, "this identity is not an approver")
    return approver


def _form(event: Mapping[str, object], *, approver: Approver, store: Store) -> dict[str, object]:
    """``GET``: the queue of the approver's tenants as one form per proposal
    (:mod:`crucible_trader.exception_decision_form`). Reads only."""
    query = cast("dict[str, str]", event.get("queryStringParameters") or {})
    wanted = query.get("tenant")
    tenants = approver.tenants if wanted is None else approver.tenants & {wanted}
    who = {"reviewer": approver.reviewer, "tenant": wanted}
    if wanted is not None and not tenants:
        return _refuse(403, f"{approver.reviewer} may not decide for tenant {wanted}", **who)
    reader = read_only(store, reason="the decision form only reads proposals and decisions")
    try:
        items = queue_items(load_proposals(reader), load_decisions(reader), tenants=tenants)
    except (DecisionError, ValueError) as exc:
        return _refuse(503, f"cannot read the queue: {exc}", **who)
    nonce = secrets.token_urlsafe(18)
    _log(200, "form", items=len(items), **who)
    return {
        "statusCode": 200,
        "headers": {
            "content-type": "text/html; charset=utf-8",
            "cache-control": "no-store",
            "content-security-policy": content_security_policy(nonce),
            "referrer-policy": "no-referrer",
            "x-content-type-options": "nosniff",
        },
        "body": render_queue_page(items, reviewer=approver.reviewer, tenants=tenants, nonce=nonce),
    }


def handle(
    event: Mapping[str, object],
    *,
    config: WriterConfig,
    keys: Callable[[str], tuple[int, int] | None],
    store: Store,
    now: float,
) -> dict[str, object]:
    """One Lambda Function URL request (payload 2.0) to one HTTP response."""
    request_context = cast("dict[str, dict[str, object]]", event.get("requestContext") or {})
    method = request_context.get("http", {}).get("method")
    headers = {k.lower(): v for k, v in cast("dict[str, str]", event.get("headers") or {}).items()}
    if method == "GET":
        found = _approver(headers, config=config, keys=keys, now=now)
        if not isinstance(found, Approver):
            return found
        return _form(event, approver=found, store=store)
    if method != "POST":
        return _refuse(405, "only POST records a decision (GET serves the form)")
    origin = headers.get("origin")
    if origin is not None and origin != config.allowed_origin:
        return _refuse(403, "request from another origin")
    if not headers.get("content-type", "").startswith("application/json"):
        return _refuse(415, "the body must be application/json")
    found = _approver(headers, config=config, keys=keys, now=now)
    if not isinstance(found, Approver):
        return found
    approver = found
    who = {"reviewer": approver.reviewer}
    try:
        request = _parse_body(event)
    except (ValueError, binascii.Error) as exc:
        return _refuse(400, f"bad request: {exc}", **who)
    tenant, proposal_id = cast("str", request["tenant"]), cast("str", request["proposal_id"])
    who |= {"tenant": tenant, "proposal_id": proposal_id}
    if tenant not in approver.tenants:
        return _refuse(403, f"{approver.reviewer} may not decide for tenant {tenant}", **who)
    reader = read_only(store, reason="the decision writer only reads proposals")
    proposal = next(
        (
            p
            for p in load_proposals(reader)
            if p.get("proposal_id") == proposal_id and p.get("tenant") == tenant
        ),
        None,
    )
    if proposal is None:
        return _refuse(404, f"no stored proposal {proposal_id} for tenant {tenant}", **who)
    try:
        document = decide(
            proposal,
            verdict=cast("str", request["verdict"]),
            decided_by=approver.reviewer,
            decided_at=_now_stamp(now),
            note=request["note"] or "",
            nature=request["nature"],
            action=request["action"],
            supersedes=request["supersedes"],
        )
    except DecisionError as exc:
        return _refuse(400, str(exc), **who)
    try:
        key = record_decision(store, document)
    except DecisionError as exc:
        return _refuse(409, str(exc), **who)
    decision_id = document["decision_id"]
    _log(201, "recorded", key=key, decision_id=decision_id, verdict=document["verdict"], **who)
    return _respond(
        201,
        {
            "key": key,
            "decision_id": decision_id,
            "verdict": document["verdict"],
            "decided_by": approver.reviewer,
            "decided_at": document["decided_at"],
        },
    )


_STATE: list[tuple[WriterConfig, AccessKeys, Store]] = []


def handler(event: Mapping[str, object], context: object = None) -> dict[str, object]:
    """The Lambda entry point: configuration and keys are built once per container."""
    del context
    if not _STATE:
        import os  # noqa: PLC0415 - the environment is read only by the entry point

        try:
            config = load_config(os.environ)
        except WriterConfigError as exc:
            return _refuse(503, f"the writer is not configured: {exc}")
        _STATE.append((config, AccessKeys(config.certs_url), open_store(config.store_uri)))
    config, keys, store = _STATE[0]
    return handle(event, config=config, keys=keys, store=store, now=time.time())
