"""The reviewer's decision writer (alpha-engine-config-I13121, back office 1d, option b).

The load-bearing properties: nothing is written without a verified Access
identity that maps to an approver for the proposal's tenant; the reviewer name
comes from that identity and never from the form; and the write goes through
the decision record's own create-only path.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import pathlib
import random

import pytest
from botocore.exceptions import ClientError
from crucible.store import LocalStore, S3Store
from s3_fake import FakeS3

from crucible_trader import exception_decision_writer as w
from crucible_trader.exception_decision import decision_key, load_decisions, validate_decision
from crucible_trader.exception_proposal import PROPOSER, RULESET_VERSION

DAY = "2026-09-11"
NOW = 1_800_000_000.0
TEAM = "team.cloudflareaccess.com"
AUD = "a" * 64
ORIGIN = "https://console.example"
TENANT = "nous-ergon-paper"
EMAIL = "reviewer@example.com"


def _prime(rng: random.Random, bits: int) -> int:
    def probably_prime(n: int) -> bool:
        if n % 2 == 0:
            return False
        d, r = n - 1, 0
        while d % 2 == 0:
            d, r = d // 2, r + 1
        for _ in range(24):
            x = pow(rng.randrange(2, n - 1), d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    while True:
        candidate = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if probably_prime(candidate):
            return candidate


class Key:
    """A test RSA key (1024-bit; the verifier is size-generic)."""

    def __init__(self, seed: int) -> None:
        rng = random.Random(seed)
        self.e = 65537
        while True:
            p, q = _prime(rng, 512), _prime(rng, 512)
            phi = (p - 1) * (q - 1)
            if p != q and phi % self.e:
                break
        self.n, self.d = p * q, pow(self.e, -1, phi)

    def sign(self, data: bytes) -> bytes:
        size = (self.n.bit_length() + 7) // 8
        digest = (
            bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(data).digest()
        )
        em = b"\x00\x01" + b"\xff" * (size - 3 - len(digest)) + b"\x00" + digest
        return pow(int.from_bytes(em, "big"), self.d, self.n).to_bytes(size, "big")

    def jwk(self, kid: str) -> dict[str, str]:
        def enc(n: int) -> str:
            return b64(n.to_bytes((n.bit_length() + 7) // 8, "big"))

        return {"kid": kid, "kty": "RSA", "alg": "RS256", "n": enc(self.n), "e": enc(self.e)}


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="module")
def key() -> Key:
    return Key(1)


@pytest.fixture(scope="module")
def other_key() -> Key:
    return Key(2)


def token(key: Key, *, kid: str = "k1", header: dict | None = None, **overrides) -> str:
    claims = {
        "iss": f"https://{TEAM}",
        "aud": [AUD],
        "exp": NOW + 600,
        "nbf": NOW - 10,
        "iat": NOW - 10,
        "email": EMAIL,
        "type": "app",
        **overrides,
    }
    claims = {k: v for k, v in claims.items() if v is not None}
    head = b64(json.dumps(header or {"alg": "RS256", "kid": kid}).encode())
    body = b64(json.dumps(claims).encode())
    return f"{head}.{body}.{b64(key.sign(f'{head}.{body}'.encode()))}"


def proposal(n: int = 1, *, tenant: str = TENANT) -> dict[str, object]:
    return {
        "schema_version": "exception_proposal.v1",
        "proposal_id": f"{n:032x}",
        "exception_id": f"exc-{n}",
        "tenant": tenant,
        "trading_day": DAY,
        "source": "broker_reconciliation",
        "class": "share_delta",
        "nature": "defect",
        "action": "investigate_shares",
        "rationale": "why",
        "rule_id": "shares.delta",
        "ruleset_version": RULESET_VERSION,
        "promotable": False,
        "held": False,
        "proposer": PROPOSER,
        "proposed_at": "2026-09-11T21:00:00Z",
    }


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    s = LocalStore(tmp_path / "store")
    document = {
        "schema_version": "exception_proposal.v1",
        "proposals": [proposal(1), proposal(2, tenant="second-tenant")],
    }
    s.put_bytes(f"trader/exception_proposals/{DAY}.json", json.dumps(document).encode())
    return s


ENV = {
    "ACCESS_TEAM_DOMAIN": TEAM,
    "ACCESS_AUD": AUD,
    "ALLOWED_ORIGIN": ORIGIN,
    "DECISION_APPROVERS": json.dumps({EMAIL: {"reviewer": "brian", "tenants": [TENANT]}}),
    "CRUCIBLE_STORE": "unused",
}


@pytest.fixture
def config() -> w.WriterConfig:
    return w.load_config(ENV)


def event(body: object, *, jwt: str | None, method: str = "POST", **headers: str) -> dict:
    h = {"content-type": "application/json", "origin": ORIGIN, **headers}
    if jwt is not None:
        h["Cf-Access-Jwt-Assertion"] = jwt
    return {
        "requestContext": {"http": {"method": method}},
        "headers": h,
        "body": body if isinstance(body, str) else json.dumps(body),
        "isBase64Encoded": False,
    }


def call(ev, *, config, store, key: Key, now: float = NOW):
    keys = {"k1": (key.n, key.e)}
    response = w.handle(ev, config=config, keys=keys.get, store=store, now=now)
    return response["statusCode"], json.loads(response["body"])


AGREE = {"proposal_id": f"{1:032x}", "tenant": TENANT, "verdict": "agree"}


class TestRecords:
    def test_an_agreement_is_recorded_under_the_verified_reviewer(self, config, store, key, capsys):
        status, body = call(event(AGREE, jwt=token(key)), config=config, store=store, key=key)
        assert status == 201 and body["decided_by"] == "brian" and body["verdict"] == "agree"
        assert body["key"] == decision_key(DAY, body["decision_id"])
        [stored] = load_decisions(store)
        validate_decision(stored)
        assert stored["decided_by"] == "brian" and stored["decided_at"] == "2027-01-15T08:00:00Z"
        log = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert log["outcome"] == "recorded" and log["decision_id"] == body["decision_id"]
        assert EMAIL not in json.dumps(log)

    def test_a_correction_and_a_base64_body(self, config, store, key):
        request = {
            **AGREE,
            "verdict": "correct",
            "nature": "event",
            "action": "await_broker_corporate_action",
            "note": "a split",
        }
        ev = event("", jwt=token(key))
        ev["body"], ev["isBase64Encoded"] = (
            base64.b64encode(json.dumps(request).encode()).decode(),
            True,
        )
        status, body = call(ev, config=config, store=store, key=key)
        assert status == 201 and body["verdict"] == "correct"

    def test_a_changed_mind_must_supersede_the_standing_decision(self, config, store, key):
        _, first = call(event(AGREE, jwt=token(key)), config=config, store=store, key=key)
        defer = {**AGREE, "verdict": "defer", "note": "waiting on IB"}
        status, body = call(event(defer, jwt=token(key)), config=config, store=store, key=key)
        assert status == 409 and "supersede" in body["error"]
        defer["supersedes"] = first["decision_id"]
        status, _ = call(event(defer, jwt=token(key)), config=config, store=store, key=key)
        assert status == 201 and len(load_decisions(store)) == 2

    def test_a_decision_the_record_refuses_is_a_bad_request(self, config, store, key):
        bad = {**AGREE, "verdict": "correct", "nature": "event", "action": "investigate_shares"}
        status, body = call(event(bad, jwt=token(key)), config=config, store=store, key=key)
        assert status == 400 and "note" in body["error"] and load_decisions(store) == []


def s3_with_proposals(client: FakeS3) -> S3Store:
    s = S3Store("bucket", "crucible", client=client)
    document = {"schema_version": "exception_proposal.v1", "proposals": [proposal(1)]}
    s.put_bytes(f"trader/exception_proposals/{DAY}.json", json.dumps(document).encode())
    return s


class RefusingRetention(FakeS3):
    def put_object_retention(self, **kw: object) -> dict[str, object]:
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObjectRetention"
        )


def emf_lines(out: str) -> list[dict]:
    return [d for d in map(json.loads, out.strip().splitlines()) if "_aws" in d]


class TestWormRetention:
    """alpha-engine-config-I13191: a recorded decision is a book of record."""

    def test_a_decision_is_locked_compliance_to_year_end_plus_five(self, config, key, capsys):
        client = FakeS3(now=lambda: dt.datetime(2027, 1, 15, 8, 0, tzinfo=dt.UTC))
        store = s3_with_proposals(client)
        status, body = call(event(AGREE, jwt=token(key)), config=config, store=store, key=key)
        assert status == 201 and body["retention"] == "locked until 2033-01-01"
        assert client.retention[f"crucible/{body['key']}"] == {
            "Mode": "COMPLIANCE",
            "RetainUntilDate": dt.datetime(2033, 1, 1, tzinfo=dt.UTC),
        }
        assert len(client.retention) == 1  # the proposals file is not the writer's to lock
        [emf] = emf_lines(capsys.readouterr().out)
        assert emf[w.RETENTION_METRIC] == 0
        assert emf["_aws"]["CloudWatchMetrics"][0]["Namespace"] == w.RETENTION_METRIC_NAMESPACE

    def test_a_refused_lock_still_records_the_decision_and_says_so(self, config, key, capsys):
        client = RefusingRetention(now=lambda: dt.datetime(2027, 1, 15, 8, 0, tzinfo=dt.UTC))
        store = s3_with_proposals(client)
        status, body = call(event(AGREE, jwt=token(key)), config=config, store=store, key=key)
        assert status == 201 and body["retention"] == "refused"
        assert len(load_decisions(store)) == 1
        out = capsys.readouterr().out
        [emf] = emf_lines(out)
        assert emf[w.RETENTION_METRIC] == 1
        log = json.loads(out.strip().splitlines()[-1])
        assert log["retention"] == "refused" and "AccessDenied" in log["retention_reason"]

    def test_a_store_without_object_lock_locks_nothing(self, config, store, key):
        status, body = call(event(AGREE, jwt=token(key)), config=config, store=store, key=key)
        assert status == 201 and body["retention"].startswith("not applicable")

    def test_a_failure_around_the_lock_never_undoes_the_decision(
        self, config, store, key, monkeypatch, capsys
    ):
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("lock path down")

        monkeypatch.setattr(w, "lock_books_records", boom)
        status, body = call(event(AGREE, jwt=token(key)), config=config, store=store, key=key)
        assert status == 201 and body["retention"] == "refused"
        assert len(load_decisions(store)) == 1
        assert emf_lines(capsys.readouterr().out)[0][w.RETENTION_METRIC] == 1


class TestRefusesTheRequest:
    @pytest.mark.parametrize(
        ("ev", "status"),
        [
            (event(AGREE, jwt=None, method="GET"), 405),
            (event(AGREE, jwt=None, origin="https://evil.example"), 403),
            (event(AGREE, jwt=None, **{"content-type": "text/plain"}), 415),
            (event(AGREE, jwt=None), 401),
        ],
    )
    def test_before_any_identity_check(self, config, store, key, ev, status):
        assert call(ev, config=config, store=store, key=key)[0] == status
        assert load_decisions(store) == []

    def test_an_event_without_context_or_headers(self, config, store, key):
        assert call({}, config=config, store=store, key=key)[0] == 405

    @pytest.mark.parametrize(
        "body",
        [
            "not json",
            "[1]",
            {**AGREE, "decided_by": "someone-else"},
            {"tenant": TENANT, "verdict": "agree"},
            {**AGREE, "note": 5},
        ],
    )
    def test_a_malformed_body(self, config, store, key, body):
        status, response = call(event(body, jwt=token(key)), config=config, store=store, key=key)
        assert status == 400 and response["error"].startswith("bad request")

    def test_a_non_text_body(self, config, store, key):
        ev = event(AGREE, jwt=token(key))
        ev["body"] = 7
        assert call(ev, config=config, store=store, key=key)[0] == 400

    def test_an_identity_that_is_not_an_approver(self, config, store, key):
        ev = event(AGREE, jwt=token(key, email="someone@example.com"))
        assert call(ev, config=config, store=store, key=key) == (
            403,
            {"error": "this identity is not an approver"},
        )

    def test_a_tenant_the_approver_does_not_hold(self, config, store, key):
        ev = event({**AGREE, "tenant": "second-tenant", "proposal_id": f"{2:032x}"}, jwt=token(key))
        assert call(ev, config=config, store=store, key=key)[0] == 403

    def test_a_proposal_of_another_tenant_is_not_found(self, config, store, key):
        ev = event({**AGREE, "proposal_id": f"{2:032x}"}, jwt=token(key))
        assert call(ev, config=config, store=store, key=key)[0] == 404

    def test_unreachable_signing_keys(self, config, store, key):
        def keys(kid: str):
            raise w.KeysUnavailable("down")

        ev = event(AGREE, jwt=token(key))
        response = w.handle(ev, config=config, keys=keys, store=store, now=NOW)
        assert response["statusCode"] == 503


class TestVerifyAccessJwt:
    def verify(self, jwt: str, key: Key, now: float = NOW):
        return w.verify_access_jwt(
            jwt, keys={"k1": (key.n, key.e)}.get, audience=AUD, issuer=f"https://{TEAM}", now=now
        )

    def test_a_valid_token_and_a_string_audience(self, key):
        assert self.verify(token(key), key)["email"] == EMAIL
        assert self.verify(token(key, aud=AUD, nbf=None, iat=None), key)["aud"] == AUD

    @pytest.mark.parametrize(
        ("jwt_args", "reason"),
        [
            ({"iss": "https://other.cloudflareaccess.com"}, "another issuer"),
            ({"aud": ["b" * 64]}, "another application"),
            ({"exp": NOW - 31}, "expired"),
            ({"exp": None}, "expired"),
            ({"exp": "soon"}, "not a number"),
            ({"exp": True}, "not a number"),
            ({"nbf": NOW + 60}, "not valid yet"),
            ({"iat": NOW + 60}, "not valid yet"),
            ({"email": None, "common_name": "abc.access"}, "service tokens"),
            ({"common_name": "abc.access"}, "service tokens"),
            ({"kid": "k2"}, "unknown key"),
            ({"header": {"alg": "HS256", "kid": "k1"}}, "RS256"),
            ({"header": {"alg": "RS256"}}, "RS256"),
        ],
    )
    def test_refuses(self, key, jwt_args, reason):
        with pytest.raises(w.AccessDenied, match=reason):
            self.verify(token(key, **jwt_args), key)

    def test_refuses_another_keys_signature(self, key, other_key):
        with pytest.raises(w.AccessDenied, match="signature"):
            self.verify(token(other_key), key)

    @pytest.mark.parametrize("jwt", ["", "a.b", "!!!.e30.e30", "W10.e30.e30"])
    def test_refuses_a_malformed_token(self, key, jwt):
        with pytest.raises(w.AccessDenied):
            self.verify(jwt, key)

    def test_a_signature_of_the_wrong_length_or_a_tiny_key(self, key):
        assert not w._rs256_valid((key.n, key.e), b"x", b"\x01")
        assert not w._rs256_valid((2**255 + 1, 3), b"x", b"\x00" * 32)


class TestLoadConfig:
    def test_reads_every_field(self, config):
        assert config.approvers[EMAIL] == w.Approver("brian", frozenset({TENANT}))
        assert config.certs_url == f"https://{TEAM}/cdn-cgi/access/certs"

    def test_emails_match_case_insensitively(self):
        env = {**ENV, "DECISION_APPROVERS": json.dumps({" Reviewer@Example.com": ENV_ENTRY})}
        assert EMAIL in w.load_config(env).approvers

    @pytest.mark.parametrize(
        ("name", "value", "reason"),
        [
            ("ACCESS_AUD", " ", "ACCESS_AUD is not set"),
            ("DECISION_APPROVERS", "{", "not JSON"),
            ("DECISION_APPROVERS", "{}", "non-empty"),
            ("DECISION_APPROVERS", '{"x": 1}', "mapped to an object"),
            (
                "DECISION_APPROVERS",
                '{"x": {"reviewer": "b", "tenants": []}}',
                "reviewer and tenants",
            ),
            ("DECISION_APPROVERS", '{"x": {"reviewer": "b", "tenants": [""]}}', "malformed tenant"),
        ],
    )
    def test_refuses(self, name, value, reason):
        with pytest.raises(w.WriterConfigError, match=reason):
            w.load_config({**ENV, name: value})


ENV_ENTRY = {"reviewer": "brian", "tenants": [TENANT]}


class TestAccessKeys:
    def test_caches_refetches_on_rotation_and_on_expiry(self, key, other_key):
        clock = [NOW]
        fetched: list[str] = []
        documents = [
            {"keys": [key.jwk("k1"), {"kid": "ec", "kty": "EC"}]},
            {"keys": [key.jwk("k1"), other_key.jwk("k2")]},
            {"keys": [other_key.jwk("k2")]},
        ]

        def fetch(url: str) -> bytes:
            fetched.append(url)
            return json.dumps(documents[len(fetched) - 1]).encode()

        keys = w.AccessKeys("u", fetch=fetch, clock=lambda: clock[0], ttl=3600, min_refetch=30)
        assert keys("k1") == (key.n, key.e) and keys("ec") is None and len(fetched) == 1
        assert keys("k2") is None and len(fetched) == 1  # inside min_refetch
        clock[0] += 31
        assert keys("k2") == (other_key.n, other_key.e) and len(fetched) == 2
        clock[0] += 3601
        assert keys("k1") is None and len(fetched) == 3

    @pytest.mark.parametrize(
        "payload", [b"<html>", b"[]", b'{"keys": [{"kid": "k", "kty": "RSA"}]}']
    )
    def test_an_unreadable_document_is_unavailable(self, payload):
        keys = w.AccessKeys("u", fetch=lambda url: payload)
        with pytest.raises(w.KeysUnavailable):
            keys("k1")

    def test_the_default_fetch_reads_the_url(self, monkeypatch):
        class Response(io.BytesIO):
            pass

        seen: list[tuple[str, float]] = []

        def urlopen(url, timeout):
            seen.append((url, timeout))
            return Response(b'{"keys": []}')

        monkeypatch.setattr(w.urllib.request, "urlopen", urlopen)
        assert w._http_get("https://x/certs") == b'{"keys": []}' and seen == [
            ("https://x/certs", 5)
        ]


class TestHandler:
    @pytest.fixture(autouse=True)
    def fresh(self, monkeypatch):
        monkeypatch.setattr(w, "_STATE", [])

    def test_an_unconfigured_writer_refuses_everything(self, monkeypatch):
        for name in ENV:
            monkeypatch.delenv(name, raising=False)
        assert w.handler(event(AGREE, jwt=None))["statusCode"] == 503

    def test_builds_its_state_once_and_writes(self, monkeypatch, store, key, tmp_path):
        for name, value in {**ENV, "CRUCIBLE_STORE": str(tmp_path / "store")}.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(
            w.urllib.request,
            "urlopen",
            lambda url, timeout: io.BytesIO(json.dumps({"keys": [key.jwk("k1")]}).encode()),
        )
        monkeypatch.setattr(w.time, "time", lambda: NOW)
        assert w.handler(event(AGREE, jwt=token(key)), None)["statusCode"] == 201
        assert w.handler(event(AGREE, jwt=None), None)["statusCode"] == 401
        assert len(w._STATE) == 1 and len(load_decisions(store)) == 1
