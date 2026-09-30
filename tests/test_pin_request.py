"""`crucible_trader.pin_request`: whether the queued trader pin owes a smoke
(alpha-engine-config-I11545). Asserted against a real `LocalStore`, and against
a real `S3Store` over a fake client for the one case a local store cannot
produce: a 403 that must never read as "no request".
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from crucible.release import TRADER_PIN_KEY
from crucible.store import LocalStore, S3Store
from s3_fake import FakeS3

from crucible_trader import pin_request
from crucible_trader.pin_request import REQUEST_KEY, UNREADABLE_EXIT, decide, main

SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40


def _pin(store, sha: str) -> None:
    store.put_bytes(
        TRADER_PIN_KEY,
        json.dumps(
            {
                "sha": sha,
                "target": "trader",
                "pinned_at": "2026-09-08T21:00:00Z",
                "smoke_run_id": "01JG0000000000000000000000",
                "smoke_status": "ok",
                "smoke_manifest_key": f"runs/trader.smoke/2026-09-08/{sha[:12]}/run.json",
            }
        ).encode(),
    )


def _request(store, sha: str = SHA_B, from_sha: str | None = SHA_A, **overrides) -> None:
    document = {
        "schema_version": "trader_pin_request.v1",
        "sha": sha,
        "from_sha": from_sha,
        "requested_by": "brian",
        "requested_at": "2026-09-08T18:00:00Z",
        "run_url": None,
    } | overrides
    store.put_bytes(REQUEST_KEY, json.dumps(document).encode())


class TestDecide:
    def test_nothing_queued_is_none(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _pin(store, SHA_A)
        assert decide(store).state == "none"

    def test_a_request_while_the_pin_names_from_sha_is_fresh(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _pin(store, SHA_A)
        _request(store)
        decision = decide(store)
        assert (decision.state, decision.sha) == ("fresh", SHA_B)
        assert decision.line().startswith(f"fresh {SHA_B} pending from {SHA_A}")

    def test_a_request_from_an_unset_pin_is_fresh(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _request(store, from_sha=None)
        assert decide(store).state == "fresh"

    def test_the_pinned_sha_is_noop(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _pin(store, SHA_B)
        _request(store)
        assert decide(store).state == "noop"

    def test_a_pin_moved_after_the_request_is_stale(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _pin(store, SHA_C)
        _request(store)
        decision = decide(store)
        assert decision.state == "stale" and SHA_C in decision.reason

    def test_noop_is_decided_before_stale(self, tmp_path) -> None:
        """Requesting the current pin cancels, whatever `from_sha` says."""
        store = LocalStore(tmp_path)
        _pin(store, SHA_C)
        _request(store, sha=SHA_C, from_sha=SHA_A)
        assert decide(store).state == "noop"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"schema_version": "trader_pin_request.v0"},
            {"sha": "abc"},
            {"sha": None},
            {"from_sha": "ABC"},
            {"from_sha": 7},
        ],
    )
    def test_a_nonconforming_request_is_unreadable_not_none(self, tmp_path, overrides) -> None:
        store = LocalStore(tmp_path)
        _pin(store, SHA_A)
        _request(store, **overrides)
        with pytest.raises(pin_request.PinRequestUnreadableError, match="does not conform"):
            decide(store)

    def test_a_request_that_is_not_json_is_unreadable(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(REQUEST_KEY, b"{not json")
        with pytest.raises(pin_request.PinRequestUnreadableError, match="not a JSON document"):
            decide(store)

    def test_an_unreadable_pin_is_unreadable(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _request(store)
        store.put_bytes(TRADER_PIN_KEY, json.dumps({"sha": SHA_A}).encode())
        with pytest.raises(pin_request.PinRequestUnreadableError, match="cannot read trader/"):
            decide(store)


class _DeniedS3(FakeS3):
    """The trader identity without the grant: S3 answers 403 to the listing."""

    def get_paginator(self, name):
        def paginate(**kw):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")

        return SimpleNamespace(paginate=paginate)


class _HeadDeniedS3(FakeS3):
    """The trader identity as granted on the box: listing works, but a HEAD or
    GET on an ABSENT key is a 403 -- S3's answer when the caller's
    `s3:ListBucket` does not cover the key. Measured 2026-09-30 on
    `trader/release_pin`, which has never been set."""

    def head_object(self, **kw):
        if kw["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "403"}}, "HeadObject")
        return super().head_object(**kw)

    def get_object(self, **kw):
        if kw["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        return super().get_object(**kw)


class TestTheStoreRefusing:
    def _store(self, client) -> S3Store:
        return S3Store("bucket", "crucible", client=client)

    def test_a_403_is_an_error_naming_the_grant_never_none(self) -> None:
        store = self._store(_DeniedS3(lambda: dt.datetime(2026, 9, 8, tzinfo=dt.UTC)))
        with pytest.raises(pin_request.PinRequestUnreadableError) as caught:
            decide(store)
        message = str(caught.value)
        assert "AccessDenied" in message and "s3:ListBucket" in message
        assert "not 'no request'" in message

    def test_another_failure_is_an_error_without_the_grant_hint(self) -> None:
        class _Broken:
            def list_keys(self, prefix):
                raise OSError("connection reset")

        with pytest.raises(pin_request.PinRequestUnreadableError) as caught:
            decide(_Broken())
        assert "connection reset" in str(caught.value)
        assert "s3:ListBucket" not in str(caught.value)

    def test_a_real_s3_store_with_the_grant_reads_the_request(self) -> None:
        client = FakeS3(lambda: dt.datetime(2026, 9, 8, tzinfo=dt.UTC))
        store = self._store(client)
        _pin(store, SHA_A)
        _request(store)
        assert decide(store).state == "fresh"

    def test_a_never_set_pin_behind_a_head_403_reads_as_unset(self) -> None:
        store = self._store(_HeadDeniedS3(lambda: dt.datetime(2026, 9, 30, tzinfo=dt.UTC)))
        _request(store, from_sha=None)
        decision = decide(store)
        assert (decision.state, decision.sha) == ("fresh", SHA_B)
        assert pin_request.read_trader_pin(store) is None

    def test_a_set_pin_behind_a_head_403_store_is_still_read(self) -> None:
        store = self._store(_HeadDeniedS3(lambda: dt.datetime(2026, 9, 30, tzinfo=dt.UTC)))
        _pin(store, SHA_A)
        _request(store)
        assert decide(store).state == "fresh"
        assert pin_request.read_trader_pin(store) == SHA_A

    def test_a_refused_pin_listing_names_the_grant(self) -> None:
        class _PinListDenied(FakeS3):
            def get_paginator(self, name):
                inner = super().get_paginator(name)

                def paginate(**kw):
                    if kw.get("Prefix", "").endswith(TRADER_PIN_KEY):
                        raise ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")
                    return inner.paginate(**kw)

                return SimpleNamespace(paginate=paginate)

        store = self._store(_PinListDenied(lambda: dt.datetime(2026, 9, 30, tzinfo=dt.UTC)))
        _request(store, from_sha=None)
        with pytest.raises(pin_request.PinRequestUnreadableError) as caught:
            decide(store)
        message = str(caught.value)
        assert f"cannot read {TRADER_PIN_KEY}" in message and "s3:ListBucket" in message

    def test_the_error_code_reader_tolerates_a_non_aws_exception(self) -> None:
        assert pin_request._error_code(ValueError("x")) is None
        assert pin_request._error_code(ClientError({"Error": {}}, "op")) is None


class TestMain:
    def test_it_prints_one_decision_line(self, tmp_path, monkeypatch, capsys) -> None:
        store = LocalStore(tmp_path)
        _pin(store, SHA_A)
        _request(store)
        monkeypatch.setenv(pin_request.STORE_VAR, str(tmp_path))
        assert main([]) == 0
        (line,) = capsys.readouterr().out.splitlines()
        assert line.split(" ", 2)[:2] == ["fresh", SHA_B]

    def test_none_prints_a_dash_for_the_sha(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv(pin_request.STORE_VAR, str(tmp_path))
        assert main() == 0
        assert capsys.readouterr().out.startswith("none - ")

    def test_an_unreadable_request_exits_distinctly(self, tmp_path, monkeypatch, capsys) -> None:
        LocalStore(tmp_path).put_bytes(REQUEST_KEY, b"[]x")
        monkeypatch.setenv(pin_request.STORE_VAR, str(tmp_path))
        assert main([]) == UNREADABLE_EXIT
        captured = capsys.readouterr()
        assert captured.out == "" and "pin_request:" in captured.err

    def test_an_unset_store_is_a_usage_error(self, monkeypatch, capsys) -> None:
        monkeypatch.delenv(pin_request.STORE_VAR, raising=False)
        assert main([]) == 2
        assert "unset" in capsys.readouterr().err

    def test_arguments_are_refused(self, capsys) -> None:
        assert main(["--sha", SHA_B]) == 2
        assert "takes no arguments" in capsys.readouterr().err
