"""`Store.read_json` of `crucible_trader.box.trade_alert`, which the ported suite never calls."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from crucible_trader.box.trade_alert import Store


class _NoSuchKey(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, Any]) -> None:
        self.objects = objects

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key == "prefix/broken.json":
            raise _ClientError("InternalError")
        if Key == "prefix/denied.json":
            raise _ClientError("AccessDenied")
        if Key not in self.objects:
            raise _NoSuchKey(Key)
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}


def _store() -> Store:
    objects = {"prefix/doc.json": {"a": 1}, "prefix/list.json": [1, 2]}
    return Store(_S3(objects), "bucket", "prefix")  # type: ignore[arg-type]


def test_read_json_returns_the_document() -> None:
    assert _store().read_json("doc.json") == {"a": 1}


def test_read_json_reads_a_non_object_document_as_absent() -> None:
    assert _store().read_json("list.json") is None


def test_read_json_reads_a_missing_key_and_access_denied_as_absent() -> None:
    assert _store().read_json("absent.json") is None
    assert _store().read_json("denied.json") is None


def test_read_json_raises_any_other_error() -> None:
    with pytest.raises(_ClientError):
        _store().read_json("broken.json")
