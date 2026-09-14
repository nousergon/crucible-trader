"""An in-memory S3 client for the store paths the drill seal touches.

`crucible.gate` counts an unannounced drill only on a seal whose S3
`LastModified` precedes the fire, and a `LocalStore` has no such time -- so the
end-to-end status test runs a real `crucible.store.S3Store` over this client.
`LastModified` is stamped by the fake from `now` on every put, never by the
caller's payload: a test controls it only the way the service does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from botocore.exceptions import ClientError


class FakeS3:
    def __init__(self, now: Callable[[], dt.datetime]) -> None:
        self.objects: dict[str, bytes] = {}
        self.modified: dict[str, dt.datetime] = {}
        self.now = now

    def _etag(self, key: str) -> str:
        return hashlib.md5(self.objects[key], usedforsecurity=False).hexdigest()

    @staticmethod
    def _error(code: str) -> ClientError:
        return ClientError({"Error": {"Code": code}}, "op")

    def put_object(self, **kw: Any) -> dict[str, Any]:
        key = kw["Key"]
        if "IfNoneMatch" in kw and key in self.objects:
            raise self._error("PreconditionFailed")
        self.objects[key] = kw["Body"]
        self.modified[key] = self.now()
        return {"ETag": f'"{self._etag(key)}"'}

    def get_object(self, **kw: Any) -> dict[str, Any]:
        if kw["Key"] not in self.objects:
            raise self._error("NoSuchKey")
        payload = self.objects[kw["Key"]]
        return {"Body": SimpleNamespace(read=lambda: payload)}

    def head_object(self, **kw: Any) -> dict[str, Any]:
        key = kw["Key"]
        if key not in self.objects:
            raise self._error("404")
        return {"ETag": f'"{self._etag(key)}"', "LastModified": self.modified[key]}

    def get_paginator(self, name: str) -> Any:
        objects = self.objects

        def paginate(**kw: Any):
            prefix = kw.get("Prefix") or ""
            yield {"Contents": [{"Key": k} for k in sorted(objects) if k.startswith(prefix)]}

        return SimpleNamespace(paginate=paginate)


class FakeSsm:
    """`ssm.put_parameter`, recorded; a second write of one name is refused the
    way `Overwrite=False` refuses it."""

    def __init__(self) -> None:
        self.parameters: dict[str, dict[str, Any]] = {}

    def put_parameter(self, **kw: Any) -> dict[str, Any]:
        if kw["Name"] in self.parameters and not kw.get("Overwrite"):
            raise ClientError({"Error": {"Code": "ParameterAlreadyExists"}}, "PutParameter")
        self.parameters[kw["Name"]] = kw
        return {"Version": 1}
