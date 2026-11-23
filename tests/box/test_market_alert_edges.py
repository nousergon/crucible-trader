"""Branches of `crucible_trader.box.market_alert` the ported suite leaves unread.

The broker reads use a fake SDK client (nothing connects anywhere); every account
name below is a synthetic placeholder.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible_trader.box import market_alert as m

NOW = dt.datetime(2026, 11, 11, 21, 7, tzinfo=dt.UTC)  # 16:07 ET
DAY = "2026-11-10"


def _done(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_the_default_runner_shells_out_with_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, argv=argv)
        return _done(0, "out")

    monkeypatch.setattr(m.subprocess, "run", fake)
    assert m._run(["echo", "x"]).stdout == "out"  # pyright: ignore[reportPrivateUsage]
    assert seen["argv"] == ["echo", "x"] and seen["timeout"] == 60 and seen["check"] is False


def test_boot_time_is_none_when_systemd_does_not_say() -> None:
    assert m.booted_at(lambda _argv: _done(0, "Other=1")) is None


def test_the_checkout_on_main_is_unknown_when_git_cannot_say() -> None:
    assert m.executor_on_main(lambda _argv: _done(128, "")) is None
    assert m.executor_on_main(lambda _argv: _done(0, "onlyone\n")) is None
    assert m.executor_sha(lambda _argv: _done(128, "")) is None


def test_the_boot_pull_log_without_a_started_line_says_nothing(tmp_path: Path) -> None:
    log = tmp_path / "boot-pull.log"
    log.write_text("unrelated line\n")
    assert m.boot_pull_failures(log) is None
    assert m.boot_pull_failures(tmp_path / "absent.log") is None


def test_quantity_reads_garbage_and_ibs_unset_marker_as_zero() -> None:
    q = m._quantity  # pyright: ignore[reportPrivateUsage]
    assert q(True) == 0.0 and q("5") == 0.0 and q(None) == 0.0
    assert q(1.7976931348623157e308) == 0.0 and q(-3) == 0.0
    assert q(12) == 12.0 and q(2.5) == 2.5


def _trade(symbol: str = "SPY", action: str = "SELL", filled: float = 0, **order: Any) -> Any:
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol),
        order=SimpleNamespace(action=action, orderType="MKT", tif="DAY", totalQuantity=10, **order),
        orderStatus=SimpleNamespace(filled=filled),
    )


def test_a_broker_order_is_read_from_the_sdk_trade_with_safe_defaults() -> None:
    order = m._broker_order(_trade(filled=4))  # pyright: ignore[reportPrivateUsage]
    assert order == m.OpenOrder("SPY", "SELL", "MKT", "DAY", 10, 4)
    blank = m._broker_order(object())  # pyright: ignore[reportPrivateUsage]
    assert blank == m.OpenOrder("?", "?", "?", "?", 0, 0)


class _FakeIB:
    """The slice of the `ib_async.IB` client `read_broker` calls."""

    def __init__(self, accounts: list[str], *, connect_error: Exception | None = None) -> None:
        self.accounts = accounts
        self.connect_error = connect_error
        self.calls: list[str] = []
        self.connected_with: dict[str, Any] = {}

    def connect(self, host: str, port: int, **kwargs: Any) -> None:
        self.connected_with = {"host": host, "port": port, **kwargs}
        if self.connect_error is not None:
            raise self.connect_error

    def managedAccounts(self) -> list[str]:  # noqa: N802
        return self.accounts

    def reqAllOpenOrders(self) -> None:  # noqa: N802
        self.calls.append("reqAllOpenOrders")

    def openTrades(self) -> list[Any]:  # noqa: N802
        return [_trade("AAA", "BUY", 3)]

    def reqCompletedOrders(self, api_only: bool) -> list[Any]:  # noqa: N802
        assert api_only is True
        return [_trade("BBB", "SELL", 0, filledQuantity=10)]

    def disconnect(self) -> None:
        self.calls.append("disconnect")


def _with_sdk(monkeypatch: pytest.MonkeyPatch, ib: _FakeIB) -> None:
    monkeypatch.setattr(m, "load_sdk", lambda: SimpleNamespace(IB=lambda: ib))


def test_the_broker_read_is_readonly_and_classifies_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ib = _FakeIB(["DUPLACEHOLDER1"])
    _with_sdk(monkeypatch, ib)
    broker = m.read_broker({"CRUCIBLE_TRADER_IB_HOST": "gateway.test"}, 44, want_orders=True)
    assert broker.account_kind == "paper" and broker.error is None
    assert broker.open_orders == (m.OpenOrder("AAA", "BUY", "MKT", "DAY", 10, 3),)
    assert broker.done == (m.OpenOrder("BBB", "SELL", "MKT", "DAY", 10, 10),)
    assert ib.connected_with == {
        "host": "gateway.test",
        "port": m.PAPER_PORT,
        "clientId": 44,
        "timeout": 15,
        "readonly": True,
    }
    assert ib.calls == ["reqAllOpenOrders", "disconnect"]


@pytest.mark.parametrize(
    ("accounts", "kind"),
    [([], "unknown"), (["DUPLACEHOLDER1"], "paper"), (["PLACEHOLDER2"], "live")],
)
def test_without_orders_only_the_account_kind_is_read(
    monkeypatch: pytest.MonkeyPatch, accounts: list[str], kind: str
) -> None:
    ib = _FakeIB(accounts)
    _with_sdk(monkeypatch, ib)
    broker = m.read_broker({}, 44, want_orders=False)
    assert broker == m.Broker(account_kind=kind)
    assert ib.calls == ["disconnect"], "the session is closed even on the early return"


def test_a_port_that_is_not_a_number_is_a_failed_line_not_a_crash() -> None:
    broker = m.read_broker({"CRUCIBLE_TRADER_IB_PORT": "x"}, 44, want_orders=False)
    assert broker.error == "the trader's gateway port is not a number"


def test_a_refused_connect_is_reported_and_never_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_sdk(monkeypatch, _FakeIB([], connect_error=TimeoutError("no answer")))
    broker = m.read_broker({}, 44, want_orders=True)
    assert broker.error == "TimeoutError: no answer"


def test_an_sdk_that_cannot_load_is_reported_and_never_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise ModuleNotFoundError("No module named 'ib_async'")

    monkeypatch.setattr(m, "load_sdk", missing)
    assert m.read_broker({}, 44, want_orders=True).error == (
        "ModuleNotFoundError: No module named 'ib_async'"
    )


def test_a_failure_mid_read_is_reported_and_the_session_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ib = _FakeIB(["DUPLACEHOLDER1"])

    def boom() -> list[Any]:
        raise RuntimeError("gateway dropped")

    ib.openTrades = boom  # type: ignore[method-assign]
    _with_sdk(monkeypatch, ib)
    broker = m.read_broker({}, 44, want_orders=True)
    assert broker.error == "RuntimeError: gateway dropped" and ib.calls[-1] == "disconnect"


def test_when_reads_an_iso_instant_and_refuses_anything_else() -> None:
    assert m._when("2026-11-10T15:30:00Z") == dt.datetime(  # pyright: ignore[reportPrivateUsage]
        2026, 11, 10, 15, 30, tzinfo=dt.UTC
    )
    assert m._when("not a time") is None  # pyright: ignore[reportPrivateUsage]
    assert m._when(5) is None  # pyright: ignore[reportPrivateUsage]


# --- single lines of the checklists ---------------------------------------------------


def _open_facts(**kw: Any) -> m.OpenFacts:
    base: dict[str, Any] = {
        "today": "2026-11-11",
        "day": DAY,
        "session": m.Unit(),
        "gateway": m.Unit({"ActiveState": "active"}),
        "broker": m.Broker(),
        "book": None,
        "watch": None,
        "run": None,
        "now": NOW,
    }
    base.update(kw)
    return m.OpenFacts(**base)


def _close_facts(**kw: Any) -> m.CloseFacts:
    base: dict[str, Any] = {
        "today": "2026-11-11",
        "day": DAY,
        "session": m.Unit({"ActiveState": "inactive", "Result": "success"}),
        "broker": m.Broker(open_orders=(), done=()),
        "run": {"status": "ok"},
        "shortfall": None,
        "watch": None,
        "timers": {},
    }
    base.update(kw)
    return m.CloseFacts(**base)


def test_a_code_pull_still_running_is_a_wait_line() -> None:
    check = m._code_check(  # pyright: ignore[reportPrivateUsage]
        _open_facts(boot_pull=m.Unit({"ActiveState": "activating"}), executor_sha="abc1234")
    )
    assert check.mark == "wait" and check.text.startswith("Code pull still running")


def test_a_gateway_service_that_is_down_is_a_cross() -> None:
    check = m._gateway_check(  # pyright: ignore[reportPrivateUsage]
        _open_facts(gateway=m.Unit({"ActiveState": "failed"}))
    )
    assert check.mark == "fail" and check.text == "IB Gateway service is failed"


def test_a_finished_watch_is_ok_and_an_unknown_watch_state_is_a_cross() -> None:
    watch = {"state": "complete", "updated_at_utc": "2026-11-11T20:00:00Z"}
    done = m._watch_check(_open_facts(watch=watch))  # pyright: ignore[reportPrivateUsage]
    assert done.mark == "ok" and done.text.startswith("Entry/exit watch complete")
    odd = m._watch_check(  # pyright: ignore[reportPrivateUsage]
        _open_facts(watch={**watch, "state": "crashed"})
    )
    assert odd.mark == "fail" and odd.text.startswith("Entry/exit watch is crashed")


def test_a_session_still_running_after_the_hard_stop_is_a_cross() -> None:
    session = m.Unit({"ActiveState": "active", "SubState": "running"})
    if not session.running:  # the property's own definition decides what "running" is
        session = m.Unit({"ActiveState": "activating"})
    check = m._close_session_check(_close_facts(session=session))  # pyright: ignore[reportPrivateUsage]
    assert check.mark == "fail" and "still running after its 16:05 hard stop" in check.text


def test_a_run_record_that_ended_badly_is_a_cross_with_its_reason() -> None:
    bad = {"status": "failed", "reason": "the book was empty"}
    check = m._close_session_check(_close_facts(run=bad))  # pyright: ignore[reportPrivateUsage]
    assert check.mark == "fail" and "its run ended failed: the book was empty" in check.text
    no_reason = m._close_session_check(  # pyright: ignore[reportPrivateUsage]
        _close_facts(run={"status": "failed"})
    )
    assert "no reason recorded" in no_reason.text


def test_fills_the_run_filed_but_the_notice_could_not_read_are_a_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "run_lists_output", lambda _run, _key: True)
    check = m._trades_check(_close_facts())  # pyright: ignore[reportPrivateUsage]
    assert check.mark == "fail" and "could not be read" in check.text


def test_a_broker_that_could_not_be_read_is_a_cross_on_the_broker_line() -> None:
    check = m._broker_check(  # pyright: ignore[reportPrivateUsage]
        _close_facts(broker=m.Broker(error="TimeoutError: no answer"))
    )
    assert check.mark == "fail" and check.text.startswith("Broker not read")


# --- gathering the close, and the entry point ---------------------------------------------


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _NoSuchKey(Exception):
    pass


class _S3:
    class exceptions:  # noqa: N801 -- mirrors boto3's client.exceptions
        NoSuchKey = _NoSuchKey

    def __init__(self, objects: dict[str, Any], *, broken: set[str]) -> None:
        self.objects, self.broken = objects, broken

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key in self.broken:
            raise _ClientError("InternalError")
        if Key not in self.objects:
            raise _ClientError("AccessDenied")
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}


def test_an_unreadable_close_document_is_listed_as_unreadable() -> None:
    broken = f"runs/trader.session/{DAY}/run.json"
    store = m.Store(_S3({}, broken={broken}), "bucket", "")  # type: ignore[arg-type]
    facts = m.gather_close(
        store,
        {},
        now=NOW,
        client_id=44,
        run=lambda _argv: _done(0, ""),
        broker=lambda _env, _cid, *, want_orders: m.Broker(open_orders=(), done=()),
    )
    assert facts.unreadable == (f"{broken} (InternalError)",)
    assert facts.run is None


def _main_env(monkeypatch: pytest.MonkeyPatch, outcomes: list[m.Outcome]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setenv(m.CLIENT_ID_VAR, "44")
    monkeypatch.setattr(m, "read_env_file", lambda _path: {"CRUCIBLE_TRADER_STORE_URI": "s3://x"})
    monkeypatch.setattr(m, "open_store", lambda _env: "store")

    def fake_notify(kind: str, store: object, env: object, **kwargs: Any) -> m.Outcome:
        seen.update(kwargs, kind=kind, store=store, env=env)
        return outcomes.pop(0)

    monkeypatch.setattr(m, "notify", fake_notify)
    return seen


def test_main_runs_the_notice_and_maps_the_outcome_to_an_exit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _main_env(
        monkeypatch, [m.Outcome("sent", "sent"), m.Outcome("failed", "the notice was not sent")]
    )
    assert m.main(["open"], now=NOW) == 0
    assert seen["kind"] == "open" and seen["client_id"] == 44 and seen["store"] == "store"
    assert m.main(["close"], now=NOW) == 1
    assert seen["kind"] == "close"


def test_a_held_session_is_named_on_the_trades_line() -> None:
    held = {
        "status": "ok",
        "metrics": [
            {"name": "trader_session_served", "value": 0, "status_reason": "inputs not published"}
        ],
    }
    check = m._trades_check(_close_facts(run=held))  # pyright: ignore[reportPrivateUsage]
    assert check.mark == "fail"
    assert check.text == "Trades executed today: none — the session held (inputs not published)"
