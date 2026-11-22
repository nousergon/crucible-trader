"""`crucible_trader.box.notify`: the three Telegram senders ported from the v1 executor notifier.

The expected strings below are literal output of the executor's own notifier
(`executor.notifier`, crucible-executor `main`, rendered locally with the
transport stubbed and no network), pasted rather than recomputed: the port must
produce the same bytes (alpha-engine-config-I14129).
"""

from __future__ import annotations

import logging
import re

import pytest

from crucible_trader.box import notify

Sent = list[tuple[str, dict[str, object]]]

TRADES, OPS = 11, 22


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL", raising=False)
    monkeypatch.setenv("FLOW_DOCTOR_TELEGRAM_THREAD_TRADES", str(TRADES))
    monkeypatch.setenv("FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH", str(OPS))
    monkeypatch.setattr(notify, "_REROUTE_REPORTED", set())


def _transport(monkeypatch: pytest.MonkeyPatch, refuse: set[int | None] = frozenset()) -> Sent:  # type: ignore[assignment]
    sent: Sent = []

    def fake(text: str, **kwargs: object) -> bool:
        sent.append((text, kwargs))
        return kwargs.get("message_thread_id") not in refuse

    monkeypatch.setattr(notify, "send_message", fake)
    return sent


TRADE_CASES = [
    (
        dict(
            action="BUY",
            ticker="AAPL",
            shares=10,
            price=123.456,
            trigger="breakout",
            source="v2",
            fill_time="2026-11-20T14:35:07Z",
        ),
        "🟢 *BUY AAPL*\nShares: 10 @ $123.46\nTime: 2026-11-20 09:35:07 ET\nTrigger: "
        "breakout\nSource: v2",
    ),
    (
        dict(
            action="SELL",
            ticker="MSFT",
            shares=5,
            price=400.0,
            trigger="stop",
            source="v2",
            fill_time="2026-11-20T19:05:00",
            realized_pnl=-1234.5,
            realized_return_pct=-2.34,
            realized_alpha_pct=-1.26,
            days_held=12,
        ),
        "🔴 *SELL MSFT*\nShares: 5 @ $400.00\nTime: 2026-11-20 14:05:00 ET\nTrigger: "
        "stop\nSource: v2\n"
        "Realized P&L: $-1,234.50 (-2.3%)\nAlpha vs SPY: -1.3% | Held: 12d",
    ),
    (
        dict(
            action="REDUCE",
            ticker="NVDA",
            shares=3,
            price=99.5,
            fill_time="2026-07-06T13:30:00-04:00",
            realized_pnl=50.0,
        ),
        "🟡 *REDUCE NVDA*\nShares: 3 @ $99.50\nTime: 2026-07-06 13:30:00 ET\nTrigger: \nSource: "
        "daemon\n"
        "Realized P&L: $+50.00",
    ),
    (
        dict(
            action="COVER",
            ticker="XYZ",
            shares=1,
            price=1.0,
            fill_time="2026-01-15T15:00:00Z",
            realized_alpha_pct=3.0,
        ),
        "⚪ *COVER XYZ*\nShares: 1 @ $1.00\nTime: 2026-01-15 10:00:00 ET\nTrigger: \nSource: "
        "daemon\n"
        "Alpha vs SPY: +3.0%",
    ),
]


@pytest.mark.parametrize(("kwargs", "expected"), TRADE_CASES)
def test_trade_alert_text_is_the_executors(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], expected: str
) -> None:
    sent = _transport(monkeypatch)
    assert notify.send_trade_alert(**kwargs) is True  # type: ignore[arg-type]
    assert sent == [(expected, {"message_thread_id": TRADES})]


@pytest.mark.parametrize("fill_time", [None, "not a time"])
def test_a_missing_or_unparseable_fill_time_renders_now_in_eastern(
    monkeypatch: pytest.MonkeyPatch, fill_time: str | None
) -> None:
    sent = _transport(monkeypatch)
    notify.send_trade_alert("BUY", "AAPL", 1, 2.0, fill_time=fill_time)
    time_line = sent[0][0].split("\n")[2]
    assert time_line.startswith("Time: 20") and time_line.endswith(" ET")


def test_a_trade_alert_re_routes_in_the_executors_order(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sent = _transport(monkeypatch, refuse={TRADES, OPS})
    with caplog.at_level(logging.WARNING):
        assert notify.send_trade_alert(**TRADE_CASES[0][0]) is True  # type: ignore[arg-type]
    body = TRADE_CASES[0][1]
    note = f"[re-routed: Telegram topic {TRADES} refused this message]\n{body}"
    assert sent == [
        (body, {"message_thread_id": TRADES}),
        (note, {"message_thread_id": OPS}),
        (note, {"message_thread_id": None}),
    ]
    assert "delivered via topic None" in caplog.text


def test_the_second_re_route_of_a_refusing_topic_is_only_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _transport(monkeypatch, refuse={TRADES})
    with caplog.at_level(logging.WARNING):
        notify.send_trade_alert(**TRADE_CASES[0][0])  # type: ignore[arg-type]
        caplog.clear()
        notify.send_trade_alert(**TRADE_CASES[0][0])  # type: ignore[arg-type]
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_a_message_no_route_accepts_is_not_delivered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sent = _transport(monkeypatch, refuse={TRADES, OPS, None})
    with caplog.at_level(logging.ERROR):
        assert notify.send_trade_alert(**TRADE_CASES[0][0]) is False  # type: ignore[arg-type]
    assert [kw["message_thread_id"] for _, kw in sent] == [TRADES, OPS, None]
    assert "NOT delivered on any route" in caplog.text


def test_essential_status_text_and_route_order(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _transport(monkeypatch, refuse={OPS})
    assert notify.send_daemon_status("*Order book written*\nline two", essential=True) is True
    assert sent == [
        ("*Order book written*\nline two", {"message_thread_id": OPS}),
        (
            f"[re-routed: Telegram topic {OPS} refused this message]\n*Order book written*\nline "
            "two",
            {"message_thread_id": None},
        ),
    ]


def test_non_essential_status_is_one_attempt_to_ops_health(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _transport(monkeypatch, refuse={OPS})
    assert notify.send_daemon_status("plain status") is False
    assert sent == [("plain status", {"message_thread_id": OPS})]


def test_trading_notice_text_and_route_order(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _transport(monkeypatch, refuse={TRADES, OPS})
    assert notify.send_trading_notice("*Market open*\nall good", auto_fix="Not needed") is True
    assert sent == [
        ("*Market open*\nall good", {"message_thread_id": TRADES, "auto_fix": "Not needed"}),
        ("*Market open*\nall good", {"message_thread_id": OPS, "auto_fix": "Not needed"}),
        ("*Market open*\nall good", {"message_thread_id": None, "auto_fix": "Not needed"}),
    ]


def test_trading_notice_to_the_first_route_logs_nothing_and_a_total_failure_is_false(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _transport(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert notify.send_trading_notice("m", auto_fix="x") is True
    assert caplog.records == []
    _transport(monkeypatch, refuse={TRADES, OPS, None})
    with caplog.at_level(logging.ERROR):
        assert notify.send_trading_notice("m", auto_fix="x") is False
    assert "NOT delivered on any route" in caplog.text


def test_unset_or_malformed_topic_ids_fall_back_to_the_bare_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLOW_DOCTOR_TELEGRAM_THREAD_TRADES", "not-a-number")
    monkeypatch.delenv("FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH")
    sent = _transport(monkeypatch)
    assert notify.send_trading_notice("m", auto_fix="x") is True
    assert sent == [("m", {"message_thread_id": None, "auto_fix": "x"})]
    assert notify._topic_thread_id("no-such-severity") is None  # pyright: ignore[reportPrivateUsage]


def test_no_flow_doctor_import_and_no_new_secret_names() -> None:
    import pathlib

    text = pathlib.Path(notify.__file__).read_text(encoding="utf-8")
    assert "flow_doctor" not in text.replace("FLOW_DOCTOR_TELEGRAM_THREAD_", "")
    assert set(re.findall(r'"(FLOW_DOCTOR_TELEGRAM_THREAD_[A-Z_]+)"', text)) == {
        "FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL",
        "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
        "FLOW_DOCTOR_TELEGRAM_THREAD_TRADES",
    }
    assert not re.search(r"^\s*(from|import)\s+(boto3|flow_doctor|nousergon_lib)", text, re.M)
