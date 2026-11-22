"""Telegram senders for the v2 box: the three the alert modules use.

A port of ``send_trade_alert``, ``send_daemon_status`` and ``send_trading_notice``
from the v1 executor's notifier (alpha-engine-config-I14129, part of
alpha-engine-config-I14079). Signatures and message text are byte-identical to
that notifier, and the credentials and topic ids come from the same places: the
bot token and chat id are resolved by :func:`krepis.telegram.send_message`
(``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``, else the secret store behind
:func:`krepis.telegram.get_secret`), and the forum topics from the
``FLOW_DOCTOR_TELEGRAM_THREAD_*`` environment variables. No new secret name.

**flow-doctor is not a dependency of this package.** The executor notifier
routed trade alerts and daemon status through flow-doctor and fell back to the
transport only when flow-doctor was inactive or past its daily cap. Here there
is no flow-doctor, so every message is sent through ``krepis.telegram``
directly, to the topic flow-doctor would have chosen for its severity (trades:
``#trades``; daemon status: ops-health). That makes every message behave like
an executor message that flow-doctor's cap held back: ``essential=True`` status
messages walk the re-route order (own topic, ops-health, the bare chat) and a
non-essential one is a single attempt.

Nothing here raises on a failed send: each function returns whether the message
was delivered, and logs why not.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from krepis.telegram import send_message

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

_TOPIC_ENV_BY_SEVERITY = {
    "critical": "FLOW_DOCTOR_TELEGRAM_THREAD_CRITICAL",
    "error": "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
    "warning": "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH",
    "info": "FLOW_DOCTOR_TELEGRAM_THREAD_TRADES",
}
# One ERROR per process per refusing topic; later re-routes stay WARNING so a
# broken topic is reported once a day, not once per trade.
_REROUTE_REPORTED: set[int | None] = set()


def _topic_thread_id(severity: str) -> int | None:
    raw = os.environ.get(_TOPIC_ENV_BY_SEVERITY.get(severity, ""), "")
    try:
        return int(raw) if raw.strip() else None
    except ValueError:
        return None


def _routes(severity: str) -> list[int | None]:
    """The message's own topic, then ops-health, then the chat with no topic."""
    primary = _topic_thread_id(severity)
    routes: list[int | None] = [primary]
    ops_health = _topic_thread_id("warning")
    if ops_health is not None and ops_health not in routes:
        routes.append(ops_health)
    if None not in routes:
        routes.append(None)
    return routes


def _send_routed(message: str, severity: str, label: str) -> bool:
    """Deliver ``message`` on the first route Telegram accepts, saying which worked."""
    routes = _routes(severity)
    primary = routes[0]
    for i, thread in enumerate(routes):
        text = (
            message
            if i == 0
            else f"[re-routed: Telegram topic {primary} refused this message]\n{message}"
        )
        if not send_message(text, message_thread_id=thread):
            continue
        if i == 0:
            logger.info("Telegram message sent: %s", label)
        elif primary not in _REROUTE_REPORTED:
            _REROUTE_REPORTED.add(primary)
            logger.error(
                "Telegram topic %s refused a message; delivered via topic %s instead: %s",
                primary,
                thread,
                label,
            )
        else:
            logger.warning(
                "Telegram message re-routed from topic %s to %s: %s", primary, thread, label
            )
        return True
    logger.error("Telegram message NOT delivered on any route (tried topics %s): %s", routes, label)
    return False


def _format_fill_time(fill_time: str | None) -> str:
    """Render a trade's execution timestamp in US/Eastern for the alert.

    Falls back to "now" (ET) when ``fill_time`` is missing or unparseable.
    """
    if fill_time:
        try:
            parsed = datetime.fromisoformat(str(fill_time).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            parsed = parsed.astimezone(_ET)
        except ValueError:
            parsed = datetime.now(_ET)
    else:
        parsed = datetime.now(_ET)
    return parsed.strftime("%Y-%m-%d %H:%M:%S ET")


def _format_trade_message(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str,
    source: str,
    fill_time: str | None = None,
    realized_pnl: float | None = None,
    realized_return_pct: float | None = None,
    realized_alpha_pct: float | None = None,
    days_held: int | None = None,
) -> str:
    emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "REDUCE": "\U0001f7e1"}.get(action, "⚪")
    lines = [
        f"{emoji} *{action} {ticker}*",
        f"Shares: {shares} @ ${price:.2f}",
        f"Time: {_format_fill_time(fill_time)}",
        f"Trigger: {trigger}",
        f"Source: {source}",
    ]
    # Realized P&L only applies to closing/reducing trades; an entry has no
    # prior fill to compare against, so realized_pnl stays None.
    if realized_pnl is not None:
        pct_str = f" ({realized_return_pct:+.1f}%)" if realized_return_pct is not None else ""
        lines.append(f"Realized P&L: ${realized_pnl:+,.2f}{pct_str}")
    if realized_alpha_pct is not None:
        held_str = f" | Held: {days_held}d" if days_held is not None else ""
        lines.append(f"Alpha vs SPY: {realized_alpha_pct:+.1f}%{held_str}")
    return "\n".join(lines)


def send_trade_alert(
    action: str,
    ticker: str,
    shares: int,
    price: float,
    trigger: str = "",
    source: str = "daemon",
    fill_time: str | None = None,
    realized_pnl: float | None = None,
    realized_return_pct: float | None = None,
    realized_alpha_pct: float | None = None,
    days_held: int | None = None,
) -> bool:
    """Send a Telegram push notification for a trade execution (``#trades`` topic)."""
    msg = _format_trade_message(
        action,
        ticker,
        shares,
        price,
        trigger,
        source,
        fill_time=fill_time,
        realized_pnl=realized_pnl,
        realized_return_pct=realized_return_pct,
        realized_alpha_pct=realized_alpha_pct,
        days_held=days_held,
    )
    return _send_routed(msg, "info", f"{action} {ticker}")


def send_daemon_status(message: str, *, essential: bool = False) -> bool:
    """Send a general status message (daemon start/stop, errors, IB events).

    Sends through ``krepis.telegram`` directly; flow-doctor is not a dependency
    of this package, so there is no daily cap in the path. ``essential=True``
    marks a message the operator acts on (the daily order book, a crash, a
    failed order): it is sent directly on every route in turn (ops-health, then
    the bare chat) instead of being dropped when the topic refuses it. A
    non-essential message is one attempt to the ops-health topic.
    """
    if essential:
        label = message.split("\n", 1)[0].replace("*", "").strip()
        return _send_routed(message, "warning", label)
    return bool(send_message(message, message_thread_id=_topic_thread_id("warning")))


def send_trading_notice(message: str, *, auto_fix: str) -> bool:
    """Send a daily trading notice straight to the ``#trades`` topic.

    For the v2 market-open and market-close notices (alpha-engine-config-I12024):
    they carry exactly the footer their sender declares (``auto_fix``;
    ``INFORMATIONAL`` carries none). Routes, in order: the ``#trades`` topic,
    the ops-health topic, then the chat with no topic.
    """
    routes = _routes("info")
    primary = routes[0]
    label = message.split("\n", 1)[0].replace("*", "").strip()
    for i, thread in enumerate(routes):
        if send_message(message, message_thread_id=thread, auto_fix=auto_fix):
            if i:
                logger.warning(
                    "Trading notice re-routed from topic %s to %s: %s", primary, thread, label
                )
            return True
    logger.error("Trading notice NOT delivered on any route (tried topics %s): %s", routes, label)
    return False
