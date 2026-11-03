"""Positions the trader inherits from the v1 executor, declared once, by name.

Brian's ruling of 2026-11-01 (`alpha-engine-config-I13005`): paper orders move
from the v1 executor to this trader from the 2026-11-03 session. The paper
account still holds v1's book, and :func:`crucible_trader.order_router.plan_orders`
refuses any account name its own book does not contain
(:class:`~crucible_trader.order_router.ForeignPositionError`). That refusal is
right for an account SHARED with a live v1 trader: a delta against v1's names
would liquidate them under it. Once v1 stops trading the same refusal would
stop every session, so the handover is declared here rather than the guard
loosened.

**``effective_from`` is a BOOK trading day**, the day
:class:`~crucible_trader.order_router.PaperOrderRouter` is opened for, not the
calendar day the orders go out. A session trades the previous close's book, so
the first session after v1 stopped (calendar 2026-11-03) trades the 2026-11-02
book. Declared as the calendar day, that session found nothing inherited and
refused SPY (``ForeignPositionError``, 2026-11-03 15:45Z, no order placed).

**What a declaration permits, and nothing more.** From ``effective_from`` on,
a name listed here that the book does not contain is planned to ZERO (sold
whole) instead of refused. A name the book does contain is sized by the book
as usual. Every other foreign name is still refused, and every risk cap still
applies to the sells. A name stays declared after it is sold, so a session
that fails half-way retries the rest the next day; a declared name the
account no longer holds plans nothing.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import cast

__all__ = [
    "DECLARED_INHERITED_PATH",
    "InheritedPositionsError",
    "inherited_for",
    "load_inherited",
]

DECLARED_INHERITED_PATH = Path(__file__).parent / "declared_inherited_positions.json"


class InheritedPositionsError(ValueError):
    """The declaration cannot be read. Raised, never skipped."""


def load_inherited(path: Path = DECLARED_INHERITED_PATH) -> tuple[str, frozenset[str]]:
    """``(effective_from, symbols)`` from the declaration file."""
    document = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(document, dict):
        raise InheritedPositionsError(f"{path.name}: not a JSON object")
    top = cast("dict[str, object]", document)
    if top.get("schema_version") != 1:
        raise InheritedPositionsError(f"{path.name}: not a schema_version 1 document")
    for field in ("effective_from", "source", "ruling"):
        value = top.get(field)
        if not isinstance(value, str) or not value.strip():
            raise InheritedPositionsError(f"{path.name}: `{field}` must be a non-empty string")
    effective_from = str(top["effective_from"])
    dt.date.fromisoformat(effective_from)
    raw = top.get("symbols")
    if not isinstance(raw, list) or not raw:
        raise InheritedPositionsError(f"{path.name}: `symbols` must be a non-empty list")
    symbols = cast("list[object]", raw)
    if not all(isinstance(s, str) and s.strip() == s and s for s in symbols):
        raise InheritedPositionsError(f"{path.name}: every symbol must be a non-empty string")
    if len(set(symbols)) != len(symbols):
        raise InheritedPositionsError(f"{path.name}: a symbol is declared twice")
    return effective_from, frozenset(cast("list[str]", symbols))


def inherited_for(trading_day: str, path: Path = DECLARED_INHERITED_PATH) -> frozenset[str]:
    """The declared names the router may sell to zero on ``trading_day``."""
    effective_from, symbols = load_inherited(path)
    if dt.date.fromisoformat(trading_day) < dt.date.fromisoformat(effective_from):
        return frozenset()
    return symbols
