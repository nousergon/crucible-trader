"""The tier-3 evidence pack: what a reviewer needs to decide a cash break from
its queue card alone (`alpha-engine-config-I14134`, part 4 of
`alpha-engine-config-I13814`).

Four things, all derived from stored documents and none computed afresh:

- the arithmetic: start balance, fills, interest, dividends and flows against
  the expected total and IB's figure, from the reconciliation document's
  ``cash_components`` (`alpha-engine-config-I14133`);
- a raw source key for each input (the same mapping as
  :func:`crucible_trader.ledger_views.source_key`);
- the break's day-by-day history;
- the exact journal entry approving it would post, from the same
  :func:`crucible_trader.ledger.clearing_postings` the clearing booking uses.

:func:`cash_pack` is pure. A share break has no pack (``None``).
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
from collections.abc import Iterable, Mapping
from typing import cast

from crucible.calendar import is_trading_day
from crucible.keys import trader_broker_statement_key, trader_reconciliation_key

from crucible_trader.ledger import Entry, clearing_postings
from crucible_trader.order_intent_reconciliation import order_intent_reconciliation_key

#: Pack field -> the ``cash_components`` key that carries its dollars.
_COMPONENTS = (
    ("start_balance", "start_balance_usd"),
    ("fills", "fills_usd"),
    ("interest", "interest_usd"),
    ("dividends", "dividends_usd"),
    ("flows", "flows_usd"),
    ("expected", "expected_cash_usd"),
    ("broker", "broker_cash_usd"),
)

_CENT = decimal.Decimal("0.01")


def usd(value: object) -> str | None:
    """``"$1,234.56"`` for a dollar amount, ``"-$1,234.56"`` for a negative one,
    ``None`` for no amount. Rounded to the cent, half up; never ``-$0.00``."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    cents = decimal.Decimal(str(value)).quantize(_CENT, rounding=decimal.ROUND_HALF_UP)
    text = f"{abs(cents):,.2f}"
    return f"-${text}" if cents < 0 else f"${text}"


def _sub(record: Mapping[str, object], name: str, field: str) -> object:
    inner = record.get(name)
    return cast("Mapping[str, object]", inner).get(field) if isinstance(inner, Mapping) else None


def _input_sources(
    record: Mapping[str, object], day: str, anchor: object
) -> list[dict[str, object]]:
    """Each input's role and raw source key. Without an anchor day (a
    reconciliation with no anchor has no ``cash_components``) the two keys that
    name it are left out rather than guessed."""
    sources: list[dict[str, object]] = []
    if isinstance(anchor, str) and anchor:
        sources.append({"role": "start_balance", "record": trader_broker_statement_key(anchor)})
    sources.append({"role": "fills", "record": order_intent_reconciliation_key(day)})
    if isinstance(anchor, str) and anchor:
        sources.append(
            {
                "role": "interest",
                "record": [trader_broker_statement_key(anchor), trader_broker_statement_key(day)],
            }
        )
    sources.append({"role": "dividends", "record": trader_reconciliation_key(day)})
    manifest = _sub(record, "detected_by", "manifest_key")
    if isinstance(manifest, str) and manifest:
        sources.append({"role": "flows", "record": manifest})
    sources.append({"role": "broker", "record": trader_broker_statement_key(day)})
    return sources


def trading_days_between(first: str, last: str) -> list[str]:
    """Every NYSE trading day from ``first`` to ``last`` inclusive."""
    day, end = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    days: list[str] = []
    while day <= end:
        if is_trading_day(day):
            days.append(day.isoformat())
        day += dt.timedelta(days=1)
    return days


def break_history(
    record: Mapping[str, object], records: Iterable[Mapping[str, object]], as_of: str
) -> list[dict[str, object]]:
    """The break's days, oldest first, from its own day to ``as_of``: ``open``
    (with that day's amount) where a stored record has the same tenant, source,
    class and instrument, ``tied`` (amount 0.0) on a trading day with none."""
    same = ("tenant", "source", "class", "instrument")
    by_day: dict[str, Mapping[str, object]] = {}
    for other in records:
        if all(other.get(f) == record.get(f) for f in same):
            by_day.setdefault(str(other.get("trading_day")), other)
    return [
        {"day": d, "amount": by_day[d].get("amount"), "status": "open"}
        if d in by_day
        else {"day": d, "amount": 0.0, "status": "tied"}
        for d in trading_days_between(str(record["trading_day"]), as_of)
    ]


def _journal_rows(gap: Entry | None) -> list[dict[str, str]] | None:
    if gap is None:
        return None
    rows: list[dict[str, str]] = []
    for posting in clearing_postings(gap):
        cents = posting.amount_cents
        if not cents:
            continue  # a share posting moves no dollars
        rows.append(
            {
                "account": posting.account,
                "debit": usd(max(cents, 0) / 100) or "$0.00",
                "credit": usd(max(-cents, 0) / 100) or "$0.00",
            }
        )
    return rows


def cash_pack(
    record: Mapping[str, object],
    reconciliation_doc: Mapping[str, object],
    history: Iterable[Mapping[str, object]],
    gap_entry: Entry | None,
    *,
    as_of: str | None = None,
) -> dict[str, object] | None:
    """The evidence pack for one exception ``record``, or ``None`` for anything
    but a dollar break. ``history`` is the stored exception records to draw the
    break's days from (:func:`break_history`, up to ``as_of``, default the
    break's own day); ``gap_entry`` is the ledger ``cash_gap`` entry citing the
    exception, or ``None`` when none does (``approval_journal`` is then null).
    The seven dollar fields are all ``None`` when the document has no
    ``cash_components``."""
    if record.get("unit") != "usd":
        return None
    day = str(record["trading_day"])
    components = reconciliation_doc.get("cash_components")
    parts: Mapping[str, object] = (
        cast("Mapping[str, object]", components) if isinstance(components, Mapping) else {}
    )
    pack: dict[str, object] = {name: usd(parts.get(key)) for name, key in _COMPONENTS}
    pack["input_sources"] = _input_sources(record, day, parts.get("anchor_trading_day"))
    pack["break_history"] = break_history(record, history, as_of or day)
    pack["approval_journal"] = _journal_rows(gap_entry)
    return pack
