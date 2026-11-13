"""Plain words for a break (alpha-engine-config-I13121, back office 1d).

Brian 2026-11-12: "I need to be able to understand what we are doing here. From
the UX alone." Every surface that lists a break (the reviewer's form, and the
console's queue through ``exception_score.v1``) takes its words from here, so a
row says what is different, what each side shows and what is proposed, and a
code or an id never stands in for a sentence.

Pure: no store, clock or network.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

__all__ = [
    "ACTION_TEXT",
    "NATURE_TEXT",
    "headline",
    "plain_fields",
    "side_figures",
    "sub",
]

#: The proposed fix in plain words (Brian 2026-11-12: "I need to be able to
#: understand what we are doing here. From the UX alone."). One sentence per
#: action in :data:`~crucible_trader.exception_proposal.ACTIONS`; a test holds
#: the two sets equal so a new action cannot reach the page as a bare code.
ACTION_TEXT: dict[str, str] = {
    "await_broker_corporate_action": "Wait for IB to book it. Nothing changes in our books yet.",
    "book_corporate_action_manually": (
        "Book the corporate action in our books by hand, from IB's notice."
    ),
    "accept_partial_fill": "Accept the partial fill as final.",
    "investigate_cash": "Find the missing or wrong cash entry before anything is booked.",
    "investigate_shares": "Find the missing or wrong share entry before anything is booked.",
    "investigate_position_count": "Find which holding is missing on one side.",
    "investigate_order": "Check the order with IB before anything is booked.",
    "rerun_after_feed": "Run the check again once the late data has arrived.",
    "refresh_anchor": "Restart the daily comparison from today's IB statement.",
    "investigate_detector": (
        "Check our own checker: the difference looks like a fault in detection."
    ),
}

#: What kind of difference the machine thinks this is, in plain words.
NATURE_TEXT: dict[str, str] = {
    "event": "a real event at the broker",
    "defect": "a mistake on our side",
    "detector": "our checker misfired or lacked data",
}


def _money(value: float) -> str:
    return f"${abs(value):,.2f}"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def sub(record: Mapping[str, object], name: str) -> Mapping[str, object]:
    """A nested object of the record, or an empty one."""
    found = record.get(name)
    return cast("Mapping[str, object]", found) if isinstance(found, Mapping) else {}


def _side_value(record: Mapping[str, object], side: str) -> float | None:
    return _number(sub(record, side).get("value"))


def _signed_money(value: float) -> str:
    return _money(value) if value >= 0 else "-" + _money(value)


def _plain(value: float) -> str:
    return f"{value:,g}"


def headline(proposal: Mapping[str, object], record: Mapping[str, object] | None) -> str:
    """One plain sentence saying what is different, from the exception record
    when there is one. Unescaped: the caller escapes."""
    rec = record or {}
    kind = proposal.get("class")
    amount = _number(rec.get("amount"))
    instrument = rec.get("instrument")
    if kind == "cash_delta" and amount is not None:
        more = amount > 0
        return f"IB shows {_money(amount)} {'more' if more else 'less'} cash than our books expect"
    if kind == "stale_anchor":
        return (
            "The daily check skipped a day, so this day's comparison started from an old statement"
        )
    if kind == "corporate_action" and instrument:
        return f"A corporate action on {instrument} is not yet booked at IB"
    detail = rec.get("detail") or proposal.get("rationale") or kind
    return str(detail)


def side_figures(record: Mapping[str, object] | None) -> list[tuple[str, str]]:
    """Our books, the broker and the difference, as ``(label, text)`` pairs."""
    if not record:
        return []
    expected, actual = _side_value(record, "expected"), _side_value(record, "actual")
    amount = _number(record.get("amount"))
    usd = record.get("unit") == "usd"
    fmt = _signed_money if usd else _plain
    cells: list[tuple[str, str]] = []
    if expected is not None:
        cells.append(("Our books", fmt(expected)))
    if actual is not None:
        cells.append(("IB says", fmt(actual)))
    if amount is not None:
        sign = "+" if amount > 0 else "-" if amount < 0 else ""
        cells.append(("Difference", sign + (_money(amount) if usd else f"{abs(amount):,g}")))
    return cells


def plain_fields(
    proposal: Mapping[str, object], record: Mapping[str, object] | None
) -> dict[str, str | None]:
    """The plain-language view of one waiting proposal, one string per column:
    the headline, what the check saw, the two sides and the difference, the
    likely kind of difference, the proposed fix and why. ``None`` where the
    source holds nothing (a month-end close line has no exception record)."""
    figures = dict(side_figures(record))
    detail = (record or {}).get("detail")
    nature = cast("str", proposal.get("nature") or "")
    action = cast("str", proposal.get("action") or "")
    likely = NATURE_TEXT.get(nature)
    return {
        "headline": headline(proposal, record),
        "what_checked": str(detail) if detail else None,
        "our_books": figures.get("Our books"),
        "broker_says": figures.get("IB says"),
        "difference": figures.get("Difference"),
        "likely_cause": likely[0].upper() + likely[1:] if likely else None,
        "proposed_fix": ACTION_TEXT.get(action, action) or None,
        "why": cast("str | None", proposal.get("rationale")),
    }
