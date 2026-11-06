"""The three-way corporate-action check: Polygon against IB against the ledger.

`alpha-engine-config-I13197`: the back office control standard's row
"Three-way check" (nous-ergon-ops `plans/2026-11-02-back-office-implementation.md`;
1g, `-I13128`, reads both sources). Polygon's daily scan
(:mod:`crucible_trader.corporate_action_feed`) is the independent source the
ledger books from; IB's own word is the second source; the ledger
(:mod:`crucible_trader.ledger`) is the book of record. Each share-affecting
action is compared across all three, and a disagreement is a
`corporate_action_source_mismatch` finding of its own: filed as an
``exception.v1`` record, proposed, queued, aged and paged like every other
finding. It is never resolved by preferring one source.

**Which IB source.** The one the trader already reads: the paper account's
statement (``positions()`` share counts, :mod:`crucible_trader.broker_statement`)
for the prior session and today, net of the day's executions
(``reqExecutions()``). Whatever IB did to a holding outside a trade is its own
word on a corporate action, and reading it needs no new credential, grant or
connection. IB's itemised corporate-action notices (the Flex statement's
CorporateActions section) would need a Flex token, a new secret on the trading
box, so they are not read here (`alpha-engine-config-I13196`).

**What each leg is**, as a non-trade share change on the day:

* ``polygon``: what the day's splits do to IB's prior-session holding;
* ``ib``: today's shares minus the prior session's minus the day's executions;
* ``ledger``: what the ledger books for the same splits on ITS prior holding,
  by the same rule as :func:`crucible_trader.ledger.book_day` (whole shares
  only; cash in lieu is not modelled). ``None`` when the ledger was not
  readable for the account: that leg is then recorded as unavailable, the two
  other legs are still compared, and the ledger's own step reports why.

**What is NOT a mismatch, stated rather than hidden.**

* A split IB has not applied yet (its shares are the pre-action holding):
  that is the reconciler's own `corporate_action` finding, so the row is
  ``ib_not_applied`` and names it. One disagreement is one finding (the
  control arm requires exactly one per planted break).
* An action the reconciler cannot apply (a non-whole split, a spin-off): it is
  already an `unhandled_corporate_action` finding; the row records it as
  ``not_comparable``.
* IB moving shares outside a trade with NO Polygon action: recorded as an
  ``ib_only`` row and already a `share_delta` finding. It is not called a
  corporate action, because a missed execution (the gateway's day boundary,
  `alpha-engine-config-I12376` class 1) has the same signature.
* A cash dividend: IB's account summary carries one cash figure, so its leg is
  the reconciler's aggregate cash tie (credited, not credited, or a cash
  break), recorded per dividend row. A dividend IB has not credited is already
  a `corporate_action` finding and an owed dividend.

Pure: no store, no broker, no clock.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol


class CorporateAction(Protocol):
    """The fields of :class:`crucible_trader.reconciliation.CorporateAction`
    read here (a Protocol, because the reconciler imports this module)."""

    @property
    def ticker(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def ratio(self) -> float | None: ...


CHECK_SCHEMA_VERSION = "corporate_action_check.v1"

#: The finding class a disagreement between the sources raises.
MISMATCH_CLASS = "corporate_action_source_mismatch"

IB_SOURCE = (
    "ib_paper statement: positions() share counts, prior session vs today, net of the "
    "day's reqExecutions()"
)


def _ledger_split_change(held: int, actions: Iterable[CorporateAction]) -> int:
    """The share change :func:`crucible_trader.ledger.book_day` books for the
    day's splits on ``held``: each whole-share rebase, nothing for the rest."""
    change = 0
    for action in actions:
        if action.ratio is None or action.ratio <= 0:
            continue
        rebased = held * action.ratio
        if float(rebased).is_integer() and int(rebased) != held:
            change += int(rebased) - held
    return change


def _polygon_split_change(held: int, actions: Iterable[CorporateAction]) -> int | None:
    """What the day's splits do to ``held``, as the reconciler applies them
    (ratios compounded). ``None`` when the result is not whole shares."""
    ratio = 1.0
    for action in actions:
        if action.ratio is None or not math.isfinite(action.ratio) or action.ratio <= 0:
            return None
        ratio *= action.ratio
    rebased = held * ratio
    if not float(rebased).is_integer():
        return None
    return int(rebased) - held


def share_rows(
    *,
    anchor: Mapping[str, int],
    broker: Mapping[str, int],
    net_fills: Mapping[str, int],
    actions: Mapping[str, Sequence[CorporateAction]],
    resolved: frozenset[str],
    ledger: Mapping[str, int] | None,
    handled_kinds: tuple[str, ...],
) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]  # JSON rows
    """One row per instrument where any source shows a share-affecting event.

    Only RESOLVED instruments are compared: an unresolved one is already a
    `corporate_action_check_incomplete` finding, and comparing it would read
    "no Polygon action" into a feed that never answered."""
    rows: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
    for ticker in sorted(set(anchor) | set(broker) | set(net_fills) | set(actions)):
        if ticker not in resolved:
            continue
        held = anchor.get(ticker, 0)
        ib = broker.get(ticker, 0) - held - net_fills.get(ticker, 0)
        share_actions = [a for a in actions.get(ticker, ()) if a.kind != "cash_dividend"]
        if not share_actions:
            if ib:
                rows.append(
                    {
                        "instrument": ticker,
                        "polygon": None,
                        "polygon_shares": 0,
                        "ib_shares": ib,
                        "ledger_shares": None if ledger is None else 0,
                        "status": "ib_only",
                        "finding": "share_delta",
                        "detail": "IB moved shares outside a trade and Polygon reports no "
                        "action; reported as share_delta (a missed execution looks the same)",
                    }
                )
            continue
        described = [
            f"{a.kind} {a.ratio!r}" if a.kind == "split" else a.kind for a in share_actions
        ]
        splits = [a for a in share_actions if a.kind == "split"]
        polygon = (
            None
            if any(a.kind not in handled_kinds for a in share_actions)
            else _polygon_split_change(held, splits)
        )
        ours = None if ledger is None else _ledger_split_change(ledger.get(ticker, 0), splits)
        row: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
            "instrument": ticker,
            "polygon": described,
            "polygon_shares": polygon,
            "ib_shares": ib,
            "ledger_shares": ours,
        }
        if polygon is None:
            row["status"] = "not_comparable"
            row["finding"] = "unhandled_corporate_action"
            row["detail"] = "an action the reconciler cannot apply; unhandled_corporate_action"
        elif ib == polygon and ours in (None, polygon):
            row["status"] = "agree"
            row["finding"] = None
            row["detail"] = "every source agrees" + (
                "" if ours is not None else " (ledger leg unavailable)"
            )
        elif ib == 0 and polygon != 0:
            # IB still shows the pre-action holding: the reconciler's own
            # `corporate_action` finding ("not yet applied at the broker") is
            # this disagreement's one finding.
            row["status"] = "ib_not_applied"
            row["finding"] = "corporate_action"
            row["detail"] = (
                f"Polygon {described} implies {polygon:+d} shares; IB has not applied it"
            )
        else:
            row["status"] = "mismatch"
            row["finding"] = MISMATCH_CLASS
            row["detail"] = (
                f"Polygon {described} implies {polygon:+d} shares; IB moved {ib:+d} outside "
                "a trade; the ledger books "
                + ("n/a (unavailable)" if ours is None else f"{ours:+d}")
                + "; no source is preferred, find which one is wrong"
            )
        rows.append(row)
    return rows


def dividend_rows(
    dividends: Mapping[str, float],
    *,
    unpaid: Iterable[str],
    cash_break: bool,
) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]  # JSON rows
    """One row per dividend paying today: Polygon's amount and IB's aggregate
    cash tie. The ledger books the receivable from the same Polygon row."""
    not_credited = set(unpaid)
    rows: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
    for ticker, amount in sorted(dividends.items()):
        if not amount:
            continue
        status = (
            "ib_not_credited" if ticker in not_credited else "cash_break" if cash_break else "agree"
        )
        rows.append(
            {
                "instrument": ticker,
                "polygon_usd": round(amount, 2),
                "ib": status,
                "status": "agree" if status == "agree" else "reported",
                "detail": {
                    "agree": "IB's cash ties with this dividend credited",
                    "ib_not_credited": "IB has not credited it; a corporate_action finding "
                    "and an owed dividend",
                    "cash_break": "IB's cash does not tie; the cash_delta finding covers it",
                }[status],
            }
        )
    return rows


def check_document(
    shares: list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
    dividends: list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
    *,
    ledger_source: str,
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # JSON document
    """The section the broker reconciliation document carries every day."""
    return {
        "schema_version": CHECK_SCHEMA_VERSION,
        "sources": {
            "polygon": "nousergon-data corporate_actions/scans (corporate_action_scan.v1)",
            "ib": IB_SOURCE,
            "ledger": ledger_source,
        },
        "shares": shares,
        "dividends": dividends,
        "mismatches": sum(1 for r in shares if r["status"] == "mismatch"),
        "disagreements": sum(1 for r in shares if r["status"] in ("mismatch", "ib_not_applied")),
    }
