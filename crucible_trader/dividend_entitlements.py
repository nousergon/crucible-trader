"""Declared dividend entitlements, and the dividends still owed by the broker.

**Declared entitlements.** A dividend is paid on the shares held at its record
date (:mod:`crucible_trader.corporate_action_feed`), read from the trader's own
stored broker statement for that session. A dividend whose record date is
BEFORE the trader's first stored statement has no such statement, and its
entitlement was left unresolved: on 2026-10-30 SPY's dividend (ex-date
2026-09-18, record session 2026-09-17, first stored statement 2026-10-09)
failed the reconcile as `corporate_action_check_incomplete`.

Brian's ruling (2026-10-30, "can't you just back into it?"): DERIVE the
entitlement rather than ask for it. Each entry in
:data:`DECLARED_ENTITLEMENTS_PATH` names the shares held at the record session,
where they were read, and is marked ``estimated``. It is then CROSS-CHECKED by
the credit itself: the owed amount (shares x amount per share) must agree with
the cash the broker credits to the cent band, or the credit is an unexplained
`cash_delta` -- a derived entitlement never confirms itself.

**Owed dividends.** :func:`owed_dividends` collects, from the stored
reconciliation results, every dividend the broker had not applied on its pay
date and has not credited since, plus every declared entitlement whose pay date
has passed and was not credited. The reconciler matches a later credit against
them (:class:`~crucible_trader.reconciliation.OwedDividend`).

**Declared waivers** (`alpha-engine-config-I12024`). An owed dividend the
broker will never credit would fail every reconcile from
:data:`~crucible_trader.reconciliation.OWED_DIVIDEND_MAX_DAYS` until it ages
out of :data:`OWED_LOOKBACK_DAYS`. Measured 2026-11-09: IB's paper account
credited neither MU's 2026-10-29 dividend (11.85, 79 shares held at the
2026-10-13 record session) nor SPY's 2026-10-30 one (18.89) -- broker cash
agreed with fills and interest alone on every session from each pay date to
2026-11-09. Each entry in :data:`DECLARED_WAIVERS_PATH` names one owed dividend
by ticker, pay date and EXACT amount, with the ruling, its date and the
evidence. A waiver takes the dividend out of the owed list and the reconciler
records it as ``dividends_waived`` (:class:`~crucible_trader.reconciliation.DividendWaiver`);
an amount off by a cent is refused, and a dividend nobody waived still fails.

The ledger books nothing for a waiver: a dividend's receivable is booked on its
pay date, and both waived pay dates precede the paper ledger's 2026-11-03
opening, so there is no receivable to write off. A waiver for a dividend whose
receivable IS on the ledger needs a write-off entry first.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from crucible_trader.reconciliation import DividendWaiver, OwedDividend

DECLARED_ENTITLEMENTS_PATH = Path(__file__).parent / "declared_dividend_entitlements.json"
DECLARED_WAIVERS_PATH = Path(__file__).parent / "declared_dividend_waivers.json"

#: The first session whose stored result is read for owed dividends. Results
#: before it predate the corporate-action source (every instrument was
#: unresolved, so no dividend was ever expected and none can be owed).
OWED_BOOTSTRAP_FROM = "2026-10-29"

#: How far back the stored results are read. Past
#: :data:`~crucible_trader.reconciliation.OWED_DIVIDEND_MAX_DAYS` an owed
#: dividend is already a finding every day, so a longer look-back adds nothing.
OWED_LOOKBACK_DAYS = 31

#: The detail line a pre-``dividends_unpaid`` result wrote for an unpaid
#: dividend (`reconciliation.reconcile`). Read only for those results.
_LEGACY_UNPAID = re.compile(
    r"cash is short exactly the (?P<ticker>\S+) dividend \((?P<amount>[0-9.]+)\)"
)


class DeclaredEntitlementError(ValueError):
    """The declared-entitlements file cannot be read. Raised, never skipped."""


def load_declared_entitlements(
    path: Path = DECLARED_ENTITLEMENTS_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    """The declared entitlements keyed by ``(ticker, pay_date)``."""
    document = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(document, dict):
        raise DeclaredEntitlementError(f"{path.name}: not a JSON object")
    top = cast("dict[str, object]", document)
    if top.get("schema_version") != 1:
        raise DeclaredEntitlementError(f"{path.name}: not a schema_version 1 document")
    entries = top.get("entitlements")
    if not isinstance(entries, list):
        raise DeclaredEntitlementError(f"{path.name}: `entitlements` is not a list")
    declared: dict[tuple[str, str], dict[str, object]] = {}
    for index, raw in enumerate(cast("list[object]", entries)):
        where = f"{path.name} entitlements[{index}]"
        if not isinstance(raw, dict):
            raise DeclaredEntitlementError(f"{where}: not an object")
        entry = cast("dict[str, object]", raw)
        for field in ("ticker", "pay_date", "record_session", "source", "ruling"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DeclaredEntitlementError(f"{where}: `{field}` must be a non-empty string")
        shares, per_share = entry.get("entitled_shares"), entry.get("amount_per_share")
        if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
            raise DeclaredEntitlementError(f"{where}: `entitled_shares` must be a whole count")
        if isinstance(per_share, bool) or not isinstance(per_share, int | float) or per_share <= 0:
            raise DeclaredEntitlementError(f"{where}: `amount_per_share` must be positive")
        if entry.get("estimated") is not True:
            raise DeclaredEntitlementError(
                f"{where}: a declared entitlement is derived, so `estimated` must be true"
            )
        dt.date.fromisoformat(str(entry["pay_date"]))
        dt.date.fromisoformat(str(entry["record_session"]))
        key = (str(entry["ticker"]), str(entry["pay_date"]))
        if key in declared:
            raise DeclaredEntitlementError(f"{where}: a second entitlement for {key}")
        declared[key] = entry
    return declared


def load_declared_waivers(path: Path = DECLARED_WAIVERS_PATH) -> tuple[DividendWaiver, ...]:
    """The declared dividend waivers. A malformed file is raised, never skipped."""
    document = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(document, dict):
        raise DeclaredEntitlementError(f"{path.name}: not a JSON object")
    top = cast("dict[str, object]", document)
    if top.get("schema_version") != 1:
        raise DeclaredEntitlementError(f"{path.name}: not a schema_version 1 document")
    entries = top.get("waivers")
    if not isinstance(entries, list):
        raise DeclaredEntitlementError(f"{path.name}: `waivers` is not a list")
    waivers: list[DividendWaiver] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(cast("list[object]", entries)):
        where = f"{path.name} waivers[{index}]"
        if not isinstance(raw, dict):
            raise DeclaredEntitlementError(f"{where}: not an object")
        entry = cast("dict[str, object]", raw)
        for field in ("ticker", "pay_date", "ruling", "ruled_on"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DeclaredEntitlementError(f"{where}: `{field}` must be a non-empty string")
        amount = entry.get("amount_usd")
        if isinstance(amount, bool) or not isinstance(amount, int | float) or amount <= 0:
            raise DeclaredEntitlementError(f"{where}: `amount_usd` must be positive")
        if abs(amount * 100 - round(amount * 100)) > 1e-6:
            raise DeclaredEntitlementError(f"{where}: `amount_usd` must be whole cents")
        evidence = entry.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(e, str) and e.strip() for e in cast("list[object]", evidence))
        ):
            raise DeclaredEntitlementError(
                f"{where}: `evidence` must be a non-empty list of non-empty strings"
            )
        dt.date.fromisoformat(str(entry["pay_date"]))
        dt.date.fromisoformat(str(entry["ruled_on"]))
        waiver = DividendWaiver(
            ticker=str(entry["ticker"]),
            pay_date=str(entry["pay_date"]),
            amount_usd=float(amount),
            ruling=str(entry["ruling"]),
            ruled_on=str(entry["ruled_on"]),
            evidence=tuple(cast("list[str]", evidence)),
        )
        if waiver.key in seen:
            raise DeclaredEntitlementError(f"{where}: a second waiver for {waiver.key}")
        seen.add(waiver.key)
        waivers.append(waiver)
    return tuple(waivers)


def declared_owed(entry: Mapping[str, object]) -> OwedDividend:
    """The owed amount a declared entitlement implies, marked estimated."""
    shares = cast("int", entry["entitled_shares"])
    per_share = cast("float", entry["amount_per_share"])
    return OwedDividend(
        ticker=str(entry["ticker"]),
        pay_date=str(entry["pay_date"]),
        amount_usd=round(shares * per_share, 2),
        entitlement_source=f"declared: {entry['source']}",
        estimated=True,
    )


def _owed_from_document(document: Mapping[str, object], day: str) -> list[OwedDividend]:
    if "dividends_unpaid" in document:
        return [
            OwedDividend(
                ticker=str(row["ticker"]),
                pay_date=str(row["pay_date"]),
                amount_usd=float(cast("float", row["amount_usd"])),
                entitlement_source=str(row.get("entitlement_source") or ""),
                estimated=bool(row.get("estimated")),
            )
            for row in cast("list[dict[str, str | float | bool]]", document["dividends_unpaid"])
        ]
    owed: list[OwedDividend] = []
    for row in cast("list[dict[str, object]]", document.get("discrepancies") or []):
        match = _LEGACY_UNPAID.search(str(row.get("detail") or ""))
        if row.get("klass") == "corporate_action" and match:
            owed.append(OwedDividend(match["ticker"], day, float(match["amount"])))
    return owed


def owed_dividends(
    results: Mapping[str, Mapping[str, object]],
    *,
    today: str,
    declared: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[OwedDividend, ...]:
    """Dividends still owed by the broker before ``today``.

    ``results`` maps a session to its stored reconciliation document. Only
    sessions from :data:`OWED_BOOTSTRAP_FROM`, within :data:`OWED_LOOKBACK_DAYS`
    and before ``today`` are read.
    """
    floor = max(
        OWED_BOOTSTRAP_FROM,
        (dt.date.fromisoformat(today) - dt.timedelta(days=OWED_LOOKBACK_DAYS)).isoformat(),
    )
    candidates: dict[tuple[str, str], OwedDividend] = {}
    credited: set[tuple[str, str]] = set()
    for day in sorted(results):
        if not floor <= day < today:
            continue
        document = results[day]
        for owed in _owed_from_document(document, day):
            candidates.setdefault(owed.key, owed)
        for row in cast("list[dict[str, object]]", document.get("dividend_credits") or []):
            credited.add((str(row["ticker"]), str(row["pay_date"])))
    for key, entry in declared.items():
        if key[1] < today and key[1] >= floor:
            # A declared entitlement replaces nothing a stored result recorded.
            candidates.setdefault(key, declared_owed(entry))
    return tuple(owed for key, owed in sorted(candidates.items()) if key not in credited)
