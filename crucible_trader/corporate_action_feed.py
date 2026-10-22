"""The reconciler's corporate-action source: nousergon-data's daily Polygon scan.

`alpha-engine-config-I11076`, option A. Brian's ruling of 2026-10-22: IB is the
single source for holdings and cash, Polygon for prices and corporate actions.

**What it reads.** nousergon-data writes one strict scan per trading day,
``corporate_actions/scans/{day}.json`` (``corporate_action_scan.v1``, written by
its MorningEnrich unit D17, `corporate_actions/scan_manifest.py`). The object
exists only if every Polygon query for the day succeeded, so:

* object present: every instrument this reconcile looks at is RESOLVED, except
  the ones this module cannot settle (below), and the actions it lists are the
  day's actions;
* object absent or unreadable: nothing is resolved, and the reconciler reports
  every instrument as `corporate_action_check_incomplete`. The same state as
  before this module existed, never a false clean.

**What it turns into actions.**

* ``splits`` -> ``split`` with ``ratio = split_to / split_from``.
* ``dividends_paid`` -> ``cash_dividend`` on the PAY date, for the shares held
  at the RECORD date. Under T+1 settlement the record date is the ex-date, so
  the entitled holding is the broker's own statement for the session before
  the ex-date, read from the trader's stored statements. That is why a dividend
  is checked against a stored statement and never against today's book: a
  position sold between the record and pay dates is still paid.
* ``declared`` (a spin-off and the like) -> an action of that kind, which the
  reconciler refuses as `unhandled_corporate_action`. Loud by design.
* ``dividends_ex`` is not cash and moves no shares; it is not used here.

**What it leaves unresolved.** A paid dividend whose entitlement cannot be
established (no stored statement for the record date, no ex-date in the row)
or whose currency is not USD. Each such ticker that is in the book is reported
as `corporate_action_check_incomplete`; one that is not in the book shows up
as unexplained cash if it was in fact paid.

**Symbols.** Polygon writes share classes with a dot (``BRK.B``); IB with a
space (``BRK B``). :func:`ib_symbol` is the one mapping.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from crucible_trader.reconciliation import CorporateAction, CorporateActionSet

logger = logging.getLogger(__name__)

SCAN_SCHEMA_VERSION = "corporate_action_scan.v1"

#: Where the scans live, as ``s3://bucket/prefix``. Required on the box
#: (alpha-engine-config `crucible-trader/trader.env`); no default, because a
#: bucket name is not a code default in this fleet.
SCANS_URI_ENV = "CRUCIBLE_TRADER_CORPORATE_ACTION_SCANS_URI"


class CorporateActionScanError(ValueError):
    """A scan exists and is not a scan this module can trust. Raised, not skipped."""


@dataclasses.dataclass(frozen=True)
class ScanRead:
    """One scan as read: where from, the bytes (for the run's input hash), the
    parsed document. ``document`` is ``None`` when there is no usable scan, and
    ``reason`` then says why."""

    uri: str
    payload: bytes | None
    document: Mapping[str, Any] | None
    reason: str | None = None


def ib_symbol(polygon_ticker: str) -> str:
    """``BRK.B`` -> ``BRK B``. Every other symbol is unchanged."""
    return polygon_ticker.replace(".", " ")


def scan_uri(base_uri: str, day: str) -> str:
    return f"{base_uri.rstrip('/')}/{day}.json"


def read_scan(base_uri: str, day: str, *, s3_client: Any = None) -> ScanRead:
    """Read ``day``'s scan. A missing or unreadable object is a :class:`ScanRead`
    with no document and a reason; a present object that does not parse raises."""
    uri = scan_uri(base_uri, day)
    if not uri.startswith("s3://"):
        raise CorporateActionScanError(f"{SCANS_URI_ENV} must be an s3:// URI, got {base_uri!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if s3_client is None:
        import boto3  # noqa: PLC0415 - only the live path needs it

        s3_client = boto3.client("s3")
    try:
        # LIST the exact key first: under a prefix-conditioned list grant a GET
        # on an absent key is a 403, which would read as a permission fault.
        listing = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if not any(o.get("Key") == key for o in listing.get("Contents", ())):
            reason = f"corporate-action scan {uri} does not exist"
            logger.warning("%s; every instrument stays unresolved", reason)
            return ScanRead(uri, None, None, reason)
        payload = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 - every read failure is "unknown", reported
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
        reason = f"corporate-action scan {uri} could not be read ({code})"
        logger.warning("%s; every instrument stays unresolved", reason)
        return ScanRead(uri, None, None, reason)
    try:
        document = json.loads(payload.decode("utf-8"))
    except ValueError as exc:
        raise CorporateActionScanError(f"{uri} is not JSON: {exc}") from exc
    return ScanRead(uri, payload, document)


def _validate(document: Mapping[str, Any], day: str, uri: str) -> None:
    if document.get("schema") != SCAN_SCHEMA_VERSION:
        raise CorporateActionScanError(
            f"{uri}: schema {document.get('schema')!r} is not {SCAN_SCHEMA_VERSION!r}"
        )
    if document.get("trading_day") != day:
        raise CorporateActionScanError(
            f"{uri}: trading_day {document.get('trading_day')!r} is not {day}"
        )
    if document.get("complete") is not True:
        raise CorporateActionScanError(f"{uri}: the scan does not claim to be complete")
    for field in ("splits", "dividends_paid", "declared"):
        if not isinstance(document.get(field), list):
            raise CorporateActionScanError(f"{uri}: {field!r} is not a list")


def actions_from_scan(
    document: Mapping[str, Any],
    *,
    day: str,
    universe: Iterable[str],
    holdings_on: Callable[[str], Mapping[str, int] | None],
    previous_trading_day: Callable[[dt.date], dt.date],
    uri: str = "<scan>",
) -> CorporateActionSet:
    """The day's :class:`CorporateActionSet` from one scan document.

    ``holdings_on(day)`` returns the broker's stored positions for a session,
    or ``None`` when no statement was stored for it.
    """
    _validate(document, day, uri)
    universe = frozenset(universe)
    unresolved: set[str] = set()
    actions: list[CorporateAction] = []

    for row in document["splits"]:
        ticker = ib_symbol(str(row["ticker"]))
        if ticker not in universe:
            continue
        split_from, split_to = row.get("split_from"), row.get("split_to")
        ratio = None
        if split_from and split_to:
            ratio = float(split_to) / float(split_from)
        actions.append(CorporateAction(ticker=ticker, kind="split", ratio=ratio))

    holdings_cache: dict[str, Mapping[str, int] | None] = {}
    for row in document["dividends_paid"]:
        ticker = ib_symbol(str(row["ticker"]))
        currency = str(row.get("currency") or "").upper()
        ex_date = row.get("ex_dividend_date")
        if currency not in ("", "USD"):
            if ticker in universe:
                unresolved.add(ticker)
            continue
        if not ex_date:
            if ticker in universe:
                unresolved.add(ticker)
            continue
        record_session = previous_trading_day(dt.date.fromisoformat(str(ex_date))).isoformat()
        if record_session not in holdings_cache:
            holdings_cache[record_session] = holdings_on(record_session)
        held = holdings_cache[record_session]
        if held is None:
            if ticker in universe:
                unresolved.add(ticker)
            continue
        shares = int(held.get(ticker, 0))
        if shares == 0:
            continue
        actions.append(
            CorporateAction(
                ticker=ticker,
                kind="cash_dividend",
                amount_per_share=float(row["cash_amount"]),
                entitled_shares=shares,
            )
        )

    for row in document["declared"]:
        ticker = ib_symbol(str(row["ticker"]))
        if ticker in universe:
            actions.append(CorporateAction(ticker=ticker, kind=str(row.get("kind") or "declared")))

    if unresolved:
        logger.warning(
            "corporate-action scan %s: %d held instruments left unresolved: %s",
            uri,
            len(unresolved),
            ", ".join(sorted(unresolved)),
        )
    return CorporateActionSet(resolved=universe - unresolved, actions=tuple(actions))


def scans_uri_from_env(env: Mapping[str, str] | None = None) -> str | None:
    value = (env if env is not None else os.environ).get(SCANS_URI_ENV, "").strip()
    return value or None


def stored_holdings(store: Any) -> Callable[[str], Mapping[str, int] | None]:
    """``holdings_on`` over the trader's own stored broker statements."""
    from crucible.keys import trader_broker_statement_key  # noqa: PLC0415

    from crucible_trader.broker_statement import statement_from_document  # noqa: PLC0415

    def holdings_on(day: str) -> Mapping[str, int] | None:
        key = trader_broker_statement_key(day)
        if not store.exists(key):
            return None
        statement = statement_from_document(json.loads(store.get_bytes(key).decode("utf-8")))
        return dict(statement.positions)

    return holdings_on


def scan_provider(base_uri: str | None, *, s3_client: Any = None):
    """The live :data:`~crucible_trader.reconciliation_job.CorporateActionProvider`.

    ``base_uri`` ``None`` (the env var is not set) resolves nothing, which is
    the reconciler's own fail-loud state. A scan that was read is recorded as a
    run input, content-hashed, under its S3 URI.
    """
    from crucible_trader.reconciliation_job import previous_trading_day  # noqa: PLC0415

    def provide(ctx: Any, day: str, universe: frozenset[str]) -> CorporateActionSet:
        if base_uri is None:
            logger.warning("%s is not set; every instrument stays unresolved", SCANS_URI_ENV)
            return CorporateActionSet(resolved=frozenset())
        read = read_scan(base_uri, day, s3_client=s3_client)
        if read.document is None:
            return CorporateActionSet(resolved=frozenset())
        assert read.payload is not None
        ctx.record_input(read.uri, read.payload, SCAN_SCHEMA_VERSION)
        return actions_from_scan(
            read.document,
            day=day,
            universe=universe,
            holdings_on=stored_holdings(ctx.store),
            previous_trading_day=previous_trading_day,
            uri=read.uri,
        )

    return provide
