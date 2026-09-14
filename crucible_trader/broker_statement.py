"""What the broker says the account holds: positions and cash, as one document.

`alpha-engine-config-I10651`. The reconciler compares an EXPECTED book against a
:class:`BrokerStatement`, and the statement is the only thing in this package
that is allowed to speak for the broker. Two properties are enforced here rather
than requested, because the v1 reconciler lost both:

1. **A statement is the broker's own word, never a reconstruction.** v1's
   ledger-replay backfill synthesised a book from the trades table and reported
   20 positions against a real 7 (+51% NAV). There is no constructor in this
   module that builds a statement from fills or a ledger: `source` is a closed
   literal naming a broker read, and :func:`statement_from_document` refuses any
   other value. A replayed book is an EXPECTED side; it cannot be filed as the
   ACTUAL one.
2. **A statement is a paper account's, until phase 6 says otherwise.** The
   :class:`IbPaperStatementSource` adapter refuses any account id that is not an
   IB paper account (`DU…`). The paper -> live crossing is a reserved action
   (plan §9.5 row 4); a reconciler that could silently read a live account would
   be the first component to cross it.

The adapter takes an already-connected `ib_insync.IB`-shaped client rather than
constructing one. That keeps this package free of a broker SDK dependency (the
connection, its host and its client id are the box's, not the library's) and
makes the read fakeable in tests without monkeypatching an SDK.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol

#: The schema a stored statement declares.
BROKER_STATEMENT_SCHEMA_VERSION = "broker_statement.v1"

#: The one broker read this package knows. Closed on purpose (see module doc).
StatementSource = Literal["ib_paper"]
STATEMENT_SOURCES: tuple[str, ...] = ("ib_paper",)

#: IB paper accounts are issued with this prefix. A live account is `U…`.
PAPER_ACCOUNT_PREFIX = "DU"

#: The account-summary tag carrying settled cash. `NetLiquidation` is NOT read:
#: a NAV derived from the broker's own positions plus the broker's own cash
#: reconciles the broker against itself and grades green forever (v1
#: `reconciliation_audit.py`'s documented tautology).
CASH_TAG = "TotalCashValue"


class BrokerStatementError(ValueError):
    """A statement could not be read, or what was read is not a statement."""


@dataclasses.dataclass(frozen=True)
class BrokerStatement:
    """Positions (whole shares, non-zero) and cash for one account on one day."""

    account: str
    trading_day: str
    positions: Mapping[str, int]
    cash_usd: float
    source: StatementSource = "ib_paper"

    def __post_init__(self) -> None:
        if self.source not in STATEMENT_SOURCES:
            raise BrokerStatementError(
                f"statement source {self.source!r} is not a broker read "
                f"({', '.join(STATEMENT_SOURCES)}). A book reconstructed from fills or a "
                "ledger is an expected side and cannot be filed as the broker's statement "
                "-- that substitution is how v1 reported 20 positions against a real 7."
            )
        if not math.isfinite(self.cash_usd):
            raise BrokerStatementError(f"cash_usd={self.cash_usd!r} is not a finite number")
        for ticker, shares in self.positions.items():
            if not ticker or not isinstance(shares, int) or isinstance(shares, bool):
                raise BrokerStatementError(
                    f"position {ticker!r}={shares!r} is not a whole-share count on a named "
                    "instrument"
                )
            if shares == 0:
                raise BrokerStatementError(
                    f"position {ticker!r} is zero. A closed position is absent from a "
                    "statement, never present at zero: a zero row would count toward the "
                    "position total the reconciler compares."
                )
        object.__setattr__(self, "positions", dict(sorted(self.positions.items())))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": BROKER_STATEMENT_SCHEMA_VERSION,
            "account": self.account,
            "trading_day": self.trading_day,
            "source": self.source,
            "positions": dict(self.positions),
            "cash_usd": self.cash_usd,
        }


def statement_from_document(document: Mapping[str, Any]) -> BrokerStatement:
    """Parse a stored statement, refusing anything that is not exactly one."""
    if document.get("schema_version") != BROKER_STATEMENT_SCHEMA_VERSION:
        raise BrokerStatementError(
            f"schema_version {document.get('schema_version')!r} is not "
            f"{BROKER_STATEMENT_SCHEMA_VERSION!r}"
        )
    expected = {"schema_version", "account", "trading_day", "source", "positions", "cash_usd"}
    if set(document) != expected:
        raise BrokerStatementError(
            f"statement fields {sorted(document)} are not exactly {sorted(expected)}"
        )
    return BrokerStatement(
        account=document["account"],
        trading_day=document["trading_day"],
        positions=document["positions"],
        cash_usd=float(document["cash_usd"]),
        source=document["source"],
    )


class StatementSourceProtocol(Protocol):
    """Anything that can produce today's statement. The live one is IB paper."""

    def read_statement(self, trading_day: str) -> BrokerStatement: ...


class _IbClient(Protocol):  # the subset of `ib_insync.IB` read here
    def managedAccounts(self) -> list[str]: ...  # noqa: N802 - SDK spelling
    def positions(self) -> Iterable[Any]: ...
    def accountSummary(self) -> Iterable[Any]: ...  # noqa: N802 - SDK spelling


class IbPaperStatementSource:
    """Reads positions and cash from a connected IB paper session.

    Every refusal below is a case v1 either skipped or defaulted:

    * exactly one managed account, and it is a paper account;
    * `positions()` rows for that account only, stock contracts only (an option
      or a future would be priced and counted as shares of its underlying);
    * whole shares (a fractional quantity cannot be compared exactly and is not
      rounded into agreement);
    * one position row per symbol (two rows would be summed by a dict and hide
      a duplicate lot);
    * exactly one base-currency `TotalCashValue` row for the account.
    """

    def __init__(self, client: _IbClient) -> None:
        self._client = client

    def read_statement(self, trading_day: str) -> BrokerStatement:
        accounts = list(self._client.managedAccounts())
        if len(accounts) != 1:
            raise BrokerStatementError(
                f"IB session manages {len(accounts)} accounts ({accounts}); reconciliation "
                "needs exactly one, or it cannot say whose book it compared"
            )
        account = accounts[0]
        if not account.startswith(PAPER_ACCOUNT_PREFIX):
            raise BrokerStatementError(
                f"account {account!r} is not an IB paper account ({PAPER_ACCOUNT_PREFIX}…). "
                "The paper -> live crossing is a reserved ruling (plan §9.5); this reader "
                "refuses rather than reconciling a live book by accident."
            )

        positions: dict[str, int] = {}
        for row in self._client.positions():
            if row.account != account:
                continue
            contract = row.contract
            if contract.secType != "STK":
                raise BrokerStatementError(
                    f"{contract.symbol!r} is a {contract.secType!r} contract; only stock is "
                    "reconciled as shares, and silently skipping it would drop a real position"
                )
            quantity = float(row.position)
            if not quantity.is_integer():
                raise BrokerStatementError(
                    f"{contract.symbol!r} holds a fractional {quantity!r} shares; it is not "
                    "rounded into agreement"
                )
            if contract.symbol in positions:
                raise BrokerStatementError(
                    f"{contract.symbol!r} appears twice in the IB position list"
                )
            if quantity != 0:
                positions[contract.symbol] = int(quantity)

        cash_rows = [
            item
            for item in self._client.accountSummary()
            if item.account == account and item.tag == CASH_TAG and item.currency == "USD"
        ]
        if len(cash_rows) != 1:
            raise BrokerStatementError(
                f"expected exactly one USD {CASH_TAG} row for {account}, found "
                f"{len(cash_rows)}. Cash that cannot be read is not zero."
            )
        return BrokerStatement(
            account=account,
            trading_day=trading_day,
            positions=positions,
            cash_usd=float(cash_rows[0].value),
        )
