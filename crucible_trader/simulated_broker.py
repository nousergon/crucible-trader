"""The staging trader's broker: simulated fills at the session close, no broker at all.

`alpha-engine-config-I12507` (staging plan S-19, epic I12488). In staging the
trader runs its daily session against staging predictions with a DRY-RUN broker:
the book is planned by the code prod runs and filled by simulation, and nothing
in this module can reach IB.

**Selected by the environment, never by a routing value.** Two switches, both
read by :meth:`StagingSessionConfig.from_env`:

* ``NE_ENV`` (the fleet's environment selector, `nousergon_lib.env`): only
  ``staging`` selects this module. Unset or ``prod`` is the prod path, which
  is left exactly as it was.
* :data:`SIMULATED_FILLS_VAR`: ``on`` makes a staging session fill its plan
  by simulation (:class:`SimulatedFillRouter`); unset or ``off`` leaves a
  staging session a shadow session (:class:`~crucible_trader.order_router.ShadowRouter`).
  Anything else raises, so a typo never reads as either.

Three refusals, each raised before any broker SDK is loaded:

1. **Staging never routes.** :data:`~crucible_trader.order_router.ROUTING_VAR`
   present at all in staging raises
   :class:`~crucible_trader.order_router.StagingRoutingRefusedError`
   (`order_router.assert_staging_never_routes`, also called by
   `RouterConfig.from_env`, so no entry point can build a routing config there).
   The fleet registry must agree: :func:`assert_registry_forbids_routing` asks
   `nousergon_lib.env`'s ``assert_routing_allowed`` about every route this
   package knows, and a staging registry that would allow one is refused
   (`trader.routing_allowed`, nousergon-lib#499; `alpha-engine-config-I12685`).
2. **Prod never simulates.** :data:`SIMULATED_FILLS_VAR` ``on`` outside staging
   raises :class:`SimulatedFillsRefusedError`: a simulated fill must never be
   filed as a prod served day.
3. **The caps are the prod caps.** A simulated session reads the same two cap
   variables the paper router reads, required, no defaults, and plans through
   the same :func:`~crucible_trader.order_router.plan_orders`, so the plan, the
   caps and the refusals are the code prod runs.

**The fill price.** Each planned order fills IN FULL at the session's close for
its symbol: ``close_raw`` on the session's trading day in the price panel
compiled for that day (`crucible.keys.data_panel_key`) -- the panel the book's
construction was decided on. The same close is the decision price, so the
simulated slippage is ZERO, declared (:data:`SIMULATED_SLIPPAGE_BPS`), never
measured. A name the panel does not close is
:class:`~crucible_trader.order_router.PriceUnavailableError` and the whole plan
is refused, exactly as in paper.

**What is filed** (`alpha-engine-config-I12685`, option a). Simulated fills
are first-class harness records, and none of them is an execution measurement:

* `trader/execution_shortfall/{day}.json` is ``computed`` with
  ``fill_source: simulated_close`` (:data:`FILL_SOURCE`). The harness
  (`crucible.execution.reduce_execution_window`) names such a session and
  never pools its orders, so a zero-slippage fill cannot read as a measured
  zero. A session that filled nothing is ``no_orders``, as in paper.
* The served-day evidence records the day as ``simulated``
  (`crucible.models.SessionModeLiteral`), which the phase-4 clause counts only
  under ``NE_ENV=staging`` and never as ``live``; `contract.record_session`
  refuses the mode anywhere else.
* `trader/simulated_fills/{day}.json` (:data:`SIMULATED_FILLS_SCHEMA_VERSION`)
  is kept as the debug copy: every fill with its price source.

**The hold-book control has no broker either** (:class:`StagingHoldControl`).
It reports no open orders and no accepted orders, refuses to cancel or close,
and has no statement to read: a hold-book halt in staging FAILS the run instead
of engaging the store's kill switch, because there is no broker book to
measure. A halted day is not a served day either way.

**The simulated book is carried across days** (`alpha-engine-config-I13305`).
Until 2026-11-06 it started every session all cash, the defect prod had until
crucible-trader#121: an engine that measures its turnover budget against an
empty book never binds, and staging could not see the churn it existed to
catch. A served simulated session now files its closing positions at
:func:`simulated_positions_key` (the opening positions plus every simulated
fill), and the next session opens from the newest such document filed for an
EARLIER trading day: the router plans against those shares and hands the
daily session the same :func:`~crucible_trader.order_router.held_weights`
prod derives from the broker statement. No document at all is inception (all
cash), recorded as such in the next document's ``carried_from``.

**Carried shares follow stock splits** (codex cfff90d3). The book is valued at
the session's ``close_raw``, which a split rebases; shares carried unchanged
across a 2:1 split would be valued at half and re-bought. Every split effective
on a trading day after ``carried_from``, through the session's own day,
rebases the carried shares (``floor(shares * split_to / split_from)``, the
fractional share dropped as a broker pays it in cash). The splits come from
the same Polygon scans the reconciler reads
(:mod:`crucible_trader.corporate_action_feed`, ``CRUCIBLE_TRADER_CORPORATE_ACTION_SCANS_URI``).
With that unset nothing is rebased and the document records ``splits: null``,
never an empty list; with it set, a day whose scan is missing, or a split
with no ratio on a carried name, refuses the session rather than guess.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]  # no pandas-stubs in the lock; the read is cast at its one call
from crucible.calendar import is_trading_day
from crucible.keys import data_panel_key
from crucible.store import Store

from crucible_trader.broker_session import OrderRefusedError
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.construction import TargetBook
from crucible_trader.execution_shortfall import build_no_orders_document, build_shortfall_document
from crucible_trader.kill_switch import AcceptedOrder, assert_trading_permitted, utcnow
from crucible_trader.order_book import ShareBasis
from crucible_trader.order_router import (
    MAX_ORDER_NOTIONAL_VAR,
    MAX_SESSION_NOTIONAL_VAR,
    ROUTING_ENABLED_VALUE,
    STAGING_ENVIRONMENT,
    ExecutionRecord,
    PlannedOrder,
    PriceUnavailableError,
    RouterSettingsError,
    RoutingRefusedError,
    StagingRoutingRefusedError,
    _positive_float,  # pyright: ignore[reportPrivateUsage]  # the paper router's own cap parse, not a second one
    assert_staging_never_routes,
    held_weights,
    is_staging,
    plan_orders,
    rebalance_band_usd,
)

__all__ = [
    "FILL_SOURCE",
    "SIMULATED_FILLS_SCHEMA_VERSION",
    "SIMULATED_FILLS_VAR",
    "SIMULATED_POSITIONS_SCHEMA_VERSION",
    "PanelCloseSource",
    "SimulatedBrokerError",
    "SimulatedFillRouter",
    "SimulatedFillsRefusedError",
    "StagingHoldControl",
    "StagingSessionConfig",
    "assert_registry_forbids_routing",
    "simulated_fills_key",
    "simulated_positions_key",
]

#: The staging-only switch. Unset or ``off``: a staging session is shadow.
SIMULATED_FILLS_VAR = "CRUCIBLE_TRADER_SIMULATED_FILLS"
SIMULATED_FILLS_ENABLED_VALUE = "on"
SIMULATED_FILLS_DISABLED_VALUES: frozenset[str] = frozenset({"", "off"})

#: Where every simulated fill's price came from: `execution_shortfall.v1`'s
#: ``fill_source`` value `crucible.execution.FILL_SOURCE_SIMULATED_CLOSE`
#: (restated, not imported, so this module imports on a release wheel that
#: predates the field; `tests/test_simulated_broker.py` asserts they agree).
FILL_SOURCE = "simulated_close"

#: Every order route this package can build. The registry is asked about each.
KNOWN_ROUTES: tuple[str, ...] = (ROUTING_ENABLED_VALUE,)
#: Declared, not measured: the fill is the close the decision was priced at.
SIMULATED_SLIPPAGE_BPS = 0.0
SIMULATED_FILLS_SCHEMA_VERSION = "trader_simulated_fills.v1"
SIMULATED_DISPOSITION = "simulated"
SIMULATED_STATUS = "SimulatedFill"


class SimulatedFillsRefusedError(RoutingRefusedError):
    """Simulated fills were asked for outside staging. Prod never files one."""


class SimulatedBrokerError(RoutingRefusedError):
    """A broker read the staging session has no broker to answer."""


def simulated_fills_key(trading_day: str) -> str:
    """The session's simulated fills, beside its shortfall document."""
    return f"trader/simulated_fills/{trading_day}.json"


SIMULATED_POSITIONS_SCHEMA_VERSION = "trader_simulated_positions.v1"
SIMULATED_POSITIONS_PREFIX = "trader/simulated_positions/"


def simulated_positions_key(trading_day: str) -> str:
    """The simulated book's closing positions for a served staging session."""
    return f"{SIMULATED_POSITIONS_PREFIX}{trading_day}.json"


#: ``day -> {symbol: ratio or None}`` for the splits effective that day, or
#: ``None`` when the day's splits cannot be known.
SplitSource = Callable[[str], Mapping[str, float | None] | None]


def scan_split_source(base_uri: str, *, s3_client: object = None) -> SplitSource:
    """Each day's splits from the corporate-action scans under ``base_uri``."""

    def splits_on(day: str) -> Mapping[str, float | None] | None:
        # Imported here: the feed imports the reconcile job, which the prod
        # session never needs for this module's sake.
        from crucible_trader.corporate_action_feed import (  # noqa: PLC0415
            actions_from_scan,
            ib_symbol,
            read_scan,
        )
        from crucible_trader.reconciliation_job import previous_trading_day  # noqa: PLC0415

        read = read_scan(base_uri, day, s3_client=s3_client)
        if read.document is None:
            return None
        raw = read.document.get("splits")
        rows = cast(list[dict[str, object]], raw if isinstance(raw, list) else [])
        found = actions_from_scan(  # validates the scan; a bad one raises
            read.document,
            day=day,
            universe=frozenset(ib_symbol(str(r.get("ticker"))) for r in rows),
            holdings_on=lambda _day: None,  # dividends are not this reader's question
            previous_trading_day=previous_trading_day,
            uri=read.uri,
        )
        return {a.ticker: a.ratio for a in found.actions if a.kind == "split"}

    return splits_on


def split_source_from_env(env: Mapping[str, str] | None = None) -> SplitSource | None:
    """The scans the box's environment names, or ``None`` when it names none."""
    value = (env if env is not None else os.environ).get(SPLIT_SCANS_VAR, "").strip()
    return scan_split_source(value) if value else None


#: The reconciler's scans location (`crucible_trader.corporate_action_feed.SCANS_URI_ENV`),
#: restated so the prod path never imports the feed for this module's sake.
SPLIT_SCANS_VAR = "CRUCIBLE_TRADER_CORPORATE_ACTION_SCANS_URI"


def apply_splits(
    positions: Mapping[str, int],
    *,
    after: str,
    through: str,
    splits: SplitSource,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """``positions`` rebased by every split effective after ``after`` through
    ``through`` (trading days), and what was applied. Raises
    :class:`SimulatedBrokerError` for a day whose splits are unknown or a
    carried name's split with no ratio."""
    book = dict(positions)
    applied: list[dict[str, object]] = []
    day = dt.date.fromisoformat(after)
    end = dt.date.fromisoformat(through)
    while True:
        day += dt.timedelta(days=1)
        if day > end:
            break
        if not is_trading_day(day):
            continue
        found = splits(day.isoformat())
        if found is None:
            raise SimulatedBrokerError(
                f"the corporate-action scan for {day} is missing; a carried book across an "
                "unknown split would be valued at the wrong close"
            )
        for symbol in sorted(set(found) & set(book)):
            ratio = found[symbol]
            if ratio is None or not (math.isfinite(ratio) and ratio > 0):
                raise SimulatedBrokerError(
                    f"{symbol} split on {day} with no usable ratio ({ratio!r}); the carried "
                    "shares cannot be rebased"
                )
            before = book[symbol]
            book[symbol] = math.floor(before * ratio + 1e-9)
            applied.append(
                {
                    "trading_day": day.isoformat(),
                    "symbol": symbol,
                    "ratio": ratio,
                    "shares_before": before,
                    "shares_after": book[symbol],
                }
            )
    return {s: n for s, n in book.items() if n}, applied


def opening_positions(store: Store, trading_day: str) -> tuple[dict[str, int], str | None]:
    """``(positions, carried_from)``: the newest closing positions filed before ``trading_day``.

    ``({}, None)`` when none was ever filed: the inception book, all cash. A
    document for ``trading_day`` itself or later is never read, so a re-run
    of a session opens from the same book its first run did.
    """
    days = sorted(
        key[len(SIMULATED_POSITIONS_PREFIX) : -len(".json")]
        for key in store.list_keys(SIMULATED_POSITIONS_PREFIX)
        if key.endswith(".json")
    )
    earlier = [d for d in days if d < trading_day]
    if not earlier:
        return {}, None
    day = earlier[-1]
    document = cast(dict[str, object], json.loads(store.get_bytes(simulated_positions_key(day))))
    if document.get("schema_version") != SIMULATED_POSITIONS_SCHEMA_VERSION:
        raise SimulatedBrokerError(
            f"{simulated_positions_key(day)} is {document.get('schema_version')!r}, not "
            f"{SIMULATED_POSITIONS_SCHEMA_VERSION!r}; the book it carries cannot be read"
        )
    raw = cast(dict[str, object], document.get("positions") or {})
    positions: dict[str, int] = {}
    for symbol, shares in raw.items():
        if not isinstance(shares, int) or isinstance(shares, bool) or shares < 0:
            raise SimulatedBrokerError(
                f"{simulated_positions_key(day)} holds {shares!r} shares of {symbol}; a "
                "simulated book is long only, in whole shares"
            )
        if shares:
            positions[str(symbol)] = shares
    return positions, day


class RoutingRegistry(Protocol):
    """What this module reads of a `nousergon_lib.env.Environment`."""

    def assert_routing_allowed(self, route: str) -> None: ...


def assert_registry_forbids_routing(
    environment: Callable[[str], RoutingRegistry] | None = None,
) -> None:
    """Refuse a staging run unless the fleet registry denies staging every route.

    The trader's own ``NE_ENV=staging`` guard already refuses the routing
    switch. This is the second authority (`alpha-engine-config-I12685`, the
    S-19 design): `nousergon_lib.env`'s ``environments.staging.trader.
    routing_allowed`` is ``[]``, and ``assert_routing_allowed`` must raise for
    every route in :data:`KNOWN_ROUTES`. A registry that would allow one is a
    registry staging's trader disagrees with, and the run stops before anything
    opens. A registry that cannot be read stops it too.

    Called on the staging path only, and imported lazily: the prod path neither
    reads the registry nor needs a `nousergon-lib` that carries the field.
    ``environment`` resolves a registry environment by name (default
    `nousergon_lib.env.resolve`).
    """
    if environment is None:
        from nousergon_lib import env as fleet_env  # noqa: PLC0415  # staging only

        environment = fleet_env.resolve
    resolved = environment(STAGING_ENVIRONMENT)
    allowed = [route for route in KNOWN_ROUTES if _registry_allows(resolved, route)]
    if allowed:
        raise StagingRoutingRefusedError(
            f"the fleet registry allows staging to route {allowed}. Staging never routes "
            "(S-19): its trader simulates fills and the IB paper account is not staging's, "
            "so a registry that says otherwise is refused rather than obeyed."
        )


def _registry_allows(resolved: RoutingRegistry, route: str) -> bool:
    """``True`` unless the registry's ``assert_routing_allowed`` refuses ``route``."""
    from nousergon_lib.env import RoutingNotAllowedError  # noqa: PLC0415  # staging only

    try:
        resolved.assert_routing_allowed(route)
    except RoutingNotAllowedError:
        return False
    return True


@dataclasses.dataclass(frozen=True)
class StagingSessionConfig:
    """How a STAGING session runs. Only :meth:`from_env` decides whether one does."""

    simulated: bool
    max_order_notional_usd: float | None = None
    max_session_notional_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.simulated:
            return
        for name in ("max_order_notional_usd", "max_session_notional_usd"):
            value = getattr(self, name)
            if value is None or not (math.isfinite(value) and value > 0):
                raise RouterSettingsError(f"simulated fills are on with {name}={value!r}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> StagingSessionConfig | None:
        """``None`` outside staging (the prod path, untouched); raises on every refusal."""
        source = os.environ if env is None else env
        flag = (source.get(SIMULATED_FILLS_VAR) or "").strip()
        if flag not in SIMULATED_FILLS_DISABLED_VALUES and flag != SIMULATED_FILLS_ENABLED_VALUE:
            raise RouterSettingsError(
                f"{SIMULATED_FILLS_VAR}={flag!r} is not a value. Unset or 'off' is no "
                f"simulation; {SIMULATED_FILLS_ENABLED_VALUE!r} simulates fills in staging. "
                "Nothing else is accepted, so a typo can neither switch it on nor hide as off."
            )
        if not is_staging(source):
            if flag == SIMULATED_FILLS_ENABLED_VALUE:
                raise SimulatedFillsRefusedError(
                    f"{SIMULATED_FILLS_VAR}=on outside staging. A simulated fill must never be "
                    "filed as a prod served day; simulated fills run only under NE_ENV=staging."
                )
            return None
        assert_staging_never_routes(source)
        assert_registry_forbids_routing()
        if flag != SIMULATED_FILLS_ENABLED_VALUE:
            return cls(simulated=False)
        return cls(
            simulated=True,
            max_order_notional_usd=_positive_float(source, MAX_ORDER_NOTIONAL_VAR),
            max_session_notional_usd=_positive_float(source, MAX_SESSION_NOTIONAL_VAR),
        )


def _panel_closes(payload: bytes, trading_day: str) -> list[tuple[str, float]]:
    """``(ticker, close_raw)`` for every ``trading_day`` row of a panel parquet."""
    frame = pd.read_parquet(io.BytesIO(payload), columns=["trading_day", "ticker", "close_raw"])  # pyright: ignore[reportUnknownMemberType]  # pandas ships no complete stubs here
    records = cast(list[dict[str, object]], frame.to_dict(orient="records"))  # pyright: ignore[reportUnknownMemberType]  # pandas ships no complete stubs here
    return [
        (str(row["ticker"]), float(cast(float, row["close_raw"])))
        for row in records
        if str(row["trading_day"]) == trading_day
    ]


class PanelCloseSource:
    """Each symbol's close on ``trading_day``, from the panel compiled for that day."""

    def __init__(self, store: Store, trading_day: str) -> None:
        self._store = store
        self._day = trading_day
        self._closes: dict[str, float] | None = None

    @property
    def source_key(self) -> str:
        return data_panel_key(self._day)

    def _load(self) -> dict[str, float]:
        key = self.source_key
        if not self._store.exists(key):
            raise PriceUnavailableError(
                f"the price panel is absent at {key}; a simulated fill has no close to fill at"
            )
        rows = _panel_closes(self._store.get_bytes(key), self._day)
        tickers = [ticker for ticker, _ in rows]
        if len(set(tickers)) != len(tickers):
            raise PriceUnavailableError(
                f"the panel at {key} carries more than one {self._day} row for a ticker; "
                "a fill price chosen between two closes would be a guess"
            )
        return dict(rows)

    def __call__(self, symbol: str) -> float | None:
        if self._closes is None:
            self._closes = self._load()
        return self._closes.get(symbol)


class StagingHoldControl:
    """The hold-book safeguard's :class:`~crucible_trader.kill_switch.BrokerControl`
    in staging, where there is no broker. It changes nothing and reads nothing."""

    def statement(self, trading_day: str) -> BrokerStatement:
        raise SimulatedBrokerError(
            f"staging has no broker statement for {trading_day}; a hold-book halt in staging "
            "fails the run rather than engage a kill switch over a book no broker holds"
        )

    def open_order_count(self) -> int:
        return 0

    def cancel_all(self) -> None:
        raise OrderRefusedError("the staging hold control has no broker and cancels nothing")

    def submit_close(self, symbol: str, shares: int) -> None:
        raise OrderRefusedError(
            f"the staging hold control has no broker and closes nothing ({symbol})"
        )

    def orders_accepted_since(self, instant: dt.datetime) -> list[AcceptedOrder]:
        return []

    def wait(self, seconds: float) -> None:
        return None


class SimulatedFillRouter:
    """:class:`~crucible_trader.session.OrderRouter` that fills by simulation.

    Imports no broker SDK and holds no connection; ``enabled`` is ``False``
    because nothing it does reaches a broker.
    """

    enabled = False
    simulated = True

    def __init__(
        self,
        config: StagingSessionConfig,
        *,
        store: Store,
        trading_day: str,
        clock: Callable[[], dt.datetime] = utcnow,
        splits: SplitSource | None = None,
    ) -> None:
        if not config.simulated:
            raise SimulatedFillsRefusedError(f"simulated fills are off ({SIMULATED_FILLS_VAR})")
        self._config = config
        self._store = store
        self._day = trading_day
        self._clock = clock
        self._closes = PanelCloseSource(store, trading_day)
        self._planned: tuple[PlannedOrder, ...] = ()
        #: The names the plan priced, for the order book's share columns.
        self._sized_names: frozenset[str] | None = None
        self._band_usd = 0.0
        self._records: list[ExecutionRecord] = []
        self._opening, self._carried_from = opening_positions(store, trading_day)
        self._splits: list[dict[str, object]] | None = None
        source = split_source_from_env() if splits is None else splits
        if source is not None:
            self._splits = []
            if self._carried_from is not None and self._opening:
                self._opening, self._splits = apply_splits(
                    self._opening, after=self._carried_from, through=trading_day, splits=source
                )

    @property
    def opening(self) -> dict[str, int]:
        """The simulated book held into the session, in shares."""
        return dict(self._opening)

    def held_weights(self, portfolio_notional_usd: float) -> dict[str, float]:
        """The carried simulated book as weights, valued at the session's closes.

        The same :func:`~crucible_trader.order_router.held_weights` the paper
        router applies to the broker statement, so staging constructs against
        the book semantics prod does.
        """
        return held_weights(
            self._opening, quote=self._closes, portfolio_notional_usd=portfolio_notional_usd
        )

    def closing_positions(self) -> dict[str, int]:
        """The opening book plus every simulated fill. Names sold to zero drop out."""
        book = dict(self._opening)
        for record in self._records:
            decision = record.order.decision
            filled = round(record.filled_quantity)
            signed = filled if decision.side == "buy" else -filled
            book[decision.symbol] = book.get(decision.symbol, 0) + signed
        short = sorted(s for s, n in book.items() if n < 0)
        if short:
            raise SimulatedBrokerError(
                f"the simulated fills leave {short} short; the plan sold more than was held"
            )
        return {s: n for s, n in sorted(book.items()) if n}

    def simulated_positions_payload(self, *, trading_day: str, now: dt.datetime) -> bytes:
        """The closing simulated book, which the next session opens from."""
        document = {
            "schema_version": SIMULATED_POSITIONS_SCHEMA_VERSION,
            "environment": "staging",
            "trading_day": trading_day,
            "generated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "carried_from": self._carried_from,
            "splits": self._splits,
            "opening": dict(sorted(self._opening.items())),
            "positions": self.closing_positions(),
        }
        return json.dumps(document, indent=2, sort_keys=True).encode("utf-8")

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._records)

    def batches(self, book: TargetBook) -> list[tuple[PlannedOrder, ...]]:
        if book.trading_day != self._day:
            raise RoutingRefusedError(
                f"the book is for {book.trading_day}; this router was opened for {self._day}"
            )
        order_cap = self._config.max_order_notional_usd
        session_cap = self._config.max_session_notional_usd
        assert order_cap is not None and session_cap is not None  # __post_init__ refuses None
        sells, buys = plan_orders(
            book,
            positions=self._opening,
            quote=self._closes,
            clock=self._clock,
            max_order_notional_usd=order_cap,
            max_session_notional_usd=session_cap,
        )
        self._planned = sells + buys
        self._band_usd = rebalance_band_usd(book)
        self._sized_names = frozenset(t for t, w in book.weights.items() if w > 0) | set(
            self._opening
        )
        return [batch for batch in (sells, buys) if batch]

    def share_basis(self) -> ShareBasis | None:
        """The opening book and the panel closes the plan sized at (`order_book.ShareBasis`).

        None before :meth:`batches`. The closes are the same cached panel the
        plan read, so pricing here again reads nothing new.
        """
        if self._sized_names is None:
            return None
        prices = {s: p for s in sorted(self._sized_names) if (p := self._closes(s)) is not None}
        return ShareBasis(positions=dict(self._opening), prices=prices, band_usd=self._band_usd)

    def send(self, batch: Sequence[PlannedOrder]) -> None:
        for order in batch:
            # The kill switch, re-read before every simulated fill, as in paper.
            assert_trading_permitted(self._store, self._day)
            self._records.append(
                ExecutionRecord(
                    order=order,
                    disposition=SIMULATED_DISPOSITION,
                    broker_status=SIMULATED_STATUS,
                    submitted_at=self._clock(),
                    filled_quantity=float(order.decision.quantity),
                    fill_price=float(order.decision.decision_price),
                )
            )

    def shortfall_document(
        self, *, champion: str, trading_day: str, now: dt.datetime
    ) -> dict[str, object]:
        """``computed`` with ``fill_source: simulated_close``; ``no_orders`` when
        nothing was filled. Never a broker document: the harness names a
        simulated session and never pools its orders."""
        if not self._records:
            return build_no_orders_document(
                trading_day=trading_day,
                champion=champion,
                reason=(
                    f"simulated session: none of {len(self._planned)} planned order(s) "
                    f"filled; no order reached a broker (NE_ENV=staging, "
                    f"{SIMULATED_FILLS_VAR}=on)"
                ),
                now=now,
            )
        return build_shortfall_document(
            trading_day=trading_day,
            champion=champion,
            fills=[record.fill for record in self._records],
            now=now,
            fill_source=FILL_SOURCE,
        )

    def simulated_fills_document(
        self, *, champion: str, trading_day: str, now: dt.datetime
    ) -> dict[str, object]:
        """Every simulated fill, labelled as one."""
        return {
            "schema_version": SIMULATED_FILLS_SCHEMA_VERSION,
            "environment": "staging",
            "trading_day": trading_day,
            "champion": champion,
            "generated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fill_source": FILL_SOURCE,
            "price_source": f"{self._closes.source_key}#close_raw",
            "slippage_bps": SIMULATED_SLIPPAGE_BPS,
            "n_planned": len(self._planned),
            "n_filled": len(self._records),
            "orders": [{**r.to_document(), "fill_source": FILL_SOURCE} for r in self._records],
        }

    def simulated_fills_payload(
        self, *, champion: str, trading_day: str, now: dt.datetime
    ) -> bytes:
        document = self.simulated_fills_document(
            champion=champion, trading_day=trading_day, now=now
        )
        return json.dumps(document, indent=2, sort_keys=True).encode("utf-8")

    def close(self) -> None:
        return None
