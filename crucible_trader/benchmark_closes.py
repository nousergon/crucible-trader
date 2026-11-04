"""Benchmark closes for the performance statement, read beside the price panel.

`alpha-engine-config-I13201` (back office 1f). The statement benchmarks the fund
against SPY and QQQ (:data:`~crucible_trader.performance_statement.BENCHMARKS`)
at their settled closes. The v2 price panel carries only the symbols a slot or
the attribution spec declares, and QQQ is neither, so every statement's QQQ leg
was ``null``. The panel is the TRADING input: a symbol added to it is a symbol
the feature layer builds over, and a compile that cannot find it fails the
day. So the statement reads the benchmarks it lacks from a separate path that
nothing on the trading side reads.

**Source.** nousergon-data's daily closes,
``{CRUCIBLE_TRADER_BENCHMARK_CLOSES_URI}/{day}.parquet`` (one row per ticker,
``Close`` and a per-row ``source``). Brian's ground truth for prices is Polygon
(2026-10-22), so only a row whose ``source`` is ``polygon`` is taken; the
evening's first pass is another vendor's and is overwritten by Polygon the
next morning. The object lives seven days, while a statement needs its
month's first and last close, so the trader CAPTURES the closes:
:func:`capture_benchmark_closes` runs in every `trader.reconcile` and files each
recent trading day's Polygon benchmark closes once, create-only, at
:func:`benchmark_closes_key`. A day whose row is not Polygon's yet is tried
again on the next run.

**Reading.** :func:`with_benchmark_closes` wraps a close source (the panel) so
that a benchmark the panel has no close for is read from the captured
document. Only :data:`BENCHMARK_SYMBOLS` are ever filled, and only where the
panel has nothing: a position is still valued from the panel alone.

**Failure isolation.** Neither function raises. A capture that cannot read or
file reports through one MetricRecord (:data:`CAPTURE_METRIC`); a missing
capture is a missing close, which the statement states as ``null`` with a
reason and its metric reads WATCH, exactly as before.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from crucible.calendar import is_trading_day
from crucible.store import ETAG_ABSENT, PointerConflictError, Store

#: Kept here, not imported, so this module has no cycle with the statement;
#: ``tests`` assert it equals ``performance_statement.BENCHMARKS``.
BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ")
BENCHMARK_CLOSES_URI_ENV = "CRUCIBLE_TRADER_BENCHMARK_CLOSES_URI"
BENCHMARK_CLOSES_PREFIX = "trader/benchmark_closes/"
BENCHMARK_CLOSES_SCHEMA_VERSION = "benchmark_closes.v1"
CAPTURE_METRIC = "benchmark_closes_capture"
#: The vendor whose close is ground truth; any other row is not taken.
GROUND_TRUTH_SOURCE = "polygon"
#: Trading days back from the run day a capture tries (the object lives 7 days).
CAPTURE_LOOKBACK_TRADING_DAYS = 4

CloseSource = Callable[[str, Sequence[str]], Mapping[str, float]]


def benchmark_closes_key(day: str) -> str:
    dt.date.fromisoformat(day)
    return f"{BENCHMARK_CLOSES_PREFIX}{day}.json"


def benchmark_closes_uri_from_env(env: Mapping[str, str] | None = None) -> str | None:
    value = (env if env is not None else os.environ).get(BENCHMARK_CLOSES_URI_ENV, "").strip()
    return value or None


def _recent_trading_days(day: str, n: int) -> list[str]:
    current, out = dt.date.fromisoformat(day), list[str]()
    while len(out) < n:
        current -= dt.timedelta(days=1)
        if is_trading_day(current):
            out.append(current.isoformat())
    return sorted(out)


@dataclasses.dataclass(frozen=True)
class _Read:
    closes: dict[str, float]
    reason: str | None


def _read_day(base_uri: str, day: str, s3_client: Any) -> _Read:  # pyright: ignore[reportExplicitAny]
    """The day's Polygon closes for :data:`BENCHMARK_SYMBOLS`, or a reason."""
    import pandas as pd  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]  # no pandas-stubs in the lock

    uri = f"{base_uri.rstrip('/')}/{day}.parquet"
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    try:
        payload = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        frame = pd.read_parquet(io.BytesIO(payload))  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:  # noqa: BLE001 - absent, denied or unreadable: try again later
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
        return _Read({}, f"{day}: {uri} not read ({code})")
    closes: dict[str, float] = {}
    for symbol in BENCHMARK_SYMBOLS:
        if symbol not in frame.index:
            return _Read({}, f"{day}: no {symbol} row")
        row = cast("Mapping[str, object]", frame.loc[symbol])
        if str(row.get("source")) != GROUND_TRUTH_SOURCE:
            return _Read({}, f"{day}: {symbol} is not a {GROUND_TRUTH_SOURCE} close yet")
        close = float(pd.to_numeric(row.get("Close"), errors="coerce"))  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
        if not math.isfinite(close) or close <= 0:
            return _Read({}, f"{day}: {symbol} close {row.get('Close')!r} is not a price")
        closes[symbol] = close
    return _Read(closes, None)


def capture_benchmark_closes(
    ctx: Any,  # pyright: ignore[reportExplicitAny]  # crucible.runner.RunContext
    *,
    day: str,
    now: str,
    base_uri: str | None = None,
    s3_client: Any = None,  # pyright: ignore[reportExplicitAny]
) -> list[str]:
    """File the Polygon benchmark closes of each recent trading day not yet
    filed. Returns the days filed. NEVER raises; unset URI = nothing to do."""
    base_uri = base_uri if base_uri is not None else benchmark_closes_uri_from_env()
    if base_uri is None:
        return []
    filed: list[str] = []
    pending: list[str] = []
    try:
        if not base_uri.startswith("s3://"):
            raise ValueError(f"{BENCHMARK_CLOSES_URI_ENV} must be an s3:// URI, got {base_uri!r}")
        if s3_client is None:
            s3_client = _default_s3_client()
        store: Store = ctx.store
        for d in _recent_trading_days(day, CAPTURE_LOOKBACK_TRADING_DAYS):
            key = benchmark_closes_key(d)
            if store.exists(key):
                continue
            read = _read_day(base_uri, d, s3_client)
            if read.reason is not None:
                pending.append(read.reason)
                continue
            document = {
                "schema_version": BENCHMARK_CLOSES_SCHEMA_VERSION,
                "trading_day": d,
                "closes": read.closes,
                "source": GROUND_TRUTH_SOURCE,
                "read_from": f"{base_uri.rstrip('/')}/{d}.parquet",
                "captured_at": now,
            }
            payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
            try:
                ctx.record_output_cas(key, ETAG_ABSENT, payload, BENCHMARK_CLOSES_SCHEMA_VERSION)
            except PointerConflictError:
                continue  # a concurrent run filed it first; a capture is never rewritten
            filed.append(d)
    except Exception as exc:  # noqa: BLE001 - must never change what the job decides or pages
        _metric(ctx, now, "FAIL", f"benchmark closes NOT captured: {type(exc).__name__}: {exc}")
        return filed
    _metric(
        ctx,
        now,
        "WATCH" if pending else "OK",
        f"filed {filed or 'nothing new'}" + (f"; not yet: {'; '.join(pending)}" if pending else ""),
    )
    return filed


def _default_s3_client() -> Any:  # pyright: ignore[reportExplicitAny]
    """The box's own credential chain (the trader's role)."""
    import boto3  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]  # only the live path

    return boto3.client("s3")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def with_benchmark_closes(store: Store, closes: CloseSource) -> CloseSource:
    """``closes``, with a benchmark it has no close for read from the capture."""

    def source(day: str, tickers: Sequence[str]) -> Mapping[str, float]:
        found = dict(closes(day, tickers))
        missing = [t for t in tickers if t in BENCHMARK_SYMBOLS and t not in found]
        if not missing:
            return found
        try:
            document = json.loads(store.get_bytes(benchmark_closes_key(day)).decode("utf-8"))
            captured = cast("Mapping[str, float]", document["closes"])
        except Exception:  # noqa: BLE001 - no capture is a missing close, stated as such
            return found
        for t in missing:
            if t in captured:
                found[t] = float(captured[t])
        return found

    return source


def _metric(ctx: Any, now: str, status: str, reason: str) -> None:  # pyright: ignore[reportExplicitAny]
    try:
        ctx.record_metric(
            {
                "name": CAPTURE_METRIC,
                "module": "crucible_trader.benchmark_closes",
                "metric_type": "operational",
                "value": None,
                "unit": None,
                "n_floor": 0,
                "status": status,
                "status_reason": reason[:400],
                "source_path": BENCHMARK_CLOSES_PREFIX,
                "last_updated_utc": now,
            }
        )
    except Exception:  # noqa: BLE001, S110 - a metric must not raise into the trader either
        pass


__all__ = [
    "BENCHMARK_CLOSES_PREFIX",
    "BENCHMARK_CLOSES_URI_ENV",
    "BENCHMARK_SYMBOLS",
    "CAPTURE_METRIC",
    "benchmark_closes_key",
    "benchmark_closes_uri_from_env",
    "capture_benchmark_closes",
    "with_benchmark_closes",
]
