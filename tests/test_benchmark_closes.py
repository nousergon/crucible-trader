"""Benchmark closes beside the panel (`alpha-engine-config-I13201`): captured once
per day from the Polygon close, read only for a benchmark the panel lacks, and
never raising into the trader. All prices are synthetic."""

from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import sys
import types

import pandas as pd
import pytest
from crucible.runner import RunContext
from crucible.store import LocalStore, PointerConflictError

from crucible_trader import benchmark_closes as bc
from crucible_trader.benchmark_closes import (
    BENCHMARK_CLOSES_URI_ENV,
    BENCHMARK_SYMBOLS,
    CAPTURE_METRIC,
    benchmark_closes_key,
    benchmark_closes_uri_from_env,
    capture_benchmark_closes,
    with_benchmark_closes,
)
from crucible_trader.exception_record import PAPER_FUND_TENANT
from crucible_trader.performance_statement import BENCHMARKS, file_statement

URI = "s3://example-bucket/staging/daily_closes"
NOW = "2026-12-01T21:00:00Z"
RUN_DAY = "2026-12-01"  # the four trading days before it: 11-24, 11-25, 11-27, 11-30


def parquet(rows: dict[str, tuple[object, str]]) -> bytes:
    frame = pd.DataFrame(
        [{"ticker": t, "Close": c, "source": s} for t, (c, s) in rows.items()]
    ).set_index("ticker")
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


class Denied(Exception):
    response = {"Error": {"Code": "AccessDenied"}}  # noqa: RUF012


class FakeS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.gets: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
        assert Bucket == "example-bucket"
        self.gets.append(Key)
        if Key not in self.objects:
            raise Denied
        return {"Body": io.BytesIO(self.objects[Key])}


def key(day: str) -> str:
    return f"staging/daily_closes/{day}.parquet"


POLYGON = {"SPY": (500.0, "polygon"), "QQQ": (400.0, "polygon"), "AAA": (1.0, "polygon")}


def ctx_for(store: LocalStore) -> RunContext:
    return RunContext(
        run_id="01JG0000000000000000000000",
        job="trader.reconcile",
        trading_day=dt.date.fromisoformat(RUN_DAY),
        calendar_date=dt.date.fromisoformat(RUN_DAY),
        store=store,
        seed=0,
        started=dt.datetime(2026, 12, 1, 21, 0, tzinfo=dt.UTC),
    )


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def test_the_symbols_are_the_statements_benchmarks():
    assert BENCHMARK_SYMBOLS == BENCHMARKS


def test_the_uri_comes_from_the_environment():
    assert benchmark_closes_uri_from_env({BENCHMARK_CLOSES_URI_ENV: f" {URI} "}) == URI
    assert benchmark_closes_uri_from_env({}) is None


class TestCapture:
    def test_files_each_recent_polygon_day_once_and_waits_for_the_rest(self, store):
        s3 = FakeS3(
            {
                key("2026-11-24"): parquet(POLYGON),
                key("2026-11-25"): parquet({**POLYGON, "QQQ": (401.0, "yfinance")}),
                key("2026-11-27"): parquet({"SPY": (502.0, "polygon")}),
                key("2026-11-30"): parquet({**POLYGON, "SPY": (float("nan"), "polygon")}),
            }
        )
        ctx = ctx_for(store)
        assert capture_benchmark_closes(ctx, day=RUN_DAY, now=NOW, base_uri=URI, s3_client=s3) == [
            "2026-11-24"
        ]
        document = json.loads(store.get_bytes(benchmark_closes_key("2026-11-24")))
        assert document["closes"] == {"SPY": 500.0, "QQQ": 400.0}
        assert document["source"] == "polygon"
        (metric,) = ctx.metrics
        assert metric["name"] == CAPTURE_METRIC and metric["status"] == "WATCH"
        reason = metric["status_reason"]
        assert "QQQ is not a polygon close yet" in reason
        assert "no QQQ row" in reason
        assert "is not a price" in reason
        # A filed day is never read again; the next run takes what has landed.
        s3.objects[key("2026-11-25")] = parquet(POLYGON)
        s3.objects[key("2026-11-27")] = parquet(
            {t: (str(c), src) for t, (c, src) in {**POLYGON, "QQQ": ("n/a", "polygon")}.items()}
        )
        s3.gets.clear()
        again = ctx_for(store)
        assert capture_benchmark_closes(
            again, day=RUN_DAY, now=NOW, base_uri=URI, s3_client=s3
        ) == ["2026-11-25"]
        assert key("2026-11-24") not in s3.gets

    def test_an_unreadable_day_is_tried_again_and_a_full_capture_reads_ok(self, store):
        s3 = FakeS3({key(d): parquet(POLYGON) for d in ("2026-11-24", "2026-11-25", "2026-11-27")})
        ctx = ctx_for(store)
        capture_benchmark_closes(ctx, day=RUN_DAY, now=NOW, base_uri=URI, s3_client=s3)
        assert "2026-11-30.parquet not read (AccessDenied)" in ctx.metrics[0]["status_reason"]
        s3.objects[key("2026-11-30")] = parquet(POLYGON)
        done = ctx_for(store)
        assert capture_benchmark_closes(done, day=RUN_DAY, now=NOW, base_uri=URI, s3_client=s3) == [
            "2026-11-30"
        ]
        assert done.metrics[0]["status"] == "OK"

    def test_no_uri_is_nothing_to_do(self, store, monkeypatch):
        monkeypatch.delenv(BENCHMARK_CLOSES_URI_ENV, raising=False)
        ctx = ctx_for(store)
        assert capture_benchmark_closes(ctx, day=RUN_DAY, now=NOW) == []
        assert ctx.metrics == []

    def test_a_bad_uri_is_a_fail_metric_never_a_raise(self, store):
        ctx = ctx_for(store)
        assert capture_benchmark_closes(ctx, day=RUN_DAY, now=NOW, base_uri="/local/path") == []
        assert ctx.metrics[0]["status"] == "FAIL"

    def test_the_live_path_builds_its_own_client(self, store, monkeypatch):
        s3 = FakeS3({})
        monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda _name: s3))
        monkeypatch.setenv(BENCHMARK_CLOSES_URI_ENV, URI)
        capture_benchmark_closes(ctx_for(store), day=RUN_DAY, now=NOW)
        assert len(s3.gets) == 4

    def test_a_concurrent_capture_is_not_rewritten_and_a_metric_failure_is_swallowed(self, store):
        class Racing:
            def __init__(self) -> None:
                self.store = store

            def record_output_cas(self, *_a, **_k):
                raise PointerConflictError("raced")

            def record_metric(self, _m):
                raise RuntimeError("metrics down")

        s3 = FakeS3({key("2026-11-30"): parquet(POLYGON)})
        assert (
            capture_benchmark_closes(Racing(), day=RUN_DAY, now=NOW, base_uri=URI, s3_client=s3)
            == []
        )


class TestReading:
    def test_a_benchmark_the_panel_lacks_is_read_from_the_capture(self, store):
        store.put_bytes(
            benchmark_closes_key("2026-11-30"),
            json.dumps({"closes": {"SPY": 9.0, "QQQ": 400.0}}).encode(),
        )

        def panel(_day, tickers):
            return {t: 1.0 for t in tickers if t in ("SPY", "AAA")}

        source = with_benchmark_closes(store, panel)
        # The panel's own close wins; only QQQ is filled; a held name never is.
        assert source("2026-11-30", ["SPY", "QQQ", "AAA", "BBB"]) == {
            "SPY": 1.0,
            "QQQ": 400.0,
            "AAA": 1.0,
        }
        assert source("2026-11-27", ["QQQ"]) == {}  # no capture: a missing close
        assert source("2026-11-30", ["AAA"]) == {"AAA": 1.0}

    def test_a_capture_missing_the_symbol_leaves_it_missing(self, store):
        store.put_bytes(benchmark_closes_key("2026-11-30"), json.dumps({"closes": {}}).encode())
        assert with_benchmark_closes(store, lambda _d, _t: {})("2026-11-30", ["QQQ"]) == {}

    def test_the_statement_benchmarks_qqq_from_the_capture(self, store):
        for day, close in (("2026-11-25", 400.0), ("2026-11-30", 410.0)):
            store.put_bytes(
                benchmark_closes_key(day), json.dumps({"closes": {"QQQ": close}}).encode()
            )
        ctx = ctx_for(store)
        close_doc = {
            "status": "closed",
            "tenant": PAPER_FUND_TENANT,
            "account": "DU1",
            "month": "2026-11",
            "opened_in_month": True,
            "external_flows_cents": 0,
            "nav_start_cents": 1_000_000,
            "nav_end_cents": 1_100_000,
            "start_trading_day": "2026-11-25",
            "last_trading_day": "2026-11-30",
            "pnl": {},
        }
        statement = file_statement(
            ctx, close_doc, now=NOW, closes=lambda _d, t: {s: 100.0 for s in t if s == "SPY"}
        )
        assert statement is not None
        assert statement["benchmarks"]["QQQ"]["return"] == pytest.approx(0.025)
        assert ctx.metrics[-1]["status"] == "OK"


def test_keys():
    assert benchmark_closes_key("2026-11-30") == "trader/ledger/benchmarks/2026-11-30.json"
    with pytest.raises(ValueError):
        benchmark_closes_key("nope")
    assert bc.CAPTURE_LOOKBACK_TRADING_DAYS == 4
