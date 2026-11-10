"""The day's order book (Brian 2026-11-05): every name considered, kept."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest
from crucible.slots.strategy import BookUniverse
from crucible.store import LocalStore

from crucible_trader.books_retention import is_books_key
from crucible_trader.construction import TargetBook
from crucible_trader.order_book import (
    ORDER_BOOK_METRIC,
    STAGE_AS_TRADED,
    STAGE_PRE_OPEN,
    STAGE_SESSION,
    as_traded_key,
    changed_targets,
    order_book_document,
    order_book_key,
    record_order_book,
    write_pre_open_order_book,
)

NOW = dt.datetime(2026, 11, 5, 15, 0, tzinfo=dt.UTC)


def _book() -> TargetBook:
    tickers = ("AAA", "BBB", "CCC", "DDD", "SPY", "$CASH")
    universe = BookUniverse(
        tickers=tickers, sectors=("t", "t", "f", "f", "idx", "cash"), benchmark_idx=4, cash_idx=5
    )
    return TargetBook(
        trading_day="2026-11-05",
        s_champion="s:arm:x",
        m_champion="m:arm:y",
        u_champion=None,
        universe=universe,
        weights={"AAA": 0.6, "BBB": 0.0, "CCC": 0.3, "DDD": 0.0, "SPY": 0.0, "$CASH": 0.1},
        turnover_one_way_ratio=0.5,
        cost_bps=3.0,
        portfolio_notional_usd=100_000.0,
        evidence={},
        metric={},
        session_inputs_key="k",
        alpha={"AAA": 0.02, "BBB": 0.01, "CCC": 0.015, "DDD": -0.01, "SPY": 0.0, "$CASH": 0.0},
        eligible={"AAA": True, "BBB": True, "CCC": True, "DDD": False, "SPY": False, "$CASH": True},
    )


class _Ctx:
    def __init__(self, store) -> None:
        self.store = store
        self.metrics: list[dict] = []
        self.outputs: list[dict] = []

    def record_output(self, key, payload, schema_version="v1") -> None:
        self.store.put_bytes(key, payload)
        self.outputs.append({"key": key, "schema_version": schema_version})

    def record_output_cas(self, key, expected, payload, schema_version="v1") -> str:
        version = self.store.compare_and_swap(key, expected, payload)
        self.outputs.append({"key": key, "schema_version": schema_version})
        return version

    def record_metric(self, metric) -> None:
        self.metrics.append(metric)


def test_every_name_is_a_row_with_what_the_book_does_with_it() -> None:
    doc = order_book_document(
        _book(), previous_weights={"BBB": 0.2, "AAA": 0.6}, written_at_utc="t"
    )
    actions = {r["ticker"]: r["action"] for r in doc["rows"]}
    assert actions == {
        "AAA": "hold",
        "BBB": "exit",
        "CCC": "enter",
        "DDD": "ineligible",
        "SPY": "benchmark",
        "$CASH": "cash",
    }
    aaa = next(r for r in doc["rows"] if r["ticker"] == "AAA")
    assert aaa["target_notional_usd"] == 60_000.0 and aaa["alpha_rank"] == 1
    assert aaa["target_weight_pct"] == 60.0 and aaa["book_position"] == 1
    assert doc["summary"]["positions"] == 2 and doc["previous_book"] == "given"
    assert [r["ticker"] for r in doc["rows"]][:2] == ["AAA", "CCC"]  # biggest target first


def test_an_eligible_name_with_no_weight_is_not_selected() -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc="t")
    assert {r["ticker"]: r["action"] for r in doc["rows"]}["BBB"] == "not_selected"
    assert doc["previous_book"] == "all_cash"
    # codex ac64b346aa382999: an all-cash previous book records cash at 1.0.
    cash = next(r for r in doc["rows"] if r["ticker"] == "$CASH")
    assert cash["previous_weight"] == 1.0 and cash["weight_change"] == pytest.approx(-0.9)


def test_the_book_is_written_under_its_day_and_is_a_book_of_record(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    assert key == order_book_key("2026-11-05") == "trader/order_books/2026-11-05.json"
    assert is_books_key(key)
    doc = json.loads(ctx.store.get_bytes(key))
    assert doc["schema_version"] == "order_book.v1" and len(doc["rows"]) == 6
    assert doc["stage"] == STAGE_SESSION
    assert "no pre-open book" in ctx.metrics[-1]["status_reason"]
    assert ctx.metrics[-1]["name"] == ORDER_BOOK_METRIC and ctx.metrics[-1]["status"] == "OK"


def test_a_failed_write_never_stops_the_session(tmp_path) -> None:
    class Refusing(_Ctx):
        def record_output_cas(self, *a, **k) -> str:
            raise PermissionError("AccessDenied")

    ctx = Refusing(LocalStore(tmp_path))
    assert record_order_book(ctx, _book(), previous_weights=None, now=NOW) is None
    assert (
        ctx.metrics[-1]["status"] == "FAIL" and "AccessDenied" in ctx.metrics[-1]["status_reason"]
    )


def test_a_weight_moved_up_or_down_is_increase_or_reduce() -> None:
    doc = order_book_document(
        _book(), previous_weights={"AAA": 0.4, "CCC": 0.5}, written_at_utc="t"
    )
    actions = {r["ticker"]: r["action"] for r in doc["rows"]}
    assert actions["AAA"] == "increase" and actions["CCC"] == "reduce"


def test_a_refused_lock_is_a_fail_metric_but_the_book_stays_written(tmp_path, monkeypatch) -> None:
    import crucible_trader.order_book as ob

    monkeypatch.setattr(
        ob, "lock_books_records", lambda store, keys, **k: ([], {keys[0]: "no lock"})
    )
    ctx = _Ctx(LocalStore(tmp_path))
    assert record_order_book(ctx, _book(), previous_weights=None, now=NOW) is not None
    assert ctx.metrics[-1]["status"] == "FAIL" and "no lock" in ctx.metrics[-1]["status_reason"]


def test_a_metric_that_cannot_be_recorded_does_not_raise(tmp_path) -> None:
    class NoMetrics(_Ctx):
        def record_metric(self, metric) -> None:
            raise RuntimeError("manifest closed")

    assert record_order_book(
        NoMetrics(LocalStore(tmp_path)), _book(), previous_weights=None, now=NOW
    )


# --- The pre-open book (alpha-engine-config-I12024, Brian 2026-11-09/10) -----

PRE_OPEN = dt.datetime(2026, 11, 5, 13, 40, tzinfo=dt.UTC)  # 08:40 ET


def _with_weights(**weights: float) -> TargetBook:
    book = _book()
    return dataclasses.replace(book, weights={**book.weights, **weights})


def test_the_pre_open_book_is_written_once_at_the_day_key_and_locked(tmp_path, monkeypatch) -> None:
    import crucible_trader.order_book as ob

    locked: list[str] = []
    monkeypatch.setattr(
        ob, "lock_books_records", lambda store, keys, **k: (locked.extend(keys) or keys, {})
    )
    store = LocalStore(tmp_path)
    key, written = write_pre_open_order_book(store, _book(), previous_weights=None, now=PRE_OPEN)
    assert written and key == order_book_key("2026-11-05") and locked == [key]
    doc = json.loads(store.get_bytes(key))
    assert doc["stage"] == STAGE_PRE_OPEN and doc["written_at_utc"] == "2026-11-05T13:40:00Z"
    # A rerun leaves the first book exactly as it was.
    before = store.get_bytes(key)
    again = write_pre_open_order_book(
        store, _with_weights(AAA=0.5, CCC=0.4), previous_weights=None, now=NOW
    )
    assert again == (key, False) and store.get_bytes(key) == before and locked == [key]


def test_a_pre_open_book_that_cannot_be_locked_fails_the_plan(tmp_path, monkeypatch) -> None:
    import crucible_trader.order_book as ob

    monkeypatch.setattr(ob, "lock_books_records", lambda store, keys, **k: ([], {"*": "denied"}))
    with pytest.raises(RuntimeError, match="written but not locked: denied"):
        write_pre_open_order_book(LocalStore(tmp_path), _book(), previous_weights=None, now=NOW)


def test_the_session_reuses_a_pre_open_book_it_matches(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    key, _ = write_pre_open_order_book(ctx.store, _book(), previous_weights=None, now=PRE_OPEN)
    before = ctx.store.get_bytes(key)
    assert record_order_book(ctx, _book(), previous_weights=None, now=NOW) == key
    assert ctx.store.get_bytes(key) == before  # never rewritten
    assert not ctx.store.exists(as_traded_key("2026-11-05")) and ctx.outputs == []
    metric = ctx.metrics[-1]
    assert metric["status"] == "OK" and metric["value"] == 1.0
    assert "reused" in metric["status_reason"] and "pre_open" in metric["status_reason"]


def test_a_session_book_that_differs_is_kept_under_its_own_key(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    key, _ = write_pre_open_order_book(ctx.store, _book(), previous_weights=None, now=PRE_OPEN)
    before = ctx.store.get_bytes(key)
    traded = record_order_book(ctx, _with_weights(AAA=0.5, BBB=0.1), previous_weights=None, now=NOW)
    assert traded == as_traded_key("2026-11-05") == "trader/order_books/as_traded/2026-11-05.json"
    assert is_books_key(traded) and ctx.store.get_bytes(key) == before
    doc = json.loads(ctx.store.get_bytes(traded))
    assert doc["stage"] == STAGE_AS_TRADED
    assert doc["pre_open_book"] == {
        "key": key,
        "stage": STAGE_PRE_OPEN,
        "written_at_utc": "2026-11-05T13:40:00Z",
        "changed_targets": ["AAA", "BBB"],
    }
    assert ctx.outputs == [{"key": traded, "schema_version": "order_book.v1"}]
    assert ctx.metrics[-1]["status"] == "OK" and "2 target(s)" in ctx.metrics[-1]["status_reason"]


def test_an_as_traded_book_that_cannot_be_locked_is_a_fail_metric(tmp_path, monkeypatch) -> None:
    import crucible_trader.order_book as ob

    ctx = _Ctx(LocalStore(tmp_path))
    write_pre_open_order_book(ctx.store, _book(), previous_weights=None, now=PRE_OPEN)
    monkeypatch.setattr(
        ob, "lock_books_records", lambda store, keys, **k: ([], {keys[0]: "no lock"})
    )
    traded = record_order_book(ctx, _with_weights(AAA=0.5), previous_weights=None, now=NOW)
    assert traded == as_traded_key("2026-11-05")
    assert ctx.metrics[-1]["status"] == "FAIL" and "no lock" in ctx.metrics[-1]["status_reason"]


def test_a_name_in_one_book_only_is_a_changed_target() -> None:
    a = {"rows": [{"ticker": "AAA", "target_weight": 0.5}, {"ticker": "BBB", "target_weight": 0.1}]}
    b = {"rows": [{"ticker": "AAA", "target_weight": 0.5}, {"ticker": "CCC", "target_weight": 0.0}]}
    assert changed_targets(a, b) == ["BBB", "CCC"]
