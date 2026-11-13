"""The day's order book (Brian 2026-11-05): every name considered, kept."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.keys import features_key
from crucible.slots.strategy import BookUniverse
from crucible.store import LocalStore

from crucible_trader.books_retention import is_books_key
from crucible_trader.construction import TargetBook
from crucible_trader.order_book import (
    HELD_SESSIONS_WALK_LIMIT,
    ORDER_BOOK_METRIC,
    ShareBasis,
    held_sessions_from_order_books,
    order_book_document,
    order_book_key,
    record_order_book,
    sector_labels,
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
    assert ctx.metrics[-1]["name"] == ORDER_BOOK_METRIC and ctx.metrics[-1]["status"] == "OK"


def test_a_failed_write_never_stops_the_session(tmp_path) -> None:
    class Refusing(_Ctx):
        def record_output(self, *a, **k) -> None:
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


# -- the minimum hold's clock (alpha-engine-config-I12024) -----------------------


def _write_book(store, day: str, held: dict[str, float], *, stamps: dict | None = None) -> None:
    """A minimal `order_book.v1` for ``day``: ``held`` names at a positive target,
    one not-held name and the two sentinels. ``stamps`` adds `held_sessions`."""
    rows = [
        {"ticker": t, "target_weight": w, "action": "hold"} for t, w in sorted(held.items())
    ] + [
        {"ticker": "ZZZ", "target_weight": 0.0, "action": "not_selected"},
        {"ticker": "SPY", "target_weight": 0.0, "action": "benchmark"},
        {"ticker": "$CASH", "target_weight": 0.05, "action": "cash"},
    ]
    if stamps is not None:
        for row in rows:
            if row["ticker"] in stamps:
                row["held_sessions"] = stamps[row["ticker"]]
    store.put_bytes(
        order_book_key(day), json.dumps({"trading_day": day, "rows": rows}).encode("utf-8")
    )


#: Consecutive NYSE sessions, oldest first. 2026-11-11 (Veterans Day) is a
#: trading day; 2026-11-07/08 are a weekend.
D1, D2, D3, D4, D5 = "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"
TODAY = "2026-11-11"


class TestTheClockIsDerivedFromTheTradersOwnBooks:
    def test_no_previous_book_is_no_clock(self, tmp_path) -> None:
        assert held_sessions_from_order_books(LocalStore(tmp_path), TODAY) == {}

    def test_consecutive_books_count_and_a_gap_ends_the_run(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_book(store, D1, {"AAA": 0.3, "OLD": 0.2})  # before the gap: never counted
        # D2 has no book (a held day): the gap.
        _write_book(store, D3, {"AAA": 0.3, "BBB": 0.2})
        _write_book(store, D4, {"AAA": 0.3, "BBB": 0.2, "CCC": 0.1})
        _write_book(store, D5, {"AAA": 0.3, "BBB": 0.2, "CCC": 0.1, "DDD": 0.1})
        assert held_sessions_from_order_books(store, TODAY) == {
            "AAA": 3,
            "BBB": 3,
            "CCC": 2,
            "DDD": 1,
        }

    def test_a_name_that_left_the_book_restarts_its_count(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_book(store, D3, {"AAA": 0.3})
        _write_book(store, D4, {"BBB": 0.3})  # AAA sold
        _write_book(store, D5, {"AAA": 0.3, "BBB": 0.3})  # AAA bought back
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 1, "BBB": 2}

    def test_a_stamped_book_is_read_alone(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        _write_book(store, D4, {"AAA": 0.3})
        _write_book(store, D5, {"AAA": 0.3, "BBB": 0.2}, stamps={"AAA": 7, "BBB": 1, "ZZZ": 0})
        reads: list[str] = []
        real = store.get_bytes
        monkeypatch.setattr(store, "get_bytes", lambda key: reads.append(key) or real(key))
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 7, "BBB": 1}
        assert reads == [order_book_key(D5)]

    def test_the_walk_finishes_on_an_older_stamped_book(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_book(store, D3, {"AAA": 0.3}, stamps={"AAA": 4})
        _write_book(store, D4, {"AAA": 0.3, "BBB": 0.2})
        _write_book(store, D5, {"AAA": 0.3, "BBB": 0.2})
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 6, "BBB": 2}

    @pytest.mark.parametrize("bad", [0, None, True, "3"])
    def test_a_held_row_without_a_whole_count_is_walked_not_trusted(self, tmp_path, bad) -> None:
        store = LocalStore(tmp_path)
        _write_book(store, D4, {"AAA": 0.3})
        _write_book(store, D5, {"AAA": 0.3}, stamps={"AAA": bad})
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 2}

    def test_the_walk_is_capped(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        for day in (D1, D2, D3, D4, D5):
            _write_book(store, day, {"AAA": 0.3})
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 5}
        assert held_sessions_from_order_books(store, TODAY, max_books=2) == {"AAA": 2}
        assert HELD_SESSIONS_WALK_LIMIT == 30

    def test_an_unreadable_or_misfiled_book_ends_the_run(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_book(store, D3, {"AAA": 0.3})
        store.put_bytes(order_book_key(D4), b"{not json")
        _write_book(store, D5, {"AAA": 0.3})
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 1}
        _write_book(store, D4, {"AAA": 0.3})
        doc = json.loads(store.get_bytes(order_book_key(D4)))
        store.put_bytes(order_book_key(D4), json.dumps({**doc, "trading_day": D3}).encode())
        assert held_sessions_from_order_books(store, TODAY) == {"AAA": 1}
        store.put_bytes(order_book_key(D5), json.dumps({"trading_day": D5}).encode())
        assert held_sessions_from_order_books(store, TODAY) == {}


class TestTheBookStampsTheClockItCarriesForward:
    def test_each_row_carries_its_count(self) -> None:
        doc = order_book_document(
            _book(),
            previous_weights={"BBB": 0.2, "AAA": 0.6},
            written_at_utc="t",
            held_sessions={"AAA": 4, "BBB": 2},
        )
        stamps = {r["ticker"]: r["held_sessions"] for r in doc["rows"]}
        # AAA held four sessions and held again: five. CCC entered here: one.
        # BBB exited and DDD was never held: zero. The sentinels carry none.
        assert stamps == {"AAA": 5, "BBB": 0, "CCC": 1, "DDD": 0, "SPY": None, "$CASH": None}

    def test_with_no_clock_every_held_name_starts_at_one(self) -> None:
        doc = order_book_document(_book(), previous_weights=None, written_at_utc="t")
        assert {r["ticker"]: r["held_sessions"] for r in doc["rows"]}["AAA"] == 1

    def test_tomorrows_clock_is_one_read_of_todays_book(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        key = record_order_book(
            _Ctx(store),
            _book(),
            previous_weights={"AAA": 0.6},
            now=NOW,
            held_sessions={"AAA": 2},
        )
        assert key == order_book_key("2026-11-05")
        assert held_sessions_from_order_books(store, "2026-11-06") == {"AAA": 3, "CCC": 1}


def _write_sectors(store, day: str, labels: dict[str, object]) -> str:
    key = features_key(DEFAULT_FEATURE_VERSION, day)
    frame = pd.DataFrame(
        {
            "trading_day": [day] * len(labels),
            "ticker": list(labels),
            "sector_raw": list(labels.values()),
        }
    )
    store.put_bytes(key, frame.to_parquet(index=False))
    return key


def test_each_row_carries_its_gics_sector_label_beside_the_engine_sector(tmp_path) -> None:
    """alpha-engine-config-I13578: Brian read every holding as `__unsectored__`.
    The label is display only; ``sector`` stays the engine's own value."""
    ctx = _Ctx(LocalStore(tmp_path))
    source = _write_sectors(
        ctx.store,
        "2026-11-05",
        {"AAA": "Information Technology", "BBB": " Financials ", "CCC": None, "SPY": None},
    )
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    doc = json.loads(ctx.store.get_bytes(key))
    labels = {r["ticker"]: r["sector_label"] for r in doc["rows"]}
    assert labels == {
        "AAA": "Information Technology",
        "BBB": "Financials",
        "CCC": None,  # the feature layer left it unlabelled
        "DDD": None,  # not in the feature artifact
        "SPY": None,
        "$CASH": None,
    }
    assert {r["ticker"]: r["sector"] for r in doc["rows"]}["AAA"] == "t"
    assert doc["sector_label_source"] == source


def test_no_feature_artifact_leaves_the_sector_labels_null_and_the_book_written(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    doc = json.loads(ctx.store.get_bytes(key))
    assert {r["sector_label"] for r in doc["rows"]} == {None}
    assert doc["sector_label_source"] is None and ctx.metrics[-1]["status"] == "OK"


def test_an_unreadable_feature_artifact_never_costs_the_book(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    ctx.store.put_bytes(features_key(DEFAULT_FEATURE_VERSION, "2026-11-05"), b"not parquet")
    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW)
    assert key is not None and ctx.metrics[-1]["status"] == "OK"
    doc = json.loads(ctx.store.get_bytes(key))
    assert {r["sector_label"] for r in doc["rows"]} == {None}


def test_each_held_row_carries_the_shares_the_session_sized(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))
    basis = ShareBasis(
        positions={"AAA": 500, "BBB": 100},
        prices={"AAA": 110.0, "BBB": 50.0, "CCC": 33.0},
    )
    key = record_order_book(
        ctx,
        _book(),
        previous_weights={"AAA": 0.55, "BBB": 0.05},
        now=NOW,
        share_basis=lambda: basis,
    )
    doc = json.loads(ctx.store.get_bytes(key))
    rows = {r["ticker"]: r for r in doc["rows"]}
    # AAA 0.6 * 100k / 110 = 545.45 -> 545; CCC 0.3 * 100k / 33 = 909.09 -> 909.
    assert (rows["AAA"]["previous_shares"], rows["AAA"]["target_shares"]) == (500, 545)
    assert rows["AAA"]["share_price_usd"] == 110.0
    assert (rows["BBB"]["previous_shares"], rows["BBB"]["target_shares"]) == (100, 0)
    assert (rows["CCC"]["previous_shares"], rows["CCC"]["target_shares"]) == (0, 909)
    assert (rows["DDD"]["previous_shares"], rows["DDD"]["target_shares"]) == (0, 0)
    for sentinel in ("SPY", "$CASH"):
        assert rows[sentinel]["target_shares"] is None
        assert rows[sentinel]["previous_shares"] is None
    assert doc["share_basis"] == "router_plan"
    # The trade, signed: + buys, - sells; no notional where there is no price.
    trades = {t: (r["trade_shares"], r["trade_notional_usd"]) for t, r in rows.items()}
    assert trades == {
        "AAA": (45, 4950.0),
        "BBB": (-100, -5000.0),
        "CCC": (909, 29997.0),
        "DDD": (0, None),
        "SPY": (None, None),
        "$CASH": (None, None),
    }
    assert rows["AAA"]["weight_change_pct"] == 5.0
    assert rows["BBB"]["weight_change_pct"] == -5.0


def test_without_a_share_basis_the_share_columns_are_null() -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc="t")
    assert {(r["previous_shares"], r["target_shares"]) for r in doc["rows"]} == {(None, None)}
    assert {(r["trade_shares"], r["trade_notional_usd"]) for r in doc["rows"]} == {(None, None)}
    assert doc["share_basis"] is None


def test_a_failing_share_basis_never_costs_the_book(tmp_path) -> None:
    ctx = _Ctx(LocalStore(tmp_path))

    def broken() -> ShareBasis:
        raise RuntimeError("router state unreadable")

    key = record_order_book(ctx, _book(), previous_weights=None, now=NOW, share_basis=broken)
    assert key is not None and ctx.metrics[-1]["status"] == "OK"
    doc = json.loads(ctx.store.get_bytes(key))
    assert doc["share_basis"] is None


def test_a_weighted_name_with_no_usable_price_has_no_target_shares() -> None:
    basis = ShareBasis(positions={}, prices={"AAA": float("nan")})
    assert basis.target_shares("AAA", 0.5, 100_000.0) is None
    assert basis.target_shares("BBB", 0.5, 100_000.0) is None
    assert basis.target_shares("BBB", 0.0, 100_000.0) == 0


def test_a_feature_artifact_without_a_sector_column_labels_nothing(tmp_path) -> None:
    store = LocalStore(tmp_path)
    frame = pd.DataFrame({"trading_day": ["2026-11-05"], "ticker": ["AAA"]})
    store.put_bytes(
        features_key(DEFAULT_FEATURE_VERSION, "2026-11-05"), frame.to_parquet(index=False)
    )
    assert sector_labels(store, "2026-11-05", ["AAA"]) == ({}, None)


# --- the optimizer block (Brian 2026-11-12: the dashboard's optimizer charts) ---------


def test_the_book_carries_what_the_optimizer_solved_and_its_settings() -> None:
    import dataclasses

    import numpy as np

    from crucible_trader.construction import optimizer_record

    class _Params:
        risk_aversion = 4.0
        vol_target_annual = None
        covariance_shrinkage = "ledoit_wolf"
        min_hold_days = 5

    diagnostics = {
        "status": "optimal",
        "portfolio_vol_ann": np.float64(0.21),
        "active_share_vs_benchmark": 0.83,
        "n_active_positions": np.int64(5),
        "churn_controls": {"min_hold_days": 5, "floored": ["SNDK"]},
        "aim_weights": np.ones(3),  # one per name: not a fact about the day
        "not_a_display_key": 1,
    }
    record = optimizer_record(diagnostics, _Params())
    assert record["solved"] == {
        "status": "optimal",
        "portfolio_vol_ann": 0.21,
        "active_share_vs_benchmark": 0.83,
        "n_active_positions": 5,
        "churn_controls": {"min_hold_days": 5, "floored": ["SNDK"]},
    }
    assert record["settings"] == {
        "risk_aversion": 4.0,
        "vol_target_annual": None,
        "covariance_shrinkage": "ledoit_wolf",
        "min_hold_days": 5,
    }
    book = dataclasses.replace(_book(), optimizer=record)
    doc = order_book_document(book, previous_weights=None, written_at_utc="t")
    assert json.loads(json.dumps(doc))["optimizer"] == record


def test_a_value_that_will_not_serialise_is_dropped_not_fatal() -> None:
    import numpy as np

    from crucible_trader.construction import optimizer_record

    class _Params:
        risk_aversion = object()
        min_hold_days = 5

    record = optimizer_record(
        {
            "portfolio_vol_ann": float("nan"),
            "status": object(),
            "expected_alpha": 0.01,
            "cost_model": {"breaks": np.array([1.0, 2.0])},
        },
        _Params(),
    )
    assert record == {
        "solved": {"expected_alpha": 0.01, "cost_model": {"breaks": [1.0, 2.0]}},
        "settings": {"min_hold_days": 5},
    }


def test_a_book_without_an_optimizer_block_writes_null() -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc="t")
    assert doc["optimizer"] is None


def test_the_book_carries_reader_facing_percentages_and_an_exchange_clock_build_time() -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc="2026-11-13T14:32:14Z")
    assert doc["built_at_et"] == "2026-11-13 09:32 ET"
    assert doc["summary"]["cash_weight_pct"] == round(doc["summary"]["cash_weight"] * 100, 3)
    assert doc["turnover_one_way_pct"] == round(doc["turnover_one_way_ratio"] * 100, 3)


@pytest.mark.parametrize("stamp", ["t", "2026-11-13T14:32:14"])
def test_an_unreadable_or_zoneless_build_time_reads_null(stamp) -> None:
    doc = order_book_document(_book(), previous_weights=None, written_at_utc=stamp)
    assert doc["built_at_et"] is None


def test_a_resize_the_plan_left_inside_the_band_reads_as_the_shares_held() -> None:
    basis = ShareBasis(
        positions={"AAA": 300, "BBB": 100}, prices={"AAA": 100.0, "BBB": 50.0}, band_usd=1_000.0
    )
    # AAA 300 -> 299 is $100 inside the band: held. BBB 100 -> 400 is $15,000: sized.
    assert basis.target_shares("AAA", 0.299, 100_000.0) == 300
    assert basis.target_shares("BBB", 0.2, 100_000.0) == 400
    # Entries are never banded.
    assert basis.target_shares("CCC", 0.0, 100_000.0) == 0
    assert (
        ShareBasis(positions={}, prices={"AAA": 100.0}, band_usd=1_000.0).target_shares(
            "AAA", 0.005, 100_000.0
        )
        == 5
    )
