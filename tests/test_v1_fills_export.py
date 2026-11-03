"""v1's trade log exports as the dry run's --fills (alpha-engine-config-I13124)."""

from __future__ import annotations

import json
import pathlib
import runpy
import sqlite3
import sys

import pytest

from crucible_trader.back_office_dry_run import load_fills
from crucible_trader.v1_fills_export import export_fills, main, read_fills

# v1's `trades` columns the export reads, plus the two it must NOT read.
_SCHEMA = """
CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY, date TEXT NOT NULL, ticker TEXT NOT NULL,
    action TEXT NOT NULL, shares INTEGER NOT NULL, filled_shares INTEGER,
    fill_price REAL, fill_time TEXT, commission_usd REAL, status TEXT, trading_day TEXT
)
"""

ROWS = [
    # an entry: v1's `trading_day` is a day early and `shares` is not what filled
    (
        "t1",
        "2026-10-02",
        "AAA",
        "ENTER",
        400,
        425,
        146.49,
        "2026-10-02T13:37:00",
        2.126275,
        "Filled",
        "2026-10-01",
    ),
    (
        "t2",
        "2026-10-08",
        "BBB",
        "EXIT",
        600,
        1856,
        61.4327,
        "2026-10-08T13:31:45",
        11.634363,
        "Filled",
        "2026-10-07",
    ),
    (
        "t3",
        "2026-10-08",
        "CCC",
        "REDUCE",
        5,
        5,
        10.0,
        "2026-10-08T13:31:40",
        None,
        "Filled",
        "2026-10-07",
    ),
    # not October, and not filled: neither exports
    (
        "t4",
        "2026-11-02",
        "AAA",
        "EXIT",
        425,
        425,
        150.0,
        "2026-11-02T13:30:00",
        1.0,
        "Filled",
        "2026-10-30",
    ),
    ("t5", "2026-10-09", "DDD", "ENTER", 10, 0, None, None, None, "Cancelled", "2026-10-08"),
]


def make_db(path: pathlib.Path, rows=ROWS) -> pathlib.Path:
    connection = sqlite3.connect(path)
    connection.execute(_SCHEMA)
    connection.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def db(tmp_path) -> pathlib.Path:
    return make_db(tmp_path / "trades.db")


def test_a_month_of_v1_trades_reads_as_signed_fills_on_their_session(db):
    fills = read_fills(db, "2026-10")
    assert [f["exec_id"] for f in fills] == ["v1:t1", "v1:t3", "v1:t2"]
    entry, small, exit_ = fills
    assert entry == {
        "trading_day": "2026-10-02",
        "ticker": "AAA",
        "shares": 425,
        "cash_usd": -62260.376275,  # -425 x 146.49 - 2.126275
        "commission_usd": 2.126275,
        "exec_id": "v1:t1",
    }
    assert exit_["shares"] == -1856 and exit_["trading_day"] == "2026-10-08"
    assert exit_["cash_usd"] == pytest.approx(1856 * 61.4327 - 11.634363, abs=1e-6)
    assert small["commission_usd"] is None and small["cash_usd"] == 50.0


def test_the_month_ends_at_its_last_day(db):
    assert [f["exec_id"] for f in read_fills(db, "2026-11")] == ["v1:t4"]
    assert read_fills(db, "2026-12") == []


def test_a_filled_row_that_is_not_a_fill_stops_the_export(tmp_path):
    bad = ("t9", "2026-10-05", "EEE", "HEDGE", 1, 1, 1.0, "x", 0.0, "Filled", "2026-10-02")
    with pytest.raises(ValueError, match="v1 trade t9 .* not a fill"):
        read_fills(make_db(tmp_path / "bad.db", [bad]), "2026-10")


def test_the_file_is_what_the_dry_run_reads(db, tmp_path, capsys):
    out = tmp_path / "fills.jsonl"
    assert main(["--db", str(db), "--month", "2026-10", "--out", str(out)]) == 0
    assert "2026-10: 3 v1 fill(s)" in capsys.readouterr().err
    fills = load_fills(out)
    assert sorted(fills) == ["2026-10-02", "2026-10-08"]
    assert fills["2026-10-08"][1].shares == -1856
    assert fills["2026-10-02"][0].exec_id == "v1:t1"
    assert json.loads(out.read_text().splitlines()[0])["ticker"] == "AAA"


def test_the_source_is_never_written(db, tmp_path):
    before = db.read_bytes()
    assert export_fills(db, "2026-10", tmp_path / "f.jsonl") == 3
    assert db.read_bytes() == before


def test_a_missing_db_and_an_existing_out_are_refused(db, tmp_path):
    with pytest.raises(SystemExit):
        main(
            ["--db", str(tmp_path / "nope.db"), "--month", "2026-10", "--out", str(tmp_path / "o")]
        )
    existing = tmp_path / "existing.jsonl"
    existing.write_text("keep\n")
    with pytest.raises(SystemExit):
        main(["--db", str(db), "--month", "2026-10", "--out", str(existing)])
    assert existing.read_text() == "keep\n"


def test_runs_as_a_module(db, tmp_path, monkeypatch):
    out = tmp_path / "module.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["v1_fills_export", "--db", str(db), "--month", "2026-10", "--out", str(out)]
    )
    sys.modules.pop("crucible_trader.v1_fills_export", None)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("crucible_trader.v1_fills_export", run_name="__main__")
    assert exited.value.code == 0 and out.is_file()
