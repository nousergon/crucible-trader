"""The declared v1 handover (`crucible_trader.inherited_positions`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible_trader.inherited_positions import (
    InheritedPositionsError,
    inherited_for,
    load_inherited,
)

V1_BOOK = {
    "ANF", "DECK", "DOCS", "DUOL", "LNTH", "MRNA", "MU", "NVDA", "PATH", "PLTR", "SPY", "UAL",
}  # fmt: skip


def test_the_committed_declaration_names_v1s_book_from_the_handover_session() -> None:
    effective_from, symbols = load_inherited()
    assert effective_from == "2026-11-03"
    assert symbols == V1_BOOK


def test_nothing_is_inherited_before_the_handover_session() -> None:
    assert inherited_for("2026-11-02") == frozenset()
    assert inherited_for("2026-11-03") == V1_BOOK
    assert inherited_for("2026-11-04") == V1_BOOK


def _write(tmp_path: Path, **overrides: object) -> Path:
    doc: dict[str, object] = {
        "schema_version": 1,
        "effective_from": "2026-11-03",
        "symbols": ["AAA"],
        "source": "s",
        "ruling": "r",
    }
    doc.update(overrides)
    path = tmp_path / "inherited.json"
    path.write_text(json.dumps(doc))
    return path


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"ruling": ""}, "ruling"),
        ({"source": None}, "source"),
        ({"symbols": []}, "symbols"),
        ({"symbols": ["AAA", "AAA"]}, "twice"),
        ({"symbols": [" AAA"]}, "symbol"),
        ({"symbols": [1]}, "symbol"),
    ],
)
def test_a_malformed_declaration_is_refused(tmp_path, overrides, match) -> None:
    with pytest.raises(InheritedPositionsError, match=match):
        load_inherited(_write(tmp_path, **overrides))


def test_a_non_object_or_bad_date_is_refused(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text("[]")
    with pytest.raises(InheritedPositionsError, match="object"):
        load_inherited(path)
    with pytest.raises(ValueError):
        load_inherited(_write(tmp_path, effective_from="11/03/2026"))
