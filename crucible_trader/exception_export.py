"""Export historical reconciliation breaks as `exception.v1` records. Read-only.

`alpha-engine-config-I12892` phase 0: the exception desk's queue has to start
with every break since the v2 cutover, not only the ones filed after the daily
job learned to write records (:func:`crucible_trader.reconciliation_job.file_exception_records`).
This rebuilds them from what the job already stored:

* every ``broker_reconciliation.v1`` document at
  `crucible.keys.trader_reconciliation_key` (findings, ``control_arm``, ``void``);
* every ``order_intent_reconciliation`` document beside it, where one exists;
* each day's `trader.reconcile` manifest, read through
  `crucible.manifest.read_manifest` (validated), for the detection time
  (``finished``).

It converts with the SAME functions the job uses
(:mod:`crucible_trader.exception_record`), so an exported break carries the same
``exception_id`` as the record the job files for it. A stored document a record
cannot be built from raises, naming its key: an export that skipped it would be
a queue that silently lost a break.

The store is wrapped in `crucible.store.read_only`, so this module structurally
cannot write to it. Output is JSON lines -- one record per line -- to stdout or
``--out``, a local file.

    python -m crucible_trader.exception_export --store URI --since YYYY-MM-DD \\
        [--until YYYY-MM-DD] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import cast

from crucible.keys import trader_reconciliation_key  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs; same import is baselined in every other module
from crucible.manifest import read_manifest  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs
from crucible.store import Store, open_store, read_only  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs

from crucible_trader.exception_record import (
    DETECTING_JOB,
    ExceptionRecordError,
    records_from_broker_reconciliation,
    records_from_order_intent_reconciliation,
)
from crucible_trader.order_intent_reconciliation import (
    ORDER_INTENT_RECONCILIATION_PREFIX,
    order_intent_reconciliation_key,
)
from crucible_trader.reconciliation_job import RECONCILIATION_PREFIX

__all__ = ["export_exceptions", "main"]


def _days(store: Store, prefix: str) -> set[str]:
    return {
        key.removeprefix(prefix).removesuffix(".json")
        for key in store.list_keys(prefix)
        if key.endswith(".json")
    }


def _load(store: Store, key: str) -> Mapping[str, object]:
    raw: object = json.loads(store.get_bytes(key).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ExceptionRecordError(f"{key} is not a JSON object")
    return cast("dict[str, object]", raw)


def export_exceptions(
    store: Store, *, since: str, until: str | None = None
) -> list[dict[str, object]]:
    """Every break stored for a trading day in ``[since, until]`` (``until``
    open-ended when ``None``), as validated `exception.v1` documents, by day,
    in the order the job would file them."""
    out: list[dict[str, object]] = []
    for day in sorted(_days(store, RECONCILIATION_PREFIX)):
        if day < since or (until is not None and day > until):
            continue
        key = trader_reconciliation_key(day)
        try:
            manifest = cast("dict[str, object]", read_manifest(store, DETECTING_JOB, day))
            document = _load(store, key)
            # `finished` is validated by `read_manifest`; the record re-checks the stamp.
            records = records_from_broker_reconciliation(
                document, detected_at=str(manifest["finished"])
            )
            intent_key = order_intent_reconciliation_key(day)
            if store.exists(intent_key):
                records += records_from_order_intent_reconciliation(
                    _load(store, intent_key), void=document["void"] is True
                )
            out.extend(r.to_document() for r in records)
        except Exception as exc:
            raise ExceptionRecordError(f"{key}: {type(exc).__name__}: {exc}") from exc
    orphan_intent_days = {
        d
        for d in _days(store, ORDER_INTENT_RECONCILIATION_PREFIX)
        if d >= since and (until is None or d <= until)
    } - _days(store, RECONCILIATION_PREFIX)
    if orphan_intent_days:
        raise ExceptionRecordError(
            "order-intent reconciliation documents with no broker reconciliation "
            f"beside them: {sorted(orphan_intent_days)}"
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible_trader.exception_export",
        description="Read-only export of stored reconciliation breaks as exception.v1 records.",
    )
    parser.add_argument("--store", default=None, help="store URI (default: $CRUCIBLE_STORE)")
    parser.add_argument("--since", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--until", default=None, type=dt.date.fromisoformat)
    parser.add_argument(
        "--out", default=None, type=pathlib.Path, help="local file (default: stdout)"
    )
    args = parser.parse_args(argv)
    store = read_only(
        open_store(cast("str | None", args.store)),
        reason="exception_export is read-only",
    )
    until = cast("dt.date | None", args.until)
    records = export_exceptions(
        store,
        since=cast("dt.date", args.since).isoformat(),
        until=until.isoformat() if until is not None else None,
    )
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    out = cast("pathlib.Path | None", args.out)
    if out is None:
        sys.stdout.write(lines)
    else:
        out.write_text(lines, encoding="utf-8")
    sys.stderr.write(f"{len(records)} exception records exported\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
