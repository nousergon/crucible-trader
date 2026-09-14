"""The trader's paper smoke: the gate `crucible release.pin --target trader` reads.

`alpha-engine-config-I10649` deliverable 2. One run, inside
`crucible.runner.run_job` so it always files
`runs/trader.smoke/{day}/{sha12}/run.json` with `release_sha` set to the build
under test -- the exact manifest `crucible.release.passing_trader_smoke` selects:

1. **the running wheel is the build under test** -- the installed `crucible`
   distribution's version must be the one `crucible.release.wheel_filename_for`
   names for ``sha``. A smoke that ran on a different wheel certifies nothing;
2. **connect to IB Gateway PAPER** through a `readonly` API session
   (:func:`crucible_trader.broker_session.connect`), INSIDE the job, so a
   logged-out gateway still files a failed manifest naming the human step;
3. **read the book** through :class:`~crucible_trader.broker_session.ReadOnlyBroker`
   and :class:`~crucible_trader.broker_statement.IbPaperStatementSource`, which
   refuse unreadable positions and unreadable cash rather than reading them as
   empty or zero;
4. **place no order.** The session handed to the reader has no order method at
   all; the test suite asserts zero order calls against a fake client that
   records every one.

`scripts/trader_paper_smoke.sh` is the box-side entry: it installs the pinned
wheel into a fresh environment and calls :func:`main` inside it.

**Registered in the harness contract** (crucible-PR297): `trader.smoke` is in
`crucible.models.JOB_VALUES`, so the runner's single writer files a
schema-valid manifest that `crucible.release.passing_trader_smoke` can select.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from crucible.release import (
    TRADER_PIN_KEY,
    TRADER_SMOKE_JOB,
    assert_sha,
    read_pointer,
    wheel_filename_for,
)
from crucible.runner import RunContext, run_job
from crucible.store import Store, open_store

from crucible_trader.broker_session import GatewayAddress, ReadOnlyBroker, connect, load_sdk
from crucible_trader.broker_statement import IbPaperStatementSource
from crucible_trader.settings import Settings

SMOKE_SCHEMA_VERSION = "trader_paper_smoke.v1"
PAPER_SMOKE_PREFIX = "trader/paper_smoke/"
METRIC_MODULE = "crucible_trader.paper_smoke"


def paper_smoke_key(trading_day: str, sha: str) -> str:
    """Declared here only until `crucible.keys` owns it (as `reconciliation_key`)."""
    return f"{PAPER_SMOKE_PREFIX}{trading_day}/{assert_sha(sha)}.json"


class RunningWheelMismatchError(RuntimeError):
    """The installed `crucible` is not the wheel named by the sha in question."""


def open_configured_store(settings: Settings) -> Store:
    """The store `Settings` names. `Settings` admits `file://` for tests and laptop
    dry runs; `crucible.store.open_store` takes a bare local path for that backend
    and refuses an unknown scheme, so the one translation lives here."""
    uri = settings.store_uri
    return open_store(uri.removeprefix("file://") if uri.startswith("file://") else uri)


def installed_crucible_version() -> str:
    return importlib.metadata.version("crucible")


def expected_version(sha: str) -> str:
    """`crucible-{version}-py3-none-any.whl` -> `{version}`, from the harness's own
    derivation rather than restated here."""
    return wheel_filename_for(sha).split("-")[1]


def assert_running_wheel(sha: str, installed_version: str) -> None:
    want = expected_version(sha)
    if installed_version != want:
        raise RunningWheelMismatchError(
            f"the running crucible wheel is {installed_version!r}, not {want!r} (release "
            f"{sha}). A smoke on another build certifies nothing about this one."
        )


def running_wheel_disagreement(store: Store, installed_version: str) -> str | None:
    """I10649 deliverable 5's page condition, for the trader session to check at start.

    ``None`` when `trader/release_pin` names the running wheel; the reason
    otherwise, including an unset pin (a trader running on a wheel nobody
    pinned). The session raises on a non-None result inside its own `run_job`,
    which makes its manifest `failed` -- the page.
    """
    sha, _ = read_pointer(store, TRADER_PIN_KEY)
    if sha is None:
        return f"{TRADER_PIN_KEY} is unset while the trader runs crucible {installed_version!r}"
    want = expected_version(sha)
    if installed_version != want:
        return (
            f"{TRADER_PIN_KEY} names {sha} ({want!r}) but the trader is running crucible "
            f"{installed_version!r}"
        )
    return None


def smoke_cycle(ctx: RunContext, *, sha: str, client: Any, installed_version: str) -> dict:
    """The job body. Raises on every failure; returns the output document."""
    assert_running_wheel(sha, installed_version)
    statement = IbPaperStatementSource(ReadOnlyBroker(client)).read_statement(
        ctx.trading_day.isoformat()
    )
    key = paper_smoke_key(statement.trading_day, sha)
    document = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "release_sha": sha,
        "crucible_version": installed_version,
        "account": statement.account,
        "trading_day": statement.trading_day,
        "n_positions": len(statement.positions),
        "cash_readable": True,
    }
    ctx.record_output(
        key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"), SMOKE_SCHEMA_VERSION
    )
    ctx.record_rows(rows_in=len(statement.positions), rows_out=1)
    ctx.record_metric(
        {
            "name": "trader_paper_smoke_book_readable",
            "module": METRIC_MODULE,
            "metric_type": "operational",
            "value": 1.0,
            "unit": "ratio",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"paper book read on release {sha[:12]}: {len(statement.positions)} positions, "
                "cash readable, through a session with no order method"
            ),
            "source_path": key,
            "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return document


def run_smoke(
    store: Store,
    sha: str,
    *,
    connector: Callable[[], Any],
    installed_version: str,
    trading_day: dt.date | None = None,
    run_mode: str | None = None,
) -> RunContext:
    """Run one smoke as `trader.smoke`, always disconnecting what it connected."""
    assert_sha(sha)
    clients: list[Any] = []

    def body(ctx: RunContext) -> dict:
        client = connector()
        clients.append(client)
        return smoke_cycle(ctx, sha=sha, client=client, installed_version=installed_version)

    try:
        return run_job(
            TRADER_SMOKE_JOB,
            body,
            store=store,
            trading_day=trading_day,
            release_sha=sha,
            discriminator=sha[:12],
            run_mode=run_mode,
            # One attempt: a gateway session that is not there is a human action,
            # not a transient the runner's retry class should absorb.
            transient_retry=False,
        )
    finally:
        for client in clients:
            client.disconnect()


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    sdk_loader: Callable[[], Any] = load_sdk,
    version_reader: Callable[[], str] = installed_crucible_version,
) -> int:
    parser = argparse.ArgumentParser(prog="crucible_trader.paper_smoke")
    parser.add_argument("--release", required=True, metavar="SHA")
    parser.add_argument("--run-mode", choices=("live", "replay"), default="live")
    args = parser.parse_args(argv)
    source = None if environ is None else dict(environ)
    settings = Settings.from_env(source)
    address = GatewayAddress.from_env(source)
    run_smoke(
        open_configured_store(settings),
        args.release,
        connector=lambda: connect(address, readonly=True, sdk_loader=sdk_loader),
        installed_version=version_reader(),
        run_mode=args.run_mode,
    )
    return 0
