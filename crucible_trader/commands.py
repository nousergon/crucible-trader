"""The operator commands: the kill switch, the fire drill, and reconciliation.

`alpha-engine-config-I10650`, `-I11071`. Each command is one
`crucible.runner.run_job` run, so every fire, release, drill and reconcile
files a manifest whatever happens:

    kill-switch fire --mode flatten|freeze --cause TEXT     job trader.kill_switch
    kill-switch release --cause TEXT                        job trader.kill_switch
    fire-drill seal --from DAY --to DAY --fire-at INSTANT    operator identity; seals a drill
    fire-drill run --kind KIND --announced                  job trader.fire_drill
    fire-drill run --kind KIND --unannounced --schedule-id ID --window-from DAY
                   --window-to DAY --fire-at INSTANT        nonce in $CRUCIBLE_TRADER_DRILL_NONCE
    fire-drill status --from YYYY-MM-DD --to YYYY-MM-DD     read-only; exit 1 unless done
    reconcile run [--genesis]                                job trader.reconcile

`seal` runs under the OPERATOR identity (it writes an SSM SecureString the
trader's identity is denied) and never connects to the broker
(`crucible_trader.drill_schedule`, `alpha-engine-config-I10761`). `status` is the
harness's own phase-4 reading (`crucible.gate.kill_switch_fire_drill_reading`).

Invoked on the box as ``python -c 'import sys; from crucible_trader.commands
import main; sys.exit(main(sys.argv[1:]))' <command> ...`` (one line).
(no `__main__` block: this package's coverage floor has no exclusions).

`fire` and `run` connect to IB Gateway PAPER with an order-capable session (the
switch cancels and closes); `release` and `status` never connect. `reconcile run`
connects READ-ONLY (`crucible_trader.broker_session.ReadOnlyBroker`) -- it never
places an order -- and its `source`/`fills` come from that session via
`crucible_trader.reconcile_entrypoint`; `cash_flows` and `corporate_actions`
have no live source yet and default to the reconciler's own documented
fail-loud state (see that module's docstring; follow-up `alpha-engine-config-I11072`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from crucible.gate import kill_switch_fire_drill_reading
from crucible.runner import RunContext, run_job

from crucible_trader.broker_session import GatewayAddress, connect, load_sdk
from crucible_trader.drill_schedule import (
    NONCE_ENV_VAR,
    SealOpening,
    seal_schedule,
)
from crucible_trader.fire_drill import DRILL_KINDS, FIRE_DRILL_JOB, drill_cycle
from crucible_trader.kill_switch import (
    KILL_SWITCH_JOB,
    IbBrokerControl,
    fire,
    record_outcome,
    release,
    utcnow,
)
from crucible_trader.paper_smoke import open_configured_store
from crucible_trader.reconcile_entrypoint import (
    fills_from_ib,
    live_corporate_actions,
    live_statement_source,
)
from crucible_trader.reconciliation_job import RECONCILE_JOB, reconcile_cycle
from crucible_trader.settings import Settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crucible_trader.commands")
    top = parser.add_subparsers(dest="command", required=True)

    switch = top.add_parser("kill-switch").add_subparsers(dest="action", required=True)
    fire_p = switch.add_parser("fire")
    fire_p.add_argument("--mode", choices=("flatten", "freeze"), required=True)
    fire_p.add_argument("--cause", required=True)
    release_p = switch.add_parser("release")
    release_p.add_argument("--cause", required=True)

    drill = top.add_parser("fire-drill").add_subparsers(dest="action", required=True)
    run_p = drill.add_parser("run")
    run_p.add_argument("--kind", choices=DRILL_KINDS, required=True)
    announced = run_p.add_mutually_exclusive_group(required=True)
    announced.add_argument("--announced", dest="announced", action="store_true")
    announced.add_argument("--unannounced", dest="announced", action="store_false")
    for flag in ("--schedule-id", "--window-from", "--window-to", "--fire-at"):
        run_p.add_argument(flag)
    seal_p = drill.add_parser("seal")
    seal_p.add_argument("--from", dest="window_start", required=True)
    seal_p.add_argument("--to", dest="window_end", required=True)
    seal_p.add_argument("--fire-at", dest="fire_at", required=True)
    status_p = drill.add_parser("status")
    status_p.add_argument("--from", dest="window_start", required=True)
    status_p.add_argument("--to", dest="window_end", required=True)

    reconcile = top.add_parser("reconcile").add_subparsers(dest="action", required=True)
    reconcile_run_p = reconcile.add_parser("run")
    reconcile_run_p.add_argument("--genesis", action="store_true")

    for sub in (fire_p, release_p, run_p, reconcile_run_p):
        sub.add_argument("--run-mode", choices=("live", "replay"), default="live")
    return parser


def default_ssm_client() -> Any:
    """The SSM client `fire-drill seal` writes the nonce with, from the process's
    own credential chain (the operator's, never the trader box's)."""
    import boto3  # noqa: PLC0415 - only the seal command needs it

    return boto3.client("ssm")


_SEAL_FLAGS = ("schedule_id", "window_from", "window_to", "fire_at")


def _seal_opening(
    parser: argparse.ArgumentParser, args: argparse.Namespace, source: Mapping[str, str]
) -> SealOpening | None:
    given = [flag for flag in _SEAL_FLAGS if getattr(args, flag) is not None]
    if args.announced:
        if given:
            parser.error("an --announced drill takes no seal flags")
        return None
    missing = [f"--{flag.replace('_', '-')}" for flag in _SEAL_FLAGS if flag not in given]
    if missing:
        parser.error(f"an --unannounced drill fires under a sealed schedule: missing {missing}")
    nonce = (source.get(NONCE_ENV_VAR) or "").strip()
    if not nonce:
        parser.error(f"an --unannounced drill needs the seal's nonce in ${NONCE_ENV_VAR}")
    return SealOpening(args.schedule_id, args.window_from, args.window_to, args.fire_at, nonce)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    sdk_loader: Callable[[], Any] = load_sdk,
    clock: Callable[[], dt.datetime] = utcnow,
    printer: Callable[[str], None] = print,
    ssm_factory: Callable[[], Any] = default_ssm_client,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source = None if environ is None else dict(environ)
    store = open_configured_store(Settings.from_env(source))

    if args.command == "fire-drill" and args.action == "status":
        reading = kill_switch_fire_drill_reading(
            store,
            dt.date.fromisoformat(args.window_start),
            dt.date.fromisoformat(args.window_end),
        )
        verdict = "DONE" if reading.met else "UNMEASURABLE" if reading.unmeasurable else "NOT DONE"
        printer(f"{verdict}: {reading.detail}")
        return 0 if reading.met else 1

    if args.command == "fire-drill" and args.action == "seal":
        seal = seal_schedule(
            store,
            ssm_factory(),
            window_start=args.window_start,
            window_end=args.window_end,
            fire_instant=args.fire_at,
            clock=clock,
        )
        printer(
            f"SEALED {seal.schedule_id}: {seal.key}; nonce in SSM {seal.parameter} (not printed). "
            f"At {seal.fire_instant}: {NONCE_ENV_VAR}=<nonce> fire-drill run --kind KIND "
            f"--unannounced --schedule-id {seal.schedule_id} --window-from {seal.window_start} "
            f"--window-to {seal.window_end} --fire-at {seal.fire_instant}"
        )
        return 0

    seal_opening = (
        _seal_opening(parser, args, source if source is not None else os.environ)
        if args.command == "fire-drill"
        else None
    )

    if args.command == "kill-switch" and args.action == "release":
        job, stamp, connects = KILL_SWITCH_JOB, "release", False
    elif args.command == "kill-switch":
        job, stamp, connects = KILL_SWITCH_JOB, f"fire-{args.mode}", True
    elif args.command == "reconcile":
        job, stamp, connects = RECONCILE_JOB, "run", True
    else:
        job, stamp, connects = FIRE_DRILL_JOB, args.kind, True

    address = GatewayAddress.from_env(source) if connects else None
    clients: list[Any] = []

    def body(ctx: RunContext) -> Any:
        if job == RECONCILE_JOB:
            sdk = sdk_loader()
            client = connect(address, readonly=True, sdk_loader=lambda: sdk)
            clients.append(client)
            return reconcile_cycle(
                ctx,
                source=live_statement_source(client),
                fills=fills_from_ib(client, ctx.trading_day.isoformat()),
                cash_flows=(),
                corporate_actions=live_corporate_actions(),
                genesis=args.genesis,
            )
        if address is None:
            return release(ctx, cause=args.cause, clock=clock)
        sdk = sdk_loader()
        client = connect(address, readonly=False, sdk_loader=lambda: sdk)
        clients.append(client)
        broker = IbBrokerControl(client, sdk)
        if job == KILL_SWITCH_JOB:
            outcome = fire(ctx, broker, mode=args.mode, cause=args.cause, clock=clock)
            record_outcome(ctx, outcome)
            return outcome.to_document()
        return drill_cycle(
            ctx,
            broker=broker,
            kind=args.kind,
            announced=args.announced,
            seal=seal_opening,
            clock=clock,
        )

    # `trader.reconcile` writes exactly one manifest per trading day, per its
    # own contract in `reconciliation_job.py` (no discriminator there), and
    # its transient-retry default matches the harness's own (a rerun on a
    # classified transient is safe: the job's outputs are content-addressed
    # and re-anchor identically). The kill switch and fire drill are each a
    # deliberate human action fired once, never retried underneath the caller.
    try:
        run_job(
            job,
            body,
            store=store,
            discriminator=(
                None if job == RECONCILE_JOB else (lambda ctx: f"{stamp}-{ctx.started:%H%M%S}")
            ),
            run_mode=args.run_mode,
            transient_retry=job == RECONCILE_JOB,
        )
    finally:
        for client in clients:
            client.disconnect()
    return 0
