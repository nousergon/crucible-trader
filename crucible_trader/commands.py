"""The operator commands: the kill switch (one command) and the fire drill.

`alpha-engine-config-I10650`. Each command is one `crucible.runner.run_job` run,
so every fire, release and drill files a manifest whatever happens:

    kill-switch fire --mode flatten|freeze --cause TEXT     job trader.kill_switch
    kill-switch release --cause TEXT                        job trader.kill_switch
    fire-drill run --kind KIND (--announced|--unannounced)  job trader.fire_drill
    fire-drill status --from YYYY-MM-DD --to YYYY-MM-DD     read-only; exit 1 unless done

Invoked on the box as ``python -c 'import sys; from crucible_trader.commands
import main; sys.exit(main(sys.argv[1:]))' <command> ...`` (one line).
(no `__main__` block: this package's coverage floor has no exclusions).

`fire` and `run` connect to IB Gateway PAPER with an order-capable session (the
switch cancels and closes); `release` and `status` never connect.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from crucible.runner import RunContext, run_job

from crucible_trader.broker_session import GatewayAddress, connect, load_sdk
from crucible_trader.fire_drill import (
    DRILL_KINDS,
    FIRE_DRILL_JOB,
    drill_cycle,
    evaluate_drills,
    read_drills,
)
from crucible_trader.kill_switch import (
    KILL_SWITCH_JOB,
    IbBrokerControl,
    fire,
    record_outcome,
    release,
    utcnow,
)
from crucible_trader.paper_smoke import open_configured_store
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
    status_p = drill.add_parser("status")
    status_p.add_argument("--from", dest="window_start", required=True)
    status_p.add_argument("--to", dest="window_end", required=True)

    for sub in (fire_p, release_p, run_p):
        sub.add_argument("--run-mode", choices=("live", "replay"), default="live")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    sdk_loader: Callable[[], Any] = load_sdk,
    clock: Callable[[], dt.datetime] = utcnow,
    printer: Callable[[str], None] = print,
) -> int:
    args = _parser().parse_args(argv)
    source = None if environ is None else dict(environ)
    store = open_configured_store(Settings.from_env(source))

    if args.command == "fire-drill" and args.action == "status":
        reading = evaluate_drills(
            read_drills(store), window_start=args.window_start, window_end=args.window_end
        )
        printer(f"{'DONE' if reading.done else 'NOT DONE'}: {reading.reason}")
        return 0 if reading.done else 1

    if args.command == "kill-switch" and args.action == "release":
        job, stamp, connects = KILL_SWITCH_JOB, "release", False
    elif args.command == "kill-switch":
        job, stamp, connects = KILL_SWITCH_JOB, f"fire-{args.mode}", True
    else:
        job, stamp, connects = FIRE_DRILL_JOB, args.kind, True

    address = GatewayAddress.from_env(source) if connects else None
    clients: list[Any] = []

    def body(ctx: RunContext) -> Any:
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
            ctx, broker=broker, kind=args.kind, announced=args.announced, clock=clock
        )

    try:
        run_job(
            job,
            body,
            store=store,
            discriminator=lambda ctx: f"{stamp}-{ctx.started:%H%M%S}",
            run_mode=args.run_mode,
            transient_retry=False,
        )
    finally:
        for client in clients:
            client.disconnect()
    return 0
