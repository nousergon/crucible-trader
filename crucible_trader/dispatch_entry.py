"""The entry point a DISPATCHED box runs for a `trader.*` job.

`alpha-engine-config-I13159`. The v2 Dispatcher boots a spot box that runs
`crucible <job>`, and `crucible.cli` has no `trader.*` subcommand: the trader's
jobs (`crucible.models.TRADER_JOB_VALUES`) are run by THIS package, never by
the harness. So a dispatched `trader.*` box installs this package's
content-hashed wheel on top of the release venv (the CI `wheel` job builds it)
and runs this console script instead::

    crucible-trader-dispatch trader.session [--run-mode live|replay]

It is a lookup and nothing else. The job name maps through
:data:`JOB_MODULES` to one `crucible_trader.<module>.main`, called with the
remaining arguments exactly as `scripts/trader_pinned.sh <module>` calls it on
the executor box. A job not in the table is a malformed dispatch.

**Staging only, for now.** A dispatched box is a spot instance with no IB
Gateway beside it, and prod's trader runs from the executor box through
`scripts/trader_pinned.sh`, which this module does not touch. So the job runs
only when ``NE_ENV`` is one of :data:`DISPATCH_ENVIRONMENTS`; the Dispatcher
exports ``NE_ENV=staging`` on staging boxes and nothing in prod
(alpha-engine-config-I12509), so a prod dispatch refuses here before any
trader code is imported. Under staging `daily_session.main` itself never loads
the broker SDK (`alpha-engine-config-I12507`).

Exit 2 for both refusals: that is the code the Dispatcher's bootstrap reads as
"malformed dispatch, no run manifest written" (`alpha-engine-config-I10134`),
which is the truth for an unknown job or a box in the wrong environment.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

#: Dispatch job name -> `crucible_trader.<module>`, whose ``main(argv)`` runs it.
#: Every value names a module with a box entry point ``main(argv) -> int``.
JOB_MODULES: Mapping[str, str] = {
    "trader.session": "daily_session",
}

#: The fleet environment selector (`nousergon_lib.env.ENV_VAR`).
ENVIRONMENT_VAR = "NE_ENV"

#: The environments a dispatched trader box may run in. Prod is absent on
#: purpose: see the module docstring.
DISPATCH_ENVIRONMENTS: frozenset[str] = frozenset({"staging"})

USAGE_EXIT = 2


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    error: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
) -> int:
    """Run the `crucible_trader` module :data:`JOB_MODULES` names for ``argv[0]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    job = args[0] if args else ""
    if job not in JOB_MODULES:
        error(
            f"crucible-trader-dispatch: unknown trader job {job!r}; "
            f"this box runs only {sorted(JOB_MODULES)}"
        )
        return USAGE_EXIT
    environment = env.get(ENVIRONMENT_VAR, "")
    if environment not in DISPATCH_ENVIRONMENTS:
        error(
            f"crucible-trader-dispatch: {job} refused: {ENVIRONMENT_VAR}={environment!r}, "
            f"a dispatched trader box runs only under {sorted(DISPATCH_ENVIRONMENTS)}; "
            "prod's trader runs from the executor box (scripts/trader_pinned.sh)"
        )
        return USAGE_EXIT
    module = importer(f"crucible_trader.{JOB_MODULES[job]}")
    entry: Callable[[list[str]], int] = module.main
    return entry(args[1:])
