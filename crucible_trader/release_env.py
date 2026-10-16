"""The requirements a box script installs beside the PINNED release wheel.

Every box script (`scripts/trader_pinned.sh`, `scripts/trader_reconcile.sh`,
`scripts/trader_paper_smoke.sh`) builds a fresh virtualenv from two sources:

* the release wheel `trader/release_pin` names, which is the crucible that runs;
* this repository's `uv.lock`, exported as requirements, for everything else.

The lock was resolved against THIS checkout's crucible git pin, not against the
release. The two are allowed to differ -- the pin moves when a PR here needs a
newer harness, the release moves only through the trader's own smoke -- so the
lock carries crucible's transitive pins at the checkout pin's versions. When the
release wheel pins one of those packages to a different exact version, the
install is unsatisfiable and the unit dies before its entry point runs.

Measured 2026-10-14/15 (`alpha-engine-config-I12280`): crucible-trader#28/#33
moved the checkout pin to crucible 38c68fc, whose `krepis[openai]==0.59.72` the
lock then carried, while the trader still ran release 2cf1a85, which requires
`krepis[openai]==0.59.71`. `uv pip install` refused (``your requirements are
unsatisfiable``), and every session, reconcile and shadow-books unit failed with
no run record written.

**The release wheel is the authority for what it requires.** This module drops
every locked line whose pinned version the wheel's own requirement for that
package does not admit, names each drop on stderr (the unit's journal), and
leaves the rest of the lock exactly as locked. A package the wheel does not
name, or whose locked version it admits, is untouched, so the environment stays
the lock wherever the lock and the release agree.

    crucible_trader.release_env.main([wheel, locked_requirements])

prints the reconciled requirements on stdout (called by the scripts; no
`__main__` block, the coverage floor has no exclusions). A wheel that is not a
wheel raises, which fails the unit before anything installs.
"""

from __future__ import annotations

import pathlib
import sys
import zipfile
from collections.abc import Iterable, Sequence
from email.parser import Parser
from typing import TextIO

from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name

USAGE = "usage: release_env <release wheel> <locked requirements.txt>"


def wheel_requirements(wheel: pathlib.Path) -> dict[NormalizedName, list[Requirement]]:
    """The wheel's own unconditional requirements, by normalized name.

    Read from its ``*.dist-info/METADATA`` ``Requires-Dist`` headers. A
    requirement behind a marker that is false here (an extra nobody installs,
    another platform) is not a requirement of this install and is skipped.
    """
    with zipfile.ZipFile(wheel) as archive:
        metadata = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        if len(metadata) != 1:
            raise ValueError(f"{wheel} carries {len(metadata)} dist-info METADATA files, not 1")
        headers = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"))
    wanted: dict[NormalizedName, list[Requirement]] = {}
    for line in headers.get_all("Requires-Dist") or []:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        wanted.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    return wanted


def reconcile(
    locked: Iterable[str], wanted: dict[NormalizedName, list[Requirement]]
) -> tuple[list[str], list[str]]:
    """Split the locked lines into (kept, dropped-with-reason).

    A line is dropped only when it pins a version that some requirement the
    wheel declares for that package does not admit. Blank lines and comments
    are not requirements and are left out of both lists.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for raw in locked:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        locked_req = Requirement(line)
        pins = [spec.version for spec in locked_req.specifier if spec.operator == "=="]
        refusing = [
            str(requirement)
            for requirement in wanted.get(canonicalize_name(locked_req.name), [])
            if pins and not requirement.specifier.contains(pins[0], prereleases=True)
        ]
        if refusing:
            dropped.append(f"the release wheel requires {', '.join(refusing)}; dropping {line}")
        else:
            kept.append(line)
    return kept, dropped


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    if len(args) != 2:
        print(USAGE, file=err)
        return 2
    wheel, locked = (pathlib.Path(arg) for arg in args)
    kept, dropped = reconcile(
        locked.read_text(encoding="utf-8").splitlines(), wheel_requirements(wheel)
    )
    for reason in dropped:
        print(f"release_env: {reason}", file=err)
    for line in kept:
        print(line, file=out)
    return 0
