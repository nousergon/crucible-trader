"""The decision writer Lambda is packaged --no-deps (nous-ergon-ops
``alpha-engine-back-office-decision-writer/requirements.txt``): crucible-trader,
crucible, krepis and tzdata only, with boto3 from the runtime. Importing its
handler must therefore load none of the trader's heavy dependencies, or the
deploy's import smoke check fails and the reviewer's form cannot ship.
"""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY = ("pydantic", "jsonschema", "numpy", "yaml", "pandas", "pyarrow")


def test_writer_import_closure_has_no_heavy_dependencies() -> None:
    code = (
        "import json, sys\n"
        "import crucible_trader.exception_decision_writer\n"
        f"print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True
    )
    loaded: list[str] = json.loads(out.stdout)
    assert loaded == []
