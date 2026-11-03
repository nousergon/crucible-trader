"""The dispatched-box entry point: a declared lookup, staging only (alpha-engine-config-I13159)."""

from __future__ import annotations

import importlib
import inspect
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from crucible.models import TRADER_JOB_VALUES

from crucible_trader import dispatch_entry
from crucible_trader.dispatch_entry import DISPATCH_ENVIRONMENTS, JOB_MODULES, USAGE_EXIT, main

STAGING = {"NE_ENV": "staging"}


class _Recorder:
    """Stands in for `crucible_trader.<module>`; records what `main` was handed."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.imported: list[str] = []
        self.code = code

    def importer(self, name: str) -> ModuleType:
        self.imported.append(name)
        module = ModuleType(name)

        def module_main(argv: list[str]) -> int:
            self.calls.append(argv)
            return self.code

        module.main = module_main
        return module


def _never_imported(name: str) -> ModuleType:
    raise AssertionError(f"a refused dispatch must import no trader code (asked for {name})")


def test_session_runs_daily_session_with_the_remaining_arguments():
    rec = _Recorder()
    code = main(["trader.session", "--run-mode", "replay"], environ=STAGING, importer=rec.importer)
    assert code == 0
    assert rec.imported == ["crucible_trader.daily_session"]
    assert rec.calls == [["--run-mode", "replay"]]


def test_the_module_exit_code_is_the_box_exit_code():
    rec = _Recorder(code=1)
    assert main(["trader.session"], environ=STAGING, importer=rec.importer) == 1
    assert rec.calls == [[]]


@pytest.mark.parametrize("argv", [[], ["trader.reconcile"], ["daily_session"], ["data.daily"]])
def test_an_undeclared_job_is_a_malformed_dispatch(argv):
    lines: list[str] = []
    code = main(argv, environ=STAGING, importer=_never_imported, error=lines.append)
    assert code == USAGE_EXIT
    assert len(lines) == 1 and "unknown trader job" in lines[0]


@pytest.mark.parametrize("environ", [{}, {"NE_ENV": "prod"}, {"NE_ENV": ""}, {"NE_ENV": "Staging"}])
def test_a_box_outside_staging_refuses_before_importing_the_trader(environ):
    lines: list[str] = []
    code = main(["trader.session"], environ=environ, importer=_never_imported, error=lines.append)
    assert code == USAGE_EXIT
    assert len(lines) == 1 and "trader_pinned.sh" in lines[0]


def test_defaults_read_process_argv_environment_and_stderr(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["crucible-trader-dispatch", "trader.session"])
    monkeypatch.setenv("NE_ENV", "prod")
    assert main(importer=_never_imported) == USAGE_EXIT
    assert "NE_ENV='prod'" in capsys.readouterr().err

    rec = _Recorder()
    monkeypatch.setenv("NE_ENV", "staging")
    assert main(importer=rec.importer) == 0
    assert rec.calls == [[]]


def test_prod_is_not_a_dispatch_environment():
    assert DISPATCH_ENVIRONMENTS == frozenset({"staging"})


def test_every_declared_job_is_a_trader_job_with_a_real_box_entry_point():
    assert set(JOB_MODULES) <= set(TRADER_JOB_VALUES)
    for module_name in JOB_MODULES.values():
        module = importlib.import_module(f"crucible_trader.{module_name}")
        params = list(inspect.signature(module.main).parameters.values())
        assert params[0].name == "argv"
        # Everything after argv is keyword-only with a default: `main(argv)` is a full call.
        assert all(p.kind is p.KEYWORD_ONLY and p.default is not p.empty for p in params[1:])


def test_the_console_script_names_this_entry_point():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert scripts == {"crucible-trader-dispatch": "crucible_trader.dispatch_entry:main"}
    assert dispatch_entry.main is main
