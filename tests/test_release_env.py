"""The box's venv installs the release wheel's own pins over this repository's
lock wherever the two disagree (`alpha-engine-config-I12280`).

The regression case is the measured one: on 2026-10-14/15 the lock carried
crucible 38c68fc's `krepis==0.59.72` while the trader ran release 2cf1a85, whose
wheel requires `krepis[openai]==0.59.71`. `uv pip install` refused the pair and
every trader unit died before its entry point, writing no run record.
"""

from __future__ import annotations

import io
import pathlib
import zipfile

import pytest
from packaging.requirements import Requirement

from crucible_trader.release_env import USAGE, main, reconcile, wheel_requirements

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"

#: `Requires-Dist` of release 2cf1a85's wheel, the lines that matter here.
RELEASE_2CF1A85 = [
    "boto3>=1.43.105",
    "krepis[openai]==0.59.71",
    "nousergon-lib[gates,github-app]==0.124.165",
    "pydantic>=2.9",
    "nousergon-lib[arcticdb]==0.124.165; extra == 'arcticdb'",
]

#: The lock as exported at crucible-trader bb33804 (crucible pin 38c68fc).
LOCKED_AT_38C68FC = """\
boto3==1.43.113
colorama==0.4.6 ; sys_platform == 'win32'
ib-async==2.1.0
krepis==0.59.72
nousergon-lib==0.124.165
pydantic==2.14.1
"""


def _wheel(tmp_path: pathlib.Path, requires: list[str], *, metadata_files: int = 1) -> pathlib.Path:
    path = tmp_path / "crucible-0.1.0-py3-none-any.whl"
    body = "Metadata-Version: 2.4\nName: crucible\nVersion: 0.1.0\n" + "".join(
        f"Requires-Dist: {line}\n" for line in requires
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("crucible/__init__.py", "")
        for index in range(metadata_files):
            archive.writestr(f"crucible{index or ''}-0.1.0.dist-info/METADATA", body)
        # A vendored METADATA deeper in the tree is not the wheel's own.
        archive.writestr("crucible/_vendor/x-1.0.dist-info/METADATA", "Name: x\n")
    return path


class TestTheWheelsOwnRequirements:
    def test_unconditional_requirements_are_read_by_normalized_name(self, tmp_path) -> None:
        wanted = wheel_requirements(_wheel(tmp_path, RELEASE_2CF1A85))

        assert sorted(wanted) == ["boto3", "krepis", "nousergon-lib", "pydantic"]
        assert [str(r) for r in wanted["krepis"]] == ["krepis[openai]==0.59.71"]

    def test_an_extra_nobody_installs_is_not_a_requirement(self, tmp_path) -> None:
        wanted = wheel_requirements(_wheel(tmp_path, RELEASE_2CF1A85))

        assert [str(r) for r in wanted["nousergon-lib"]] == [
            "nousergon-lib[gates,github-app]==0.124.165"
        ]

    def test_a_wheel_with_no_requirements_requires_nothing(self, tmp_path) -> None:
        assert wheel_requirements(_wheel(tmp_path, [])) == {}

    @pytest.mark.parametrize("count", [0, 2])
    def test_a_wheel_without_exactly_one_metadata_is_refused(self, tmp_path, count) -> None:
        with pytest.raises(ValueError, match=f"carries {count} dist-info METADATA"):
            wheel_requirements(_wheel(tmp_path, RELEASE_2CF1A85, metadata_files=count))


class TestTheLockYieldsOnlyWhereTheReleaseDisagrees:
    def test_the_measured_krepis_conflict_is_dropped_and_named(self, tmp_path) -> None:
        wanted = wheel_requirements(_wheel(tmp_path, RELEASE_2CF1A85))

        kept, dropped = reconcile(LOCKED_AT_38C68FC.splitlines(), wanted)

        assert "krepis==0.59.72" not in kept
        assert dropped == [
            "the release wheel requires krepis[openai]==0.59.71; dropping krepis==0.59.72"
        ]

    def test_every_line_the_release_admits_stays_exactly_as_locked(self, tmp_path) -> None:
        wanted = wheel_requirements(_wheel(tmp_path, RELEASE_2CF1A85))

        kept, _ = reconcile(LOCKED_AT_38C68FC.splitlines(), wanted)

        assert kept == [
            "boto3==1.43.113",
            "colorama==0.4.6 ; sys_platform == 'win32'",
            "ib-async==2.1.0",
            "nousergon-lib==0.124.165",
            "pydantic==2.14.1",
        ]

    def test_a_floor_the_lock_is_below_drops_the_lock_line(self) -> None:
        """A release NEWER than the lock: its raised floor wins too."""
        wanted = {"boto3": [_req("boto3>=1.43.120")]}

        kept, dropped = reconcile(["boto3==1.43.113"], wanted)

        assert kept == [] and dropped == [
            "the release wheel requires boto3>=1.43.120; dropping boto3==1.43.113"
        ]

    def test_names_are_compared_normalized(self) -> None:
        wanted = {"nousergon-lib": [_req("nousergon_lib==0.124.166")]}

        kept, dropped = reconcile(["Nousergon.Lib==0.124.165"], wanted)

        assert kept == [] and len(dropped) == 1

    def test_a_line_with_no_exact_pin_is_kept(self) -> None:
        kept, dropped = reconcile(["krepis>=0.59"], {"krepis": [_req("krepis==0.59.71")]})

        assert kept == ["krepis>=0.59"] and dropped == []

    def test_blank_lines_and_comments_are_not_requirements(self) -> None:
        kept, dropped = reconcile(["", "   ", "# via crucible", "ib-async==2.1.0"], {})

        assert kept == ["ib-async==2.1.0"] and dropped == []


class TestTheScriptsEntryPoint:
    def test_it_prints_the_reconciled_lock_and_journals_each_drop(self, tmp_path) -> None:
        locked = tmp_path / "locked.txt"
        locked.write_text(LOCKED_AT_38C68FC, encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()

        code = main([str(_wheel(tmp_path, RELEASE_2CF1A85)), str(locked)], stdout=out, stderr=err)

        assert code == 0
        assert "krepis" not in out.getvalue() and "ib-async==2.1.0\n" in out.getvalue()
        assert err.getvalue() == (
            "release_env: the release wheel requires krepis[openai]==0.59.71; "
            "dropping krepis==0.59.72\n"
        )

    def test_it_defaults_to_the_process_streams(self, tmp_path, capsys, monkeypatch) -> None:
        locked = tmp_path / "locked.txt"
        locked.write_text("ib-async==2.1.0\n", encoding="utf-8")
        wheel = _wheel(tmp_path, [])
        monkeypatch.setattr("sys.argv", ["release_env", str(wheel), str(locked)])

        assert main() == 0
        assert capsys.readouterr().out == "ib-async==2.1.0\n"

    @pytest.mark.parametrize("argv", [[], ["one"], ["one", "two", "three"]])
    def test_anything_but_two_paths_is_a_usage_error(self, argv) -> None:
        err = io.StringIO()

        assert main(argv, stdout=io.StringIO(), stderr=err) == 2
        assert err.getvalue() == USAGE + "\n"


@pytest.mark.parametrize(
    "name", ["trader_pinned.sh", "trader_reconcile.sh", "trader_paper_smoke.sh"]
)
def test_every_wheel_script_installs_the_reconciled_lock(name: str) -> None:
    """The raw export never reaches `uv pip install`; the reconciled one does."""
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    export = text.find("""grep -v '^ *#' > "$work/locked.txt\"""")
    reconciled = text.find(
        "from crucible_trader.release_env import main; sys.exit(main(sys.argv[1:]))' \\\n"
        '  "$work/$wheel" "$work/locked.txt" > "$work/requirements.txt"'
    )
    install = text.find('-r "$work/requirements.txt" "$work/$wheel"')

    assert -1 < export < reconciled < install, name
    assert text.count("requirements.txt") == 2, name


def _req(text: str) -> Requirement:
    return Requirement(text)
