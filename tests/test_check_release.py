"""Check release readiness without creating tags or contacting GitHub."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_release


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\n- Future work.\n\n"
        "## 0.1.0 - 2026-09-21\n\n### Added\n\n- Verified snapshots.\n\n"
        "## 0.0.1 - 2026-08-01\n\n- Earlier work.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_extracts_only_matching_release(release_root: Path) -> None:
    assert check_release.release_notes("v0.1.0", root=release_root) == (
        "### Added\n\n- Verified snapshots.\n"
    )


@pytest.mark.parametrize("tag", ["0.1.0", "v0.2.0", "v0.1.0-extra", "v0.1.0\n", ""])
def test_tag_must_match_package_version(release_root: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="does not match project.version"):
        check_release.release_notes(tag, root=release_root)


@pytest.mark.parametrize(
    "metadata",
    [
        "",
        'project = "invalid"',
        "[project]",
        '[project]\nversion = ""',
        '[project]\nversion = " 0.1.0"',
        "[project]\nversion = 1",
    ],
)
def test_version_must_be_declared_as_a_string(release_root: Path, metadata: str) -> None:
    (release_root / "pyproject.toml").write_text(metadata, encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty project.version string"):
        check_release.release_notes("v0.1.0", root=release_root)


@pytest.mark.parametrize(
    ("changelog", "message"),
    [
        ("## Unreleased\n\n- Work.\n", "exactly one"),
        ("## 0.1.0 - 2026-09-21\n- Work.\n## 0.1.0 - 2026-09-22\n- More.\n", "exactly one"),
        ("## 0.1.0\n\n- Work.\n", "Release heading must be"),
        ("## 0.1.0 - 20260921\n\n- Work.\n", "Release heading must be"),
        ("## 0.1.0 - 2026-9-21\n\n- Work.\n", "Release heading must be"),
        ("## 0.1.0 - 2026-02-29\n\n- Work.\n", "Invalid release date"),
        ("## 0.1.0 - 2026-13-01\n\n- Work.\n", "Invalid release date"),
        ("## 0.1.0 - 2026-09-21\n", "must contain release notes"),
        ("## 0.1.0 - 2026-09-21\n\n### Added\n\n### Fixed\n", "must contain release notes"),
    ],
)
def test_rejects_missing_or_invalid_release_sections(
    release_root: Path, changelog: str, message: str
) -> None:
    (release_root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        check_release.release_notes("v0.1.0", root=release_root)


def test_preserves_markdown_and_accepts_valid_leap_day(release_root: Path) -> None:
    notes = "### Added\n\n- Literal `$(example)` and **café**.\n\n```sh\nexample --help\n```\n"
    (release_root / "CHANGELOG.md").write_text(
        "## 0.1.0 - 2024-02-29\n\n" + notes, encoding="utf-8"
    )
    assert check_release.release_notes("v0.1.0", root=release_root) == notes


def test_cli_can_write_notes(
    release_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_release, "ROOT", release_root)
    output = release_root / "build/release-notes.md"
    assert check_release.main(["--tag", "v0.1.0", "--notes-output", str(output)]) == 0
    assert output.read_bytes() == b"### Added\n\n- Verified snapshots.\n"
    assert "Validated release metadata" in capsys.readouterr().out


def test_cli_can_check_without_writing(release_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_release, "ROOT", release_root)
    before = {path.name: path.read_bytes() for path in release_root.iterdir()}
    assert check_release.main(["--tag", "v0.1.0"]) == 0
    assert {path.name: path.read_bytes() for path in release_root.iterdir()} == before


def test_failed_validation_preserves_previous_notes(
    release_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_release, "ROOT", release_root)
    output = release_root / "notes.md"
    output.write_bytes(b"Previous notes\n")
    assert check_release.main(["--tag", "v9.0.0", "--notes-output", str(output)]) == 1
    assert output.read_bytes() == b"Previous notes\n"
    assert "Release validation failed" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["missing", "malformed", "output_directory"])
def test_cli_reports_file_errors(
    release_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    monkeypatch.setattr(check_release, "ROOT", release_root)
    if failure == "missing":
        (release_root / "CHANGELOG.md").unlink()
    elif failure == "malformed":
        (release_root / "pyproject.toml").write_text("[invalid", encoding="utf-8")
    assert check_release.main(["--tag", "v0.1.0", "--notes-output", str(release_root)]) == 1
    assert "Release validation failed" in capsys.readouterr().err


def test_standalone_command_needs_only_standard_library(release_root: Path) -> None:
    scripts = release_root / "scripts"
    scripts.mkdir()
    script = scripts / "check_release.py"
    script.write_bytes(Path(check_release.__file__).read_bytes())
    output = release_root / "notes.md"
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--tag", "v0.1.0", "--notes-output", str(output)],
        cwd=release_root.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b"### Added\n\n- Verified snapshots.\n"
