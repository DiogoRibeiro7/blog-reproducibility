"""Check saved artifacts independently of the renderer and current checkout."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.manifest import ROOT
from scripts.verify_reproduction import main, verify_report


@pytest.fixture
def saved_report(tmp_path: Path) -> Path:
    """Create a portable report with two independently hashed outputs."""
    directory = tmp_path / "original"
    figures = []
    for name in ("statistics/first.png", "physics/second.png"):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = b"\x89PNG\r\n\x1a\n" + name.encode()
        path.write_bytes(content)
        figures.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    report = directory / "reproduction.json"
    report.write_text(json.dumps({"schema_version": 1, "figures": figures}), encoding="utf-8")
    return report


def test_verifies_relocated_outputs_from_another_working_directory(
    saved_report: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "download"
    shutil.copytree(saved_report.parent, copied)
    (copied / "unrelated.png").write_bytes(b"an older run")
    before = {path: path.read_bytes() for path in copied.rglob("*") if path.is_file()}
    monkeypatch.chdir(tmp_path)
    assert verify_report(Path("download/reproduction.json")) == 2
    assert {path: path.read_bytes() for path in copied.rglob("*") if path.is_file()} == before


def test_verification_command_succeeds(
    saved_report: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(saved_report)]) == 0
    assert "Verified checksums for 2 figures" in capsys.readouterr().out


def test_standalone_command_works_outside_the_checkout(saved_report: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/verify_reproduction.py"), str(saved_report)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified checksums for 2 figures" in result.stdout


@pytest.mark.parametrize("damage", ["changed", "missing", "directory"])
def test_reports_damaged_figure(
    saved_report: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    target = saved_report.parent / "physics/second.png"
    if damage == "changed":
        target.write_bytes(b"changed")
    else:
        target.unlink()
        if damage == "directory":
            target.mkdir()
    before = saved_report.read_bytes()
    assert main([str(saved_report)]) == 1
    error = capsys.readouterr().err
    assert "Verification failed:" in error
    assert "physics/second.png" in error
    assert saved_report.read_bytes() == before


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"schema_version": 2, "figures": []},
        {"schema_version": True, "figures": []},
        {"schema_version": 1.0, "figures": []},
        {"schema_version": 1},
        {"schema_version": 1, "figures": {}},
        {"schema_version": 1, "figures": []},
        {"schema_version": 1, "figures": [None]},
        {"schema_version": 1, "figures": [{}]},
        {"schema_version": 1, "figures": [{"path": "figure.png", "sha256": 42}]},
        {"schema_version": 1, "figures": [{"path": "figure.png", "sha256": "abc"}]},
        {"schema_version": 1, "figures": [{"path": "figure.png", "sha256": "z" * 64}]},
    ],
)
def test_rejects_invalid_report_structure(saved_report: Path, document: object) -> None:
    saved_report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_report(saved_report)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../outside.png",
        "/outside.png",
        "C:/outside.png",
        "C:outside.png",
        "folder\\figure.png",
        "folder//figure.png",
        "folder/../figure.png",
        "./figure.png",
        "folder/figure.png:stream",
        "\x00.png",
    ],
)
def test_rejects_invalid_paths_before_hashing(
    saved_report: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def unexpected_hash(*args: object, **kwargs: object) -> None:
        pytest.fail("All report paths must be validated before hashing figures")

    document = json.loads(saved_report.read_text(encoding="utf-8"))
    document["figures"][1]["path"] = name
    saved_report.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(hashlib, "file_digest", unexpected_hash)
    with pytest.raises(ValueError, match="relative POSIX path"):
        verify_report(saved_report)


def test_rejects_duplicate_figure_paths(saved_report: Path) -> None:
    document = json.loads(saved_report.read_text(encoding="utf-8"))
    document["figures"].append(document["figures"][0])
    saved_report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate figure path"):
        verify_report(saved_report)


def test_rejects_symlink_outside_report_directory(saved_report: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    target = saved_report.parent / "physics/second.png"
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")
    with pytest.raises(ValueError, match="outside the report directory"):
        verify_report(saved_report)


@pytest.mark.parametrize("content", [b"invalid json", b"\xff"])
def test_command_reports_unreadable_json(
    saved_report: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    saved_report.write_bytes(content)
    assert main([str(saved_report)]) == 1
    assert "Verification failed:" in capsys.readouterr().err


def test_command_reports_missing_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.json")]) == 1
    assert "missing.json" in capsys.readouterr().err
