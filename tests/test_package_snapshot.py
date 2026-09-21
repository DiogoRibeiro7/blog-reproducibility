"""Exercise portable snapshots and rejection of stale or incomplete artifacts."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import package_snapshot
from scripts.manifest import ROOT
from scripts.package_snapshot import create_snapshot, main
from scripts.reproduce import input_hashes
from scripts.verify_reproduction import verify_figures


@pytest.fixture
def snapshot_root(tmp_path: Path) -> Path:
    """Prepare real archives and a recorded run from a minimal checkout."""
    files = {
        "LICENSE": b"MIT license\n",
        "README.md": b"Reproduce this checkout.\n",
        "pyproject.toml": b'[project]\nname = "blog-reproducibility"\nversion = "0.1.0"\n',
        "poetry.lock": b"# locked dependencies\n",
        "src/blog_reproducibility/__init__.py": b'__version__ = "0.1.0"\n',
        "scripts/verify_reproduction.py": (ROOT / "scripts/verify_reproduction.py").read_bytes(),
        "data/example.csv": b"value\n42\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    distribution = tmp_path / "dist"
    distribution.mkdir()
    with ZipFile(distribution / "example.whl", "w") as wheel:
        for name, content in files.items():
            if name.startswith("src/"):
                wheel.writestr(name.removeprefix("src/"), content)
        wheel.writestr(
            "blog_reproducibility-0.1.0.dist-info/METADATA",
            "Name: blog-reproducibility\nVersion: 0.1.0\n",
        )
    with tarfile.open(distribution / "example.tar.gz", "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo("example/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    figures = tmp_path / "build/figures"
    (figures / "statistics").mkdir(parents=True)
    content = b"\x89PNG\r\n\x1a\nrecorded output"
    (figures / "statistics/example.png").write_bytes(content)
    (figures / "unrelated.png").write_bytes(b"unrelated old run")
    (figures / "reproduction.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": input_hashes(tmp_path),
                "figures": [
                    {
                        "path": "statistics/example.png",
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_snapshot_is_complete_and_can_verify_figures_after_extraction(snapshot_root: Path) -> None:
    root = snapshot_root
    report = root / "build/figures/reproduction.json"
    snapshot = create_snapshot(root / "build/snapshots/release.zip", report, root=root)
    destination = root / "extracted"
    with ZipFile(snapshot) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "README.md",
            "LICENSE",
            "SHA256SUMS",
            "verify_reproduction.py",
            "distributions/example.whl",
            "distributions/example.tar.gz",
            "figures/reproduction.json",
            "figures/statistics/example.png",
        }
        inventory = {}
        for line in archive.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            inventory[name] = digest
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert set(inventory) == set(archive.namelist()) - {"SHA256SUMS"}
        assert archive.read("figures/reproduction.json") == report.read_bytes()
        assert archive.read("distributions/example.whl") == (root / "dist/example.whl").read_bytes()
        archive.extractall(destination)
    result = subprocess.run(
        [sys.executable, "-I", "verify_reproduction.py", "figures/reproduction.json"],
        cwd=destination,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified checksums for 1 figures" in result.stdout


@pytest.mark.parametrize("change", ["changed", "missing", "extra"])
def test_report_must_describe_current_inputs(snapshot_root: Path, change: str) -> None:
    root = snapshot_root
    report = root / "build/figures/reproduction.json"
    document = json.loads(report.read_text(encoding="utf-8"))
    if change == "changed":
        document["inputs"]["data/example.csv"] = "0" * 64
    elif change == "missing":
        del document["inputs"]["data/example.csv"]
    else:
        document["inputs"]["data/removed.csv"] = "0" * 64
    report.write_text(json.dumps(document), encoding="utf-8")
    output = root / "snapshot/release.zip"
    with pytest.raises(ValueError, match="Report inputs do not match"):
        create_snapshot(output, report, root=root)
    assert not output.parent.exists()


def test_stale_distributions_are_rejected(snapshot_root: Path) -> None:
    (snapshot_root / "data/example.csv").write_bytes(b"value\n99\n")
    output = snapshot_root / "release.zip"
    with pytest.raises(RuntimeError, match="Source archive differs"):
        create_snapshot(
            output, snapshot_root / "build/figures/reproduction.json", root=snapshot_root
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "Name: another-project\nVersion: 0.1.0\n",
        "Name: blog-reproducibility\nVersion: 9.0.0\n",
    ],
)
def test_wheel_metadata_must_match_project(snapshot_root: Path, metadata: str | None) -> None:
    wheel = snapshot_root / "dist/example.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "blog_reproducibility/__init__.py",
            (snapshot_root / "src/blog_reproducibility/__init__.py").read_bytes(),
        )
        if metadata is not None:
            archive.writestr("example.dist-info/METADATA", metadata)
    with pytest.raises(ValueError, match="METADATA|name or version"):
        create_snapshot(
            snapshot_root / "release.zip",
            snapshot_root / "build/figures/reproduction.json",
            root=snapshot_root,
        )
    assert not (snapshot_root / "release.zip").exists()


@pytest.mark.parametrize("distribution", ["missing", "ambiguous"])
def test_distribution_selection_must_be_unambiguous(snapshot_root: Path, distribution: str) -> None:
    if distribution == "missing":
        (snapshot_root / "dist/example.whl").unlink()
    else:
        (snapshot_root / "dist/another.whl").write_bytes(b"ambiguous")
    with pytest.raises(RuntimeError, match="Expected one wheel and one source archive"):
        create_snapshot(
            snapshot_root / "release.zip",
            snapshot_root / "build/figures/reproduction.json",
            root=snapshot_root,
        )


def test_existing_snapshot_is_never_overwritten(snapshot_root: Path) -> None:
    output = snapshot_root / "release.zip"
    output.write_bytes(b"previous snapshot")
    with pytest.raises(FileExistsError, match="already exists"):
        create_snapshot(
            output, snapshot_root / "build/figures/reproduction.json", root=snapshot_root
        )
    assert output.read_bytes() == b"previous snapshot"


def test_snapshot_requires_zip_extension(snapshot_root: Path) -> None:
    with pytest.raises(ValueError, match=".zip extension"):
        create_snapshot(
            snapshot_root / "release.txt",
            snapshot_root / "build/figures/reproduction.json",
            root=snapshot_root,
        )


def test_changed_figures_are_rejected_during_packaging(
    snapshot_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def verify_then_change(report: object, directory: Path) -> int:
        count = verify_figures(report, directory)
        (directory / "statistics/example.png").write_bytes(b"changed after verification")
        return count

    monkeypatch.setattr(package_snapshot, "verify_figures", verify_then_change)
    with pytest.raises(ValueError, match="Figure changed while packaging"):
        create_snapshot(
            snapshot_root / "release.zip",
            snapshot_root / "build/figures/reproduction.json",
            root=snapshot_root,
        )
    assert not (snapshot_root / "release.zip").exists()


def test_snapshot_command_reports_success_and_failure(
    snapshot_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(package_snapshot, "ROOT", snapshot_root)
    assert main([]) == 0
    assert "Created verified snapshot" in capsys.readouterr().out
    assert main([]) == 1
    assert "already exists" in capsys.readouterr().err


def test_figure_cannot_replace_the_bundled_report(snapshot_root: Path) -> None:
    directory = snapshot_root / "build/figures"
    original = directory / "reproduction.json"
    document = json.loads(original.read_text(encoding="utf-8"))
    content = b"a figure with a conflicting name"
    original.write_bytes(content)
    document["figures"] = [
        {"path": "reproduction.json", "sha256": hashlib.sha256(content).hexdigest()}
    ]
    report = directory / "run.json"
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with a snapshot file"):
        create_snapshot(snapshot_root / "release.zip", report, root=snapshot_root)
    assert not (snapshot_root / "release.zip").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows disallows line breaks in filenames")
def test_snapshot_paths_must_fit_checksum_inventory(snapshot_root: Path) -> None:
    directory = snapshot_root / "build/figures"
    report = directory / "reproduction.json"
    document = json.loads(report.read_text(encoding="utf-8"))
    content = b"figure with a line break in its name"
    name = "line\nbreak.png"
    (directory / name).write_bytes(content)
    document["figures"] = [{"path": name, "sha256": hashlib.sha256(content).hexdigest()}]
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="line breaks or backslashes"):
        create_snapshot(snapshot_root / "release.zip", report, root=snapshot_root)
    assert not (snapshot_root / "release.zip").exists()
