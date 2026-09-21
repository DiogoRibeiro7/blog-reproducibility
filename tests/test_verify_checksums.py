"""Verify downloaded inventories without installing the scientific dependencies."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import verify_checksums
from scripts.verify_checksums import main, verify_inventory


@pytest.fixture
def inventory(tmp_path: Path) -> Path:
    directory = tmp_path / "snapshot"
    records = []
    for name, content in {
        "distributions/example.whl": b"wheel contents\x00\xff",
        "figures/a figure.png": b"recorded figure",
        "README.md": b"Read this snapshot.\n",
    }.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
    path = directory / "SHA256SUMS"
    path.write_text("".join(records), encoding="utf-8", newline="\n")
    return path


def test_verifies_relocated_inventory_without_changing_files(
    inventory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "download"
    shutil.copytree(inventory.parent, destination)
    (destination / "unlisted.txt").write_bytes(b"unrelated download")
    before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    monkeypatch.chdir(tmp_path)
    assert verify_inventory(Path("download/SHA256SUMS")) == 3
    assert {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
def test_accepts_portable_line_endings_and_final_line_without_newline(
    inventory: Path, ending: str
) -> None:
    lines = inventory.read_text(encoding="utf-8").splitlines()
    separator = ending or "\n"
    inventory.write_bytes((separator.join(lines) + ending).encode("utf-8"))
    assert verify_inventory(inventory) == 3


def test_preserves_spaces_and_unicode_in_paths(inventory: Path) -> None:
    name = "figures/ café\u2028result .png"
    content = b"another figure"
    (inventory.parent / name).write_bytes(content)
    inventory.write_text(f"{hashlib.sha256(content).hexdigest()}  {name}\n", encoding="utf-8")
    assert verify_inventory(inventory) == 1


@pytest.mark.parametrize("damage", ["changed", "missing", "directory"])
def test_reports_damaged_file(
    inventory: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    target = inventory.parent / "distributions/example.whl"
    if damage == "changed":
        target.write_bytes(b"changed download")
    else:
        target.unlink()
        if damage == "directory":
            target.mkdir()
    before = inventory.read_bytes()
    assert main([str(inventory)]) == 1
    assert "distributions/example.whl" in capsys.readouterr().err
    assert inventory.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        "",
        "\n",
        "# SHA256SUMS\n",
        "abc  README.md\n",
        "g" * 64 + "  README.md\n",
        "0" * 64 + " README.md\n",
        "0" * 64 + "  \n",
        "0" * 64 + "  README.md\n\n",
    ],
)
def test_rejects_malformed_inventory(inventory: Path, content: str) -> None:
    inventory.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="inventory|record"):
        verify_inventory(inventory)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.txt",
        "/outside.txt",
        "C:/outside.txt",
        "C:outside.txt",
        "folder\\file.txt",
        "folder//file.txt",
        "folder/../file.txt",
        "./file.txt",
        "folder/file.txt:stream",
        "\x00.txt",
    ],
)
def test_rejects_invalid_paths_before_hashing(
    inventory: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def unexpected_hash(*args: object, **kwargs: object) -> None:
        pytest.fail("Validate every inventory path before hashing any file")

    with inventory.open("a", encoding="utf-8") as stream:
        stream.write(f"{'0' * 64}  {name}\n")
    monkeypatch.setattr(hashlib, "file_digest", unexpected_hash)
    with pytest.raises(ValueError, match="relative POSIX path"):
        verify_inventory(inventory)


def test_rejects_duplicate_paths(inventory: Path) -> None:
    contents = inventory.read_text(encoding="utf-8")
    inventory.write_text(contents + contents, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate checksum path"):
        verify_inventory(inventory)


@pytest.mark.parametrize("outside", [False, True])
def test_rejects_symlink_escape_and_duplicate_alias(
    inventory: Path, tmp_path: Path, outside: bool
) -> None:
    target = tmp_path / "outside.txt" if outside else inventory.parent / "README.md"
    if outside:
        target.write_bytes(b"outside snapshot")
    alias = inventory.parent / "alias.txt"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")
    with inventory.open("a", encoding="utf-8") as stream:
        stream.write(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  alias.txt\n")
    message = "outside the inventory directory" if outside else "Duplicate checksum path"
    with pytest.raises(ValueError, match=message):
        verify_inventory(inventory)


@pytest.mark.parametrize("failure", ["missing", "encoding"])
def test_cli_reports_unreadable_inventory(
    inventory: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    if failure == "missing":
        inventory.unlink()
    else:
        inventory.write_bytes(b"\xff")
    assert main([str(inventory)]) == 1
    assert "Checksum verification failed:" in capsys.readouterr().err


def test_cli_reports_success(inventory: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(inventory)]) == 0
    assert "Verified checksums for 3 files" in capsys.readouterr().out


def test_standalone_command_needs_only_standard_library(inventory: Path, tmp_path: Path) -> None:
    script = tmp_path / "verify_checksums.py"
    script.write_bytes(Path(verify_checksums.__file__).read_bytes())
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), str(inventory)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified checksums for 3 files" in result.stdout
