"""Bundle verified distributions and saved figures into a portable snapshot."""

import argparse
import hashlib
import io
import json
import sys
import tarfile
import tomllib
from email import message_from_bytes
from pathlib import Path
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from scripts.check_distribution import check_contents, find_distributions
from scripts.manifest import ROOT
from scripts.reproduce import input_hashes
from scripts.verify_reproduction import verify_figures

SNAPSHOT_README = """# Reproducibility snapshot

This archive contains the wheel and source distribution, the figures recorded by
the included reproduction report, and standalone checksum and figure verifiers.

After extracting the snapshot, verify all listed files with Python 3.11 or newer
on Windows, macOS, or Linux:

    python verify_checksums.py SHA256SUMS

This checks the distributions, report, figures, license, README, and verifiers
against the included inventory. It leaves files unchanged and ignores unlisted
files. The inventory itself is not checksummed.

To check just the figures against their reproduction report:

    python verify_reproduction.py figures/reproduction.json

SHA256SUMS lists every other file in this snapshot. Check all files on Linux with
`sha256sum -c SHA256SUMS`, or on macOS with `shasum -a 256 -c SHA256SUMS`.

To reproduce the computations, extract the source archive in distributions/ and
follow its README.md. It includes the dependency lock, article manifest, scripts,
tests, and data. The wheel contains the importable scientific package.

figures/reproduction.json records the run's source hashes, Git state, environment,
and figure checksums. Keep this report with the figures when sharing the snapshot.
"""


def create_snapshot(output: Path, report_path: Path, *, root: Path = ROOT) -> Path:
    """Package a checked run without rebuilding artifacts or replacing snapshots."""
    output = output.resolve()
    if output.suffix != ".zip":
        raise ValueError("Snapshot output must have a .zip extension")
    if output.exists():
        raise FileExistsError(f"Snapshot already exists; choose another output path: {output}")

    wheel, source = find_distributions(root / "dist")
    check_contents(wheel, source, root=root)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    with ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel must contain exactly one distribution METADATA file")
        metadata = message_from_bytes(archive.read(metadata_paths[0]))
    if metadata["Name"] != project["name"] or metadata["Version"] != project["version"]:
        raise ValueError("Wheel name or version does not match pyproject.toml; rebuild dist/")

    report_path = report_path.resolve()
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    verify_figures(report, report_path.parent)
    if report.get("inputs") != input_hashes(root):
        raise ValueError("Report inputs do not match the current checkout; rerun reproduction")

    payloads = {
        "README.md": SNAPSHOT_README.encode("utf-8"),
        "LICENSE": (root / "LICENSE").read_bytes(),
        "verify_checksums.py": (root / "scripts/verify_checksums.py").read_bytes(),
        "verify_reproduction.py": (root / "scripts/verify_reproduction.py").read_bytes(),
        "figures/reproduction.json": report_bytes,
        f"distributions/{wheel.name}": wheel.read_bytes(),
        f"distributions/{source.name}": source.read_bytes(),
    }
    for figure in report["figures"]:
        name = figure["path"]
        archive_name = f"figures/{name}"
        if archive_name in payloads:
            raise ValueError(f"Figure conflicts with a snapshot file: {name}")
        content = (report_path.parent / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != figure["sha256"]:
            raise ValueError(f"Figure changed while packaging: {name}")
        payloads[archive_name] = content
    if any(any(character in name for character in "\r\n\\") for name in payloads):
        raise ValueError("Snapshot paths cannot contain line breaks or backslashes")
    payloads["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(payloads.items())
    ).encode("utf-8")

    # Finish compression before creating the destination. Exclusive creation also
    # prevents overwriting a file that appeared after the initial existence check.
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as snapshot:
        for name, content in sorted(payloads.items()):
            snapshot.writestr(name, content)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(buffer.getbuffer())
    return output


def main(argv: list[str] | None = None) -> int:
    """Create a snapshot for review or a manually published release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "build/figures/reproduction.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/snapshots/reproducibility-snapshot.zip"
    )
    args = parser.parse_args(argv)
    try:
        output = create_snapshot(args.output, args.report, root=ROOT)
    except (OSError, ValueError, RuntimeError, BadZipFile, tarfile.TarError) as error:
        print(f"Snapshot packaging failed: {error}", file=sys.stderr)
        return 1
    print(f"Created verified snapshot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
