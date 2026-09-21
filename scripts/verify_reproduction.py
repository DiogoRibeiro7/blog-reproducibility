"""Verify saved figure checksums against a reproduction report without rendering."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


def verify_report(report_path: Path) -> int:
    """Check every recorded figure, resolving paths from the report's directory."""
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("The reproduction report must be a JSON object")
    version = report.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported reproduction report schema_version: {version!r}")
    figures = report.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError("The reproduction report must contain a nonempty figures list")

    root = report_path.parent
    seen = set()
    planned = []
    for index, figure in enumerate(figures):
        if not isinstance(figure, dict):
            raise ValueError(f"Invalid figure record at index {index}")
        name, digest = figure.get("path"), figure.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError(f"Figure {index} must have string path and sha256 fields")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 checksum for figure {index}: {name!r}")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(character in name for character in ("\\", ":", "\x00"))
        ):
            raise ValueError(f"Figure path must be a portable relative POSIX path: {name!r}")
        target = root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Figure path resolves outside the report directory: {name!r}")
        if target in seen:
            raise ValueError(f"Duplicate figure path: {name!r}")
        seen.add(target)
        planned.append((name, target, digest))

    for name, target, digest in planned:
        if not target.is_file():
            raise FileNotFoundError(f"Missing figure file: {name}")
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(planned)


def main(argv: list[str] | None = None) -> int:
    """Verify artifacts without changing them or invoking any renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to the saved reproduction.json report.")
    args = parser.parse_args(argv)
    try:
        count = verify_report(args.report)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified checksums for {count} figures: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
