"""Verify a portable SHA256SUMS inventory using only Python's standard library."""

import argparse
import hashlib
import re
import sys
from pathlib import Path, PurePosixPath


def verify_inventory(inventory_path: Path) -> int:
    """Check every listed file relative to the inventory, without changing files."""
    inventory_path = inventory_path.resolve()
    root = inventory_path.parent
    lines = inventory_path.read_text(encoding="utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("Checksum inventory must contain at least one file")

    seen = set()
    planned = []
    for number, line in enumerate(lines, start=1):
        record = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if record is None:
            raise ValueError(
                f"Invalid checksum record on line {number}; expected SHA256, two spaces, path"
            )
        digest, name = record.groups()
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(character in name for character in ("\\", ":", "\x00"))
        ):
            raise ValueError(f"Checksum path must be a portable relative POSIX path: {name!r}")
        target = root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Checksum path resolves outside the inventory directory: {name!r}")
        if target in seen:
            raise ValueError(f"Duplicate checksum path: {name!r}")
        seen.add(target)
        planned.append((name, target, digest))

    for name, target, digest in planned:
        if not target.is_file():
            raise FileNotFoundError(f"Missing checksum file: {name}")
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(planned)


def main(argv: list[str] | None = None) -> int:
    """Verify downloaded files without installing dependencies or rendering figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help="Path to an extracted snapshot's SHA256SUMS.")
    args = parser.parse_args(argv)
    try:
        count = verify_inventory(args.inventory)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Checksum verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified checksums for {count} files: {args.inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
