"""Validate a version tag and extract its dated changelog entry for a GitHub release."""

import argparse
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def release_notes(tag: str, *, root: Path = ROOT) -> str:
    """Require matching package metadata and one nonempty, dated release section."""
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version.strip() or version != version.strip():
        raise ValueError("pyproject.toml must declare a nonempty project.version string")
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match project.version; expected 'v{version}'")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^##[ \t]+([^\n]+)", changelog, re.MULTILINE))
    matches = [
        index
        for index, heading in enumerate(headings)
        if heading.group(1).strip().split(" - ", 1)[0] == version
    ]
    if len(matches) != 1:
        raise ValueError(
            f"CHANGELOG.md must contain exactly one '## {version} - YYYY-MM-DD' section"
        )
    index = matches[0]
    heading = headings[index]
    title = heading.group(1).strip()
    if not re.fullmatch(re.escape(version) + r" - [0-9]{4}-[0-9]{2}-[0-9]{2}", title):
        raise ValueError(f"Release heading must be '## {version} - YYYY-MM-DD'")
    try:
        date.fromisoformat(title.rsplit(" - ", 1)[1])
    except ValueError as error:
        raise ValueError(f"Invalid release date in heading: {title}") from error

    end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
    notes = changelog[heading.end() : end].strip()
    if not any(line.strip() and not line.lstrip().startswith("#") for line in notes.splitlines()):
        raise ValueError(f"Release section for {version} must contain release notes")
    return notes + "\n"


def main(argv: list[str] | None = None) -> int:
    """Check release readiness, optionally saving notes after successful validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Existing or proposed tag, such as v0.1.0.")
    parser.add_argument("--notes-output", type=Path, help="Write the matching release notes here.")
    args = parser.parse_args(argv)
    try:
        notes = release_notes(args.tag, root=ROOT)
        if args.notes_output is not None:
            args.notes_output.parent.mkdir(parents=True, exist_ok=True)
            args.notes_output.write_text(notes, encoding="utf-8", newline="\n")
    except (OSError, ValueError) as error:
        print(f"Release validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated release metadata and changelog for {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
