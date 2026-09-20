"""Verify registered data inputs against the committed SHA-256 inventory."""

import argparse
import hashlib
import sys
from collections.abc import Iterable
from difflib import unified_diff
from pathlib import Path

import yaml

from scripts.manifest import ROOT, Article, load_manifest

INVENTORY = Path("data/SHA256SUMS")


def render_checksums(articles: Iterable[Article], *, root: Path = ROOT) -> str:
    """Describe each distinct manifest input once, in a stable order."""
    paths = sorted(
        {path for article in articles for path in article.inputs}, key=lambda path: path.as_posix()
    )
    records = []
    for path in paths:
        with (root / path).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(f"{digest}  {path.as_posix()}\n")
    return "".join(records)


def verify_checksums(articles: Iterable[Article], *, root: Path = ROOT) -> None:
    """Reject changed bytes and missing, stale, or duplicate inventory entries."""
    expected = render_checksums(articles, root=root)
    recorded = (root / INVENTORY).read_text(encoding="utf-8")
    if recorded != expected:
        difference = "".join(
            unified_diff(
                recorded.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=INVENTORY.as_posix(),
                tofile="current manifest inputs",
            )
        )
        raise ValueError(
            "Data checksum inventory does not match the registered inputs. "
            "Review the data and provenance changes before running "
            "`python -m scripts.check_data --update`.\n" + difference
        )


def main(argv: list[str] | None = None) -> int:
    """Check inputs by default; replace the inventory only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true", help="Record the current inputs after reviewing changes."
    )
    args = parser.parse_args(argv)
    try:
        articles = load_manifest(ROOT)
        if args.update:
            (ROOT / INVENTORY).write_text(
                render_checksums(articles.values(), root=ROOT), encoding="utf-8", newline="\n"
            )
            print(f"Updated {INVENTORY.as_posix()}; review and commit it with the data changes.")
        else:
            verify_checksums(articles.values(), root=ROOT)
            print("All registered data inputs match their committed SHA-256 checksums.")
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Data verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
