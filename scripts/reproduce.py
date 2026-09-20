"""Validate the article manifest and reproduce its figures with provenance."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scripts.check_data import verify_checksums
from scripts.manifest import ROOT, Article, load_manifest


def sha256(path: Path) -> str:
    """Hash a reproducibility input or generated artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_hashes(root: Path) -> dict[str, str]:
    """Snapshot source and data without including Python's generated caches."""
    source_paths = [root / "pyproject.toml", root / "poetry.lock"]
    for directory in ("src", "scripts", "articles", "data"):
        source_paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(source_paths)}


def git_state(root: Path) -> dict[str, str | bool | None]:
    """Describe the checkout, allowing execution from a source archive."""
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(toplevel).resolve() != root.resolve():
            return {"revision": None, "dirty": None}
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": bool(status.strip())}


def reproduce(articles: list[Article], output_dir: Path, *, root: Path = ROOT) -> Path:
    """Render into a fresh staging directory, then save figures and a run report."""
    output_dir = output_dir.resolve()
    report_path = output_dir / "reproduction.json"
    # A failed rerun must not leave a previous success report at this location.
    report_path.unlink(missing_ok=True)
    inputs = input_hashes(root)
    checkout = git_state(root)
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    figures: list[dict[str, str]] = []
    with TemporaryDirectory(prefix="blog-reproduce-") as temporary:
        staging = Path(temporary)
        for article in articles:
            # A shared script may also render figures owned by another article.
            # Keep each invocation separate until all expected outputs are verified.
            article_staging = staging / article.identifier
            relative_dir = article.figures[0].parent.relative_to("build/figures")
            for script in article.scripts:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(root / script),
                        "--output-dir",
                        str(article_staging / relative_dir),
                    ],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"{script.as_posix()} failed:\n{result.stderr or result.stdout}"
                    )
            for figure in article.figures:
                relative = figure.relative_to("build/figures")
                generated = article_staging / relative
                if not generated.is_file() or not generated.read_bytes().startswith(
                    b"\x89PNG\r\n\x1a\n"
                ):
                    raise RuntimeError(f"{article.identifier}: missing or invalid PNG: {relative}")
                figures.append(
                    {
                        "article": article.identifier,
                        "path": relative.as_posix(),
                        "sha256": sha256(generated),
                    }
                )
        if input_hashes(root) != inputs:
            raise RuntimeError(
                "Source or data files changed during reproduction; rerun when stable."
            )
        for figure_record in figures:
            destination = output_dir / figure_record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / figure_record["article"] / figure_record["path"], destination)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git": checkout,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": "Agg",
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in distributions())),
        "articles": [article.identifier for article in articles],
        "inputs": inputs,
        "figures": figures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    """Run repository reproduction commands without requiring shell-specific tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List registered article identifiers.")
    mode.add_argument(
        "--check", action="store_true", help="Validate metadata and local file references."
    )
    parser.add_argument("--article", help="Reproduce one article; the default is all articles.")
    parser.add_argument(
        "--domain", help="Select articles by scientific domain, such as statistics."
    )
    parser.add_argument("--search", help="Filter identifiers and module names by a search string.")
    parser.add_argument("--json", action="store_true", help="Emit --list results as JSON.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/figures")
    args = parser.parse_args(argv)
    if args.json and not args.list:
        parser.error("--json requires --list")
    if args.check and (args.article or args.domain or args.search):
        parser.error("--check validates the entire manifest and cannot be filtered")
    try:
        articles = load_manifest()
        verify_checksums(articles.values(), root=ROOT)
        if args.article and args.article not in articles:
            parser.error(
                f"unknown article {args.article!r}; use --list to see available identifiers"
            )
        selected = [articles[args.article]] if args.article else list(articles.values())
        if args.domain:
            domains = sorted(
                {domain for article in articles.values() for domain in article.domains}
            )
            if args.domain not in domains:
                parser.error(f"unknown domain {args.domain!r}; choose from {', '.join(domains)}")
            selected = [article for article in selected if args.domain in article.domains]
        if args.search:
            needle = args.search.casefold()
            selected = [
                article
                for article in selected
                if needle in " ".join((article.identifier, *article.modules)).casefold()
            ]
        if not selected:
            parser.error("no articles match the requested filters")
        if args.list:
            if args.json:
                print(
                    json.dumps(
                        [
                            {
                                "identifier": article.identifier,
                                "publication_status": article.publication_status,
                                "domains": article.domains,
                                "source_url": article.source_url,
                                "scripts": [path.as_posix() for path in article.scripts],
                                "figures": [path.as_posix() for path in article.figures],
                                "data_inputs": [path.as_posix() for path in article.inputs],
                            }
                            for article in selected
                        ],
                        indent=2,
                    )
                )
            else:
                print("\n".join(article.identifier for article in selected))
        elif args.check:
            print(f"Validated {len(articles)} articles, local references, and data checksums.")
        else:
            report = reproduce(selected, args.output_dir)
            print(f"Reproduced {len(selected)} articles. Report: {report}")
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        print(f"Reproduction failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
