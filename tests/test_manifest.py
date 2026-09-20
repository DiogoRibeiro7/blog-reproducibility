"""Prevent article mappings from silently drifting away from the repository."""

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.manifest import ROOT, load_manifest


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    """Create a disposable checkout containing the registered local inputs."""
    for directory in ("articles", "src", "scripts", "tests", "data"):
        shutil.copytree(
            ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__")
        )
    return tmp_path


def change_entry(root: Path, field: str, value: object) -> None:
    """Edit the first article in a disposable manifest."""
    path = root / "articles/manifest.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = next(iter(document["articles"].values()))
    entry[field] = value
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def test_repository_manifest_is_valid() -> None:
    articles = load_manifest()
    assert len(articles) >= 2
    assert sum(len(article.figures) for article in articles.values()) >= 3


def test_drafts_are_registered_without_a_published_permalink() -> None:
    drafts = [entry for entry in load_manifest().values() if entry.publication_status == "draft"]
    assert drafts
    for entry in drafts:
        assert entry.permalink == ""
        assert "/_drafts/" in entry.source_url


def test_identifiers_cannot_collide_across_publication_states(manifest_root: Path) -> None:
    path = manifest_root / "articles/manifest.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    identifier, entry = next(iter(document["articles"].items()))
    draft = copy.deepcopy(entry)
    draft["path"] = "_drafts/ideas/example.md"
    del draft["permalink"]
    document.setdefault("drafts", {})[identifier] = draft
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unique across articles and drafts"):
        load_manifest(manifest_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modules", ["blog_reproducibility.missing"], "Missing file"),
        ("figure_scripts", ["scripts/figures/missing.py"], "Missing file"),
        ("tests", ["tests/missing.py"], "Missing file"),
        ("data_inputs", ["data/missing.csv"], "Missing file"),
        ("benchmark_harness", "scripts/benchmarks/missing.py", "Missing file"),
        ("benchmark_harness", "tests/test_package.py", "must be inside scripts/benchmarks"),
        ("figure_scripts", ["tests/test_package.py"], "must be inside scripts/figures"),
        ("tests", ["tests/../README.md"], "does not match"),
        ("tests", ["/tmp/outside.py"], "does not match"),
        ("tests", ["C:/outside.py"], "does not match"),
        ("tests", ["tests/README.md"], "must have the .py extension"),
        ("generated_figures", ["figures/out.png"], "must be inside build/figures"),
        ("generated_figures", ["build/figures/out.svg"], "expected a .png"),
        (
            "generated_figures",
            ["build/figures/a/one.png", "build/figures/b/two.png"],
            "share one output",
        ),
        ("figure_scripts", [], "non-empty"),
        ("tests", ["tests/test_package.py", "tests/test_package.py"], "non-unique"),
        ("unexpected", True, "Additional properties"),
    ],
)
def test_manifest_rejects_broken_contracts(
    manifest_root: Path, field: str, value: object, message: str
) -> None:
    change_entry(manifest_root, field, value)
    with pytest.raises(ValueError, match=message):
        load_manifest(manifest_root)


def test_manifest_rejects_output_collisions(manifest_root: Path) -> None:
    path = manifest_root / "articles/manifest.yml"
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["articles"]["duplicate"] = copy.deepcopy(next(iter(document["articles"].values())))
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate generated figure"):
        load_manifest(manifest_root)
