"""Verify that the article reference follows the executable manifest."""

from pathlib import Path

import pytest

from scripts import catalog
from scripts.manifest import load_manifest


def test_catalog_includes_every_registered_source_and_output() -> None:
    articles = load_manifest()
    content = catalog.render_catalog(articles)
    for identifier, article in articles.items():
        assert f"## {identifier}\n" in content
        assert article.source_url in content
        for path in (*article.scripts, *article.tests, *article.inputs):
            assert f"](../{path.as_posix()})" in content
        for path in article.figures:
            assert f"`{path.as_posix()}`" in content


def test_catalog_check_detects_stale_content_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    path = tmp_path / "docs/articles.md"
    assert catalog.main(["--check"]) == 1
    assert not path.exists()
    assert catalog.main([]) == 0
    assert catalog.main(["--check"]) == 0
    path.write_text("stale\n", encoding="utf-8")
    assert catalog.main(["--check"]) == 1
    assert path.read_text(encoding="utf-8") == "stale\n"
