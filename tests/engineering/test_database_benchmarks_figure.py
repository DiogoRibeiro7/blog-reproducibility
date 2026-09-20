"""Tests for database article figure rendering."""

from pathlib import Path

from blog_reproducibility.engineering.database_benchmarks_figure import render_database_figures


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    first, second = render_database_figures(output_dir=tmp_path)

    assert first.path == tmp_path / "database_rows_versus_columns.png"
    assert second.path == tmp_path / "database_index_helps_and_hurts.png"

    for artifact in (first, second):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert (first.width, first.height) == (1536, 832)
    assert (second.width, second.height) == (1536, 672)
