"""Tests for inflammation-marker article figure rendering."""

from pathlib import Path

from blog_reproducibility.health.inflammation_markers_figure import render_inflammation_figures


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    first, second = render_inflammation_figures(output_dir=tmp_path)

    assert first.path == tmp_path / "inflammation_crp_scale.png"
    assert second.path == tmp_path / "inflammation_trials_forest.png"

    for artifact in (first, second):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert (first.width, first.height) == (1152, 704)
    assert (second.width, second.height) == (1152, 704)
