"""Tests for results-rhetoric article figure rendering."""

from pathlib import Path

from blog_reproducibility.health.results_rhetoric_figure import render_results_rhetoric_figures


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    first, second = render_results_rhetoric_figures(output_dir=tmp_path)

    assert first.path == tmp_path / "results_rhetoric_retention.png"
    assert second.path == tmp_path / "results_rhetoric_testimonial_wall.png"

    for artifact in (first, second):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert (first.width, first.height) == (1152, 704)
    assert (second.width, second.height) == (1152, 736)
