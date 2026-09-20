"""Tests for poll-selection figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.poll_selection_figure import render_poll_selection_figures


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    precision, weighting = render_poll_selection_figures(output_dir=tmp_path)

    assert precision.path == tmp_path / "science_poll_selection_precision.png"
    assert weighting.path == tmp_path / "science_poll_selection_weighting.png"

    for artifact in (precision, weighting):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000
        assert artifact.width == 1440
        assert artifact.height == 880
