"""Tests for screening article figure rendering."""

from pathlib import Path

from blog_reproducibility.health.screening_survival_figure import render_screening_survival_figures


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    first, second = render_screening_survival_figures(output_dir=tmp_path)

    assert first.path == tmp_path / "science_screening_survival_and_mortality.png"
    assert second.path == tmp_path / "science_screening_duration_selection.png"

    for artifact in (first, second):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert (first.width, first.height) == (1600, 944)
    assert (second.width, second.height) == (1440, 848)
