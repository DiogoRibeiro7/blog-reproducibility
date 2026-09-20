"""Tests for testimonial figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.testimonial_selection_figure import (
    render_testimonial_figures,
)


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    selection, counterfactual = render_testimonial_figures(output_dir=tmp_path)

    assert selection.path == tmp_path / "science_testimonial_selection.png"
    assert counterfactual.path == tmp_path / "science_testimonial_counterfactual.png"

    for artifact in (selection, counterfactual):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert selection.width == 1440
    assert selection.height == 928
    assert counterfactual.width == 1440
    assert counterfactual.height == 848
