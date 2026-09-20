"""Tests for p-value evidence figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.pvalue_evidence_figure import (
    render_pvalue_evidence_figures,
)


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    selected, comparison = render_pvalue_evidence_figures(output_dir=tmp_path)

    assert selected.path == tmp_path / "science_pvalue_selected_studies.png"
    assert comparison.path == tmp_path / "science_pvalue_study_comparison.png"

    for artifact in (selected, comparison):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert selected.width == 1440
    assert selected.height == 864
    assert comparison.width == 1440
    assert comparison.height == 752
