"""Tests for confidence-set figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.confidence_sets_figure import render_confidence_set_figures


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    components, basins = render_confidence_set_figures(output_dir=tmp_path)

    assert components.path == tmp_path / "confidence_set_components.png"
    assert basins.path == tmp_path / "confidence_set_profile_basins.png"

    for artifact in (components, basins):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert components.width == 1536
    assert components.height == 672
    assert basins.width == 1152
    assert basins.height == 672
