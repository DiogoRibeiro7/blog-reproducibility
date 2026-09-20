"""Tests for climate evidence figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.climate_evidence_figure import (
    render_climate_evidence_figures,
)


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    shift, evidence = render_climate_evidence_figures(output_dir=tmp_path)

    assert shift.path == tmp_path / "science_weather_climate_shift.png"
    assert evidence.path == tmp_path / "science_climate_kl_evidence.png"

    for artifact in (shift, evidence):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert (shift.width, shift.height) == (1312, 688)
    assert (evidence.width, evidence.height) == (1600, 720)
