"""Tests for the risk communication figure rendering."""

from pathlib import Path

from blog_reproducibility.health.risk_communication_figure import (
    render_concentration_figure,
    render_relative_risk_figure,
)


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    concentration = render_concentration_figure(output_dir=tmp_path)
    risk = render_relative_risk_figure(output_dir=tmp_path)

    assert concentration.path == tmp_path / "science_concentration_and_amount.png"
    assert risk.path == tmp_path / "science_relative_absolute_risk.png"

    for artifact in (concentration, risk):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000

    assert (concentration.width, concentration.height) == (1440, 656)
    assert (risk.width, risk.height) == (1440, 672)
