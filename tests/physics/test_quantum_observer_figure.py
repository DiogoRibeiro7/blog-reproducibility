"""Tests for quantum measurement figure rendering."""

from pathlib import Path

from blog_reproducibility.physics.quantum_observer_figure import (
    render_quantum_observer_figures,
)


def test_renderers_write_both_expected_pngs(tmp_path: Path) -> None:
    """Both article figures should render as non-empty PNG files."""
    visibility, eraser = render_quantum_observer_figures(output_dir=tmp_path)

    assert visibility.path == tmp_path / "science_quantum_marker_visibility.png"
    assert eraser.path == tmp_path / "science_quantum_eraser_conditioning.png"

    for artifact in (visibility, eraser):
        assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert artifact.path.stat().st_size > 10_000
        assert (artifact.width, artifact.height) == (1440, 864)
