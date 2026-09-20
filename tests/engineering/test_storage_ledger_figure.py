"""Tests for storage ledger draft figure rendering."""

from pathlib import Path

from blog_reproducibility.engineering.storage_ledger_figure import render_storage_figure


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The draft figure should render as a non-empty PNG file."""
    artifact = render_storage_figure(output_dir=tmp_path)

    assert artifact.path == tmp_path / "environment_storage_constraints.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (2080, 640)
