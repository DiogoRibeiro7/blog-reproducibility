"""Tests for the acceptance sampling operating characteristic figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.acceptance_sampling import example_payload
from blog_reproducibility.statistics.acceptance_sampling_figure import (
    render_acceptance_sampling_figure,
)

# The payload is closed form and fast, so the figure renders the real curves.
SUMMARY = example_payload()


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_acceptance_sampling_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "acceptance_sampling_oc.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)


def test_renderer_computes_the_payload_when_none_is_given(tmp_path: Path) -> None:
    """Without a summary the renderer computes the curves itself."""
    artifact = render_acceptance_sampling_figure(output_dir=tmp_path)

    assert artifact.path.exists()
