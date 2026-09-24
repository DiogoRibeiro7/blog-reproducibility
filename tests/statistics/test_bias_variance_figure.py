"""Tests for the bias-variance and lasso path figure rendering."""

from collections.abc import Callable
from pathlib import Path

import pytest

from blog_reproducibility.common.plotting import FigureArtifact
from blog_reproducibility.statistics.bias_variance_figure import (
    render_bias_variance_figure,
    render_regularization_paths_figure,
)


@pytest.mark.parametrize(
    ("render", "slug"),
    [
        (render_bias_variance_figure, "bias_variance"),
        (render_regularization_paths_figure, "regularization_paths"),
    ],
)
def test_renderer_writes_the_expected_png(
    tmp_path: Path, render: Callable[..., FigureArtifact], slug: str
) -> None:
    """Each article figure should render as a non-empty PNG file."""
    artifact = render(output_dir=tmp_path)

    assert artifact.path == tmp_path / f"{slug}.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
