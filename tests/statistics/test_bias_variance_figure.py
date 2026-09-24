"""Tests for the bias-variance and lasso path figure rendering."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from blog_reproducibility.common.plotting import FigureArtifact, save_figure
from blog_reproducibility.statistics import bias_variance_figure
from blog_reproducibility.statistics.bias_variance import lasso_coefficient_path
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


def test_the_penalty_axis_decreases_from_left_to_right(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label says "decreasing": the largest penalty sits on the left, the smallest on the right.

    The published image set the limits from large to small and then inverted the
    axis, which cancelled out and left the penalty growing to the right.
    """
    saved: list[Figure] = []

    def capture(figure: Figure, *, slug: str, output_dir: Path) -> FigureArtifact:
        saved.append(figure)
        return save_figure(figure, slug=slug, output_dir=output_dir)

    monkeypatch.setattr(bias_variance_figure, "save_figure", capture)
    render_regularization_paths_figure(output_dir=tmp_path)

    (axis,) = saved[0].axes
    penalties = lasso_coefficient_path().penalties
    largest, smallest = float(penalties[0]), float(penalties[-1])
    left, right = axis.get_xlim()
    assert axis.get_xscale() == "log"
    assert "decreasing" in axis.get_xlabel()
    assert axis.xaxis_inverted()  # matplotlib's word for values that fall to the right
    assert left == pytest.approx(largest)
    assert right < smallest
    # In screen coordinates the penalties run leftwards to rightwards in decreasing order.
    screen = axis.transData.transform(np.column_stack([penalties, np.zeros_like(penalties)]))
    assert np.all(np.diff(screen[:, 0]) > 0)
    assert screen[0, 0] == pytest.approx(axis.transAxes.transform((0.0, 0.0))[0])
    # The survivors' labels sit to the right of the smallest penalty, inside the axes.
    labels = [child for child in axis.get_children() if isinstance(child, Annotation)]
    assert sorted(label.get_text() for label in labels) == [f"$x_{{{j}}}$" for j in (1, 2, 3, 6, 8)]
    for label in labels:
        assert label.xy[0] == pytest.approx(smallest)
        assert label.get_horizontalalignment() == "left"
