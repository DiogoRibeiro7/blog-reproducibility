"""Tests for the drift figure rendering."""

from collections.abc import Callable
from pathlib import Path

import pytest

from blog_reproducibility.common.plotting import FigureArtifact
from blog_reproducibility.machine_learning.drift import EDGES, STAGES
from blog_reproducibility.machine_learning.drift_figure import (
    ARROWS,
    BOXES,
    render_architecture_figure,
    render_concept_drift_figure,
    render_feature_drift_figure,
    render_prediction_drift_figure,
)


@pytest.mark.parametrize(
    ("render", "slug", "size"),
    [
        (render_architecture_figure, "drift_monitoring_architecture", (1696, 832)),
        (render_feature_drift_figure, "feature_drift_distribution", (1152, 672)),
        (render_concept_drift_figure, "concept_drift_boundary", (1152, 672)),
        (render_prediction_drift_figure, "prediction_drift_threshold", (1152, 672)),
    ],
)
def test_renderer_writes_the_expected_png(
    tmp_path: Path,
    render: Callable[..., FigureArtifact],
    slug: str,
    size: tuple[int, int],
) -> None:
    """Each article figure should render as a non-empty PNG file."""
    artifact = render(output_dir=tmp_path)

    assert artifact.path == tmp_path / f"{slug}.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == size


def test_diagram_draws_every_stage_and_edge() -> None:
    """The diagram has a box per stage and an arrow per edge, and nothing else."""
    assert set(BOXES) == set(STAGES)
    assert set(ARROWS) == set(EDGES)
