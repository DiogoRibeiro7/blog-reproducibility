"""Tests for the sequential-boundaries figure rendering."""

from pathlib import Path

import numpy as np

from blog_reproducibility.statistics.sequential_testing import (
    FigureBoundaries,
    GroupSequentialBoundary,
)
from blog_reproducibility.statistics.sequential_testing_figure import render_sequential_figure

# Rendering needs only the critical values; the recursion is not rerun.
BOUNDARIES = FigureBoundaries(
    naive=1.96,
    pocock=GroupSequentialBoundary((2.49,) * 7, (0.013,) * 7, 0.05),
    obrien_fleming=GroupSequentialBoundary(
        tuple(float(2.06 * np.sqrt(7 / k)) for k in range(1, 8)), (0.01,) * 7, 0.05
    ),
    mixture=tuple(float(3 + 2.3 / day) for day in range(1, 29)),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_sequential_figure(output_dir=tmp_path, boundaries=BOUNDARIES)

    assert artifact.path == tmp_path / "sequential_boundaries.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
