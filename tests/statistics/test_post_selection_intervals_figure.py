"""Tests for the post-selection interval figure rendering."""

from pathlib import Path

import numpy as np

from blog_reproducibility.statistics.post_selection_intervals import (
    FIGURE_MEANS,
    ProcedureCurves,
    WidthCurve,
)
from blog_reproducibility.statistics.post_selection_intervals_figure import (
    render_selective_interval_runaway_figure,
    render_selective_width_by_procedure_figure,
)

# Rendering needs only the curves; recomputing them would rerun 6.8 million draws.
DISTANCES = np.geomspace(0.01, 4.0, 12)
WIDTH_CURVE = WidthCurve(
    distances=tuple(float(d) for d in DISTANCES),
    widths=tuple(float(3.69 / d + 3.92) for d in DISTANCES),
)
PROCEDURE_CURVES = ProcedureCurves(
    means=FIGURE_MEANS,
    hard_median=tuple(15.0 - 2.7 * m for m in FIGURE_MEANS),
    hard_p90=tuple(84.0 - 19.0 * m for m in FIGURE_MEANS),
    randomised_median=tuple(5.13 - 0.23 * m for m in FIGURE_MEANS),
    randomised_p90=tuple(5.31 - 0.18 * m for m in FIGURE_MEANS),
    randomised_selected=tuple(0.08 + 0.21 * m for m in FIGURE_MEANS),
    split=5.54,
)


def test_runaway_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The two-panel figure should render as a non-empty PNG file."""
    artifact = render_selective_interval_runaway_figure(output_dir=tmp_path, summary=WIDTH_CURVE)

    assert artifact.path == tmp_path / "selective_interval_runaway.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1536, 672)


def test_runaway_renderer_computes_its_own_curve(tmp_path: Path) -> None:
    """Without a precomputed curve the renderer computes the 60 exact widths itself."""
    artifact = render_selective_interval_runaway_figure(output_dir=tmp_path)

    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (artifact.width, artifact.height) == (1536, 672)


def test_procedure_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The width-by-procedure figure should render as a non-empty PNG file."""
    artifact = render_selective_width_by_procedure_figure(
        output_dir=tmp_path, summary=PROCEDURE_CURVES
    )

    assert artifact.path == tmp_path / "selective_width_by_procedure.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
