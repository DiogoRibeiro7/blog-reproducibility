"""Tests for the feature-selection leakage figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.selection_leakage import (
    CANDIDATE_FEATURES,
    LeakageRow,
    SelectionLeakageSummary,
)
from blog_reproducibility.machine_learning.selection_leakage_figure import (
    render_selection_leakage_figure,
)

# Rendering needs only the curves; recomputing them would refit 2,800 classifiers.
SUMMARY = SelectionLeakageSummary(
    samples=100,
    kept_features=10,
    replications=40,
    rows=tuple(
        LeakageRow(pool, 0.55 + 0.04 * index, 0.5 + (-1) ** index * 0.01)
        for index, pool in enumerate(CANDIDATE_FEATURES)
    ),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_selection_leakage_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "cv_selection_leakage.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
