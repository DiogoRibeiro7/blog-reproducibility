"""Tests for the permutation-importance figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.permutation_importance import (
    FEATURES,
    Extrapolation,
    ImportanceSummary,
)
from blog_reproducibility.machine_learning.permutation_importance_figure import (
    render_permutation_importance_figure,
)

# Rendering needs only the importances; recomputing them would refit five forests.
SUMMARY = ImportanceSummary(
    features=FEATURES,
    train_mse=0.6,
    test_mse=1.1,
    permutation=(3.8, 1.0, 0.45, 0.0),
    training_permutation=(3.9, 1.2, 0.74, 0.15),
    without_duplicate=(8.1, 0.5, 0.0),
    group_pair=8.2,
    conditional=(0.29, 0.02, 0.43, 0.0),
    drop_column=(0.18, -0.001, 0.27, 0.004),
    extrapolation=Extrapolation(0.98, 0.03, 0.0, 0.48),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_permutation_importance_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "permutation_importance_methods.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1664, 544)
