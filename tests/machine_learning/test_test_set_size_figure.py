"""Tests for the test-set size ranking figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.test_set_size import RankingRow
from blog_reproducibility.machine_learning.test_set_size_figure import (
    render_test_set_size_figure,
)

SIZES = (200, 500, 1000, 2000, 5000, 10000, 20000)
ROWS = tuple(
    RankingRow(
        test_size=size,
        same_set=min(1.0, 0.6 + 0.07 * index),
        separate_sets=min(1.0, 0.58 + 0.06 * index),
        paired_significant=min(1.0, 0.05 * 1.8**index),
        intervals_disjoint=min(1.0, 0.002 * 2.7**index),
    )
    for index, size in enumerate(SIZES)
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_test_set_size_figure(output_dir=tmp_path, rows=ROWS)

    assert artifact.path == tmp_path / "test_set_size_ranking.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
