"""Tests for the winner's-curse figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.winners_curse import OptimismCurve
from blog_reproducibility.machine_learning.winners_curse_figure import (
    render_winners_curse_figure,
)

CANDIDATES = (1, 2, 5, 10, 20, 50, 100, 200, 500)
CURVES = tuple(
    OptimismCurve(
        validation_size=size,
        candidates=CANDIDATES,
        optimism=tuple(scale * index for index in range(len(CANDIDATES))),
        equal_candidates_bound=tuple(1.2 * scale * index for index in range(len(CANDIDATES))),
    )
    for size, scale in ((200, 0.9), (1000, 0.35), (5000, 0.1))
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_winners_curse_figure(output_dir=tmp_path, curves=CURVES)

    assert artifact.path == tmp_path / "winners_curse_optimism.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
