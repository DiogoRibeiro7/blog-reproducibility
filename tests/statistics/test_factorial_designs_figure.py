"""Tests for the one-factor-at-a-time against factorial design figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.factorial_designs import (
    FACTORIAL_RUNS,
    OFAT_RUNS,
    DesignRow,
    FactorialSummary,
    closed_forms,
)
from blog_reproducibility.statistics.factorial_designs_figure import (
    render_factorial_designs_figure,
)


def _row(design: str, runs: int, share: float) -> DesignRow:
    return DesignRow(design, runs, 1500, round(1500 * share), share, 1.0, share, share)


# Rendering needs only the shares; round numbers stand in for the 15,000 experiments.
SUMMARY = FactorialSummary(
    ofat=tuple(
        _row("one factor at a time", runs, share)
        for runs, share in zip(OFAT_RUNS, (0.09, 0.07, 0.03, 0.01, 0.004, 0.001), strict=True)
    ),
    factorial=tuple(
        _row("factorial", runs, share)
        for runs, share in zip(FACTORIAL_RUNS, (0.69, 0.70, 0.77, 0.86), strict=True)
    ),
    closed_forms=closed_forms(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_factorial_designs_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "factorial_vs_ofat_optimum.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
