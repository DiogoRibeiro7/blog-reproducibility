"""Tests for the extreme value tail figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.extreme_values import (
    BootstrapIntervals,
    DependentRecord,
    ExtremeValueSummary,
    MaximumDistribution,
    PotFit,
    RecordEstimates,
    TailCurves,
)
from blog_reproducibility.statistics.extreme_values_figure import render_extreme_values_figure

# Rendering needs only the curves; the payload would refit the tail and bootstrap it.
LEVELS = tuple(130 + 490 * i / 399 for i in range(400))
OBSERVED = tuple(265 - 135 * (i / 525) ** 0.25 for i in range(526))
FIT = PotFit(0.98, 130.0, 526, 0.19, 10.6, 0.02, 302.0, 425.0)
SUMMARY = ExtremeValueSummary(
    curves=TailCurves(
        levels=LEVELS,
        true=tuple(0.02 * (1 + 0.25 * (x - 130) / 8.0) ** -4 for x in LEVELS),
        normal=tuple(0.02 * 10 ** (-(((x - 130) / 12) ** 2)) for x in LEVELS),
        pareto=tuple(0.02 * (1 + 0.19 * (x - 130) / 10.6) ** (-1 / 0.19) for x in LEVELS),
        observed_levels=OBSERVED,
        observed_probabilities=tuple((i + 1) / 26_280 for i in range(526)),
    ),
    record=RecordEstimates(26_280, 267, 326, 502, 265, 318, 100, 14, 159, 166, 50, 180),
    fit=FIT,
    bootstrap=BootstrapIntervals(500, (255.0, 355.0), (319.0, 563.0)),
    thresholds=(FIT,),
    dependent=DependentRecord(0.8, 526, 194, 0.37, FIT),
    maxima=(MaximumDistribution(0.0, 283, 253, 328, 0.74),),
    normal_limit_ten_year=160.0,
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_extreme_values_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "extreme_value_tail_plot.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 736)
