"""Tests for the intermittent demand forecast bias figure rendering."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from blog_reproducibility.common.plotting import FigureArtifact, save_figure
from blog_reproducibility.statistics import intermittent_demand_figure
from blog_reproducibility.statistics.intermittent_demand import (
    METHODS,
    IntermittentSummary,
    MethodRow,
    closed_forms,
)
from blog_reproducibility.statistics.intermittent_demand_figure import (
    render_intermittent_demand_figure,
)

# Rendering needs only the averages and their errors; the series are not simulated.
AVERAGES = (0.0, 0.796, 0.800, 0.828, 0.787)
SUMMARY = IntermittentSummary(
    series=40,
    forecasts_per_series=234,
    rows=tuple(
        MethodRow(name, average, 0.01 if average else 0.0, (average,), average - 0.8, 1.27, 1.8)
        for name, average in zip(METHODS, AVERAGES, strict=True)
    ),
    closed_forms=closed_forms(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_intermittent_demand_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "intermittent_forecast_bias.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)


def test_the_title_names_the_one_method_far_off_the_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The title singles out the forecast of zero, not "only one method is wrong".

    The published title said only one method was wrong about the rate, but
    undebiased Croston is 3.5 percent above it too (2.8 standard errors); the
    forecast of zero is the one far off. The bars are unchanged, and the longer
    title still ends inside the axes, so the saved image is no wider.
    """
    saved: list[Figure] = []

    def capture(figure: Figure, *, slug: str, output_dir: Path) -> FigureArtifact:
        saved.append(figure)
        return save_figure(figure, slug=slug, output_dir=output_dir)

    monkeypatch.setattr(intermittent_demand_figure, "save_figure", capture)
    render_intermittent_demand_figure(output_dir=tmp_path, summary=SUMMARY)

    (axis,) = saved[0].axes
    title = axis.get_title(loc="left")
    assert title == "Only the zero forecast is far off the rate, and it wins on absolute error"
    assert "only one method" not in title.lower()
    bars = [patch for patch in axis.patches if isinstance(patch, Rectangle)]
    assert [bar.get_width() for bar in bars] == list(AVERAGES)
    assert [label.get_text() for label in axis.get_yticklabels()] == list(METHODS)
    (title_text,) = [
        child
        for child in axis.get_children()
        if isinstance(child, Text) and child.get_text() == title
    ]
    assert title_text.get_window_extent().x1 <= axis.get_window_extent().x1
