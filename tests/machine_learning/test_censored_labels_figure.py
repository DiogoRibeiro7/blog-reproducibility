"""Tests for the censored-labels figure rendering."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from blog_reproducibility.common.plotting import FigureArtifact, save_figure
from blog_reproducibility.machine_learning import censored_labels_figure
from blog_reproducibility.machine_learning.censored_labels import (
    COHORTS,
    CensoredLabelsSummary,
    ChurnRates,
    CohortRow,
    ModelScore,
)
from blog_reproducibility.machine_learning.censored_labels_figure import render_cohorts_figure

# Rendering needs only the cohort table; recomputing it would refit every model.
SUMMARY = CensoredLabelsSummary(
    training_customers=150,
    short_followup_training=50,
    person_month_rows=1200,
    cohorts=tuple(
        CohortRow(lower, upper, 100, 0.48, 0.1 + 0.15 * index, 0.49, 0.47)
        for index, (lower, upper) in enumerate(COHORTS)
    ),
    scores=(ModelScore("naive", 0.76, 0.19),),
    naive_own_label_auc=0.9,
    naive_new_customer=0.006,
    true_new_customer=0.446,
    test_rates=ChurnRates(0.49, 0.42, 0.49),
    snapshot_positive_rate=0.54,
    tenure_free_mean=0.54,
)
TRUTH = "True 12-month probability"
MODELS = (
    "Naive: churned by extract date",
    "Fixed 12-month horizon",
    "Discrete-time hazard model",
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_cohorts_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "censored_labels_cohorts.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)


def test_the_truth_is_drawn_over_the_hazard_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The truth line is dashed and above every model line, and still first in the legend.

    Once the hazard model matched the truth its line covered the thin black truth
    line entirely; drawn on top and dashed, the truth shows with the hazard line
    visible through its gaps. The data are unchanged.
    """
    saved: list[Figure] = []

    def capture(figure: Figure, *, slug: str, output_dir: Path) -> FigureArtifact:
        saved.append(figure)
        return save_figure(figure, slug=slug, output_dir=output_dir)

    monkeypatch.setattr(censored_labels_figure, "save_figure", capture)
    render_cohorts_figure(output_dir=tmp_path, summary=SUMMARY)

    (axis,) = saved[0].axes
    lines = {line.get_label(): line for line in axis.get_lines()}
    assert set(lines) == {TRUTH, *MODELS}
    truth = lines[TRUTH]
    assert truth.get_linestyle() == "--"
    for label in MODELS:
        assert lines[label].get_linestyle() == "-"
        assert truth.get_zorder() > lines[label].get_zorder()
    assert np.array_equal(np.asarray(truth.get_ydata()), [row.true for row in SUMMARY.cohorts])
    hazard = np.asarray(lines[MODELS[2]].get_ydata())
    assert np.array_equal(hazard, [row.hazard for row in SUMMARY.cohorts])

    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [TRUTH, *MODELS]
    handle = legend.legend_handles[0]
    assert isinstance(handle, Line2D)
    assert handle.get_linestyle() == "--"
