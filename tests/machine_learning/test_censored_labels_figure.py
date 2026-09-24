"""Tests for the censored-labels figure rendering."""

from pathlib import Path

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


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_cohorts_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "censored_labels_cohorts.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
