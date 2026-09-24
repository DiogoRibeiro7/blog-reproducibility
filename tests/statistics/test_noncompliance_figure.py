"""Tests for the non-compliance estimator figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.noncompliance import (
    ALWAYS_TAKERS,
    COMPLIANCE_RATES,
    TRUE_EFFECT,
    USERS,
    ComplianceRow,
    Estimates,
    NoncomplianceSummary,
    SampleSizeRow,
    scenario_row,
)
from blog_reproducibility.statistics.noncompliance_figure import render_noncompliance_figure

# Rendering needs only the means; recomputing them would rerun 4,000 experiments.
_SPREAD = Estimates(0.17, 0.17, 0.18, 0.27)
SUMMARY = NoncomplianceSummary(
    effect=TRUE_EFFECT,
    users=USERS,
    rows=tuple(
        ComplianceRow(
            compliance=c,
            always_takers=ALWAYS_TAKERS,
            replications=500,
            simulated=Estimates(2 * c, 2 + 3.5 * (1 - c), 2 + 3 * (1 - c), 2.0),
            expected=Estimates(2 * c, 2 + 3.5 * (1 - c), 2 + 3 * (1 - c), 2.0),
            spread=_SPREAD,
        )
        for c in COMPLIANCE_RATES
    ),
    article=scenario_row(0.6, 0.1, 2.0),
    inert_feature=scenario_row(0.6, 0.1, 0.0),
    one_sided=scenario_row(0.7, 0.0, 2.0),
    sample_sizes=(SampleSizeRow(0.6, 294.3, 27.01),),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_noncompliance_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "noncompliance_estimators.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
