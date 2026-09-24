"""Tests for the propensity score estimator figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.propensity_scores import (
    CONDITIONS,
    TRUE_EFFECT,
    UNITS,
    ConditionRow,
    Estimates,
    PropensitySummary,
)
from blog_reproducibility.statistics.propensity_scores_figure import (
    render_propensity_scores_figure,
)

# Rendering needs only the absolute biases; recomputing them would rerun 1,600 studies.
_BIASES = (
    (1.67, 0.004, 0.004, 0.006, 0.005),
    (3.63, 0.189, 0.060, 0.006, 0.004),
    (1.67, 0.004, 0.241, 0.231, 0.003),
    (2.48, 0.000, 0.072, 0.289, 0.004),
)
_SPREAD = Estimates(0.1, 0.1, 0.2, 0.15, 0.1)
SUMMARY = PropensitySummary(
    effect=TRUE_EFFECT,
    units=UNITS,
    rows=tuple(
        ConditionRow(
            condition=condition,
            replications=400,
            mean=Estimates(*(TRUE_EFFECT + b for b in biases)),
            standard_deviation=_SPREAD,
            root_mean_squared_error=_SPREAD,
            absolute_bias=Estimates(*biases),
            naive_limit=TRUE_EFFECT + biases[0],
            regression_limit=TRUE_EFFECT - biases[1],
        )
        for condition, biases in zip(CONDITIONS, _BIASES, strict=True)
    ),
    overlap=(),
    balance=(),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_propensity_scores_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "propensity_estimator_bias.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
