"""Tests for the learning-curve figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.learning_curves import (
    TRAINING_SIZES,
    CurvePoint,
    LearningCurve,
    LearningCurveSummary,
    PowerLawFit,
)
from blog_reproducibility.machine_learning.learning_curves_figure import (
    render_learning_curves_figure,
)


def _curve(model: str, fit: PowerLawFit) -> LearningCurve:
    return LearningCurve(
        model,
        tuple(CurvePoint(n, fit.predict(n), 0.001, fit.predict(n) - 0.02) for n in TRAINING_SIZES),
    )


# Rendering needs only the curves and fits; recomputing them would retrain 48 models.
LOGISTIC = PowerLawFit(0.294, 1.4, 0.69)
BOOSTING = PowerLawFit(0.206, 2.4, 0.60)
SUMMARY = LearningCurveSummary(
    bayes_error=0.205,
    positive_rate=0.506,
    fitted_sizes=5,
    curves=(_curve("logistic", LOGISTIC), _curve("boosting", BOOSTING)),
    fits=(LOGISTIC, BOOSTING),
    noisy_curve=_curve("boosting", BOOSTING),
    noisy_fit=BOOSTING,
    noisy_test_floor=0.264,
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_learning_curves_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "learning_curves_extrapolation.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
