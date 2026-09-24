"""Check the propensity score estimators against the website, closed forms, article and figure.

The figure's payload (1,600 studies, each with a logistic fit and a k-d tree
match) takes about 12 seconds, so these tests do not recompute it. They check
instead that the port is draw for draw identical to a transcription of the
website's loop, including its full 40 Newton steps, pin the first study and
the mean of the first 40 in each condition exactly, and test the figure's
claims on those first 40 studies of each condition, which are the figure's own.

The full run reproduces all three of the article's tables at their printed
precision (for example 3.67, 1.81 and 2.00 for the naive, regression and doubly
robust means when the outcome model is wrong, and biases of -0.24 and -0.23 for
matching and weighting when the propensity model is wrong) and the figure's
biases: regression 0.004, 0.189, 0.004 and 0.000 across the four conditions,
matching 0.004, 0.060, 0.241 and 0.072, weighting 0.006, 0.006, 0.231 and 0.289,
doubly robust 0.005, 0.004, 0.003 and 0.005; the naive difference is off by
1.67, 3.63, 1.67 and 2.48, the alt text's "between 1.7 and 3.6".

Under poor overlap many scores are clipped to the same value, and the matched
control among tied ones is whatever SciPy's k-d tree returns first; that choice
alone moves a study's matching estimate by up to 0.57. The matching values in
that condition are therefore compared only with the website's own k-d tree,
not pinned, since another SciPy version could break the ties differently.

The closed forms (the naive difference, the misspecified regression's limit of
1.81, the covariates' standardised differences and the share of scores outside
``[0.1, 0.9]``) are checked against a large sample and against Stein's identity.
The article's overlap and balance tables come from later draws and are checked
against them at their noise level. Three remarks in its prose do not hold as
written: the naive bias is 83 percent of the effect, not 84 (1.67 / 2, rounded
up); matching's spread under poor overlap is 3.1 times its spread with both
models correct, not quadrupled; and the misspecified regression's 95 percent
interval excludes the truth in only 30 percent of the figure's studies (121 of
400), not as a rule, since its bias of 0.19 is 1.6 standard errors.
"""

from math import sqrt

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate

from blog_reproducibility.statistics.propensity_scores import (
    CLIP,
    CONDITIONS,
    REPLICATIONS,
    TREATMENT_COEFFICIENTS,
    TRUE_EFFECT,
    ConditionRow,
    condition_estimates,
    doubly_robust,
    draw_study,
    estimate_all,
    fit_propensity,
    inverse_probability_weighting,
    naive_difference,
    naive_difference_limit,
    propensity_matching,
    regression_adjustment,
    regression_limit,
    share_outside,
    skip_studies,
    standardised_difference,
    summarise,
)

SHORT_REPLICATIONS = 40
SHORT = condition_estimates(estimated=SHORT_REPLICATIONS)
ROWS = [summarise(condition, values) for condition, values in zip(CONDITIONS, SHORT, strict=True)]

# The website generator's first study, and the mean of its first 40, in each condition.
# None: the poor-overlap matching estimate depends on the k-d tree's tie-breaking.
FIRST_STUDIES = (
    (3.725876, 2.003271, 2.144460, 2.082380, 1.997080),
    (5.748943, 1.708148, 2.296913, 1.886816, 1.895552),
    (3.658010, 1.916818, 1.653479, 1.774841, 1.938039),
    (4.485818, 1.981401, None, 2.276829, 1.994327),
)
FIRST_MEANS = (
    (3.6466, 1.9866, 1.9588, 1.9812, 1.9823),
    (5.6390, 1.7944, 2.1000, 1.9760, 1.9810),
    (3.6692, 1.9987, 1.7688, 1.7673, 1.9988),
    (4.4763, 2.0043, None, 2.2461, 1.9857),
)


def _website_draw(
    r: np.random.Generator, n: int = 4000, strength: float = 1.0, nonlinear: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Transcribe the website generator's study."""
    x = r.normal(0, 1, (n, 3))
    logit = strength * (0.9 * x[:, 0] + 0.6 * x[:, 1] - 0.5 * x[:, 2])
    d = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    base = 2.0 * x[:, 0] + 1.0 * x[:, 1] + 0.5 * x[:, 2]
    if nonlinear:
        base = base + 1.0 * x[:, 0] ** 3 - 1.0 * x[:, 0] * x[:, 1]
    return x, d, 10 + base + 2.0 * d + r.normal(0, 2, n)


def _website_fit(
    x: NDArray[np.float64], d: NDArray[np.int64], iters: int = 40
) -> NDArray[np.float64]:
    """Transcribe the website generator's logistic fit, every Newton step of it."""
    a = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(a.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a @ b))
        w = p * (1 - p) + 1e-9
        b += np.linalg.solve((a * w[:, None]).T @ a + 1e-8 * np.eye(a.shape[1]), a.T @ (d - p))
    scores: NDArray[np.float64] = 1 / (1 + np.exp(-a @ b))
    return scores


def _website_estimates(
    x: NDArray[np.float64], d: NDArray[np.int64], y: NDArray[np.float64], ps: NDArray[np.float64]
) -> tuple[float, ...]:
    """Transcribe the website generator's five estimates."""
    from scipy.spatial import cKDTree

    a = np.column_stack([np.ones(len(x)), d, x])
    reg = np.linalg.lstsq(a, y, rcond=None)[0][1]
    tr, co = np.where(d == 1)[0], np.where(d == 0)[0]
    _, j = cKDTree(ps[co][:, None]).query(ps[tr][:, None])
    mat = np.mean(y[tr] - y[co][j])
    w1, w0 = d / ps, (1 - d) / (1 - ps)
    w = np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)
    b = np.column_stack([np.ones(len(x)), x])
    m1 = b @ np.linalg.lstsq(b[d == 1], y[d == 1], rcond=None)[0]
    m0 = b @ np.linalg.lstsq(b[d == 0], y[d == 0], rcond=None)[0]
    dr = np.mean(m1 - m0 + d * (y - m1) / ps - (1 - d) * (y - m0) / (1 - ps))
    return tuple(float(v) for v in (y[d == 1].mean() - y[d == 0].mean(), reg, mat, w, dr))


def _z(row: ConditionRow, field: str) -> float:
    """Bias of one estimator in standard errors of its mean over the short run."""
    error = float(getattr(row.standard_deviation, field)) / sqrt(row.replications)
    return (float(getattr(row.mean, field)) - TRUE_EFFECT) / error


def test_the_first_studies_are_pinned() -> None:
    """The first study and the first 40 studies' means, as the website's generator computes them."""
    for values, first, means in zip(SHORT, FIRST_STUDIES, FIRST_MEANS, strict=True):
        assert values.shape == (SHORT_REPLICATIONS, 5)
        for j, (one, mean) in enumerate(zip(first, means, strict=True)):
            if one is not None and mean is not None:
                assert round(float(values[0, j]), 6) == one
                assert round(float(values[:, j].mean()), 4) == mean


def test_the_studies_match_the_website_loop() -> None:
    """The first two studies of every condition, with all 40 Newton steps, estimate for estimate."""
    generator = np.random.default_rng(0)
    for condition, values in zip(CONDITIONS, SHORT, strict=True):
        for rep in range(2):
            x, d, y = _website_draw(
                generator, strength=condition.strength, nonlinear=condition.nonlinear
            )
            features = x[:, :2] if condition.wrong_propensity else x
            ps = np.clip(_website_fit(features, d), *CLIP)
            assert tuple(float(v) for v in values[rep]) == _website_estimates(x, d, y, ps)
        for _ in range(REPLICATIONS - 2):
            _website_draw(generator)


def test_the_fit_reads_the_last_iterate_off_its_cycle() -> None:
    """Every step count gives the scores of that many plain Newton steps, bit for bit."""
    rng = np.random.default_rng(1)
    for strength, wrong in ((1.0, False), (1.0, True), (2.5, False), (0.5, False)):
        for _ in range(3):
            x, d, _ = draw_study(rng, 1500, strength=strength)
            features = x[:, :2] if wrong else x
            for steps in (0, 3, 13, 40):
                assert np.array_equal(
                    fit_propensity(features, d, steps), _website_fit(features, d, steps)
                )
            # The article's 60 steps reach the same scores as the figure's 40.
            assert fit_propensity(features, d, 60) == pytest.approx(
                fit_propensity(features, d), rel=1e-12
            )


def test_skipping_a_study_consumes_its_draws() -> None:
    """Skipped studies leave the generator where drawing them would."""
    drawn, skipped = np.random.default_rng(2), np.random.default_rng(2)
    for _ in range(3):
        draw_study(drawn, 500, strength=2.5, nonlinear=True)
    skip_studies(skipped, 3, 500)
    assert np.array_equal(draw_study(drawn, 50)[2], draw_study(skipped, 50)[2])


def test_the_naive_difference_is_off_the_scale() -> None:
    """The alt text: the unadjusted bias is 1.7 to 3.6, far off the axis, as its limits are."""
    for row in ROWS:
        bias = row.mean.naive - TRUE_EFFECT
        assert 1.6 < bias < 3.7
        assert bias > 0.36
        error = row.standard_deviation.naive / sqrt(row.replications)
        assert abs(row.mean.naive - row.naive_limit) < 4 * error
    limits = sorted(row.naive_limit - TRUE_EFFECT for row in ROWS)
    assert 1.65 < limits[0] < 1.75
    assert 3.55 < limits[-1] < 3.65


def test_regression_fails_only_when_the_outcome_model_is_wrong() -> None:
    """The alt text: regression adjustment is biased by 0.19 there and unbiased elsewhere."""
    for row in ROWS:
        z = _z(row, "regression")
        if row.condition.nonlinear:
            assert z < -8
            error = row.standard_deviation.regression / sqrt(row.replications)
            assert abs(row.mean.regression - row.regression_limit) < 4 * error
            assert row.regression_limit == pytest.approx(1.81, abs=0.005)
        else:
            assert abs(z) < 4
            assert row.regression_limit == pytest.approx(TRUE_EFFECT)


def test_matching_and_weighting_fail_when_the_propensity_model_is_wrong() -> None:
    """The alt text: both are biased by about a quarter when the score omits a confounder."""
    for row in ROWS:
        for field in ("matching", "weighting"):
            z = _z(row, field)
            if row.condition.wrong_propensity:
                assert z < -8
                assert getattr(row.mean, field) - TRUE_EFFECT == pytest.approx(-0.23, abs=0.03)
            elif row.condition.strength == 1.0:
                assert abs(z) < 4
    # Weighting also fails where overlap is poor, where the article reports a bias of 0.29.
    assert _z(ROWS[3], "weighting") > 4


def test_the_doubly_robust_estimator_sits_at_zero() -> None:
    """The alt text: no detectable bias in any condition."""
    for row in ROWS:
        assert abs(_z(row, "doubly_robust")) < 4
        assert row.absolute_bias.doubly_robust < 0.03


def test_the_closed_forms_match_a_large_sample() -> None:
    """400,000 units: naive differences, the misspecified regression, balance and overlap."""
    linear = draw_study(np.random.default_rng(3), 400_000)
    x, d, y = draw_study(np.random.default_rng(3), 400_000, nonlinear=True)
    assert np.array_equal(linear[1], d)
    assert naive_difference(*linear) == pytest.approx(naive_difference_limit(), abs=0.05)
    assert naive_difference(x, d, y) == pytest.approx(
        naive_difference_limit(nonlinear=True), abs=0.08
    )
    assert regression_adjustment(x, d, y) == pytest.approx(
        regression_limit(nonlinear=True), abs=0.04
    )
    treated, untreated = x[d == 1].mean(axis=0), x[d == 0].mean(axis=0)
    observed = (treated - untreated) / x.std(axis=0)
    assert observed == pytest.approx(standardised_difference(), abs=0.015)
    logit = x @ np.array(TREATMENT_COEFFICIENTS)
    score = 1 / (1 + np.exp(-logit))
    outside = float(np.mean((score < 0.1) | (score > 0.9)))
    assert outside == pytest.approx(share_outside(1.0), abs=0.002)


def test_the_closed_forms_by_hand() -> None:
    """Stein's identity, symmetry, a linear outcome and vanishing confounding."""
    beta = np.array(TREATMENT_COEFFICIENTS)
    spread = sqrt(float(beta @ beta))

    def slope(z: float) -> float:
        p = 1 / (1 + np.exp(-z))
        return float(p * (1 - p) * np.exp(-(z**2) / (2 * spread**2)))

    mean_slope = integrate.quad(slope, -40, 40)[0] / (spread * sqrt(2 * np.pi))
    # E[x sigma(beta x)] = beta E[sigma'(beta x)], with half the units treated.
    assert standardised_difference() == pytest.approx(tuple(4 * beta * mean_slope), rel=1e-9)
    assert naive_difference_limit() == pytest.approx(
        TRUE_EFFECT + 4 * mean_slope * (2.0 * 0.9 + 1.0 * 0.6 - 0.5 * 0.5), rel=1e-9
    )
    assert regression_limit(2.5) == pytest.approx(TRUE_EFFECT)
    assert naive_difference_limit(1e-6) == pytest.approx(TRUE_EFFECT, abs=1e-5)
    assert share_outside(0.0) == 0.0
    assert share_outside(1e6) == pytest.approx(1.0, abs=1e-5)


def test_the_article_overlap_and_balance_tables() -> None:
    """One study each: shares outside [0.1, 0.9] and standardised differences within noise."""
    printed_outside = {0.5: 0.000, 1.0: 0.057, 2.0: 0.372, 3.0: 0.549}
    for strength, value in printed_outside.items():
        expected = share_outside(strength)
        error = sqrt(max(expected * (1 - expected), 1e-6) / 4000)
        assert abs(value - expected) < 0.0005 + 3 * error
    before = (0.69, 0.46, -0.41)
    error = sqrt(4 / 20_000)
    for value, expected in zip(before, standardised_difference(), strict=True):
        assert abs(value - expected) < 0.005 + 3 * error
    # After weighting with the correct model the differences are zero up to noise.
    for value in (0.01, 0.01, -0.00):
        assert abs(value) < 0.005 + 3 * error


def test_the_misspecified_regression_interval_rarely_excludes_the_truth() -> None:
    """The first 40 misspecified studies: the 95 percent interval misses 2 in a minority."""
    rng = np.random.default_rng(0)
    skip_studies(rng, REPLICATIONS)
    misses, widths = [], []
    for _ in range(SHORT_REPLICATIONS):
        x, d, y = draw_study(rng, nonlinear=True)
        design = np.column_stack([np.ones(len(x)), d, x])
        coefficients, residual, *_ = np.linalg.lstsq(design, y, rcond=None)
        variance = float(residual[0]) / (len(y) - design.shape[1])
        error = sqrt(variance * float(np.linalg.inv(design.T @ design)[1, 1]))
        misses.append(abs(coefficients[1] - TRUE_EFFECT) > 1.959964 * error)
        widths.append(1.959964 * error)
    assert np.mean(misses) < 0.5
    assert TRUE_EFFECT - regression_limit(nonlinear=True) < float(np.mean(widths))


def test_the_public_estimators_match_the_bundle() -> None:
    """Each estimator on its own gives what estimate_all reports; trimming changes the sample."""
    x, d, y = draw_study(np.random.default_rng(4), 1500, strength=2.5)
    ps = np.clip(fit_propensity(x, d), *CLIP)
    bundle = estimate_all(x, d, y, ps)
    assert naive_difference(x, d, y) == bundle.naive
    assert regression_adjustment(x, d, y) == bundle.regression
    assert propensity_matching(x, d, y, ps) == bundle.matching
    assert inverse_probability_weighting(x, d, y, ps) == bundle.weighting
    assert doubly_robust(x, d, y, ps) == bundle.doubly_robust
    assert inverse_probability_weighting(x, d, y, ps, trim=0.05) != bundle.weighting


def test_a_randomised_study_needs_no_adjustment() -> None:
    """With a constant score every weighting estimator is the difference in means."""
    x, d, y = draw_study(np.random.default_rng(5), 800, strength=0.0)
    flat = np.full(len(d), 0.5)
    estimates = estimate_all(x, d, y, flat)
    assert estimates.weighting == pytest.approx(estimates.naive)
    assert fit_propensity(x, d, 0) == pytest.approx(flat)


def test_summarise_computes_the_table_columns() -> None:
    """Mean, spread, error and absolute bias of a hand-made table."""
    values = np.array([[1.0, 2.0, 3.0, 2.0, 2.5], [3.0, 2.0, 1.0, 2.0, 1.5]])
    row = summarise(CONDITIONS[0], values)
    assert row.replications == 2
    assert (row.mean.naive, row.mean.matching, row.mean.doubly_robust) == (2.0, 2.0, 2.0)
    assert row.standard_deviation.naive == pytest.approx(1.0)
    assert row.root_mean_squared_error.naive == pytest.approx(1.0)
    assert row.root_mean_squared_error.regression == 0.0
    assert row.absolute_bias.weighting == 0.0


def test_the_simulation_is_deterministic_under_a_seed() -> None:
    """The same seed reproduces a run; another seed does not."""
    first = condition_estimates(6, replications=2, units=300)
    again = condition_estimates(6, replications=2, units=300)
    other = condition_estimates(7, replications=2, units=300)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not np.array_equal(first[0], other[0])


def test_invalid_inputs_are_rejected() -> None:
    """Bad designs, mismatched arrays, scores outside (0, 1) and empty arms are refused."""
    rng = np.random.default_rng(0)
    x, d, y = draw_study(rng, 200)
    ps = np.clip(fit_propensity(x, d), *CLIP)
    with pytest.raises(ValueError):
        draw_study(rng, 1)
    with pytest.raises(ValueError):
        draw_study(rng, 100, strength=-1.0)
    with pytest.raises(ValueError):
        estimate_all(x, d, y[:-1], ps)
    with pytest.raises(ValueError):
        estimate_all(x, d + 1, y, ps)
    with pytest.raises(ValueError):
        estimate_all(x, d, y, np.ones_like(ps))
    with pytest.raises(ValueError):
        naive_difference(x, np.zeros_like(d), y)
    with pytest.raises(ValueError):
        inverse_probability_weighting(x, d, y, ps, trim=0.5)
    with pytest.raises(ValueError):
        fit_propensity(x, d[:-1])
    with pytest.raises(ValueError):
        condition_estimates(replications=2, estimated=3)
    with pytest.raises(ValueError):
        naive_difference_limit(0.0)
    with pytest.raises(ValueError):
        share_outside(1.0, band=0.5)
    with pytest.raises(ValueError):
        summarise(CONDITIONS[0], np.ones((3, 4)))
