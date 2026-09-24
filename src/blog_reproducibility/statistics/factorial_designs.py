"""One factor at a time against factorial designs, for the article comparing the two.

A process has four two-level factors in coded units (-1 at the baseline, +1
changed) and yield

    y = 10 + 2 A + 1.5 B + 0 C + 0.5 D + 2.5 AB + e,    e ~ N(0, 2^2).

The best settings raise A, B and D (C does nothing, so two of the sixteen
corners are best), lifting the mean yield from 8.5 at the baseline to 16.5.
The main effects, averaged over the other factors, are 4, 3, 0 and 1, and the
AB interaction effect is 5; but from the baseline, with the other factor at -1,
raising A changes the yield by ``4 - 5 = -1`` and raising B by ``3 - 5 = -2``.

One factor at a time runs the baseline and each single change ``n`` times and
keeps every change whose mean beat the baseline's. With ``s = sigma / sqrt(n)``
and the baseline's mean at ``m_0 + s z``, the changes are kept independently, so
the experiment ends at the best setting with probability

    P = integral phi(z) Phi(-1 / s - z) Phi(-2 / s - z) Phi(1 / s - z) dz,

which falls towards zero as ``n`` grows: more runs measure the wrong effects
more precisely. Its effect estimates have standard deviation ``sigma sqrt(2 /
n)``, against ``2 sigma / sqrt(N)`` for every effect of an ``N``-run two-level
factorial.

A factorial design is analysed by least squares on the main effects and the
two-factor interactions (in the eight-run half fraction with ``I = ABCD`` only
AB, AC and AD, each aliased with another pair) and ends at the corner with the
highest prediction. The large effects put A and B at +1 almost always; given
that, the half fraction raises D when ``b_D + b_AD > 0``, a normal of mean 0.5
and variance ``2 sigma^2 / 8``, so with probability ``Phi(0.5)``. The full
factorial of ``N`` runs raises D when ``v > -sign(u w) min(|u|, |w|)``, with
``u = b_C + b_AC + b_BC`` and ``v = b_D + b_AD + b_BD`` of variance ``3 sigma^2 / N``
and ``w = b_CD`` of variance ``sigma^2 / N``; the minimum has survival function
``4 Phi(-m / s_u) Phi(-m / s_w)``, which leaves a one-dimensional integral. A
or B ends up wrong in under 0.2 percent of half-fraction experiments and under
0.01 percent of full-factorial ones, so these are the chances of ending at the
best setting to that accuracy.

The figure is one computation from a generator seeded at 0: 1,500 experiments
one factor at a time for each budget of 10, 15, 25, 40, 60 and 100 runs (two to
twenty runs per setting), then 1,500 for the half fraction and for the full
factorial once, twice and four times, each experiment drawing its runs' noise
in turn. Drawing each budget's noise at once gives the website's draws, and
each experiment is still analysed with its own least-squares fit, so every
choice is the website's. The article's tables come from the same seed consumed
in a different order, with 4,000 experiments per row, and are not reproduced;
the tests compare them with the closed forms.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, pi, sqrt
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, special

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "BASELINE",
    "COEFFICIENTS",
    "CORNERS",
    "EXPERIMENTS",
    "FACTORIAL_RUNS",
    "FACTORS",
    "HALF_FRACTION",
    "INTERCEPT",
    "NOISE_SD",
    "OFAT_RUNS",
    "SEED",
    "ClosedForms",
    "DesignRow",
    "FactorialSummary",
    "closed_forms",
    "design_for_runs",
    "effect_standard_error",
    "ends_at_best",
    "example_payload",
    "factorial_choices",
    "factorial_d_probability",
    "interaction_detection_probability",
    "mean_yield",
    "model_columns",
    "ofat_choices",
    "ofat_effect_sd",
    "ofat_success_probability",
]

SEED: Final[int] = 0
FACTORS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
INTERCEPT: Final[float] = 10.0
COEFFICIENTS: Final[Mapping[str, float]] = MappingProxyType(
    {"A": 2.0, "B": 1.5, "C": 0.0, "D": 0.5, "AB": 2.5}
)
NOISE_SD: Final[float] = 2.0
EXPERIMENTS: Final[int] = 1500
# Runs one factor at a time (five settings, so two to twenty runs each), and factorial runs.
OFAT_RUNS: Final[tuple[int, ...]] = (10, 15, 25, 40, 60, 100)
FACTORIAL_RUNS: Final[tuple[int, ...]] = (8, 16, 32, 64)


def _read_only(array: NDArray[np.float64]) -> NDArray[np.float64]:
    array.setflags(write=False)
    return array


# The sixteen corners in the website's order, A slowest and D fastest.
CORNERS: Final[NDArray[np.float64]] = _read_only(
    np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)
)
# The 2^(4-1) half fraction with defining relation I = ABCD.
HALF_FRACTION: Final[NDArray[np.float64]] = _read_only(CORNERS[np.prod(CORNERS, axis=1) == 1])
BASELINE: Final[NDArray[np.float64]] = _read_only(-np.ones(4))


@dataclass(frozen=True, slots=True)
class DesignRow:
    """How often one design's experiments ended at the best setting, and why not."""

    design: str
    runs: int
    experiments: int
    successes: int
    share: float
    share_a_and_b_raised: float
    share_d_raised: float
    expected_share: float


@dataclass(frozen=True, slots=True)
class ClosedForms:
    """Population values behind the article's tables."""

    baseline_yield: float
    best_yield: float
    main_effects: tuple[float, ...]
    interaction_effect: float
    ofat_effects: tuple[float, ...]
    ofat_effect_sd: float
    factorial_effect_se: tuple[float, ...]
    half_fraction_effect_se: float
    ofat_runs_for_equal_precision: float
    interaction_detection: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FactorialSummary:
    """The figure's two curves and the closed forms."""

    ofat: tuple[DesignRow, ...]
    factorial: tuple[DesignRow, ...]
    closed_forms: ClosedForms


def _coefficients(coefficients: Mapping[str, float]) -> dict[str, float]:
    if set(coefficients) != {"A", "B", "C", "D", "AB"}:
        raise ValueError("coefficients must give A, B, C, D and AB")
    return {name: real(value, name=name) for name, value in coefficients.items()}


def mean_yield(
    settings: NDArray[np.float64], coefficients: Mapping[str, float] = COEFFICIENTS
) -> NDArray[np.float64]:
    """Mean yield at each row of coded settings, in the order A, B, C, D."""
    x = np.asarray(settings, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("settings must have one column per factor")
    beta = _coefficients(coefficients)
    a, b, c, d = x.T
    mean: NDArray[np.float64] = (
        INTERCEPT
        + beta["A"] * a
        + beta["B"] * b
        + beta["C"] * c
        + beta["D"] * d
        + beta["AB"] * a * b
    )
    return mean


def model_columns(settings: NDArray[np.float64], *, half: bool = False) -> NDArray[np.float64]:
    """Intercept, main effects and two-factor interactions (AB, AC, AD only for a half fraction)."""
    x = np.asarray(settings, dtype=np.float64)
    pairs = [(0, 1), (0, 2), (0, 3)] if half else list(itertools.combinations(range(4), 2))
    return np.column_stack(
        [np.ones(len(x))] + [x[:, i] for i in range(4)] + [x[:, i] * x[:, j] for i, j in pairs]
    )


def design_for_runs(runs: int) -> NDArray[np.float64]:
    """The half fraction for eight runs, else the full factorial replicated ``runs / 16`` times."""
    n = count(runs, name="runs", minimum=8)
    if n == 8:
        return HALF_FRACTION.copy()
    if n % 16:
        raise ValueError("runs must be eight or a multiple of sixteen")
    return np.vstack([CORNERS] * (n // 16))


def ofat_choices(
    rng: np.random.Generator,
    per_setting: int,
    experiments: int = EXPERIMENTS,
    *,
    sigma: float = NOISE_SD,
) -> NDArray[np.float64]:
    """Settings chosen by each one-factor-at-a-time experiment, one row per experiment.

    Each experiment runs the baseline and then each single change ``per_setting``
    times, in that order, and raises every factor whose change beat the baseline.
    """
    n = count(per_setting, name="per_setting", minimum=1)
    reps = count(experiments, name="experiments", minimum=1)
    settings = np.vstack([BASELINE, BASELINE + 2 * np.eye(4)])
    noise = rng.normal(0, positive(sigma, name="sigma"), (reps, 5, n))
    means = (mean_yield(settings)[None, :, None] + noise).mean(axis=-1)
    chosen = np.tile(BASELINE, (reps, 1))
    chosen[means[:, 1:] - means[:, :1] > 0] = 1
    return chosen


def factorial_choices(
    rng: np.random.Generator,
    design: NDArray[np.float64],
    experiments: int = EXPERIMENTS,
    *,
    sigma: float = NOISE_SD,
) -> NDArray[np.float64]:
    """Corner chosen by each factorial experiment: the best prediction of its least-squares fit."""
    runs = np.asarray(design, dtype=np.float64)
    if runs.ndim != 2 or runs.shape[1] != 4 or not np.all(np.abs(runs) == 1):
        raise ValueError("design must be rows of four coded settings of -1 or +1")
    reps = count(experiments, name="experiments", minimum=1)
    half = len(runs) % 16 != 0
    columns = model_columns(runs, half=half)
    corners = model_columns(CORNERS, half=half)
    noise = rng.normal(0, positive(sigma, name="sigma"), (reps, len(runs)))
    responses = mean_yield(runs) + noise
    chosen = np.empty((reps, 4))
    for i in range(reps):
        coefficients, *_ = np.linalg.lstsq(columns, responses[i], rcond=None)
        chosen[i] = CORNERS[(corners @ coefficients).argmax()]
    return chosen


def ends_at_best(chosen: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each chosen setting raises A, B and D (C does not matter)."""
    x = np.asarray(chosen, dtype=np.float64)
    best: NDArray[np.bool_] = (x[:, 0] == 1) & (x[:, 1] == 1) & (x[:, 3] == 1)
    return best


def ofat_effect_sd(per_setting: int, *, sigma: float = NOISE_SD) -> float:
    """Standard deviation of a one-at-a-time effect: ``sigma sqrt(2 / n)``."""
    return positive(sigma, name="sigma") * sqrt(
        2 / count(per_setting, name="per_setting", minimum=1)
    )


def effect_standard_error(runs: int, *, sigma: float = NOISE_SD) -> float:
    """Standard error of any effect of an ``N``-run two-level design: ``2 sigma / sqrt(N)``."""
    return 2 * positive(sigma, name="sigma") / sqrt(count(runs, name="runs", minimum=1))


def ofat_success_probability(
    per_setting: int,
    *,
    coefficients: Mapping[str, float] = COEFFICIENTS,
    sigma: float = NOISE_SD,
) -> float:
    """Chance one factor at a time raises A, B and D, integrating over the baseline's mean."""
    n = count(per_setting, name="per_setting", minimum=1)
    settings = np.vstack([BASELINE, BASELINE + 2 * np.eye(4)])
    means = mean_yield(settings, coefficients)
    s = positive(sigma, name="sigma") / sqrt(n)
    a, b, d = (float(means[k] - means[0]) / s for k in (1, 2, 4))

    def integrand(z: float) -> float:
        return _pdf(z) * _cdf(a - z) * _cdf(b - z) * _cdf(d - z)

    value, _ = integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-14, epsrel=1e-12, limit=200)
    return float(value)


def _pdf(z: float) -> float:
    return exp(-0.5 * z * z) / sqrt(2 * pi)


def _cdf(z: float) -> float:
    return float(special.ndtr(z))


def factorial_d_probability(
    runs: int, *, d_coefficient: float = COEFFICIENTS["D"], sigma: float = NOISE_SD
) -> float:
    """Chance a factorial design raises D, given that it raises A and B.

    C has no effect and no factor interacts with D, so the terms that decide D
    are centred on ``d_coefficient`` and the others on zero, as are their aliases.
    """
    n = count(runs, name="runs", minimum=8)
    design_for_runs(n)
    mean = real(d_coefficient, name="d_coefficient")
    variance = positive(sigma, name="sigma") ** 2 / n
    if n == 8:
        return _cdf(mean / sqrt(2 * variance))
    s_u = s_v = sqrt(3 * variance)
    s_w = sqrt(variance)

    def keep(m: float) -> float:
        return 0.5 * (_cdf((mean - m) / s_v) + _cdf((mean + m) / s_v))

    def integrand(m: float) -> float:
        slope = 0.5 * (_pdf((mean + m) / s_v) - _pdf((mean - m) / s_v)) / s_v
        return slope * 4 * _cdf(-m / s_u) * _cdf(-m / s_w)

    tail, _ = integrate.quad(integrand, 0, np.inf, epsabs=1e-14, epsrel=1e-12, limit=200)
    return keep(0.0) + float(tail)


def interaction_detection_probability(
    runs: int, *, coefficients: Mapping[str, float] = COEFFICIENTS, sigma: float = NOISE_SD
) -> float:
    """Chance the AB estimate is more than two standard errors from zero."""
    effect = 2 * _coefficients(coefficients)["AB"] / effect_standard_error(runs, sigma=sigma)
    return _cdf(effect - 2) + _cdf(-2 - effect)


def _row(design: str, runs: int, chosen: NDArray[np.float64], expected: float) -> DesignRow:
    hits = ends_at_best(chosen)
    return DesignRow(
        design=design,
        runs=runs,
        experiments=int(hits.size),
        successes=int(hits.sum()),
        share=float(hits.mean()),
        share_a_and_b_raised=float(np.mean((chosen[:, 0] == 1) & (chosen[:, 1] == 1))),
        share_d_raised=float(np.mean(chosen[:, 3] == 1)),
        expected_share=expected,
    )


def closed_forms(*, coefficients: Mapping[str, float] = COEFFICIENTS) -> ClosedForms:
    """The article's population values: yields, effects and their precision."""
    beta = _coefficients(coefficients)
    best = np.array([[1.0, 1.0, 1.0, 1.0]])
    settings = np.vstack([BASELINE, BASELINE + 2 * np.eye(4)])
    means = mean_yield(settings, beta)
    return ClosedForms(
        baseline_yield=float(mean_yield(BASELINE[None], beta)[0]),
        best_yield=float(mean_yield(best, beta)[0]),
        main_effects=tuple(2 * beta[f] for f in FACTORS),
        interaction_effect=2 * beta["AB"],
        ofat_effects=tuple(float(m - means[0]) for m in means[1:]),
        ofat_effect_sd=ofat_effect_sd(3),
        factorial_effect_se=tuple(effect_standard_error(n) for n in (16, 32, 64)),
        half_fraction_effect_se=effect_standard_error(8),
        # N runs one at a time give sigma sqrt(10 / N); a factorial gives sigma sqrt(4 / N).
        ofat_runs_for_equal_precision=5 * ofat_effect_sd(1) ** 2 / effect_standard_error(1) ** 2,
        interaction_detection=tuple(interaction_detection_probability(n) for n in (16, 32, 64)),
    )


def example_payload() -> FactorialSummary:
    """Run the figure's experiments in its order and put each share beside its closed form."""
    rng = np.random.default_rng(SEED)
    ofat = tuple(
        _row(
            "one factor at a time",
            runs,
            ofat_choices(rng, runs // 5),
            ofat_success_probability(runs // 5),
        )
        for runs in OFAT_RUNS
    )
    factorial = tuple(
        _row(
            "half fraction" if runs == 8 else "full factorial",
            runs,
            factorial_choices(rng, design_for_runs(runs)),
            factorial_d_probability(runs),
        )
        for runs in FACTORIAL_RUNS
    )
    return FactorialSummary(ofat=ofat, factorial=factorial, closed_forms=closed_forms())
