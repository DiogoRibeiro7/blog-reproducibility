"""Sample ratio mismatch and the bias it signals, for the article on the SRM check.

Under correct 50/50 assignment the treated count is binomial, so the observed
split can be tested against the intended one with a one-degree-of-freedom
chi-square goodness-of-fit test; with ``D`` the difference between the arm
sizes and ``N`` their sum, the statistic is ``D^2 / N``. At a threshold ``alpha``
it fires on deviations from an even split larger than
``z_(1 - alpha / 2) * 0.5 / sqrt(N)``.

The article's experiment has a million users, a ten percent baseline and a true
lift of one percent. A fifth of users are slow and convert at half the rate of
the rest, and a share ``d`` of all treated users, taken only from the slow ones,
is lost before logging. Because the lost users convert less than average, the
measured lift tends to

    (1 + lift) (1 - d r) / (1 - d) - 1,

where ``r`` is the slow users' conversion relative to the average, and the
logged treated share to ``(1 - d) / (2 - d)``. The check sees ``D`` with mean
``-d n / 2`` and standard deviation close to ``sqrt(n)``, which gives its
detection probability. The measured lift also has sampling error, about 0.6 of
the true lift at a million users. The figure plots the bias, the mean signed
error ``(measured - lift) / lift`` over its experiments, in which that sampling
error averages out; a mean absolute error would keep it, and would sit near half
the effect even when nothing is lost.

The figure runs 60 experiments at each of six drop shares from one generator
seeded at 0, and is reproduced draw for draw. The article's tables run their
own simulations (200 replications, other drop shares, an A/A check first) from
a generator used for other draws before them, and are not reproduced; its
closed-form table of detectable deviations and the arithmetic of its opening
example are.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
from scipy import stats

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "ALARM_THRESHOLD",
    "BASE_RATE",
    "DEFAULT_DESIGN",
    "DETECTION_SIZES",
    "DROP_SHARES",
    "REPLICATIONS",
    "SEED",
    "SLOW_RATIO",
    "SLOW_SHARE",
    "TRUE_LIFT",
    "USERS",
    "ArticleNumbers",
    "Design",
    "DetectableDeviation",
    "DropRow",
    "DropTheory",
    "ExperimentOutcome",
    "MismatchSummary",
    "alarm_probability",
    "article_numbers",
    "detectable_deviation",
    "drop_theory",
    "example_payload",
    "expected_lift",
    "expected_treated_share",
    "lift_standard_error",
    "run_experiment",
    "simulate_drops",
    "srm_p_value",
]

SEED: Final[int] = 0
USERS: Final[int] = 1_000_000
TRUE_LIFT: Final[float] = 0.01
BASE_RATE: Final[float] = 0.10
SLOW_SHARE: Final[float] = 0.20
SLOW_RATIO: Final[float] = 0.5
ALARM_THRESHOLD: Final[float] = 0.001
DROP_SHARES: Final[tuple[float, ...]] = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02)
REPLICATIONS: Final[int] = 60
DETECTION_SIZES: Final[tuple[int, ...]] = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
# The article's opening example: a million users logged 502,470 to 497,530.
OPENING_CONTROL: Final[int] = 502_470
OPENING_TREATMENT: Final[int] = 497_530


@dataclass(frozen=True, slots=True)
class Design:
    """The simulated experiment: its size, true lift, baseline and slow segment."""

    users: int = USERS
    lift: float = TRUE_LIFT
    base_rate: float = BASE_RATE
    slow_share: float = SLOW_SHARE
    slow_ratio: float = SLOW_RATIO

    def __post_init__(self) -> None:
        count(self.users, name="users", minimum=2)
        positive(self.lift, name="lift")
        probability(self.base_rate, name="base_rate", inclusive=False)
        probability(self.slow_share, name="slow_share", inclusive=False)
        positive(self.slow_ratio, name="slow_ratio")
        rates = self.segment_rates()
        if min(rates) <= 0 or max(rates) * (1 + self.lift) > 1:
            raise ValueError("every segment's conversion rate must lie in (0, 1] when treated")

    def segment_rates(self) -> tuple[float, float]:
        """Conversion rates of slow and of other users, averaging to the base rate."""
        share, ratio = self.slow_share, self.slow_ratio
        return (
            self.base_rate * ratio,
            self.base_rate * (1 + share * (1 - ratio) / (1 - share)),
        )


DEFAULT_DESIGN: Final[Design] = Design()


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """What one logged experiment shows: the lift, the treated share and the SRM p-value."""

    measured_lift: float
    treated_share: float
    p_value: float


@dataclass(frozen=True, slots=True)
class DropRow:
    """Simulated experiments at one drop share: the figure's two series and two means."""

    drop: float
    relative_bias: float
    alarm_rate: float
    mean_lift: float
    mean_treated_share: float


@dataclass(frozen=True, slots=True)
class DropTheory:
    """Large-sample expectations at one drop share."""

    drop: float
    expected_lift: float
    relative_bias: float
    treated_share: float
    lift_standard_error: float
    alarm_probability: float


@dataclass(frozen=True, slots=True)
class DetectableDeviation:
    """Smallest deviation from an even split the check flags at one sample size."""

    users: int
    deviation: float
    users_off_even: float


@dataclass(frozen=True, slots=True)
class ArticleNumbers:
    """The opening example's arithmetic and the table of detectable deviations."""

    arm_size_sd: float
    shortfall: int
    shortfall_in_sd: float
    shortfall_p_value: float
    detectable: tuple[DetectableDeviation, ...]


@dataclass(frozen=True, slots=True)
class MismatchSummary:
    """The figure's simulation, its large-sample expectations and the article's numbers."""

    rows: tuple[DropRow, ...]
    theory: tuple[DropTheory, ...]
    article: ArticleNumbers


def srm_p_value(control: int, treatment: int, *, treated_share: float = 0.5) -> float:
    """Chi-square goodness-of-fit p-value of the observed split against the intended one."""
    a = count(control, name="control")
    b = count(treatment, name="treatment")
    share = probability(treated_share, name="treated_share", inclusive=False)
    total = a + b
    if total == 0:
        raise ValueError("the experiment must have at least one user")
    expected_a, expected_b = total * (1 - share), total * share
    statistic = (a - expected_a) ** 2 / expected_a + (b - expected_b) ** 2 / expected_b
    return float(stats.chi2.sf(statistic, 1))


def run_experiment(
    rng: np.random.Generator, drop: float, design: Design = DEFAULT_DESIGN
) -> ExperimentOutcome:
    """Assign, segment and convert users, then lose a share of treated slow users.

    The draws are the article's: arm, slow flag and conversion uniforms for every
    user, then, only when something is dropped, one more uniform per user.
    """
    share = probability(drop, name="drop")
    n = design.users
    treated = rng.integers(0, 2, n) == 1
    slow = rng.random(n) < design.slow_share
    slow_rate, fast_rate = design.segment_rates()
    thresholds = np.where(slow, slow_rate, fast_rate)
    thresholds *= np.where(treated, 1 + design.lift, 1.0)
    converted = rng.random(n) < thresholds
    kept_treated = treated
    if share > 0:
        lost = treated & slow
        lost &= rng.random(n) < min(1.0, share / design.slow_share)
        kept_treated = treated & ~lost
    control = ~treated

    n_treated = int(np.count_nonzero(kept_treated))
    n_control = int(np.count_nonzero(control))
    conversions_treated = int(np.count_nonzero(converted & kept_treated))
    conversions_control = int(np.count_nonzero(converted & control))
    if n_treated == 0 or conversions_control == 0:
        raise ValueError("both arms need users and the control arm a conversion")
    measured = (conversions_treated / n_treated) / (conversions_control / n_control) - 1
    return ExperimentOutcome(
        measured_lift=measured,
        treated_share=n_treated / (n_treated + n_control),
        p_value=srm_p_value(n_control, n_treated),
    )


def simulate_drops(
    seed: int = SEED,
    *,
    drops: tuple[float, ...] = DROP_SHARES,
    design: Design = DEFAULT_DESIGN,
    replications: int = REPLICATIONS,
    threshold: float = ALARM_THRESHOLD,
) -> tuple[DropRow, ...]:
    """Run the figure's experiments for each drop share in turn from one generator."""
    rng = np.random.default_rng(count(seed, name="seed"))
    reps = count(replications, name="replications", minimum=1)
    alpha = probability(threshold, name="threshold", inclusive=False)
    shares = tuple(probability(drop, name="drop") for drop in drops)

    rows = []
    for drop in shares:
        outcomes = [run_experiment(rng, drop, design) for _ in range(reps)]
        errors = np.array([(o.measured_lift - design.lift) / design.lift for o in outcomes])
        alarms = np.array([o.p_value < alpha for o in outcomes], dtype=np.float64)
        rows.append(
            DropRow(
                drop=drop,
                relative_bias=float(errors.mean()),
                alarm_rate=float(alarms.mean()),
                mean_lift=float(np.mean([o.measured_lift for o in outcomes])),
                mean_treated_share=float(np.mean([o.treated_share for o in outcomes])),
            )
        )
    return tuple(rows)


def _lost_share(drop: float, design: Design) -> float:
    """Share of treated users lost: the drop, capped at the slow segment it comes from."""
    return min(probability(drop, name="drop"), design.slow_share)


def expected_lift(drop: float, design: Design = DEFAULT_DESIGN) -> float:
    """Large-sample measured lift when a share of treated slow users is lost."""
    lost = _lost_share(drop, design)
    return (1 + design.lift) * (1 - lost * design.slow_ratio) / (1 - lost) - 1


def expected_treated_share(drop: float, design: Design = DEFAULT_DESIGN) -> float:
    """Large-sample share of logged users who are treated."""
    lost = _lost_share(drop, design)
    return (1 - lost) / (2 - lost)


def lift_standard_error(drop: float, design: Design = DEFAULT_DESIGN) -> float:
    """Delta-method standard error of the measured lift, a ratio of two proportions."""
    lost = _lost_share(drop, design)
    control_rate = design.base_rate
    treated_rate = control_rate * (1 + expected_lift(drop, design))
    n_control = design.users / 2
    n_treated = design.users * (1 - lost) / 2
    relative_variance = (1 - treated_rate) / (n_treated * treated_rate) + (1 - control_rate) / (
        n_control * control_rate
    )
    return treated_rate / control_rate * sqrt(relative_variance)


def alarm_probability(
    drop: float, design: Design = DEFAULT_DESIGN, *, threshold: float = ALARM_THRESHOLD
) -> float:
    """Normal approximation to the chance that the check fires.

    Each assigned user adds one to ``D`` with probability ``(1 - d) / 2``,
    subtracts one with probability ``1 / 2`` and is lost otherwise, so ``D`` has
    mean ``-d n / 2`` and variance ``n (1 - d / 2 - d^2 / 4)``; the statistic is
    ``D^2 / N`` with ``N`` close to ``n (1 - d / 2)``.
    """
    lost = _lost_share(drop, design)
    critical = float(stats.norm.isf(probability(threshold, name="threshold", inclusive=False) / 2))
    kept = 1 - lost / 2
    centre = -(lost / 2) * sqrt(design.users / kept)
    spread = sqrt((kept - lost**2 / 4) / kept)
    return float(
        stats.norm.cdf((-critical - centre) / spread) + stats.norm.sf((critical - centre) / spread)
    )


def drop_theory(drop: float, design: Design = DEFAULT_DESIGN) -> DropTheory:
    """Every large-sample expectation at one drop share."""
    lift = expected_lift(drop, design)
    return DropTheory(
        drop=drop,
        expected_lift=lift,
        relative_bias=(lift - design.lift) / design.lift,
        treated_share=expected_treated_share(drop, design),
        lift_standard_error=lift_standard_error(drop, design),
        alarm_probability=alarm_probability(drop, design),
    )


def detectable_deviation(users: int, threshold: float = ALARM_THRESHOLD) -> float:
    """Smallest deviation of the treated share from one half that the check flags."""
    n = count(users, name="users", minimum=1)
    alpha = probability(threshold, name="threshold", inclusive=False)
    return float(stats.norm.ppf(1 - alpha / 2)) * 0.5 / sqrt(n)


def article_numbers() -> ArticleNumbers:
    """The opening example's arithmetic and the article's detectable-deviation table."""
    total = OPENING_CONTROL + OPENING_TREATMENT
    arm_sd = sqrt(total * 0.5 * 0.5)
    shortfall = total // 2 - OPENING_TREATMENT
    return ArticleNumbers(
        arm_size_sd=arm_sd,
        shortfall=shortfall,
        shortfall_in_sd=shortfall / arm_sd,
        shortfall_p_value=srm_p_value(OPENING_CONTROL, OPENING_TREATMENT),
        detectable=tuple(
            DetectableDeviation(n, detectable_deviation(n), n * detectable_deviation(n))
            for n in DETECTION_SIZES
        ),
    )


def example_payload() -> MismatchSummary:
    """Return the figure's simulation, its expectations and the article's closed forms."""
    return MismatchSummary(
        rows=simulate_drops(),
        theory=tuple(drop_theory(drop) for drop in DROP_SHARES),
        article=article_numbers(),
    )
