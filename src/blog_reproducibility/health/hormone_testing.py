"""Panel-size and reference-change arithmetic for the hormone-testing article.

Two calculations sit behind the article. The first is the chance that a healthy
person has at least one result outside a 95% reference interval when several
analytes are measured at once, with and without correlation between them. The
second is the reference change value: how far two results from the same person
must differ before biological variation alone stops explaining the difference.

The reference change value is ``z * sqrt(2) * sqrt(CV_within^2 + CV_analytical^2)``
in percent. The ``sqrt(2)`` is there because two measurements each carry the
variation, and it is why the thresholds are so large.

Sources are named beside each constant.
"""

from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import NormalDist
from typing import Final

from blog_reproducibility.common.validation import count, non_negative, probability, real

__all__ = [
    "ANCKAERT_PROGESTERONE",
    "CASALS",
    "DANESE_RCV",
    "NAUGLER",
    "STANCZYK",
    "WITHIN_SUBJECT_CV",
    "Z",
    "HormoneSummary",
    "any_flag",
    "any_flag_correlated",
    "example_payload",
    "reference_change_value",
]

# Within-subject biological variation, CV in %, median of the studies in the EFLM
# Biological Variation Database (https://biologicalvariation.eu, read 2026-09-19);
# serum or plasma, healthy adults.
WITHIN_SUBJECT_CV: Final[dict[str, float]] = {
    "free T4": 4.8,
    "SHBG": 8.3,
    "FSH": 9.5,
    "testosterone": 14.3,
    "estradiol": 15.0,
    "cortisol": 16.2,
    "TSH": 17.8,
    "progesterone": 18.5,
    "DHEAS": 20.0,
    "LH": 25.2,
    "insulin": 25.4,
    "prolactin": 45.0,
    "free testosterone": 55.0,
}
# Casals G, et al. Clin Biochem 2011;44:665-668. Late-night salivary cortisol:
# analytical CV, within-subject CV, and the reference change value the authors report.
CASALS: Final[dict[str, float]] = {
    "analytical": 15.4,
    "within subject": 34.1,
    "published RCV": 104.0,
}
# Danese E, et al. Clin Chem Lab Med 2024;62:2287-2293. Salivary cortisol,
# reference change value by time of day, %.
DANESE_RCV: Final[tuple[int, int]] = (96, 245)
# Naugler C, Ma I. Can Fam Physician 2018;64:202-203. Mean abnormal result rate
# across 1,340 family physicians, %.
NAUGLER: Final[dict[str, float]] = {"observed abnormal": 8.6, "expected in the healthy": 5.0}
# Stanczyk FZ, et al. Menopause 2019;26:966-971. Measured content against the label.
STANCZYK: Final[dict[str, tuple[float, float, float]]] = {
    "estradiol capsules, mg": (0.365, 0.551, 0.5),
    "progesterone capsules, mg": (90.8, 135.0, 100.0),
}
# Anckaert E, et al. Pract Lab Med 2021;25:e00211. Median serum progesterone, nmol/L.
ANCKAERT_PROGESTERONE: Final[dict[str, float]] = {"follicular": 0.212, "luteal": 28.8}

Z: Final[float] = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True, slots=True)
class HormoneSummary:
    """Every number the hormone article reports."""

    chance_of_a_flag_percent: dict[int, float]
    expected_flags_in_a_panel_of_12: float
    chance_of_a_flag_correlated_percent: float
    false_positive_share_percent: int
    reference_change_values_percent: dict[str, int]
    salivary_cortisol_rcv_percent: float
    estradiol_capsule_against_label_percent: tuple[int, int]
    progesterone_capsule_against_label_percent: tuple[int, int]
    progesterone_luteal_over_follicular: int


def any_flag(analytes: int, coverage: float = 0.95) -> float:
    """Chance that at least one of independent results falls outside its interval."""
    number = count(analytes, name="analytes", minimum=0)
    inside = probability(coverage, name="coverage")
    return 1.0 - inside**number


def any_flag_correlated(
    analytes: int,
    correlation: float,
    *,
    repeats: int = 40_000,
    seed: int = 20260919,
) -> float:
    """The same chance when every pair of analytes shares the given correlation.

    Each analyte is a shared standard normal component scaled by
    ``sqrt(correlation)`` plus its own scaled by ``sqrt(1 - correlation)``, which
    gives an equicorrelated panel. The generator is built from ``seed`` here, so
    the estimate repeats and never depends on global random state.
    """
    number = count(analytes, name="analytes", minimum=0)
    shared_variance = probability(correlation, name="correlation")
    draws = count(repeats, name="repeats", minimum=1)
    rng = Random(count(seed, name="seed"))

    shared = sqrt(shared_variance)
    own = sqrt(1.0 - shared_variance)
    hits = 0
    for _ in range(draws):
        common = rng.gauss(0.0, 1.0)
        if any(abs(shared * common + own * rng.gauss(0.0, 1.0)) > Z for _ in range(number)):
            hits += 1
    return hits / draws


def reference_change_value(within_subject_cv: float, analytical_cv: float = 0.0) -> float:
    """Smallest difference between two results, in %, that variation alone rarely gives."""
    biological = non_negative(within_subject_cv, name="within_subject_cv")
    analytical = non_negative(analytical_cv, name="analytical_cv")
    return Z * sqrt(2.0) * sqrt(biological**2 + analytical**2)


def _against_label(measured: tuple[float, float, float]) -> tuple[int, int]:
    """Return the low and high measured content as a percentage of the label."""
    low, high, label = measured
    return (round(100 * (low / label - 1)), round(100 * (high / label - 1)))


def example_payload() -> HormoneSummary:
    """Return the numbers the article reports."""
    observed = NAUGLER["observed abnormal"]
    expected = NAUGLER["expected in the healthy"]

    return HormoneSummary(
        chance_of_a_flag_percent={
            number: round(100 * any_flag(number), 1) for number in (1, 5, 12, 20, 40)
        },
        expected_flags_in_a_panel_of_12=round(0.05 * 12, 1),
        chance_of_a_flag_correlated_percent=round(100 * any_flag_correlated(12, 0.5), 1),
        false_positive_share_percent=round(100 * expected / observed),
        reference_change_values_percent={
            analyte: round(reference_change_value(cv)) for analyte, cv in WITHIN_SUBJECT_CV.items()
        },
        salivary_cortisol_rcv_percent=round(
            reference_change_value(CASALS["within subject"], CASALS["analytical"]), 1
        ),
        estradiol_capsule_against_label_percent=_against_label(STANCZYK["estradiol capsules, mg"]),
        progesterone_capsule_against_label_percent=_against_label(
            STANCZYK["progesterone capsules, mg"]
        ),
        progesterone_luteal_over_follicular=round(
            ANCKAERT_PROGESTERONE["luteal"] / ANCKAERT_PROGESTERONE["follicular"]
        ),
    )


def flag_curve(largest_panel: int = 40) -> tuple[tuple[int, float], ...]:
    """Return the independent-analyte curve drawn by the panel figure."""
    size = count(largest_panel, name="largest_panel", minimum=1)
    return tuple((number, 100 * any_flag(number)) for number in range(1, size + 1))


def sorted_reference_changes() -> tuple[tuple[str, float], ...]:
    """Return the analytes ordered by their reference change value."""
    values = {name: reference_change_value(cv) for name, cv in WITHIN_SUBJECT_CV.items()}
    return tuple(sorted(values.items(), key=lambda item: real(item[1], name="rcv")))
