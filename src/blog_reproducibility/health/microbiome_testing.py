"""Closure, flag counts, and repeatability for the microbiome-testing article.

Three calculations, all exact except the last, which is a one-dimensional
integral evaluated on a fixed grid.

**Closure.** A report gives shares, not counts. When one taxon's absolute count
changes and nothing else does, every other share moves, because they must still
sum to one. If the taxon that grew held share ``s`` and its count multiplied by
``k``, every other share is multiplied by ``1 / (1 + s(k - 1))``.

**Flags.** With a 95% range per taxon, five taxa in every hundred fall outside
in a healthy person by construction.

**Repeatability.** Given the intraclass correlation of a taxon within one
person, two samples are bivariate normal with that correlation, and the chance
that a value in the top 5% is in the top 5% again follows by integration.

Sources are named beside each constant.
"""

from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

from blog_reproducibility.common.bivariate import conditional_top_share
from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "BERRY",
    "SERVETAS",
    "STANDARD",
    "VANDEPUTTE",
    "MicrobiomeSummary",
    "any_flag",
    "apparent_change",
    "example_payload",
    "flagged_again",
]

STANDARD: Final[NormalDist] = NormalDist()

# Vandeputte D, et al. Nat Commun 2021;12:6740. Twenty women sampled daily for six
# weeks. Share of genera whose abundance varied more within a person than between
# people, meaning an intraclass correlation below 0.5.
VANDEPUTTE: Final[dict[str, int]] = {"absolute abundance, %": 78, "relative abundance, %": 36}
# Servetas SL, et al. Commun Biol 2026;9:269. One homogenised stool sample, three
# kits sent to each of seven companies.
SERVETAS: Final[dict[str, int]] = {
    "companies": 7,
    "unique taxa": 1208,
    "genera found by all": 3,
    "fewest genera": 34,
    "most genera": 906,
}
# Berry SE, et al. Nat Med 2020;26:964-973. Variance in the glucose response after
# identical meals, % explained.
BERRY: Final[dict[str, float]] = {"meal macronutrients": 15.4, "gut microbiome": 6.0}


@dataclass(frozen=True, slots=True)
class MicrobiomeSummary:
    """Every number the microbiome article reports."""

    other_shares_when_a_30_percent_taxon_doubles: int
    other_shares_when_a_30_percent_taxon_rises_tenfold: int
    other_shares_when_a_5_percent_taxon_rises_tenfold: int
    expected_flags_among_100_taxa: int
    chance_of_a_flag_among_100_taxa_percent: float
    expected_flags_with_interquartile_ranges: int
    flagged_again_percent: dict[str, float]
    microbiome_over_macronutrients: float


def apparent_change(share: float, fold: float) -> float:
    """Relative change in every other taxon's share when one taxon's count grows.

    ``share`` is what the growing taxon held before, and ``fold`` is what its
    absolute count multiplied by. The result is negative for growth: nothing
    happened to the other taxa, and their shares fall anyway.
    """
    held = probability(share, name="share")
    multiplier = positive(fold, name="fold")
    return 1.0 / (1.0 + held * (multiplier - 1.0)) - 1.0


def any_flag(taxa: int, coverage: float = 0.95) -> float:
    """Chance that at least one of independent taxa falls outside its range."""
    number = count(taxa, name="taxa", minimum=0)
    inside = probability(coverage, name="coverage")
    return 1.0 - inside**number


def flagged_again(reliability: float, *, top: float = 0.05, step: float = 0.001) -> float:
    """Chance a taxon in its top share repeats that in a second sample.

    Two samples from one person are bivariate standard normal with correlation
    ``reliability``, so this is the shared tail-agreement integral.
    """
    return conditional_top_share(reliability, top=top, step=step)


def example_payload() -> MicrobiomeSummary:
    """Return the numbers the article reports."""
    return MicrobiomeSummary(
        other_shares_when_a_30_percent_taxon_doubles=round(100 * apparent_change(0.30, 2)),
        other_shares_when_a_30_percent_taxon_rises_tenfold=round(100 * apparent_change(0.30, 10)),
        other_shares_when_a_5_percent_taxon_rises_tenfold=round(100 * apparent_change(0.05, 10)),
        expected_flags_among_100_taxa=round(0.05 * 100),
        chance_of_a_flag_among_100_taxa_percent=round(100 * any_flag(100), 1),
        expected_flags_with_interquartile_ranges=round(0.5 * 100),
        flagged_again_percent={
            str(icc): round(100 * flagged_again(icc), 1) for icc in (0.3, 0.5, 0.7, 0.9)
        },
        microbiome_over_macronutrients=round(
            BERRY["gut microbiome"] / BERRY["meal macronutrients"], 2
        ),
    )


def closure_curve(share: float, points: int = 51) -> tuple[tuple[float, float], ...]:
    """Return the fold change and apparent change pairs drawn by the closure figure."""
    size = count(points, name="points", minimum=2)
    folds = [10 ** (index / (size - 1)) for index in range(size)]
    return tuple((fold, 100 * apparent_change(share, fold)) for fold in folds)
