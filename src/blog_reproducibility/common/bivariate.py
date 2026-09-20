"""Tail agreement between two correlated standard normal measurements.

Two articles ask the same question of different data. The microbiome article
asks how often a taxon flagged as high in one stool sample is flagged again in
the next. The leaky-gut article asks how often someone whose surrogate test is
in its top fifth really is in the top fifth of measured permeability. Both are
the same integral over a standard bivariate normal pair, so it lives here once.
"""

from math import sqrt
from statistics import NormalDist
from typing import Final

from blog_reproducibility.common.validation import positive, probability

__all__ = ["STANDARD", "conditional_top_share"]

STANDARD: Final[NormalDist] = NormalDist()


def conditional_top_share(
    correlation: float,
    *,
    top: float = 0.05,
    step: float = 0.001,
) -> float:
    """P(second value in its top share | first value in its top share).

    The pair is standard bivariate normal with the given correlation. The
    integral is evaluated on a fixed midpoint grid out to eight standard
    deviations, so the result is deterministic and the accuracy is set by
    ``step`` rather than by a sampler.

    At zero correlation the answer is ``top`` itself, because the second value
    knows nothing about the first; at correlation one it is certain.
    """
    rho = probability(correlation, name="correlation")
    share = probability(top, name="top", inclusive=False)
    spacing = positive(step, name="step")

    if rho >= 1.0:
        return 1.0

    cut = STANDARD.inv_cdf(1.0 - share)
    spread = sqrt(1.0 - rho**2)

    total = 0.0
    position = cut
    while position < 8.0:
        middle = position + spacing / 2.0
        above = 1.0 - STANDARD.cdf((cut - rho * middle) / spread)
        total += STANDARD.pdf(middle) * above * spacing
        position += spacing
    return total / share
