"""Random-walk Metropolis sampling, for the article on Markov chain Monte Carlo.

Each step proposes ``x' = x + N(0, step^2)`` and accepts it with probability
``min(1, p(x') / p(x))``. Only the ratio enters, so the target needs no
normalising constant, which is the point of the method. The article runs four
chains from dispersed starting points on a normal target with mean 2 and
standard deviation 0.8, discards the first 1,000 draws of each, and pools the
rest.

Agreement between chains is measured by the Gelman-Rubin statistic, which
compares between-chain and within-chain variance; values near one mean the
chains are exploring the same distribution.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import log
from typing import Final

import numpy as np
from numpy.typing import NDArray

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "BURN_IN",
    "CHAINS",
    "SEED",
    "STEPS",
    "TARGET_MEAN",
    "TARGET_SD",
    "ChainSummary",
    "example_payload",
    "gelman_rubin",
    "metropolis_chain",
    "run_chains",
    "target_log_density",
]

SEED: Final[int] = 20260816
TARGET_MEAN: Final[float] = 2.0
TARGET_SD: Final[float] = 0.8
CHAINS: Final[int] = 4
STEPS: Final[int] = 4000
BURN_IN: Final[int] = 1000
PROPOSAL_SD: Final[float] = 0.9
START_SD: Final[float] = 3.0


@dataclass(frozen=True, slots=True)
class ChainSummary:
    """Pooled posterior moments, acceptance, and convergence of the four chains."""

    pooled_mean: float
    pooled_sd: float
    acceptance_rate: float
    r_hat: float


def target_log_density(value: float) -> float:
    """Unnormalised log density of ``N(2, 0.8^2)``."""
    return -0.5 * ((value - TARGET_MEAN) / TARGET_SD) ** 2


def metropolis_chain(
    log_density: Callable[[float], float],
    start: float,
    *,
    steps: int,
    proposal_sd: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], int]:
    """Run one chain, returning its draws and the number of accepted proposals.

    A uniform is drawn at every step, even when acceptance is certain, so the
    random stream does not depend on the target.
    """
    current = real(start, name="start")
    scale = positive(proposal_sd, name="proposal_sd")
    current_density = log_density(current)
    draws = np.empty(count(steps, name="steps", minimum=1))
    accepted = 0
    for index in range(draws.size):
        proposal = current + rng.normal(0, scale)
        proposal_density = log_density(proposal)
        uniform = rng.random()
        if uniform == 0 or log(uniform) < proposal_density - current_density:
            current, current_density = proposal, proposal_density
            accepted += 1
        draws[index] = current
    return draws, accepted


def run_chains(*, seed: int = SEED) -> tuple[tuple[NDArray[np.float64], ...], float]:
    """Run the article's four chains and return them with the acceptance rate."""
    rng = np.random.default_rng(count(seed, name="seed"))
    chains, accepted = [], 0
    for _ in range(CHAINS):
        start = float(rng.normal(TARGET_MEAN, START_SD))
        draws, hits = metropolis_chain(
            target_log_density, start, steps=STEPS, proposal_sd=PROPOSAL_SD, rng=rng
        )
        chains.append(draws)
        accepted += hits
    return tuple(chains), accepted / (CHAINS * STEPS)


def gelman_rubin(chains: tuple[NDArray[np.float64], ...]) -> float:
    """Potential scale reduction factor of equal-length chains."""
    if len(chains) < 2 or len({chain.size for chain in chains}) != 1 or chains[0].size < 2:
        raise ValueError("need at least two equal-length chains of two or more draws")
    stacked = np.vstack(chains)
    length = stacked.shape[1]
    within = float(stacked.var(axis=1, ddof=1).mean())
    between = float(length * stacked.mean(axis=1).var(ddof=1))
    pooled = (length - 1) / length * within + between / length
    return float(np.sqrt(pooled / within))


def example_payload() -> ChainSummary:
    """Return the pooled posterior and convergence diagnostics after burn-in."""
    chains, acceptance = run_chains()
    kept = tuple(chain[BURN_IN:] for chain in chains)
    pooled = np.concatenate(kept)
    return ChainSummary(
        pooled_mean=float(pooled.mean()),
        pooled_sd=float(pooled.std()),
        acceptance_rate=acceptance,
        r_hat=gelman_rubin(kept),
    )
