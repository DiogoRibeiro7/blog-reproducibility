"""Check the Metropolis sampler recovers its target and the diagnostic behaves.

The pooled chains are compared with the known target, and the Gelman-Rubin
statistic with its values on chains that do and do not agree.
"""

import numpy as np
import pytest

from blog_reproducibility.mathematics.metropolis import (
    TARGET_MEAN,
    TARGET_SD,
    example_payload,
    gelman_rubin,
    metropolis_chain,
    run_chains,
    target_log_density,
)

SUMMARY = example_payload()


def test_pooled_draws_recover_the_target() -> None:
    """Twelve thousand kept draws land on mean 2 and standard deviation 0.8."""
    assert SUMMARY.pooled_mean == pytest.approx(TARGET_MEAN, abs=0.05)
    assert SUMMARY.pooled_sd == pytest.approx(TARGET_SD, abs=0.03)


def test_chains_agree() -> None:
    """The figure's claim: the four chains explore the same posterior."""
    assert SUMMARY.r_hat == pytest.approx(1.0, abs=0.01)
    # A 0.9 step on a 0.8 target accepts most proposals, as random walks tuned for 1-D do.
    assert 0.5 < SUMMARY.acceptance_rate < 0.8


def test_gelman_rubin_flags_disagreeing_chains() -> None:
    """Chains centred apart give a large statistic; identical distributions give about one."""
    rng = np.random.default_rng(0)
    agreeing = tuple(rng.normal(0, 1, 2000) for _ in range(4))
    split = (rng.normal(0, 1, 2000), rng.normal(5, 1, 2000))

    assert gelman_rubin(agreeing) == pytest.approx(1.0, abs=0.01)
    assert gelman_rubin(split) > 2


def test_acceptance_depends_only_on_the_density_ratio() -> None:
    """A flat target accepts every proposal; the draws are then a plain random walk."""
    rng = np.random.default_rng(1)
    draws, accepted = metropolis_chain(lambda _: 0.0, 0.0, steps=100, proposal_sd=1.0, rng=rng)

    assert accepted == 100
    replay = np.random.default_rng(1)
    steps = []
    for _ in range(100):
        steps.append(replay.normal(0, 1.0))
        replay.random()
    np.testing.assert_allclose(draws, np.cumsum(steps))


def test_log_density_is_the_unnormalised_target() -> None:
    """Zero at the mode, falling by half a unit per standard deviation squared."""
    assert target_log_density(TARGET_MEAN) == 0.0
    assert target_log_density(TARGET_MEAN + TARGET_SD) == pytest.approx(-0.5)


def test_chains_are_deterministic() -> None:
    """The same seed reproduces every chain."""
    first, rate = run_chains(seed=3)
    second, again = run_chains(seed=3)

    assert rate == again
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_invalid_inputs_are_rejected() -> None:
    """Bad step sizes, lengths, and chain sets are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        metropolis_chain(target_log_density, 0.0, steps=10, proposal_sd=0.0, rng=rng)
    with pytest.raises(ValueError):
        metropolis_chain(target_log_density, 0.0, steps=0, proposal_sd=1.0, rng=rng)
    with pytest.raises(ValueError):
        gelman_rubin((np.zeros(5),))
    with pytest.raises(ValueError):
        gelman_rubin((np.zeros(5), np.zeros(4)))
