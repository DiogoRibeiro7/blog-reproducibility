"""Check the monitoring simulation's mechanism and the figure's claim.

The null mechanism is checked on large single days, where the expected null
share and the bias from dropping nulls are known in closed form. The seeded
curves then carry the figure's claim: the null-rate monitor fires as soon as
the rate moves, while the metric monitor is blind at small rates.
"""

import numpy as np
import pytest

from blog_reproducibility.data_science.null_monitoring import (
    NULL_RATES,
    QUIET_NULL_RATE,
    example_payload,
    quiet_baseline,
    simulate_day,
)

SUMMARY = example_payload()
SEGMENT_MEANS = np.array([12.0, 18.0, 30.0])
SEGMENT_SHARES = np.array([0.55, 0.30, 0.15])
NULL_WEIGHTS = np.array([0.6, 1.0, 2.2])


def test_day_matches_the_closed_form() -> None:
    """Null share and the metric after dropping nulls match their expectations."""
    rate = 0.2
    null_by_segment = rate * NULL_WEIGHTS
    expected_nulls = float(SEGMENT_SHARES @ null_by_segment)
    kept = SEGMENT_SHARES * (1 - null_by_segment)
    expected_metric = float(kept @ SEGMENT_MEANS / kept.sum())

    metric, nulls = simulate_day(rate, np.random.default_rng(0), rows=2_000_000)

    assert nulls == pytest.approx(expected_nulls, abs=0.001)
    assert metric == pytest.approx(expected_metric, abs=0.02)
    # Heavy rows go missing more often, so dropping nulls pulls the metric down.
    assert expected_metric < float(SEGMENT_SHARES @ SEGMENT_MEANS)


def test_quiet_baseline_is_centred_on_the_normal_rate() -> None:
    """Sixty quiet days average close to the 2 percent null rate."""
    baseline = SUMMARY.baseline

    assert baseline.null_rate_mean == pytest.approx(QUIET_NULL_RATE * 0.95, abs=0.001)
    assert baseline.metric_sd > 0 and baseline.null_rate_sd > 0


def test_null_monitor_fires_as_soon_as_the_rate_moves() -> None:
    """The figure's claim: certainty from the first step above the quiet rate."""
    rows = {round(row.null_rate, 2): row for row in SUMMARY.detection}

    assert rows[0.02].null_rate_monitor < 0.02
    for rate in NULL_RATES[1:]:
        assert rows[round(rate, 2)].null_rate_monitor == 1.0


def test_metric_monitor_is_blind_at_small_rates() -> None:
    """The metric barely moves until several percent of rows are missing."""
    rows = {round(row.null_rate, 2): row for row in SUMMARY.detection}
    metric = [row.metric_monitor for row in SUMMARY.detection]

    assert rows[0.02].metric_monitor < 0.02
    assert rows[0.03].metric_monitor < 0.05
    assert rows[0.05].metric_monitor < 0.3
    assert rows[0.12].metric_monitor == 1.0
    assert metric == sorted(metric)


def test_simulation_is_deterministic() -> None:
    """The same seed reproduces the same baseline."""
    first = quiet_baseline(np.random.default_rng(4), days=5)

    assert first == quiet_baseline(np.random.default_rng(4), days=5)


def test_invalid_inputs_are_rejected() -> None:
    """Impossible rates, empty days, and a day of nothing but nulls are refused."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_day(1.5, rng)
    with pytest.raises(ValueError):
        simulate_day(0.1, rng, rows=0)
    # One row at a full null rate: with this seed it arrives null, leaving no metric.
    with pytest.raises(ValueError):
        simulate_day(1.0, np.random.default_rng(0), rows=1)
    with pytest.raises(ValueError):
        quiet_baseline(rng, days=1)
