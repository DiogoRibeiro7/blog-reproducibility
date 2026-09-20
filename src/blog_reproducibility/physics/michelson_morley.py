"""What a stationary aether predicts for a rotating interferometer.

This module replaces `code/michelson_morley.py` in the website repository, which
was a box plot of ten numbers labelled ``# Hypothetical data`` with no article
behind it. Nothing was migrated from it: the arithmetic below is written from
the physics.

**The prediction.** In an aether at rest, light travelling a distance ``L`` and
back along the direction of motion takes ``(2L/c) / (1 - b^2)``, and across it
``(2L/c) / sqrt(1 - b^2)``, with ``b = v/c``. In one orientation the optical
paths therefore differ by

    d = 2L * (1/(1 - b^2) - 1/sqrt(1 - b^2)),

which is ``L b^2`` to leading order. Rotating the apparatus by a quarter turn
swaps the two arms, so the difference changes sign and the fringes move by
``2d``:

    shift = 4L/lambda * (1/(1 - b^2) - 1/sqrt(1 - b^2)),

which is ``2 L b^2 / lambda`` to leading order in ``b``. The factor of two from
the rotation is easy to lose and doubles the answer.

**The scale.** Michelson and Morley's 1887 apparatus folded the light path with
mirrors to about eleven metres per arm and used sodium light near 590 nm. The
Earth's orbital speed is about 30 km/s, giving ``b`` near 1e-4. Those numbers
predict a shift of roughly four tenths of a fringe, well above what the
apparatus could resolve, and the experiment saw far less.

The exact expression is kept alongside the leading-order one, because the point
of the example is that the predicted effect is second order in ``b`` and still
large enough to see. Only the ratio ``v/c`` matters, so the SI value of ``c`` is
not needed anywhere.

**Cancellation.** Written as a difference of two travel times the exact form
loses most of its significant digits: at ``b`` near 1e-4 the two times agree to
eight places and their difference is what is wanted. The identity

    1/(1 - b^2) - 1/sqrt(1 - b^2) = b^2 / ((1 - b^2) (1 + sqrt(1 - b^2)))

removes the subtraction, so the value below is accurate to full precision. The
test suite computes it the naive way as well, and records how much is lost.
"""

from dataclasses import dataclass
from math import sqrt

from blog_reproducibility.common.validation import positive, probability

__all__ = [
    "ARM_LENGTH_METRES",
    "EARTH_ORBITAL_SPEED_M_PER_S",
    "SODIUM_WAVELENGTH_METRES",
    "SPEED_OF_LIGHT_M_PER_S",
    "FringePrediction",
    "MichelsonMorleySummary",
    "example_payload",
    "prediction",
    "expected_fringe_shift",
    "leading_order_fringe_shift",
    "speed_from_fringe_shift",
]

# Exact by the SI definition of the metre.
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
# The 1887 apparatus: the light path was folded by mirrors to about 11 m per arm.
ARM_LENGTH_METRES = 11.0
# Sodium D light, the source they used.
SODIUM_WAVELENGTH_METRES = 5.9e-7
# The Earth's mean orbital speed.
EARTH_ORBITAL_SPEED_M_PER_S = 29_780.0


@dataclass(frozen=True, slots=True)
class FringePrediction:
    """What a quarter turn should shift the fringes by, under a stationary aether."""

    arm_length_metres: float
    wavelength_metres: float
    speed_m_per_s: float
    beta: float
    exact_shift: float
    leading_order_shift: float


@dataclass(frozen=True, slots=True)
class MichelsonMorleySummary:
    """The prediction at several speeds, against what the apparatus could resolve."""

    at_earth_orbital_speed: FringePrediction
    by_speed: dict[str, FringePrediction]
    resolvable_shift: float
    speed_implied_by_the_resolvable_shift: float


def _beta(speed_m_per_s: float) -> float:
    """Return v/c, rejecting speeds at or above light speed."""
    speed = positive(speed_m_per_s, name="speed_m_per_s")
    ratio = speed / SPEED_OF_LIGHT_M_PER_S
    if ratio >= 1.0:
        raise ValueError("speed_m_per_s must be below the speed of light")
    return ratio


def expected_fringe_shift(
    speed_m_per_s: float = EARTH_ORBITAL_SPEED_M_PER_S,
    *,
    arm_length_metres: float = ARM_LENGTH_METRES,
    wavelength_metres: float = SODIUM_WAVELENGTH_METRES,
) -> float:
    """Fringe shift on a quarter turn, from the exact light-travel times.

    The quarter turn swaps the arms, so the shift is twice the optical path
    difference of a single orientation.
    """
    beta = _beta(speed_m_per_s)
    length = positive(arm_length_metres, name="arm_length_metres")
    wavelength = positive(wavelength_metres, name="wavelength_metres")

    squared = beta * beta
    root = sqrt(1 - squared)
    # Algebraically identical to (1/(1-b^2) - 1/sqrt(1-b^2)), without the
    # subtraction of two nearly equal numbers.
    difference = squared / ((1 - squared) * (1 + root))
    return 4 * length / wavelength * difference


def leading_order_fringe_shift(
    speed_m_per_s: float = EARTH_ORBITAL_SPEED_M_PER_S,
    *,
    arm_length_metres: float = ARM_LENGTH_METRES,
    wavelength_metres: float = SODIUM_WAVELENGTH_METRES,
) -> float:
    """The same shift to leading order, ``2 L beta^2 / lambda``."""
    beta = _beta(speed_m_per_s)
    length = positive(arm_length_metres, name="arm_length_metres")
    wavelength = positive(wavelength_metres, name="wavelength_metres")

    return 2 * length * beta * beta / wavelength


def speed_from_fringe_shift(
    shift: float,
    *,
    arm_length_metres: float = ARM_LENGTH_METRES,
    wavelength_metres: float = SODIUM_WAVELENGTH_METRES,
) -> float:
    """Invert the leading-order relation: what speed would produce this shift?

    This is how a null result becomes an upper bound on the speed through the
    aether. The inversion uses the leading-order form, which is accurate to
    better than one part in a million at these speeds.
    """
    observed = positive(shift, name="shift")
    length = positive(arm_length_metres, name="arm_length_metres")
    wavelength = positive(wavelength_metres, name="wavelength_metres")

    return SPEED_OF_LIGHT_M_PER_S * sqrt(observed * wavelength / (2 * length))


def prediction(
    speed_m_per_s: float = EARTH_ORBITAL_SPEED_M_PER_S,
    *,
    arm_length_metres: float = ARM_LENGTH_METRES,
    wavelength_metres: float = SODIUM_WAVELENGTH_METRES,
) -> FringePrediction:
    """Bundle both forms of the prediction at one speed."""
    return FringePrediction(
        arm_length_metres=arm_length_metres,
        wavelength_metres=wavelength_metres,
        speed_m_per_s=speed_m_per_s,
        beta=_beta(speed_m_per_s),
        exact_shift=expected_fringe_shift(
            speed_m_per_s,
            arm_length_metres=arm_length_metres,
            wavelength_metres=wavelength_metres,
        ),
        leading_order_shift=leading_order_fringe_shift(
            speed_m_per_s,
            arm_length_metres=arm_length_metres,
            wavelength_metres=wavelength_metres,
        ),
    )


def example_payload(*, resolvable_shift: float = 0.01) -> MichelsonMorleySummary:
    """Return the prediction against the shift the apparatus could resolve.

    ``resolvable_shift`` defaults to a hundredth of a fringe, which is the order
    of what the 1887 apparatus could distinguish. It is a stated assumption of
    this calculation, not a measurement reproduced here.
    """
    resolvable = probability(resolvable_shift, name="resolvable_shift", inclusive=False)

    return MichelsonMorleySummary(
        at_earth_orbital_speed=prediction(),
        by_speed={
            str(int(speed)): prediction(speed)
            for speed in (3_000.0, 10_000.0, EARTH_ORBITAL_SPEED_M_PER_S, 100_000.0)
        },
        resolvable_shift=resolvable,
        speed_implied_by_the_resolvable_shift=speed_from_fringe_shift(resolvable),
    )
