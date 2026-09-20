"""Input checks shared by the numerical models.

Article models take their parameters from the command line, from a notebook, or
from a test, so they validate what they are given rather than propagating a
silent ``nan``. Booleans are rejected everywhere a number is expected: ``True``
is an ``int`` in Python, and a model that accepts it reports a result for a
parameter nobody meant to pass.
"""

from math import isfinite


def real(value: object, *, name: str) -> float:
    """Return a finite real value, rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def positive(value: object, *, name: str) -> float:
    """Return a finite, strictly positive real value."""
    numeric = real(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def non_negative(value: object, *, name: str) -> float:
    """Return a finite real value that is zero or greater."""
    numeric = real(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be zero or greater")
    return numeric


def probability(value: object, *, name: str, inclusive: bool = True) -> float:
    """Return a finite probability, on [0, 1] or on (0, 1)."""
    numeric = real(value, name=name)
    valid = 0.0 <= numeric <= 1.0 if inclusive else 0.0 < numeric < 1.0

    if not valid:
        interval = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return numeric


def count(value: object, *, name: str, minimum: int = 0) -> int:
    """Return an integer of at least ``minimum``, rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or greater")
    return value
