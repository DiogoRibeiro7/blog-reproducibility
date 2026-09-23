"""Solow steady state for the article on the Solow growth model and its extensions.

With Cobb-Douglas output per effective worker ``f(k) = k**alpha``, capital
accumulates while saving ``s f(k)`` exceeds the break-even investment
``(n + g + delta) k`` needed to keep ``k`` constant. Concavity makes the two
cross once, at

    k* = (s / (n + g + delta)) ** (1 / (1 - alpha)),

and the steady state is stable: below ``k*`` saving outruns break-even and
capital grows, above it capital shrinks.

The golden-rule capital stock maximises steady-state consumption. It solves
``f'(k) = n + g + delta``; an economy saving less than ``alpha`` settles below it.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "DEFAULT_PARAMETERS",
    "SolowParameters",
    "SolowSummary",
    "break_even_investment",
    "capital_path",
    "example_payload",
    "golden_rule_capital",
    "output_per_worker",
    "saving",
    "steady_state_capital",
]


@dataclass(frozen=True, slots=True)
class SolowParameters:
    """Technology, saving, and growth rates of the Solow model."""

    capital_share: float = 0.35
    saving_rate: float = 0.25
    population_growth: float = 0.01
    technology_growth: float = 0.02
    depreciation: float = 0.05

    def __post_init__(self) -> None:
        probability(self.capital_share, name="capital_share", inclusive=False)
        probability(self.saving_rate, name="saving_rate", inclusive=False)
        real(self.population_growth, name="population_growth")
        real(self.technology_growth, name="technology_growth")
        real(self.depreciation, name="depreciation")
        if self.break_even_rate <= 0:
            raise ValueError("population growth, technology growth, and depreciation must sum > 0")

    @property
    def break_even_rate(self) -> float:
        """Investment per unit of capital needed to hold ``k`` constant."""
        return self.population_growth + self.technology_growth + self.depreciation


DEFAULT_PARAMETERS: Final[SolowParameters] = SolowParameters()


@dataclass(frozen=True, slots=True)
class SolowSummary:
    """The steady state the article's figure marks, with the golden rule for context."""

    parameters: SolowParameters
    steady_state_capital: float
    steady_state_output: float
    steady_state_consumption: float
    golden_rule_capital: float


def _capital(capital: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(capital, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("capital must be finite and non-negative")
    return values


def output_per_worker(
    capital: ArrayLike, parameters: SolowParameters = DEFAULT_PARAMETERS
) -> NDArray[np.float64]:
    """Cobb-Douglas output per effective worker, ``k ** alpha``."""
    return np.asarray(_capital(capital) ** parameters.capital_share, dtype=np.float64)


def saving(
    capital: ArrayLike, parameters: SolowParameters = DEFAULT_PARAMETERS
) -> NDArray[np.float64]:
    """Gross investment per effective worker, ``s f(k)``."""
    return parameters.saving_rate * output_per_worker(capital, parameters)


def break_even_investment(
    capital: ArrayLike, parameters: SolowParameters = DEFAULT_PARAMETERS
) -> NDArray[np.float64]:
    """Investment needed to keep capital per effective worker constant."""
    return parameters.break_even_rate * _capital(capital)


def steady_state_capital(parameters: SolowParameters = DEFAULT_PARAMETERS) -> float:
    """The unique positive capital stock where saving equals break-even investment."""
    ratio = parameters.saving_rate / parameters.break_even_rate
    return float(ratio ** (1 / (1 - parameters.capital_share)))


def golden_rule_capital(parameters: SolowParameters = DEFAULT_PARAMETERS) -> float:
    """The steady-state capital stock that maximises consumption per worker."""
    ratio = parameters.capital_share / parameters.break_even_rate
    return float(ratio ** (1 / (1 - parameters.capital_share)))


def capital_path(
    initial_capital: float,
    periods: int,
    parameters: SolowParameters = DEFAULT_PARAMETERS,
) -> tuple[float, ...]:
    """Iterate ``k_(t+1) = k_t + s f(k_t) - (n + g + delta) k_t`` from ``k_0``."""
    capital = positive(initial_capital, name="initial_capital")
    path = [capital]
    for _ in range(count(periods, name="periods")):
        capital += float(saving(capital, parameters) - break_even_investment(capital, parameters))
        path.append(capital)
    return tuple(path)


def example_payload() -> SolowSummary:
    """Return the steady state behind the article's figure."""
    capital = steady_state_capital()
    output = float(output_per_worker(capital))
    return SolowSummary(
        parameters=DEFAULT_PARAMETERS,
        steady_state_capital=capital,
        steady_state_output=output,
        steady_state_consumption=(1 - DEFAULT_PARAMETERS.saving_rate) * output,
        golden_rule_capital=golden_rule_capital(),
    )
