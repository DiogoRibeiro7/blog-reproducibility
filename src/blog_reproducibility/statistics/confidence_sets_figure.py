"""Figure renderers for the confidence-sets article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.confidence_sets import (
    Z,
    accepted_set,
    p_value,
    profile_search,
)

ALPHA = 0.05


def _linspace(low: float, high: float, points: int) -> list[float]:
    """Return evenly spaced values, endpoints included."""
    step = (high - low) / (points - 1)
    return [low + index * step for index in range(points)]


def render_components_figure(*, output_dir: Path) -> FigureArtifact:
    """Draw the p-value curve and accepted set for two observations."""
    use_house_style()
    grid = _linspace(-1.8, 1.8, 3601)
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)

    for axis, y in zip(axes, (0.04, 0.50), strict=True):
        pieces = accepted_set(y)
        axis.plot(
            grid,
            [p_value(theta, y) for theta in grid],
            color=PALETTE[0],
            lw=2.2,
            label="p-value of each candidate",
        )
        axis.axhline(ALPHA, color=INK_MUTED, lw=1.4, ls=":", label="level, 0.05")
        for index, (low, high) in enumerate(pieces):
            axis.axvspan(
                low,
                high,
                color=PALETTE[0],
                alpha=0.14,
                lw=0,
                label="accepted: the confidence set" if index == 0 else None,
            )
        axis.plot(
            [pieces[0][0], pieces[-1][1]],
            [-0.07, -0.07],
            color=PALETTE[1],
            lw=3,
            solid_capstyle="butt",
            label="convex hull of the set",
            clip_on=False,
        )
        axis.set_ylim(-0.12, 1.05)
        axis.set_xlabel("candidate value of the parameter")
        axis.set_title(f"observed {y:.2f}: {len(pieces)} components")

    axes[0].set_ylabel("p-value")
    # The rejected region around zero leaves the middle of the panel empty.
    axes[0].legend(loc="center", fontsize=8.5)
    # The house style turns constrained layout on; these panels were laid out tight.
    figure.set_layout_engine("tight")

    return save_figure(figure, slug="confidence_set_components", output_dir=output_dir)


def render_profile_basins_figure(*, output_dir: Path) -> FigureArtifact:
    """Draw the test statistic over the nuisance parameter and both local optima."""
    use_house_style()
    sigma, psi, y = 0.06, 0.40, 0.0

    def statistic(lam: float) -> float:
        return ((y - psi - ((lam**2 - 1.0) ** 2 - 0.3 * lam)) / sigma) ** 2

    grid = _linspace(-1.7, 1.7, 2001)
    figure, axis = plt.subplots()
    axis.plot(
        grid,
        [statistic(lam) for lam in grid],
        color=PALETTE[0],
        lw=2.4,
        label="test statistic along the nuisance parameter",
    )
    axis.axhline(Z**2, color=INK_MUTED, lw=1.4, ls=":", label="critical value, 3.84")

    annotations = (
        (-1.2, PALETTE[1], "local optimiser started at -1.2\nstops here: reject", (10, 12)),
        (1.2, PALETTE[2], "started at 1.2: the true minimum, 2.48\naccept (p = 0.115)", (-150, 28)),
    )
    for start, colour, text, offset in annotations:
        search = profile_search(start, psi=psi, y=y, sigma=sigma)
        axis.plot(
            [search.minimizer],
            [search.statistic],
            marker="o",
            ms=9,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=5,
        )
        axis.annotate(
            text,
            (search.minimizer, search.statistic),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
        )

    axis.set_yscale("log")
    axis.set_ylim(1, 3e4)
    axis.set_xlabel("nuisance parameter")
    axis.set_ylabel("test statistic for the target value 0.40")
    axis.set_title("Two basins: which one the optimiser finds decides the inference")
    axis.legend(loc="upper center")

    return save_figure(figure, slug="confidence_set_profile_basins", output_dir=output_dir)


def render_confidence_set_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the confidence-sets article."""
    return (
        render_components_figure(output_dir=output_dir),
        render_profile_basins_figure(output_dir=output_dir),
    )
