"""Figure renderers for the leaky-gut article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL,
    SURFACE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.health.leaky_gut import (
    POWER,
    TOP,
    TURPIN,
    correlation_interval,
    risks_by_test,
    share_truly_high,
)


def render_surrogate_figure(*, output_dir: Path) -> FigureArtifact:
    """Plot what a positive surrogate result is worth against the correlation."""
    use_house_style()
    estimate = correlation_interval(POWER["r squared"], int(POWER["participants"]))
    grid = [index / 100 for index in range(100)]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.axvspan(
        0,
        estimate.high,
        color=SEQUENTIAL[0],
        alpha=0.55,
        linewidth=0,
        label="95% interval for the commercial kit (39 people)",
    )
    axis.axhline(
        100 * TOP,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (3, 3)),
        label="picking people at random",
    )
    axis.plot(
        grid,
        [100 * share_truly_high(value) for value in grid],
        color=PALETTE[0],
        label="a test with this correlation",
    )

    here = 100 * share_truly_high(estimate.correlation)
    axis.plot(
        [estimate.correlation],
        [here],
        "o",
        color=PALETTE[1],
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    axis.annotate(
        f"observed correlation {estimate.correlation:.2f}: {here:.0f}%",
        xy=(estimate.correlation, here),
        xytext=(34, -46),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 100)
    axis.set_xlabel("correlation between the test and measured permeability")
    axis.set_ylabel("test positives whose permeability is truly high, %")
    axis.set_title("What a positive result is worth depends on the correlation")
    axis.legend(loc="upper left", fontsize=8.5)

    return save_figure(figure, slug="leaky_gut_surrogate_test", output_dir=output_dir)


def render_relatives_figure(*, output_dir: Path) -> FigureArtifact:
    """Turn the published hazard ratio into absolute risks under three assumptions."""
    use_house_style()
    overall = TURPIN["developed Crohn's disease"] / TURPIN["relatives"]
    ratio = TURPIN["hazard ratio"]
    shares = (0.1, 0.2, 0.3)
    width = 0.36

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    series = (
        ("normal permeability test", PALETTE[0]),
        ("abnormal permeability test", PALETTE[1]),
    )
    for index, (label, colour) in enumerate(series):
        values = [100 * risks_by_test(share, overall, ratio)[index] for share in shares]
        positions = [position + (index - 0.5) * width for position in range(len(shares))]
        axis.bar(positions, values, width=width * 0.94, color=colour, label=label)
        for position, value in zip(positions, values, strict=True):
            axis.text(
                position,
                value + 0.25,
                f"{value:.1f}%",
                ha="center",
                fontsize=9,
                color=INK_SECONDARY,
            )

    axis.set_xticks(range(len(shares)))
    axis.set_xticklabels([f"if {100 * share:.0f}% of relatives test abnormal" for share in shares])
    axis.set_ylim(0, 12)
    axis.set_ylabel("developed Crohn's disease in about eight years, %")
    axis.set_title("A threefold risk is still a small risk: most abnormal tests lead to nothing")
    axis.legend(loc="upper right")
    axis.grid(axis="x", visible=False)

    return save_figure(figure, slug="leaky_gut_relatives_risk", output_dir=output_dir)


def render_leaky_gut_figures(*, output_dir: Path) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the leaky-gut article."""
    return (
        render_surrogate_figure(output_dir=output_dir),
        render_relatives_figure(output_dir=output_dir),
    )
