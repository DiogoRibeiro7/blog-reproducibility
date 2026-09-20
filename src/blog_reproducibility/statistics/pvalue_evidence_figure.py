"""Figure renderers for the p-value evidence article."""

from pathlib import Path

import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    INK_MUTED,
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.statistics.pvalue_evidence import (
    STANDARD_NORMAL,
    selected_studies,
    study_summary,
)


def render_selected_studies_figure(*, output_dir: Path) -> FigureArtifact:
    """Render expected retained-study composition across alpha thresholds."""
    rows = [selected_studies(alpha=alpha) for alpha in (0.05, 0.01, 0.001)]
    use_house_style()

    figure, axis = plt.subplots(figsize=(9, 5.4))
    false_counts = [row.null_rejections for row in rows]
    true_counts = [row.signal_rejections for row in rows]

    axis.bar(
        range(3),
        false_counts,
        width=0.55,
        color=PALETTE[1],
        label="Null-generated rejections",
    )
    axis.bar(
        range(3),
        true_counts,
        width=0.55,
        bottom=false_counts,
        color=PALETTE[0],
        label="Signal-generated rejections",
    )

    for x, row in enumerate(rows):
        retained = row.signal_rejections + row.null_rejections
        axis.text(
            x,
            retained + 18.0,
            f"{retained:.1f} retained\n{100.0 * row.signal_given_rejection:.1f}% signal",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set(
        xticks=range(3),
        xticklabels=["0.05", "0.01", "0.001"],
        ylim=(0, 1190),
        xlabel="Prespecified significance threshold",
        ylabel="Expected rejections per 10,000 studies",
        title="The retained studies depend on both error rates and the starting mixture",
    )
    axis.legend(loc="upper right", fontsize=8.5)
    figure.text(
        0.02,
        0.01,
        "Synthetic model: 90% null, 10% signal; signal Z ~ N(2, 1). Counts are expectations.",
        color=INK_MUTED,
        fontsize=8,
    )

    return save_figure(
        figure,
        slug="science_pvalue_selected_studies",
        output_dir=output_dir,
    )


def render_study_comparison_figure(*, output_dir: Path) -> FigureArtifact:
    """Render two nearly identical estimates with different significance labels."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(9, 4.7))

    studies = (
        ("Study B", 0.19, 0.11),
        ("Study A", 0.20, 0.10),
    )
    for y, (_, estimate, se) in enumerate(studies):
        summary = study_summary(estimate, se)
        axis.errorbar(
            estimate,
            y,
            xerr=STANDARD_NORMAL.inv_cdf(0.975) * se,
            fmt="o",
            capsize=5,
            color=PALETTE[0],
            markersize=7,
        )
        axis.annotate(
            f"p = {summary.p_value:.4f}",
            (0.42, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )

    axis.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    axis.set(
        xlim=(-0.08, 0.60),
        ylim=(-0.7, 1.7),
        yticks=[0, 1],
        yticklabels=["Study B", "Study A"],
        xlabel="Estimated effect and 95% normal-model confidence interval",
        title="Nearly identical estimates can receive opposite significance labels",
    )
    figure.text(
        0.02,
        0.01,
        "Invented independent studies of the same estimand; the difference is 0.01 (SE 0.149).",
        color=INK_MUTED,
        fontsize=8,
    )

    return save_figure(
        figure,
        slug="science_pvalue_study_comparison",
        output_dir=output_dir,
    )


def render_pvalue_evidence_figures(
    *,
    output_dir: Path,
) -> tuple[FigureArtifact, FigureArtifact]:
    """Render both figures used by the p-value evidence article."""
    return (
        render_selected_studies_figure(output_dir=output_dir),
        render_study_comparison_figure(output_dir=output_dir),
    )
