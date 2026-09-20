"""Figure renderer for the release-vintages article."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from blog_reproducibility.common.plotting import (
    PALETTE,
    FigureArtifact,
    save_figure,
    use_house_style,
)
from blog_reproducibility.time_series.release_vintages import load_releases

END = datetime.fromisoformat("2025-07-01T12:00:00+00:00")


def render_vintages_figure(*, output_dir: Path) -> FigureArtifact:
    """Contrast what was known at the time with the final figure carried backwards."""
    use_house_style()
    releases = load_releases()
    # Matplotlib draws datetimes fine, but its stubs only describe numbers, so
    # the conversion is made explicit here rather than silenced.
    dates = [
        float(mdates.date2num(release.released_at))  # type: ignore[no-untyped-call]
        for release in releases
    ]
    end = float(mdates.date2num(END))  # type: ignore[no-untyped-call]
    values = [release.value for release in releases]

    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    axis.step(
        [*dates, end],
        [*values, values[-1]],
        where="post",
        color=PALETTE[0],
        label="Estimate available at the time",
    )
    axis.scatter(dates, values, color=PALETTE[0], zorder=3)
    axis.hlines(
        values[-1],
        dates[0],
        end,
        color=PALETTE[1],
        linestyle="--",
        label="Third estimate copied backwards (hindsight)",
    )
    for moment, release in zip(dates, releases, strict=True):
        axis.annotate(
            f"{release.release_name}: {release.value:+.1f}%",
            (moment, release.value),
            xytext=(5, 12),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set(
        ylim=(-0.6, -0.1),
        xlim=(dates[0], end),
        xlabel="Date information becomes available (2025, UTC)",
        ylabel="Real GDP growth (%, annualized)",
        title="One quarter of economic activity, three dated estimates",
    )
    axis.xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)  # type: ignore[no-untyped-call]
    )
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")  # type: ignore[no-untyped-call]
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=9)

    return save_figure(figure, slug="gdp_release_vintages_2026", output_dir=output_dir)
