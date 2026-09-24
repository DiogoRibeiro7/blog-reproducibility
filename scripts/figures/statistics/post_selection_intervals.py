"""Reproduce the exact post-selection intervals, their widths and both figures."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.statistics.post_selection_intervals import example_payload
from blog_reproducibility.statistics.post_selection_intervals_figure import (
    render_selective_interval_runaway_figure,
    render_selective_width_by_procedure_figure,
)


def main() -> None:
    """Print the numerical payload and optionally render both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR / "statistics",
        help="Directory for generated PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print calculations without writing figures.",
    )
    args = parser.parse_args()

    summary = example_payload()
    payload = to_jsonable(summary)
    if not args.dry_run:
        payload["figures"] = describe_figures(
            (
                render_selective_interval_runaway_figure(
                    output_dir=args.output_dir, summary=summary.width_curve
                ),
                render_selective_width_by_procedure_figure(
                    output_dir=args.output_dir, summary=summary.procedure_curves
                ),
            )
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
