"""Reproduce the Type S and Type M closed forms, the article's tables and the figure."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.statistics.design_analysis import example_payload
from blog_reproducibility.statistics.design_analysis_figure import render_design_analysis_figure


def main() -> None:
    """Print the numerical payload and optionally render the figure."""
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
        help="Print calculations without writing the figure.",
    )
    args = parser.parse_args()

    summary = example_payload()
    payload = to_jsonable(summary)
    if not args.dry_run:
        payload["figures"] = describe_figures(
            (render_design_analysis_figure(output_dir=args.output_dir, summary=summary),)
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
