"""Reproduce the adaptive baseline calculations and figure."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.health.adaptive_baseline import example_payload
from blog_reproducibility.health.adaptive_baseline_figure import render_baseline_figure


def main() -> None:
    """Print every step of each learning rate and optionally render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR / "health",
        help="Directory for generated PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print calculations without writing the figure.",
    )
    args = parser.parse_args()

    payload: dict[str, object] = {"baselines": to_jsonable(example_payload())}
    if not args.dry_run:
        payload["figures"] = describe_figures((render_baseline_figure(output_dir=args.output_dir),))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
