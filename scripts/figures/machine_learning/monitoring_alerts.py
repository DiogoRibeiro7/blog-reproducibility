"""Reproduce the drift-monitoring power sweep, false-alert bursts and both figures."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.machine_learning.monitoring_alerts import example_payload
from blog_reproducibility.machine_learning.monitoring_alerts_figure import (
    render_alert_bursts_figure,
    render_power_curves_figure,
)


def main() -> None:
    """Print the numerical payload and optionally render the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR / "machine_learning",
        help="Directory for generated PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print calculations without writing the figures.",
    )
    args = parser.parse_args()

    summary = example_payload()
    payload = to_jsonable(summary)
    if not args.dry_run:
        payload["figures"] = describe_figures(
            (
                render_power_curves_figure(output_dir=args.output_dir, rows=summary.power),
                render_alert_bursts_figure(output_dir=args.output_dir, bursts=summary.bursts),
            )
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
