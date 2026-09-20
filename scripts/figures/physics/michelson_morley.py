"""Print the aether-drift prediction for a rotating interferometer.

No article embeds a figure for this, so the script only reports numbers.
"""

import argparse
import json

from blog_reproducibility.common.reporting import to_jsonable
from blog_reproducibility.physics.michelson_morley import example_payload


def main() -> None:
    """Print the predicted fringe shift and the speed a null result bounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolvable-shift",
        type=float,
        default=0.01,
        help="Fringe shift the apparatus is assumed able to resolve.",
    )
    args = parser.parse_args()

    print(
        json.dumps(to_jsonable(example_payload(resolvable_shift=args.resolvable_shift)), indent=2)
    )


if __name__ == "__main__":
    main()
