"""CUH Respiratory Virus Simulation Dashboard package.

This package contains a Dash application for simulating and visualizing
respiratory virus spread within a hospital setting.  For installation and usage
instructions, please refer to the README.md file in the repository root, which describes
how to set up the environment and run the application.

The dashboard runs a local web server that can be accessed via a web browser at
http://localhost:8050 by default. For details on how to use the dashboard once it is running,
please see the documentation available at `tutorial.pdf` (TODO: create this file).
"""

import argparse

from virus_sim_dashboard.app import app

parser = argparse.ArgumentParser(
    prog="uv run dashboard",
    description="CUH Respiratory Virus Simulation Dashboard",
    epilog="© 2026 Yin-Chi Chan",
    add_help=True,
)
parser.add_argument(
    "-H",
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host address to run the Dash app on (default: 127.0.0.1)",
)
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=8050,
    help="Port number to run the Dash app on (default: 8050)",
)

DEBUG_DEFAULT = True
parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    default=DEBUG_DEFAULT,
    help=f"Run the Dash app in debug mode (default: {DEBUG_DEFAULT})",
)


def main() -> None:
    """Main entry point to run the Dash app."""
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
