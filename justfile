# Default: List available commands
default:
    @just --list --color=always | less -F -R

# Run isort and ruff on the Python source code
codestyle:
    ./code_style.sh

# Launch the dashboard
dashboard:
    @uv run dashboard -H 0.0.0.0
