# Default: List available commands
default:
    @just --list --color=always | less -F -R

# Run isort and ruff on the Python source code
codestyle:
    ./code_style.sh

# Launch the dashboard
dashboard-prod:
    @uv run --no-dev dashboard -H 0.0.0.0 --no-debug

# Launch the dashboard in dev mode
dashboard:
    @uv run dashboard -p 8888

# Build the Docker container
build:
    @docker build . -t yinchi/virus-sim-dashboard

# Launch the Docker container
docker:
    @docker run --rm -d -p 8050:8050 --name virus-sim-dashboard yinchi/virus-sim-dashboard:latest

# Stop the Docker container
docker-stop:
    @docker stop virus-sim-dashboard -t 3
