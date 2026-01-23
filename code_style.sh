#!/usr/bin/env bash
# Run isort and ruff on the src/ directory

WORKING_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"

echo cd "$WORKING_DIR"
cd "$WORKING_DIR"

echo
echo "====="
echo "ISORT"
echo "====="
uv run isort --show-files -w 100 ./src/

echo
echo "==========="
echo "RUFF FORMAT"
echo "==========="
uv run ruff format ./src/  # ruff config in .ruff.toml

echo
echo "================"
echo "RUFF CHECK --FIX"
echo "================"
uv run ruff check --fix ./src/

echo
echo "====="
echo "MYPY"
echo "====="
uv run mypy ./src/  # mypy config in pyproject.toml
