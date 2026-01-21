#!/usr/bin/env bash
# Run isort and ruff on the src/ directory

WORKING_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"

echo cd "$WORKING_DIR"
cd "$WORKING_DIR"

echo
echo "====="
echo "ISORT"
echo "====="
isort --show-files -w 100 ./src/

echo
echo "==========="
echo "RUFF FORMAT"
echo "==========="
ruff format ./src/

echo
echo "================"
echo "RUFF CHECK --FIX"
echo "================"
ruff check --fix ./src/
