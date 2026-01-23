"""Common definitions for the Dash application."""

import json
import pathlib

COPY = "\u00a9"
NDASH = "\u2013"
NBSP = "\u00a0"


def jsonify(obj: object) -> str:
    """Convert an object to a minified JSON string for comparison purposes."""
    return json.dumps(obj, sort_keys=True, indent=None, separators=(",", ":"))


def find_upwards(filename: str, start: pathlib.Path | str | None = None) -> pathlib.Path | None:
    """Search for a file by looking in the current directory and then moving upwards.

    Args:
        filename: Name of the file to search for.
        start: Directory to start searching from.  If None, uses the current working directory.

    Returns:
        The path to the found file, or None if not found.
    """
    if start is None:
        curr_dir = pathlib.Path.cwd()
    else:
        curr_dir = pathlib.Path(start).resolve()

        # If `start` is a file, start from its parent directory
        if not curr_dir.is_dir():
            curr_dir = curr_dir.parent

    root = curr_dir.anchor

    while True:
        candidate = curr_dir / filename
        if candidate.exists():
            return candidate

        if str(curr_dir) == root:
            break

        curr_dir = curr_dir.parent

    return None


def js_file(path: str) -> str:
    """Read a JavaScript file from the src/js/ directory and return its contents as a string.

    Args:
        path: Relative path to the JavaScript file within the src/js/ directory.
    """
    # Use pyproject.toml as an anchor to find the src/js directory
    project_root = find_upwards("pyproject.toml")
    if not project_root:
        raise FileNotFoundError("Could not find 'pyproject.toml' to locate 'src/js' directory.")
    src_dir = project_root.parent / "src/js"
    with open(src_dir / path, "r", encoding="utf-8") as f:
        return f.read()


DEFAULT_FIGURE_LAYOUT = {
    "width": 1200,
    "height": 350,
    "legend": {
        "yanchor": "bottom",
        "y": 1,
        "xanchor": "left",
        "x": 0,
        "font_size": 14,
        "orientation": "h",
    },
    "title": {
        "text": "Daily patient arrivals for {disease_name}",
        "font": {"size": 20, "weight": 700},
    },
    "xaxis": {"tickfont": {"size": 14}},
    "yaxis": {"tickfont": {"size": 14}},
    "hovermode": "x unified",
}
