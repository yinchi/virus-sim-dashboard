"""Dark/Light theme toggle."""

import dash
import dash_mantine_components as dmc
import plotly.io as pio
from dash import Input, Output, Patch, State, callback
from dash.development.base_component import Component as DashComponent
from dash_iconify import DashIconify

from virus_sim_dashboard.components.common import main_ids
from virus_sim_dashboard.util import js_file

THEME_TOGGLE_ID = "theme-toggle-switch"


def layout() -> DashComponent:
    """Theme toggle component."""
    return dmc.Switch(
        offLabel=DashIconify(
            icon="material-symbols:light-mode-rounded",
            width=15,
            color=dmc.DEFAULT_THEME["colors"]["yellow"][8],
        ),
        onLabel=DashIconify(
            icon="material-symbols:dark-mode-rounded",
            width=15,
            color=dmc.DEFAULT_THEME["colors"]["yellow"][6],
        ),
        id=THEME_TOGGLE_ID,
        persistence=True,
        color="grey",
        size="lg",
    )


# Client-side callback to toggle the theme
dash.clientside_callback(
    js_file("theme_toggle.js"),
    Output(THEME_TOGGLE_ID, "checked"),  # Dummy output, callback returns `no_update`
    Input(THEME_TOGGLE_ID, "checked"),
)

# Callback to update the theme in figure components
dmc.add_figure_templates()
REDUCED_GRAPH_MARGINS = {"l": 30, "r": 30, "t": 90, "b": 30}
pio.templates["plotly"].layout.margin = REDUCED_GRAPH_MARGINS
pio.templates["mantine_dark"].layout.margin = REDUCED_GRAPH_MARGINS


GRAPH_IDS = {
    "type": "graph",
    "id": dash.ALL,
}


@callback(
    Output(GRAPH_IDS, "figure"),
    Input(main_ids.STEPPER, "active"),  # Dummy input to trigger on step change
    Input(THEME_TOGGLE_ID, "checked"),
    State(GRAPH_IDS, "id"),
)
def update_figure(_, is_dark_mode, ids):
    """Apply light/dark theme to dcc.Graph figures.

    Each figure must have a `dict` component ID instead of a string, with a `type` of "graph".
    """
    template = pio.templates["mantine_dark"] if is_dark_mode else pio.templates["plotly"]
    patched_figures = []
    for _ in ids:
        patched_fig = Patch()  # Patch corresponding to the nth graph
        patched_fig["layout"]["template"] = template
        patched_figures.append(patched_fig)

    return patched_figures
