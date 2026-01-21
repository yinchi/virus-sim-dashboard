"""Dark/Light theme toggle."""

import dash
import dash_mantine_components as dmc
from dash import Input, Output
from dash.development.base_component import Component as DashComponent
from dash_iconify import DashIconify

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
