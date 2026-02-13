"""Step 4: Simulation and Results Visualization."""

from collections.abc import Generator
import json
from typing import NamedTuple

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components.common import main_ids


# Component IDs
class StepFourIDs(NamedTuple):
    """Component IDs used in Step 4."""

    BTN_PREV: str = "step4-btn-prev"


step4_ids = StepFourIDs()


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 4."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Text(
                "Step 4 contents go here.",
            )
            yield dmc.Button("TEMP download data", id="temp-download-btn")
            yield dcc.Download(id="temp-download")
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step4_ids.BTN_PREV)
    return ret


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Input(step4_ids.BTN_PREV, "n_clicks"),
    prevent_initial_call=True,
)
def step4_on_prev(
    _: int,
) -> int:
    """Handle 'Previous' button click to go back a step."""
    return 2  # Go back to step 3 (index 2)


@callback(
    Output("temp-download", "data"),
    Input("temp-download-btn", "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def temp_download_data(_: int, main_store_data: dict) -> dict:
    """Temporary callback to download data."""
    
    return {
        "content": json.dumps(main_store_data),
        "filename": "main_store_data.json",
    }
