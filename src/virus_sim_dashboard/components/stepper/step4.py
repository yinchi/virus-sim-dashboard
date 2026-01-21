"""Step 4: Simulation and Results Visualization."""

from collections.abc import Generator

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from .common import ID_BTN_NEXT, ID_MAIN_STORE, ID_STEPPER


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 4."""
    with dmc.Stack(None, id="step4-stack", gap="md", m=0, p=0) as ret:
        yield dmc.Text(
            "Step 4 contents go here.",
        )
    return ret


@callback(
    Output(ID_STEPPER, "active", allow_duplicate=True),
    Output(ID_MAIN_STORE, "data", allow_duplicate=True),
    Input(ID_BTN_NEXT, "n_clicks"),
    Input("step4-stack", "id"),  # Dummy input to differentiate callbacks
    State(ID_STEPPER, "active"),
    State(ID_MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step4_on_next(
    _: int,
    _2: str,
    active_step: int,
    main_store_data: dict,
) -> tuple[int, dict]:
    """Handle 'Next' button click to advance the stepper."""
    # Only proceed if we are on step 4 (active_step == 3)
    if active_step != 3:
        raise dash.exceptions.PreventUpdate

    # TODO: Add necessary inputs to callback
    # TODO: Validate inputs and update main_store_data as needed
    return active_step + 1, main_store_data
