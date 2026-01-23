"""Step 3: Length-of-Stay fitting."""

from collections.abc import Generator
from typing import NamedTuple

import dash_mantine_components as dmc
from dash import Input, Output, State, callback
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components.common import main_ids


# Component IDs
class StepThreeIDs(NamedTuple):
    """Component IDs used in Step 3."""

    BTN_PREV: str = "step3-btn-prev"
    BTN_NEXT: str = "step3-btn-next"


step3_ids = StepThreeIDs()


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 3."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Text(
                "Step 3 contents go here.",
            )
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step3_ids.BTN_PREV)
            yield dmc.Button("Next", id=step3_ids.BTN_NEXT)
    return ret


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Input(step3_ids.BTN_PREV, "n_clicks"),
    prevent_initial_call=True,
)
def step3_on_prev(
    _: int,
) -> int:
    """Handle 'Previous' button click to go back a step."""
    return 1  # Go back to step 2 (index 1)


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Output(main_ids.MAIN_STORE, "data", allow_duplicate=True),
    Input(step3_ids.BTN_NEXT, "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step3_on_next(
    _: int,
    main_store_data: dict,
) -> tuple[int, dict]:
    """Handle 'Next' button click to advance the stepper."""
    # TODO: Add necessary inputs to callback
    # TODO: Validate inputs and update main_store_data as needed
    return 3, main_store_data  # Advance to step 4 (index 3)
