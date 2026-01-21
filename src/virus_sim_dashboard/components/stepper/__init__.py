"""Main stepper component for the Dash application.

A DMC Stepper component that guides users through the main steps of the virus
simulation analysis.

1. Data upload
2. Parameter configuration for daily arrival rates
3. Parameter configuration for patient lengths-of-stay
4. Simulation execution and results visualization
"""

from collections.abc import Generator

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from . import step1, step2, step3, step4
from .common import ID_BTN_BACK, ID_BTN_NEXT, ID_MAIN_STORE, ID_STEPPER


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Main content area layout."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        yield dcc.Store(id=ID_MAIN_STORE, data={})
        with dmc.Stepper(
            id=ID_STEPPER,
            active=0,
            size="sm",
            allowNextStepsSelect=False,
        ):
            with dmc.StepperStep(
                label="Upload Data",
                description="Upload patient stay data for analysis",
            ):
                yield step1.layout()

            with dmc.StepperStep(
                label="Scenario Configuration",
                description="Set parameters for daily patient arrival rates",
            ):
                yield step2.layout()

            with dmc.StepperStep(
                label="LoS Analysis",
                description="Fit distributions for patient LoS",
            ):
                yield step3.layout()

            with dmc.StepperStep(
                label="Run Simulation",
                description="Simulate model and visualize results",
            ):
                yield step4.layout()
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Back", id=ID_BTN_BACK, color="gray")
            yield dmc.Button("Next", id=ID_BTN_NEXT)
    return ret


@callback(
    Output(ID_BTN_BACK, "disabled"),
    Output(ID_BTN_NEXT, "disabled"),
    Input(ID_STEPPER, "active"),
    State(ID_STEPPER, "children"),
)
def update_button_states(active: int, steps: list) -> tuple[bool, bool]:
    """Enable/disable the back and next buttons based on the current step."""
    # Get the number of steps in the stepper
    n_steps = len([step["type"] for step in steps if step["type"] == "StepperStep"])

    back_disabled = active == 0
    next_disabled = active == n_steps - 1
    return back_disabled, next_disabled


# Client-side callback for the 'Back' button
dash.clientside_callback(
    "(n_clicks, active) => Math.max(active - 1, 0)",
    Output(ID_STEPPER, "active", allow_duplicate=True),
    Input(ID_BTN_BACK, "n_clicks"),
    State(ID_STEPPER, "active"),
    prevent_initial_call=True,
)

# Each `step{n}` module should define a callback for the 'Next' button
# to validate inputs and advance the stepper.
