"""Main stepper component for the Dash application.

A DMC Stepper component that guides users through the main steps of the virus
simulation analysis.
"""

from collections.abc import Generator

import dash_mantine_components as dmc
from dash import dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components.common import main_ids
from virus_sim_dashboard.components.stepper import step1, step2, step3, step4, step5


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Main content area layout."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        yield dcc.Store(id=main_ids.MAIN_STORE, data={})
        with dmc.Stepper(
            id=main_ids.STEPPER,
            active=0,
            size="sm",
            allowNextStepsSelect=False,
        ):
            with dmc.StepperStep(
                label="Upload data",
                description="Upload patient stay data",
            ):
                yield step1.layout()

            with dmc.StepperStep(
                label="Define patient groups",
                description="Group patients based on age, etc.",
            ):
                yield step2.layout()

            with dmc.StepperStep(
                label="Length-of-stay analysis",
                description="Fit patient LoS distributions",
            ):
                yield step3.layout()

            with dmc.StepperStep(
                label="Arrival Scenario",
                description="Create patient arrival scenario",
            ):
                yield step4.layout()

            with dmc.StepperStep(
                label="Simulate!",
                description="Run simulation and visualize results",
            ):
                yield step5.layout()
    return ret
