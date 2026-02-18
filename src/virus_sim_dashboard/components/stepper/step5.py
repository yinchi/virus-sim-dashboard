"""Step 5: Simulation and Results Visualization."""

import time
from collections.abc import Generator
from io import BytesIO
from typing import Callable

import dash_mantine_components as dmc
import diskcache
import pandas as pd
from dash import DiskcacheManager, Input, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components.common import main_ids, step5_ids


# region layout
@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 5."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        yield dmc.Title("Step 5: Simulation and Results Visualization", order=2, ta="center")
        with dmc.Group(None, gap="md"):
            yield dmc.Button(
                "Download full config (.xlsx)",
                disabled=True,
                size="xl",
                id=step5_ids.BTN_DOWNLOAD_CONFIG,
            )
            yield dcc.Download(id=step5_ids.DOWNLOAD_CONFIG)
            yield dmc.Button("Run simulation", color="green", size="xl", id=step5_ids.BTN_SIMULATE)
        # Progress indicator for simulation running status
        yield dmc.Title("Simulation Progress", order=3)
        with dmc.Group(None, id=step5_ids.GROUP_SIM_PROGRESS, gap="md", style={"display": "none"}):
            yield dmc.Text("0/30", id=step5_ids.TEXT_SIM_PROGRESS)
            yield dmc.Progress(
                color="blue", id=step5_ids.PROGRESS_SIM_PROGRESS, size="xl", w=900, value=0
            )
        yield dmc.Stack(
            None,  # Empty initially
            id=step5_ids.STACK_SIMULATION_RESULTS,
            gap="md",
            m=0,
            p=0,
        )
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step5_ids.BTN_PREV)
    return ret


# endregion


# region callbacks
@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Input(step5_ids.BTN_PREV, "n_clicks"),
    prevent_initial_call=True,
)
def step5_on_prev(
    _: int,
) -> int:
    """Handle 'Previous' button click to go back a step."""
    return 3  # Go back to step 4 (index 3)


@callback(
    Output(step5_ids.DOWNLOAD_CONFIG, "data"),
    Input(step5_ids.BTN_DOWNLOAD_CONFIG, "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step5_on_download_config(
    _: int,
    main_store_data: dict,
) -> dict:
    """Handle 'Download Config' button click to download the current config as an Excel file."""
    # Empty workbook for demo purposes
    # TODO: populate the workbook with the actual config data from main_store_data
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            ws = writer.book.active
            ws.title = "Config"
            ws["A1"] = "Disease Name"
            ws["B1"] = main_store_data.get("step1", {}).get("disease_name", "N/A")
        excel_bytes = buffer.getvalue()
    return dcc.send_bytes(excel_bytes, "config.xlsx")


@callback(
    Output(step5_ids.STACK_SIMULATION_RESULTS, "children"),
    Output(step5_ids.TEXT_SIM_PROGRESS, "children"),
    Output(step5_ids.PROGRESS_SIM_PROGRESS, "value"),
    Input(step5_ids.BTN_SIMULATE, "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
    background=True,  # run in background to avoid blocking UI
    manager=DiskcacheManager(
        diskcache.Cache("./.dash_cache"),
    ),
    running=[
        (Output(step5_ids.BTN_SIMULATE, "disabled"), True, False),
        (Output(step5_ids.PROGRESS_SIM_PROGRESS, "animated"), True, False),
        (Output(step5_ids.GROUP_SIM_PROGRESS, "style"), {"display": "flex"}, {"display": "none"}),
        (Output(step5_ids.STACK_SIMULATION_RESULTS, "children"), None, None),
    ],
    progress=[
        Output(step5_ids.TEXT_SIM_PROGRESS, "children"),
        Output(step5_ids.PROGRESS_SIM_PROGRESS, "value"),
    ],
    interval=200,  # update progress every 200ms
)
@composition
def step5_on_simulate(
    set_progress: Callable[[tuple[str, float]], None],  # progress_text, progress_pct
    _: int,
    main_store_data: dict,
):
    """Handle 'Run Simulation' button click to run the simulation and update progress."""
    # Simulate a long-running process with progress updates

    total_steps = 30
    for i in range(total_steps):
        time.sleep(0.5)  # Simulate work by sleeping
        set_progress((f"Running: {i + 1}/{total_steps} iterations", (i + 1) / total_steps * 100))

    # Once simulation is done, return the results visualization (placeholder for now)
    with dmc.Stack(None) as ret:
        yield dmc.Text("Simulation complete! Results go here.")

    # Return the children of the stack (populate the existing stack in the layout)
    # and reset the progress indicators (will be hidden with "display: none")
    return ret.children, f"Running: 0/{total_steps} iterations", 0


# endregion
