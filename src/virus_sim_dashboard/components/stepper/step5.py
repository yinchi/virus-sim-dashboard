"""Step 5: Simulation and Results Visualization."""

import json
import typing
from collections.abc import Generator
from io import BytesIO
from typing import Callable

import dash
import dash_mantine_components as dmc
import diskcache
import pandas as pd
from dash import DiskcacheManager, Input, NoUpdate, Output, Patch, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from plotly import graph_objects as go

from virus_sim_dashboard.components.common import main_ids, step5_ids
from virus_sim_dashboard.sim import (
    EnvironmentFactory,
    SimMultipleResult,
    get_quantiles,
    sim_multiple,
)
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT


# region layout
@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 5."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        yield dmc.Title("Step 5: Simulation and Results Visualization", order=2, ta="center")
        with dmc.Group(None, gap="md"):
            # yield dmc.Button(
            #     "Download full config (.json)",
            #     size="xl",
            #     id="temp-download-json-btn",
            # )
            yield dmc.Button(
                "Download full config (.xlsx)",
                disabled=True,
                size="xl",
                id=step5_ids.BTN_DOWNLOAD_CONFIG,
            )
            yield dcc.Download(id=step5_ids.DOWNLOAD_CONFIG)
            yield dmc.Button("Run simulation", color="green", size="xl", id=step5_ids.BTN_SIMULATE)
        # Progress indicator for simulation running status
        with dmc.Stack(None, id=step5_ids.STACK_SIM_PROGRESS, gap="md", style={"display": "none"}):
            yield dmc.Title("Simulation Progress", order=3)
            with dmc.Group(None, gap="md"):
                yield dmc.Text("0/30", id=step5_ids.TEXT_SIM_PROGRESS)
                yield dmc.Progress(
                    color="blue", id=step5_ids.PROGRESS_SIM_PROGRESS, size="xl", w=900, value=0
                )
        yield dcc.Store(id=step5_ids.STORE_SIMULATION_RESULTS)  # Store for simulation results

        with dmc.Stack(
            None,
            id=step5_ids.STACK_SIM_RESULTS,
            gap="md",
            style={"display": "none"},  # hide until we have results to show
        ):
            yield dmc.Title("Simulation Results", order=3)
            yield dmc.MultiSelect(
                # Since we don't know the age groups in advance, default to 0+
                size="md",
                data=[{"label": "0+", "value": "0+"}],
                value=["0+"],
                label="Filter by Age Groups",
                id=step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS,
                placeholder="Click here to select age groups",
            )
            yield dmc.Button(
                "Select All",
                id=step5_ids.BTN_OUTPUT_GROUPINGS_ALL,
            )
            yield dmc.Title("GIM Beds Occupancy (selected age groups)", order=4)

            layout_gim: dict[str, typing.Any] = DEFAULT_FIGURE_LAYOUT.copy()
            layout_gim["title"]["text"] = "GIM Beds Occupancy (daily maximum)"
            figure_gim = go.Figure(layout=layout_gim)
            yield dcc.Graph(
                id=step5_ids.GRAPH_GIM_BEDS_OCCUPANCY,
                figure=figure_gim,
            )

            yield dmc.Title("ICU Beds Occupancy (selected age groups)", order=4)

            layout_icu: dict[str, typing.Any] = DEFAULT_FIGURE_LAYOUT.copy()
            layout_icu["title"]["text"] = "ICU Beds Occupancy (daily maximum)"
            figure_icu = go.Figure(layout=layout_icu)
            yield dcc.Graph(
                id=step5_ids.GRAPH_ICU_BEDS_OCCUPANCY,
                figure=figure_icu,
            )

            with dmc.Group(None, gap="md"):
                yield dmc.Button(
                    "Download simulation results (.xlsx)",
                    disabled=True,
                    id=step5_ids.BTN_DOWNLOAD_SIM_RESULTS,
                )
                yield dcc.Download(id=step5_ids.DOWNLOAD_SIM_RESULTS)

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
    Output(step5_ids.DOWNLOAD_CONFIG, "data", allow_duplicate=True),
    Input("temp-download-json-btn", "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step5_on_download_json(
    _: int,
    main_store_data: dict,
) -> dict:
    """Handle 'Download JSON' button click to download the current config as a JSON file."""
    return {
        "filename": "config.json",
        "type": "application/json",
        "base64": False,
        "content": json.dumps(main_store_data, indent=4, sort_keys=False),
    }


@callback(
    Output(step5_ids.DOWNLOAD_CONFIG, "data", allow_duplicate=True),
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
    Output(step5_ids.STORE_SIMULATION_RESULTS, "data"),  # dict
    Output(step5_ids.TEXT_SIM_PROGRESS, "children"),  # str
    Output(step5_ids.PROGRESS_SIM_PROGRESS, "value"),  # float
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
        (Output(step5_ids.STACK_SIM_PROGRESS, "style"), {"display": "flex"}, {"display": "none"}),
        (Output(step5_ids.STACK_SIM_RESULTS, "style"), {"display": "none"}, {"display": "flex"}),
    ],
    progress=[
        Output(step5_ids.TEXT_SIM_PROGRESS, "children"),
        Output(step5_ids.PROGRESS_SIM_PROGRESS, "value"),
    ],
    interval=200,  # update progress every 200ms
)
def step5_on_simulate(
    set_progress: Callable[[tuple[str, float]], None],  # progress_text, progress_pct
    _: int,
    main_store_data: dict,
) -> tuple[dict, str, float]:
    """Handle 'Run Simulation' button click to run the simulation and update progress."""
    # Simulate a long-running process with progress updates

    n_runs = 30
    env_factory = EnvironmentFactory(main_store_data)
    results = sim_multiple(env_factory, n_runs=n_runs, set_progress=set_progress)

    # Return the simulation results data (to be stored in dcc.Store)
    # and reset the final progress text and progress bar value
    return results.to_dict(), f"Running: 0/{n_runs} iterations", 0


@callback(
    Output(step5_ids.BTN_DOWNLOAD_SIM_RESULTS, "disabled"),
    Input(step5_ids.STORE_SIMULATION_RESULTS, "data"),
    State(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "value"),
    prevent_initial_call=True,
)
def step5_update_download_button(
    simulation_results_data: dict,
    selected_age_groups: list[str],
) -> bool:
    """Enable or disable the 'Download Simulation Results' button."""
    # Disable if we don't have simulation results or if no age groups are selected
    return simulation_results_data is None or len(selected_age_groups) == 0


@callback(
    Output(step5_ids.DOWNLOAD_SIM_RESULTS, "data"),
    Input(step5_ids.BTN_DOWNLOAD_SIM_RESULTS, "n_clicks"),
    State(step5_ids.STORE_SIMULATION_RESULTS, "data"),
    State(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "value"),
    prevent_initial_call=True,
)
def step5_on_download_sim_results(
    _: int,
    simulation_results_data: dict,
    selected_age_groups: list[str],
) -> dict[str, typing.Any] | NoUpdate:
    """Handle button click to download the results as an Excel file."""
    if simulation_results_data is None:
        return dash.no_update  # No results to download
    if not selected_age_groups:
        return dash.no_update  # No age groups selected, so no data to download
    results = SimMultipleResult.from_dict(simulation_results_data)
    gim_summary = get_quantiles(results.gim, selected_age_groups)
    icu_summary = get_quantiles(results.icu, selected_age_groups)

    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            gim_summary.to_excel(writer, sheet_name="GIM Beds Occupancy")
            icu_summary.to_excel(writer, sheet_name="ICU Beds Occupancy")
        excel_bytes = buffer.getvalue()

    return dcc.send_bytes(
        excel_bytes,
        filename=f"simulation_results_{'_'.join(selected_age_groups)}.xlsx",
        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@callback(
    Output(step5_ids.STACK_SIM_RESULTS, "style"),
    Output(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "data"),
    Output(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "value", allow_duplicate=True),
    Input(step5_ids.STORE_SIMULATION_RESULTS, "data"),
    Input(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "value"),
    prevent_initial_call=True,
)
def step5_show_sim_results(
    simulation_results_data: dict,
    selected_age_groups: list[str],
) -> tuple[dict, list[dict[str, str]] | NoUpdate, list[str] | NoUpdate]:
    """Update the results stack with visualizations based on the simulation results."""
    if simulation_results_data is None:
        stack_display_style = {"display": "none"}  # No results to show yet
    else:
        stack_display_style = {"display": "flex"}  # Show the results stack when we have results

    sim_results = SimMultipleResult.from_dict(simulation_results_data)
    age_groups = sorted(sim_results.gim[0].columns)

    # If selected age groups is not a subset of available age groups, default to all age groups
    new_multiselect_data: list[dict[str, str]] | dash.NoUpdate
    new_multiselect_value: list[str] | dash.NoUpdate

    if not set(selected_age_groups).issubset(set(age_groups)):
        new_multiselect_data = [{"label": ag, "value": ag} for ag in age_groups]
        new_multiselect_value = list(age_groups)
    else:
        new_multiselect_data = dash.no_update
        new_multiselect_value = dash.no_update

    return (stack_display_style, new_multiselect_data, new_multiselect_value)


@callback(
    Output(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "value", allow_duplicate=True),
    Input(step5_ids.BTN_OUTPUT_GROUPINGS_ALL, "n_clicks"),
    State(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "data"),
    prevent_initial_call=True,
)
def step5_on_select_all_groupings(
    _: int,
    multiselect_data: list[dict[str, str]],
) -> list[str]:
    """Handle 'Select All' button click to select all age groups in the multiselect."""
    return sorted([item["value"] for item in multiselect_data])


@callback(
    Output(step5_ids.GRAPH_GIM_BEDS_OCCUPANCY, "figure", allow_duplicate=True),
    Output(step5_ids.GRAPH_ICU_BEDS_OCCUPANCY, "figure", allow_duplicate=True),
    Input(step5_ids.STORE_SIMULATION_RESULTS, "data"),
    Input(step5_ids.MULTISELECT_SIM_OUTPUT_GROUPINGS, "value"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step5_update_simulation_graphs(
    simulation_results_data: dict,
    selected_age_groups: list[str],
    main_store_data: dict,
) -> tuple[go.Figure, go.Figure]:
    """Update the GIM and ICU occupancy graphs."""
    disease_name = main_store_data.get("step1", {}).get("disease_name", "<unknown disease>")

    figure_gim = Patch()
    figure_icu = Patch()

    figure_gim["layout"]["title"]["text"] = (
        f"GIM Beds Occupancy for {disease_name} (selected age groups)"
    )
    figure_icu["layout"]["title"]["text"] = (
        f"ICU Beds Occupancy for {disease_name} (selected age groups)"
    )

    if simulation_results_data is None:
        return figure_gim, figure_icu  # Return empty figures if no data

    results = SimMultipleResult.from_dict(simulation_results_data)
    try:
        gim_summary = get_quantiles(results.gim, selected_age_groups)
        icu_summary = get_quantiles(results.icu, selected_age_groups)

        figure_gim["data"] = [
            go.Scatter(
                x=gim_summary.index,
                y=gim_summary[0.1],
                line_width=0,
                name="Bottom Decile",
            ),
            go.Scatter(
                x=gim_summary.index,
                y=gim_summary[0.9],
                line_width=0,
                name="Top Decile",
                fill="tonexty",
                fillcolor="rgba(127,127,255,0.5)",
            ),
            go.Scatter(
                x=gim_summary.index,
                y=gim_summary[0.25],
                line_width=0,
                name="Lower Quartile",
            ),
            go.Scatter(
                x=gim_summary.index,
                y=gim_summary[0.75],
                line_width=0,
                name="Upper Quartile",
                fill="tonexty",
                fillcolor="rgba(127,127,255,1)",
            ),
            go.Scatter(
                x=gim_summary.index,
                y=gim_summary[0.5],
                line_width=2,
                line_color="black",
                name="Median",
            ),
        ]

        figure_icu["data"] = [
            go.Scatter(
                x=icu_summary.index,
                y=icu_summary[0.1],
                line_width=0,
                name="Bottom Decile",
            ),
            go.Scatter(
                x=icu_summary.index,
                y=icu_summary[0.9],
                line_width=0,
                name="Top Decile",
                fill="tonexty",
                fillcolor="rgba(127,127,255,0.5)",
            ),
            go.Scatter(
                x=icu_summary.index,
                y=icu_summary[0.25],
                line_width=0,
                name="Lower Quartile",
            ),
            go.Scatter(
                x=icu_summary.index,
                y=icu_summary[0.75],
                line_width=0,
                name="Upper Quartile",
                fill="tonexty",
                fillcolor="rgba(127,127,255,1)",
            ),
            go.Scatter(
                x=icu_summary.index,
                y=icu_summary[0.5],
                line_width=2,
                line_color="black",
                name="Median",
            ),
        ]

        return figure_gim, figure_icu
    except KeyError:
        # Fix issue where this callback gets triggered before the multiselect options are
        # populated (defaulting to "0+"), causing a KeyError in get_quantiles
        raise dash.exceptions.PreventUpdate


# endregion
