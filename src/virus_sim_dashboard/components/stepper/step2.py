"""Step 2: Scenario Configuration."""

from base64 import b64decode
from collections.abc import Generator
from io import BytesIO
from typing import Literal

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, Patch, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from plotly import graph_objects as go

from virus_sim_dashboard.components.common import main_ids, step2_ids, step3_ids
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT, jsonify

# Options for start timestamps
START_OPTS = [
    {"label": "First Positive Sample", "value": "FirstPosCollected"},
    {"label": "Admission", "value": "Admission"},
    {"label": "Whichever is earlier", "value": "Earlier"},
    {"label": "Whichever is later", "value": "Later"},
]
START_OPTS_TYPE = Literal[
    "FirstPosCollected",
    "Admission",
    "Earlier",
    "Later",
]


# region Layout
@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 2."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Title("Step 2: Patient settings", order=2, ta="center")
            yield start_timestamp_controls()
            yield dcc.Graph(
                id=step2_ids.GRAPH_DAILY_ARRIVALS,
                figure=go.Figure(layout=DEFAULT_FIGURE_LAYOUT),
            )
            with dmc.Group(align="center", gap="md"):
                yield dmc.Checkbox(
                    id=step2_ids.CHECKBOX_SHOW_ROLLING_AVG,
                    label="Show rolling average",
                    size="sm",
                    checked=True,
                )
                yield dmc.NumberInput(
                    id=step2_ids.NUMINPUT_ROLLING_AVG_DAYS,
                    label="Window size (days)",
                    min=2,
                    max=7,
                    step=1,
                    value=3,
                    size="sm",
                    w=150,
                )
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step2_ids.BTN_PREV)
            yield dmc.Button("Next", id=step2_ids.BTN_NEXT)
    return ret


@composition
def start_timestamp_controls() -> Generator[DashComponent, None, DashComponent]:
    """Controls for selecting options for start timestamps.

    The start timestamp for a patient can be chosen from either their admission time
    or the time of their first positive sample collection.
    """
    with dmc.Stack(None, gap="md", m=0, p=0) as ret:
        yield dmc.Title("Start timestamp option:", order=3)
        yield dmc.Select(
            id=step2_ids.SELECT_START_OPT_COMMUNITY,
            label="Community-acquired cases",
            description=(
                "Select the start timestamp for community-acquired cases. "
                'This includes all cases containing "community" in the Acquisition column '
                "(case insensitive)."
            ),
            data=START_OPTS,
            value="Admission",
            allowDeselect=False,
        )
        yield dmc.Select(
            id=step2_ids.SELECT_START_OPT_OTHER,
            label="Other cases",
            description=(
                "Select the start timestamp for other cases. "
                'This includes all cases not containing "community" in the Acquisition '
                "column (case insensitive)."
            ),
            data=START_OPTS,
            value="FirstPosCollected",
            allowDeselect=False,
        )
    return ret


# endregion


# region Callbacks
@callback(
    Output(step2_ids.GRAPH_DAILY_ARRIVALS, "figure", allow_duplicate=True),
    Output(step3_ids.GRAPH_DAILY_ARRIVALS, "figure", allow_duplicate=True),
    Input(step2_ids.SELECT_START_OPT_COMMUNITY, "value"),
    Input(step2_ids.SELECT_START_OPT_OTHER, "value"),
    Input(main_ids.STEPPER, "active"),
    Input(step2_ids.CHECKBOX_SHOW_ROLLING_AVG, "checked"),
    Input(step2_ids.NUMINPUT_ROLLING_AVG_DAYS, "value"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,  # OK since this is not the first step;
)
def update_daily_arrivals_graph(
    start_opt_community: START_OPTS_TYPE,
    start_opt_other: START_OPTS_TYPE,
    active_step: int,
    show_rolling_avg: bool,
    rolling_avg_days: int,
    main_store_data: dict,
) -> tuple[go.Figure, go.Figure]:
    """Update the daily arrivals graph based on selected start timestamp options.

    We also update the same graph in Step 3 to keep it consistent.  In other words,
    both elements of the returned tuple should be identical.
    """
    if active_step != 1:
        raise dash.exceptions.PreventUpdate  # Only update when on Step 2
    if "step1" not in main_store_data or "patient_stays" not in main_store_data["step1"]:
        raise dash.exceptions.PreventUpdate  # Missing data from Step 1

    stays_df = process_stay_data(
        main_store_data,
        start_opt_community,
        start_opt_other,
    )

    # Count daily arrivals and fill in gaps with 0
    stays_df = stays_df.set_index("Start")
    daily_arrivals = stays_df.resample("D").size()

    counts_df = pd.DataFrame(
        {
            "Date": daily_arrivals.index,
            "Daily Arrivals": daily_arrivals.to_numpy(),
        }
    ).set_index("Date")

    if show_rolling_avg:
        counts_df["Rolling Avg"] = (
            counts_df["Daily Arrivals"]
            .rolling(window=rolling_avg_days, min_periods=1, center=True)
            .mean()
        )

    disease_name = main_store_data.get("step1", {}).get("disease_name", "Disease")

    # Create the figure
    # Use Patch() to get the existing figure object and modify it
    # Both outputs are identical so only call Patch() once and reuse it for the second output
    patched_fig = Patch()
    patched_fig["layout"]["title"]["text"] = f"Daily Patient Arrivals, {disease_name}"
    patched_fig["data"] = []
    patched_fig["data"].append(
        go.Scatter(
            x=counts_df.index,
            y=counts_df["Daily Arrivals"],
            name="Daily Arrivals",
            mode="lines",
            line=dict(color="lightblue", width=1),
        )
    )
    if show_rolling_avg:
        patched_fig["data"].append(
            go.Scatter(
                x=counts_df.index,
                y=counts_df["Rolling Avg"],
                name=f"{rolling_avg_days}-day Rolling Avg",
                mode="lines",
                line=dict(color="orange", width=2),
            )
        )

    return patched_fig, patched_fig


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Input(step2_ids.BTN_PREV, "n_clicks"),
    prevent_initial_call=True,
)
def step2_on_prev(
    _: int,
) -> int:
    """Handle 'Previous' button click to go back a step."""
    return 0  # Go back to step 1 (index 0)


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Output(main_ids.MAIN_STORE, "data", allow_duplicate=True),
    Output(step3_ids.STORE_FIT_LOS_DISTS_RESULTS, "data", allow_duplicate=True),
    Input(step2_ids.BTN_NEXT, "n_clicks"),
    State(step2_ids.SELECT_START_OPT_COMMUNITY, "value"),
    State(step2_ids.SELECT_START_OPT_OTHER, "value"),
    State(main_ids.MAIN_STORE, "data"),
    State(step3_ids.STORE_FIT_LOS_DISTS_RESULTS, "data"),
    prevent_initial_call=True,
)
def step2_on_next(
    _: int,
    start_opt_community: START_OPTS_TYPE,
    start_opt_other: START_OPTS_TYPE,
    main_store_data: dict,
    los_fit_results: dict | None,
) -> tuple[int, dict, dict | None]:
    """Handle 'Next' button click to advance the stepper."""
    # Since start timestamp options are selected from dropdowns, they should always be valid
    # Store data is also always valid since validation is done when Update buttons are clicked
    step2_dict = {
        "start_opt_community": start_opt_community,
        "start_opt_other": start_opt_other,
    }

    if jsonify(main_store_data.get("step2", {})) != jsonify(step2_dict):
        main_store_data["step2"] = step2_dict
        # Invalidate and discard any data beyond step 2
        main_store_data = {k: v for k, v in main_store_data.items() if k in ["step1", "step2"]}
        los_fit_results = None  # Clear LOS fit results from Step 3

    return 2, main_store_data, los_fit_results  # Advance to step 3 (index 2)


# endregion


# region Helper functions
def process_stay_data(
    main_store_data: dict,
    start_opt_community: START_OPTS_TYPE,
    start_opt_other: START_OPTS_TYPE,
) -> pd.DataFrame:
    """Validate and preprocess the uploaded patient stay data."""
    encoded_stays = main_store_data["step1"]["patient_stays"]
    stays_df = pd.read_feather(BytesIO(b64decode(encoded_stays)))

    def start_datetime(row):
        is_community = "community" in str(row["Acquisition"]).lower()
        option = start_opt_community if is_community else start_opt_other
        if option == "Admission":
            return row["Admission"]
        if option == "FirstPosCollected":
            return row["FirstPosCollected"]
        if option == "Earlier":
            return min(row["Admission"], row["FirstPosCollected"])
        # option == "Later"
        return max(row["Admission"], row["FirstPosCollected"])

    stays_df["Start"] = stays_df.apply(start_datetime, axis=1)
    stays_df["Start"] = pd.to_datetime(stays_df["Start"])

    return stays_df


# endregion
