"""Step 2: Scenario Configuration."""

from base64 import b64decode
from collections.abc import Generator
from io import BytesIO
from typing import Literal, NamedTuple

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, NoUpdate, Output, Patch, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from plotly import graph_objects as go

from virus_sim_dashboard.components.common import main_ids
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT, jsonify


# Component IDs
class StepTwoIDs(NamedTuple):
    """Component IDs used in Step 2."""

    BTN_PREV: str = "step2-btn-prev"
    BTN_NEXT: str = "step2-btn-next"

    SELECT_START_OPT_COMMUNITY: str = "step2-select-start-opt-community"
    SELECT_START_OPT_OTHER: str = "step2-select-start-opt-other"

    GRAPH_DAILY_ARRIVALS: dict[str, str] = {  # need dict for wildcard callbacks
        "type": "graph",
        "id": "step2-daily-arrivals",
    }
    CHECKBOX_SHOW_ROLLING_AVG: str = "step2-checkbox-show-rolling-avg"
    NUMINPUT_ROLLING_AVG_DAYS: str = "step2-numinput-rolling-avg-days"

    TEXTINPUT_AGE_BREAKPOINTS_GIM: str = "step2-textinput-age-breakpoints-gim"
    BTN_AGE_GROUPS_GIM_UPDATE: str = "step2-btn-age-groups-gim-update"
    TEXT_AGE_GROUPS_GIM: str = "step2-text-age-groups-gim"
    STORE_AGE_GROUPS_GIM: str = "step2-store-age-groups-gim"

    TEXTINPUT_AGE_BREAKPOINTS_ICU: str = "step2-textinput-age-breakpoints-icu"
    BTN_AGE_GROUPS_ICU_UPDATE: str = "step2-btn-age-groups-icu-update"
    TEXT_AGE_GROUPS_ICU: str = "step2-text-age-groups-icu"
    STORE_AGE_GROUPS_ICU: str = "step2-store-age-groups-icu"

    DATEPICKER_START: str = "step2-datepicker-start"
    DATEPICKER_END: str = "step2-datepicker-end"
    TABLE_PATIENT_COUNTS_GIM: str = "step2-table-patient-counts-gim"
    TABLE_PATIENT_COUNTS_ICU: str = "step2-table-patient-counts-icu"


step2_ids = StepTwoIDs()

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
            yield dmc.Title("Data summary", order=3)
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
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield age_group_section()
            yield patient_counts_section()
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
        yield dmc.Title("Start timestamp option:", order=4)
        yield dmc.Select(
            id=step2_ids.SELECT_START_OPT_COMMUNITY,
            label="Community-acquired cases",
            description=(
                "Select the start timestamp for community-acquired cases. "
                'This includes all cases containing "community" in the Acquisition column '
                "(case insensitive)."
            ),
            data=START_OPTS,
            value="FirstPosCollected",
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


@composition
def age_group_section() -> Generator[DashComponent, None, DashComponent]:
    """Controls for defining age groups."""
    with dmc.Stack(None, gap="md", m=0, p=0) as ret:
        yield dmc.Title("Age group settings", order=3)
        yield dmc.Text(
            "Define age group breakpoints for GIM-only and ICU patients. "
            'For example, "16,65" creates the age groups 0-15, 16-64, and 65+. '
            "An empty string will create a single age group covering all ages.",
            size="sm",
        )

        yield dmc.Title("GIM-only patients", order=4)
        with dmc.Group(align="flex-start"):
            yield dmc.TextInput(id=step2_ids.TEXTINPUT_AGE_BREAKPOINTS_GIM, w=300, value="16,65")
            yield dmc.Button("Update", id=step2_ids.BTN_AGE_GROUPS_GIM_UPDATE)
            with dmc.Group():
                yield dmc.Text("Current groups: ", fw=700, span=True)
                yield dmc.Text("0-15, 16-64, 65+", id=step2_ids.TEXT_AGE_GROUPS_GIM, span=True)
                yield dcc.Store(
                    id=step2_ids.STORE_AGE_GROUPS_GIM,
                    data=[
                        "Age >= 0 and Age < 16",
                        "Age >= 16 and Age < 65",
                        "Age >= 65",
                    ],  # match initial value of text input
                )

        yield dmc.Title("ICU patients", order=4)
        with dmc.Group(align="flex-start"):
            yield dmc.TextInput(id=step2_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, w=300, value="")
            yield dmc.Button("Update", id=step2_ids.BTN_AGE_GROUPS_ICU_UPDATE)
            with dmc.Group():
                yield dmc.Text("Current groups: ", fw=700, span=True)
                yield dmc.Text("0+", id=step2_ids.TEXT_AGE_GROUPS_ICU, span=True)
                yield dcc.Store(
                    id=step2_ids.STORE_AGE_GROUPS_ICU,
                    data=["Age >= 0"],  # match initial value of text input
                )
    return ret


@composition
def patient_counts_section() -> Generator[DashComponent, None, DashComponent]:
    """Controls and display components for patient counts, given a set of age group settings."""
    with dmc.Stack(None, gap="md", m=0, p=0) as ret:
        yield dmc.Title("Patient Counts", order=4)
        yield dmc.Text(
            "Age groups set in the current step will be used for length-of-stay distribution "
            "fitting in Step 3. It is recommended to have at least 30 patients in each group "
            "(reduce the number of age groups if necessary). You will have the option of combining "
            "groups in Step 3 for length-of-stay analysis.",
            size="sm",
        )
        with dmc.Group(align="flex-start", justify="space-between"):
            with dmc.Stack(gap="md"):
                yield dmc.Text("Analysis period", fw=700)
                with dmc.Group(align="flex-start", gap="lg"):
                    yield dmc.DatePickerInput(
                        id=step2_ids.DATEPICKER_START,
                        value="01 Jan 2023",
                        valueFormat="DD MMM YYYY",
                        label="Start date for arrivals",
                    )
                    yield dmc.DatePickerInput(
                        id=step2_ids.DATEPICKER_END,
                        value="01 Jun 2023",
                        valueFormat="DD MMM YYYY",
                        label="End date for arrivals",
                    )
            with dmc.Stack(gap="md"):
                yield dmc.Text("GIM-only patients", fw=700)
                yield dmc.Table(
                    id=step2_ids.TABLE_PATIENT_COUNTS_GIM,
                    w=300,
                    data={
                        "head": ["Age Group", "Survived", "Died"],
                        "body": [
                            ["0-15", 111, 11],
                            ["16-64", 222, 44],
                            ["65+", 555, 99],
                        ],
                    },
                )
            with dmc.Stack(gap="md"):
                yield dmc.Text("ICU patients", fw=700)
                yield dmc.Table(
                    id=step2_ids.TABLE_PATIENT_COUNTS_ICU,
                    w=300,
                    data={
                        "head": ["Age Group", "Survived", "Died"],
                        "body": [
                            ["0+", 111, 11],
                        ],
                    },
                )
    return ret


# endregion


# region Callbacks
@callback(
    Output(step2_ids.GRAPH_DAILY_ARRIVALS, "figure", allow_duplicate=True),
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
) -> go.Figure:
    """Update the daily arrivals graph based on selected start timestamp options."""
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
    patched_fig = Patch()  # use Patch() to get the existing figure object
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

    return patched_fig


@callback(
    Output(step2_ids.TEXT_AGE_GROUPS_GIM, "children"),
    Output(step2_ids.STORE_AGE_GROUPS_GIM, "data"),
    Output(step2_ids.TEXTINPUT_AGE_BREAKPOINTS_GIM, "error"),
    Input(step2_ids.BTN_AGE_GROUPS_GIM_UPDATE, "n_clicks"),
    State(step2_ids.TEXTINPUT_AGE_BREAKPOINTS_GIM, "value"),
)
def update_age_groups_gim(
    _n: int,
    breakpoints_str: str,
) -> tuple[str, dict[str, str], None] | tuple[NoUpdate, NoUpdate, str]:
    """Update displayed GIM age groups based on breakpoints input.

    Also sets an error message if the input is invalid, or clears it if valid.
    """
    try:
        breakpoints = [int(bp.strip()) for bp in breakpoints_str.split(",") if bp.strip() != ""]
        breakpoints = sorted(set(breakpoints))
        age_groups = compute_age_groups(breakpoints)
        return ", ".join(age_groups.keys()), age_groups, None
    except ValueError:
        return dash.no_update, dash.no_update, "Invalid breakpoints"


@callback(
    Output(step2_ids.TEXT_AGE_GROUPS_ICU, "children"),
    Output(step2_ids.STORE_AGE_GROUPS_ICU, "data"),
    Output(step2_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, "error"),
    Input(step2_ids.BTN_AGE_GROUPS_ICU_UPDATE, "n_clicks"),
    State(step2_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, "value"),
)
def update_age_groups_icu(
    _n: int,
    breakpoints_str: str,
) -> tuple[str, dict[str, str], None] | tuple[NoUpdate, NoUpdate, str]:
    """Update displayed ICU age groups based on breakpoints input.

    Also sets an error message if the input is invalid, or clears it if valid.
    """
    try:
        breakpoints = [int(bp.strip()) for bp in breakpoints_str.split(",") if bp.strip() != ""]
        breakpoints = sorted(set(breakpoints))
        age_groups = compute_age_groups(breakpoints)
        return ", ".join(age_groups.keys()), age_groups, None
    except ValueError:
        return dash.no_update, dash.no_update, "Invalid breakpoints"


@callback(
    Output(step2_ids.TABLE_PATIENT_COUNTS_GIM, "data"),
    Input(step2_ids.STORE_AGE_GROUPS_GIM, "data"),
    Input(step2_ids.DATEPICKER_START, "value"),
    Input(step2_ids.DATEPICKER_END, "value"),
    Input(main_ids.STEPPER, "active"),
    Input(step2_ids.SELECT_START_OPT_COMMUNITY, "value"),
    Input(step2_ids.SELECT_START_OPT_OTHER, "value"),
    State(main_ids.MAIN_STORE, "data"),
)
def update_patient_counts_gim(
    age_groups: dict[str, str],
    start_date: str,
    end_date: str,
    active_step: int,
    start_opt_community: START_OPTS_TYPE,
    start_opt_other: START_OPTS_TYPE,
    main_store_data: dict,
) -> dict:
    """Update GIM patient counts table based on age groups and analysis period."""
    if active_step != 1:
        raise dash.exceptions.PreventUpdate  # Only update when on Step 2
    if "step1" not in main_store_data or "patient_stays" not in main_store_data["step1"]:
        raise dash.exceptions.PreventUpdate  # Missing data from Step 1

    stays_df = process_stay_data(main_store_data, start_opt_community, start_opt_other)

    # Test: print total patients for Dec 27 2024
    print(
        "Total patients admitted on 27 Dec 2024:",
        (
            (stays_df["Start"] >= pd.to_datetime("2024-12-27"))
            & (stays_df["Start"] < pd.to_datetime("2024-12-28"))
        ).sum(),
    )

    print()

    gim_only_df = stays_df.loc[
        stays_df.ICUAdmission.isna()
        & (stays_df.Start >= pd.to_datetime(start_date))
        & (stays_df.Start < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    ]  # Filter to GIM-only patients in analysis period

    head = ["Age Group", "Survived", "Died"]
    body = []
    for age_group, query_str in age_groups.items():
        group_df = gim_only_df.query(query_str)  # Filter to patients in this age group
        is_dead = group_df.Summary.str.lower().isin(["dead", "deceased"])
        survived_count = (~is_dead).sum()
        died_count = is_dead.sum()
        body.append([age_group, int(survived_count), int(died_count)])

    return {"head": head, "body": body}


@callback(
    Output(step2_ids.TABLE_PATIENT_COUNTS_ICU, "data"),
    Input(step2_ids.STORE_AGE_GROUPS_ICU, "data"),
    Input(step2_ids.DATEPICKER_START, "value"),
    Input(step2_ids.DATEPICKER_END, "value"),
    Input(main_ids.STEPPER, "active"),
    Input(step2_ids.SELECT_START_OPT_COMMUNITY, "value"),
    Input(step2_ids.SELECT_START_OPT_OTHER, "value"),
    State(main_ids.MAIN_STORE, "data"),
)
def update_patient_counts_icu(
    age_groups: dict[str, str],
    start_date: str,
    end_date: str,
    active_step: int,
    start_opt_community: START_OPTS_TYPE,
    start_opt_other: START_OPTS_TYPE,
    main_store_data: dict,
) -> dict:
    """Update ICU patient counts table based on age groups and analysis period."""
    if active_step != 1:
        raise dash.exceptions.PreventUpdate  # Only update when on Step 2
    if "step1" not in main_store_data or "patient_stays" not in main_store_data["step1"]:
        raise dash.exceptions.PreventUpdate  # Missing data from Step 1

    stays_df = process_stay_data(main_store_data, start_opt_community, start_opt_other)
    icu_df = stays_df.loc[
        stays_df.ICUAdmission.notna()
        & (stays_df.Start >= pd.to_datetime(start_date))
        & (stays_df.Start < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    ]  # Filter to ICU patients in analysis period

    head = ["Age Group", "Survived", "Died"]
    body = []
    for age_group, query_str in age_groups.items():
        group_df = icu_df.query(query_str)  # Filter to patients in this age group
        is_dead = group_df.Summary.str.lower().isin(["dead", "deceased"])
        survived_count = (~is_dead).sum()
        died_count = is_dead.sum()
        body.append([age_group, int(survived_count), int(died_count)])

    return {"head": head, "body": body}


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
    Input(step2_ids.BTN_NEXT, "n_clicks"),
    State(step2_ids.SELECT_START_OPT_COMMUNITY, "value"),
    State(step2_ids.SELECT_START_OPT_OTHER, "value"),
    State(step2_ids.STORE_AGE_GROUPS_GIM, "data"),
    State(step2_ids.STORE_AGE_GROUPS_ICU, "data"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step2_on_next(
    _: int,
    start_opt_community: START_OPTS_TYPE,
    start_opt_other: START_OPTS_TYPE,
    age_groups_gim: dict[str, str],
    age_groups_icu: dict[str, str],
    main_store_data: dict,
) -> tuple[int, dict]:
    """Handle 'Next' button click to advance the stepper."""
    # Since start timestamp options are selected from dropdowns, they should always be valid
    # Store data is also always valid since validation is done when Update buttons are clicked
    step2_dict = {
        "start_opt_community": start_opt_community,
        "start_opt_other": start_opt_other,
        "age_groups_gim": age_groups_gim,
        "age_groups_icu": age_groups_icu,
    }

    if jsonify(main_store_data.get("step2", {})) != jsonify(step2_dict):
        main_store_data["step2"] = step2_dict
        # Invalidate and discard any data beyond step 2
        main_store_data = {k: v for k, v in main_store_data.items() if k in ["step1", "step2"]}

    return 2, main_store_data  # Advance to step 3 (index 2)


# endregion


# region Helper functions
def compute_age_groups(breakpoints: list[int]) -> dict[str, str]:
    """Compute age group strings from breakpoints.

    Breakpoints are left-inclusive and right-exclusive, except for the last group (n and above).

    Args:
        breakpoints: A list of integer breakpoints.

    Returns:
        A dict where keys are age group strings and values are corresponding pandas query strings.
    """
    age_groups = {}
    lower_bound = 0
    for bp in breakpoints:
        age_groups[f"{lower_bound}-{bp - 1}"] = f"Age >= {lower_bound} and Age < {bp}"
        lower_bound = bp
    age_groups[f"{lower_bound}+"] = f"Age >= {lower_bound}"
    return age_groups


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
