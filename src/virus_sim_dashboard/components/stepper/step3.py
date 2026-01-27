"""Step 3: Length-of-Stay fitting."""

from collections.abc import Generator

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, NoUpdate, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components.common import main_ids, step3_ids
from virus_sim_dashboard.components.stepper.step2 import START_OPTS, process_stay_data
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT

start_opts_map = {opt["value"]: opt["label"] for opt in START_OPTS}


# region Layout
@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 3."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Title("Step 3: Length-of-Stay fitting", order=2, ta="center")
            with dmc.TypographyStylesProvider():
                yield dcc.Markdown(
                    id=step3_ids.MARKDOWN_CONFIG_SUMMARY,
                    children="Data summary will be displayed here.",
                )
        yield dcc.Graph(
            id=step3_ids.GRAPH_DAILY_ARRIVALS,
            figure=go.Figure(layout=DEFAULT_FIGURE_LAYOUT),
        )
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Title("Fitting period", order=3)
            with dmc.Group(None, gap="lg"):
                yield dmc.DatePickerInput(
                    id=step3_ids.DATEPICKER_START,
                    value="01 Jan 2023",
                    valueFormat="DD MMM YYYY",
                    label="Start date for arrivals",
                )
                yield dmc.DatePickerInput(
                    id=step3_ids.DATEPICKER_END,
                    value="01 Jun 2023",
                    valueFormat="DD MMM YYYY",
                    label="End date for arrivals",
                )
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield age_group_section()
            yield patient_counts_section()
        yield los_groupings_section()
        with dmc.Group(None, justify="center"):
            yield dmc.Button(
                "Fit Length-of-Stay Distributions",
                id=step3_ids.BTN_FIT_LOS_DISTS,
                size="xl",
                radius="md",
                color="orange",
            )
            yield dcc.Store(
                id=step3_ids.STORE_FIT_LOS_DISTS_RESULTS,
                data=None,  # to be filled with fitting results
            )
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step3_ids.BTN_PREV)
            yield dmc.Button("Next", id=step3_ids.BTN_NEXT)
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
            yield dmc.TextInput(id=step3_ids.TEXTINPUT_AGE_BREAKPOINTS_GIM, w=300, value="16,65")
            yield dmc.Button("Update", id=step3_ids.BTN_AGE_GROUPS_GIM_UPDATE)
            with dmc.Group():
                yield dmc.Text("Current groups: ", fw=700, span=True)
                yield dmc.Text("0-15, 16-64, 65+", id=step3_ids.TEXT_AGE_GROUPS_GIM, span=True)
                yield dcc.Store(
                    id=step3_ids.STORE_AGE_GROUPS_GIM,
                    data=[
                        "Age >= 0 and Age < 16",
                        "Age >= 16 and Age < 65",
                        "Age >= 65",
                    ],  # match initial value of text input
                )

        yield dmc.Title("ICU patients", order=4)
        with dmc.Group(align="flex-start"):
            yield dmc.TextInput(id=step3_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, w=300, value="")
            yield dmc.Button("Update", id=step3_ids.BTN_AGE_GROUPS_ICU_UPDATE)
            with dmc.Group():
                yield dmc.Text("Current groups: ", fw=700, span=True)
                yield dmc.Text("0+", id=step3_ids.TEXT_AGE_GROUPS_ICU, span=True)
                yield dcc.Store(
                    id=step3_ids.STORE_AGE_GROUPS_ICU,
                    data=["Age >= 0"],  # match initial value of text input
                )
    return ret


@composition
def patient_counts_section() -> Generator[DashComponent, None, DashComponent]:
    """Controls and display components for patient counts, given a set of age group settings."""
    with dmc.Stack(None, gap="md", m=0, p=0) as ret:
        yield dmc.Title("Patient Counts", order=4)
        with dmc.Group(align="flex-start", gap="xl"):
            with dmc.Stack(gap="md"):
                yield dmc.Text("GIM-only patients", fw=700)
                yield dmc.Table(
                    id=step3_ids.TABLE_PATIENT_COUNTS_GIM,
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
                    id=step3_ids.TABLE_PATIENT_COUNTS_ICU,
                    w=300,
                    data={
                        "head": ["Age Group", "Survived", "Died"],
                        "body": [
                            ["0+", 111, 11],
                        ],
                    },
                )
    return ret


@composition
def los_groupings_section() -> Generator[DashComponent, None, DashComponent]:
    """Controls for defining length-of-stay groupings."""
    with dmc.Stack(None, gap="md", m=0, p=0) as ret:
        yield dmc.Title("Length-of-Stay Groupings", order=3)
        yield dmc.Text(
            "Define how age groups will be combined for length-of-stay analysis. "
            "Groups assigned the same numeric label will be combined. "
            "It is recommended to have at least 30 patients in each combined group.",
            size="sm",
        )
        with dmc.Group(align="flex-start", gap="xl"):
            with dmc.Stack(gap="md"):
                yield dmc.Text("GIM-only patients", fw=700)
                yield dmc.Table(
                    id=step3_ids.TABLE_LOS_GROUPS_GIM,
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
                    id=step3_ids.TABLE_LOS_GROUPS_ICU,
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
    Output(step3_ids.MARKDOWN_CONFIG_SUMMARY, "children"),
    Input(main_ids.MAIN_STORE, "data"),
)
def update_config_summary(
    main_store_data: dict,
) -> str:
    """Update the configuration summary based on main store data."""
    if not main_store_data or "step2" not in main_store_data:
        raise dash.exceptions.PreventUpdate  # Step 2 data not available yet

    start_opt_community = main_store_data["step2"]["start_opt_community"]
    start_opt_other = main_store_data["step2"]["start_opt_other"]

    return f"""\
### Configuration Summary:

- **Start timestamp for community-acquired cases:**
  {start_opts_map.get(start_opt_community, start_opt_community)}
- **Start timestamp for other cases:** {start_opts_map.get(start_opt_other, start_opt_other)}
"""


@callback(
    Output(step3_ids.TEXT_AGE_GROUPS_GIM, "children"),
    Output(step3_ids.STORE_AGE_GROUPS_GIM, "data"),
    Output(step3_ids.TEXTINPUT_AGE_BREAKPOINTS_GIM, "error"),
    Input(step3_ids.BTN_AGE_GROUPS_GIM_UPDATE, "n_clicks"),
    State(step3_ids.TEXTINPUT_AGE_BREAKPOINTS_GIM, "value"),
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
    Output(step3_ids.TEXT_AGE_GROUPS_ICU, "children"),
    Output(step3_ids.STORE_AGE_GROUPS_ICU, "data"),
    Output(step3_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, "error"),
    Input(step3_ids.BTN_AGE_GROUPS_ICU_UPDATE, "n_clicks"),
    State(step3_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, "value"),
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
    Output(step3_ids.TABLE_PATIENT_COUNTS_GIM, "data"),
    Input(step3_ids.STORE_AGE_GROUPS_GIM, "data"),
    Input(step3_ids.DATEPICKER_START, "value"),
    Input(step3_ids.DATEPICKER_END, "value"),
    Input(main_ids.STEPPER, "active"),
    State(main_ids.MAIN_STORE, "data"),
)
def update_patient_counts_gim(
    age_groups: dict[str, str],
    start_date: str,
    end_date: str,
    active_step: int,
    main_store_data: dict,
) -> dict:
    """Update GIM patient counts table based on age groups and analysis period."""
    if active_step != 2:
        raise dash.exceptions.PreventUpdate  # Only update when on Step 3
    if "step1" not in main_store_data or "patient_stays" not in main_store_data["step1"]:
        raise dash.exceptions.PreventUpdate  # Missing data from Step 1

    if (
        "step2" not in main_store_data
        or "start_opt_community" not in main_store_data["step2"]
        or "start_opt_other" not in main_store_data["step2"]
    ):
        raise dash.exceptions.PreventUpdate  # Missing data from Step 2

    start_opt_community = main_store_data["step2"]["start_opt_community"]
    start_opt_other = main_store_data["step2"]["start_opt_other"]

    stays_df = process_stay_data(main_store_data, start_opt_community, start_opt_other)

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
    Output(step3_ids.TABLE_PATIENT_COUNTS_ICU, "data"),
    Input(step3_ids.STORE_AGE_GROUPS_ICU, "data"),
    Input(step3_ids.DATEPICKER_START, "value"),
    Input(step3_ids.DATEPICKER_END, "value"),
    Input(main_ids.STEPPER, "active"),
    State(main_ids.MAIN_STORE, "data"),
)
def update_patient_counts_icu(
    age_groups: dict[str, str],
    start_date: str,
    end_date: str,
    active_step: int,
    main_store_data: dict,
) -> dict:
    """Update ICU patient counts table based on age groups and analysis period."""
    if active_step != 2:
        raise dash.exceptions.PreventUpdate  # Only update when on Step 3
    if "step1" not in main_store_data or "patient_stays" not in main_store_data["step1"]:
        raise dash.exceptions.PreventUpdate  # Missing data from Step 1

    if (
        "step2" not in main_store_data
        or "start_opt_community" not in main_store_data["step2"]
        or "start_opt_other" not in main_store_data["step2"]
    ):
        raise dash.exceptions.PreventUpdate  # Missing data from Step 2

    start_opt_community = main_store_data["step2"]["start_opt_community"]
    start_opt_other = main_store_data["step2"]["start_opt_other"]

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
    Output(step3_ids.TABLE_LOS_GROUPS_GIM, "children"),
    Input(step3_ids.STORE_AGE_GROUPS_GIM, "data"),
)
def update_los_groups_gim(
    age_groups: dict[str, str],
) -> list[DashComponent]:
    """Update GIM length-of-stay groupings table based on age groups.

    The groupings table provides a set of number inputs using the same table structure as the
    patient counts table in component `step3_ids.TABLE_PATIENT_COUNTS_GIM`.
    """
    table_cols = ["Age Group", "Survived", "Died"]
    table_head = dmc.TableThead(dmc.TableTr([dmc.TableTh(col) for col in table_cols]))
    num_input_opts = {
        "min": 1,
        "step": 1,
        "allowNegative": False,
        "allowDecimal": False,
        "w": 80,
        "hideControls": True,
    }

    table_body_rows = []

    counter = 1
    for age_group in age_groups:
        items = [
            age_group,
            dmc.NumberInput(
                id={
                    "type": "los-grouping-input",
                    "pathway": "gim/survived",
                    "age_group": age_group,
                },
                value=counter,
                **num_input_opts,
            ),
            dmc.NumberInput(
                id={
                    "type": "los-grouping-input",
                    "pathway": "gim/died",
                    "age_group": age_group,
                },
                value=counter + 1,
                **num_input_opts,
            ),
        ]
        table_body_rows.append([dmc.TableTd(item) for item in items])
        counter += 2
    table_body = dmc.TableTbody([dmc.TableTr(row) for row in table_body_rows])

    # Return both thead and tbody as children of the table
    return [table_head, table_body]


@callback(
    Output(step3_ids.TABLE_LOS_GROUPS_ICU, "children"),
    Input(step3_ids.STORE_AGE_GROUPS_ICU, "data"),
)
def update_los_groups_icu(
    age_groups: dict[str, str],
) -> list[DashComponent]:
    """Update ICU length-of-stay groupings table based on age groups.

    The groupings table provides a set of number inputs using the same table structure as the
    patient counts table in component `step3_ids.TABLE_PATIENT_COUNTS_ICU`.
    """
    table_cols = ["Age Group", "Survived", "Died"]
    table_head = dmc.TableThead(dmc.TableTr([dmc.TableTh(col) for col in table_cols]))
    num_input_opts = {
        "min": 1,
        "step": 1,
        "allowNegative": False,
        "allowDecimal": False,
        "w": 80,
        "hideControls": True,
    }

    table_body_rows = []

    counter = 1
    for age_group in age_groups:
        items = [
            age_group,
            dmc.NumberInput(
                id={
                    "type": "los-grouping-input",
                    "pathway": "icu/survived",
                    "age_group": age_group,
                },
                value=counter,
                **num_input_opts,
            ),
            dmc.NumberInput(
                id={
                    "type": "los-grouping-input",
                    "pathway": "icu/died",
                    "age_group": age_group,
                },
                value=counter + 1,
                **num_input_opts,
            ),
        ]
        table_body_rows.append([dmc.TableTd(item) for item in items])
        counter += 2
    table_body = dmc.TableTbody([dmc.TableTr(row) for row in table_body_rows])

    # Return both thead and tbody as children of the table
    return [table_head, table_body]


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


# endregion
