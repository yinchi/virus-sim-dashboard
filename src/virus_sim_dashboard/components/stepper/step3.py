"""Step 3: Length-of-Stay fitting."""

from base64 import b64encode
from collections.abc import Generator
from io import BytesIO
from typing import Literal, cast

import dash
import dash_mantine_components as dmc
import matplotlib
import pandas as pd
import plotly.graph_objects as go
import reliability
from dash import Input, NoUpdate, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import Series

from virus_sim_dashboard.components.common import main_ids, step3_ids
from virus_sim_dashboard.components.stepper.step2 import START_OPTS, process_stay_data
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT, jsonify

start_opts_map = {opt["value"]: opt["label"] for opt in START_OPTS}

LosGroupTuple = tuple[
    Literal["gim", "icu"],  # ward type
    Literal["survived", "died"],  # outcome
    str,  # age group string
]

matplotlib.use("Agg")


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
                    value="01 Aug 2024",
                    valueFormat="DD MMM YYYY",
                    label="Start date for arrivals",
                )
                yield dmc.DatePickerInput(
                    id=step3_ids.DATEPICKER_END,
                    value="31 May 2025",
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
        with dmc.Stack(None, id=step3_ids.STACK_LOS_FIT_RESULTS, gap="md", m=0, p=0):
            yield dmc.Title("Length-of-Stay Fitting Results", order=3)
            with dmc.Stack(
                None,
                gap="md",
                m=0,
                p=0,
                id=step3_ids.STACK_LOS_FIT_RESULTS,
            ):
                yield dmc.Text(
                    "No results yet, or results have been cleared due to changes in previous "
                    "inputs.",
                    c="red",
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
    Output(step3_ids.STORE_FIT_LOS_DISTS_RESULTS, "data"),
    Input(step3_ids.BTN_FIT_LOS_DISTS, "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    State(step3_ids.DATEPICKER_START, "value"),
    State(step3_ids.DATEPICKER_END, "value"),
    State({"type": "los-grouping-input", "pathway": dash.ALL, "age_group": dash.ALL}, "value"),
    State({"type": "los-grouping-input", "pathway": dash.ALL, "age_group": dash.ALL}, "id"),
)
def fit_los_dists(
    _n: int,
    main_store_data: dict,
    start_date_str: str,
    end_date_str: str,
    los_grouping_values: list[int],
    los_grouping_ids: list[dict[str, str]],
) -> dict:
    """Fit length-of-stay distributions based on user-defined groupings.

    Returns:
        A dict containing fitting results.
    """
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

    start_date = pd.to_datetime(start_date_str)
    # Include any time within the end date by adding one day and using "<" comparison
    end_date = pd.to_datetime(end_date_str) + pd.Timedelta(days=1)

    stays_df = stays_df.loc[
        (stays_df.Start >= start_date) & (stays_df.Start < end_date)
    ]  # Filter to analysis period

    # A mapping from LOS groupings to an integer label
    los_grouping_map: dict[LosGroupTuple, int] = {}
    for value, id_dict in zip(los_grouping_values, los_grouping_ids):
        assert "pathway" in id_dict and "age_group" in id_dict, "Invalid los grouping id"

        _source, _outcome = id_dict["pathway"].split("/")
        source = cast(Literal["gim", "icu"], _source)
        outcome = cast(Literal["survived", "died"], _outcome)
        los_grouping_map[(source, outcome, id_dict["age_group"])] = value

    try:
        return fit_los_dists_helper(stays_df, los_grouping_map)
    except NotImplementedError:
        print("LOS fitting not implemented yet.")
        return {"error": "LOS fitting not implemented yet."}
    except Exception as e:
        print(f"Error during LOS fitting: {e}")
        return {"error": str(e)}


@callback(
    Output(step3_ids.STACK_LOS_FIT_RESULTS, "children"),
    Input(step3_ids.STORE_FIT_LOS_DISTS_RESULTS, "data"),
)
def update_los_fit_results_display(
    fit_results: dict | None,
) -> list[DashComponent]:
    """Update the LOS fitting results display based on fitting results data."""
    if fit_results is None:
        return [
            dmc.Text(
                "No results yet, or results have been cleared due to changes in previous inputs.",
                c="red",
            )
        ]
    if "error" in fit_results and fit_results["error"] is not None:
        return [dmc.Text(f"Error during fitting: {fit_results['error']}", c="red")]

    return display_results_components(fit_results)


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Output(main_ids.MAIN_STORE, "data", allow_duplicate=True),
    Input(step3_ids.BTN_NEXT, "n_clicks"),
    State(main_ids.MAIN_STORE, "data"),
    State(step3_ids.STORE_FIT_LOS_DISTS_RESULTS, "data"),
    prevent_initial_call=True,
)
def step3_on_next(
    _: int,
    main_store_data: dict,
    los_fit_results: dict | None,
) -> tuple[int, dict]:
    """Handle 'Next' button click to advance the stepper."""
    if los_fit_results is None or ("error" in los_fit_results and los_fit_results["error"]):
        # Cannot proceed without valid LOS fitting results
        raise dash.exceptions.PreventUpdate

    step3_dict = {
        "los_fit_results": los_fit_results,
    }
    if jsonify(main_store_data.get("step3", {})) != jsonify(step3_dict):
        main_store_data["step3"] = step3_dict
        # Invalidate and discard any data beyond step 3
        main_store_data = {
            k: v for k, v in main_store_data.items() if k in ["step1", "step2", "step3"]
        }
    return 3, main_store_data  # Advance to step 4 (index 3)


@callback(
    Output(step3_ids.BTN_NEXT, "disabled"),
    Input(step3_ids.STORE_FIT_LOS_DISTS_RESULTS, "data"),
)
def update_next_button_state(
    fit_results: dict | None,
) -> bool:
    """Enable 'Next' button only if LOS fitting results are available and no errors occurred."""
    return fit_results is None or ("error" in fit_results and fit_results["error"])


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


DAY = pd.Timedelta(days=1)


def fit_los_dists_helper(
    stays_df: pd.DataFrame,
    los_grouping_map: dict[LosGroupTuple, int],
) -> dict:
    """Fit length-of-stay distributions based on provided groupings.

    Args:
        stays_df: DataFrame containing patient stay data.
        los_grouping_map: A mapping from LOS groupings to integer labels.
    """
    print()
    print("Starting length-of-stay distribution fitting...")

    # For each unique label in los_grouping_map.values(), which keys correspond to it?
    los_mapping_gim: dict[int, list[LosGroupTuple]] = {}
    los_mapping_icu: dict[int, list[LosGroupTuple]] = {}
    for key, label in los_grouping_map.items():
        if key[0] == "gim":
            los_mapping_gim.setdefault(label, []).append(key)
        else:
            los_mapping_icu.setdefault(label, []).append(key)

    los_mapping_gim = {k: sorted(los_mapping_gim[k]) for k in sorted(los_mapping_gim)}
    los_mapping_icu = {k: sorted(los_mapping_icu[k]) for k in sorted(los_mapping_icu)}

    # Temp: save the dataframe so we can test fitting externally before implementing here
    # stays_df.to_feather("assets/private/stays_df_for_fitting.feather")
    # print("Saved stays_df to assets/private/stays_df_for_fitting.feather for external fitting.")

    data = stays_df.copy()

    # Clean up ICU dates (ensure ICUAdmission and ICUDischarge are after Start)
    data.loc[data.ICUAdmission < data.Start, "ICUAdmission"] = data.loc[
        data.ICUAdmission < data.Start, "Start"
    ]
    data.loc[data.ICUDischarge < data.Start, ["ICUAdmission", "ICUDischarge"]] = pd.NaT

    # GIM-only patients
    # For each label, extract the subset of patients corresponding to that LOS grouping
    los_data_gim = {}
    for label in los_mapping_gim:
        query_fragments = []
        print(f'Processing GIM LOS grouping label "{label}"...')
        keys = los_mapping_gim[label]
        label_groups = {}
        n_patients = 0
        for key in keys:
            query = query_str(key)  # Convert key (e.g. ("gim", "died", "65+")) to a query string
            query_fragments.append(f"({query})")
            subset = data.query(query)
            n_patients += len(subset)
            label_groups[key] = {
                "count": len(subset),
                # Proportion: this subset relative to all patients
                "p_total": len(subset) / len(data),
            }
        for key, val in label_groups.items():
            # Compute proportion within this LOS grouping
            val["p_in_label"] = val["count"] / n_patients if n_patients > 0 else 0.0
        # Make JSON-able label group keys
        label_groups_jsonable = {str(k): v for k, v in label_groups.items()}
        combined_query = " or ".join(query_fragments)
        filtered_data = data.query(combined_query)
        if len(filtered_data) < 3:
            # Distribution for LOS fitting has 3 parameters, so need at least 3 data points
            raise ValueError(
                f"Not enough patients ({len(filtered_data)}) for LOS fitting in GIM LOS "
                f"grouping label {label}"
            )
        los_data_gim[label] = {
            "n_patients": n_patients,
            "probability": n_patients / len(data),
            "label_groups": label_groups_jsonable,
            "query": combined_query,
        }
        los_series: Series[pd.Timedelta] = filtered_data.Discharge - filtered_data.Start
        los_series_days = los_series / DAY
        fit_result = reliability.Fitters.Fit_Lognormal_3P(
            los_series_days.to_numpy(),
            show_probability_plot=True,
            print_results=False,
        )
        ax = fit_result.probability_plot
        ax.set_title(f"GIM LOS Fit for GIM LOS Grouping Label {label}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Proportion of Patients discharged")
        fig_str = fig_to_b64(ax.figure)
        plt.close(ax.figure)
        los_data_gim[label]["los_fit"] = {
            "distribution": "Lognormal_3P",
            "parameters": fit_result.distribution.parameters,
            "pp_plot_b64": fig_str,
        }

    # pprint(los_data_gim)

    # ICU patients
    los_data_icu = {}
    for label in los_mapping_icu:
        query_fragments = []
        print(f'Processing ICU LOS grouping label "{label}"...')
        keys = los_mapping_icu[label]
        n_patients = 0
        label_groups = {}
        for key in keys:
            query = query_str(key)  # Convert key (e.g. ("icu", "died", "65+")) to a query string
            query_fragments.append(f"({query})")
            subset = data.query(query)
            n_patients += len(subset)
            label_groups[key] = {
                "count": len(subset),
                # Proportion: this subset relative to all patients
                "p_total": len(subset) / len(data),
            }
        for key, val in label_groups.items():
            # Compute proportion within this LOS grouping
            val["p_in_label"] = val["count"] / n_patients if n_patients > 0 else 0.0
        # Make JSON-able label group keys
        label_groups_jsonable = {str(k): v for k, v in label_groups.items()}
        combined_query = " or ".join(query_fragments)
        filtered_data = data.query(combined_query)
        if len(filtered_data) < 3:
            # Distribution for LOS fitting has 3 parameters, so need at least 3 data points
            raise ValueError(
                f"Not enough patients ({len(filtered_data)} < 3) for LOS fitting in ICU LOS "
                f"grouping label {label}"
            )
        los_data_icu[label] = {
            "n_patients": n_patients,
            "probability": n_patients / len(data),
            "label_groups": label_groups_jsonable,
            "query": combined_query,
            # "data": filtered_data,
        }

        # Pre-ICU LOS
        n_icu = len(filtered_data)
        df_with_pre_icu = filtered_data.loc[filtered_data.ICUAdmission > filtered_data.Start]
        n_with_pre_icu = len(df_with_pre_icu)
        los_data_icu[label]["prob_pre_icu"] = n_with_pre_icu / n_icu if n_icu > 0 else 0.0

        pre_icu_series: Series[pd.Timedelta] = df_with_pre_icu.ICUAdmission - df_with_pre_icu.Start
        pre_icu_series_days = pre_icu_series / DAY
        if len(pre_icu_series_days) < 3:
            raise ValueError(
                f"Not enough patients ({len(pre_icu_series_days)} < 3) for Pre-ICU LOS fitting "
                f"in ICU LOS grouping label {label}"
            )
        fit_result = reliability.Fitters.Fit_Lognormal_3P(
            pre_icu_series_days.to_numpy(),
            show_probability_plot=True,
            print_results=False,
        )
        ax = fit_result.probability_plot
        ax.set_title(f"Pre-ICU LOS Fit for ICU LOS Grouping Label {label}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Proportion of Patients discharged")
        fig_str = fig_to_b64(ax.figure)
        plt.close(ax.figure)
        los_data_icu[label]["pre_icu_los_fit"] = {
            "distribution": "Lognormal_3P",
            "parameters": fit_result.distribution.parameters,
            "pp_plot_b64": fig_str,
        }

        # ICU LOS
        icu_series: Series[pd.Timedelta] = filtered_data.ICUDischarge - filtered_data.ICUAdmission
        icu_series_days = icu_series / DAY
        if len(icu_series_days) < 3:
            raise ValueError(
                f"Not enough patients ({len(icu_series_days)} < 3) for ICU LOS fitting "
                f"in ICU LOS grouping label {label}"
            )
        fit_result = reliability.Fitters.Fit_Lognormal_3P(
            icu_series_days.to_numpy(),
            show_probability_plot=True,
            print_results=False,
        )
        ax = fit_result.probability_plot
        ax.set_title(f"ICU LOS Fit for ICU LOS Grouping Label {label}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Proportion of Patients discharged")
        fig_str = fig_to_b64(ax.figure)
        plt.close(ax.figure)
        los_data_icu[label]["icu_los_fit"] = {
            "distribution": "Lognormal_3P",
            "parameters": fit_result.distribution.parameters,
            "pp_plot_b64": fig_str,
        }

        # Post-ICU LOS
        df_with_post_icu = filtered_data.loc[filtered_data.ICUDischarge < filtered_data.Discharge]
        n_with_post_icu = len(df_with_post_icu)
        los_data_icu[label]["prob_post_icu"] = n_with_post_icu / n_icu if n_icu > 0 else 0.0

        post_icu_series: Series[pd.Timedelta] = (
            df_with_post_icu.Discharge - df_with_post_icu.ICUDischarge
        )
        post_icu_series_days = post_icu_series / DAY
        if len(post_icu_series_days) < 3:
            return {
                "error": f"Not enough patients ({len(post_icu_series_days)} < 3) for Post-ICU LOS "
                f"fitting in ICU LOS grouping label {label}"
            }
        fit_result = reliability.Fitters.Fit_Lognormal_3P(
            post_icu_series_days.to_numpy(),
            show_probability_plot=True,
            print_results=False,
        )
        ax = fit_result.probability_plot
        ax.set_title(f"Post-ICU LOS Fit for ICU LOS Grouping Label {label}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Proportion of Patients discharged")
        fig_str = fig_to_b64(ax.figure)
        plt.close(ax.figure)
        los_data_icu[label]["post_icu_los_fit"] = {
            "distribution": "Lognormal_3P",
            "parameters": fit_result.distribution.parameters,
            "pp_plot_b64": fig_str,
        }

    # pprint(los_data_icu)

    return {
        "error": None,
        "gim": los_data_gim,
        "icu": los_data_icu,
    }


def query_str(los_tuple: LosGroupTuple) -> str:
    """Convert a LOS grouping tuple to a pandas query string.

    For example, ("gim", "died", "65+") becomes
    "ICUAdmission.isnull() and Summary.str.lower() in ['dead', 'deceased'] and Age >= 65"
    """
    ward, outcome, age_group = los_tuple
    conditions = []

    # GIM/ICU
    if ward == "icu":
        conditions.append("ICUAdmission.notnull()")
    else:
        conditions.append("ICUAdmission.isnull()")

    # Outcome
    if outcome == "died":
        conditions.append("Summary.str.lower() in ['dead', 'deceased']")
    else:
        conditions.append("Summary.str.lower() not in ['dead', 'deceased']")

    # Age group
    if age_group.endswith("+"):
        age_limit = int(age_group[:-1])
        conditions.append(f"Age >= {age_limit}")
    else:
        age_min, age_max = age_group.split("-")
        # age_min, age_max should be strings representing integers
        conditions.append(f"Age >= {int(age_min)} and Age < {int(age_max) + 1}")
    return " and ".join(conditions)


def fig_to_b64(fig: Figure) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG string."""
    fig_out = BytesIO()
    fig.savefig(fig_out, format="png")
    return b64encode(fig_out.getvalue()).decode("utf-8")


@composition
def display_results_components(
    fit_results: dict,
) -> Generator[DashComponent, None, list[DashComponent]]:
    """Create display components for LOS fitting results.

    Components will be embedded in a dmc.Stack.

    Args:
        fit_results: The fitting results dict returned by `fit_los_dists_helper`.
    """
    with dmc.Tabs(
        None,
        orientation="vertical",
        value="gim-1",
        styles={
            "tabLabel": {"text-align": "right"},
        },
    ) as ret:
        with dmc.TabsList(None):
            yield dmc.TabsTab("GIM-only patients", value="gim", disabled=True, ta="right")
            for label in sorted(fit_results.get("gim", {})):
                yield dmc.TabsTab(f"Group {label}", value=f"gim-{label}", ta="right")

            yield dmc.TabsTab("ICU patients", value="icu", disabled=True, ta="right")
            for label in sorted(fit_results.get("icu", {})):
                yield dmc.TabsTab(
                    f"Group {label}", value=f"icu-{label}", color="orange", ta="right"
                )

        for label in sorted(fit_results.get("gim", {})):
            with dmc.TabsPanel(None, value=f"gim-{label}", mx="md"):
                with dmc.Stack(gap="md"):
                    result = fit_results["gim"][label]
                    yield dmc.Title(f"GIM-only Patients: LOS Grouping Label {label}", order=4)
                    yield dmc.Text(
                        [
                            dmc.Text("Number of patients: ", span=True, fw=700),
                            f"{result['n_patients']} ({result['probability'] * 100:.2f}%)",
                        ]
                    )
                    if len(result.get("label_groups", [])) == 1:
                        # Single associated grouping
                        key, val = [*result["label_groups"].items()][0]
                        yield dmc.Text(
                            [
                                dmc.Text("Associated grouping: ", span=True, fw=700),
                                f"{key} (n={val['count']}, {val['p_total']:.2%} of total)",
                            ]
                        )
                    else:
                        # Multiple associated groupings
                        yield dmc.Text("Associated grouping(s):", fw=700)
                        with dmc.List(None):
                            for key, val in result["label_groups"].items():
                                yield dmc.ListItem(
                                    f"{key} (n={val['count']}, {val['p_total']:.2%} of "
                                    f"total, {val['p_in_label']:.2%} within group)"
                                )
                    yield dmc.Text(
                        [
                            dmc.Text("Fitted LOS Distribution: ", span=True, fw=700),
                            f"{result['los_fit']['distribution']}"
                            f"""({
                                ", ".join(
                                    map(lambda f: str(round(f, 4)), result["los_fit"]["parameters"])
                                )
                            })""",
                        ]
                    )
                    yield dmc.Image(
                        src=f"data:image/png;base64,{result['los_fit']['pp_plot_b64']}",
                        alt="Probability Plot",
                        maw=800,
                    )

        for label in sorted(fit_results.get("icu", {})):
            with dmc.TabsPanel(None, value=f"icu-{label}", mx="md"):
                with dmc.Stack(gap="md"):
                    result = fit_results["icu"][label]
                    yield dmc.Title(f"ICU Patients: LOS Grouping Label {label}", order=4)
                    yield dmc.Text(
                        [
                            dmc.Text("Number of patients: ", span=True, fw=700),
                            f"{result['n_patients']} ({result['probability'] * 100:.2f}%)",
                        ]
                    )
                    if len(result.get("label_groups", [])) == 1:
                        # Single associated grouping
                        key = result["label_groups"][0]
                        yield dmc.Text(
                            [
                                dmc.Text("Associated grouping: ", span=True, fw=700),
                                f"{key} (n={val['count']}, {val['p_total']:.2%} of total)"
                            ]
                        )
                    else:
                        # Multiple associated groupings
                        yield dmc.Text("Associated grouping(s):", fw=700)
                        with dmc.List(None):
                            for key, val in result["label_groups"].items():
                                yield dmc.ListItem(
                                    f"{key} (n={val['count']}, {val['p_total']:.2%} of total, "
                                    f"{val['p_in_label']:.2%} within group)"
                                )
                    yield dmc.Text(
                        [
                            dmc.Text("With pre-ICU stay: ", span=True, fw=700),
                            f"{result['prob_pre_icu'] * 100:.2f}%",
                        ]
                    )
                    yield dmc.Text(
                        [
                            dmc.Text("With post-ICU stay: ", span=True, fw=700),
                            f"{result['prob_post_icu'] * 100:.2f}%",
                        ]
                    )
                    yield dmc.Text(
                        [
                            dmc.Text("Fitted LOS Distribution (pre-ICU stay): ", span=True, fw=700),
                            f"{result['pre_icu_los_fit']['distribution']}"
                            f"""({
                                ", ".join(
                                    map(
                                        lambda f: str(round(f, 4)),
                                        result["pre_icu_los_fit"]["parameters"],
                                    )
                                )
                            })""",
                        ]
                    )
                    yield dmc.Image(
                        src=f"data:image/png;base64,{result['pre_icu_los_fit']['pp_plot_b64']}",
                        alt="Probability Plot",
                        maw=800,
                    )
                    yield dmc.Text(
                        [
                            dmc.Text(
                                "Fitted LOS Distribution (main ICU stay): ", span=True, fw=700
                            ),
                            f"{result['icu_los_fit']['distribution']}"
                            f"""({
                                ", ".join(
                                    map(
                                        lambda f: str(round(f, 4)),
                                        result["icu_los_fit"]["parameters"],
                                    )
                                )
                            })""",
                        ]
                    )
                    yield dmc.Image(
                        src=f"data:image/png;base64,{result['icu_los_fit']['pp_plot_b64']}",
                        alt="Probability Plot",
                        maw=800,
                    )
                    yield dmc.Text(
                        [
                            dmc.Text(
                                "Fitted LOS Distribution (post-ICU stay): ", span=True, fw=700
                            ),
                            f"{result['post_icu_los_fit']['distribution']}"
                            f"""({
                                ", ".join(
                                    map(
                                        lambda f: str(round(f, 4)),
                                        result["post_icu_los_fit"]["parameters"],
                                    )
                                )
                            })""",
                        ]
                    )
                    yield dmc.Image(
                        src=f"data:image/png;base64,{result['post_icu_los_fit']['pp_plot_b64']}",
                        alt="Probability Plot",
                        maw=800,
                    )

    return [ret]


# endregion
