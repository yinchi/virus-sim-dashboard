"""Step 2: Scenario Configuration."""

from collections.abc import Generator
from typing import NamedTuple

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from plotly import graph_objects as go

from virus_sim_dashboard.components.common import main_ids
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT


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

    TEXTINPUT_AGE_BREAKPOINTS_GIM: str = "step2-textinput-age-breakpoints-gim"
    BTN_AGE_GROUPS_GIM_UPDATE: str = "step2-btn-age-groups-gim-update"
    TEXT_AGE_GROUPS_GIM: str = "step2-text-age-groups-gim"
    TEXTINPUT_AGE_BREAKPOINTS_ICU: str = "step2-textinput-age-breakpoints-icu"
    BTN_AGE_GROUPS_ICU_UPDATE: str = "step2-btn-age-groups-icu-update"
    TEXT_AGE_GROUPS_ICU: str = "step2-text-age-groups-icu"

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
                yield dmc.Text("", id=step2_ids.TEXT_AGE_GROUPS_GIM, span=True)

        yield dmc.Title("ICU patients", order=4)
        with dmc.Group(align="flex-start"):
            yield dmc.TextInput(id=step2_ids.TEXTINPUT_AGE_BREAKPOINTS_ICU, w=300, value="")
            yield dmc.Button("Update", id=step2_ids.BTN_AGE_GROUPS_ICU_UPDATE)
            with dmc.Group():
                yield dmc.Text("Current groups: ", fw=700, span=True)
                yield dmc.Text("", id=step2_ids.TEXT_AGE_GROUPS_ICU, span=True)
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
            "groups in the Step 3 for length-of-stay analysis.",
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
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step2_on_next(
    _: int,
    main_store_data: dict,
) -> tuple[int, dict]:
    """Handle 'Next' button click to advance the stepper."""
    # TODO: Add necessary inputs to callback
    # TODO: Validate inputs and update main_store_data as needed
    return 2, main_store_data  # Advance to step 3 (index 2)


# endregion
