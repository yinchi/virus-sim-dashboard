"""Step 4: Scenario Definition."""

from base64 import b64decode, b64encode
from collections.abc import Generator
from io import BytesIO

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from plotly import express as px

from virus_sim_dashboard.components.common import main_ids, step4_ids
from virus_sim_dashboard.util import jsonify


# region layout
@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 4."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Title("Step 4: Scenario Definition", order=2, ta="center")
        with dmc.Tabs(
            None,
            id=step4_ids.TABS_CONFIG_OPTION,
            variant="pills",
            color="orange",
            value="upload",
        ):
            with dmc.TabsList(None, grow=True, bd="1px solid var(--mantine-color-default-border)"):
                yield dmc.TabsTab(
                    "Option 1: Upload scenario config", value="upload", fz="lg", fw=900
                )
                yield dmc.TabsTab(
                    "Option 2: Scenario fitter tool (coming soon)", value="manual", fz="lg", fw=900
                )
            yield upload_config_tab()
            with dmc.TabsPanel(None, value="manual", py="md"):
                yield dmc.Text("Not yet implemented, coming soon!", c="red")
        yield jitter_controls()
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step4_ids.BTN_PREV)
            yield dmc.Button("Next", id=step4_ids.BTN_NEXT, disabled=True)  # TODO: Step 5
    return ret


@composition
def upload_config_tab() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the 'Upload scenario config' tab in Step 4."""
    with dmc.TabsPanel(None, value="upload", py="md") as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Text(
                "Upload an Excel file with tables containing the number of daily "
                "arrivals and time-of-day distribution.  See the example file for "
                "details on the required format."
            )
            with dmc.Group(None, gap="md"):
                yield dmc.Button(
                    "Download example config",
                    id=step4_ids.BTN_DOWNLOAD_EXAMPLE_SCENARIO,
                    color="green",
                )
                yield dcc.Download(id=step4_ids.DOWNLOAD_EXAMPLE_SCENARIO)
                with dcc.Upload(None, id=step4_ids.UPLOAD_SCENARIO_FILE):
                    yield dmc.Button("Upload config", id=step4_ids.BTN_UPLOAD_SCENARIO)
            with dmc.Stack(None, id=step4_ids.STACK_SCENARIO_OPT_UPLOAD, gap="md", m=0, p=0):
                yield dmc.Title("Scenario parameters", order=3)
                yield dmc.Title("Daily arrivals", order=4)
                yield dmc.Text("Daily arrivals graph goes here.")
                yield dmc.Title("Time-of-day distribution", order=4)
                yield dmc.Text("Time-of-day distribution graph goes here.")
        yield dcc.Store(id=step4_ids.STORE_UPLOADED_SCENARIO)
    return ret


@composition
def jitter_controls() -> Generator[DashComponent, None, DashComponent]:
    """Controls for adding jitter to the daily arrivals."""
    with dmc.Stack(None, gap="md") as ret:
        yield dmc.Title("Jitter", order=3)
        yield dmc.Text(
            "Add some randomness to the daily arrivals to "
            "simulate real-world variability.  For example, if the number of "
            "arrivals on a given day is 10 and the jitter is set to 20%, the actual "
            "number of arrivals for that day could be anywhere between 8 and 12. "
            "Note rounding is applied to ensure the number of arrivals is always a "
            "whole number."
        )
        with dmc.Group(None, gap="md"):
            yield dmc.NumberInput(
                id=step4_ids.NUMINPUT_JITTER,
                value=0,
                label="Jitter (0-100%)",
                min=0,
                max=100,
                step=1,
                # Reject invalid inputs, works since no invalid value is a prefix of a valid value
                clampBehavior="strict",
                allowDecimal=False,
                allowNegative=False,
                allowLeadingZeros=False,
                suffix="%",
            )
    return ret


# endregion


# region callbacks
@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Input(step4_ids.BTN_PREV, "n_clicks"),
    prevent_initial_call=True,
)
def step4_on_prev(
    _: int,
) -> int:
    """Handle 'Previous' button click to go back a step."""
    return 2  # Go back to step 3 (index 2)


@callback(
    Output(step4_ids.DOWNLOAD_EXAMPLE_SCENARIO, "data"),
    Input(step4_ids.BTN_DOWNLOAD_EXAMPLE_SCENARIO, "n_clicks"),
    prevent_initial_call=True,
)
def download_example_scenario(_: int) -> dict:
    """Callback to download example scenario config."""
    return dcc.send_file(
        "assets/scenario.xlsx",
        filename="example_scenario.xlsx",
        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@callback(
    Output(step4_ids.STORE_UPLOADED_SCENARIO, "data"),
    Input(step4_ids.UPLOAD_SCENARIO_FILE, "contents"),
    State(step4_ids.UPLOAD_SCENARIO_FILE, "filename"),
    prevent_initial_call=True,
)
def handle_uploaded_scenario(contents: str | None, filename: str | None) -> dict | None:
    """Callback to handle uploaded scenario config file.

    Extract Excel tables and encode them as base64 strings to store in dcc.Store for
    later processing.
    """
    # TODO: Add component to layout to show upload errors and add it to callback outputs
    if contents is None or filename is None:
        raise dash.exceptions.PreventUpdate
    if not filename.lower().endswith(".xlsx"):
        raise ValueError("Invalid file type. Please upload an .xlsx file.")

    # read file contents and store in an BytesIO object for later processing
    content_string = contents.split(",", maxsplit=1)[1]
    decoded = b64decode(content_string)
    file_data = BytesIO(decoded)

    # Read pandas DataFrames from the Excel file and convert to feather format in-memory
    dailies_bytes = BytesIO()
    hourlies_bytes = BytesIO()
    pd.read_excel(file_data, sheet_name="Daily Arrivals").to_feather(dailies_bytes)
    pd.read_excel(file_data, sheet_name="Hourly Distribution").to_feather(hourlies_bytes)

    # Encode the feather byte buffers as base64 and return a dict
    return {
        "dailies": b64encode(dailies_bytes.getvalue()).decode(),
        "hourlies": b64encode(hourlies_bytes.getvalue()).decode(),
    }


@callback(
    Output(step4_ids.STACK_SCENARIO_OPT_UPLOAD, "children"),
    Input(step4_ids.STORE_UPLOADED_SCENARIO, "data"),
)
@composition
def update_scenario_opt_upload_stack(
    uploaded_data: dict | None,
) -> Generator[DashComponent, None, DashComponent]:
    """Update the scenario options stack to show graphs based on the uploaded scenario data."""
    if uploaded_data is None:
        with dmc.Stack(None, gap="md", m=0, p=0) as ret:
            yield dmc.Title("Scenario parameters", order=3)
            yield dmc.Text("No scenario uploaded yet.", c="red")
        return ret

    # Decode the base64-encoded feather data and read into pandas DataFrames
    dailies_bytes = BytesIO(b64decode(uploaded_data["dailies"]))
    dailies = pd.read_feather(dailies_bytes)
    dailies.date = pd.to_datetime(dailies.date)  # ensure date column is datetime
    dailies = dailies.set_index("date")

    hourlies_bytes = BytesIO(b64decode(uploaded_data["hourlies"]))
    hourlies = pd.read_feather(hourlies_bytes).set_index("hour")

    with dmc.Stack(None, gap="md", m=0, p=0) as ret:
        yield dmc.Title("Scenario parameters", order=3)
        yield dcc.Graph(
            id=step4_ids.GRAPH_DAILIES_OPT_UPLOAD,
            figure=px.line(
                dailies,
                x=dailies.index,
                y="count",
                title="Daily arrivals",
                labels={"date": "Date", "count": "Number of arrivals"},
            ),
        )
        yield dcc.Graph(
            id=step4_ids.GRAPH_HOURLIES_OPT_UPLOAD,
            figure=px.line(
                hourlies,
                x=hourlies.index,
                y="probability",
                title="Time-of-day distribution",
                labels={"hour": "Hour of day", "probability": "Probability"},
            ),
        )
    return ret


@callback(
    Output(step4_ids.BTN_NEXT, "disabled"),
    Input(step4_ids.STORE_UPLOADED_SCENARIO, "data"),
    Input(step4_ids.TABS_CONFIG_OPTION, "value"),
)
def update_next_button_disabled(uploaded_data: dict | None, tab_value: str | None) -> bool:
    """Enable the 'Next' button based on component inputs."""
    if tab_value == "upload":
        return uploaded_data is None
    # tab_value == "manual"
    return True  # Manual option not implemented yet


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Output(main_ids.MAIN_STORE, "data", allow_duplicate=True),
    Input(step4_ids.BTN_NEXT, "n_clicks"),
    State(step4_ids.TABS_CONFIG_OPTION, "value"),
    State(step4_ids.STORE_UPLOADED_SCENARIO, "data"),
    State(step4_ids.NUMINPUT_JITTER, "value"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step4_on_next(
    _: int,
    tab_value: str | None,
    uploaded_data: dict | None,
    jitter_value: int,
    main_store_data: dict,
) -> tuple[int, dict]:
    """Handle 'Next' button click to go to the next step and store scenario data."""
    # Handle data update based on which scenario config option is selected.

    # Upload option: extract scenario data from the uploaded file
    if tab_value == "upload":
        # Ensure we have a valid uploaded scenario before proceeding
        if uploaded_data is None:
            raise dash.exceptions.PreventUpdate
        step4_dict = {
            "dailies": uploaded_data["dailies"],
            "hourlies": uploaded_data["hourlies"],
            "jitter": jitter_value / 100,
        }

    # Manual option: compute scenario based on user inputs (start/end, peak value, etc.)
    # TODO: Use a Store to save the manual config; use a separate callback for populating the Store.
    # The callback should clear the Store if the config is invalid, which will in turn disable the
    # 'Next' button until the config is valid again (when the "Manual" Tab is selected).
    else:
        raise dash.exceptions.PreventUpdate

    # Only update the main store if the new scenario data is different from what's already stored
    if jsonify(main_store_data.get("step4", {})) != jsonify(step4_dict):
        main_store_data["step4"] = step4_dict
        # Invalidate and discard any data beyond step 4
        main_store_data = {
            k: v for k, v in main_store_data.items() if k in ["step1", "step2", "step3", "step4"]
        }

    return 4, main_store_data  # Go to step 5 (index 4)


# endregion
