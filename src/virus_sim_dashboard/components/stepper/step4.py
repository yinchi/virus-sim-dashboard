"""Step 4: Scenario Definition."""

from base64 import b64decode, b64encode
from collections.abc import Callable, Generator, Sequence
from io import BytesIO

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from plotly import express as px
from scipy.optimize import curve_fit
from scipy.stats import beta, rv_continuous

from virus_sim_dashboard.components.common import main_ids, step4_ids
from virus_sim_dashboard.components.stepper.step2 import process_stay_data
from virus_sim_dashboard.util import DEFAULT_FIGURE_LAYOUT, jsonify

DAY = pd.Timedelta(days=1)


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
                yield dmc.TabsTab("Option 2: Scenario fitter tool", value="manual", fz="lg", fw=900)
            yield upload_config_tab()
            yield manual_config_tab()
        yield jitter_controls()
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Previous", id=step4_ids.BTN_PREV)
            yield dmc.Button("Next", id=step4_ids.BTN_NEXT, disabled=True)
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
                    id=step4_ids.OPT_UPLOAD_BTN_DOWNLOAD_EXAMPLE,
                    color="green",
                )
                yield dcc.Download(id=step4_ids.OPT_UPLOAD_DOWNLOAD_EXAMPLE)
                with dcc.Upload(None, id=step4_ids.OPT_UPLOAD_UPLOAD):
                    yield dmc.Button("Upload config", id=step4_ids.OPT_UPLOAD_BTN_UPLOAD)
            with dmc.Stack(None, id=step4_ids.OPT_UPLOAD_STACK_SCENARIO, gap="md", m=0, p=0):
                yield dmc.Text("Placeholder")
        yield dcc.Store(id=step4_ids.OPT_UPLOAD_STORE_SCENARIO, data=None)
    return ret


@composition
def manual_config_tab() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the 'Scenario fitter tool' tab in Step 4."""
    with dmc.TabsPanel(None, value="manual", py="md") as ret:
        with dmc.Stack(None, gap="md", m=0, p=0):
            yield dmc.Title("Curve fitting parameters", order=3)
            with dmc.Group(None, gap="md"):
                yield dmc.DatePickerInput(
                    id=step4_ids.OPT_MANUAL_DPICKER_FIT_START,
                    value="01 Aug 2024",
                    valueFormat="DD MMM YYYY",
                    label="Start date for arrivals",
                )
                yield dmc.DatePickerInput(
                    id=step4_ids.OPT_MANUAL_DPICKER_FIT_END,
                    value="31 May 2025",
                    valueFormat="DD MMM YYYY",
                    label="End date for arrivals",
                )
            with dmc.Group(None, gap="md"):
                yield dmc.Checkbox(
                    id=step4_ids.OPT_MANUAL_CHECKBOX_ZERO_START,
                    label="Force start date to have zero arrivals",
                    size="md",
                )
                yield dmc.Checkbox(
                    id=step4_ids.OPT_MANUAL_CHECKBOX_ZERO_END,
                    label="Force end date to have zero arrivals",
                    size="md",
                )
            with dmc.Group(None, gap="md"):
                yield dmc.Button("Fit to data", id=step4_ids.OPT_MANUAL_BTN_FIT_CURVE)
            with dmc.Stack(
                None,
                id=step4_ids.OPT_MANUAL_STACK_FIT_RESULTS,
                gap="md",
                m=0,
                p=0,
                style={"display": "none"},
            ):
                with dmc.TypographyStylesProvider():
                    yield dcc.Markdown(
                        children="""\
#### Fitted curve parameters

- **Peak date**: 01 Jan 2025 (42.0 days from start date)
- **Peak value**: 30.0 arrivals/day
- **Start value**: 0.0 arrivals/day
- **End value**: 0.0 arrivals/day"""
                    )
                yield dmc.Button(
                    "Apply fitted parameters to scenario", id=step4_ids.OPT_MANUAL_BTN_APPLY_FIT
                )
            yield dmc.Title("Scenario parameters", order=3)
            with dmc.Group(None, gap="md"):
                yield dmc.DatePickerInput(
                    id=step4_ids.OPT_MANUAL_DPICKER_START,
                    value="01 Aug 2025",
                    valueFormat="DD MMM YYYY",
                    label="Start date for arrivals",
                )
                yield dmc.DatePickerInput(
                    id=step4_ids.OPT_MANUAL_DPICKER_END,
                    value="31 May 2026",
                    valueFormat="DD MMM YYYY",
                    label="End date for arrivals",
                )
                yield dmc.DatePickerInput(
                    id=step4_ids.OPT_MANUAL_DPICKER_PEAK,
                    value="01 Jan 2026",
                    valueFormat="DD MMM YYYY",
                    label="Peak date for arrivals",
                )
            with dmc.Group(None, gap="md"):
                yield dmc.NumberInput(
                    id=step4_ids.OPT_MANUAL_NUMINPUT_PEAK,
                    value=30,
                    label="Peak value (arrivals/day)",
                    min=0,
                    allowNegative=False,
                    allowLeadingZeros=False,
                )
                yield dmc.NumberInput(
                    id=step4_ids.OPT_MANUAL_NUMINPUT_START,
                    value=0,
                    label="Start value (arrivals/day)",
                    min=0,
                    allowNegative=False,
                    allowLeadingZeros=False,
                )
                yield dmc.NumberInput(
                    id=step4_ids.OPT_MANUAL_NUMINPUT_END,
                    value=0,
                    label="End value (arrivals/day)",
                    min=0,
                    allowNegative=False,
                    allowLeadingZeros=False,
                )
                yield dmc.NumberInput(
                    id=step4_ids.OPT_MANUAL_NUMINPUT_CONCENTRATION,
                    value=90,
                    label="Concentration parameter (higher = more peaked)",
                    min=0.1,
                    allowNegative=False,
                    allowLeadingZeros=False,
                )
            # Main plot for the fitted and scenario daily arrivals curve
            yield dcc.Graph(
                id=step4_ids.OPT_MANUAL_GRAPH_DAILIES,
                figure=go.Figure(layout=DEFAULT_FIGURE_LAYOUT),
            )
            # Time-of-day distribution plot
            with dmc.Stack(None, gap="md", m=0, p=0):
                yield dmc.Title("Time-of-day distribution", order=4)
                yield dmc.Text(
                    'Select "Fit to data" above to fit a time-of-day distribution to the '
                    "provided data.",
                    id=step4_ids.OPT_MANUAL_TEXT_NO_HOURLIES,
                    c="red",
                )
                yield dcc.Graph(
                    id=step4_ids.OPT_MANUAL_GRAPH_HOURLIES,
                    figure=go.Figure(layout=DEFAULT_FIGURE_LAYOUT),
                    style={"display": "none"},
                )
            yield dcc.Store(id=step4_ids.OPT_MANUAL_STORE_FIT_RESULTS, data=None)
        yield dcc.Store(id=step4_ids.OPT_MANUAL_STORE_DAILIES, data=None)
        yield dcc.Store(id=step4_ids.OPT_MANUAL_STORE_HOURLIES, data=None)
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
            "whole number.",
            size="sm",
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
    Output(step4_ids.OPT_UPLOAD_DOWNLOAD_EXAMPLE, "data"),
    Input(step4_ids.OPT_UPLOAD_BTN_DOWNLOAD_EXAMPLE, "n_clicks"),
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
    Output(step4_ids.OPT_UPLOAD_STORE_SCENARIO, "data"),
    Input(step4_ids.OPT_UPLOAD_UPLOAD, "contents"),
    State(step4_ids.OPT_UPLOAD_UPLOAD, "filename"),
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
    Output(step4_ids.OPT_UPLOAD_STACK_SCENARIO, "children"),
    Input(step4_ids.OPT_UPLOAD_STORE_SCENARIO, "data"),
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
            id=step4_ids.OPT_UPLOAD_GRAPH_DAILIES,
            figure=px.line(
                dailies,
                x=dailies.index,
                y="count",
                title="Daily arrivals",
                labels={"date": "Date", "count": "Number of arrivals"},
            ),
        )
        yield dcc.Graph(
            id=step4_ids.OPT_UPLOAD_GRAPH_HOURLIES,
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
    Input(step4_ids.TABS_CONFIG_OPTION, "value"),
    Input(step4_ids.OPT_UPLOAD_STORE_SCENARIO, "data"),
    Input(step4_ids.OPT_MANUAL_STORE_DAILIES, "data"),
    Input(step4_ids.OPT_MANUAL_STORE_HOURLIES, "data"),
)
def update_next_button_disabled(
    tab_value: str | None,
    uploaded_data: dict | None,
    manual_dailies: str | None,
    manual_hourlies: str | None,
) -> bool:
    """Enable the 'Next' button based on component inputs."""
    if tab_value == "upload":
        return uploaded_data is None
    return manual_dailies is None or manual_hourlies is None


@callback(
    Output(step4_ids.OPT_MANUAL_STACK_FIT_RESULTS, "style"),
    Output(step4_ids.OPT_MANUAL_STACK_FIT_RESULTS, "children"),
    Output(step4_ids.OPT_MANUAL_STORE_FIT_RESULTS, "data"),
    Output(step4_ids.OPT_MANUAL_DPICKER_FIT_START, "error"),
    Output(step4_ids.OPT_MANUAL_DPICKER_FIT_END, "error"),
    Input(step4_ids.OPT_MANUAL_BTN_FIT_CURVE, "n_clicks"),
    # Clear results if main store data changes in a way that would invalidate the fit
    Input(main_ids.MAIN_STORE, "data"),
    State(step4_ids.OPT_MANUAL_DPICKER_FIT_START, "value"),
    State(step4_ids.OPT_MANUAL_DPICKER_FIT_END, "value"),
    State(step4_ids.OPT_MANUAL_CHECKBOX_ZERO_START, "checked"),
    State(step4_ids.OPT_MANUAL_CHECKBOX_ZERO_END, "checked"),
    prevent_initial_call=True,
)
def fit_curve_and_update_results(
    _: int,
    main_data: dict,
    fit_start: str | None,
    fit_end: str | None,
    zero_left: bool,
    zero_right: bool,
) -> tuple[dict, list[DashComponent], dict | None, str | None, str | None]:
    """Fit a curve to the provided data and update the results stack and store."""
    fit_start_date = pd.to_datetime(fit_start) if fit_start is not None else None
    fit_end_date = pd.to_datetime(fit_end) if fit_end is not None else None

    errors: list[str | None] = [None, None]  # [fit_start_error, fit_end_error]

    # Check date inputs
    if fit_start_date is None:
        errors[0] = "Date is required"
    if fit_end_date is None:
        errors[1] = "Date is required"
    if fit_start_date is not None and fit_end_date is not None and fit_start_date > fit_end_date:
        errors[0] = " "
        errors[1] = "End date must be after start date"
    if any(errors):
        return {"display": "none"}, [], None, errors[0], errors[1]

    # For type checker; above checks ensure these are not None
    assert fit_start_date is not None and fit_end_date is not None

    # Check for required data in main store
    disease_name = main_data.get("step1", {}).get("disease_name")
    start_opt_community = main_data.get("step2", {}).get("start_opt_community")
    start_opt_other = main_data.get("step2", {}).get("start_opt_other")
    if not all([disease_name, start_opt_community, start_opt_other]):
        return {"display": "none"}, [], None, None, None

    # Else, no-op if trigger was the data
    if dash.callback_context.triggered and dash.callback_context.triggered[0]["prop_id"].startswith(
        main_ids.MAIN_STORE
    ):
        raise dash.exceptions.PreventUpdate

    # Checkpoint: trigger was the button and we have valid inputs and required data

    # Process stay data and filter to the fit date range
    stay_data = process_stay_data(main_data, start_opt_community, start_opt_other)
    stay_data = stay_data.loc[
        (stay_data["Start"] >= fit_start_date) & (stay_data["Start"] <= fit_end_date)
    ]
    if stay_data.empty:
        return {"display": "none"}, [], None, None, "No data in selected date range"

    # Resample to daily counts
    dailies = stay_data.set_index("Start").resample("D").size()

    # Get hourly distribution from stay_data
    hourlies = stay_data["Start"].dt.hour.value_counts().sort_index()

    # Fit the curve to the daily data
    fit_results = fit_beta(dailies, zero_left=zero_left, zero_right=zero_right)

    print(f"Fitted curve parameters: {fit_results}")

    store_value = {
        "fit_results": fit_results,
        "hourlies": hourlies.tolist(),
    }

    # Calculate the peak date from the fitted curve's mode parameter
    peak_mode = fit_results["mode"]
    peak_n_days = peak_mode * ((fit_end_date - fit_start_date) / DAY)
    peak_date = fit_start_date + pd.Timedelta(days=peak_n_days)
    # Round to nearest day for display purposes
    peak_date_rounded = peak_date.round("D")

    ret_style = {"display": "block"}
    ret_children = [
        dmc.TypographyStylesProvider(
            dcc.Markdown(f"""\
#### Fitted curve parameters
- **Peak date**: {peak_date_rounded.date()} ({peak_n_days:.1f} days from start date)
- **Peak value**: {fit_results["y_max"]:.1f} arrivals/day
- **Start value**: {fit_results.get("y_left", 0.0):.1f} arrivals/day
- **End value**: {fit_results.get("y_right", 0.0):.1f} arrivals/day
- **Concentration**: {fit_results.get("concentration", 10.0):.1f}""")
        ),
        dmc.Button("Apply fitted parameters to scenario", id=step4_ids.OPT_MANUAL_BTN_APPLY_FIT),
        dmc.Text(
            "Clicking the button above will populate the scenario config with the "
            "fitted curve parameters, with dates shifted to align with the selected start date. "
            "These can then be further edited manually if desired.",
            c="dimmed",
            size="sm",
        ),
    ]
    return ret_style, ret_children, store_value, None, None


@callback(
    Output(step4_ids.OPT_MANUAL_TEXT_NO_HOURLIES, "style"),
    Output(step4_ids.OPT_MANUAL_GRAPH_HOURLIES, "style"),
    Output(step4_ids.OPT_MANUAL_GRAPH_HOURLIES, "figure", allow_duplicate=True),
    Output(step4_ids.OPT_MANUAL_STORE_HOURLIES, "data", allow_duplicate=True),
    Input(step4_ids.OPT_MANUAL_STORE_FIT_RESULTS, "data"),
    prevent_initial_call=True,
)
def update_hourly_distribution_plot(
    store_value: dict | None,
) -> tuple[dict, dict, Patch, str | None]:
    """Update the time-of-day distribution plot based on the fitted curve results."""
    if store_value is None or "hourlies" not in store_value:
        return {"display": "block"}, {"display": "none"}, Patch(), None

    hourlies = store_value["hourlies"]
    hourlies_df = pd.DataFrame({"hour": range(24), "count": hourlies})
    hourlies_df["probability"] = hourlies_df["count"] / hourlies_df["count"].sum()

    fig = Patch()
    fig["layout"]["title"]["text"] = "Time-of-day distribution of arrivals"
    fig["layout"]["xaxis"]["title"]["text"] = "Hour of day"
    fig["layout"]["yaxis"]["title"]["text"] = "Probability"
    fig["data"] = [
        go.Scatter(
            x=hourlies_df["hour"],
            y=hourlies_df["probability"],
            name="Probability",
        )
    ]
    fig["layout"]["yaxis"]["rangemode"] = "tozero"

    print(fig)

    ret_hourlies_df = hourlies_df[["hour", "probability"]]
    ret_hourlies_df["hour"] = ret_hourlies_df["hour"].astype(int)
    ret_hourlies_df["probability"] = ret_hourlies_df["probability"].astype(float)
    bytes_buffer = BytesIO()
    ret_hourlies_df.to_feather(bytes_buffer)
    ret_hourlies_b64 = b64encode(bytes_buffer.getvalue()).decode()

    return {"display": "none"}, {"display": "block"}, fig, ret_hourlies_b64


@callback(
    Output(step4_ids.OPT_MANUAL_GRAPH_DAILIES, "figure", allow_duplicate=True),
    Output(step4_ids.OPT_MANUAL_STORE_DAILIES, "data", allow_duplicate=True),
    Output(step4_ids.OPT_MANUAL_DPICKER_END, "error"),
    Output(step4_ids.OPT_MANUAL_DPICKER_PEAK, "error"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_PEAK, "error"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_START, "error"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_END, "error"),
    Input(step4_ids.TABS_CONFIG_OPTION, "value"),
    Input(step4_ids.OPT_MANUAL_STORE_FIT_RESULTS, "data"),
    Input(step4_ids.OPT_MANUAL_DPICKER_START, "value"),
    Input(step4_ids.OPT_MANUAL_DPICKER_END, "value"),
    Input(step4_ids.OPT_MANUAL_DPICKER_PEAK, "value"),
    Input(step4_ids.OPT_MANUAL_NUMINPUT_PEAK, "value"),
    Input(step4_ids.OPT_MANUAL_NUMINPUT_START, "value"),
    Input(step4_ids.OPT_MANUAL_NUMINPUT_END, "value"),
    Input(step4_ids.OPT_MANUAL_NUMINPUT_CONCENTRATION, "value"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def update_dailies_plot(
    tab_value: str | None,
    store_value: dict | None,
    scenario_start: str | None,
    scenario_end: str | None,
    scenario_peak: str | None,
    scenario_peak_value: float | None,
    scenario_start_value: float | None,
    scenario_end_value: float | None,
    scenario_concentration: float | None,
    main_store_data: dict,
) -> tuple[Patch | str | None, ...]:
    """Update the daily arrivals plot based on the fitted curve results and scenario config."""
    if tab_value != "manual":
        # Plot is not on visible tab, no need to update
        raise dash.exceptions.PreventUpdate

    fig = Patch()
    disease_name: str = main_store_data.get("step1", {}).get("disease_name", "Disease")
    fig["layout"]["title"]["text"] = f"Daily arrivals curve for {disease_name}"
    fig["layout"]["xaxis"]["title"]["text"] = "Date"
    fig["layout"]["yaxis"]["title"]["text"] = "Number of arrivals"
    fig["layout"]["yaxis"]["rangemode"] = "tozero"

    start_opt_community = main_store_data.get("step2", {}).get("start_opt_community")
    start_opt_other = main_store_data.get("step2", {}).get("start_opt_other")
    if not all([disease_name, scenario_start, scenario_end, scenario_peak]):
        # Not enough information to plot daily arrivals, early exit with empty figure
        fig["data"] = []
        return (fig, None, None, None, None, None, None)

    assert scenario_start is not None and scenario_end is not None and scenario_peak is not None

    # Process patient stay data and add curve to figure
    stay_data = (
        process_stay_data(main_store_data, start_opt_community, start_opt_other)
        .set_index("Start")
        .resample("D")
        .size()
    )
    fig["data"] = [
        go.Scatter(
            x=stay_data.index,
            y=stay_data.values,
            name="Daily arrivals",
        )
    ]

    # If we have fitted curve results, add the curve to the plot
    if store_value and "fit_results" in store_value:
        fit_results = store_value["fit_results"]
        fit_start_date = pd.to_datetime(fit_results["start_date"])
        fit_end_date = pd.to_datetime(fit_results["end_date"])
        scaled_xs = np.linspace(0, 1, (fit_end_date - fit_start_date).days + 1)
        fit_fn = my_beta_scaled(
            mode=fit_results["mode"],
            concentration=fit_results["concentration"],
            y_max=fit_results["y_max"],
            y_left=fit_results.get("y_left", 0),
            y_right=fit_results.get("y_right", 0),
        )
        fit_x = pd.date_range(start=fit_start_date, end=fit_end_date)
        fit_y = fit_fn(scaled_xs)

        fig["data"].append(
            go.Scatter(
                x=fit_x,
                y=fit_y,
                name="Fitted curve",
            )
        )

    # Validate scenario config dates and values before plotting the scenario curve
    end_date_err_idx = 0
    peak_date_err_idx = 1
    peak_value_err_idx = 2
    start_value_err_idx = 3
    end_value_err_idx = 4
    errors: list[str | None] = [None] * 5

    scenario_start_date = pd.to_datetime(scenario_start)
    scenario_end_date = pd.to_datetime(scenario_end)
    scenario_peak_date = pd.to_datetime(scenario_peak)
    if scenario_start_date >= scenario_end_date:
        errors[end_date_err_idx] = "End date must be after start date"
    if not (scenario_start_date <= scenario_peak_date <= scenario_end_date):
        errors[peak_date_err_idx] = "Peak date must be between start and end date"

    if scenario_peak_value is None or scenario_peak_value < 0:
        errors[peak_value_err_idx] = "Peak value must be non-negative"
    if scenario_start_value is None or scenario_start_value < 0:
        errors[start_value_err_idx] = "Start value must be non-negative"
    if (
        scenario_start_value is not None
        and scenario_peak_value is not None
        and scenario_start_value >= scenario_peak_value
    ):
        errors[start_value_err_idx] = "Start value must be less than peak value"
    if scenario_end_value is None or scenario_end_value < 0:
        errors[end_value_err_idx] = "End value must be non-negative"
    if (
        scenario_end_value is not None
        and scenario_peak_value is not None
        and scenario_end_value >= scenario_peak_value
    ):
        errors[end_value_err_idx] = "End value must be less than peak value"

    # Default scenario data to None
    ret_dailies = None

    # If we have valid scenario config inputs, add the scenario curve to the plot
    if not any(errors):
        # Type checker assertions; above checks ensure these are not None
        assert scenario_concentration is not None
        assert scenario_peak_value is not None
        assert scenario_start_value is not None
        assert scenario_end_value is not None

        # Map the scenario date range to [0, 1] for the x values of the curve function
        # n_days = end - start + 1
        n_days = (scenario_end_date - scenario_start_date).days + 1
        xs = np.linspace(0, 1, n_days)
        mode = (scenario_peak_date - scenario_start_date).days / n_days
        mode = np.clip(mode, 0.001, 0.999)

        scenario_fn = my_beta_scaled(
            mode=mode,
            concentration=scenario_concentration,
            y_max=scenario_peak_value,
            y_left=scenario_start_value,
            y_right=scenario_end_value,
        )
        scenario_y = scenario_fn(xs)
        scenario_x = [
            scenario_start_date + pd.Timedelta(days=int(x))
            for x in np.linspace(0, n_days - 1, n_days)
        ]
        fig["data"].append(
            go.Scatter(
                x=scenario_x,
                y=scenario_y,
                name="Scenario curve",
            )
        )

        # Prepare scenario data to store in dcc.Store for use in later steps
        scenario_df = pd.DataFrame({"date": scenario_x, "count": scenario_y})
        bytes_buffer = BytesIO()
        scenario_df.to_feather(bytes_buffer)
        ret_dailies = b64encode(bytes_buffer.getvalue()).decode()

    return (fig, ret_dailies, *errors)


@callback(
    Output(step4_ids.OPT_MANUAL_DPICKER_END, "value"),
    Output(step4_ids.OPT_MANUAL_DPICKER_PEAK, "value"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_PEAK, "value"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_START, "value"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_END, "value"),
    Output(step4_ids.OPT_MANUAL_NUMINPUT_CONCENTRATION, "value"),
    Input(step4_ids.OPT_MANUAL_BTN_APPLY_FIT, "n_clicks"),
    State(step4_ids.OPT_MANUAL_STORE_FIT_RESULTS, "data"),
    State(step4_ids.OPT_MANUAL_DPICKER_START, "value"),
    prevent_initial_call=True,
)
def apply_fit_to_scenario_config(
    _: int,
    store_value: dict | None,
    scenario_start: str | None,
) -> tuple[str, str, float, float, float, float]:
    """Apply the fitted curve parameters to the scenario config inputs."""
    fit_results = store_value.get("fit_results", None) if store_value else None

    if fit_results is None or scenario_start is None:
        raise dash.exceptions.PreventUpdate

    # Calculate the end date and peak date based on the fitted curve's mode and the selected
    # start date
    scenario_start_date = pd.to_datetime(scenario_start)
    fit_start_date = pd.to_datetime(fit_results["start_date"])
    fit_end_date = pd.to_datetime(fit_results["end_date"])

    peak_mode = fit_results["mode"]
    peak_n_days = peak_mode * ((fit_end_date - fit_start_date) / DAY)
    # Apply rounding
    peak_n_days = round(peak_n_days, 1)
    scenario_peak_date = scenario_start_date + pd.Timedelta(days=peak_n_days)
    # No rounding for end date since inputs do not have time-of-day component
    scenario_end_date = scenario_start_date + (fit_end_date - fit_start_date)

    return (
        scenario_end_date.strftime("%Y-%m-%d"),
        scenario_peak_date.strftime("%Y-%m-%d"),
        round(fit_results["y_max"], 1),
        round(fit_results.get("y_left", 0), 1),
        round(fit_results.get("y_right", 0), 1),
        round(fit_results.get("concentration", 10.0), 1),
    )


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Output(main_ids.MAIN_STORE, "data", allow_duplicate=True),
    Input(step4_ids.BTN_NEXT, "n_clicks"),
    State(step4_ids.TABS_CONFIG_OPTION, "value"),
    State(step4_ids.OPT_UPLOAD_STORE_SCENARIO, "data"),
    State(step4_ids.OPT_MANUAL_STORE_DAILIES, "data"),
    State(step4_ids.OPT_MANUAL_STORE_HOURLIES, "data"),
    State(step4_ids.NUMINPUT_JITTER, "value"),
    State(main_ids.MAIN_STORE, "data"),
    prevent_initial_call=True,
)
def step4_on_next(
    _: int,
    tab_value: str | None,
    uploaded_data: dict | None,
    manual_dailies: str | None,
    manual_hourlies: str | None,
    jitter_value: int,
    main_store_data: dict,
) -> tuple[int, dict]:
    """Handle 'Next' button click to go to the next step and store scenario data.

    Returns the next step index and the updated main store data.  Step 4 data contains the keys
    - `dailies`: base64-encoded feather data for the daily arrivals curve
    - `hourlies`: base64-encoded feather data for the time-of-day distribution
    - `jitter`: jitter value as a float between 0 and 1

    regardless of whether the scenario was uploaded or manually configured.
    """
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
    else:
        if manual_dailies is None or manual_hourlies is None:
            raise dash.exceptions.PreventUpdate

        step4_dict = {
            "dailies": manual_dailies,
            "hourlies": manual_hourlies,
            "jitter": jitter_value / 100,
        }

    # Only update the main store if the new scenario data is different from what's already stored
    if jsonify(main_store_data.get("step4", {})) != jsonify(step4_dict):
        main_store_data["step4"] = step4_dict
        # Invalidate and discard any data beyond step 4
        main_store_data = {
            k: v for k, v in main_store_data.items() if k in ["step1", "step2", "step3", "step4"]
        }

    return 4, main_store_data  # Go to step 5 (index 4)


# endregion


# region helper functions
def my_beta(mode: float, concentration: float) -> rv_continuous:
    """Generate a beta distribution with a specified mode and concentration.

    For a concave beta distribution, i.e. α > 1 and β > 1, the mode and concentration are given by:
    mode = (α - 1) / (α + β - 2)
    c = α + β - 2

    Solving for α and β gives:
    α = mode * c + 1
    β = (1 - mode) * c + 1

    See: https://en.wikipedia.org/wiki/Beta_distribution#Mode_and_concentration
    """
    if not (0 < mode < 1):
        raise ValueError("Mode must be between 0 and 1")
    if concentration <= 0:
        raise ValueError("Concentration must be greater than 0")

    alpha = mode * concentration + 1
    beta_param = (1 - mode) * concentration + 1
    return beta(alpha, beta_param)


def my_beta_scaled(
    mode: float,
    concentration: float,
    y_max: float = 1.0,
    y_left: float = 0.0,
    y_right: float = 0.0,
) -> Callable[[float | Sequence[float] | np.ndarray], float | np.ndarray]:
    """Generate a scaled beta distribution.

    Args:
        mode: The mode of the beta distribution (between 0 and 1).
        concentration: The concentration parameter of the beta distribution (greater than 0).
        y_left: The minimum value of the scaled PDF.
        y_right: The maximum value of the scaled PDF.
        y_max: The maximum value of the scaled PDF (used for scaling).

    Returns:
        A function that takes a value x (between 0 and 1) and returns the scaled PDF value.
    """
    unscaled_dist = my_beta(mode, concentration)

    # Calculate the maximum value of the unscaled PDF
    unscaled_max = unscaled_dist.pdf(mode)

    # Calculate scaling factors for the left and right sides of the distribution
    scale_factor_left = (y_max - y_left) / unscaled_max
    scale_factor_right = (y_max - y_right) / unscaled_max

    # Shift and scale the PDF to fit within the specified y_left and y_right bounds
    def scaled_pdf_scalar(x: float) -> float:
        if x < mode:
            return y_left + scale_factor_left * unscaled_dist.pdf(x)
        return y_right + scale_factor_right * unscaled_dist.pdf(x)

    # Vectorize the scaled PDF function to handle array inputs
    def scaled_pdf_vectorized(x: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.array([scaled_pdf_scalar(xi) for xi in x])

    def scaled_pdf(x: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        if isinstance(x, (float, int)):
            return scaled_pdf_scalar(x)
        return scaled_pdf_vectorized(x)

    return scaled_pdf


def fit_beta(dailies: pd.Series, zero_left: bool = True, zero_right: bool = True):
    """Fit a beta distribution to the given daily data.

    Args:
        dailies: A pandas Series with daily counts, indexed by date.
        zero_left: If True, the left y-intercept of the fitted curve is fixed at 0.
        zero_right: If True, the right y-intercept of the fitted curve is fixed at 0.

    Returns:
        The parameters of the fitted beta distribution (compatible with `my_beta_scaled`).
    """
    # Scale the dates to the interval [0,1]
    df = pd.DataFrame(
        {
            "ScaledDate": np.linspace(0, 1, len(dailies)),
            "Admissions": dailies.to_numpy(),
        },
        index=dailies.index,
    )

    # For the function names, "z" stands for zero and "f" stands for free (non-zero)
    def fit_zz(mode, conc, y_max):
        return my_beta_scaled(mode, conc, y_max=y_max, y_left=0.0, y_right=0.0)

    def fit_fz(mode, conc, y_max, left):
        return my_beta_scaled(mode, conc, y_max=y_max, y_left=left, y_right=0.0)

    def fit_zf(mode, conc, y_max, right):
        return my_beta_scaled(mode, conc, y_max=y_max, y_left=0.0, y_right=right)

    def fit_ff(mode, conc, y_max, left, right):
        return my_beta_scaled(mode, conc, y_max=y_max, y_left=left, y_right=right)

    # Initial estimate for mode: scaled date corresponding to the maximum daily count
    initial_mode = df.loc[df["Admissions"].idxmax(), "ScaledDate"]
    # Initial estimate for concentration
    initial_concentration = 10.0
    # Initial estimate for y_max: maximum daily count
    initial_y_max = df["Admissions"].max()
    # Initial estimates for y_left and y_right: first and last daily counts
    initial_y_left = df["Admissions"].iloc[0] if not zero_left else 0.0
    initial_y_right = df["Admissions"].iloc[-1] if not zero_right else 0.0

    initials = (
        (initial_mode, initial_concentration, initial_y_max)
        if zero_left and zero_right
        else (initial_mode, initial_concentration, initial_y_max, initial_y_left)
        if not zero_left and zero_right
        else (initial_mode, initial_concentration, initial_y_max, initial_y_right)
        if zero_left and not zero_right
        else (initial_mode, initial_concentration, initial_y_max, initial_y_left, initial_y_right)
    )

    lower_bounds = (
        (0.0, 0.0, 0.0)
        if zero_left and zero_right
        else (0.0, 0.0, 0.0, 0.0)
        if not zero_left and zero_right
        else (0.0, 0.0, 0.0, 0.0)
        if zero_left and not zero_right
        else (0.0, 0.0, 0.0, 0.0, 0.0)
    )

    upper_bounds = (
        (1.0, np.inf, np.inf)
        if zero_left and zero_right
        else (1.0, np.inf, np.inf, df["Admissions"].max())
        if not zero_left and zero_right
        else (1.0, np.inf, np.inf, df["Admissions"].max())
        if zero_left and not zero_right
        else (1.0, np.inf, np.inf, df["Admissions"].max(), df["Admissions"].max())
    )

    fit_func = (
        fit_zz
        if zero_left and zero_right
        else fit_fz
        if not zero_left and zero_right
        else fit_zf
        if zero_left and not zero_right
        else fit_ff
    )

    params, _ = curve_fit(
        lambda x, *params: fit_func(*params)(x),
        df["ScaledDate"],
        df["Admissions"],
        p0=initials,
        bounds=(lower_bounds, upper_bounds),
    )

    return {
        "start_date": df.index[0],
        "end_date": df.index[-1],
        "mode": params[0],
        "concentration": params[1],
        "y_max": params[2],
        "y_left": params[3] if not zero_left else 0.0,
        "y_right": params[4] if not zero_right else 0.0,
    }


# endregion
