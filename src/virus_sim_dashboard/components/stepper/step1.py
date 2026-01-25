"""Step 1: File upload."""

import io
from base64 import b64decode
from collections.abc import Generator
from typing import NamedTuple

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components.common import main_ids
from virus_sim_dashboard.util import find_upwards, jsonify

DISEASES = ["COVID-19", "Influenza", "RSV", "Other"]

FILE_INPUT_INSTRUCTIONS = [
    """\
Upload patient stay data in CSV or Excel format.  The example file shows the expected \
columns and data formats; extra columns will be ignored.  Column names must match the expected \
names (as shown in the example file) exactly.  For Excel input, the data must start in cell A1 of \
the first sheet.""",
    """\
⚠️ Note that for timestamp columns, we will try to infer the format automatically; however, to \
avoid ambiguity, it is recommended to use the format YYYY-MM-DD HH:MM:SS (ISO 8601).""",
]


def format_validation_errors(filename: str, errors: list[str]) -> str:
    """Format validation errors as a markdown string for display."""
    if not errors:
        return ""
    return f"""\
**{len(errors)} validation errors for {filename}:**

{"\n".join(["- " + err for err in errors])}
"""


# Component IDs
class StepOneIDs(NamedTuple):
    """Component IDs used in Step 1."""

    SELECT_DISEASE_NAME: str = "step1-disease-select"
    TEXTINPUT_CUSTOM_DISEASE: str = "step1-disease-custom"
    BUTTON_DOWNLOAD_STAYS_EXAMPLE: str = "step1-stays-example-btn"
    DOWNLOAD_STAYS_EXAMPLE: str = "step1-stays-example"
    UPLOAD_STAYS_FILE: str = "step1-stays-upload"
    TEXT_UPLOAD_STATUS: str = "step1-stays-upload-status"
    STYLES_PROVIDER: str = "step1-stays-upload-details-wrapper"
    TEXT_UPLOAD_DETAILS: str = "step1-stays-upload-details"
    STORE_PARSED_DATA: str = "step1-parsed-data-store"
    BTN_NEXT: str = "step1-btn-next"


step1_ids = StepOneIDs()


# region Layout
@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Contents of the dmc.StepperStep for Step 1."""
    with dmc.Stack(None, gap="xl", m=0, p=0) as ret:
        with dmc.Stack(None, id="step1-stack", gap="md", m=0, p=0):
            yield dcc.Store(id=step1_ids.STORE_PARSED_DATA, data=None)
            yield dmc.Title("Step 1: File upload", order=2, ta="center")
            yield dmc.Title("Name of respiratory illness", order=3)
            with dmc.Group(None, align="flex-start", gap="md"):
                yield dmc.Select(
                    id=step1_ids.SELECT_DISEASE_NAME,
                    data=DISEASES,
                    description='Select from dropdown or select "Other" to enter a custom name.',
                    value=DISEASES[0],
                    allowDeselect=False,
                    w=300,
                )
                yield dmc.TextInput(
                    id=step1_ids.TEXTINPUT_CUSTOM_DISEASE,
                    mt=33.8,
                    placeholder="Enter custom disease name",
                    w=300,
                )
            yield dmc.Title("Patient stay data", order=3)
            for par in FILE_INPUT_INSTRUCTIONS:
                yield dmc.Text(par, size="sm", c="dimmed")
            with dmc.Group(None, align="flex-start", gap="md"):
                yield dmc.Button(
                    "Download example .xlsx file",
                    id=step1_ids.BUTTON_DOWNLOAD_STAYS_EXAMPLE,
                    bg="green",
                )
                yield dcc.Download(id=step1_ids.DOWNLOAD_STAYS_EXAMPLE)
                with dcc.Upload(None, id=step1_ids.UPLOAD_STAYS_FILE):
                    yield dmc.Button("Upload patient stay data")
                yield dmc.Text(
                    "No file uploaded", id=step1_ids.TEXT_UPLOAD_STATUS, c="dimmed", mt="0.5rem"
                )
            with dmc.TypographyStylesProvider(
                None, c="red", id=step1_ids.STYLES_PROVIDER, display="none"
            ):
                initial_errors: list[str] = []
                markdown_str = format_validation_errors("", initial_errors)
                yield dcc.Markdown(markdown_str, id=step1_ids.TEXT_UPLOAD_DETAILS)
        with dmc.Group(None, gap="md"):
            yield dmc.Button("Next", id=step1_ids.BTN_NEXT, disabled=True)

    return ret


# endregion


# region Callbacks
@callback(
    Output(step1_ids.TEXTINPUT_CUSTOM_DISEASE, "styles"),
    Input(step1_ids.SELECT_DISEASE_NAME, "value"),
)
def toggle_custom_disease_input(selected_disease: str) -> dict:
    """Show/hide the custom disease name input based on selection.

    Show the input only when "Other" is selected.  Note that the value of the input is not
    cleared when hidden.
    """
    # Based on https://www.dash-mantine-components.com/components/textinput#styles-api
    # we want to target the `wrapper` element to hide/show the input
    if selected_disease == "Other":
        return {}  # Default styles for everything
    return {"wrapper": {"display": "none"}}  # Hide the input


@callback(
    Output(step1_ids.STYLES_PROVIDER, "display"),
    Input(step1_ids.TEXT_UPLOAD_DETAILS, "children"),
)
def toggle_upload_details_display(details_children: str) -> str:
    """Show/hide the upload status details based on whether there are any details to show.

    Args:
        details_children: Content of the Markdown component showing upload status details.
    """
    if details_children.strip():
        return "block"
    return "none"


@callback(
    Output(step1_ids.DOWNLOAD_STAYS_EXAMPLE, "data"),
    Input(step1_ids.BUTTON_DOWNLOAD_STAYS_EXAMPLE, "n_clicks"),
    prevent_initial_call=True,
)
def download_stays_example(_: int) -> dict:
    """Provide example patient stay data file for download."""
    # Use the preprocessed flu data as the example file
    assets_dir = find_upwards("assets", __file__)
    if not assets_dir:
        raise FileNotFoundError("Could not find 'assets' directory for example file.")
    asset_path = assets_dir / "flu_preprocessed.xlsx"
    return dcc.send_file(
        asset_path,
        filename=asset_path.name,
        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@callback(
    Output(step1_ids.STORE_PARSED_DATA, "data"),
    Output(step1_ids.TEXT_UPLOAD_STATUS, "children"),
    Output(step1_ids.TEXT_UPLOAD_DETAILS, "children"),
    Input(step1_ids.UPLOAD_STAYS_FILE, "contents"),
    State(step1_ids.UPLOAD_STAYS_FILE, "filename"),
    prevent_initial_call=True,
)
def handle_stays_file_upload(
    contents: str | None,
    filename: str | None,
) -> tuple[list | None, str, str]:
    """Handle patient stay data file upload.

    If the file is valid, parse and store the data; otherwise, return validation errors.

    Args:
        contents: Base64-encoded contents of the uploaded file.
        filename: Name of the uploaded file.

    Returns:
        A tuple containing
        1. The parsed patient stay data as a list-of-dicts, or None if validation failed.
        2. A status message for display.
        3. A detailed markdown string with validation errors, if any.
    """
    # Check if a file was uploaded
    if contents is None or filename is None:
        data = None
        status_msg = "No file uploaded"
        details_msg = ""
        return data, status_msg, details_msg
    status_msg = f"Uploaded file: {filename}"

    # Check file type and parse accordingly
    if filename.endswith(".xlsx"):
        # Extract the base64-encoded content, assume data is in the first sheet and starts at A1
        try:
            _, b64_content = contents.split(",", 1)
            decoded_content = b64decode(b64_content)
            df = pd.read_excel(io.BytesIO(decoded_content))
            data, errors = process_patient_stay_data(df)
        except Exception as e:
            data, errors = None, [f"Error parsing Excel file: {str(e)}"]
    elif filename.endswith(".csv"):
        # Extract the base64-encoded content
        try:
            _, b64_content = contents.split(",", 1)
            decoded_content = b64decode(b64_content)
            df = pd.read_csv(io.StringIO(decoded_content.decode()))
            data, errors = process_patient_stay_data(df)
        except Exception as e:
            data, errors = None, [f"Error parsing CSV file: {str(e)}"]
    else:
        data, errors = None, ["Unsupported file format. Please upload a .xlsx or .csv file."]
    details_msg = format_validation_errors(filename, errors)
    return data, status_msg, details_msg


@callback(
    Output(step1_ids.BTN_NEXT, "disabled"),
    Input(step1_ids.STORE_PARSED_DATA, "data"),
)
def toggle_next_button(parsed_data: list | None) -> bool:
    """Enable/disable the 'Next' button based on whether valid data has been uploaded.

    Args:
        parsed_data: The parsed patient stay data from the upload.  Is reset to None if the
            uploaded data is invalid (`handle_stays_file_upload` callback).
    """
    return parsed_data is None


@callback(
    Output(main_ids.STEPPER, "active", allow_duplicate=True),
    Output(main_ids.MAIN_STORE, "data", allow_duplicate=True),
    Input(step1_ids.BTN_NEXT, "n_clicks"),
    State(step1_ids.SELECT_DISEASE_NAME, "value"),
    State(step1_ids.TEXTINPUT_CUSTOM_DISEASE, "value"),
    State(main_ids.MAIN_STORE, "data"),
    State(step1_ids.STORE_PARSED_DATA, "data"),
    prevent_initial_call=True,
)
def step1_on_next(
    _: int,
    disease_name: str,
    custom_disease_name: str,
    main_store_data: dict,
    parsed_stays_data: dict | None,
) -> tuple[int, dict]:
    """Handle 'Next' button click to advance the stepper."""
    disease_name_final = custom_disease_name.strip() if disease_name == "Other" else disease_name

    if parsed_stays_data is None:
        # No valid data uploaded (should not happen due to UI constraints)
        raise dash.exceptions.PreventUpdate

    old_step1_data = main_store_data.get("step1", {})
    new_step1_data = {"disease_name": disease_name_final, "patient_stays": parsed_stays_data}

    # If data has changed, update current step data and discard downstream step data
    if jsonify(old_step1_data) != jsonify(new_step1_data):
        main_store_data = {"step1": new_step1_data}

    return 1, main_store_data  # Advance to Step 2 (index 1)


# endregion


# region Helper functions
STAYS_COLS_NUMERIC = ["Age"]
STAYS_COLS_DATETIME = [
    "Admission",
    "Discharge",
    "ICUAdmission",
    "ICUDischarge",
    "Readmission",
    "ReadmissionDischarge",
    "FirstPosCollected",
]
STAYS_COL_STRING = ["Acquisition", "Summary"]


def process_patient_stay_data(df: pd.DataFrame) -> tuple[list[dict] | None, list[str]]:
    """Validate and preprocess the uploaded patient stay data.

    Args:
        df: DataFrame containing the patient stay data.

    Returns:
        A tuple containing
        1. A list-of-dicts representation of the validated and preprocessed data, or None if
          validation failed.
        2. A list of validation error messages (empty if no errors).
    """
    errors = []

    # Check for column presence and cast to the expected types, collecting errors as we go
    for col in STAYS_COLS_NUMERIC:
        if col not in df.columns:
            errors.append(f"Missing required numeric column: {col}")
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except ValueError:
                errors.append(f"Column '{col}' must be numeric.")
    for col in STAYS_COLS_DATETIME:
        if col not in df.columns:
            errors.append(f"Missing required datetime column: {col}")
        else:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
            except ValueError:
                errors.append(f"Column '{col}' must be datetime.")
    for col in STAYS_COL_STRING:
        if col not in df.columns:
            errors.append(f"Missing required string column: {col}")
        else:
            try:
                df[col] = df[col].astype(str)
            except ValueError:
                errors.append(f"Column '{col}' must be string.")

    if errors:
        return None, errors

    # No errors, proceed to pre-process data as needed

    # Keep only the expected columns
    df = df[STAYS_COLS_NUMERIC + STAYS_COLS_DATETIME + STAYS_COL_STRING]

    df = df.loc[
        # Keep only admitted patients
        (df.Summary.str.lower() != "not admitted")
        # Ensure mandatory datetime fields are present
        & df.Admission.notna()
        & df.Discharge.notna()
        & df.FirstPosCollected.notna()
        # Ensure ICU and/or readmission stay is complete if present
        & (df.ICUAdmission.isna() | df.ICUDischarge.notna())
        & (df.Readmission.isna() | df.ReadmissionDischarge.notna())
    ]

    if df.empty:
        errors.append("No valid patient stay records found after preprocessing.")
    if (df.Admission > df.Discharge).any():
        errors.append("Some records have Admission date after Discharge date.")
    if (
        df.ICUAdmission.notna() & df.ICUDischarge.notna() & (df.ICUAdmission > df.ICUDischarge)
    ).any():
        errors.append("Some records have ICU Admission date after ICU Discharge date.")
    if (
        df.Readmission.notna()
        & df.ReadmissionDischarge.notna()
        & (df.Readmission > df.ReadmissionDischarge)
    ).any():
        errors.append("Some records have Readmission date after Readmission Discharge date.")

    if errors:
        return None, errors

    # All validations passed, return the processed data with no errors
    return df.to_dict(orient="records"), []


# endregion
