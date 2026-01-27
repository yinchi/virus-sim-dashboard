"""Common definitions for the Stepper component and subcomponents."""

from typing import NamedTuple


class MainIDs(NamedTuple):
    """Component IDs used across all Stepper steps."""

    MAIN_STORE: str = "main-store"
    """ID for the main Store component holding application state."""

    STEPPER: str = "main-stepper"
    """ID for the main Stepper component."""


main_ids = MainIDs()


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


step2_ids = StepTwoIDs()


class StepThreeIDs(NamedTuple):
    """Component IDs used in Step 3."""

    BTN_PREV: str = "step3-btn-prev"
    BTN_NEXT: str = "step3-btn-next"

    MARKDOWN_CONFIG_SUMMARY: str = "step3-markdown-config-summary"

    GRAPH_DAILY_ARRIVALS: dict[str, str] = {  # need dict for wildcard callbacks
        "type": "graph",
        "id": "step3-daily-arrivals",
    }

    TEXTINPUT_AGE_BREAKPOINTS_GIM: str = "step3-textinput-age-breakpoints-gim"
    BTN_AGE_GROUPS_GIM_UPDATE: str = "step3-btn-age-groups-gim-update"
    TEXT_AGE_GROUPS_GIM: str = "step3-text-age-groups-gim"
    STORE_AGE_GROUPS_GIM: str = "step3-store-age-groups-gim"

    TEXTINPUT_AGE_BREAKPOINTS_ICU: str = "step3-textinput-age-breakpoints-icu"
    BTN_AGE_GROUPS_ICU_UPDATE: str = "step3-btn-age-groups-icu-update"
    TEXT_AGE_GROUPS_ICU: str = "step3-text-age-groups-icu"
    STORE_AGE_GROUPS_ICU: str = "step3-store-age-groups-icu"

    DATEPICKER_START: str = "step3-datepicker-start"
    DATEPICKER_END: str = "step3-datepicker-end"
    TABLE_PATIENT_COUNTS_GIM: str = "step3-table-patient-counts-gim"
    TABLE_PATIENT_COUNTS_ICU: str = "step3-table-patient-counts-icu"
    TABLE_LOS_GROUPS_GIM: str = "step3-table-los-groups-gim"
    TABLE_LOS_GROUPS_ICU: str = "step3-table-los-groups-icu"

    BTN_FIT_LOS_DISTS: str = "step3-btn-fit-los-dists"
    STORE_FIT_LOS_DISTS_RESULTS: str = "step3-store-fit-los-dists-results"


step3_ids = StepThreeIDs()
