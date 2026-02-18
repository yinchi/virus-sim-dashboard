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

    TEXTINPUT_AGE_BREAKPOINTS: str = "step3-textinput-age-breakpoints-gim"
    BTN_AGE_GROUPS_UPDATE: str = "step3-btn-age-groups-gim-update"
    TEXT_AGE_GROUPS: str = "step3-text-age-groups-gim"
    STORE_AGE_GROUPS: str = "step3-store-age-groups-gim"

    DATEPICKER_START: str = "step3-datepicker-start"
    DATEPICKER_END: str = "step3-datepicker-end"
    TABLE_PATIENT_COUNTS_GIM: str = "step3-table-patient-counts-gim"
    TABLE_PATIENT_COUNTS_ICU: str = "step3-table-patient-counts-icu"
    TABLE_LOS_GROUPS_GIM: str = "step3-table-los-groups-gim"
    TABLE_LOS_GROUPS_ICU: str = "step3-table-los-groups-icu"

    BTN_FIT_LOS_DISTS: str = "step3-btn-fit-los-dists"
    STORE_FIT_LOS_DISTS_RESULTS: str = "step3-store-fit-los-dists-results"
    STACK_LOS_FIT_RESULTS: str = "step3-stack-los-fit-results"


step3_ids = StepThreeIDs()


class StepFourIDs(NamedTuple):
    """Component IDs used in Step 4."""

    BTN_PREV: str = "step4-btn-prev"
    BTN_NEXT: str = "step4-btn-next"

    TABS_CONFIG_OPTION: str = "step4-tabs-config-option"
    BTN_DOWNLOAD_EXAMPLE_SCENARIO: str = "step4-button-download-example-scenario"
    DOWNLOAD_EXAMPLE_SCENARIO: str = "step4-download-example-scenario"
    UPLOAD_SCENARIO_FILE: str = "step4-upload-scenario-file"
    BTN_UPLOAD_SCENARIO: str = "step4-btn-upload-scenario"
    STORE_UPLOADED_SCENARIO: str = "step4-store-uploaded-scenario"

    STACK_SCENARIO_OPT_UPLOAD: str = "step4-stack-scenario-opt-upload"
    STACK_SCENARIO_OPT_MANUAL: str = "step4-stack-scenario-opt-manual"

    NUMINPUT_JITTER: str = "step4-numinput-jitter"

    GRAPH_DAILIES_OPT_UPLOAD: dict[str, str] = {  # need dict for wildcard callbacks
        "type": "graph",
        "id": "step4-dailies-opt-upload",
    }
    GRAPH_HOURLIES_OPT_UPLOAD: dict[str, str] = {  # need dict for wildcard callbacks
        "type": "graph",
        "id": "step4-hourlies-opt-upload",
    }
    GRAPH_DAILIES_OPT_MANUAL: dict[str, str] = {  # need dict for wildcard callbacks
        "type": "graph",
        "id": "step4-dailies-opt-manual",
    }
    GRAPH_HOURLIES_OPT_MANUAL: dict[str, str] = {  # need dict for wildcard callbacks
        "type": "graph",
        "id": "step4-hourlies-opt-manual",
    }


step4_ids = StepFourIDs()


class StepFiveIDs(NamedTuple):
    """Component IDs used in Step 5."""

    BTN_PREV: str = "step5-btn-prev"

    DOWNLOAD_CONFIG: str = "step5-download-config"
    BTN_DOWNLOAD_CONFIG: str = "step5-download-config"

    BTN_SIMULATE: str = "step5-btn-simulate"
    GROUP_SIM_PROGRESS: str = "step5-sim-progress"
    TEXT_SIM_PROGRESS: str = "step5-sim-progress-text"
    PROGRESS_SIM_PROGRESS: str = "step5-sim-progress-bar"
    STACK_SIMULATION_RESULTS: str = "step5-simulation-results"


step5_ids = StepFiveIDs()
