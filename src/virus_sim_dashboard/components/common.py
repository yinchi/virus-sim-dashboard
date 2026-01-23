"""Common definitions for the Stepper component and subcomponents."""

from typing import NamedTuple


class MainIDs(NamedTuple):
    """Component IDs used across all Stepper steps."""

    MAIN_STORE: str = "main-store"
    """ID for the main Store component holding application state."""

    STEPPER: str = "main-stepper"
    """ID for the main Stepper component."""


main_ids = MainIDs()
