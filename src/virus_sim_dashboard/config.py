"""App configuration settings."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the Dash app."""

    app_width: int = 1234  # in pixels, (min width required to fit the header content)
    header_height: int = 90  # in pixels
    footer_height: int = 40  # in pixels


config = Config()
