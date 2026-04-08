"""App configuration settings."""

from dataclasses import dataclass
from typing import Literal

from reliability.Distributions import (
    Gamma_Distribution,
    Lognormal_Distribution,
    Weibull_Distribution,
)
from reliability.Fitters import Fit_Gamma_3P, Fit_Lognormal_3P, Fit_Weibull_3P


@dataclass
class Config:
    """Configuration for the Dash app."""

    app_width: int = 1234  # in pixels, (min width required to fit the header content)
    header_height: int = 90  # in pixels
    footer_height: int = 50  # in pixels


# Allowed distribution types for length of stay modeling (can be extended as needed)

LOS_DISTRIBUTION_TYPE = Literal["Lognormal_3P", "Gamma_3P", "Weibull_3P"]
LOS_DISTRIBUTION_CLASSES: dict[LOS_DISTRIBUTION_TYPE, type] = {
    "Lognormal_3P": Lognormal_Distribution,
    "Gamma_3P": Gamma_Distribution,
    "Weibull_3P": Weibull_Distribution,
}
LOS_FITTERS: dict[LOS_DISTRIBUTION_TYPE, type] = {
    "Lognormal_3P": Fit_Lognormal_3P,
    "Gamma_3P": Fit_Gamma_3P,
    "Weibull_3P": Fit_Weibull_3P,
}

# Type to use for LoS fitting and sampling (can be changed to other supported types as needed)
LOS_DIST: LOS_DISTRIBUTION_TYPE = "Weibull_3P"


config = Config()
