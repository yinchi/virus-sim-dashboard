"""Home page for the virus simulation dashboard, containing the main Stepper component."""

from collections.abc import Generator

import dash
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.components import stepper as main_stepper
from virus_sim_dashboard.config import config

dash.register_page(__name__, path="/")


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Main content area layout."""
    with dmc.AppShellMain(
        None, w=config.app_width, px=0, m="md", pt=config.header_height, pb=config.footer_height
    ) as ret:
        yield main_stepper.layout()
    return ret
