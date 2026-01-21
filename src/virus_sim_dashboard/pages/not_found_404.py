"""Custom 404 Not Found page for the Dash application."""

from collections.abc import Generator

import dash
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.config import config

dash.register_page(__name__)


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Main content area layout."""
    with dmc.AppShellMain(
        None, w=config.app_width, px=0, m="md", pt=config.header_height, pb=config.footer_height
    ) as ret:
        with dmc.Stack(gap="md", m=0, p=0):
            yield dmc.Title("404 — Not Found", order=2)
            yield dmc.Text("The page you are looking for does not exist.")
    return ret
