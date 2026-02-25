"""The application shell for the Dash app."""

from collections.abc import Generator
from datetime import date

import dash
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition
from dash_iconify import DashIconify

from virus_sim_dashboard.components import theme_toggle
from virus_sim_dashboard.config import config
from virus_sim_dashboard.util import COPY, NBSP, NDASH


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Dash app layout (root component)."""
    # TODO: Replace placeholder with actual dashboard components
    with dmc.MantineProvider() as ret:
        with dmc.AppShell(
            None,
            header={"height": config.header_height},
            footer={"height": config.footer_height},
            miw=config.app_width,
            p=0,
            m=0,
        ):
            yield my_app_header()
            yield dash.page_container
            yield my_app_footer()
    return ret


@composition
def my_app_header() -> Generator[DashComponent, None, DashComponent]:
    """Header component for the Dash app."""
    with dmc.AppShellHeader(None, miw=config.app_width, h=config.header_height, p=0, m=0) as ret:
        with dmc.Group(
            justify="space-between",
            align="center",
            flex=1,
            miw=config.app_width,
            h="100%",
            px="md",
        ):
            yield header_left()
            yield header_right()
    return ret


@composition
def header_left() -> Generator[DashComponent, None, DashComponent]:
    """Left side of the header."""
    with dmc.Group(p=0, m=0, gap=0, align="center") as ret:
        yield dmc.Image(
            src=dash.get_asset_url("logo-cuh.png"),
            h=f"{config.header_height}px",
            w="auto",
            fit="scale-down",
        )
        yield dmc.Title("CUH respiratory viruses modelling web app", order=1)
    return ret


@composition
def header_right() -> Generator[DashComponent, None, DashComponent]:
    """Right side of the header."""
    with dmc.Group(p=0, m=0, gap="xl", align="center") as ret:
        yield dmc.Anchor(
            "Documentation ↗️",
            c="inherit",
            href="https://github.com/yinchi/virus-sim-dashboard/tree/main/docs",
            target="_blank",
        )
        yield dmc.Anchor(
            "About ↗️",
            c="inherit",
            href="/about",
            target="_blank",
        )
        yield theme_toggle.layout()
    return ret


@composition
def my_app_footer() -> Generator[DashComponent, None, DashComponent]:
    """Footer component for the Dash app."""
    with dmc.AppShellFooter(None, miw=config.app_width, h=config.footer_height, p=0, m=0) as ret:
        with dmc.Group(
            justify="space-between", h="100%", px="sm", pt="5", pb="5"
        ):
            with dmc.Text():
                yield dmc.Text(copyright_str())
            with dmc.Anchor(href="http://github.com/yinchi/virus-sim-dashboard", target="_blank"):
                yield DashIconify(icon="ion:logo-github", height=16)
                yield dmc.Text(f"{NBSP}Github", span=True)
    return ret


def copyright_str():
    """Returns the copyright string."""
    year = date.today().year
    return (
        f"{COPY} 2025{f'{NDASH}{year}' if year > 2025 else ''} "
        "Yin-Chi Chan, Institute for Manufacturing, "
        "University of Cambridge"
    )
