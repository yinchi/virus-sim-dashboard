"""About page for the virus simulation dashboard."""

from collections.abc import Generator

import dash
import dash_mantine_components as dmc
from dash import dcc
from dash.development.base_component import Component as DashComponent
from dash_compose import composition

from virus_sim_dashboard.config import config

dash.register_page(__name__)

ABOUT_MD = f"""\
## About

This dashboard was created by Yin-Chi Chan at the Institute for Manufacturing, University of
Cambridge for Cambridge University Hospitals (CUH) NHS Foundation Trust.

**Links:**
- [Tutorial (PDF)](/assets/tutorial.pdf)
- [GitHub repository](http://github.com/yinchi/virus-sim-dashboard)

**Related publication:**

- Y.-C. Chan, A. Mukherjee, N. Moretti, M. Nakaoka, et al.,
  “**A process simulation model for a histopathology laboratory**,”
  in: *2024 Winter Simulation Conference (WSC)*, Orlando: IEEE, 2024,
  pp. 846–857. ([PDF](https://yinchi.github.io/papers/ChanSimulation2023.pdf))

![CUH Logo]({dash.get_asset_url("logos.png")})
"""


@composition
def layout() -> Generator[DashComponent, None, DashComponent]:
    """Main content area layout."""
    with dmc.AppShellMain(
        None, w=config.app_width, px=0, m="md", pt=config.header_height, pb=config.footer_height
    ) as ret:
        with dmc.TypographyStylesProvider(None):
            yield dcc.Markdown(ABOUT_MD)
    return ret
