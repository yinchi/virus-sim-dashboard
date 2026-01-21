"""Main module for the CUH Respiratory Virus Simulation Dashboard Dash application."""

from dash import Dash

from virus_sim_dashboard.components import app_shell

app = Dash(
    assets_folder="../../assets",
    suppress_callback_exceptions=True,
    use_pages=True,
)


app.layout = app_shell.layout()
