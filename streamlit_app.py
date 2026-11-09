"""Entrypoint for the diabetic 30-day readmission risk dashboard."""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="InsightD | Readmission assessment",
    page_icon=":material/medical_services:",
    layout="wide",
)

st.session_state.setdefault("current_prediction", None)

project_root = Path(__file__).resolve().parent
st.logo(
    str(project_root / "assets" / "insightd_logo.svg"),
    icon_image=str(project_root / "assets" / "insightd_icon.svg"),
    size="large",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/predictor.py",
            title="Risk assessment",
            icon=":material/health_and_safety:",
            help="Review historical cohort patterns, then enter a discharge encounter profile for a research-model estimate.",
        ),
        st.Page(
            "app_pages/guide.py",
            title="User guide",
            icon=":material/menu_book:",
            help="Learn when and how to use the research dashboard, interpret results, and understand its limitations.",
        ),
    ],
    position="sidebar",
)

page.run()
