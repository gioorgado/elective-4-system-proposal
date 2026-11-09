"""Reusable descriptive cohort summaries and visualizations."""

import pandas as pd
import plotly.express as px
import streamlit as st

from model_service import (
    ADMISSION_SOURCE_LABELS,
    ADMISSION_TYPE_LABELS,
    AGE_CATEGORIES,
    DIAGNOSIS_GROUP_LABELS,
    DISCHARGE_DESTINATION_LABELS,
)

COHORT_GROUPS = {
    "Age group": ("age", None),
    "Primary diagnosis group": ("diag_1_category", DIAGNOSIS_GROUP_LABELS),
    "Admission category": ("admission_type_id", ADMISSION_TYPE_LABELS),
    "Discharge destination": (
        "discharge_disposition_id",
        DISCHARGE_DESTINATION_LABELS,
    ),
    "Source of admission": ("admission_source_id", ADMISSION_SOURCE_LABELS),
    "Length of stay": ("time_in_hospital", None),
    "Prior-year inpatient admissions": ("number_inpatient", None),
    "Prior-year emergency visits": ("number_emergency", None),
    "Total prior-year visits": ("total_prior_visits", None),
    "Insulin treatment": (
        "insulin",
        {
            "No": "Not prescribed",
            "Steady": "Prescribed; no dose change",
            "Up": "Dose increased",
            "Down": "Dose decreased",
            "Yes": "Prescribed; change not specified",
        },
    ),
}

COHORT_GROUP_BINS = {
    "Length of stay": (
        [-float("inf"), 2, 5, float("inf")],
        ["1-2 days", "3-5 days", "6+ days"],
    ),
    "Prior-year inpatient admissions": (
        [-float("inf"), 0, 1, float("inf")],
        ["0", "1", "2+"],
    ),
    "Prior-year emergency visits": (
        [-float("inf"), 0, 1, float("inf")],
        ["0", "1", "2+"],
    ),
    "Total prior-year visits": (
        [-float("inf"), 0, 2, 5, float("inf")],
        ["0", "1-2", "3-5", "6+"],
    ),
}

VISUALIZATION_OPTIONS = [
    "Readmission rate",
    "Encounter volume",
    "Outcome counts",
    "Outcome proportions",
    "Encounter mix (donut)",
    "Readmission rate trend (line)",
]
TREND_GROUPS = {"Age group", *COHORT_GROUP_BINS}
DEFAULT_VISUALIZATIONS = ["Readmission rate", "Encounter volume"]
NO_SECOND_VISUALIZATION = "None"
FIRST_VISUALIZATION_KEY = "cohort_visualization_1"
SECOND_VISUALIZATION_KEY = "cohort_visualization_2"


def available_visualizations(group_name: str) -> list[str]:
    """Keep the line chart available only when categories have a natural order."""
    return [
        option
        for option in VISUALIZATION_OPTIONS
        if option != "Readmission rate trend (line)" or group_name in TREND_GROUPS
    ]


def _update_visualizations_for_group() -> None:
    """Keep both chart selectors valid when the cohort group changes."""
    group_name = st.session_state["cohort_group"]
    allowed_visualizations = available_visualizations(group_name)
    first = st.session_state.get(FIRST_VISUALIZATION_KEY)
    if first not in allowed_visualizations:
        first = DEFAULT_VISUALIZATIONS[0]
        st.session_state[FIRST_VISUALIZATION_KEY] = first

    second = st.session_state.get(SECOND_VISUALIZATION_KEY)
    if second not in allowed_visualizations or second == first:
        second = next(
            (
                visualization
                for visualization in DEFAULT_VISUALIZATIONS
                if visualization != first and visualization in allowed_visualizations
            ),
            NO_SECOND_VISUALIZATION,
        )
        st.session_state[SECOND_VISUALIZATION_KEY] = second


def _update_second_visualization() -> None:
    """Prevent duplicate charts when the first visualization changes."""
    first = st.session_state[FIRST_VISUALIZATION_KEY]
    second = st.session_state.get(SECOND_VISUALIZATION_KEY)
    if second == first:
        st.session_state[SECOND_VISUALIZATION_KEY] = NO_SECOND_VISUALIZATION


def selected_visualizations(first: str, second: str) -> list[str]:
    """Return one or two selected chart views without duplicates."""
    return [first] if second == NO_SECOND_VISUALIZATION else [first, second]


def _age_label(value: str) -> str:
    """Turn source age bands into clinician-readable ranges."""
    try:
        lower, upper = value.strip("[]()").split("-")
        return f"{lower}-{int(upper) - 1} years"
    except (ValueError, AttributeError):
        return value


def summarize_cohort(cohort: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Aggregate encounters and readmission outcomes for one cohort grouping."""
    if group_name not in COHORT_GROUPS:
        raise ValueError(f"Unknown cohort grouping: {group_name}")

    column, labels = COHORT_GROUPS[group_name]
    if group_name in COHORT_GROUP_BINS:
        bins, bin_labels = COHORT_GROUP_BINS[group_name]
        groups = pd.cut(
            pd.to_numeric(cohort[column], errors="coerce"),
            bins=bins,
            labels=bin_labels,
        ).astype("string")
    else:
        groups = cohort[column].astype("string").fillna("Missing")
    if labels is not None:
        groups = groups.map(labels).fillna(groups)
    elif column == "age":
        groups = groups.map(_age_label)
    groups = groups.fillna("Not recorded").replace("Missing", "Not recorded")

    grouped = cohort.assign(cohort_group=groups)
    summary = (
        grouped.groupby("cohort_group", observed=True)["readmitted_30d"]
        .agg(encounters="size", readmissions="sum", readmission_rate="mean")
        .reset_index()
        .rename(columns={"cohort_group": "group"})
    )

    if column == "age":
        age_order = [_age_label(value) for value in AGE_CATEGORIES]
        summary["group"] = pd.Categorical(
            summary["group"], categories=age_order, ordered=True
        )
        summary = summary.sort_values("group")
        summary["group"] = summary["group"].astype(str)
    elif group_name in COHORT_GROUP_BINS:
        summary["group"] = pd.Categorical(
            summary["group"],
            categories=COHORT_GROUP_BINS[group_name][1],
            ordered=True,
        )
        summary = summary.sort_values("group")
        summary["group"] = summary["group"].astype(str)
    else:
        summary = summary.sort_values("readmission_rate", ascending=False)

    summary["not_readmitted"] = summary["encounters"] - summary["readmissions"]
    return summary.reset_index(drop=True)


def _build_cohort_figure(
    summary: pd.DataFrame,
    group_name: str,
    visualization: str,
    total_encounters: int,
    readmissions: int,
) -> tuple[object, str]:
    """Build one cohort visualization and its accessible description."""
    category_order = summary["group"].tolist()
    if visualization == "Readmission rate trend (line)":
        figure = px.line(
            summary,
            x="group",
            y="readmission_rate",
            markers=True,
            hover_data={
                "encounters": ":,",
                "readmissions": ":,",
                "readmission_rate": ":.1%",
            },
            category_orders={"group": category_order},
            labels={
                "group": group_name,
                "readmission_rate": "Readmitted within 30 days",
            },
        )
        figure.update_yaxes(tickformat=".0%")
        alt = f"30-day readmission rate trend across {group_name.lower()} groups"
    elif visualization == "Encounter mix (donut)":
        outcome_summary = pd.DataFrame(
            {
                "outcome": [
                    "Readmitted within 30 days",
                    "No readmission within 30 days",
                ],
                "encounters": [readmissions, total_encounters - readmissions],
            }
        )
        figure = px.pie(
            outcome_summary,
            names="outcome",
            values="encounters",
            hole=0.58,
            hover_data={"encounters": ":,"},
            labels={"outcome": "Encounter outcome"},
        )
        figure.update_traces(textinfo="percent+label")
        alt = "Overall cohort split between 30-day readmissions and other encounters"
    elif visualization == "Readmission rate":
        figure = px.bar(
            summary,
            x="group",
            y="readmission_rate",
            hover_data={
                "encounters": ":,",
                "readmissions": ":,",
                "readmission_rate": ":.1%",
            },
            category_orders={"group": category_order},
            labels={
                "group": group_name,
                "readmission_rate": "Readmitted within 30 days",
            },
        )
        figure.update_yaxes(tickformat=".0%")
        alt = f"Observed 30-day readmission rate by {group_name.lower()}"
    elif visualization == "Encounter volume":
        figure = px.bar(
            summary,
            x="group",
            y="encounters",
            hover_data={
                "readmissions": ":,",
                "readmission_rate": ":.1%",
            },
            category_orders={"group": category_order},
            labels={"group": group_name, "encounters": "Study encounters"},
        )
        alt = f"Number of study encounters by {group_name.lower()}"
    else:
        outcome_summary = summary.melt(
            id_vars=["group", "encounters", "readmission_rate"],
            value_vars=["readmissions", "not_readmitted"],
            var_name="outcome",
            value_name="encounter_count",
        )
        outcome_labels = {
            "readmissions": "Readmitted within 30 days",
            "not_readmitted": "No readmission within 30 days",
        }
        outcome_summary["outcome"] = outcome_summary["outcome"].map(outcome_labels)

        if visualization == "Outcome proportions":
            outcome_summary["proportion"] = (
                outcome_summary["encounter_count"] / outcome_summary["encounters"]
            )
            y_column = "proportion"
            y_label = "Share of encounters"
            figure = px.bar(
                outcome_summary,
                x="group",
                y=y_column,
                color="outcome",
                barmode="stack",
                hover_data={
                    "encounters": ":,",
                    "encounter_count": ":,",
                    "proportion": ":.1%",
                },
                category_orders={
                    "group": category_order,
                    "outcome": list(outcome_labels.values()),
                },
                labels={
                    "group": group_name,
                    "outcome": "Encounter outcome",
                    y_column: y_label,
                },
            )
            figure.update_yaxes(tickformat=".0%")
            alt = f"Proportion of readmitted and other encounters by {group_name.lower()}"
        else:
            y_column = "encounter_count"
            figure = px.bar(
                outcome_summary,
                x="group",
                y=y_column,
                color="outcome",
                barmode="stack",
                hover_data={
                    "encounters": ":,",
                    "encounter_count": ":,",
                    "readmission_rate": ":.1%",
                },
                category_orders={
                    "group": category_order,
                    "outcome": list(outcome_labels.values()),
                },
                labels={
                    "group": group_name,
                    "outcome": "Encounter outcome",
                    y_column: "Study encounters",
                },
            )
            alt = f"Encounter outcome counts by {group_name.lower()}"

    figure.update_xaxes(tickangle=-25)
    return figure, alt


def render_cohort_overview(cohort: pd.DataFrame) -> None:
    """Render cohort metrics and 2–3 interactive descriptive visualizations."""
    st.header("Cohort overview", icon=":material/bar_chart:")
    st.caption(
        "Historical patterns in the study dataset. These group-level summaries "
        "describe the cohort; they do not estimate an individual patient's risk."
    )
    total_encounters = len(cohort)
    readmissions = int(cohort["readmitted_30d"].sum())
    with st.container(horizontal=True):
        st.metric("Study encounters", f"{total_encounters:,}", border=True)
        st.metric("Readmitted within 30 days", f"{readmissions:,}", border=True)
        st.metric(
            "Overall cohort rate",
            f"{cohort['readmitted_30d'].mean():.1%}",
            border=True,
        )

    st.session_state.setdefault(FIRST_VISUALIZATION_KEY, DEFAULT_VISUALIZATIONS[0])
    st.session_state.setdefault(SECOND_VISUALIZATION_KEY, DEFAULT_VISUALIZATIONS[1])
    group_column, first_column, second_column = st.columns([1.25, 1, 1])
    with group_column:
        group_name = st.selectbox(
            "Group encounters by",
            list(COHORT_GROUPS),
            key="cohort_group",
            on_change=_update_visualizations_for_group,
        )
    visualization_options = available_visualizations(group_name)
    with first_column:
        first_visualization = st.selectbox(
            "Visualization 1",
            visualization_options,
            key=FIRST_VISUALIZATION_KEY,
            on_change=_update_second_visualization,
        )
    with second_column:
        second_options = [
            NO_SECOND_VISUALIZATION,
            *(
                visualization
                for visualization in visualization_options
                if visualization != first_visualization
            ),
        ]
        second_visualization = st.selectbox(
            "Visualization 2 (optional)",
            second_options,
            key=SECOND_VISUALIZATION_KEY,
        )
    visualizations = selected_visualizations(
        first_visualization,
        second_visualization,
    )
    summary = summarize_cohort(cohort, group_name)

    columns = st.columns(len(visualizations))
    for column, visualization in zip(columns, visualizations):
        with column:
            st.subheader(visualization)
            figure, alt = _build_cohort_figure(
                summary,
                group_name,
                visualization,
                total_encounters,
                readmissions,
            )
            st.plotly_chart(figure, width="stretch", alt=alt)

    st.caption(
        "Rates are unadjusted. Differences between groups do not show cause and effect."
    )

    if st.toggle("Show cohort summary table", key="show_cohort_summary"):
        table = summary.rename(
            columns={
                "group": group_name,
                "encounters": "Study encounters",
                "readmissions": "Readmitted within 30 days",
                "not_readmitted": "No readmission within 30 days",
                "readmission_rate": "Readmission rate",
            }
        )
        table["Readmission rate"] = table["Readmission rate"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(
            table,
            hide_index=True,
            alt=f"Readmission summary by {group_name.lower()}",
        )
