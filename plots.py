"""Plotly helpers for the enriched CGM dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COLUMN_LABELS, PALETTE, SIGNAL_COLUMNS, SPLIT_COLORS, STRATUM_COLORS, STUDY_GROUP_LABELS

PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(0,0,0,0)"
GRID = "rgba(136,136,136,0.18)"
TEXT = "#003366"
ANCHOR_ROLE = "Forecast anchor"
SEGMENT_RESET_ROLE = "Segment reset"
WARMUP_ROLE = "Warm-up"
ANCHOR_MARKER_SIZE = 6
ANCHOR_MARKER_OPACITY = 0.82
SEGMENT_RESET_LINE_WIDTH = 1.5
SEGMENT_RESET_LINE_DASH = "dash"
WARMUP_END_LINE_WIDTH = 1.5
WARMUP_END_LINE_DASH = "solid"
WARMUP_BAND_FILLCOLOR = "rgba(136,136,136,0.14)"
PARTIAL_STREAM_LINE_COLOR = PALETTE[2]
PARTIAL_STREAM_LINE_DASH = "dot"
PARTIAL_STREAM_LINE_WIDTH = 1.5
PARTIAL_STREAM_BAND_FILLCOLOR = "rgba(91,186,186,0.16)"
PARTIAL_STREAM_LABEL = "CGM available, other modalities incomplete"

MEDICATION_CLASS_COLORS = {
    "Metformin": "#BA2828",
    "Insulin": "#003366",
    "GLP-1 / GIP-GLP-1": "#5BBABA",
    "SGLT2 inhibitor": "#FF0000",
    "Sulfonylurea": "#888888",
    "TZD": "#F28E2B",
}

GROUP_COLORS = {
    "T2D insulin": "#BA2828",
    "Prediabetes": "#003366",
    "T2D Non Insulin": "#5BBABA",
    "Normoglycemia": "#FF0000",
    "Healthy": "#FF0000",
    "Unknown": "#888888",
}

DEMO_CATEGORY_COLORS = {
    "F": "#BA2828",
    "M": "#003366",
    "Female": "#BA2828",
    "Male": "#003366",
    "Hispanic Or Latino": "#BA2828",
    "Not Hispanic Or Latino": "#003366",
    "Prefer Not To Say": "#888888",
    "Unknown": "#888888",
    "White Or Caucasian": "#5BBABA",
    "Black Or African American": "#003366",
    "Asian": "#BA2828",
    "American Indian Or Alaska Native": "#86CCCF",
    "Native Hawaiian Or Pacific Islander": "#FF0000",
    "Middle Eastern": "#11A579",
    "North African": "#3969AC",
    "Other Race, Ethnicity Or Origin": "#FF0000",
    "Other / Rare": "#888888",
    "Not Answered": "#A0A0A0",
}

HIGH_CONTRAST_SEQUENCE = [
    "#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888",
    "#F28E2B", "#7F3C8D", "#11A579", "#3969AC", "#E68310",
    "#A0A0A0", "#4B4B8F",
]


def pretty_col(name: str) -> str:
    return COLUMN_LABELS.get(str(name), str(name).replace("_", " ").strip().title())


def pretty_split(value) -> str:
    text = str(value).lower()
    if "val" in text:
        return "Validation"
    if "test" in text:
        return "Test"
    if "train" in text:
        return "Train"
    return "Other" if text in {"nan", "none", ""} else str(value).replace("_", " ").title()


def pretty_group(value) -> str:
    if pd.isna(value):
        return "Unknown"
    raw = str(value)
    return STUDY_GROUP_LABELS.get(raw, STUDY_GROUP_LABELS.get(raw.lower(), raw.replace("_", " ").replace("-", " ").title()))



def style_fig(fig, height: int = 380, title: str | None = None):

    fig.update_layout(

        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,

        font=dict(
            color=TEXT,
            family="Inter, Arial, sans-serif"
        ),

        # ---------------- TITLE ----------------
        title=dict(
            text=title,
            x=0.02,
            y=0.995,
            xanchor="left",
            yanchor="top",
            font=dict(
                size=19,
                color=TEXT
            ),
        ) if title else None,

        # ---------------- LEGEND ----------------
        # Keep legends below the title with enough reserved top margin.
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            title_text="",
            font=dict(size=12),
            tracegroupgap=6,
        ),

        # ---------------- MARGINS ----------------
        margin=dict(
            l=58,
            r=34,
            t=80 if title else 34,
            b=20,
        ),

        height=height,
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        title_font=dict(size=13),
        tickfont=dict(size=12),
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        title_font=dict(size=13),
        tickfont=dict(size=12),
    )

    return fig


def empty_plot(message: str = "No data available", height: int = 300):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=15, color="#888888"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=height)


def plot_split_distribution(split_df: pd.DataFrame, title: str = "Participant split"):
    if split_df is None or split_df.empty or "split" not in split_df.columns:
        return empty_plot("Split table is unavailable")
    tmp = split_df.copy()
    tmp["Split"] = tmp["split"].map(pretty_split)
    counts = tmp["Split"].value_counts().reindex(["Train", "Validation", "Test", "Other"]).dropna().reset_index()
    counts.columns = ["Split", "Participants"]
    fig = px.bar(counts, x="Split", y="Participants", color="Split", color_discrete_map=SPLIT_COLORS, text="Participants")
    fig.update_traces(textposition="outside", marker_line_color="rgba(0,51,102,.28)", marker_line_width=1)
    return style_fig(fig, height=390, title=title)


def plot_stacked(df: pd.DataFrame, x: str, color: str, title: str):
    if df is None or df.empty or x not in df.columns or color not in df.columns:
        return empty_plot(f"Need {pretty_col(x)} and {pretty_col(color)}")
    tmp = df[[x, color]].copy()
    tmp[x] = tmp[x].map(pretty_split) if x == "split" else tmp[x].astype(str)
    tmp[color] = tmp[color].map(pretty_group) if color in {"study_group", "stratum", "participants_study_group"} else tmp[color].fillna("Unknown").astype(str)
    tmp = tmp.value_counts().reset_index(name="Participants")
    fig = px.bar(tmp, x=x, y="Participants", color=color, color_discrete_sequence=PALETTE, labels={x: pretty_col(x), color: pretty_col(color)})
    return style_fig(fig, height=410, title=title)


def plot_count_bar(df: pd.DataFrame, column: str, title: str, top_n: int = 25):
    if df is None or df.empty or not column or column not in df.columns:
        return empty_plot(f"Missing {pretty_col(column)}")
    values = df[column].map(pretty_group) if column in {"study_group", "stratum", "participants_study_group"} else df[column].fillna("Unknown").astype(str)
    tmp = values.value_counts().head(top_n).reset_index()
    tmp.columns = [pretty_col(column), "Participants"]
    fig = px.bar(tmp, x=pretty_col(column), y="Participants", color_discrete_sequence=[PALETTE[1]])
    fig.update_traces(marker_line_color="rgba(91,186,186,.35)", marker_line_width=1)
    return style_fig(fig, height=370, title=title)


def plot_histogram(df: pd.DataFrame, column: str, title: str, nbins: int = 45):
    if df is None or df.empty or not column or column not in df.columns:
        return empty_plot(f"Missing {pretty_col(column)}")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return empty_plot(f"No numeric values for {pretty_col(column)}")
    label = pretty_col(column)
    fig = px.histogram(values.to_frame(label), x=label, nbins=nbins, color_discrete_sequence=[PALETTE[2]])
    fig.update_traces(marker_line_color="rgba(0,51,102,.25)", marker_line_width=1)
    return style_fig(fig, height=360, title=title)



def plot_participant_timeseries(
    ts_df: pd.DataFrame,
    time_col: str | None,
    selected_signals: list[str],
    segment_boundaries_df: pd.DataFrame | None = None,
    anchor_df: pd.DataFrame | None = None,
    warmup_hours: int | float | None = None,
    segment_gap_minutes: int | float | None = None,
    title: str = "Participant signal timeline",
):
    if ts_df is None or ts_df.empty:
        return empty_plot("No participant time series loaded", height=620)
    if not time_col or time_col not in ts_df.columns:
        return empty_plot("Timestamp column could not be detected", height=620)
    labels = {col: label for label, col in SIGNAL_COLUMNS.items()}
    available = [col for col in selected_signals if col in ts_df.columns]
    if not available:
        return empty_plot("Selected signals are unavailable for this participant", height=620)

    fig = make_subplots(rows=len(available), cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=[labels.get(c, pretty_col(c)) for c in available])
    color_cycle = [PALETTE[1], PALETTE[2], PALETTE[0], PALETTE[4]]
    for idx, col in enumerate(available, start=1):
        fig.add_trace(
            go.Scatter(
                x=ts_df[time_col],
                y=pd.to_numeric(ts_df[col], errors="coerce"),
                mode="lines",
                name=labels.get(col, pretty_col(col)),
                line=dict(color=color_cycle[(idx - 1) % len(color_cycle)], width=1.6),
                connectgaps=False,
                hovertemplate="%{x}<br>%{y:.2f}<extra>" + labels.get(col, pretty_col(col)) + "</extra>",
            ),
            row=idx,
            col=1,
        )
        fig.update_yaxes(title_text=labels.get(col, pretty_col(col)), row=idx, col=1)

    all_segment_starts = pd.to_datetime(
        segment_boundaries_df.get("start", pd.Series(dtype=object)),
        errors="coerce",
    ).dropna().sort_values() if segment_boundaries_df is not None and not segment_boundaries_df.empty else pd.Series(dtype="datetime64[ns]")
    stream_start = all_segment_starts.iloc[0] if not all_segment_starts.empty else None
    if stream_start is None:
        ts_times = pd.to_datetime(ts_df[time_col], errors="coerce").dropna()
        stream_start = ts_times.min() if not ts_times.empty else None

    if not all_segment_starts.empty:
        reset_times = all_segment_starts.iloc[1:]
        for reset_time in reset_times:
            fig.add_shape(
                type="line",
                x0=reset_time,
                x1=reset_time,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(
                    color=STRATUM_COLORS[SEGMENT_RESET_ROLE],
                    width=SEGMENT_RESET_LINE_WIDTH,
                    dash=SEGMENT_RESET_LINE_DASH,
                ),
            )
        if not reset_times.empty:
            reset_label = "segment reset"
            if segment_gap_minutes is not None:
                reset_label = f"segment reset, gap > {segment_gap_minutes:g} min"
            fig.add_annotation(
                x=reset_times.iloc[0],
                y=1,
                xref="x",
                yref="paper",
                text=reset_label,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(color=STRATUM_COLORS[SEGMENT_RESET_ROLE], size=11),
            )

    raw_signal_start = pd.to_datetime(
        ts_df.loc[ts_df[available[0]].notna(), time_col], errors="coerce"
    ).min() if available else None

    if (
        stream_start is not None
        and raw_signal_start is not None
        and pd.notna(raw_signal_start)
        and raw_signal_start < stream_start
    ):
        fig.add_vrect(
            x0=raw_signal_start,
            x1=stream_start,
            fillcolor=PARTIAL_STREAM_BAND_FILLCOLOR,
            line_width=PARTIAL_STREAM_LINE_WIDTH,
            line_color=PARTIAL_STREAM_LINE_COLOR,
            line_dash=PARTIAL_STREAM_LINE_DASH,
            row="all",
            col=1,
        )
        fig.add_annotation(
            x=raw_signal_start,
            y=1,
            xref="x",
            yref="paper",
            text=PARTIAL_STREAM_LABEL,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=PARTIAL_STREAM_LINE_COLOR, size=11),
        )

    if stream_start is not None and warmup_hours is not None:
        warmup_limit = float(warmup_hours)
        warmup_end = stream_start + pd.Timedelta(hours=warmup_limit)
        if warmup_limit > 0:
            fig.add_vrect(
                x0=stream_start,
                x1=warmup_end,
                fillcolor=WARMUP_BAND_FILLCOLOR,
                line_width=0,
                row="all",
                col=1,
            )
        fig.add_shape(
            type="line",
            x0=warmup_end,
            x1=warmup_end,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color=STRATUM_COLORS[WARMUP_ROLE],
                width=WARMUP_END_LINE_WIDTH,
                dash=WARMUP_END_LINE_DASH,
            ),
        )
        fig.add_annotation(
            x=warmup_end,
            y=1,
            xref="x",
            yref="paper",
            text=f"warm-up ends, {warmup_limit:g} h from stream start",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=STRATUM_COLORS[WARMUP_ROLE], size=11),
        )

    anchor_columns = {"anchor_timestamp", "hours_since_start", "segment_id", "anchor_time_idx"}
    if anchor_df is not None and not anchor_df.empty and anchor_columns.issubset(anchor_df.columns):
        anchor_work = anchor_df.copy()
        anchor_work["anchor_timestamp"] = pd.to_datetime(
            anchor_work["anchor_timestamp"], errors="coerce"
        )
        anchor_work["hours_since_segment_start"] = pd.to_numeric(
            anchor_work["hours_since_start"], errors="coerce"
        )
        anchor_work["hours_since_stream_start"] = (
            (anchor_work["anchor_timestamp"] - stream_start).dt.total_seconds() / 3600.0
            if stream_start is not None
            else np.nan
        )
        signal_at_time = pd.DataFrame({
            "anchor_timestamp": pd.to_datetime(ts_df[time_col], errors="coerce"),
            "_anchor_y": pd.to_numeric(ts_df[available[0]], errors="coerce"),
        }).dropna(subset=["anchor_timestamp"]).drop_duplicates(
            "anchor_timestamp", keep="first"
        )
        anchor_work = anchor_work.merge(signal_at_time, on="anchor_timestamp", how="left")
        anchor_work = anchor_work.dropna(subset=["anchor_timestamp", "_anchor_y"])
        if not anchor_work.empty:
            fig.add_trace(
                go.Scatter(
                    x=anchor_work["anchor_timestamp"],
                    y=anchor_work["_anchor_y"],
                    mode="markers",
                    name=ANCHOR_ROLE,
                    marker=dict(
                        color=STRATUM_COLORS[ANCHOR_ROLE],
                        size=ANCHOR_MARKER_SIZE,
                        opacity=ANCHOR_MARKER_OPACITY,
                        symbol="circle",
                    ),
                    customdata=anchor_work[
                        ["hours_since_stream_start", "hours_since_segment_start", "segment_id", "anchor_time_idx"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{x}<br>Hours since stream start: %{customdata[0]:.2f}<br>"
                        "Hours since segment start: %{customdata[1]:.2f}<br>"
                        "Segment: %{customdata[2]}<br>Anchor index: %{customdata[3]}"
                        f"<extra>{ANCHOR_ROLE}</extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    fig.update_xaxes(title="Time", tickformat="%b %d %H:%M", rangeslider_visible=False)
    fig.update_layout(hovermode="x unified", dragmode="zoom")
    return style_fig(fig, height=max(650, 230 * len(available)), title=title)


def plot_static_feature_table(static_row: pd.DataFrame, query: str = "") -> pd.DataFrame:
    if static_row is None or static_row.empty:
        return pd.DataFrame(columns=["Feature", "Value"])
    row = static_row.iloc[0]
    values = ["" if pd.isna(v) else str(v) for v in row.values]
    out = pd.DataFrame({"Feature": [pretty_col(c) for c in row.index], "Raw feature": row.index, "Value": values})
    if query:
        q = query.lower()
        out = out[out["Feature"].astype(str).str.lower().str.contains(q, na=False) | out["Raw feature"].astype(str).str.lower().str.contains(q, na=False) | out["Value"].astype(str).str.lower().str.contains(q, na=False)]
    return out[["Feature", "Value"]]


def plot_missingness(ts_df: pd.DataFrame):
    if ts_df is None or ts_df.empty:
        return empty_plot("Load a participant time series to view missingness")
    rows = []
    for label, col in SIGNAL_COLUMNS.items():
        if col in ts_df.columns:
            rows.append({"Variable": label, "Missing fraction": ts_df[col].isna().mean()})
    if not rows:
        return empty_plot("No core signal columns available")
    miss = pd.DataFrame(rows)
    fig = px.bar(miss, x="Variable", y="Missing fraction", color_discrete_sequence=[PALETTE[0]])
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, height=340, title="Signal missingness for selected participant")


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def plot_population_violin(df: pd.DataFrame, value_col: str, group_col: str | None, title: str):
    values = _numeric_series(df, value_col)
    if values.empty:
        return empty_plot(f"No numeric values for {pretty_col(value_col)}")
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if group_col and group_col in tmp.columns:
        tmp["Group"] = tmp[group_col].map(pretty_group) if group_col in {"study_group", "stratum"} else tmp[group_col].fillna("Unknown").astype(str)
        fig = px.violin(tmp, x="Group", y=value_col, color="Group", box=True, points=False, color_discrete_sequence=PALETTE)
        fig.update_xaxes(title="", tickangle=0)
    else:
        tmp["Cohort"] = "All participants"
        fig = px.violin(tmp, x="Cohort", y=value_col, box=True, points=False, color_discrete_sequence=[PALETTE[1]])
        fig.update_xaxes(title="")
    fig.update_yaxes(title=pretty_col(value_col))
    return style_fig(fig, height=430, title=title)


def plot_cluster_box_strip(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str,
    cluster_order: list[str],
    cluster_colors: dict[str, str],
    title: str,
):
    """Box plot with jittered individual points, grouped by a frozen cluster label."""
    values = _numeric_series(df, value_col)
    if values.empty:
        return empty_plot(f"No numeric values for {pretty_col(value_col)}")
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col, cluster_col])
    fig = px.box(
        tmp,
        x=cluster_col,
        y=value_col,
        color=cluster_col,
        points="all",
        category_orders={cluster_col: cluster_order},
        color_discrete_map=cluster_colors,
    )
    fig.update_traces(
        boxpoints="all",
        jitter=0.4,
        pointpos=0,
        marker=dict(size=4, opacity=0.55),
        line=dict(width=1.4),
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title=pretty_col(value_col))
    fig.update_layout(showlegend=False)
    return style_fig(fig, height=380, title=title)


def plot_stacked_histogram(df: pd.DataFrame, value_col: str, group_col: str | None, title: str, nbins: int = 35):
    values = _numeric_series(df, value_col)
    if values.empty:
        return empty_plot(f"No numeric values for {pretty_col(value_col)}")
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if group_col and group_col in tmp.columns:
        tmp["Group"] = tmp[group_col].map(pretty_group) if group_col in {"study_group", "stratum"} else tmp[group_col].fillna("Unknown").astype(str)
        fig = px.histogram(tmp, x=value_col, color="Group", nbins=nbins, barmode="overlay", opacity=0.62, histnorm="probability density", color_discrete_sequence=PALETTE)
    else:
        fig = px.histogram(tmp, x=value_col, nbins=nbins, histnorm="probability density", color_discrete_sequence=[PALETTE[2]])
    fig.update_xaxes(title=pretty_col(value_col))
    fig.update_yaxes(title="Density")
    return style_fig(fig, height=410, title=title)


def plot_kde_curves(df: pd.DataFrame, value_col: str, group_col: str | None, title: str):
    values = _numeric_series(df, value_col)
    if values.empty or len(values) < 3:
        return empty_plot(f"Not enough values for {pretty_col(value_col)} KDE")
    lo, hi = float(values.quantile(0.01)), float(values.quantile(0.99))
    if lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return empty_plot(f"No spread for {pretty_col(value_col)}")
    xs = np.linspace(lo, hi, 180)
    fig = go.Figure()
    if group_col and group_col in df.columns:
        groups = df[group_col].map(pretty_group) if group_col in {"study_group", "stratum"} else df[group_col].fillna("Unknown").astype(str)
        tmp = pd.DataFrame({"value": pd.to_numeric(df[value_col], errors="coerce"), "group": groups}).dropna()
        for idx, (group, g) in enumerate(tmp.groupby("group")):
            vals = g["value"].to_numpy(dtype=float)
            if len(vals) < 3:
                continue
            bw = max(np.std(vals) * (len(vals) ** (-1 / 5)), 1e-6)
            density = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1) / (len(vals) * bw * np.sqrt(2 * np.pi))
            fig.add_trace(go.Scatter(x=xs, y=density, mode="lines", name=str(group), line=dict(color=PALETTE[idx % len(PALETTE)], width=2.6)))
    else:
        vals = values.to_numpy(dtype=float)
        bw = max(np.std(vals) * (len(vals) ** (-1 / 5)), 1e-6)
        density = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1) / (len(vals) * bw * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(x=xs, y=density, mode="lines", name="All participants", line=dict(color=PALETTE[1], width=2.4)))
    fig.update_xaxes(title=pretty_col(value_col))
    fig.update_yaxes(title="Estimated density")
    return style_fig(fig, height=430, title=title)


def plot_proportion_bar(df: pd.DataFrame, column: str, title: str):
    if df is None or df.empty or column not in df.columns:
        return empty_plot(f"Missing {pretty_col(column)}")
    values = df[column].map(pretty_group) if column in {"study_group", "stratum"} else df[column].fillna("Unknown").astype(str)
    tmp = values.value_counts(normalize=True).mul(100).reset_index()
    tmp.columns = [pretty_col(column), "Percent of participants"]
    fig = px.bar(tmp, x=pretty_col(column), y="Percent of participants", color=pretty_col(column), text="Percent of participants", color_discrete_sequence=PALETTE)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Participants [%]", range=[0, max(5, tmp["Percent of participants"].max() * 1.18)])
    return style_fig(fig, height=410, title=title)


def plot_clinical_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str | None,
    title: str,
):
    if x_col not in df.columns or y_col not in df.columns:
        return empty_plot(f"Missing columns for scatter plot")
    tmp = df.copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col])
    if tmp.empty:
        return empty_plot("No paired data available for scatter")
    if group_col and group_col in tmp.columns:
        tmp["Group"] = tmp[group_col].map(pretty_group) if group_col in {"study_group", "stratum"} else tmp[group_col].fillna("Unknown").astype(str)
        fig = px.scatter(tmp, x=x_col, y=y_col, color="Group", opacity=0.62, color_discrete_sequence=PALETTE)
    else:
        fig = px.scatter(tmp, x=x_col, y=y_col, opacity=0.62, color_discrete_sequence=[PALETTE[1]])
    fig.update_xaxes(title=pretty_col(x_col))
    fig.update_yaxes(title=pretty_col(y_col))
    return style_fig(fig, height=430, title=title)


def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    labels: dict[str, str] | None = None,
    title: str = "Correlation matrix",
):
    avail = [c for c in cols if c in df.columns]
    if len(avail) < 2:
        return empty_plot("Not enough columns for correlation matrix")
    tmp = df[avail].apply(pd.to_numeric, errors="coerce").dropna()
    if len(tmp) < 5:
        return empty_plot("Not enough rows with complete data for correlation")
    corr = tmp.corr()
    lbl = [(labels or {}).get(c, pretty_col(c)) for c in avail]
    z = corr.values
    text_vals = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=lbl,
        y=lbl,
        text=text_vals,
        texttemplate="%{text}",
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="r", thickness=14, len=0.72, tickvals=[-1, -0.5, 0, 0.5, 1]),
    ))
    fig.update_layout(
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    return style_fig(fig, height=490, title=title)


def plot_preprocessing_pipeline(pipeline_df: pd.DataFrame):
    if pipeline_df is None or pipeline_df.empty:
        return empty_plot("Pipeline summary unavailable", height=420)
    df = pipeline_df.copy().reset_index(drop=True)
    df["x"] = np.arange(len(df))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["x"],
        y=[0] * len(df),
        mode="markers+text+lines",
        marker=dict(size=34, color=[PALETTE[i % len(PALETTE)] for i in range(len(df))], line=dict(color="rgba(0,51,102,.35)", width=2)),
        line=dict(color="rgba(0,51,102,.30)", width=3),
        text=df["Stage"],
        textposition="bottom center",
        customdata=np.stack([df["Participants"].astype(str), df["Windows"].astype(str), df["Thresholds / rule"].astype(str)], axis=-1),
        hovertemplate="<b>%{text}</b><br>Participants: %{customdata[0]}<br>Windows: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        showlegend=False,
    ))
    for i in range(len(df) - 1):
        fig.add_annotation(x=i + 0.5, y=0, text="->", showarrow=False, font=dict(size=24, color=TEXT))
    fig.update_xaxes(visible=False, range=[-0.55, len(df) - 0.45])
    fig.update_yaxes(visible=False, range=[-0.55, 0.55])
    return style_fig(fig, height=430, title="Stage 1-4 preprocessing flow")


def _flag_series(series: pd.Series) -> pd.Series:
    """Return medication/demo indicator-like values as 0/1 floats."""
    if series.empty:
        return pd.Series(dtype=float)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0) > 0).astype(float)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "present", "current", "checked"}).astype(float)


def plot_medication_prevalence_by_stratum(df: pd.DataFrame, med_cols: dict[str, str], group_col: str, title: str):
    rows = []
    if df.empty or group_col not in df.columns:
        return style_fig(go.Figure(), height=420, title=title)
    for col, label in med_cols.items():
        if col not in df.columns:
            continue
        work = pd.DataFrame({"Study group": df[group_col].map(pretty_group), "Flag": _flag_series(df[col])})
        for group, sub in work.groupby("Study group"):
            rows.append({"Study group": group, "Drug class": label, "Participants on class [%]": sub["Flag"].mean() * 100})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return style_fig(go.Figure(), height=420, title=title)
    fig = px.bar(
        plot_df,
        x="Study group",
        y="Participants on class [%]",
        color="Drug class",
        barmode="stack",
        text="Participants on class [%]",
        color_discrete_map=MEDICATION_CLASS_COLORS,
    )
    fig.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        marker_line_color="rgba(0,51,102,.38)",
        marker_line_width=0.9,
        hovertemplate="%{x}<br>%{legendgroup}: %{y:.1f}%<extra></extra>",
    )
    fig.update_yaxes(title="Participants on medication class [%]")
    fig.update_xaxes(title="Study group")
    return style_fig(fig, height=500, title=title)


def plot_coprescription_heatmap(df: pd.DataFrame, med_cols: dict[str, str], title: str):
    cols = [col for col in med_cols if col in df.columns]
    if df.empty or not cols:
        return style_fig(go.Figure(), height=600, title=title)
    flags = pd.DataFrame({med_cols[col]: _flag_series(df[col]) for col in cols})
    labels = list(flags.columns)
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    n = max(len(flags), 1)
    for a in labels:
        for b in labels:
            matrix.loc[a, b] = ((flags[a] > 0) & (flags[b] > 0)).sum() / n * 100
    fig = px.imshow(
        matrix.astype(float),
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=[[0, "rgba(255,255,255,0.0)"], [0.5, PALETTE[2]], [1, PALETTE[1]]],
        labels=dict(x="Drug class", y="Drug class", color="Participants [%]"),
    )
    fig.update_traces(hovertemplate="%{y} + %{x}<br>%{z:.1f}% of participants<extra></extra>")
    fig.update_xaxes(
        tickangle=0,
        automargin=True,
        side="bottom",
    )
    return style_fig(fig, height=500, title=title)


def plot_medication_burden(df: pd.DataFrame, med_cols: dict[str, str], group_col: str, title: str):
    cols = [col for col in med_cols if col in df.columns]
    if df.empty or not cols or group_col not in df.columns:
        return style_fig(go.Figure(), height=420, title=title)
    work = pd.DataFrame({
        "Study group": df[group_col].map(pretty_group),
        "Diabetes drug classes [count]": pd.DataFrame({_c: _flag_series(df[_c]) for _c in cols}).sum(axis=1),
    })
    fig = px.histogram(
        work,
        x="Diabetes drug classes [count]",
        color="Study group",
        barmode="group",
        color_discrete_sequence=PALETTE,
        category_orders={"Diabetes drug classes [count]": sorted(work["Diabetes drug classes [count]"].dropna().unique())},
    )
    fig.update_xaxes(dtick=1, title="Number of diabetes drug classes")
    fig.update_yaxes(title="Participants")
    return style_fig(fig, height=460, title=title)


def plot_hba1c_vs_med_count(df: pd.DataFrame, med_cols: dict[str, str], hba1c_col: str, group_col: str, title: str):
    cols = [col for col in med_cols if col in df.columns]
    if df.empty or not cols or hba1c_col not in df.columns or group_col not in df.columns:
        return style_fig(go.Figure(), height=430, title=title)
    work = pd.DataFrame({
        "Study group": df[group_col].map(pretty_group),
        "HbA1c [%]": pd.to_numeric(df[hba1c_col], errors="coerce"),
        "Diabetes drug classes [count]": pd.DataFrame({_c: _flag_series(df[_c]) for _c in cols}).sum(axis=1),
    }).dropna(subset=["HbA1c [%]"])
    fig = px.strip(
        work,
        x="Diabetes drug classes [count]",
        y="HbA1c [%]",
        color="Study group",
        stripmode="overlay",
        color_discrete_map=GROUP_COLORS,
    )
    symbol_map = {"T2D insulin": "circle", "Prediabetes": "diamond", "T2D Non Insulin": "square", "Normoglycemia": "x", "Healthy": "x"}
    for trace in fig.data:
        trace.marker.size = 8
        trace.marker.opacity = 0.78
        trace.marker.line = dict(color="rgba(0,51,102,.42)", width=0.65)
        trace.marker.symbol = symbol_map.get(trace.name, "circle")
    fig.update_xaxes(dtick=1, title="Number of diabetes drug classes")
    fig.update_yaxes(title="HbA1c [%]")
    return style_fig(fig, height=460, title=title)


def plot_stacked_proportion(
    df: pd.DataFrame,
    x_col: str,
    color_col: str,
    title: str,
    x_title: str = "Study group",
    color_title: str = "Category",
    color_map: dict[str, str] | None = None,
    horizontal: bool = False,
    min_percent_for_label: float = 7.0,
):
    if df.empty or x_col not in df.columns or color_col not in df.columns:
        return style_fig(go.Figure(), height=420, title=title)
    work = df[[x_col, color_col]].copy()
    work[x_col] = work[x_col].map(pretty_group) if x_col in {"study_group", "stratum"} else work[x_col].fillna("Unknown").astype(str)
    work[color_col] = work[color_col].fillna("Unknown").astype(str).str.replace("_", " ").str.strip().str.title()
    counts = work.groupby([x_col, color_col], dropna=False).size().reset_index(name="Participants")
    totals = counts.groupby(x_col)["Participants"].transform("sum")
    counts["Percent within group"] = counts["Participants"] / totals * 100
    counts["Segment label"] = counts["Percent within group"].where(counts["Percent within group"] >= min_percent_for_label).map(lambda v: f"{v:.0f}%" if pd.notna(v) else "")
    cmap = color_map or {cat: DEMO_CATEGORY_COLORS.get(cat, HIGH_CONTRAST_SEQUENCE[i % len(HIGH_CONTRAST_SEQUENCE)]) for i, cat in enumerate(counts[color_col].dropna().unique())}
    if horizontal:
        fig = px.bar(
            counts,
            y=x_col,
            x="Percent within group",
            color=color_col,
            orientation="h",
            barmode="stack",
            text="Segment label",
            color_discrete_map=cmap,
            labels={x_col: x_title, color_col: color_title},
            hover_data={"Participants": True, "Percent within group": ":.1f"},
        )
        fig.update_xaxes(title="Participants within group [%]", range=[0, 100])
        fig.update_yaxes(title=x_title)
    else:
        fig = px.bar(
            counts,
            x=x_col,
            y="Percent within group",
            color=color_col,
            barmode="stack",
            text="Segment label",
            color_discrete_map=cmap,
            labels={x_col: x_title, color_col: color_title},
            hover_data={"Participants": True, "Percent within group": ":.1f"},
        )
        fig.update_yaxes(title="Participants within group [%]", range=[0, 100])
        fig.update_xaxes(title=x_title)
    fig.update_traces(textposition="inside", marker_line_color="rgba(0,51,102,.32)", marker_line_width=0.55)
    return style_fig(fig, height=520 if horizontal else 470, title=title)


def plot_crosstab_heatmap(df: pd.DataFrame, row_col: str, col_col: str, title: str):
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        return style_fig(go.Figure(), height=600, title=title)
    rows = df[row_col].map(pretty_group) if row_col in {"study_group", "stratum"} else df[row_col].fillna("Unknown").astype(str)
    cols = df[col_col].map(pretty_group) if col_col in {"study_group", "stratum", "participants_study_group", "raw_study_group"} else df[col_col].fillna("Unknown").astype(str)
    tab = pd.crosstab(rows, cols)
    if tab.empty:
        return style_fig(go.Figure(), height=430, title=title)
    fig = px.imshow(
        tab,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, "rgba(255,255,255,0.0)"], [0.5, PALETTE[2]], [1, PALETTE[1]]],
        labels=dict(x="Self-reported / source group", y="Final analysis stratum", color="Participants"),
    )
    
    return style_fig(fig, height=max(440, 60 * len(tab.index) + 230), title=title)

