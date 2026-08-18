
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CANONICAL_STREAM_CHECKPOINT_VAL_PINBALL_MGDL,
    COLUMN_LABELS,
    PERSONALIZATION_WARMUP_HOURS,
    SIGNAL_COLUMNS,
    STREAM_TIMELINE_SIGNAL,
    T2D_SUBTYPE_CLUSTER_COLORS,
    T2D_SUBTYPE_CLUSTER_INTERPRETATION,
    T2D_SUBTYPE_CLUSTER_ORDER,
    T2D_SUBTYPE_FACTOR_COLUMNS,
)
from data_loader import (
    CGM_SEGMENT_GAP_MINUTES,
    check_file_availability,
    detect_timestamp_column,
    load_canonical_stream_split,
    load_cohort,
    load_cohort_selection_metadata,
    load_cgm_participant_metrics,
    load_forecast_anchors,
    load_measurements_long,
    load_medications_long,
    load_original_condition_groups,
    load_original_participants,
    load_participant_timeseries,
    load_segments,
    load_static_features,
    load_stream_summary,
    load_t2d_subtype_clinical_factors,
    multimodal_metadata,
    segment_boundaries,
)
from metrics import participant_time_stats
from plots import (
    plot_clinical_scatter,
    plot_cluster_box_strip,
    plot_coprescription_heatmap,
    plot_crosstab_heatmap,
    plot_correlation_matrix,
    plot_histogram,
    plot_hba1c_vs_med_count,
    plot_missingness,
    plot_medication_burden,
    plot_medication_prevalence_by_stratum,
    plot_kde_curves,
    plot_population_violin,
    plot_preprocessing_pipeline,
    plot_proportion_bar,
    plot_participant_timeseries,
    plot_split_distribution,
    plot_stacked,
    plot_stacked_histogram,
    plot_stacked_proportion,
    plot_static_feature_table,
    pretty_group,
    pretty_split,
)

st.set_page_config(page_title="AI-READI Dashboard", page_icon="CGM", layout="wide")


def _require_password() -> None:
    if st.session_state.get("dashboard_authenticated", False):
        return
    st.title("AI-READI Dashboard")
    password = st.text_input("Password", type="password")
    if not password:
        st.stop()
    if password != st.secrets["dashboard_password"]:
        st.error("Incorrect password.")
        st.stop()
    st.session_state["dashboard_authenticated"] = True
    st.rerun()


_require_password()

st.markdown('''
<style>
:root { --red:#BA2828; --blue:#003366; --teal:#5BBABA; --hot:#FF0000; --gray:#888888; --panel:rgba(248,251,252,.84); }
html { scroll-behavior: smooth; }

.stApp {
  background:
    radial-gradient(circle at 0% 0%, rgba(91,186,186,.22), transparent 28%),
    
    linear-gradient(180deg, #EEF7F8 0%, #F9FCFD 48%, #EDF4F5 100%);
  color: var(--blue);
}           
.block-container { max-width: calc(100vw - 1.8rem) !important; padding: .8rem .9rem 1.4rem !important; }
[data-testid="stHorizontalBlock"] { gap: .85rem; }
[data-testid="stVerticalBlock"] { gap: .65rem; }
[data-testid="stDataFrame"] { border: 1px solid rgba(0,51,102,.12); border-radius: 10px; overflow: hidden; }
/* Remove borders around Plotly charts */
.stPlotlyChart {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 0 !important;
}

/* Align charts at same vertical level */
div[data-testid="column"] {
  align-self: flex-start !important;
}

.hero {
  text-align: center;
  padding: 3.2rem 1rem 2.2rem;
  margin: .4rem auto 1.2rem;
  max-width: 1500px;
  border-radius: 30px;
  
  animation: riseIn .65s ease-out;
}
.hero h1 { margin: 0; color: var(--blue); font-size: clamp(2rem, 3.9vw, 4.1rem); line-height: 1; letter-spacing: 0; white-space: nowrap; }
.hero p {
  max-width: 1200px;
  margin: .9rem auto 0 auto;
  color: #31556b;
  font-size: 1rem;
  text-align: center;
  line-height: 1.5;
}
.kicker { text-transform: uppercase; letter-spacing: .12em; color: var(--red); font-size: .76rem; font-weight: 800; margin-bottom: .55rem; }
.toc { display:flex; flex-wrap:wrap; gap:.45rem .6rem; margin:.55rem 0 .7rem; padding:.45rem 0; border-radius:0; background:transparent; border:0; }
.toc a { color: var(--blue); text-decoration:none; font-weight:750; font-size:.9rem; padding:.25rem .45rem; border-radius:999px; }
.toc a:hover { background:rgba(91,186,186,.18); }
.section { display:none; }
.section-heading { padding: .5rem .1rem .25rem; margin: .65rem 0 .25rem; border: 0; background: transparent; box-shadow: none; }
.section-heading h2 { margin: 0 0 .16rem; color: var(--blue); font-size: clamp(1.25rem, 2vw, 1.72rem); line-height: 1.08; letter-spacing: 0; white-space: nowrap; }
.section-heading p { color:#46687c; margin:.15rem 0 .55rem; max-width:1280px; font-size:.94rem; }
/* Center the top metrics row */
.metric-card {
  text-align: center !important;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
/* Bigger and more centered metric numbers */
.metric-label {
  text-align: center;
  width: 100%;
  color: var(--gray);
  font-size: .74rem;
  text-transform: uppercase;
  letter-spacing: .11em;
  font-weight: 850;
}
.metric-value {
  text-align: center;
  width: 100%;
  color: var(--blue);
  font-size: 1.85rem;
  font-weight: 950;
  margin-top: .25rem;
  line-height: 1.05;
}
.metric-help {
  text-align: center;
  width: 100%;
  color: #557184;
  font-size: .85rem;
  margin-top: .25rem;
}
/* Center the top navigation */
.toc {
  display: flex;
  justify-content: center !important;
  align-items: center;
  flex-wrap: wrap;
  gap: 1.1rem;
  margin: 1rem auto 1.2rem;
  padding: .55rem 1rem;
  width: fit-content;
  max-width: 95vw;
}
/* Make the top metric section more balanced */
.block-container > div:nth-of-type(3) {
  text-align: center;
}
            

.profile-card, .participant-card { min-height: 84px; padding: .72rem .8rem; border-radius: 8px; border: 1px solid rgba(0,51,102,.12); background: rgba(255,255,255,.22); box-shadow: none; }

.profile-grid {
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: 1.2rem;
  margin: 1.4rem 0 2.4rem;
  align-items: start;
}

.profile-card {
  min-height: auto;
  padding: 0;
  border: none;
  background: transparent;
  box-shadow: none;
}

.profile-card b {
  color: var(--blue);
  display: block;
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .09em;
  margin-bottom: .35rem;
}

.profile-card span {
  color: #263f51;
  font-weight: 850;
  font-size: 1.08rem;
  line-height: 1.35;
}

.pipeline-overview {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .75rem;
  margin: .8rem 0 1rem;
}
.pipeline-mini {
  padding: .65rem .7rem;
  border-left: 4px solid var(--teal);
  background: rgba(255,255,255,.22);
}
.pipeline-mini b {
  display: block;
  color: var(--blue);
  font-size: .82rem;
  line-height: 1.2;
}
.pipeline-mini span {
  display: block;
  color: var(--blue);
  font-size: 1.55rem;
  line-height: 1.05;
  font-weight: 950;
  margin: .25rem 0 .1rem;
}
.pipeline-mini em {
  display: block;
  color: #5c7280;
  font-style: normal;
  font-size: .78rem;
}
.pipeline-spotlight {
  margin: .6rem 0 1rem;
  padding: 1.1rem 1.15rem;
  background: linear-gradient(135deg, rgba(255,255,255,.42), rgba(91,186,186,.14));
  border-left: 7px solid var(--blue);
}
.pipeline-spotlight .big-number {
  color: var(--blue);
  font-size: clamp(3.2rem, 7vw, 6rem);
  line-height: .95;
  font-weight: 950;
}
.pipeline-spotlight .big-label {
  color: var(--gray);
  text-transform: uppercase;
  letter-spacing: .12em;
  font-weight: 850;
  font-size: .78rem;
  margin-top: .25rem;
}
.pipeline-spotlight h3 {
  color: var(--blue);
  margin: .8rem 0 .35rem;
  font-size: clamp(1.25rem, 2vw, 1.8rem);
  line-height: 1.05;
}
.pipeline-spotlight p {
  color: #314e60;
  font-size: 1rem;
  line-height: 1.45;
  max-width: 1180px;
  margin: 0;
}
.pipeline-chips { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; }
.pipeline-chip {
  color: var(--blue);
  border: 1px solid rgba(0,51,102,.18);
  background: rgba(255,255,255,.38);
  padding: .34rem .55rem;
  font-size: .82rem;
  font-weight: 750;
}
.pass-badge, .fail-badge, .warn-badge { display:inline-block; padding:.46rem .68rem; border-radius:999px; font-weight:800; margin:.12rem .25rem .12rem 0; }
.pass-badge { background:rgba(91,186,186,.28); color:var(--blue); border:1px solid rgba(91,186,186,.75); }
.fail-badge { background:rgba(186,40,40,.13); color:var(--red); border:1px solid rgba(186,40,40,.55); }
.warn-badge { background:rgba(136,136,136,.14); color:#394b5a; border:1px solid rgba(136,136,136,.35); }
.small-note { color:#5d7483; font-size:.85rem; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { border-radius:999px; background:rgba(255,255,255,.42); padding:.45rem .85rem; height:auto; }
@keyframes riseIn { from { transform: translateY(8px); opacity:0; } to { transform: translateY(0); opacity:1; } }
@media (max-width: 900px) { .hero h1, .section h2 { white-space: normal; } .block-container { max-width: 100vw !important; padding: .55rem !important; } .pipeline-overview { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.forecast-method-card {
  margin: .8rem 0 .25rem;
  padding: 1.1rem 1.25rem;
  border: 1px solid rgba(0,51,102,.12);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,255,255,.58), rgba(91,186,186,.12));
}
.forecast-bar-wrap {
  position: relative;
  min-height: 116px;
  padding: 1.2rem 1.4rem 2.9rem;
}
.forecast-bar {
  display: flex;
  height: 52px;
  width: min(980px, 100%);
  border: 1px solid rgba(0,51,102,.22);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.25);
}
.forecast-context,
.forecast-target {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 86px;
  font-weight: 800;
  letter-spacing: .02em;
  color: white;
  white-space: nowrap;
}
.forecast-context { background: #003366; }
.forecast-target { background: #BA2828; }
.forecast-arrow {
  position: absolute;
  top: 70px;
  transform: translateX(-50%);
  color: #003366;
  text-align: center;
  font-weight: 800;
}
.arrow-line {
  width: 2px;
  height: 26px;
  background: #FF0000;
  margin: 0 auto .2rem;
  position: relative;
}
.arrow-line:before {
  content: "";
  position: absolute;
  top: -6px;
  left: -5px;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-bottom: 8px solid #FF0000;
}
.arrow-label { font-size: .85rem; line-height: 1.18; }
.forecast-equation {
  display: flex;
  gap: .6rem;
  align-items: center;
  flex-wrap: wrap;
  margin: .4rem 0 .75rem;
  color: #003366;
}
.forecast-equation b {
  padding: .35rem .55rem;
  border-radius: 999px;
  background: rgba(91,186,186,.16);
}
.forecast-method-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: .7rem;
}
.forecast-method-grid div {
  border-left: 4px solid #5BBABA;
  padding: .55rem .7rem;
  background: rgba(255,255,255,.45);
}
.forecast-method-grid b,
.forecast-method-grid span { display: block; }
.forecast-method-grid span { font-size: 1.25rem; font-weight: 900; color: #003366; }
.forecast-ascii {
  margin: .9rem 0 0;
  padding: .75rem;
  border-radius: 10px;
  background: rgba(0,51,102,.06);
  color: #003366;
  overflow-x: auto;
  font-size: .85rem;
}
@media (max-width: 760px) {
  .forecast-method-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .forecast-context, .forecast-target { font-size: .8rem; min-width: 58px; }
}

</style>
''', unsafe_allow_html=True)


def section(anchor: str, title: str, copy: str) -> None:
    st.markdown(f'<span id="{anchor}"></span>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading"><h2>{title}</h2><p>{copy}</p></div>', unsafe_allow_html=True)


def end_section() -> None:
    return None


def fmt(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):,.{digits}f}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
    except Exception:
        pass
    return str(value)


def metric_card(label: str, value, help_text: str = "") -> None:
    display = fmt(value) if not isinstance(value, str) else value
    st.markdown(f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(display)}</div><div class="metric-help">{escape(help_text)}</div></div>', unsafe_allow_html=True)


def info_cards(items: list[tuple[str, object]]) -> None:
    html = ['<div class="profile-grid">']
    for label, value in items:
        display = fmt(value) if not isinstance(value, str) else value
        html.append(f'<div class="profile-card"><b>{escape(str(label))}</b><span>{escape(display)}</span></div>')
    html.append('</div>')
    st.markdown(''.join(html), unsafe_allow_html=True)


def pipeline_stage_visual(pipeline_df: pd.DataFrame, selected_stage: str) -> None:
    if pipeline_df.empty or selected_stage not in pipeline_df["Stage"].tolist():
        return
    row = pipeline_df[pipeline_df["Stage"] == selected_stage].iloc[0]
    context_label = str(row.get("Participant context", "Participants retained"))
    chips = [
        ("Participant change", row.get("Participants lost from previous step", "-")),
    ]
    chips_html = ''.join(
        f'<div class="pipeline-chip">{escape(label)}: {escape(fmt(value))}</div>'
        for label, value in chips
    )
    html = f"""
    <div class="pipeline-spotlight">
      <div class="big-number">{escape(fmt(row.get("Participants")))}</div>
      <div class="big-label">{escape(context_label)}</div>
      <h3>{escape(str(row.get("Stage", "-")))}</h3>
      <p><b>Strategy:</b> {escape(str(row.get("Thresholds / rule", "-")))}</p>
      <p style="margin-top:.45rem;"><b></b> {escape(str(row.get("Examples", "-")))}</p>
      <div class="pipeline-chips">{chips_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def participant_cards(df: pd.DataFrame, limit: int = 12) -> None:
    if df.empty:
        st.info("No participants match the current filters.")
        return
    cards = ['<div class="participant-grid">']
    for _, row in df.head(limit).iterrows():
        cards.append(
            f'<div class="participant-card"><b>Participant {escape(str(row.get("Participant ID", "-")))}</b>'
            f'<span>{escape(str(row.get("Split", "-")))}</span>'
            f'<div class="small-note">{escape(str(row.get("Study group", "-")))} · {escape(str(row.get("Clinical site", "-")))}<br>{escape(fmt(row.get("Valid forecast windows", "-")))} forecast windows</div></div>'
        )
    cards.append('</div>')
    st.markdown(''.join(cards), unsafe_allow_html=True)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None



def truthy_indicator(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).gt(0)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "present", "checked"})


def prepare_static_population(static_features: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if static_features.empty:
        return pd.DataFrame()
    out = static_features.copy()
    if "participant_id" in out.columns:
        out["participant_id"] = out["participant_id"].astype(str)
    rename_map = {}
    if "participants_study_group" in out.columns:
        rename_map["participants_study_group"] = "raw_study_group"
    if "participants_clinical_site" in out.columns and "clinical_site" not in out.columns:
        rename_map["participants_clinical_site"] = "clinical_site"
    out = out.rename(columns=rename_map)
    if not cohort.empty and "participant_id" in cohort.columns and "participant_id" in out.columns:
        merge_cols = [c for c in ["participant_id", "study_group", "stratum", "HbA1c", "BMI", "age"] if c in cohort.columns]
        coh = cohort[merge_cols].copy()
        coh["participant_id"] = coh["participant_id"].astype(str)
        out = out.merge(coh, on="participant_id", how="inner", suffixes=("", "_cohort"))
    if "study_group" not in out.columns:
        if "stratum" in out.columns:
            out["study_group"] = out["stratum"]
        elif "raw_study_group" in out.columns:
            out["study_group"] = out["raw_study_group"]
    return out


def add_ethnicity_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Ethnicity"] = "Unknown"
    if "demo_ethnicity_hispanic_or_latino" in out.columns:
        out.loc[truthy_indicator(out["demo_ethnicity_hispanic_or_latino"]), "Ethnicity"] = "Hispanic or Latino"
    if "demo_ethnicity_not_hispanic_or_latino" in out.columns:
        out.loc[truthy_indicator(out["demo_ethnicity_not_hispanic_or_latino"]), "Ethnicity"] = "Not Hispanic or Latino"
    if "demo_ethnicity_prefer_not_to_say" in out.columns:
        out.loc[truthy_indicator(out["demo_ethnicity_prefer_not_to_say"]), "Ethnicity"] = "Prefer not to say"
    return out


def add_tg_hdl_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"triglycerides_mgdl_baseline", "hdl_cholesterol_mgdl_baseline"}.issubset(out.columns):
        triglycerides = pd.to_numeric(out["triglycerides_mgdl_baseline"], errors="coerce")
        hdl = pd.to_numeric(out["hdl_cholesterol_mgdl_baseline"], errors="coerce").replace(0, np.nan)
        out["tg_hdl_ratio"] = triglycerides / hdl
    return out


def attach_original_condition_sources(demo_df: pd.DataFrame, participants_tsv: pd.DataFrame, condition_groups: pd.DataFrame) -> pd.DataFrame:
    out = demo_df.copy()
    if "participant_id" in out.columns:
        out["participant_id"] = out["participant_id"].astype(str)
    if not participants_tsv.empty and "participant_id" in participants_tsv.columns and "participant_id" in out.columns:
        keep = [c for c in ["participant_id", "participant_tsv_study_group", "clinical_site", "recommended_split"] if c in participants_tsv.columns]
        raw = participants_tsv[keep].copy()
        raw["participant_id"] = raw["participant_id"].astype(str)
        out = out.merge(raw, on="participant_id", how="left", suffixes=("", "_participants_tsv"))
    if not condition_groups.empty and "participant_id" in condition_groups.columns and "participant_id" in out.columns:
        cond = condition_groups.copy()
        cond["participant_id"] = cond["participant_id"].astype(str)
        out = out.merge(cond, on="participant_id", how="left")
    out["condition_file_self_report"] = out.get("condition_file_self_report", pd.Series(index=out.index, dtype=object)).fillna("No diabetes condition recorded")
    return out

def participant_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["participant_id", "Participant_ID", "subject_id", "id"])


def duration_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["dur_h", "duration_h", "duration_hours", "segment_duration_h", "duration"])


def humanize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out.rename(columns={c: COLUMN_LABELS.get(c, c.replace("_", " ").title()) for c in out.columns})


def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "split" not in df.columns or "participant_id" not in df.columns:
        return pd.DataFrame(columns=["Split", "Participants"])
    tmp = df.copy()
    tmp["Split"] = tmp["split"].map(pretty_split)
    return tmp.groupby("Split")["participant_id"].nunique().reset_index(name="Participants")


def enrich_participant_table(cohort: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty or "participant_id" not in cohort.columns:
        return pd.DataFrame()
    cols = [c for c in ["participant_id", "study_group", "stratum", "clinical_site", "age", "BMI", "HbA1c", "n_segments", "total_clean_dur_h", "longest_seg_h", "personalization_eligible"] if c in cohort.columns]
    table = cohort[cols].copy()
    if "study_group" not in table.columns and "stratum" in table.columns:
        table["study_group"] = table["stratum"]
    if not split_df.empty and {"participant_id", "split"}.issubset(split_df.columns):
        table = table.merge(split_df[["participant_id", "split"]], on="participant_id", how="left")
    else:
        table["split"] = "unknown"
    table["Split"] = table["split"].map(pretty_split)
    table["Study group"] = table.get("study_group", pd.Series(index=table.index, dtype=object)).map(pretty_group)
    table["Clinical site"] = table.get("clinical_site", pd.Series(index=table.index, dtype=object)).fillna("Unknown")
    table["Participant ID"] = table["participant_id"].astype(str)
    return table.rename(columns={
        "age": "Age [years]",
        "BMI": "BMI [kg/m2]",
        "HbA1c": "HbA1c [%]",
        "n_segments": "Clean segments",
        "total_clean_dur_h": "Clean duration [h]",
        "longest_seg_h": "Longest clean segment [h]",
        "personalization_eligible": "Personalization eligible",
    })


def selected_row(df: pd.DataFrame, participant_id: str) -> pd.Series:
    if df.empty or "participant_id" not in df.columns:
        return pd.Series(dtype=object)
    rows = df[df["participant_id"].astype(str) == str(participant_id)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)



def feature_group(static_row: pd.DataFrame, keywords: list[str], query: str = "") -> pd.DataFrame:
    if static_row.empty:
        return pd.DataFrame(columns=["Feature", "Value"])
    cols = [c for c in static_row.columns if any(k in c.lower() for k in keywords)]
    return plot_static_feature_table(static_row[cols], query=query) if cols else pd.DataFrame(columns=["Feature", "Value"])


def build_pipeline_summary() -> pd.DataFrame:
    meta = cohort_selection_meta if isinstance(cohort_selection_meta, dict) else {}
    counts = meta.get("counts", {}) or {}
    config = meta.get("config", {}) or {}

    raw_n = counts.get("raw")
    after_duration = counts.get("after_duration_floor")
    after_trim = counts.get("after_trim")
    after_segmentation = counts.get("after_segmentation")
    total_segments = counts.get("total_segments", len(segments) if not segments.empty else None)

    min_duration_h = config.get("min_duration_h", 108)
    context_h = config.get("context_h", 48)
    target_h = config.get("target_h", 1)
    gap = config.get("gap_thresholds_min", {}) or {}
    trim_bins = config.get("trim_window_bins", 12)
    trim_threshold = config.get("trim_miss_threshold", 0.20)
    input_parquet = meta.get("input_parquet", "final_multimodal_dataset*.parquet")

    rows = [
        {
            "Stage": "Raw enriched data",
            "Participants": raw_n or "-",
            "Examples": f"Input parquet: {Path(str(input_parquet)).name}",
            "Thresholds / rule": "Before cohort selection; enriched CGM, HR, RR, activity, and static clinical features are available.",
        },
        {
            "Stage": "Duration floor",
            "Participants": after_duration or "-",
            "Examples": f"Retains participants with at least {float(min_duration_h):.0f} hours of raw coverage.",
            "Thresholds / rule": f">= {float(min_duration_h):.0f} h of raw coverage, enough to support a full streamed history per participant.",
        },
        {
            "Stage": "Boundary trimming",
            "Participants": after_trim or "-",
            "Examples": "Trims leading/trailing bins where missingness is high before clean segmentation.",
            "Thresholds / rule": f"{trim_bins} bins rolling window ({trim_bins * 5 / 60:.1f} h), missingness threshold {float(trim_threshold) * 100:.0f}%.",
        },
        {
            "Stage": "Gap segmentation",
            "Participants": after_segmentation or "-",
            "Examples": f"{fmt(total_segments)} clean segments retained across participants, each streamed continuously.",
            "Thresholds / rule": f"Split at long gaps: CGM > {gap.get('cgm', 30)} min; HR/RR/activity > {gap.get('hr', 60)} min; keep segments >= {context_h + target_h} h of streamed coverage.",
        },
        {
            "Stage": "Imputation",
            "Participants": after_segmentation or "-",
            "Examples": "CGM, HR, and RR are linearly interpolated inside retained clean segments; activity is zero-filled.",
            "Thresholds / rule": "Imputation is constrained by the same modality gap thresholds; segments with remaining CGM NaNs are not streamable.",
        },
    ]
    out = pd.DataFrame(rows)
    out["Participants lost from previous step"] = pd.Series(["-"] * len(out), dtype=object)
    prev = None
    for idx, value in out["Participants"].items():
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric) and prev is not None:
            out.loc[idx, "Participants lost from previous step"] = int(prev - numeric)
        if pd.notna(numeric):
            prev = numeric

    out["Participant context"] = "Participants retained"
    return out


availability = check_file_availability()
cohort = load_cohort()
segments = load_segments()
stream_summary = load_stream_summary()
# Single source of truth for participant split assignment across the whole app,
# the same split forecast anchor computation uses (experiment_c_split_adapt6h_seed42).
split_participants = load_canonical_stream_split()
static_features = load_static_features()
original_participants = load_original_participants()
original_condition_groups = load_original_condition_groups()
metadata = multimodal_metadata()
cohort_selection_meta = load_cohort_selection_metadata()
t2d_subtype_factors = load_t2d_subtype_clinical_factors()
participants_table = enrich_participant_table(cohort, split_participants)

st.markdown('''
<div class="hero">
  <div class="kicker">Enriched multimodal cohort</div>
  <h1>AI-READI Dataset Dashboard</h1>
</div>
''', unsafe_allow_html=True)

found_count = int(availability["Found/Missing"].astype(str).str.contains("Found", na=False).sum()) if "Found/Missing" in availability.columns else 0
summary_cols = st.columns(3)
with summary_cols[0]:
    metric_card(
        "Participants",
        cohort["participant_id"].nunique() if "participant_id" in cohort.columns else None,
        "Enriched cohort",
    )
with summary_cols[1]:
    metric_card("Enriched rows", metadata.get("rows"), "Parquet metadata")
with summary_cols[2]:
    stream_cols = st.columns(2)
    with stream_cols[0]:
        metric_card("Streams", stream_summary.get("streams"), "Participant + segment")
    with stream_cols[1]:
        metric_card(
            "Forecast anchors",
            stream_summary.get("forecast_anchors"),
            "15-min stride · all splits",
        )

st.markdown('''
<div class="toc">
  <a href="#population-statistics">Population statistics</a>
  <a href="#cohort-participant-explorer">Participant explorer</a>  
  <a href="#preprocessing-pipeline">Preprocessing pipeline</a>
  <a href="#split-validation">Split validation</a>
  <a href="#phase-timeline">Streaming participant timeline</a>
  <a href="#time-series">Participant time series</a>
  <a href="#static-profile">Static feature profile</a>
  <a href="#data-quality">Data quality</a>
</div>
''', unsafe_allow_html=True)

section(
    "population-statistics",
    "Population Statistics Dashboard",
    "Cohort-level distributions and proportions for paper figures, presentations, and split sanity checks."
)
if cohort.empty:
    st.warning("cohort.csv could not be loaded, so population statistics are unavailable.")
else:
    group_col = "study_group" if "study_group" in cohort.columns else "stratum" if "stratum" in cohort.columns else None
    static_pop = prepare_static_population(static_features, cohort)
    static_group_col = "stratum" if "stratum" in static_pop.columns else "study_group" if "study_group" in static_pop.columns else None
    med_cols = {
        "med_metformin": "Metformin",
        "med_insulin": "Insulin",
        "med_glp1_or_gip_glp1": "GLP-1 / GIP-GLP-1",
        "med_sglt2": "SGLT2 inhibitor",
        "med_sulfonylurea": "Sulfonylurea",
        "med_thiazolidinedione": "TZD",
    }

    static_pop_ratio = add_tg_hdl_ratio(static_pop)

    tab_prop, tab_dist, tab_duration, tab_meds, tab_demo, tab_cgm, tab_cardio, tab_t2d = st.tabs([
        "Cohort proportions", "Clinical distributions",
        "Duration and coverage", "Medication patterns",
        "Demographic balance", "CGM-derived metrics",
        "Blood pressure & cardiometabolic", "T2D heterogeneity",
    ])
    with tab_dist:
        c1, c2 = st.columns(2)
        with c1:
            if "age" in cohort.columns:
                st.plotly_chart(plot_population_violin(cohort, "age", group_col, "Age distribution by study group"), width="stretch", key="pop_age_violin")
            if "BMI" in cohort.columns:
                st.plotly_chart(plot_kde_curves(cohort, "BMI", group_col, "BMI KDE curves"), width="stretch", key="pop_bmi_kde")
        with c2:
            if "HbA1c" in cohort.columns:
                st.plotly_chart(
                    plot_kde_curves(cohort, "HbA1c", group_col, "HbA1c density curves by study group"),
                    width="stretch",
                    key="pop_hba1c_kde",
                )
                st.caption("HbA1c is shown as density curves instead of overlapping histograms so the group-level patterns remain readable.")
        if static_pop_ratio.empty:
            st.info("Static features file not available, clinical factor panels require participant_static_features.")
        else:
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            st.caption("C-peptide, TG/HDL ratio, and waist-to-hip ratio are the same clinical factors used for the frozen T2D subtype clustering in the interpretability chapter.")
            c1, c2, c3 = st.columns(3)
            with c1:
                if "c_peptide_ngml_baseline" in static_pop_ratio.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop_ratio, "c_peptide_ngml_baseline", "study_group",
                            "C-peptide distribution by study group",
                        ),
                        width="stretch", key="pop_cpeptide_violin",
                    )
            with c2:
                if "tg_hdl_ratio" in static_pop_ratio.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop_ratio, "tg_hdl_ratio", "study_group",
                            "TG/HDL ratio distribution by study group",
                        ),
                        width="stretch", key="pop_tghdl_violin",
                    )
            with c3:
                if "waist_to_hip_ratio_baseline" in static_pop_ratio.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop_ratio, "waist_to_hip_ratio_baseline", "study_group",
                            "Waist-to-hip ratio distribution by study group",
                        ),
                        width="stretch", key="pop_waisthip_violin",
                    )
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if "clinical_systolic_bp_mmhg_baseline" in static_pop_ratio.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop_ratio, "clinical_systolic_bp_mmhg_baseline", "study_group",
                            "Systolic BP by study group",
                        ),
                        width="stretch", key="pop_dist_sbp_violin",
                    )
            with c2:
                if "clinical_diastolic_bp_mmhg_baseline" in static_pop_ratio.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop_ratio, "clinical_diastolic_bp_mmhg_baseline", "study_group",
                            "Diastolic BP by study group",
                        ),
                        width="stretch", key="pop_dist_dbp_violin",
                    )
    with tab_prop:
        c1, c2 = st.columns(2)
        with c1:
            if "clinical_site" in cohort.columns:
                st.plotly_chart(plot_proportion_bar(cohort, "clinical_site", "Clinical site proportions"), width="stretch", key="pop_site_prop")
        with c2:
            if group_col:
                st.plotly_chart(plot_proportion_bar(cohort, group_col, "Study group proportions"), width="stretch", key="pop_group_prop")
    with tab_duration:
        c1, c2 = st.columns(2)
        with c1:
            duration_col_name = "duration_h" if "duration_h" in cohort.columns else "duration_h_trimmed" if "duration_h_trimmed" in cohort.columns else None
            if duration_col_name:
                st.plotly_chart(plot_kde_curves(cohort, duration_col_name, group_col, "Participant duration KDE"), width="stretch", key="pop_duration_kde")
        with c2:
            if duration_col_name:
                st.plotly_chart(plot_stacked_histogram(cohort, duration_col_name, group_col, "Participant duration stacked histogram"), width="stretch", key="pop_duration_hist")
    with tab_meds:
        if static_pop.empty or not static_group_col:
            st.info("Static features file not available, medication plots require participant_static_features.")
        else:
            present_med_cols = {col: label for col, label in med_cols.items() if col in static_pop.columns}
            if not present_med_cols:
                st.warning("No diabetes medication flag columns were found in participant_static_features.")
            else:
                st.caption("Drug classes are non-exclusive: a participant can contribute to more than one medication class.")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_medication_prevalence_by_stratum(
                            static_pop, present_med_cols, static_group_col,
                            "Drug class prevalence by stratum",
                        ),
                        width="stretch", key="pop_med_prevalence",
                    )
                with c2:
                    st.plotly_chart(
                        plot_coprescription_heatmap(
                            static_pop, present_med_cols,
                            "Co-prescription heatmap",
                        ),
                        width="stretch", key="pop_med_coprescription",
                    )
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_medication_burden(
                            static_pop, present_med_cols, static_group_col,
                            "Medication burden by stratum",
                        ),
                        width="stretch", key="pop_med_burden",
                    )
                with c2:
                    hba1c_col = "hba1c_percent_baseline" if "hba1c_percent_baseline" in static_pop.columns else "HbA1c" if "HbA1c" in static_pop.columns else None
                    if hba1c_col:
                        st.plotly_chart(
                            plot_hba1c_vs_med_count(
                                static_pop, present_med_cols, hba1c_col, static_group_col,
                                "HbA1c vs medication count",
                            ),
                            width="stretch", key="pop_hba1c_med_count",
                        )
                    else:
                        st.info("HbA1c column not found for treatment-intensity scatter plot.")

    with tab_demo:
        if static_pop.empty or not static_group_col:
            st.info("Static features file not available, demographic balance plots require participant_static_features.")
        else:
            demo_pop = add_ethnicity_label(static_pop)
            demo_pop = attach_original_condition_sources(demo_pop, original_participants, original_condition_groups)
            st.caption(
                "Demographic distributions are normalized within each final analysis stratum. "
                "The condition heatmap uses participant IDs merged from participants.tsv and self-reported diabetes entries from clinical_data/condition_occurrence.csv."
            )
            c1, c2 = st.columns(2)
            with c1:
                if "demo_sex_at_birth" in demo_pop.columns:
                    st.plotly_chart(
                        plot_stacked_proportion(
                            demo_pop, static_group_col, "demo_sex_at_birth",
                            "Sex distribution by stratum", "Study group", "Sex at birth",
                            horizontal=True,
                        ),
                        width="stretch", key="pop_sex_by_stratum",
                    )
                else:
                    st.info("Sex-at-birth column not found.")
            with c2:
                race_col = "demo_race_primary" if "demo_race_primary" in demo_pop.columns else "demo_race_best_represents" if "demo_race_best_represents" in demo_pop.columns else None
                if race_col:
                    st.plotly_chart(
                        plot_stacked_proportion(
                            demo_pop, static_group_col, race_col,
                            "Race distribution by stratum", "Study group", "Race",
                            horizontal=True,
                        ),
                        width="stretch", key="pop_race_by_stratum",
                    )
                else:
                    st.info("Race summary column not found.")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_stacked_proportion(
                        demo_pop, static_group_col, "Ethnicity",
                        "Ethnicity distribution by stratum", "Study group", "Ethnicity",
                        horizontal=True,
                    ),
                    width="stretch", key="pop_ethnicity_by_stratum",
                )
            with c2:
                if "condition_file_self_report" in demo_pop.columns:
                    st.plotly_chart(
                        plot_crosstab_heatmap(
                            demo_pop, static_group_col, "condition_file_self_report",
                            "Original conditions file vs final stratum",
                        ),
                        width="stretch", key="pop_condition_report_heatmap",
                    )
                else:
                    st.info("Original condition file could not be merged for cross-tab heatmap.")

    with tab_cgm:
        cgm_metrics = load_cgm_participant_metrics()
        if cgm_metrics.empty:
            st.info("CGM-derived population metrics could not be computed from the enriched multimodal parquet.")
        else:
            cgm_pop = cgm_metrics.copy()
            if "participant_id" in cgm_pop.columns:
                cgm_pop["participant_id"] = cgm_pop["participant_id"].astype(str)
            merge_cols = [c for c in ["participant_id", "study_group", "stratum"] if c in cohort.columns]
            if merge_cols and "participant_id" in cgm_pop.columns:
                coh = cohort[merge_cols].copy()
                coh["participant_id"] = coh["participant_id"].astype(str)
                cgm_pop = cgm_pop.merge(coh, on="participant_id", how="left")
            cgm_group_col = "stratum" if "stratum" in cgm_pop.columns else "study_group" if "study_group" in cgm_pop.columns else None
            st.caption("Computed from CGM glucose values in the enriched multimodal parquet: mean glucose, time in range 70-180 mg/dL, and coefficient of variation.")
            c1, c2, c3 = st.columns(3)
            with c1:
                fig = plot_population_violin(cgm_pop, "mean_glucose_mgdl", cgm_group_col, "Mean glucose by stratum")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch", key="pop_cgm_mean_violin")
            with c2:
                fig = plot_population_violin(cgm_pop, "tir_70_180_pct", cgm_group_col, "Time in range by stratum")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch", key="pop_cgm_tir_violin")
            with c3:
                fig = plot_population_violin(cgm_pop, "cv_pct", cgm_group_col, "Glucose variability by stratum")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch", key="pop_cgm_cv_violin")

    with tab_cardio:
        if static_pop.empty:
            st.info("Static features file not available, cardiometabolic plots require participant_static_features.")
        else:
            # ── Row 1: Blood pressure distributions ───────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                if "clinical_systolic_bp_mmhg_baseline" in static_pop.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop, "clinical_systolic_bp_mmhg_baseline", "study_group",
                            "Systolic BP distribution by study group",
                        ),
                        width="stretch", key="pop_sbp_violin",
                    )
            with c2:
                if "clinical_diastolic_bp_mmhg_baseline" in static_pop.columns:
                    st.plotly_chart(
                        plot_population_violin(
                            static_pop, "clinical_diastolic_bp_mmhg_baseline", "study_group",
                            "Diastolic BP distribution by study group",
                        ),
                        width="stretch", key="pop_dbp_violin",
                    )
            # ── Row 2: HbA1c vs Systolic BP scatter + correlation matrix ──────
            c1, c2 = st.columns(2)
            with c1:
                if all(c in static_pop.columns for c in ["hba1c_percent_baseline", "clinical_systolic_bp_mmhg_baseline"]):
                    st.plotly_chart(
                        plot_clinical_scatter(
                            static_pop,
                            "hba1c_percent_baseline", "clinical_systolic_bp_mmhg_baseline",
                            "study_group", "HbA1c vs Systolic BP",
                        ),
                        width="stretch", key="pop_hba1c_sbp_scatter",
                    )
            with c2:
                _corr_cols = [
                    "hba1c_percent_baseline",
                    "bmi_baseline",
                    "ldl_cholesterol_mgdl_baseline",
                    "hdl_cholesterol_mgdl_baseline",
                    "triglycerides_mgdl_baseline",
                    "clinical_systolic_bp_mmhg_baseline",
                    "clinical_diastolic_bp_mmhg_baseline",
                ]
                _corr_labels = {
                    "hba1c_percent_baseline": "HbA1c",
                    "bmi_baseline": "BMI",
                    "ldl_cholesterol_mgdl_baseline": "LDL",
                    "hdl_cholesterol_mgdl_baseline": "HDL",
                    "triglycerides_mgdl_baseline": "Triglycerides",
                    "clinical_systolic_bp_mmhg_baseline": "Systolic BP",
                    "clinical_diastolic_bp_mmhg_baseline": "Diastolic BP",
                }
                st.plotly_chart(
                    plot_correlation_matrix(
                        static_pop, _corr_cols, _corr_labels,
                        "Cardiometabolic correlation matrix",
                    ),
                    width="stretch", key="pop_cardio_corr",
                )

    with tab_t2d:
        if t2d_subtype_factors.empty:
            st.info("Frozen T2D subtype clustering data could not be loaded.")
        else:
            cluster_counts = t2d_subtype_factors.drop_duplicates("participant_id").groupby("cluster")["participant_id"].nunique()
            st.caption(
                "T2D oral non-insulin participants only, split into the 3 frozen subtype clusters from the "
                "interpretability chapter (Ward-selected k=3, silhouette and bootstrap ARI validated)."
            )
            info_cards([
                (f"{cluster} (n={cluster_counts.get(cluster, '-')})", T2D_SUBTYPE_CLUSTER_INTERPRETATION[cluster])
                for cluster in T2D_SUBTYPE_CLUSTER_ORDER
            ])
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            factor_items = list(T2D_SUBTYPE_FACTOR_COLUMNS.items())
            for row_start in range(0, len(factor_items), 3):
                cols = st.columns(3)
                for col, (factor_key, factor_label) in zip(cols, factor_items[row_start:row_start + 3]):
                    with col:
                        sub = t2d_subtype_factors.loc[
                            t2d_subtype_factors["factor"] == factor_key, ["cluster", "value"]
                        ].rename(columns={"value": factor_key})
                        st.plotly_chart(
                            plot_cluster_box_strip(
                                sub, factor_key, "cluster",
                                T2D_SUBTYPE_CLUSTER_ORDER, T2D_SUBTYPE_CLUSTER_COLORS,
                                f"{factor_label} by subtype cluster",
                            ),
                            width="stretch", key=f"t2d_subtype_{factor_key}",
                        )
            st.caption(
                "Clusters were frozen before post hoc interpretation. Fasting status is unconfirmed for "
                "C-peptide and triglycerides, so TG/HDL and C-peptide profiles carry the same non-fasting caveat as the source analysis."
            )
end_section()


section(
    "cohort-participant-explorer",
    "Participant Explorer",
    "Filter participants by split and study group, then select one participant for the profile, timeline, time-series, and static-feature views below."
)

selected_participant = None
selected_split_label = "All"

if participants_table.empty:
    st.warning("No participant table could be built from cohort.csv.")
else:
    with st.expander("Filtered participant table", expanded=False):
        display_cols = [c for c in [
            "Participant ID", "Split", "Study group", "Clinical site",
            "Age [years]", "BMI [kg/m2]", "HbA1c [%]",
            "Clean segments", "Clean duration [h]", "Personalization eligible"
        ] if c in participants_table.columns]

        st.dataframe(
            participants_table[display_cols],
            width="stretch",
            hide_index=True
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.1, 1.4, 2.4])

    with f1:
        split_default = 2 if "Validation" in participants_table["Split"].unique() else 0
        split_filter = st.selectbox(
            "Split filter",
            ["All", "Train", "Validation", "Test"],
            index=split_default
        )

    with f2:
        group_values = ["All"] + sorted(participants_table["Study group"].dropna().unique().tolist())
        group_filter = st.selectbox("Study group filter", group_values)

    filtered = participants_table.copy()

    if split_filter != "All":
        filtered = filtered[filtered["Split"] == split_filter]

    if group_filter != "All":
        filtered = filtered[filtered["Study group"] == group_filter]

    ids = filtered["Participant ID"].astype(str).tolist()

    participant_labels = {
        str(row["Participant ID"]): f"{row['Participant ID']} · {row['Split']} · {row['Study group']} · {row['Clinical site']}"
        for _, row in filtered.iterrows()
    }

    with f3:
        selected_participant = st.selectbox(
            "Participant",
            ids,
            index=0 if ids else None,
            placeholder="No participant after filters",
            format_func=lambda pid: participant_labels.get(str(pid), str(pid)),
        )

    if selected_participant:
        prow = participants_table[
            participants_table["Participant ID"] == str(selected_participant)
        ].head(1)

        if not prow.empty:
            r = prow.iloc[0]
            selected_split_label = r.get("Split", "All")

            streamed_duration = r.get("Clean duration [h]")
            info_cards([
                ("Split", selected_split_label),
                ("Study group", r.get("Study group")),
                ("Clinical site", r.get("Clinical site")),
                ("Age", r.get("Age [years]")),
                ("BMI", r.get("BMI [kg/m2]")),
                ("HbA1c", r.get("HbA1c [%]")),
                ("Streamed duration", f"{streamed_duration:.1f} h" if pd.notna(streamed_duration) else "-"),
            ])

end_section()



section(
    "preprocessing-pipeline",
    "Preprocessing Steps and Cohort Selection Impact",
    "Stage-level view of how raw multimodal data becomes clean, continuous streams ready for the stateful SSM-CGM model."
)
pipeline_df = build_pipeline_summary()
if cohort_selection_meta:
    counts = cohort_selection_meta.get("counts", {})
    cfg = cohort_selection_meta.get("config", {})
    st.caption(
        f"Loaded real cohort-selection metadata from cohort_selection.py outputs: "
        f"raw {fmt(counts.get('raw'))} participants -> retained {fmt(counts.get('after_segmentation'))} participants, "
        f"{fmt(counts.get('total_segments'))} segments across continuous streams."
    )
else:
    st.warning("cohort_selection_metadata.json was not found, so this section falls back to available CSV counts.")
selected_stage = st.radio(
    "Choose preprocessing stage",
    pipeline_df["Stage"].tolist(),
    horizontal=True,
    label_visibility="collapsed",
)
pipeline_stage_visual(pipeline_df, selected_stage)

end_section()

section(
    "split-validation",
    "Train / Validation / Test Split",
    "Participant repartition and stratification for the single split manifest used everywhere in this dashboard.",
)
if split_participants.empty:
    st.warning(
        "Placeholder: the canonical split_participants.csv for the epoch-5 streaming checkpoint "
        "(experiment_c_split_adapt6h_seed42) could not be loaded, so no participant counts are shown here."
    )
else:
    st.caption(
        f"Sourced from experiment_c_split_adapt6h_seed42/split_participants.csv, the split named in the "
        f"checkpoint's config_resolved.yaml (split.existing_split_path), matching the epoch-5 checkpoint "
        f"with val pinball loss {CANONICAL_STREAM_CHECKPOINT_VAL_PINBALL_MGDL:.6f} mg/dL. "
        f"This is the same manifest used for the participant filter, split labels, and forecast anchors below."
    )
    c1, c2 = st.columns([1, 1], vertical_alignment="top")
    with c1:
        st.plotly_chart(plot_split_distribution(split_participants, "Participant repartition"), width="stretch", key="split_distribution_main")
    with c2:
        split_group_col = "stratum" if "stratum" in split_participants.columns else "study_group" if "study_group" in split_participants.columns else None
        if split_group_col:
            st.plotly_chart(plot_stacked(split_participants, "split", split_group_col, "Split by study group"), width="stretch", key="split_group_stack")
end_section()

section(
    "phase-timeline",
    "Streaming participant timeline",
    "Continuous CGM stream with real segment resets and a one-time warm-up period at the start of the stream.",
)
warmup_hours = st.radio(
    "Warm-up length",
    PERSONALIZATION_WARMUP_HOURS,
    index=len(PERSONALIZATION_WARMUP_HOURS) - 1,
    horizontal=True,
    format_func=lambda hours: f"{hours} h",
)
if selected_participant:
    timeline_df = load_participant_timeseries(selected_participant)
    timeline_time_col = detect_timestamp_column(timeline_df)
    if timeline_df.empty:
        st.warning("No time-series rows could be loaded for this participant.")
    elif not timeline_time_col:
        st.warning("No timestamp column is available for the streaming timeline.")
    else:
        try:
            participant_segments = segment_boundaries(
                timeline_df,
                "participant_id",
                timeline_time_col,
            )
        except ValueError as exc:
            participant_segments = pd.DataFrame()
            st.warning(f"Segment reset markers are unavailable: {exc}")

        forecast_anchors = load_forecast_anchors()
        participant_anchors = forecast_anchors[
            forecast_anchors["participant_id"].astype(str).eq(str(selected_participant))
        ].copy() if not forecast_anchors.empty else pd.DataFrame()

        st.caption(
            "The continuous signal carries model state forward between real segment resets. "
            "Warm-up applies once, at the very start of the participant's stream; a later segment reset "
            "does not restart it. The shaded band marks the warm-up period and the solid line marks where "
            "scored evaluation begins."
        )
        if participant_anchors.empty:
            # split_participants is the single source of truth used everywhere in this app
            # (Participant Explorer filter, split labels, this timeline, and forecast anchors),
            # so canonical_split_label and selected_split_label should always agree for any
            # participant present in the split manifest. The mismatch note below is a safeguard
            # for a still-misconfigured state (for example, a participant missing from the
            # split manifest entirely), not expected behavior for ordinary participants.
            canonical_row = split_participants[
                split_participants["participant_id"].astype(str) == str(selected_participant)
            ] if not split_participants.empty else pd.DataFrame()
            canonical_split_label = pretty_split(canonical_row.iloc[0]["split"]) if not canonical_row.empty else None

            if canonical_split_label == "Train":
                empty_anchor_message = "Train participants are not personalized, so no forecast anchors exist."
                if str(selected_split_label).strip().lower() != "train":
                    empty_anchor_message += (
                        f" Note: this participant is Train in the canonical model split "
                        f"(experiment_c_split_adapt6h_seed42) that the checkpoint and forecast anchors actually "
                        f"use, even though the filter above shows {selected_split_label}. This indicates a "
                        f"split-source misconfiguration, since both should read from the same manifest."
                    )
            else:
                total_stream_hours = (
                    (participant_segments["end"] - participant_segments["start"]).dt.total_seconds().sum() / 3600
                    if not participant_segments.empty else 0.0
                )
                if total_stream_hours <= float(warmup_hours):
                    shorter_options = [h for h in PERSONALIZATION_WARMUP_HOURS if h < total_stream_hours]
                    suggestion = (
                        f" Try a shorter warm-up length, such as {max(shorter_options)} h, to see scored anchors."
                        if shorter_options else ""
                    )
                    empty_anchor_message = (
                        f"This participant's stream is only {total_stream_hours:.1f} h long, shorter than the "
                        f"{float(warmup_hours):g} h warm-up window selected, so no scored anchors exist.{suggestion}"
                    )
                else:
                    empty_anchor_message = (
                        "No forecast anchors were found for this validation/test participant in the model's "
                        "prediction output."
                    )
            st.info(empty_anchor_message)

        st.plotly_chart(
            plot_participant_timeseries(
                timeline_df,
                timeline_time_col,
                [STREAM_TIMELINE_SIGNAL],
                segment_boundaries_df=participant_segments,
                anchor_df=participant_anchors,
                warmup_hours=warmup_hours,
                segment_gap_minutes=CGM_SEGMENT_GAP_MINUTES,
                title=f"Participant {selected_participant} continuous stream",
            ),
            width="stretch",
            key=f"stream_timeline_{selected_participant}_{warmup_hours}",
        )
else:
    st.info("Select a participant in the cohort explorer first.")
end_section()


section("time-series", "Participant Time-Series Explorer", "Large stacked plots for CGM glucose, heart rate, respiratory rate, and activity.")
ts_df = pd.DataFrame()
time_col = None
if selected_participant:
    ts_df = load_participant_timeseries(selected_participant)
    time_col = detect_timestamp_column(ts_df)
    if ts_df.empty:
        st.warning("No time-series rows could be loaded for this participant.")
    else:
        signal_options = list(SIGNAL_COLUMNS.values())
        labels_by_col = {col: label for label, col in SIGNAL_COLUMNS.items()}
        available = [c for c in signal_options if c in ts_df.columns]
        chosen = st.multiselect("Signals", signal_options, default=available, format_func=lambda c: labels_by_col.get(c, c))
        missing = [labels_by_col[c] for c in signal_options if c not in ts_df.columns]
        if missing:
            st.warning("Missing expected signals: " + ", ".join(missing))
        stats = participant_time_stats(ts_df, time_col)
        info_cards([
            ("Rows", stats.get("rows")),
            ("Time coverage", f"{stats.get('duration_h', np.nan):.1f} h" if pd.notna(stats.get("duration_h")) else "-"),
            ("Mean CGM glucose", f"{stats.get('mean_glucose', np.nan):.1f} mg/dL" if pd.notna(stats.get("mean_glucose")) else "-"),
            ("Glucose min / max", f"{stats.get('glucose_min', np.nan):.0f} / {stats.get('glucose_max', np.nan):.0f}" if pd.notna(stats.get("glucose_min")) else "-"),
            ("Mean heart rate", f"{stats.get('mean_hr', np.nan):.1f} bpm" if pd.notna(stats.get("mean_hr")) else "-"),
            ("Mean respiratory rate", f"{stats.get('mean_rr', np.nan):.1f} breaths/min" if pd.notna(stats.get("mean_rr")) else "-"),
        ])
        st.plotly_chart(
            plot_participant_timeseries(ts_df, time_col, chosen),
            width="stretch",
            key=f"timeseries_{selected_participant}_{'-'.join(chosen)}",
        )
else:
    st.info("Select a participant in the cohort explorer first.")
end_section()

section("static-profile", "Static Feature Profile", "Grouped enriched participant features displayed as a medical profile rather than a raw CSV view.")
if selected_participant:
    pcol = participant_col(static_features)
    static_row = static_features[static_features[pcol].astype(str) == str(selected_participant)] if pcol and not static_features.empty else pd.DataFrame()
    if static_row.empty:
        st.warning("No static feature row found for this participant.")
    else:
        sr = static_row.iloc[0]
        info_cards([
            ("Clinical site", sr.get("participants_clinical_site")),
            ("Study group", pretty_group(sr.get("participants_study_group"))),
            ("Age", sr.get("participants_age")),
            ("BMI", sr.get("bmi_baseline")),
            ("HbA1c", sr.get("hba1c_percent_baseline")),
            ("Insulin flag", "Yes" if sr.get("med_insulin") == 1 else "No"),
            ("Any diabetes drug", "Yes" if sr.get("med_any_diabetes_drug") == 1 else "No"),
        ])
        query = st.text_input("Search enriched static features", placeholder="Search feature or value")
        tabs = st.tabs(["Demographics", "Clinical", "Medications", "Cohort metadata", "Clinical measurements"])
        with tabs[0]:
            st.dataframe(feature_group(static_row, ["demo", "participants_"], query), width="stretch", hide_index=True)
        with tabs[1]:
            st.dataframe(feature_group(static_row, ["bmi", "hba1c", "clinical", "cholesterol", "glucose", "triglycerides", "peptide", "waist"], query), width="stretch", hide_index=True)
        with tabs[2]:
            med_df = feature_group(static_row, ["med_"], query)

            if not med_df.empty:
                # Keep only active medications / medication flags equal to 1
                med_df_active = med_df.copy()
                med_df_active["Value_numeric"] = pd.to_numeric(med_df_active["Value"], errors="coerce")
                med_df_active = med_df_active[med_df_active["Value_numeric"] == 1]

                med_df_active = med_df_active.drop(columns=["Value_numeric"])

                if med_df_active.empty:
                    st.info("No active medication flags found for this participant.")
                else:
                    med_df_active = med_df_active.rename(columns={"Feature": "Medication"})
                    med_df_active = med_df_active[["Medication"]]
                    st.dataframe(med_df_active, width="stretch", hide_index=True)
            else:
                st.info("No medication features found for this participant.")

            meds_long = load_medications_long(selected_participant)

            if not meds_long.empty:
                st.markdown("**Medication records**")

                meds_display = meds_long.copy()

                # Remove repetitive / low-value columns
                cols_to_drop = [
                    "redcap_repeat_instrument",
                    "Redcap Repeat Instrument",
                    "redcap_repeat_instance",
                    "Redcap Repeat Instance",
                ]
                meds_display = meds_display.drop(
                    columns=[c for c in cols_to_drop if c in meds_display.columns],
                    errors="ignore"
                )

                # Rename columns to readable labels
                meds_display = meds_display.rename(columns={
                    "studyid": "Participant ID",
                    "cmname": "Medication",
                    "cmroute": "Route code",
                    "cmrouteot": "Route",
                    "cmdos": "Dose",
                    "cmdosu": "Dose unit",
                    "cmdosfrq": "Frequency",
                    "rxnorm_code": "RxNorm code",
                    "rxnorm_term": "RxNorm term",
                })

                # Keep most useful columns first if they exist
                preferred_cols = [
                    "Participant ID",
                    "Medication",
                    "Dose",
                    "Dose unit",
                    "Frequency",
                    "Route",
                    "Route code",
                    "RxNorm code",
                    "RxNorm term",
                ]

                existing_preferred = [c for c in preferred_cols if c in meds_display.columns]
                remaining_cols = [c for c in meds_display.columns if c not in existing_preferred]
                meds_display = meds_display[existing_preferred + remaining_cols]

                st.dataframe(
                    meds_display.head(500),
                    width="stretch",
                    hide_index=True
                )
        with tabs[3]:
            prow = selected_row(cohort, selected_participant)
            if not prow.empty:
                st.dataframe(
                    plot_static_feature_table(pd.DataFrame([prow])),
                    width="stretch",
                    hide_index=True
                )
            else:
                st.info("No cohort metadata row found.")
        with tabs[4]:
            meas = load_measurements_long(selected_participant)
            if not meas.empty:
                st.dataframe(humanize_columns(meas.head(1000)), width="stretch", hide_index=True)
            else:
                st.info("No measurement rows found or file missing.")
else:
    st.info("Select a participant in the cohort explorer first.")
end_section()

section("data-quality", "Data Quality", "Compact diagnostics for selected participant missingness and global clean segment/window distributions.")
q1, q2 = st.columns([1, 1])
with q1:
    st.plotly_chart(plot_missingness(ts_df), width="stretch", key=f"missingness_{selected_participant}")
with q2:
    dcol = duration_col(segments)
    if dcol:
        st.plotly_chart(plot_histogram(segments, dcol, "Clean segment duration"), width="stretch", key="quality_segment_duration")
    else:
        st.info("No segment duration column found.")
end_section()
