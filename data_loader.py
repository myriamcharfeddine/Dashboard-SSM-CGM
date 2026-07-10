"""Cached defensive loaders for enriched CGM data and Experiment C split artifacts."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from config import (
    BASE_DATA_DIR,
    CORE_TS_COLS,
    ENRICHED_DATASET_DIR,
    EXPERIMENT_C_SPLIT_DIR,
    EXPECTED_FILES,
    PARTICIPANT_COL,
    RESULTS_DIR,
    TIMESTAMP_CANDIDATES,
)


def _size(path: Path) -> str:
    try:
        value = float(path.stat().st_size)
    except OSError:
        return ""
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return ""


def _matches(pattern: Path) -> list[Path]:
    s = str(pattern)
    if any(ch in s for ch in "*?[]"):
        return sorted(Path("/").glob(s.lstrip("/"))) if s.startswith("/") else sorted(Path().glob(s))
    return [pattern] if pattern.exists() else []


@st.cache_data(show_spinner=False)
def check_file_availability() -> pd.DataFrame:
    rows = []
    for label, pattern in EXPECTED_FILES.items():
        found = _matches(Path(pattern))
        if found:
            for path in found[:8]:
                rows.append({"File": label, "Found/Missing": "✅ Found", "Path": str(path), "Size": _size(path)})
            if len(found) > 8:
                rows.append({"File": label, "Found/Missing": f"✅ Found {len(found)} total", "Path": f"{len(found)-8} more not shown", "Size": ""})
        else:
            rows.append({"File": label, "Found/Missing": "⚠️ Missing", "Path": str(pattern), "Size": ""})
    return pd.DataFrame(rows)


def _warn(msg: str):
    st.warning(msg, icon="⚠️")


def _csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        _warn(f"Missing file: {path}")
    except Exception as exc:
        _warn(f"Could not read {path}: {exc}")
    return pd.DataFrame()


def _parquet(path: Path, columns: list[str] | None = None, filters=None, max_rows: int | None = None) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path, columns=columns, filters=filters)
        return df.head(max_rows) if max_rows and len(df) > max_rows else df
    except FileNotFoundError:
        _warn(f"Missing file: {path}")
    except Exception as exc:
        _warn(f"Could not read {path}: {exc}")
    return pd.DataFrame()


def _latest_multimodal() -> Path | None:
    files = sorted(ENRICHED_DATASET_DIR.glob("final_multimodal_dataset*.parquet"))
    return files[-1] if files else None


@st.cache_data(show_spinner=False)
def multimodal_metadata() -> dict:
    path = _latest_multimodal()
    if not path:
        return {"path": None, "columns": [], "rows": None}
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return {"path": str(path), "columns": pf.schema.names, "rows": pf.metadata.num_rows, "row_groups": pf.num_row_groups}
    except Exception as exc:
        _warn(f"Could not inspect multimodal parquet metadata: {exc}")
        return {"path": str(path), "columns": [], "rows": None}


@st.cache_data(show_spinner=False)
def load_cohort_selection_metadata() -> dict:
    path = ENRICHED_DATASET_DIR / "cohort_selection_metadata.json"
    if not path.exists():
        _warn(f"Missing file: {path}")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        _warn(f"Could not read cohort selection metadata: {exc}")
        return {}


@st.cache_data(show_spinner=False)
def load_original_participants() -> pd.DataFrame:
    """Load the original clinical participants.tsv file for source study-group checks."""
    path = BASE_DATA_DIR / "clinical_data" / "participants.tsv"
    try:
        df = pd.read_csv(path, sep="\t")
    except FileNotFoundError:
        _warn(f"Missing file: {path}")
        return pd.DataFrame()
    except Exception as exc:
        _warn(f"Could not read {path}: {exc}")
        return pd.DataFrame()
    if "person_id" in df.columns:
        df["participant_id"] = df["person_id"].astype(str)
    if "study_group" in df.columns:
        df = df.rename(columns={"study_group": "participant_tsv_study_group"})
    return df


@st.cache_data(show_spinner=False)
def load_original_condition_groups() -> pd.DataFrame:
    """Derive a compact self-reported diabetes-condition group from condition_occurrence.csv."""
    path = BASE_DATA_DIR / "clinical_data" / "condition_occurrence.csv"
    try:
        df = pd.read_csv(path, usecols=["person_id", "condition_source_value"])
    except FileNotFoundError:
        _warn(f"Missing file: {path}")
        return pd.DataFrame()
    except Exception as exc:
        _warn(f"Could not read {path}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    source = df["condition_source_value"].fillna("").astype(str).str.lower()
    df = df.assign(
        participant_id=df["person_id"].astype(str),
        has_t2d=source.str.contains("type ii diabetes|mhterm_dm2", regex=True),
        has_prediabetes=source.str.contains("pre-diabetes|prediabetes|mhterm_predm", regex=True),
        has_diabetes_complication=source.str.contains("diabetic retinopathy", regex=True),
    )
    grouped = df.groupby("participant_id", as_index=False).agg(
        has_t2d=("has_t2d", "max"),
        has_prediabetes=("has_prediabetes", "max"),
        has_diabetes_complication=("has_diabetes_complication", "max"),
    )
    grouped["condition_file_self_report"] = "No diabetes condition recorded"
    grouped.loc[grouped["has_diabetes_complication"], "condition_file_self_report"] = "Diabetes-related complication only"
    grouped.loc[grouped["has_prediabetes"], "condition_file_self_report"] = "Pre-diabetes condition"
    grouped.loc[grouped["has_t2d"], "condition_file_self_report"] = "Type II diabetes condition"
    return grouped[["participant_id", "condition_file_self_report"]]


@st.cache_data(show_spinner=False)
def load_cohort() -> pd.DataFrame:
    return _csv(ENRICHED_DATASET_DIR / "cohort.csv")


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    return _csv(ENRICHED_DATASET_DIR / "segments.csv")


@st.cache_data(show_spinner=False)
def load_forecast_windows() -> pd.DataFrame:
    return _csv(ENRICHED_DATASET_DIR / "forecast_windows.csv")


@st.cache_data(show_spinner=False)
def load_split_participants() -> pd.DataFrame:
    return _csv(EXPERIMENT_C_SPLIT_DIR / "split_participants.csv")


@st.cache_data(show_spinner=False)
def load_split_summary() -> pd.DataFrame:
    return _csv(EXPERIMENT_C_SPLIT_DIR / "split_summary.csv")


@st.cache_data(show_spinner=False)
def load_forecast_windows_with_split() -> pd.DataFrame:
    return _csv(EXPERIMENT_C_SPLIT_DIR / "forecast_windows_with_split.csv")


@st.cache_data(show_spinner=False)
def load_personalization_windows(split: str) -> pd.DataFrame:
    if split not in {"val", "test", "validation"}:
        return pd.DataFrame()
    name = "val" if split == "validation" else split
    return _csv(EXPERIMENT_C_SPLIT_DIR / f"{name}_personalization_windows.csv")


@st.cache_data(show_spinner=False)
def load_static_features() -> pd.DataFrame:
    p = ENRICHED_DATASET_DIR / "participant_static_features.parquet"
    if p.exists():
        return _parquet(p)
    return _csv(ENRICHED_DATASET_DIR / "participant_static_features.csv")


@st.cache_data(show_spinner=False)
def load_measurements_long(participant_id: str | None = None, max_rows: int = 10000) -> pd.DataFrame:
    p = ENRICHED_DATASET_DIR / "participant_measurements_selected_long.parquet"
    c = ENRICHED_DATASET_DIR / "participant_measurements_selected_long.csv"
    filters = [(PARTICIPANT_COL, "=", str(participant_id))] if participant_id and p.exists() else None
    df = _parquet(p, filters=filters, max_rows=max_rows) if p.exists() else _csv(c, nrows=max_rows)
    if participant_id and PARTICIPANT_COL in df.columns:
        df = df[df[PARTICIPANT_COL].astype(str) == str(participant_id)].head(max_rows)
    return df


@st.cache_data(show_spinner=False)
def load_medications_long(participant_id: str | None = None, max_rows: int = 10000) -> pd.DataFrame:
    p = ENRICHED_DATASET_DIR / "participant_medications_long.parquet"
    c = ENRICHED_DATASET_DIR / "participant_medications_long.csv"
    filters = [(PARTICIPANT_COL, "=", str(participant_id))] if participant_id and p.exists() else None
    df = _parquet(p, filters=filters, max_rows=max_rows) if p.exists() else _csv(c, nrows=max_rows)
    if participant_id and PARTICIPANT_COL in df.columns:
        df = df[df[PARTICIPANT_COL].astype(str) == str(participant_id)].head(max_rows)
    return df


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        lc = col.lower()
        if "time" in lc or "date" in lc:
            return col
    return None


@st.cache_data(show_spinner=True)
def load_participant_timeseries(participant_id: str, max_rows: int = 60000) -> pd.DataFrame:
    path = _latest_multimodal()
    if not path:
        _warn("No final_multimodal_dataset*.parquet found.")
        return pd.DataFrame()
    meta = multimodal_metadata()
    available_cols = set(meta.get("columns") or [])
    cols = [c for c in CORE_TS_COLS if c in available_cols]
    if PARTICIPANT_COL not in cols and PARTICIPANT_COL in available_cols:
        cols.insert(0, PARTICIPANT_COL)
    if not cols:
        _warn("Could not identify any expected time-series columns in the multimodal parquet.")
        return pd.DataFrame()
    filters = [(PARTICIPANT_COL, "=", str(participant_id))] if PARTICIPANT_COL in available_cols else None
    try:
        df = pd.read_parquet(path, columns=cols, filters=filters)
    except Exception as exc:
        _warn(f"Filtered parquet read failed; trying a limited fallback. Details: {exc}")
        df = _parquet(path, columns=cols, max_rows=max_rows)
        if PARTICIPANT_COL in df.columns:
            df = df[df[PARTICIPANT_COL].astype(str) == str(participant_id)]
    if len(df) > max_rows:
        df = df.head(max_rows)
        _warn(f"Participant has more than {max_rows:,} rows; showing the first {max_rows:,} rows for responsiveness.")
    tcol = detect_timestamp_column(df)
    if tcol:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.sort_values(tcol)
    else:
        _warn("Could not detect timestamp column. Expected one of timestamp, datetime, time, index_time, measurement_datetime.")
    return df


@st.cache_data(show_spinner=False)
def load_cgm_participant_metrics() -> pd.DataFrame:
    """Compute participant-level CGM summaries from the enriched multimodal parquet."""
    path = _latest_multimodal()
    if not path:
        _warn("No final_multimodal_dataset*.parquet found for CGM-derived population metrics.")
        return pd.DataFrame()
    meta = multimodal_metadata()
    available_cols = set(meta.get("columns") or [])
    needed = [PARTICIPANT_COL, "cgm_glucose_mean"]
    if not all(col in available_cols for col in needed):
        _warn("CGM-derived population metrics require participant_id and cgm_glucose_mean in the multimodal parquet.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=needed)
    except Exception as exc:
        _warn(f"Could not compute CGM-derived population metrics: {exc}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    glucose = pd.to_numeric(df["cgm_glucose_mean"], errors="coerce")
    work = pd.DataFrame({PARTICIPANT_COL: df[PARTICIPANT_COL].astype(str), "glucose": glucose}).dropna(subset=["glucose"])
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby(PARTICIPANT_COL)["glucose"]
    summary = grouped.agg(
        mean_glucose_mgdl="mean",
        cgm_rows="size",
        glucose_std="std",
    ).reset_index()
    tir = grouped.apply(lambda s: s.between(70, 180).mean() * 100).reset_index(name="tir_70_180_pct")
    summary = summary.merge(tir, on=PARTICIPANT_COL, how="left")
    summary["cv_pct"] = summary["glucose_std"] / summary["mean_glucose_mgdl"] * 100
    return summary.drop(columns=["glucose_std"])


@st.cache_data(show_spinner=False)
def detected_results() -> dict[str, pd.DataFrame]:
    folders = [p for p in RESULTS_DIR.iterdir() if p.is_dir()] if RESULTS_DIR.exists() else []
    checkpoints = list(RESULTS_DIR.rglob("checkpoints/*.ckpt")) if RESULTS_DIR.exists() else []
    metrics = list(RESULTS_DIR.rglob("metrics.csv")) if RESULTS_DIR.exists() else []
    return {
        "folders": pd.DataFrame({"result_folder": [str(p) for p in folders]}),
        "checkpoints": pd.DataFrame({"checkpoint": [str(p) for p in checkpoints]}),
        "metrics": pd.DataFrame({"metrics_csv": [str(p) for p in metrics]}),
    }
