"""Cached defensive loaders for enriched CGM data, read from Google Cloud Storage."""
from __future__ import annotations

import json

import gcsfs
import pandas as pd
import streamlit as st

from config import (
    CANONICAL_STREAM_SPLIT_PREFIX,
    CLINICAL_DATA_PREFIX,
    CORE_TS_COLS,
    ENRICHED_DATASET_PREFIX,
    EXPECTED_FILES,
    GCS_BUCKET,
    GCS_PREFIX,
    PARTICIPANT_COL,
    RESULTS_PREFIX,
    SSM_STREAM_OUTPUT_PREFIX,
    SSM_STREAM_TEST_OUTPUT_PREFIX,
    SSM_STREAM_VALIDATION_OUTPUT_PREFIX,
    T2D_SUBTYPE_CLINICAL_FACTORS_PATH,
    T2D_SUBTYPE_STRATUM,
    TIMESTAMP_CANDIDATES,
)
from vendored_cohort_selection import (
    CORE_COLS as SEGMENTATION_CORE_COLS,
    GAP_THRESHOLDS_MIN,
    _segment_participant,
)

CGM_SEGMENT_GAP_MINUTES = GAP_THRESHOLDS_MIN["cgm"]
FORECAST_ANCHOR_HORIZON_STEP = 1
FORECAST_ANCHOR_SCENARIO_MODE = "forecast_only"
FORECAST_PREDICTIONS_RELATIVE_PATH = "predictions/predictions.parquet"
FORECAST_ANCHOR_COLUMNS = [
    PARTICIPANT_COL, "segment_id", "split", "anchor_time_idx",
    "anchor_timestamp", "hours_since_start", "scenario_mode", "horizon_step",
]
FORECAST_ANCHOR_KEY_COLUMNS = [
    PARTICIPANT_COL, "segment_id", "split", "anchor_time_idx",
]
SEGMENT_BOUNDARY_COLUMNS = ["segment_id", "start", "end"]


def _gcs_key(relative_path: str) -> str:
    return f"{GCS_BUCKET}/{GCS_PREFIX}/{relative_path}"


def _strip_gcs_prefix(key: str) -> str:
    prefix = f"{GCS_BUCKET}/{GCS_PREFIX}/"
    return key[len(prefix):] if key.startswith(prefix) else key


@st.cache_resource(show_spinner=False)
def _gcs_filesystem() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(token=dict(st.secrets["gcp_service_account"]))


@st.cache_resource(show_spinner=False)
def read_parquet_from_gcs(relative_path: str, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """Read a parquet object at gs://{GCS_BUCKET}/{GCS_PREFIX}/{relative_path}."""
    with _gcs_filesystem().open(_gcs_key(relative_path), "rb") as f:
        return pd.read_parquet(f, columns=columns, filters=filters)


@st.cache_resource(show_spinner=False)
def read_csv_from_gcs(relative_path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV object at gs://{GCS_BUCKET}/{GCS_PREFIX}/{relative_path}."""
    with _gcs_filesystem().open(_gcs_key(relative_path), "rb") as f:
        return pd.read_csv(f, **kwargs)


def _gcs_exists(relative_path: str) -> bool:
    try:
        return _gcs_filesystem().exists(_gcs_key(relative_path))
    except Exception:
        return False


def _size(relative_path: str) -> str:
    try:
        value = float(_gcs_filesystem().info(_gcs_key(relative_path))["size"])
    except Exception:
        return ""
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return ""


def _matches(relative_pattern: str) -> list[str]:
    if any(ch in relative_pattern for ch in "*?[]"):
        try:
            found = sorted(_gcs_filesystem().glob(_gcs_key(relative_pattern)))
        except Exception:
            return []
        return [_strip_gcs_prefix(p) for p in found]
    return [relative_pattern] if _gcs_exists(relative_pattern) else []


@st.cache_data(show_spinner=False)
def check_file_availability() -> pd.DataFrame:
    rows = []
    for label, pattern in EXPECTED_FILES.items():
        found = _matches(pattern)
        if found:
            for relative_path in found[:8]:
                rows.append({"File": label, "Found/Missing": "✅ Found", "Path": _gcs_key(relative_path), "Size": _size(relative_path)})
            if len(found) > 8:
                rows.append({"File": label, "Found/Missing": f"✅ Found {len(found)} total", "Path": f"{len(found)-8} more not shown", "Size": ""})
        else:
            rows.append({"File": label, "Found/Missing": "⚠️ Missing", "Path": _gcs_key(pattern), "Size": ""})
    return pd.DataFrame(rows)


def _warn(msg: str):
    st.warning(msg, icon="⚠️")


def _csv(relative_path: str, **kwargs) -> pd.DataFrame:
    try:
        return read_csv_from_gcs(relative_path, **kwargs)
    except FileNotFoundError:
        _warn(f"Missing file: {relative_path}")
    except Exception as exc:
        _warn(f"Could not read {relative_path}: {exc}")
    return pd.DataFrame()


def _parquet(relative_path: str, columns: list[str] | None = None, filters=None, max_rows: int | None = None) -> pd.DataFrame:
    try:
        df = read_parquet_from_gcs(relative_path, columns=columns, filters=filters)
        return df.head(max_rows) if max_rows and len(df) > max_rows else df
    except FileNotFoundError:
        _warn(f"Missing file: {relative_path}")
    except Exception as exc:
        _warn(f"Could not read {relative_path}: {exc}")
    return pd.DataFrame()


def _latest_multimodal() -> str | None:
    files = _matches(f"{ENRICHED_DATASET_PREFIX}/final_multimodal_dataset*.parquet")
    return files[-1] if files else None


@st.cache_data(show_spinner=False)
def multimodal_metadata() -> dict:
    relative_path = _latest_multimodal()
    if not relative_path:
        return {"path": None, "columns": [], "rows": None}
    try:
        import pyarrow.parquet as pq
        with _gcs_filesystem().open(_gcs_key(relative_path), "rb") as f:
            pf = pq.ParquetFile(f)
            return {"path": relative_path, "columns": pf.schema.names, "rows": pf.metadata.num_rows, "row_groups": pf.num_row_groups}
    except Exception as exc:
        _warn(f"Could not inspect multimodal parquet metadata: {exc}")
        return {"path": relative_path, "columns": [], "rows": None}


@st.cache_data(show_spinner=False)
def load_cohort_selection_metadata() -> dict:
    relative_path = f"{ENRICHED_DATASET_PREFIX}/cohort_selection_metadata.json"
    if not _gcs_exists(relative_path):
        _warn(f"Missing file: {relative_path}")
        return {}
    try:
        with _gcs_filesystem().open(_gcs_key(relative_path), "rb") as f:
            return json.loads(f.read())
    except Exception as exc:
        _warn(f"Could not read cohort selection metadata: {exc}")
        return {}


@st.cache_data(show_spinner=False)
def load_original_participants() -> pd.DataFrame:
    """Load the original clinical participants.tsv file for source study-group checks."""
    df = _csv(f"{CLINICAL_DATA_PREFIX}/participants.tsv", sep="\t")
    if "person_id" in df.columns:
        df["participant_id"] = df["person_id"].astype(str)
    if "study_group" in df.columns:
        df = df.rename(columns={"study_group": "participant_tsv_study_group"})
    return df


@st.cache_data(show_spinner=False)
def load_original_condition_groups() -> pd.DataFrame:
    """Derive a compact self-reported diabetes-condition group from condition_occurrence.csv."""
    df = _csv(f"{CLINICAL_DATA_PREFIX}/condition_occurrence.csv", usecols=["person_id", "condition_source_value"])
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
    return _csv(f"{ENRICHED_DATASET_PREFIX}/cohort.csv")


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    return _csv(f"{ENRICHED_DATASET_PREFIX}/segments.csv")


def segment_boundaries(
    df: pd.DataFrame,
    participant_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Return real start and end timestamps for one participant's clean segments."""
    if df is None or df.empty:
        return pd.DataFrame(columns=SEGMENT_BOUNDARY_COLUMNS)

    required = {participant_col, timestamp_col, *SEGMENTATION_CORE_COLS.values()}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Segment boundary input is missing columns: {', '.join(missing)}")

    work = df.loc[:, sorted(required)].dropna(subset=[participant_col]).copy()
    work[participant_col] = work[participant_col].astype(str)
    participant_ids = work[participant_col].unique()
    if len(participant_ids) != 1:
        raise ValueError(
            "segment_boundaries expects exactly one participant; "
            f"received {len(participant_ids)}"
        )

    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    if work.empty:
        return pd.DataFrame(columns=SEGMENT_BOUNDARY_COLUMNS)

    rows = [
        {"segment_id": segment_id, "start": segment.index.min(), "end": segment.index.max()}
        for segment_id, segment in enumerate(_segment_participant(work))
    ]
    return pd.DataFrame(rows, columns=SEGMENT_BOUNDARY_COLUMNS)


@st.cache_data(show_spinner=False)
def load_forecast_anchors() -> pd.DataFrame:
    """Load one timestamped marker per canonical validation or test forecast anchor."""
    relative_paths = [
        f"{prefix}/{FORECAST_PREDICTIONS_RELATIVE_PATH}"
        for prefix in (SSM_STREAM_VALIDATION_OUTPUT_PREFIX, SSM_STREAM_TEST_OUTPUT_PREFIX)
    ]
    frames = []
    filters = [
        ("horizon_step", "==", FORECAST_ANCHOR_HORIZON_STEP),
        ("scenario_mode", "==", FORECAST_ANCHOR_SCENARIO_MODE),
    ]
    for relative_path in relative_paths:
        frame = _parquet(relative_path, columns=FORECAST_ANCHOR_COLUMNS, filters=filters)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=FORECAST_ANCHOR_COLUMNS)

    anchors = pd.concat(frames, ignore_index=True)
    anchors[PARTICIPANT_COL] = anchors[PARTICIPANT_COL].astype(str)
    anchors["anchor_timestamp"] = pd.to_datetime(
        anchors["anchor_timestamp"], errors="coerce"
    )
    invalid_timestamp_count = int(anchors["anchor_timestamp"].isna().sum())
    if invalid_timestamp_count:
        _warn(f"Excluded {invalid_timestamp_count:,} forecast anchors with invalid timestamps.")
    anchors = anchors.dropna(subset=["anchor_timestamp"])
    anchors = anchors.drop_duplicates(FORECAST_ANCHOR_KEY_COLUMNS, keep="first")
    return anchors.sort_values(
        [PARTICIPANT_COL, "segment_id", "anchor_time_idx"]
    ).reset_index(drop=True)


def _stream_anchor_count(relative_path: str) -> int | None:
    """Count anchors once per stream, never once per horizon/scenario row."""
    try:
        df = read_csv_from_gcs(relative_path)
    except Exception:
        return None
    if df.empty or "n_anchors" not in df.columns:
        return None
    if "scenario_mode" in df.columns:
        forecast_only = df[df["scenario_mode"].eq("forecast_only")]
        if not forecast_only.empty:
            df = forecast_only
    stream_cols = [c for c in ["participant_id", "segment_id"] if c in df.columns]
    if stream_cols:
        df = df.drop_duplicates(stream_cols)
    return int(pd.to_numeric(df["n_anchors"], errors="coerce").fillna(0).sum())


@st.cache_data(show_spinner=False)
def load_stream_summary() -> dict:
    """Load stateful SSM-CGM stream and true forecast-anchor totals."""
    stream_count = None
    try:
        segment_df = read_csv_from_gcs(f"{ENRICHED_DATASET_PREFIX}/segments.csv")
        stream_cols = [c for c in ["participant_id", "segment_id"] if c in segment_df.columns]
        stream_count = int(segment_df.drop_duplicates(stream_cols).shape[0]) if stream_cols else int(len(segment_df))
    except Exception:
        pass

    train_anchors = validation_anchors = None
    training_relative_path = f"{SSM_STREAM_OUTPUT_PREFIX}/metrics/training_summary.json"
    try:
        with _gcs_filesystem().open(_gcs_key(training_relative_path), "rb") as f:
            training_summary = json.loads(f.read())
        history = training_summary.get("history", [])
        if history:
            latest = history[-1]
            train_anchors = int(latest["n_train_anchors"])
            validation_anchors = int(latest["n_val_anchors"])
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
        pass

    validation_memory = f"{SSM_STREAM_VALIDATION_OUTPUT_PREFIX}/hardware/stream_state_memory.csv"
    validation_anchors = validation_anchors or _stream_anchor_count(validation_memory)
    test_memory = f"{SSM_STREAM_TEST_OUTPUT_PREFIX}/hardware/stream_state_memory.csv"
    test_anchors = _stream_anchor_count(test_memory)

    split_counts = [train_anchors, validation_anchors, test_anchors]
    total_anchors = sum(split_counts) if all(value is not None for value in split_counts) else None
    return {
        "streams": stream_count,
        "forecast_anchors": total_anchors,
        "train_anchors": train_anchors,
        "validation_anchors": validation_anchors,
        "test_anchors": test_anchors,
    }


@st.cache_data(show_spinner=False)
def load_canonical_stream_split() -> pd.DataFrame:
    """Load the single source of truth for participant split assignment.

    Sourced from CANONICAL_STREAM_SPLIT_PREFIX, referenced as
    split.existing_split_path in the checkpoint's config_resolved.yaml, this
    is the actual split the canonical streaming checkpoint was trained and
    evaluated on, and the same split forecast anchor computation uses. Every
    Train/Validation/Test label shown anywhere in the dashboard must be
    derived from this loader; EXPERIMENT_C_SPLIT_PREFIX backs the retired
    windowed-forecasting pipeline and no longer matches the streaming model.
    """
    return _csv(f"{CANONICAL_STREAM_SPLIT_PREFIX}/split_participants.csv")


@st.cache_data(show_spinner=False)
def load_t2d_subtype_clinical_factors() -> pd.DataFrame:
    """Load per-participant clinical factor values for the frozen T2D oral non-insulin subtypes.

    These are the C1/C2/C3 clusters used in the interpretability chapter,
    frozen before post hoc interpretation.
    """
    df = _csv(T2D_SUBTYPE_CLINICAL_FACTORS_PATH)
    if df.empty or not {"canonical_stratum", "display_cluster", "participant_id"}.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["canonical_stratum"] == T2D_SUBTYPE_STRATUM].copy()
    if df.empty:
        return df
    df["participant_id"] = df["participant_id"].astype(str)
    df["cluster"] = "C" + df["display_cluster"].astype(int).astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_static_features() -> pd.DataFrame:
    relative_parquet = f"{ENRICHED_DATASET_PREFIX}/participant_static_features.parquet"
    if _gcs_exists(relative_parquet):
        return _parquet(relative_parquet)
    return _csv(f"{ENRICHED_DATASET_PREFIX}/participant_static_features.csv")


@st.cache_data(show_spinner=False)
def load_measurements_long(participant_id: str | None = None, max_rows: int = 10000) -> pd.DataFrame:
    relative_parquet = f"{ENRICHED_DATASET_PREFIX}/participant_measurements_selected_long.parquet"
    relative_csv = f"{ENRICHED_DATASET_PREFIX}/participant_measurements_selected_long.csv"
    parquet_exists = _gcs_exists(relative_parquet)
    filters = [(PARTICIPANT_COL, "=", str(participant_id))] if participant_id and parquet_exists else None
    df = _parquet(relative_parquet, filters=filters, max_rows=max_rows) if parquet_exists else _csv(relative_csv, nrows=max_rows)
    if participant_id and PARTICIPANT_COL in df.columns:
        df = df[df[PARTICIPANT_COL].astype(str) == str(participant_id)].head(max_rows)
    return df


@st.cache_data(show_spinner=False)
def load_medications_long(participant_id: str | None = None, max_rows: int = 10000) -> pd.DataFrame:
    relative_parquet = f"{ENRICHED_DATASET_PREFIX}/participant_medications_long.parquet"
    relative_csv = f"{ENRICHED_DATASET_PREFIX}/participant_medications_long.csv"
    parquet_exists = _gcs_exists(relative_parquet)
    filters = [(PARTICIPANT_COL, "=", str(participant_id))] if participant_id and parquet_exists else None
    df = _parquet(relative_parquet, filters=filters, max_rows=max_rows) if parquet_exists else _csv(relative_csv, nrows=max_rows)
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
    relative_path = _latest_multimodal()
    if not relative_path:
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
        df = read_parquet_from_gcs(relative_path, columns=cols, filters=filters)
    except Exception as exc:
        _warn(f"Filtered parquet read failed; trying a limited fallback. Details: {exc}")
        df = _parquet(relative_path, columns=cols, max_rows=max_rows)
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
    relative_path = _latest_multimodal()
    if not relative_path:
        _warn("No final_multimodal_dataset*.parquet found for CGM-derived population metrics.")
        return pd.DataFrame()
    meta = multimodal_metadata()
    available_cols = set(meta.get("columns") or [])
    needed = [PARTICIPANT_COL, "cgm_glucose_mean"]
    if not all(col in available_cols for col in needed):
        _warn("CGM-derived population metrics require participant_id and cgm_glucose_mean in the multimodal parquet.")
        return pd.DataFrame()
    try:
        df = read_parquet_from_gcs(relative_path, columns=needed)
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
    try:
        prefix_key = _gcs_key(RESULTS_PREFIX)
        checkpoints = sorted(_gcs_filesystem().glob(f"{prefix_key}/**/checkpoints/*.ckpt"))
        metrics = sorted(_gcs_filesystem().glob(f"{prefix_key}/**/metrics.csv"))
    except Exception:
        checkpoints, metrics = [], []
    folders = sorted(
        {p.rsplit("/checkpoints/", 1)[0] for p in checkpoints}
        | {p.rsplit("/", 1)[0] for p in metrics}
    )
    return {
        "folders": pd.DataFrame({"result_folder": folders}),
        "checkpoints": pd.DataFrame({"checkpoint": checkpoints}),
        "metrics": pd.DataFrame({"metrics_csv": metrics}),
    }
