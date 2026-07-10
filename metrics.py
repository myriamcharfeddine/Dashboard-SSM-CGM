"""Small metric helpers for enriched CGM dashboard summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def as_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_mean(df: pd.DataFrame, col: str):
    if df is None or df.empty or not col or col not in df.columns:
        return np.nan
    return float(as_numeric(df[col]).mean())


def safe_median(df: pd.DataFrame, col: str):
    if df is None or df.empty or not col or col not in df.columns:
        return np.nan
    return float(as_numeric(df[col]).median())


def safe_min(df: pd.DataFrame, col: str):
    if df is None or df.empty or not col or col not in df.columns:
        return np.nan
    return float(as_numeric(df[col]).min())


def safe_max(df: pd.DataFrame, col: str):
    if df is None or df.empty or not col or col not in df.columns:
        return np.nan
    return float(as_numeric(df[col]).max())


def duration_hours(start, end):
    start = pd.to_datetime(start, errors="coerce")
    end = pd.to_datetime(end, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return float((end - start).total_seconds() / 3600)


def participant_time_stats(df: pd.DataFrame, time_col: str | None):
    if df is None or df.empty:
        return {}
    out = {"rows": len(df)}
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        out["start"] = t.min()
        out["end"] = t.max()
        out["duration_h"] = duration_hours(out["start"], out["end"])
    for col, key in [
        ("cgm_glucose_mean", "mean_glucose"),
        ("heart_rate_mean", "mean_hr"),
        ("respiratory_rate_mean", "mean_rr"),
        ("activity_steps_per_min", "mean_activity"),
    ]:
        out[key] = safe_mean(df, col)
    out["glucose_min"] = safe_min(df, "cgm_glucose_mean")
    out["glucose_max"] = safe_max(df, "cgm_glucose_mean")
    if "activity_steps_per_min" in df.columns:
        out["activity_total_proxy"] = float(pd.to_numeric(df["activity_steps_per_min"], errors="coerce").sum())
    return out
