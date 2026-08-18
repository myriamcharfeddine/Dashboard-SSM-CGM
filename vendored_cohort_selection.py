"""Segment-boundary logic vendored from SSM-CGM/Preprocessing/cohort_selection.py.

Only the constants and functions this dashboard actually uses (CORE_COLS,
GAP_THRESHOLDS_MIN, and the _segment_participant call chain, copied verbatim)
are included here, so the dashboard has no runtime dependency on a local
checkout of the sibling SSM-CGM repo and can run on Streamlit Community Cloud.
"""
from __future__ import annotations

import pandas as pd

BIN_MINUTES = 5
MAX_CONTEXT_H = 48
TARGET_H = 1
MIN_SEGMENT_H = MAX_CONTEXT_H + TARGET_H  # 49 h
MIN_SEGMENT_BINS = MIN_SEGMENT_H * 60 // BIN_MINUTES

GAP_THRESHOLDS_MIN = {"cgm": 30, "hr": 60, "rr": 60, "activity": 60}
GAP_THRESHOLDS_BINS = {k: v // BIN_MINUTES for k, v in GAP_THRESHOLDS_MIN.items()}

# Core modalities: key → column name in the time-series parquet
CORE_COLS: dict[str, str] = {
    "cgm": "cgm_glucose_mean",
    "hr": "heart_rate_mean",
    "rr": "respiratory_rate_mean",
    "activity": "activity_steps_per_min",
}


def _find_long_gap_mask(series: pd.Series, threshold_bins: int) -> pd.Series:
    is_null = series.isna()
    if not is_null.any():
        return pd.Series(False, index=series.index)
    run_id = (~is_null).cumsum()
    run_lens = is_null.groupby(run_id).sum()
    bad_runs = set(run_lens[run_lens > threshold_bins].index)
    if not bad_runs:
        return pd.Series(False, index=series.index)
    return is_null & run_id.isin(bad_runs)


def _segment_participant(
    grp: pd.DataFrame,
    gap_thr_bins: dict[str, int] = GAP_THRESHOLDS_BINS,
    min_seg_bins: int = MIN_SEGMENT_BINS,
) -> list[pd.DataFrame]:
    grp = grp.sort_index()
    bad = pd.Series(False, index=grp.index)
    for k, col in CORE_COLS.items():
        bad |= _find_long_gap_mask(grp[col], gap_thr_bins[k])

    good = ~bad
    if not good.any():
        return []
    seg_label = (good & (good != good.shift(fill_value=False))).cumsum()
    seg_label[bad] = 0
    segments = []
    for _, seg_grp in grp[good].groupby(seg_label[good]):
        if len(seg_grp) >= min_seg_bins:
            segments.append(seg_grp)
    return segments
