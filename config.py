"""Configuration for the AI-READI CGM enriched dataset dashboard."""
from pathlib import Path

BASE_DATA_DIR = Path("/home/myriamcharfeddine/CGM/Data/")
ENRICHED_DATASET_DIR = Path("/home/myriamcharfeddine/CGM/Data/enriched_multimodal/")
EXPERIMENT_C_SPLIT_DIR = Path("/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt48h_seed42/")
RESULTS_DIR = Path("/home/myriamcharfeddine/CGM/Data/results/")

EXPECTED_FILES = {
    "final_multimodal_dataset*.parquet": ENRICHED_DATASET_DIR / "final_multimodal_dataset*.parquet",
    "participant_static_features.parquet": ENRICHED_DATASET_DIR / "participant_static_features.parquet",
    "cohort.csv": ENRICHED_DATASET_DIR / "cohort.csv",
    "segments.csv": ENRICHED_DATASET_DIR / "segments.csv",
    "forecast_windows.csv": ENRICHED_DATASET_DIR / "forecast_windows.csv",
    "participant_measurements_selected_long.parquet": ENRICHED_DATASET_DIR / "participant_measurements_selected_long.parquet",
    "participant_medications_long.parquet": ENRICHED_DATASET_DIR / "participant_medications_long.parquet",
    "split_participants.csv": EXPERIMENT_C_SPLIT_DIR / "split_participants.csv",
    "forecast_windows_with_split.csv": EXPERIMENT_C_SPLIT_DIR / "forecast_windows_with_split.csv",
    "val_personalization_windows.csv": EXPERIMENT_C_SPLIT_DIR / "val_personalization_windows.csv",
    "test_personalization_windows.csv": EXPERIMENT_C_SPLIT_DIR / "test_personalization_windows.csv",
}

PARTICIPANT_COL = "participant_id"
PALETTE = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
PHASE_COLORS = {
    "Context": "#003366",
    "Adaptation": "#5BBABA",
    "Evaluation": "#BA2828",
    "Unused / other": "#888888",
}
SPLIT_COLORS = {
    "Train": "#003366",
    "Validation": "#5BBABA",
    "Test": "#BA2828",
    "Other": "#888888",
}

CORE_TS_COLS = [
    "participant_id",
    "timestamp_local",
    "timestamp",
    "datetime",
    "time",
    "index_time",
    "measurement_datetime",
    "cgm_glucose_mean",
    "heart_rate_mean",
    "respiratory_rate_mean",
    "activity_steps_per_min",
]
TIMESTAMP_CANDIDATES = ["timestamp_local", "timestamp", "datetime", "time", "index_time", "measurement_datetime", "ds", "window_start"]
SIGNAL_COLUMNS = {
    "CGM glucose [mg/dL]": "cgm_glucose_mean",
    "Heart rate [bpm]": "heart_rate_mean",
    "Respiratory rate [breaths/min]": "respiratory_rate_mean",
    "Activity [steps/min]": "activity_steps_per_min",
}
COLUMN_LABELS = {
    "participant_id": "Participant ID",
    "split": "Split",
    "study_group": "Study group",
    "stratum": "Study group",
    "clinical_site": "Clinical site",
    "age": "Age [years]",
    "BMI": "BMI [kg/m2]",
    "HbA1c": "HbA1c [%]",
    "n_valid_windows": "Valid forecast windows",
    "n_segments": "Clean segments",
    "total_clean_dur_h": "Clean duration [h]",
    "longest_seg_h": "Longest clean segment [h]",
    "weighted_completeness": "Weighted completeness",
    "post_weighted_completeness": "Post-cleaning completeness",
    "personalization_eligible": "Personalization eligible",
    "duration_h": "Duration [h]",
    "dur_h": "Segment duration [h]",
    "windows": "Forecast windows",
    "participants": "Participants",
    "missing_fraction": "Missing fraction",
    "variable": "Variable",
    "phase": "Phase",
    "start": "Start",
    "end": "End",
    "duration_h_phase": "Duration [h]",
    "n_windows": "Windows",
    "cgm_glucose_mean": "CGM glucose [mg/dL]",
    "heart_rate_mean": "Heart rate [bpm]",
    "respiratory_rate_mean": "Respiratory rate [breaths/min]",
    "activity_steps_per_min": "Activity [steps/min]",
    "participants_study_group": "Study group",
    "participants_age": "Age [years]",
    "participants_clinical_site": "Clinical site",
    "clinical_systolic_bp_mmhg_baseline": "Systolic BP [mmHg]",
    "clinical_diastolic_bp_mmhg_baseline": "Diastolic BP [mmHg]",
    "hba1c_percent_baseline": "HbA1c [%]",
    "bmi_baseline": "BMI [kg/m²]",
    "hdl_cholesterol_mgdl_baseline": "HDL [mg/dL]",
    "ldl_cholesterol_mgdl_baseline": "LDL [mg/dL]",
    "triglycerides_mgdl_baseline": "Triglycerides [mg/dL]",
    "clinical_resting_hr_bpm_baseline": "Resting HR [bpm]",
    "mean_glucose_mgdl": "Mean glucose [mg/dL]",
    "tir_70_180_pct": "Time in range 70-180 [%]",
    "cv_pct": "Glucose CV [%]",
    "cgm_rows": "CGM rows",
}
STUDY_GROUP_LABELS = {
    "healthy": "Healthy",
    "Healthy": "Healthy",
    "pre_diabetes_lifestyle_controlled": "Prediabetes",
    "prediabetes": "Prediabetes",
    "T2D-non-insulin": "T2D Non Insulin",
    "T2D_non_insulin": "T2D Non Insulin",
    "non_insulin_dependent": "T2D Non Insulin",
    "T2D-insulin": "T2D insulin",
    "T2D-oral": "T2D Non Insulin",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "T2D Non Insulin",
    "Oral Medication And Or Non Insulin Injectable Medication Controlled": "T2D Non Insulin",
    "insulin_dependent": "T2D insulin",
}
PLOT_TEMPLATE = "plotly_white"
