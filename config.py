"""Configuration for the AI-READI CGM enriched dataset dashboard."""

# Data is read from Google Cloud Storage rather than the local filesystem so
# the dashboard can run on Streamlit Community Cloud. Every *_PREFIX below is
# a path relative to gs://{GCS_BUCKET}/{GCS_PREFIX}/, resolved by the
# read_parquet_from_gcs / read_csv_from_gcs helpers in data_loader.py.
GCS_BUCKET = "cgmproject2025"
GCS_PREFIX = "dashboard_data"

# NOT currently present under gs://cgmproject2025/dashboard_data/ (verified
# 2026-08-18) -- participants.tsv / condition_occurrence.csv have not been
# uploaded yet. The demographic-balance tab will show its existing
# "could not be loaded" warnings until this prefix exists in the bucket.
CLINICAL_DATA_PREFIX = "clinical_data"
ENRICHED_DATASET_PREFIX = "enriched_multimodal"
EXPERIMENT_C_SPLIT_PREFIX = "experiment_c_split_adapt48h_seed42"
# Unverified in the bucket; only backs detected_results(), which nothing in
# app.py currently calls.
RESULTS_PREFIX = "results"
# The SSM-CGM model outputs are a separate local project (CGM/SSM-CGM/outputs/...)
# from the CGM/Data/ tree the other prefixes above mirror, so they are
# namespaced under ssm_cgm_outputs/ in the bucket.
SSM_STREAM_OUTPUT_PREFIX = "ssm_cgm_outputs/aireadi_stream_mamba_stateful_5epoch"
SSM_STREAM_VALIDATION_OUTPUT_PREFIX = "ssm_cgm_outputs/aireadi_stream_mamba_stateful_10epoch_eval_validation"
SSM_STREAM_TEST_OUTPUT_PREFIX = "ssm_cgm_outputs/aireadi_stream_mamba_stateful_10epoch_eval_test"
# Canonical participant split used to train the epoch-5 checkpoint at
# SSM_STREAM_OUTPUT_PREFIX -- confirmed via config_resolved.yaml
# (split.existing_split_path) inside SSM_STREAM_OUTPUT_PREFIX in the bucket,
# which still points at this exact relative path. Distinct from
# EXPERIMENT_C_SPLIT_PREFIX, which backs the retired windowed pipeline.
# NOT currently present under gs://cgmproject2025/dashboard_data/ (verified
# 2026-08-18) -- kept as this exact path so the file starts working the
# moment it's uploaded; until then, load_canonical_stream_split() returns
# empty and every Train/Validation/Test label in the dashboard falls back to
# the app's existing "canonical split could not be loaded" placeholder.
CANONICAL_STREAM_SPLIT_PREFIX = "experiment_c_split_adapt6h_seed42"
CANONICAL_STREAM_CHECKPOINT_VAL_PINBALL_MGDL = 3.286316
# Frozen T2D oral non-insulin subtype clustering (C1/C2/C3) used in the
# interpretability chapter; per-participant clinical factor values, long format.
T2D_SUBTYPE_CLINICAL_FACTORS_PATH = (
    "ssm_cgm_outputs/static_phenotype_trajectory_stratified_v2/"
    "extended_clinical_latent_dynamics_v1/01_cluster_metabolic_profiles/figure_1A_plotted_data.csv"
)
T2D_SUBTYPE_STRATUM = "t2d_oral_non_insulin"

EXPECTED_FILES = {
    "final_multimodal_dataset*.parquet": f"{ENRICHED_DATASET_PREFIX}/final_multimodal_dataset*.parquet",
    "participant_static_features.parquet": f"{ENRICHED_DATASET_PREFIX}/participant_static_features.parquet",
    "cohort.csv": f"{ENRICHED_DATASET_PREFIX}/cohort.csv",
    "segments.csv": f"{ENRICHED_DATASET_PREFIX}/segments.csv",
    "forecast_windows.csv": f"{ENRICHED_DATASET_PREFIX}/forecast_windows.csv",
    "participant_measurements_selected_long.parquet": f"{ENRICHED_DATASET_PREFIX}/participant_measurements_selected_long.parquet",
    "participant_medications_long.parquet": f"{ENRICHED_DATASET_PREFIX}/participant_medications_long.parquet",
    "split_participants.csv": f"{CANONICAL_STREAM_SPLIT_PREFIX}/split_participants.csv",
    "forecast_windows_with_split.csv": f"{EXPERIMENT_C_SPLIT_PREFIX}/forecast_windows_with_split.csv",
    "val_personalization_windows.csv": f"{EXPERIMENT_C_SPLIT_PREFIX}/val_personalization_windows.csv",
    "test_personalization_windows.csv": f"{EXPERIMENT_C_SPLIT_PREFIX}/test_personalization_windows.csv",
}

PARTICIPANT_COL = "participant_id"
PALETTE = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
STRATUM_COLORS = {
    "Forecast anchor": PALETTE[0],
    "Warm-up": PALETTE[4],
    "Segment reset": PALETTE[4],
}
PERSONALIZATION_WARMUP_HOURS = (0, 6, 12, 24, 48)
STREAM_TIMELINE_SIGNAL = "cgm_glucose_mean"
T2D_SUBTYPE_CLUSTER_ORDER = ["C1", "C2", "C3"]
T2D_SUBTYPE_CLUSTER_COLORS = {
    "C1": PALETTE[1],
    "C2": PALETTE[2],
    "C3": PALETTE[0],
}
T2D_SUBTYPE_CLUSTER_INTERPRETATION = {
    "C1": "Lower BMI and lower proxy profile, more MARD-like",
    "C2": "Obesity dominant and younger profile, MOD-like",
    "C3": "Higher C-peptide, TG/HDL, and central adiposity, SIRD-like",
}
T2D_SUBTYPE_FACTOR_COLUMNS = {
    "participants_age": "Age [years]",
    "bmi_baseline": "BMI [kg/m2]",
    "hba1c_percent_baseline": "HbA1c [%]",
    "c_peptide_ngml_baseline": "C-peptide [ng/mL]",
    "tg_hdl_ratio": "TG/HDL ratio",
    "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
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
    "c_peptide_ngml_baseline": "C-peptide [ng/mL]",
    "tg_hdl_ratio": "TG/HDL ratio",
    "waist_to_hip_ratio_baseline": "Waist-to-hip ratio",
    "participants_age": "Age [years]",
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
