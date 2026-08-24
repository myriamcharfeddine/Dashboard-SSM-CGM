# AI-READI CGM Enriched Dataset Dashboard

A password-protected Streamlit dashboard for the CGM thesis project. It covers the enriched multimodal AI-READI dataset, the preprocessing/cohort-selection pipeline, the canonical train/validation/test stream split, per-participant time-series exploration, and the T2D oral non-insulin subtype (C1/C2/C3) interpretability analysis.



## Data source: Google Cloud Storage

The dashboard does **not** read from the local filesystem. All data is loaded from GCS so the app can run on Streamlit Community Cloud, from `gs://cgmproject2025/dashboard_data/`, with each dataset under its own prefix (see [config.py](config.py)):

- `enriched_multimodal/` — enriched multimodal parquet/CSV files (cohort, segments, forecast windows, static features, measurements, medications)
- `clinical_data/` — original participants/condition-occurrence tables (not yet uploaded as of 2026-08-18; the demographic-balance tab shows "could not be loaded" warnings until it exists)
- `experiment_c_split_adapt48h_seed42/` — retired windowed-pipeline split (`RESULTS_PREFIX`/legacy)
- `experiment_c_split_adapt6h_seed42/` — canonical split for the stream/SSM-CGM model (`split_participants.csv`)
- `ssm_cgm_outputs/aireadi_stream_mamba_stateful_*` — SSM-CGM model outputs (train/val/test eval), a separate local project from the `CGM/Data/` tree
- `ssm_cgm_outputs/static_phenotype_trajectory_stratified_v2/.../figure_1A_plotted_data.csv` — frozen T2D subtype clinical-factor data (C1/C2/C3)

Access is via a GCP service account, configured through Streamlit secrets (see below) and read with `gcsfs`/`google-cloud-storage` in [data_loader.py](data_loader.py).

## Install

```bash
pip install -r requirements.txt
```

(`requirements.txt` and `requirements_dashboard.txt` are kept identical; either works.)

## Secrets

The app requires a `dashboard_password` and a `gcp_service_account` block in Streamlit secrets — it will not start without them.

- **Local development**: create `.streamlit/secrets.toml` (gitignored, never commit it) with:
  ```toml
  dashboard_password = "..."

  [gcp_service_account]
  type = "service_account"
  project_id = "..."
  # ...full service account key fields...
  ```
- **Streamlit Community Cloud**: paste the same content into the app's *Settings → Secrets* panel.

## Run

Locally:

```bash
streamlit run app.py
```

Local URL:

```text
http://localhost:8501
```

On a remote VM, forward the port first:

```bash
ssh -L 8501:localhost:8501 user@server
```

Then open `http://localhost:8501` and enter the dashboard password.

On Streamlit Community Cloud, the app is deployed straight from this repo/branch with secrets configured as above.

## Dashboard sections

After the password gate, the app is organized into these sections (in page order):

1. **Preprocessing Steps and Cohort Selection Impact** — stage-level view of how raw multimodal data becomes clean, continuous streams for the stateful SSM-CGM model.
2. **Train / Validation / Test Split** — participant repartition and stratification for the canonical split manifest used throughout the dashboard.
3. **Population Statistics Dashboard** — cohort-level distributions across tabs: Cohort proportions, Clinical distributions, Duration and coverage, Medication patterns, Demographic balance, CGM-derived metrics, Blood pressure & cardiometabolic, and T2D heterogeneity (the C1/C2/C3 subtype analysis).
4. **Participant Explorer** — filter by split/study group and select a participant, feeding the profile, timeline, time-series, and static-feature views below.
5. **Participant Time-Series Explorer** — stacked plots for CGM glucose, heart rate, respiratory rate, and activity.
6. **Static Feature Profile** — grouped enriched participant features shown as a medical profile rather than a raw table.
7. **Data Quality** — per-participant missingness diagnostics plus global clean segment/window distributions.

## Troubleshooting missing data

The dashboard checks file availability against GCS at startup (`check_file_availability()` in [data_loader.py](data_loader.py)) and is built to degrade gracefully: missing files/prefixes produce warnings in the relevant section rather than crashing the app. If a section is empty, check whether the corresponding GCS prefix/file exists under `gs://cgmproject2025/dashboard_data/` and whether the participant ID appears in that file.
