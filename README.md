# AI-READI CGM Enriched Dataset Dashboard

This is a local Streamlit dashboard for the CGM thesis project. It focuses on the enriched multimodal dataset and Experiment C split validation: participant-level enriched features, train/validation/test repartition, context/adaptation/evaluation timelines, and selected participant time series.



No code, content, or folder structure is copied from either reference.

## Install

From the dashboard folder or your project environment:

```bash
pip install -r /home/myriamcharfeddine/CGM/dashboard/requirements_dashboard.txt
```

## Run

```bash
streamlit run /home/myriamcharfeddine/CGM/dashboard/app.py
```

Local URL:

```text
http://localhost:8501
```

On a remote VM, forward the Streamlit port first:

```bash
ssh -L 8501:localhost:8501 user@server
```

Then open `http://localhost:8501` in your local browser.

## Expected Paths

The dashboard expects these default locations, defined in `config.py`:

- `/home/myriamcharfeddine/CGM/Data/`
- `/home/myriamcharfeddine/CGM/Data/enriched_multimodal/`
- `/home/myriamcharfeddine/CGM/Data/experiment_c_split_adapt48h_seed42/`
- `/home/myriamcharfeddine/CGM/Data/results/`

Expected files include:

- `final_multimodal_dataset*.parquet`
- `participant_static_features.parquet`
- `cohort.csv`
- `segments.csv`
- `forecast_windows.csv`
- `participant_measurements_selected_long.parquet`
- `participant_medications_long.parquet`
- `split_participants.csv`
- `forecast_windows_with_split.csv`
- `val_personalization_windows.csv`
- `test_personalization_windows.csv`

## Troubleshooting Missing Data

The first dashboard section shows a file availability table with found/missing status, paths, and sizes. Missing files should produce warnings, not crashes. If a section is empty, check whether the corresponding file exists in the expected folder and whether the participant ID appears in that file.

## Adding Future Experiment Outputs

For now, experiment results are intentionally minimal. To extend this later, place result folders, checkpoints, and metrics files under `/home/myriamcharfeddine/CGM/Data/results/`. The placeholder section already detects common files such as `metrics.csv`, `results*.csv`, `predictions*.parquet`, and `*.ckpt`.
