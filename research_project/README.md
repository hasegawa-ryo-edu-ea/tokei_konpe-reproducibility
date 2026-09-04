# Railway municipal structure research

This project contains only factual research artifacts, technical records, immutable raw sources, and reproducible analysis code. It does not generate paper prose or policy recommendations. SSDSE-A-2026 is the canonical base dataset for the 2020 cross-sectional analysis; e-Stat SSDS supplies explicitly labelled extensions, and Census extracts supply the railway outcome and historical analyses.

## Reproduction

1. Create a Python 3.11+ environment and install `python -m pip install -r requirements.txt`.
2. From this directory run `python scripts/run_all.py`.
3. Run `python scripts/verify_artifacts.py`; it must write `qa/final_verification.json` with `status: pass`.

The exact package versions in `requirements.txt` record the Python 3.11 reproduction environment exercised for this frozen release. Verification checks research invariants and finite results; it does not require byte-identical generated files or machine-level equality of floating-point metrics.

No credential and no network access are required to replay the supplied project: the immutable official SSDSE-A-2026 distribution, e-Stat responses, and request manifests are included below `data/raw/`. Credentials are needed only to perform new retrievals with the adjacent `../e-stat_data_get_system/` tool.

Raw API data are retained under `data/raw/`; processing outputs are regenerated but raw files are never overwritten by these scripts. `processed/municipality_panel_stable_boundary.parquet` is a strict exact-code cohort, not a claim of boundary-perfect stability. See `qa/municipality_harmonization.md`.

## Deliverable map

- `processed/municipality_cross_section_2020.parquet`: SSDSE-A-2026 canonical 2020 cross-section, joined to the Census/SSDS railway outcome.
- `processed/municipality_panel.parquet`: 2010 SSDS extension rows plus the SSDSE-A-2026 canonical 2020 rows; it is not a claim that current SSDSE values are historical observations.
- `qa/ssdse_compliance.json`, `qa/ssdse_temporal_leakage.json`, and `qa/ssdse_feature_semantic_audit.csv`: SSDSE provenance, raw/derived year-lineage, and semantic QA.
- `processed/municipality_panel_stable_boundary.parquet`: 2000/2010/2020 exact-code longitudinal cohort.
- `catalog/feature_dictionary.parquet` and `.csv`: field-level source, transformation, role, coverage, and missingness record.
- `catalog/source_registry.parquet`: request manifests, source table IDs, filters, retrieval times, and raw hashes.
- `runs/experiment_registry.parquet`: de-duplicated model/parameter/metric records.
- `analysis/`, `figures/`, `tables/`, `qa/`, and `final/`: numerical results, visual outputs, QA, and factual indexes.
