# Methodology technical record

- Data are e-Stat API responses preserved in `data/raw/`; request filters, table IDs, timestamps, and available SHA-256 hashes are in `catalog/source_registry.parquet`.
- Municipality code is the only join key. The 2000/2010/2020 file is the exact three-year code intersection after aggregate and missing-outcome exclusions: 1,307 codes / 3,921 rows. It is not a boundary-perfect claim.
- Primary SSDS target is `H730102 / H7301` for 2010/2020; it is a non-exclusive mode share. Direct-census historical item/category filters are explicit in `scripts/run_analysis.py` and `qa/source_qa.md`.
- Predictive preprocessing is fitted inside training folds. PCA/clustering are descriptive and exclude transport H730 fields. Panel estimates are within-municipality associations with a year indicator and municipality-clustered SE.
- Predictive comparison uses random and prefecture-grouped folds. The strict temporal test fits 2000 population level, age structure, and rail share to 2010, then evaluates the same feature definitions at 2010 against 2020. 2020 socioeconomic fields are never used.
- Interpretation uses held-out permutation importance and PDP/ALE. SHAP was omitted after a non-reproducible import stall; see `qa/interpretation_methods.md`.
