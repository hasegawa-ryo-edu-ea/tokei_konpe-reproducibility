#!/usr/bin/env python3
"""Create factual result/index documents; deliberately no submission prose."""
from pathlib import Path
import json,pandas as pd
R=Path(__file__).resolve().parents[1];A=R/'analysis';P=R/'processed';T=R/'tables';F=R/'final';T.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
def main():
 # One authoritative N is the published maximum 2020 cross-section file.
 keypath=F/'key_numbers.json'; key=json.loads(keypath.read_text(encoding='utf8')); cs=pd.read_parquet(P/'municipality_cross_section_2020.parquet'); balanced=pd.read_parquet(P/'municipality_panel_balanced.parquet'); strict_panel=pd.read_parquet(P/'municipality_panel_stable_boundary.parquet'); features=pd.read_parquet(A/'ml_features.parquet').feature.tolist()
 key['n_cross_section_2020']=int(len(cs)); key['n_balanced_2010_2020']=int(balanced.code.nunique()); key['n_features_ml']=len(features)
 key['cross_section_universe']={'name':'SSDSE-A-2026 canonical municipalities with observed 2020 railway target','n':int(len(cs))}; key['balanced_panel_universe']={'name':'canonical-code intersection with observed 2010 and 2020 SSDS outcome','n':int(balanced.code.nunique())}; key['strict_temporal_universe']={'name':'2000/2010/2020 exact-code historical cohort','n':int(strict_panel.code.nunique())}
 # Replace legacy/in-sample metric blocks with the canonical position-aligned OOF table.
 canonical=pd.read_csv(A/'robustness/oof_model_metrics.csv'); key.pop('cross_oof_metrics',None); key.pop('temporal_diagnostic_metrics',None); key['canonical_oof_metrics']=canonical.to_dict(orient='records'); keypath.write_text(json.dumps(key,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 perf=pd.read_csv(A/'sensitivity/model_performance.csv');x=pd.read_csv(A/'ml/xgboost_metrics.csv');cb=pd.read_csv(A/'ml/catboost_metrics.csv');allp=pd.concat([perf,x,cb],ignore_index=True);allp.to_csv(T/'all_model_performance.csv',index=False)
 strict=pd.read_parquet(P/'municipality_panel_stable_boundary.parquet');imp=pd.read_csv(A/'interpretation/oof_permutation_importance.csv');cl=pd.read_csv(A/'clustering_comparison.csv');tem=pd.read_csv(A/'temporal/temporal_validation_enhanced.csv');oof=pd.read_csv(A/'robustness/oof_model_metrics.csv');best=tem.loc[tem.rmse.idxmin()]
 text=['# Factual result record','',f'- 2020 cross-section: {pd.read_parquet(P/"municipality_cross_section_2020.parquet").shape[0]} municipalities.',f'- Strict exact-code 2000/2010/2020 cohort: {strict.code.nunique()} municipalities; {len(strict)} rows.','- Geographic GroupKFold and random KFold metrics: tables/all_model_performance.csv.','- Fold-separated five-model OOF metrics and consensus residuals: analysis/robustness/.','- Cluster method/grid metrics: analysis/clustering_comparison.csv.','- Held-out permutation importance: analysis/interpretation/oof_permutation_importance.csv.',f'- Best specified compact temporal test: {best.model}, RMSE={best.rmse:.6f}; MAE={best.mae:.6f}; R2={best.r2:.6f}; Spearman={best.spearman:.6f}; mean bias={best.mean_bias:.6f}.','- Temporal models use 2000/2010 population level, age structure, and prior rail share only; no 2020 predictors are used.','- Fractional-logit robustness and 500-replicate prefecture-cluster bootstrap: analysis/bounded/ and analysis/bootstrap/.','- Top fold-separated permutation features:',*['  - '+r.feature+': RMSE increase='+format(r.mean_importance,'.6f') for _,r in imp.head(10).iterrows()]]
 (F/'results_factual.md').write_text('\n'.join(text)+'\n',encoding='utf8')
 (F/'table_index.md').write_text('\n'.join(['# Tables','','- tables/all_model_performance.csv','- tables/model_performance.csv','- tables/cluster_method_comparison.csv','- tables/longitudinal_summary.csv','- analysis/temporal/temporal_validation_enhanced.csv','- analysis/robustness/oof_model_metrics.csv','- analysis/bounded/fractional_logit_coefficients.csv','- analysis/bootstrap/bootstrap_summary.csv','- analysis/interpretation/oof_permutation_importance.csv','- analysis/residual/consensus_residual_consistent_outliers.csv']),encoding='utf8')
 (F/'figure_index.md').write_text('\n'.join(['# Figures','',*['- '+p.name for p in sorted((R/'figures').glob('*.png'))]]),encoding='utf8')
 (F/'reproduction.md').write_text('''# Reproduction record\n\nRaw e-Stat API responses and request manifests are preserved under `data/raw/`; analysis scripts do not overwrite them.\n\n## Clean-environment execution\n\n1. Create and activate a Python 3.11+ environment.\n2. Run `pip install -r requirements.txt`.\n3. From `research_project`, run `python scripts/run_all.py`.\n4. Run `python scripts/verify_artifacts.py`. It must emit the invariant JSON and write `qa/final_verification.json` with `status: pass`.\n\n`run_all.py` rebuilds processed panels, figures, model records, QA, factual indexes, enhanced feature dictionary, and the de-duplicated registry from preserved raw inputs. Randomized estimators use `random_state=2026`; hardware-dependent floating-point variation can cause minor metric differences. No API retrieval is needed for this replay.\n''',encoding='utf8')
 (F/'methodology_technical.md').write_text('''# Methodology technical record

- Data are e-Stat API responses preserved in `data/raw/`; request filters, table IDs, timestamps, and available SHA-256 hashes are in `catalog/source_registry.parquet`.
- Municipality code is the only join key. The 2000/2010/2020 file is the exact three-year code intersection after aggregate and missing-outcome exclusions: 1,307 codes / 3,921 rows. It is not a boundary-perfect claim.
- Primary SSDS target is `H730102 / H7301` for 2010/2020; it is a non-exclusive mode share. Direct-census historical item/category filters are explicit in `scripts/run_analysis.py` and `qa/source_qa.md`.
- Predictive preprocessing is fitted inside training folds. PCA/clustering are descriptive and exclude transport H730 fields. Panel estimates are within-municipality associations with a year indicator and municipality-clustered SE.
- Predictive comparison uses random and prefecture-grouped folds. The strict temporal test fits 2000 population level, age structure, and rail share to 2010, then evaluates the same feature definitions at 2010 against 2020. 2020 socioeconomic fields are never used.
- Interpretation uses held-out permutation importance and PDP/ALE. SHAP was omitted after a non-reproducible import stall; see `qa/interpretation_methods.md`.
''',encoding='utf8')
 (F/'limitations_factual.md').write_text('''# Factual limitations record

- Primary SSDS transport fields are only available for 2010/2020; 2020 is COVID-era.
- Exact code persistence does not prove unchanged municipal boundaries.
- Transport categories and denominator category filters change across 2000, 2010, and 2020; they are not silently pooled.
- The held-out temporal model has a compact common socioeconomic feature set (population level and age structure), rather than the full 2020 cross-sectional feature universe.
- The 37 raw socioeconomic variables are a curated e-Stat subset, not an exhaustive indicator universe. Stable item codes and raw metadata are retained even where legacy text encoding affects a label.
- Railway supply/network variables and causal policy exposures are absent; results are descriptive/associational.
''',encoding='utf8')
if __name__=='__main__':main()
