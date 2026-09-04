#!/usr/bin/env python3
"""Build the data-derived fact package used to update the paper (not the DOCX)."""
from pathlib import Path
import json
import pandas as pd

R = Path(__file__).resolve().parents[1]
A, T, Q, F = R/'analysis', R/'tables', R/'qa', R/'final'

def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))

def csv_block(df, columns=None, n=None):
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    if n:
        df = df.head(n)
    return '```csv\n' + df.to_csv(index=False, float_format='%.6f') + '```'

def main():
    F.mkdir(exist_ok=True)
    comp = pd.read_csv(T/'ssdse_core_vs_expanded.csv')
    provenance = pd.read_csv(Q/'model_feature_provenance_audit.csv')
    compliance, universe = read_json(Q/'ssdse_compliance.json'), read_json(Q/'ssdse_municipality_universe_summary.json')
    key = read_json(F/'key_numbers.json')
    oof = pd.read_csv(A/'robustness/oof_model_metrics.csv')
    importance = pd.read_csv(A/'interpretation/oof_permutation_importance.csv').head(10)
    coef = pd.read_csv(A/'bounded/fractional_logit_coefficients.csv')
    boot = pd.read_csv(A/'bootstrap/bootstrap_summary.csv')
    frac_boot = boot.loc[boot.artifact.eq('fractional_logit'), ['name','coef_p025','coef_p975']]
    frac = coef.loc[coef.specification.eq('municipality_equal')].merge(frac_boot, left_on='term', right_on='name', how='left')
    frac['CI crosses zero'] = (frac.coef_p025 <= 0) & (frac.coef_p975 >= 0)
    frac['interpretation_status'] = frac.apply(lambda r: ('robust_positive' if r.coefficient > 0 else 'robust_negative') if not r['CI crosses zero'] else ('point_estimate_positive_but_bootstrap_uncertain' if r.coefficient > 0 else 'point_estimate_negative_but_bootstrap_uncertain'), axis=1)
    temporal = pd.read_csv(A/'temporal/temporal_validation_enhanced.csv')
    residual = pd.read_csv(A/'residual/consensus_residual_consistent_outliers.csv')
    gam = pd.read_csv(A/'gam_curve.csv')

    # Core membership has its own authoritative artifact. Do not infer it from
    # the Expanded-model provenance label, which intentionally names the model
    # in which every audited feature is available.
    core_artifact = pd.read_parquet(A/'core_features.parquet')
    core = core_artifact.feature.astype(str).tolist()
    provenance_features = provenance.feature.astype(str).tolist()
    missing_core = sorted(set(core) - set(provenance_features))
    if missing_core:
        raise RuntimeError(f'Core features missing from provenance audit: {missing_core}')
    expanded = [x for x in provenance_features if x not in set(core)]
    core_row = comp.loc[comp.model.eq('SSDSE-A Core')]
    expanded_row = comp.loc[comp.model.str.contains('Expanded', case=False, na=False)]
    if core_row.empty or int(core_row.iloc[0].n_features) != len(core):
        raise RuntimeError('Core feature-count mismatch between core_features.parquet and comparison table')
    if expanded_row.empty or int(expanded_row.iloc[0].n_features) != len(provenance_features):
        raise RuntimeError('Expanded feature-count mismatch between provenance audit and comparison table')

    density = frac.loc[frac.term.eq('population_density')].iloc[0].to_dict()
    aging = frac.loc[frac.term.eq('aging_share_65plus')].iloc[0].to_dict()
    facts = {
        'dataset': {'name': compliance['dataset_name'], 'version': compliance['dataset_version'], 'official_source': compliance.get('official_source'), 'raw_sha256': compliance['raw_sha256'], 'raw_municipality_count': compliance['municipality_count_raw'], 'analysis_municipality_count': compliance['municipality_count_analysis']},
        'universe': universe,
        'feature_counts': {'old_reported': 41, 'ssdse_core': len(core), 'expanded': len(provenance_features)},
        'core_features': core,
        'expanded_additions': expanded,
        'strict_temporal': {'universe_n': key['strict_three_wave_cohort_n'], 'train_n': key['longitudinal']['temporal_validation']['n_train'], 'test_n': key['longitudinal']['temporal_validation']['n_test'], 'test_n_reason': 'Rows missing an outcome or a required 2000/2010 predictor in the held-out 2010→2020 transition are excluded; the strict cohort is not the test-row count.'},
        'core_vs_expanded': comp.to_dict(orient='records'),
        'model_comparison': oof.to_dict(orient='records'),
        'fractional_logit_population_density': density,
        'fractional_logit_aging_share_65plus': aging,
        'aging_robustness': aging['interpretation_status'],
        'population_density_robustness': density['interpretation_status'],
    }
    matrix_rows = [
        ('全国1,910市区町村', '1,910 municipalities', str(compliance['municipality_count_analysis']), 'qa/ssdse_municipality_universe_summary.json', True, 'Canonical SSDSE municipality universe and observed target exclusion'),
        ('2020横断1,910', 'N=1,910', str(compliance['municipality_count_analysis']), 'final/key_numbers.json', True, 'Regenerated canonical analysis frame'),
        ('均衡1,906', 'N=1,906', str(key['n_balanced_2010_2020']), 'final/key_numbers.json', True, 'Canonical-code 2010/2020 intersection'),
        ('41特徴量', '41', str(len(provenance_features)), 'qa/model_feature_provenance_audit.csv', True, 'Expanded model now has fully audited features'),
        ('旧GroupKFold指標', 'legacy values', 'Use regenerated OOF model comparison', 'analysis/robustness/oof_model_metrics.csv', True, 'Canonical outcome universe'),
        ('旧重要度順位', 'legacy ranking', 'Use regenerated top-10 permutation ranking', 'analysis/interpretation/oof_permutation_importance.csv', True, 'Canonical model rerun'),
        ('旧fractional logit係数', 'legacy coefficients', 'Use regenerated municipality-equal coefficients', 'analysis/bounded/fractional_logit_coefficients.csv', True, 'Canonical analysis frame'),
        ('旧bootstrap CI', 'legacy intervals', 'Use regenerated prefecture-cluster bootstrap intervals', 'analysis/bootstrap/bootstrap_summary.csv', True, 'Canonical analysis frame'),
        ('旧temporal N', 'legacy temporal N', f"strict={key['strict_three_wave_cohort_n']}; train={key['longitudinal']['temporal_validation']['n_train']}; test={key['longitudinal']['temporal_validation']['n_test']}", 'final/key_numbers.json', True, 'Strict cohort and evaluable holdout rows differ'),
        ('旧temporal metrics', 'legacy metrics', 'Use regenerated temporal validation table', 'analysis/temporal/temporal_validation_enhanced.csv', True, 'Retained exact-code historical panel'),
        ('高齢化率の「頑健な負」主張', 'robust negative', aging['interpretation_status'], 'analysis/bootstrap/bootstrap_summary.csv', True, 'Bootstrap CI must determine robustness wording'),
    ]
    matrix = pd.DataFrame(matrix_rows, columns=['paper_item','old_value_or_claim','new_value_or_claim','source_artifact','mandatory_change','reason'])
    matrix.to_csv(F/'paper_update_matrix.csv', index=False)
    (F/'paper_update_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    best = oof.sort_values('r2', ascending=False).iloc[0]
    lines = [
        '# SSDSE-A-2026 Migration Report', '',
        'This is the single generated fact source for paper revision. It does not modify the paper DOCX.', '',
        '## 1. Final dataset provenance', '',
        f"- SSDSE-A official dataset: {compliance['dataset_name']} ({compliance['dataset_version']}).",
        f"- Official source: {compliance.get('official_source', 'recorded in source metadata')}; raw SHA-256: `{compliance['raw_sha256']}`.",
        f"- Raw municipalities: {compliance['municipality_count_raw']}; 2020 analysis municipalities: {compliance['municipality_count_analysis']}.",
        '- Exclusion detail is machine-readable in `qa/ssdse_analysis_exclusions.csv`; one canonical municipality has no saved H730 outcome.', '',
        '## 2. Municipality universe migration', '',
        csv_block(pd.DataFrame([{'old_N': universe['existing_code_count'], 'new_canonical_N': universe['common_code_count'], 'common': universe['common_code_count'], 'old_only': universe['existing_only_count'], 'administrative_wards': universe['classification_counts']['designated_city_administrative_ward'], 'other_noncanonical': universe['classification_counts']['other_special_region_or_noncanonical'], 'SSDSE_only': universe['ssdse_only_count'], 'final_analysis_N': compliance['municipality_count_analysis']}])) , '',
        '## 3. Bugs corrected during SSDSE migration', '',
        '- Corrected `foreign_population_share`, `single_household_share`, and `elderly_single_household_share` numerator/denominator definitions.',
        '- Added derived-feature temporal leakage checks, year-compatible e-Stat crosschecks, full model-feature provenance verification, and independent derived-formula value checks.',
        '- Removed stale legacy balanced-N and feature-count claims and fixed Core-membership generation in the paper fact package.', '',
        '## 4. Final feature architecture', '',
        f"- Old reported feature count: 41; SSDSE Core: {len(core)}; Expanded: {len(provenance_features)}.",
        '- SSDSE Core features: ' + ', '.join(core) + '.',
        '- Expanded additions beyond Core: ' + ', '.join(expanded) + '.', '',
        '## 5. SSDSE Core vs Expanded', '', csv_block(comp), '',
        '## 6. Final model comparison', '', csv_block(oof, ['model','n','rmse','mae','r2','spearman']), '',
        f"- Best R²/RMSE model: {best['model']} (R²={best['r2']:.6f}, RMSE={best['rmse']:.6f}).", '',
        '## 7. Permutation importance', '', csv_block(importance, n=10), '',
        '## 8. Fractional logit', '', csv_block(frac, ['term','coefficient','pvalue','coef_p025','coef_p975','CI crosses zero','interpretation_status']), '',
        f"- Population density: {density['interpretation_status']}; aging share: {aging['interpretation_status']}.", '',
        '## 9. GAM / nonlinear findings', '',
        'The following saved GAM curve records are the numeric basis for any nonlinear wording:', '', csv_block(gam, n=20), '',
        '## 10. Longitudinal analysis', '',
        f"- Strict exact-code universe N={key['strict_three_wave_cohort_n']}; train N={key['longitudinal']['temporal_validation']['n_train']}; test N={key['longitudinal']['temporal_validation']['n_test']}.",
        '- Test N is smaller because held-out transition rows with a missing outcome or required predictor are not evaluable; do not call all strict-universe municipalities test observations.',
        '- SSDSE-A-2026 is not retrojected into historical years.', '', csv_block(temporal, ['model','task','n_train','n_test','rmse','mae','r2','spearman']), '',
        f"- Change-score N={key['longitudinal']['change_score_n']}; first-difference N={key['longitudinal']['first_difference_n']}.", '',
        '## 11. Residual analysis', '', f"- Analysis N={compliance['municipality_count_analysis']}; residual records={len(residual)}.",
        '- Largest negative and positive consensus residual records (municipality codes; use names only if joined to an authoritative name artifact):', '',
        csv_block(pd.concat([residual.nsmallest(5, 'consensus_residual'), residual.nlargest(5, 'consensus_residual')]), ['code','consensus_residual','residual_sign_agreement']), '',
        '## 12. Old paper -> new paper update matrix', '', csv_block(matrix), '',
        'Machine-readable companions: `final/paper_update_matrix.csv` and `final/paper_update_facts.json`.', '',
        '## QA status', '',
        '- SSDSE-A is the canonical 2020 base; e-Stat SSDS is an extension; Census provides the outcome/historical source.',
        '- Reproduce with `python scripts/run_all.py` followed by `python scripts/verify_artifacts.py`.'
    ]
    (F/'SSDSE_MIGRATION_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('wrote paper update facts, matrix, and migration report')

if __name__ == '__main__':
    main()
