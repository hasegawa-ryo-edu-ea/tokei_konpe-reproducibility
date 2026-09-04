#!/usr/bin/env python3
"""Fail closed final gate for generated SSDSE analysis artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
REQUIRED = [
    'data/raw/ssdse_a_2026/SSDSE-A-2026.csv', 'data/raw/ssdse_a_2026/source_metadata.json',
    'data/raw/ssdse_a_2026/retrieval_record.json',
    'processed/municipality_panel.parquet', 'processed/municipality_cross_section_2020.parquet',
    'processed/municipality_panel_balanced.parquet', 'processed/municipality_panel_stable_boundary.parquet',
    'analysis/core_features.parquet', 'analysis/ml_features.parquet', 'analysis/robustness/oof_predictions_all_models.parquet',
    'tables/ssdse_core_vs_expanded.csv', 'final/key_numbers.json', 'final/SSDSE_MIGRATION_REPORT.md',
    'final/paper_update_matrix.csv', 'final/paper_update_facts.json',
    'qa/ssdse_compliance.json', 'qa/ssdse_temporal_leakage.json', 'qa/ssdse_derived_feature_audit.json',
    'qa/ssdse_estat_crosscheck_summary.json', 'qa/ssdse_municipality_universe_summary.json',
    'qa/model_feature_provenance_audit.json', 'qa/model_feature_provenance_audit.csv',
    'qa/artifact_consistency.json', 'qa/feature_semantic_audit.json', 'qa/temporal_target_definition.json',
    'runs/experiment_registry.parquet',
]

def load(relative):
    with (R / relative).open(encoding='utf8') as fh:
        return json.load(fh)

def main():
    missing = [p for p in REQUIRED if not (R / p).exists()]
    if missing:
        raise SystemExit('Missing artifacts: ' + ', '.join(missing))
    panel = pd.read_parquet(R / 'processed/municipality_panel.parquet')
    cross_frame = pd.read_parquet(R / 'processed/municipality_cross_section_2020.parquet')
    balanced = pd.read_parquet(R / 'processed/municipality_panel_balanced.parquet')
    strict = pd.read_parquet(R / 'processed/municipality_panel_stable_boundary.parquet')
    core_features = pd.read_parquet(R / 'analysis/core_features.parquet')
    ml_features = pd.read_parquet(R / 'analysis/ml_features.parquet')
    oof = pd.read_parquet(R / 'analysis/robustness/oof_predictions_all_models.parquet')
    comp = pd.read_csv(R / 'tables/ssdse_core_vs_expanded.csv')
    runs = pd.read_parquet(R / 'runs/experiment_registry.parquet')
    key = load('final/key_numbers.json')
    facts = load('final/paper_update_facts.json')
    report = (R/'final/SSDSE_MIGRATION_REPORT.md').read_text(encoding='utf8')
    source_meta = load('data/raw/ssdse_a_2026/source_metadata.json')
    retrieval = load('data/raw/ssdse_a_2026/retrieval_record.json')
    compliance, leak = load('qa/ssdse_compliance.json'), load('qa/ssdse_temporal_leakage.json')
    derived, crosscheck = load('qa/ssdse_derived_feature_audit.json'), load('qa/ssdse_estat_crosscheck_summary.json')
    universe, provenance = load('qa/ssdse_municipality_universe_summary.json'), load('qa/model_feature_provenance_audit.json')
    consistency, semantic = load('qa/artifact_consistency.json'), load('qa/feature_semantic_audit.json')
    target = load('qa/temporal_target_definition.json')
    oof_n = int(oof['code'].nunique()) if 'code' in oof else int(len(oof))
    cross_n = int(cross_frame['code'].nunique())
    balanced_n = int(balanced['code'].nunique())
    strict_n = int(strict['code'].nunique())
    core_n = int(len(core_features))
    ml_n = int(len(ml_features))
    core_list = core_features['feature'].astype(str).tolist()
    expanded_list = ml_features['feature'].astype(str).tolist()
    comp_core = comp.loc[comp.model.eq('SSDSE-A Core'),'n_features']
    comp_expanded = comp.loc[comp.model.str.contains('Expanded',case=False,na=False),'n_features']
    comp_core_n = int(comp_core.iloc[0]) if len(comp_core)==1 else -1
    comp_expanded_n = int(comp_expanded.iloc[0]) if len(comp_expanded)==1 else -1
    facts_core_n = int(facts.get('feature_counts',{}).get('ssdse_core',-1))
    facts_expanded_n = int(facts.get('feature_counts',{}).get('expanded',-1))
    report_token = f'SSDSE Core: {core_n}; Expanded: {ml_n}.'
    facts_core = [str(x) for x in facts.get('core_features', [])]
    facts_expanded = facts_core + [str(x) for x in facts.get('expanded_additions', [])]
    core_ok = (core_n==comp_core_n==facts_core_n==29 and ml_n==comp_expanded_n==facts_expanded_n==69 and
               core_list==facts_core and expanded_list==facts_expanded and report_token in report)
    metric_columns = ['rmse', 'mae', 'r2', 'spearman']
    metric_frames = [comp, pd.read_csv(R / 'analysis/robustness/oof_model_metrics.csv')]
    metrics_finite = all(not frame.empty and all(
        column in frame and np.isfinite(pd.to_numeric(frame[column], errors='coerce')).all()
        for column in metric_columns
    ) for frame in metric_frames)
    retrieval_ok = source_meta.get('retrieved_at')==retrieval.get('retrieved_at') and source_meta.get('raw_sha256')==retrieval.get('raw_sha256')
    checks = {
        'ssdse_compliance_status': compliance.get('status'),
        'ssdse_temporal_leakage_status': leak.get('status'),
        'ssdse_crosscheck_status': crosscheck.get('status'),
        'ssdse_universe_status': universe.get('status'),
        'model_feature_provenance_status': provenance.get('status'),
        'derived_semantic_status': semantic.get('status'),
        'core_membership_consistency_status': 'pass' if core_ok else 'fail',
        'performance_metrics_finite_status': 'pass' if metrics_finite else 'fail',
        'source_metadata_retrieval_status': 'pass' if retrieval_ok else 'fail',
        'artifact_consistency_status': consistency.get('status'),
        'feature_semantic_status': semantic.get('status'),
        'temporal_target_definition_status': target.get('status'),
        'raw_future_feature_count': leak.get('raw_future_feature_count'),
        'derived_future_feature_count': leak.get('derived_future_feature_count'),
        'future_feature_count': leak.get('future_feature_count'),
        'crosscheck_items_with_mismatch': crosscheck.get('items_with_mismatch'),
        'untracked_model_feature_count': provenance.get('untracked_model_feature_count'),
        'temporally_unsafe_model_feature_count': provenance.get('temporally_unsafe_model_feature_count'),
        'unknown_year_feature_count': provenance.get('unknown_year_feature_count'),
        'outcome_leak_feature_count': provenance.get('outcome_leak_feature_count'),
        'derived_formula_definition_mismatch_count': semantic.get('derived_formula_definition_mismatch_count'),
        'derived_formula_value_mismatch_count': semantic.get('derived_formula_value_mismatch_count'),
        'panel_duplicate_keys': int(panel.duplicated(['code', 'year']).sum()),
        'registry_duplicate_run_ids': int(runs.run_id.duplicated().sum()),
        'cross_section_n': cross_n, 'balanced_n': balanced_n, 'strict_temporal_n': strict_n,
        'ssdse_core_feature_count': core_n, 'ml_feature_count': ml_n, 'oof_n': oof_n,
        'core_vs_expanded_core_feature_count': comp_core_n,
        'core_vs_expanded_expanded_feature_count': comp_expanded_n,
        'paper_facts_core_feature_count': facts_core_n,
        'paper_facts_expanded_feature_count': facts_expanded_n,
        'key_numbers_cross_section_n': key.get('n_cross_section_2020'),
        'key_numbers_balanced_n': key.get('n_balanced_2010_2020'),
        'key_numbers_ml_feature_count': key.get('n_features_ml'),
    }
    status_keys = [k for k in checks if k.endswith('_status')]
    zero_keys = ['raw_future_feature_count', 'derived_future_feature_count', 'future_feature_count',
                 'crosscheck_items_with_mismatch', 'untracked_model_feature_count',
                 'temporally_unsafe_model_feature_count', 'unknown_year_feature_count',
                 'outcome_leak_feature_count', 'derived_formula_definition_mismatch_count',
                 'derived_formula_value_mismatch_count', 'panel_duplicate_keys', 'registry_duplicate_run_ids']
    n_ok = (oof_n == cross_n and key.get('n_cross_section_2020') == cross_n and
            key.get('n_balanced_2010_2020') == balanced_n and key.get('n_features_ml') == ml_n)
    passed = all(checks[k] == 'pass' for k in status_keys) and all(checks[k] == 0 for k in zero_keys) and n_ok
    result = {'status': 'pass' if passed else 'fail', **checks}
    (R / 'qa/final_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
