#!/usr/bin/env python3
"""Fail when factual records, model artifacts, and panel counts disagree."""
from pathlib import Path
import json,re,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];P=R/'processed';A=R/'analysis';F=R/'final';Q=R/'qa';Q.mkdir(exist_ok=True)
def m(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
 return {'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.abs(e).mean()),'r2':float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),'spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman'))}
def close(a,b): return np.isfinite(a) and np.isfinite(b) and abs(a-b)<1e-9
def main():
 errors=[]; cs=pd.read_parquet(P/'municipality_cross_section_2020.parquet'); n=int(len(cs)); key=json.loads((F/'key_numbers.json').read_text(encoding='utf8')); factual=(F/'results_factual.md').read_text(encoding='utf8')
 if key.get('n_cross_section_2020')!=n:errors.append('key_numbers n_cross_section_2020 mismatch')
 if 'cross_oof_metrics' in key or 'temporal_diagnostic_metrics' in key: errors.append('legacy metric block remains in key_numbers')
 canonical=pd.DataFrame(key.get('canonical_oof_metrics',[]))
 if canonical.empty: errors.append('canonical_oof_metrics missing from key_numbers')
 q=re.search(r'2020 cross-section: (\d+) municipalities',factual)
 if not q or int(q.group(1))!=n:errors.append('results_factual 2020 N mismatch')
 o=pd.read_parquet(A/'robustness/oof_predictions_all_models.parquet')
 if len(o)!=n or o.code.duplicated().any():errors.append('OOF rows/code keys mismatch')
 metrics=pd.read_csv(A/'robustness/oof_model_metrics.csv').set_index('model')
 for name in metrics.index:
  calc=m(o.observed,o[name+'_prediction'])
  for k,v in calc.items():
   if not close(float(metrics.loc[name,k]),v):errors.append(f'OOF {name} {k} mismatch')
  if not canonical.empty:
   for name in metrics.index:
    rr=canonical[canonical.model==name]
    if len(rr)!=1:errors.append(f'key canonical OOF row mismatch {name}')
    else:
     for k in ['rmse','mae','r2','spearman','mean_bias']:
      if not close(float(rr.iloc[0][k]),float(metrics.loc[name,k])):errors.append(f'key canonical {name} {k} mismatch')
 # Existing benchmark tables must agree with the independently regenerated OOF metrics.
 legacy=pd.concat([pd.read_csv(A/'sensitivity/model_performance.csv'),pd.read_csv(A/'ml/xgboost_metrics.csv'),pd.read_csv(A/'ml/catboost_metrics.csv')],ignore_index=True)
 mapping={'extra_trees':'extra_trees','hist_gbdt':'hist_gbdt','elastic_net':'elastic_net','xgboost':'xgboost','catboost':'catboost'}
 for legacy_name,oof_name in mapping.items():
  r=legacy[(legacy.model==legacy_name)&(legacy.split=='prefecture_group_kfold')].iloc[0]
  for k in ['rmse','mae','r2','spearman']:
   if not close(float(r[k]),float(metrics.loc[oof_name,k])):errors.append(f'legacy {legacy_name} {k} mismatch')
 temp=pd.read_csv(A/'temporal/temporal_validation_enhanced.csv'); tp=pd.read_parquet(A/'temporal/temporal_enhanced_predictions.parquet')
 for _,r in temp.iterrows():
  if r.model in tp:
   calc=m(tp.observed_2020,tp[r.model])
   for k,v in calc.items():
    if not close(float(r[k]),v): errors.append(f'temporal {r.model} {k} mismatch')
 hashes=pd.read_parquet(R/'runs/experiment_registry.parquet').dataset_hash.dropna().astype(str)
 if not hashes.map(lambda x:len(x)==64 and all(c in '0123456789abcdef' for c in x.lower())).all():errors.append('invalid experiment dataset hash')
 out={'status':'pass' if not errors else 'fail','n_cross_section_2020':n,'n_oof':int(len(o)),'checked_oof_models':metrics.index.tolist(),'checked_temporal_models':temp.model.tolist(),'errors':errors}
 (Q/'artifact_consistency.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 (Q/'metric_audit.json').write_text(json.dumps({'status':'pass' if not errors else 'fail','implementation':'All checked metrics use position-aligned numpy arrays before Spearman construction.','checked_models':out['checked_oof_models'],'checked_temporal_models':out['checked_temporal_models'],'errors':errors},ensure_ascii=False,indent=2),encoding='utf8')
 if errors: raise SystemExit('; '.join(errors))
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
