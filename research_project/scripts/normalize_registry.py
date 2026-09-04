#!/usr/bin/env python3
"""Collapse repeated identical legacy runs while retaining the newest record."""
from pathlib import Path
import pandas as pd,json
R=Path(__file__).resolve().parents[1];p=R/'runs/experiment_registry.parquet'
def main():
 d=pd.read_parquet(p);d['_t']=pd.to_datetime(d.timestamp,errors='coerce');keys=['model','split_method','feature_set','outcome','period','dataset_hash'];d=d.sort_values('_t').drop_duplicates(keys,keep='last').drop(columns='_t')
 # Legacy IDs are upgraded deterministically when necessary.
 seen=set();ids=[]
 for i,r in d.iterrows():
  base=str(r.run_id).split('_')[0] or 'run';rid=str(r.run_id)
  if rid in seen:rid=f'{base}_{i:03d}'
  seen.add(rid);ids.append(rid)
 d['run_id']=ids;d.to_parquet(p,index=False)
 for col,default in [('parameters',json.dumps({'legacy_parameters_unavailable':True})),('selection_reason','legacy imported run')]:
  d[col]=d[col].fillna(default).replace({'{}':default,'' :default})
 d.to_parquet(p,index=False)
 (R/'runs/registry_summary.json').write_text(json.dumps({'unique_runs':len(d),'duplicate_run_ids':int(d.run_id.duplicated().sum()),'columns':list(d.columns)},ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
