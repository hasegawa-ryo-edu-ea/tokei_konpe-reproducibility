#!/usr/bin/env python3
"""Fail-closed provenance audit for every Expanded-model feature."""
from pathlib import Path
import gzip,json, pandas as pd
R=Path(__file__).resolve().parents[1]; A=R/'analysis'; C=R/'catalog'; Q=R/'qa'; Q.mkdir(exist_ok=True)
OUTCOME={'rail_any_share','rail_users','commuters','rail_pp','log_commuters','oof_prediction','residual'}
def saved_ssds_years():
 out={}
 for p in (R/'data/raw/ssds_core').glob('*.jsonl.gz'):
  table=p.name.split('_')[0]
  with gzip.open(p,'rt',encoding='utf8') as fh:
   for line in fh:
    z=json.loads(line); z=z.get('attributes',z); code=z.get('@cat01',z.get('cat01')); year=str(z.get('@time',z.get('time','')))[:4]
    if code and year.isdigit(): out.setdefault(code,set()).add((int(year),table))
 return out
def main():
 f=pd.read_parquet(A/'ml_features.parquet').feature.astype(str).tolist(); core=set(pd.read_parquet(A/'core_features.parquet').feature.astype(str).tolist()); d=pd.read_parquet(C/'feature_dictionary.parquet').drop_duplicates('feature').set_index('feature')
 raw_years=saved_ssds_years(); rows=[]
 for x in f:
  q=d.loc[x] if x in d.index else None; src='' if q is None else str(q.source_dataset); year=None if q is None or pd.isna(q.reference_year) else int(q.reference_year); table=None if q is None else q.stat_table_id
  if year is None and x in raw_years and 2020 in {v[0] for v in raw_years[x]}:
   year=2020; table=';'.join(sorted({v[1] for v in raw_years[x] if v[0]==2020})); src='e-Stat SSDS municipal data (saved raw response)'
  if x=='population_density':
   year=2020; table='e-Stat SSDS saved raw A1101/B1101'; src='e-Stat SSDS municipal data'
  if year is None and 'SSDSE-A-2026 derived' in src:
   year=2020; table='SSDSE-A-2026 item lineage'
  unresolved=('e-Stat SSDS' in src and year is None) or q is None
  outcome=x in OUTCOME or x.startswith('H730')
  rows.append({'feature':x,'feature_set':'SSDSE-A + e-Stat SSDS Expanded','in_ssdse_core':x in core,'in_expanded':True,'source_dataset':src,'source_table_or_dataset_id':table,'source_item_codes':'A1101;B1101' if x=='population_density' else (None if q is None else q.item_code),'source_reference_years':'' if year is None else str(year),'max_source_reference_year':year,'observation_year':2020,'raw_or_derived':None if q is None else q.raw_or_derived,'formula':'A1101 / B1101' if x=='population_density' else (None if q is None else q.formula),'temporal_safe':False if unresolved else year is None or year<=2020,'outcome_related':outcome,'provenance_complete':not unresolved,'provenance_status':'unresolved' if unresolved else 'pass','notes':'Year is read from preserved SSDS raw @time where applicable; population_density uses saved 2020 SSDS components.'})
 out=pd.DataFrame(rows); out.to_csv(Q/'model_feature_provenance_audit.csv',index=False,encoding='utf-8-sig')
 missing_core=core-set(f)
 summary={'status':'pass' if out.provenance_complete.all() and out.temporal_safe.all() and not out.outcome_related.any() and not missing_core else 'fail','model_feature_count':len(out),'core_feature_count':int(out.in_ssdse_core.sum()),'expanded_feature_count':int(out.in_expanded.sum()),'untracked_core_feature_count':len(missing_core),'provenance_complete_count':int(out.provenance_complete.sum()),'untracked_model_feature_count':int((~out.provenance_complete).sum()),'temporally_safe_count':int(out.temporal_safe.sum()),'temporally_unsafe_model_feature_count':int((~out.temporal_safe).sum()),'unknown_year_feature_count':int(out.max_source_reference_year.isna().sum()),'outcome_leak_feature_count':int(out.outcome_related.sum())}
 (Q/'model_feature_provenance_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); print(json.dumps(summary))
 if summary['status']!='pass': raise SystemExit(1)
if __name__=='__main__':main()
