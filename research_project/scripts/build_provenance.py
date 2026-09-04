#!/usr/bin/env python3
"""Build raw-input source registry and measured QA summaries."""
from pathlib import Path
import hashlib,json,platform,sys
from datetime import datetime,timezone
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1]; RAW=R/'data/raw'; C=R/'catalog';Q=R/'qa';P=R/'processed'; C.mkdir(exist_ok=True);Q.mkdir(exist_ok=True)
def h(p):
 x=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):x.update(b)
 return x.hexdigest()
def main():
 rows=[]
 sm=RAW/'ssdse_a_2026'/'source_metadata.json'
 if sm.exists():
  x=json.loads(sm.read_text(encoding='utf8')); rows.append({'request_manifest':str(sm.relative_to(R)),'raw_values_file':x.get('raw_file'),'stats_data_id':'SSDSE-A-2026','filters':'{}','retrieved_at':x.get('retrieved_at'),'rows':x.get('municipality_count'),'pages':1,'sha256':x.get('raw_sha256'),'source_dataset':'SSDSE-A-2026','source_url':x.get('official_source_url')})
 for p in RAW.rglob('*_request.json'):
  try:
   x=json.loads(p.read_text(encoding='utf8'));v=Path(x.get('values_file',''));rows.append({'request_manifest':str(p.relative_to(R)),'raw_values_file':str(v),'stats_data_id':x.get('stats_data_id'),'filters':json.dumps(x.get('filters',{}),ensure_ascii=False),'retrieved_at':x.get('retrieved_at'),'rows':x.get('rows'),'pages':x.get('pages'),'sha256':h(v) if v.exists() else None})
  except Exception as e: rows.append({'request_manifest':str(p.relative_to(R)),'error':str(e)})
 for p in RAW.rglob('requests/**/*.json'):
  try:
   x=json.loads(p.read_text(encoding='utf8'));rows.append({'request_manifest':str(p.relative_to(R)),'raw_values_file':x.get('values_file'),'stats_data_id':x.get('stats_data_id'),'filters':json.dumps(x.get('filters',{}),ensure_ascii=False),'retrieved_at':x.get('retrieved_at'),'rows':x.get('downloaded_rows'),'pages':x.get('pages'),'sha256':None})
  except:pass
 pd.DataFrame(rows).drop_duplicates('request_manifest').to_parquet(C/'source_registry.parquet',index=False)
 d=pd.read_parquet(P/'municipality_panel.parquet'); keydup=int(d.duplicated(['code','year']).sum()); num=d.select_dtypes('number'); out=[]
 for c in num:
  x=num[c];out.append({'feature':c,'n':int(x.notna().sum()),'missing_rate':float(x.isna().mean()),'min':x.min(),'p01':x.quantile(.01),'median':x.median(),'p99':x.quantile(.99),'max':x.max(),'outlier_iqr_count':int(((x<x.quantile(.25)-3*(x.quantile(.75)-x.quantile(.25)))|(x>x.quantile(.75)+3*(x.quantile(.75)-x.quantile(.25)))).sum())})
 pd.DataFrame(out).to_csv(Q/'numeric_outlier_audit.csv',index=False)
 byyear=d.groupby('year')[num.columns].agg(lambda x:x.isna().mean()).T.reset_index().rename(columns={'index':'feature'});byyear.to_csv(Q/'missingness_by_year.csv',index=False)
 (Q/'integrity_audit.json').write_text(json.dumps({'panel_rows':len(d),'duplicate_code_year_keys':keydup,'raw_registry_records':len(rows),'generated_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform()},ensure_ascii=False,indent=2),encoding='utf8')
 # JSON standard forbids NaN; normalize legacy numeric outputs for downstream reproducibility.
 kp=R/'final/key_numbers.json'
 if kp.exists():
  def clean(v):
   if isinstance(v,float) and not np.isfinite(v): return None
   if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
   if isinstance(v,list): return [clean(x) for x in v]
   return v
  kp.write_text(json.dumps(clean(json.loads(kp.read_text(encoding='utf8'))),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
if __name__=='__main__':main()
