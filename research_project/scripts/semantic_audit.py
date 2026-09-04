#!/usr/bin/env python3
"""Verify raw labels and independently recompute SSDSE derived features."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]; C=R/'catalog'; Q=R/'qa'; P=R/'processed'; Q.mkdir(exist_ok=True)
DERIVED={
 'log_population':(['A1101'],'log1p(A1101)'),
 'aging_share_65plus':(['A1303','A1101'],'A1303 / A1101'),
 'share_75plus':(['A1419','A1101'],'A1419 / A1101'),
 'youth_share_0_14':(['A1301','A1101'],'A1301 / A1101'),
 'working_age_share_15_64':(['A1302','A1101'],'A1302 / A1101'),
 'foreign_population_share':(['A1700','A1101'],'A1700 / A1101'),
 'single_household_share':(['A810105','A710101'],'A810105 / A710101'),
 'elderly_single_household_share':(['A8301','A710101'],'A8301 / A710101'),
 'unemployment_rate':(['F1107','F1102'],'F1107 / (F1102 + F1107)'),
 'non_labor_force_share':(['F1108','F1102','F1107'],'F1108 / (F1102 + F1107 + F1108)'),
 'primary_industry_share':(['F2201','F2211','F2221'],'F2201 / (F2201 + F2211 + F2221)'),
 'secondary_industry_share':(['F2201','F2211','F2221'],'F2211 / (F2201 + F2211 + F2221)'),
 'tertiary_industry_share':(['F2201','F2211','F2221'],'F2221 / (F2201 + F2211 + F2221)'),
}

def div(n,d): return n/d.where(d>0)
def expected(frame,name):
 if name=='log_population': return np.log1p(frame.A1101.clip(lower=0))
 if name=='aging_share_65plus': return div(frame.A1303,frame.A1101)
 if name=='share_75plus': return div(frame.A1419,frame.A1101)
 if name=='youth_share_0_14': return div(frame.A1301,frame.A1101)
 if name=='working_age_share_15_64': return div(frame.A1302,frame.A1101)
 if name=='foreign_population_share': return div(frame.A1700,frame.A1101)
 if name=='single_household_share': return div(frame.A810105,frame.A710101)
 if name=='elderly_single_household_share': return div(frame.A8301,frame.A710101)
 if name=='unemployment_rate': return div(frame.F1107,frame.F1102+frame.F1107)
 if name=='non_labor_force_share': return div(frame.F1108,frame.F1102+frame.F1107+frame.F1108)
 denom=frame.F2201+frame.F2211+frame.F2221
 if name=='primary_industry_share': return div(frame.F2201,denom)
 if name=='secondary_industry_share': return div(frame.F2211,denom)
 if name=='tertiary_industry_share': return div(frame.F2221,denom)
 raise KeyError(name)

def main():
 d=pd.read_parquet(C/'feature_dictionary.parquet'); raw=d[d.raw_or_derived.eq('raw')].copy()
 bad=raw[(raw.official_item_name_raw.isna())|(raw.label_safe.isna())|(raw.official_item_name_raw!=raw.label_safe)]
 frame=pd.read_parquet(P/'municipality_cross_section_2020.parquet')
 recorded=json.loads((Q/'ssdse_derived_feature_audit.json').read_text(encoding='utf8'))
 rec={r['derived_feature']:r for r in recorded}
 checks=[]
 for name,(sources,formula) in DERIVED.items():
  r=rec.get(name,{})
  definition_ok=(r.get('formula')==formula and list(r.get('source_features',[]))==sources)
  missing=[c for c in [*sources,name] if c not in frame.columns]
  if missing:
   mismatch_count=len(frame); max_abs_diff=None
  else:
   exp=pd.to_numeric(expected(frame,name),errors='coerce').to_numpy(dtype=float)
   act=pd.to_numeric(frame[name],errors='coerce').to_numpy(dtype=float)
   same=np.isclose(act,exp,rtol=1e-10,atol=1e-12,equal_nan=True)
   mismatch_count=int((~same).sum())
   finite=np.isfinite(act)&np.isfinite(exp)
   max_abs_diff=float(np.max(np.abs(act[finite]-exp[finite]))) if finite.any() else 0.0
  checks.append({'feature':name,'expected_sources':sources,'expected_formula':formula,'definition_ok':definition_ok,'value_mismatch_count':mismatch_count,'max_absolute_difference':max_abs_diff,'missing_columns':missing,'status':'pass' if definition_ok and mismatch_count==0 and not missing else 'fail'})
 definition_bad=sum(not x['definition_ok'] for x in checks); value_bad=sum(x['value_mismatch_count'] for x in checks)
 out={'status':'pass' if bad.empty and definition_bad==0 and value_bad==0 else 'fail','raw_feature_count':int(len(raw)),'semantic_mismatch_count':int(len(bad)),'derived_feature_count':len(checks),'derived_formula_definition_mismatch_count':int(definition_bad),'derived_formula_value_mismatch_count':int(value_bad),'display_rule':'For raw SSDS features label_safe must equal official_item_name_raw from preserved metadata. Derived SSDSE features are independently recomputed from their recorded source columns and checked against expected definitions.','mismatches':bad.feature.tolist(),'derived_checks':checks}
 (Q/'feature_semantic_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 if out['status']!='pass': raise SystemExit(json.dumps(out,ensure_ascii=False))
 print(json.dumps({'status':out['status'],'raw_mismatches':len(bad),'derived_definition_mismatches':definition_bad,'derived_value_mismatches':value_bad},ensure_ascii=False))
if __name__=='__main__':main()
