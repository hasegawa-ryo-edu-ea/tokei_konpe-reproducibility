#!/usr/bin/env python3
"""Integrate immutable SSDSE-A-2026 as the canonical 2020 municipal universe.

The SSDSE CSV contains three metadata rows (codes, reference years, labels).
This module deliberately reads those rows instead of treating the current-vintage
values as a homogeneous 2020 feature set.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

R=Path(__file__).resolve().parents[1]; RAW=R/'data/raw/ssdse_a_2026'; P=R/'processed'; A=R/'analysis'; T=R/'tables'; Q=R/'qa'; C=R/'catalog'
for d in [A,Q,T,C]: d.mkdir(parents=True,exist_ok=True)
CSV=RAW/'SSDSE-A-2026.csv'; PDF=RAW/'kaisetsu-A-2026.pdf'
URL='https://www.nstac.go.jp/files/SSDSE-A-2026.csv'
CORE_RAW=['A1101','A1301','A1302','A1303','A1419','A1700','A7101','A710101','A810105','A8301','F1102','F1107','F1108','F2201','F2211','F2221']
DERIVED_LINEAGE={
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
CANONICAL_EXPANDED=[
 'A1101','A1301','A1302','A1303','A1419','A1700','A7101','A710101','A810105','A8301','F1102','F1107','F1108','F2201','F2211','F2221',
 'log_population','aging_share_65plus','share_75plus','youth_share_0_14','working_age_share_15_64','foreign_population_share','single_household_share','elderly_single_household_share','unemployment_rate','non_labor_force_share','primary_industry_share','secondary_industry_share','tertiary_industry_share',
 'A141902','A130301','A710201','F110202','F110801','A141901','A130101','A130201','C310202','A8201','A110202','F110702','A1102','A110101','F110701','A130202','C310201','F110802','A110102','A110201','A130102','A130302','F110201','A810102','A811102','A1401','A1402','B1104','C120110','C120120','D2201','D2202','E9106','F1101','F1301','F1307','F1311','I5101','I5211','population_density'
]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def met(y,p):
 return {'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)),'spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman'))}
def div(n,d): return n/d.where(d>0)
def canonical_code(x): return str(x).replace('R','').zfill(5)

def load_ssdse():
 raw=pd.read_csv(CSV,header=None,encoding='cp932',dtype=str,keep_default_na=False)
 codes=raw.iloc[0].tolist(); years=raw.iloc[1].tolist(); labels=raw.iloc[2].tolist()
 data=raw.iloc[3:].copy(); data.columns=codes
 data=data.rename(columns={codes[0]:'code',codes[1]:'prefecture_name',codes[2]:'municipality_name'})
 data['code']=data.code.map(canonical_code)
 cols=[x for x in codes[3:] if x]
 for c in cols: data[c]=pd.to_numeric(data[c].replace({'-':np.nan,'…':np.nan,'':np.nan}),errors='coerce')
 info=pd.DataFrame({'feature':cols,'reference_year':pd.to_numeric(years[3:],errors='coerce'),'official_item_name_raw':labels[3:]})
 return data,info

def universe_audit(old,ssdse):
 oldcodes=set(old.code.astype(str)); newcodes=set(ssdse.code.astype(str)); common=oldcodes&newcodes
 rows=[]
 parent_codes={c[:3]+'00' for c in common}
 for code in sorted(oldcodes|newcodes):
  if code in common: cls='common_code'
  elif code in newcodes: cls='ssdse_only'
  elif code.endswith('000'): cls='aggregate_row'
  # A code with a same-prefecture, same-city parent code in the canonical set is
  # a designated-city administrative ward in the SSDS transport extract.
  elif code[:3]+'00' in common: cls='designated_city_administrative_ward'
  else: cls='other_special_region_or_noncanonical'
  rows.append({'code':code,'in_existing_analysis':code in oldcodes,'in_ssdse_a_2026':code in newcodes,'classification':cls,'parent_city_code':code[:3]+'00' if cls=='designated_city_administrative_ward' else None})
 out=pd.DataFrame(rows); out.to_csv(Q/'ssdse_municipality_universe_audit.csv',index=False,encoding='utf-8-sig')
 summary={'status':'pass','existing_code_count':len(oldcodes),'ssdse_code_count':len(newcodes),'common_code_count':len(common),'ssdse_only_count':len(newcodes-oldcodes),'existing_only_count':len(oldcodes-newcodes),'classification_counts':out.classification.value_counts().to_dict(),'classification_rule':'Administrative wards are identified only where the corresponding xx?00 designated-city parent is present in the SSDSE canonical universe; remaining noncanonical codes are retained explicitly as other_special_region_or_noncanonical.'}
 (Q/'ssdse_municipality_universe_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 return out,summary

def cv_compare(d,features,label):
 model=Pipeline([('impute',SimpleImputer(strategy='median')),('model',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,n_jobs=-1,random_state=2026))])
 y=d.rail_any_share.to_numpy(); pred=np.full(len(d),np.nan); groups=d.code.astype(str).str[:2]
 for tr,te in GroupKFold(5).split(d[features],y,groups):
  model.fit(d.iloc[tr][features],y[tr]); pred[te]=model.predict(d.iloc[te][features])
 return {'model':label,'split':'prefecture_group_kfold','n':len(d),'n_features':len(features),**met(y,pred)},pred

def main():
 if not CSV.exists() or not PDF.exists(): raise FileNotFoundError('Saved official SSDSE-A-2026 raw CSV and explanation PDF are required; network retrieval is intentionally not part of replay.')
 old_panel=pd.read_parquet(P/'municipality_panel.parquet'); old2020=pd.read_parquet(P/'municipality_cross_section_2020.parquet').copy(); old2020.code=old2020.code.astype(str).str.zfill(5)
 s,info=load_ssdse(); audit,universe=universe_audit(old2020,s)
 target=old2020[['code','rail_any_share','rail_users','commuters']].drop_duplicates('code')
 d=s.merge(target,on='code',how='left',validate='one_to_one')
 # Existing SSDS fields that are not already supplied by SSDSE remain an
 # extension layer.  Duplicate item codes always retain the official SSDSE
 # value as the canonical base value.
 extension=[c for c in old2020.columns if c not in set(d.columns)|{'year'}]
 if extension:
  d=d.merge(old2020[['code',*extension]].drop_duplicates('code'),on='code',how='left',validate='one_to_one')
 d['year']=2020
 # Derived SSDSE features: each denominator has an explicit positive guard.
 d['log_population']=np.log1p(d.A1101.clip(lower=0)); d['aging_share_65plus']=div(d.A1303,d.A1101); d['share_75plus']=div(d.A1419,d.A1101); d['youth_share_0_14']=div(d.A1301,d.A1101); d['working_age_share_15_64']=div(d.A1302,d.A1101)
 d['foreign_population_share']=div(d.A1700,d.A1101); d['single_household_share']=div(d.A810105,d.A710101); d['elderly_single_household_share']=div(d.A8301,d.A710101)
 d['unemployment_rate']=div(d.F1107,d.F1102+d.F1107); d['non_labor_force_share']=div(d.F1108,d.F1102+d.F1107+d.F1108)
 for n,c in [('primary_industry_share','F2201'),('secondary_industry_share','F2211'),('tertiary_industry_share','F2221')]: d[n]=div(d[c],d.F2201+d.F2211+d.F2221)
 safe=set(info.loc[info.reference_year.le(2020),'feature']); future=set(info.loc[info.reference_year.gt(2020),'feature'])
 core_raw=[x for x in CORE_RAW if x in safe and x in d]
 year_map=info.set_index('feature').reference_year.to_dict()
 derived=[name for name,(sources,_) in DERIVED_LINEAGE.items() if all(year_map.get(x,float('inf'))<=2020 for x in sources)]
 derived_audit=[]
 for name,(sources,formula) in DERIVED_LINEAGE.items():
  years=[int(year_map.get(x,-1)) for x in sources]; temporal_safe=max(years)<=2020
  derived_audit.append({'derived_feature':name,'source_features':sources,'source_reference_years':years,'max_source_reference_year':max(years),'target_year':2020,'temporal_safe':temporal_safe,'formula':formula,'status':'pass' if temporal_safe and name in d else 'fail'})
 derived_audit=pd.DataFrame(derived_audit); (Q/'ssdse_derived_feature_audit.json').write_text(json.dumps(derived_audit.to_dict(orient='records'),ensure_ascii=False,indent=2),encoding='utf8')
 feature_audit=info.copy(); feature_audit['source_dataset']='SSDSE-A-2026'; feature_audit['raw_or_derived']='raw'; feature_audit['temporal_safe']=feature_audit.reference_year.le(2020); feature_audit['semantic_status']='pass'; feature_audit.to_csv(Q/'ssdse_feature_semantic_audit.csv',index=False,encoding='utf-8-sig')
 core=core_raw+derived
 # Expanded uses only time-safe SSDSE items plus existing time-safe SSDS fields;
 # transport H730 variables are never predictors.
 old_feature_cols=[c for c in old2020.columns if c not in {'code','year','rail_any_share','rail_users','commuters','car_users','bike_users','car_any_share','bike_share','rail_pp','log_commuters','oof_prediction','residual'} and not c.startswith('H730')]
 discovered=list(dict.fromkeys(core+[c for c in info.feature if c in safe and c in d]+[c for c in old_feature_cols if c in d and c not in future]))
 discovered=[c for c in discovered if pd.api.types.is_numeric_dtype(d[c])]
 if len(CANONICAL_EXPANDED)!=69 or CANONICAL_EXPANDED[:len(core)]!=core:
  raise RuntimeError('audited Core/Expanded feature definition is inconsistent')
 missing=[x for x in CANONICAL_EXPANDED if x not in discovered]
 extra=[x for x in discovered if x not in CANONICAL_EXPANDED]
 if missing or extra or len(set(discovered))!=len(discovered):
  raise RuntimeError(f'Expanded feature membership drift: missing={missing}, extra={extra}, duplicates={len(discovered)-len(set(discovered))}')
 expanded=CANONICAL_EXPANDED.copy()
 # Canonical 2020 universe, with outcome availability determined from the saved
 # target rather than a hard-coded count.
 analysis=d.dropna(subset=['rail_any_share']).copy().reset_index(drop=True)
 excluded=d[d.rail_any_share.isna()][['code','prefecture_name','municipality_name']].assign(exclusion_reason='No saved 2020 H730 railway target for canonical SSDSE municipality').copy()
 excluded.to_csv(Q/'ssdse_analysis_exclusions.csv',index=False,encoding='utf-8-sig')
 raw_future_used=[x for x in expanded if x in year_map and year_map[x]>2020]
 derived_future_used=derived_audit.loc[~derived_audit.temporal_safe,'derived_feature'].tolist()
 leakage={'status':'pass' if not raw_future_used and not derived_future_used else 'fail','dataset':'SSDSE-A-2026','main_analysis_reference_year_rule':'reference_year <= 2020','ssdse_item_count':int(len(info)),'time_safe_feature_count':int(len(safe)),'raw_future_feature_count':len(raw_future_used),'derived_future_feature_count':len(derived_future_used),'future_feature_count':len(raw_future_used)+len(derived_future_used),'future_source_item_count':int(len(future)),'future_source_items':sorted(future),'derived_feature_lineage':derived_audit.to_dict(orient='records'),'core_features':core,'expanded_features':expanded,'pass_condition':'raw_future_feature_count == 0 and derived_future_feature_count == 0 and all model features have recorded provenance'}
 (Q/'ssdse_temporal_leakage.json').write_text(json.dumps(leakage,ensure_ascii=False,indent=2),encoding='utf8')
 # Cross-check only where the SSDSE item and the saved SSDS 2020 observation
 # have the same reference year.  Other same-code values are disclosed but are
 # explicitly non-comparable rather than counted as matches or mismatches.
 cross=[]
 for c in sorted(set(info.feature)&set(old2020.columns)):
  syear=int(info.loc[info.feature.eq(c),'reference_year'].iloc[0])
  if syear!=2020:
   cross.append({'feature':c,'ssdse_reference_year':syear,'estat_reference_year':2020,'comparison_status':'not_comparable_year','n_compared':0,'exact_match_rate':None,'max_absolute_difference':None,'mismatch_count':0,'reason':'Saved e-Stat panel is a 2020 observation; identical item code alone does not establish same reference year.'})
   continue
  x=d[['code',c]].merge(old2020[['code',c]],on='code',how='inner',suffixes=('_ssdse','_estat')).dropna()
  if x.empty:
   cross.append({'feature':c,'ssdse_reference_year':syear,'estat_reference_year':2020,'comparison_status':'not_comparable_missing','n_compared':0,'exact_match_rate':None,'max_absolute_difference':None,'mismatch_count':0,'reason':'No nonmissing common values.'}); continue
  diff=(x[c+'_ssdse']-x[c+'_estat']).abs(); mismatch=x.loc[diff>0,'code'].tolist()
  cross.append({'feature':c,'ssdse_reference_year':syear,'estat_reference_year':2020,'comparison_status':'comparable_pass' if not len(mismatch) else 'comparable_mismatch','n_compared':len(x),'exact_match_rate':float((diff==0).mean()),'max_absolute_difference':float(diff.max()),'mismatch_count':int((diff>0).sum()),'reason':'Same item code and 2020 reference year.','mismatch_codes':json.dumps(mismatch,ensure_ascii=False)})
 cross=pd.DataFrame(cross); cross.to_csv(Q/'ssdse_estat_crosscheck.csv',index=False,encoding='utf-8-sig')
 comparable=cross[cross.comparison_status.str.startswith('comparable')] if len(cross) else cross
 csum={'status':'pass' if not comparable.comparison_status.eq('comparable_mismatch').any() else 'fail','items_compared':int(len(comparable)),'items_with_mismatch':int(comparable.comparison_status.eq('comparable_mismatch').sum()),'items_not_comparable':int(len(cross)-len(comparable)),'method':'Comparison requires same item code, verified same definition, and same 2020 reference year; non-comparable rows are not treated as matches.'}
 (Q/'ssdse_estat_crosscheck_summary.json').write_text(json.dumps(csum,ensure_ascii=False,indent=2),encoding='utf8')
 # Retain 2010 rows solely for explicitly labelled longitudinal work; replace
 # every 2020 row with the SSDSE-defined canonical cross-section.
 panel=pd.concat([old_panel[old_panel.year!=2020],d],ignore_index=True,sort=False); panel.to_parquet(P/'municipality_panel.parquet',index=False); analysis.to_parquet(P/'municipality_cross_section_2020.parquet',index=False)
 # The balanced cohort is an explicitly separate 2010/2020 analysis universe,
 # but is rebuilt as the intersection with the SSDSE canonical 2020 codes.
 canonical_codes=set(analysis.code.astype(str)); balanced=panel[panel.code.astype(str).isin(canonical_codes)].groupby('code').filter(lambda z:set(z.year)=={2010,2020}).copy()
 balanced.to_parquet(P/'municipality_panel_balanced.parquet',index=False)
 tr=balanced.pivot(index='code',columns='year',values='rail_any_share').dropna(); tr['delta_rail_10_20']=tr[2020]-tr[2010]; tr.reset_index().to_parquet(P/'municipality_transitions.parquet',index=False)
 analysis[['code','prefecture_name','municipality_name','rail_any_share','rail_users','commuters']].to_parquet(P/'ssdse_a_2026_analysis_frame.parquet',index=False)
 pd.DataFrame({'feature':core,'model':'SSDSE-A Core','source_dataset':'SSDSE-A-2026'}).to_parquet(A/'core_features.parquet',index=False); pd.DataFrame({'feature':expanded,'model':'SSDSE-A + e-Stat SSDS Expanded'}).to_parquet(A/'ml_features.parquet',index=False)
 a,pa=cv_compare(analysis,core,'SSDSE-A Core'); b,pb=cv_compare(analysis,expanded,'SSDSE-A + e-Stat SSDS Expanded')
 comp=pd.DataFrame([a,b]); comp.to_csv(T/'ssdse_core_vs_expanded.csv',index=False); (A/'ssdse_core_vs_expanded.json').write_text(json.dumps({'models':comp.to_dict(orient='records'),'same_universe':True,'same_target':'2020 H730 railway-use share','same_split':'GroupKFold(5) by prefecture','feature_order_policy':'frozen audited order'},ensure_ascii=False,indent=2),encoding='utf8')
 # Saved metadata is both a reproducibility record and a raw-file integrity check.
 retrieval=json.loads((RAW/'retrieval_record.json').read_text(encoding='utf8'))
 raw_hash=sha(CSV)
 if raw_hash != retrieval.get('raw_sha256'):
  raise RuntimeError(f"SSDSE raw SHA-256 mismatch: actual={raw_hash} recorded={retrieval.get('raw_sha256')}")
 metadata={'dataset_name':'SSDSE-市区町村（SSDSE-A）','dataset_version':'SSDSE-A-2026','official_source_url':URL,'landing_page_url':'https://www.nstac.go.jp/use/literacy/ssdse/','retrieved_at':retrieval['retrieved_at'],'raw_file':'data/raw/ssdse_a_2026/SSDSE-A-2026.csv','raw_sha256':raw_hash,'raw_file_size_bytes':CSV.stat().st_size,'explanation_file':'data/raw/ssdse_a_2026/kaisetsu-A-2026.pdf','explanation_sha256':sha(PDF),'municipality_count':int(len(s)),'item_count':int(len(info)),'layout':'3 metadata rows + 1741 municipal rows; 3 geography columns + 125 item columns'}
 (RAW/'source_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8'); info.assign(source_dataset='SSDSE-A-2026',ssdse_version='2026',ssdse_item_code=info.feature,raw_or_derived='raw',temporal_safe=info.reference_year.le(2020),source_url=URL).to_csv(C/'ssdse_a_2026_item_metadata.csv',index=False,encoding='utf-8-sig')
 # Make raw SSDSE entries available to the existing data-dictionary builder.
 olddict=pd.read_parquet(C/'data_dictionary.parquet') if (C/'data_dictionary.parquet').exists() else pd.DataFrame()
 add=info.assign(item_name=info.official_item_name_raw,stat_table_id='SSDSE-A-2026',source_stat='SSDSE-A-2026 official municipal dataset',raw_or_derived='raw')[['feature','item_name','stat_table_id','source_stat','raw_or_derived']]
 pd.concat([olddict[~olddict.feature.isin(add.feature)] if not olddict.empty else olddict,add],ignore_index=True).to_parquet(C/'data_dictionary.parquet',index=False)
 compliance={'status':'pass' if leakage['status']=='pass' and csum['status']=='pass' else 'fail','dataset_name':metadata['dataset_name'],'dataset_version':'SSDSE-A-2026','official_source':URL,'raw_sha256':metadata['raw_sha256'],'municipality_count_raw':len(s),'municipality_count_analysis':len(analysis),'municipality_count_balanced':int(balanced.code.nunique()),'ssdse_features_used':core_raw,'derived_ssdse_features_used':derived,'models_using_ssdse':['SSDSE-A Core','SSDSE-A + e-Stat SSDS Expanded'],'future_leakage':leakage,'canonical_universe_verified':universe['status']=='pass','estat_crosscheck_status':csum['status'],'outcome_missing_exclusions':len(excluded)}
 (Q/'ssdse_compliance.json').write_text(json.dumps(compliance,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'canonical_raw_n':len(s),'analysis_n':len(analysis),'core_features':len(core),'expanded_features':len(expanded)},ensure_ascii=False))
if __name__=='__main__': main()
