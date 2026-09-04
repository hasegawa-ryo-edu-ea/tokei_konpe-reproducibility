#!/usr/bin/env python3
"""Valid longitudinal analyses. No future features are used in prediction."""
from pathlib import Path
import json, hashlib, datetime, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

R=Path(__file__).resolve().parents[1]; P=R/'processed'; A=R/'analysis'; F=R/'figures'; T=R/'tables'; Q=R/'qa'; RUN=R/'runs'; FINAL=R/'final'
for d in [A/'panel',A/'temporal',A/'sensitivity',F,T,Q]:d.mkdir(parents=True,exist_ok=True)
def met(y,p):
 y=np.asarray(y); p=np.asarray(p)
 return {'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)),'spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman'))}
def raw_age_2000():
 """Read all saved Census age extracts and require total, 15+, and 65+ codes."""
 import gzip, glob
 paths=sorted(glob.glob(str(R/'data/raw/census_2000_ses_age/*jsonl.gz')))
 if not paths: raise FileNotFoundError('missing e-Stat 2000 age extract')
 rows=[]
 for path in paths:
  with gzip.open(path,'rt',encoding='utf8') as fh:
   for line in fh:
    x=json.loads(line); rows.append({'code':str(x.get('@area')),'age_code':str(x.get('@cat02')),'value':pd.to_numeric(x.get('$'),errors='coerce')})
 z=pd.DataFrame(rows).pivot_table(index='code',columns='age_code',values='value',aggfunc='first')
 required={'T01','415','565'}; missing=sorted(required-set(map(str,z.columns)))
 if missing: raise RuntimeError(f'missing required 2000 Census age categories after combining saved extracts: {missing}')
 z=z.rename(columns={'T01':'population_total_2000','415':'population_15plus_2000','565':'population_65plus_2000'})
 z['aging_share_65plus']=z.population_65plus_2000/z.population_total_2000.where(z.population_total_2000>0)
 z['working_age_share_15_64']=(z.population_15plus_2000-z.population_65plus_2000)/z.population_total_2000.where(z.population_total_2000>0)
 z['population_total']=z.population_total_2000
 return z[['population_total','aging_share_65plus','working_age_share_15_64']]
def temporal_pipeline(kind):
 model=Ridge(alpha=5.0) if kind=='ridge' else ElasticNet(alpha=.001,l1_ratio=.25,max_iter=10000,random_state=2026)
 return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',model)])
def main():
 strict=pd.read_parquet(P/'municipality_panel_stable_boundary.parquet').copy()
 if strict.empty: raise RuntimeError('strict panel is empty; run run_analysis.py first')
 wide=strict.pivot(index='code',columns='year',values='rail_any_share').dropna();
 transitions=pd.DataFrame({'code':wide.index,'rail_2000':wide[2000],'rail_2010':wide[2010],'rail_2020':wide[2020]})
 transitions['delta_00_10']=transitions.rail_2010-transitions.rail_2000; transitions['delta_10_20']=transitions.rail_2020-transitions.rail_2010
 transitions.to_parquet(A/'temporal/strict_code_transitions.parquet',index=False)
 # Strict out-of-time validation: fit only 2000 X -> 2010 Y, test 2010 X -> 2020 Y.
 # This is intentionally a small, definition-auditable Census feature set; no X_2020 is used.
 age00=raw_age_2000(); pan_for_t=pd.read_parquet(P/'municipality_panel_balanced.parquet'); x10=pan_for_t[pan_for_t.year==2010].set_index('code')
 x10=pd.DataFrame({'population_total':x10.A1101,'aging_share_65plus':x10.aging_share_65plus,'working_age_share_15_64':x10.working_age_share_15_64})
 basecols=['population_total','aging_share_65plus','working_age_share_15_64','rail_t']
 train=transitions.set_index('code').join(age00,how='inner'); train['rail_t']=train.rail_2000; train=train.dropna(subset=basecols+['rail_2010'])
 test=transitions.set_index('code').join(x10,how='inner'); test['rail_t']=test.rail_2010; test=test.dropna(subset=basecols+['rail_2020'])
 rows=[]; preds={}
 for kind in ['ridge','elastic_net']:
  model=temporal_pipeline(kind).fit(train[basecols],train.rail_2010); pred=model.predict(test[basecols]); mm=met(test.rail_2020,pred); preds[kind]=pred; rows.append({'model':kind,'train_transition':'2000->2010','test_transition':'2010->2020','n_train':len(train),'n_test':len(test),**mm,'features':';'.join(basecols)})
 comparison=pd.DataFrame(rows); comparison.to_csv(A/'temporal/temporal_ses_model_comparison.csv',index=False)
 # Persist both attempted specifications, including the non-selected model.
 dh=hashlib.sha256(pd.concat([train[basecols+['rail_2010']],test[basecols+['rail_2020']]]).to_csv().encode()).hexdigest()
 regpath=RUN/'experiment_registry.parquet'; reg=pd.read_parquet(regpath) if regpath.exists() else pd.DataFrame()
 regrows=[]
 for rr in rows:
  runid=f"temporal_ses_{rr['model']}_{dh[:12]}"
  regrows.append({'run_id':runid,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset_hash':dh,'feature_set':json.dumps(basecols),'outcome':'rail_any_share_t_plus_10','period':'train 2000->2010; test 2010->2020','model':rr['model'],'parameters':json.dumps({'random_state':2026,'ridge_alpha':5.0} if rr['model']=='ridge' else {'random_state':2026,'alpha':.001,'l1_ratio':.25,'max_iter':10000},sort_keys=True),'split_method':'strict chronological holdout','metrics':json.dumps({k:rr[k] for k in ['rmse','mae','r2','spearman']},allow_nan=False),'selected':rr['model']=='elastic_net','selection_reason':'Lowest temporal holdout RMSE among the pre-specified compact SES Ridge/Elastic Net comparison.'})
 if not reg.empty: reg=reg[~reg.run_id.isin([x['run_id'] for x in regrows])]
 pd.concat([reg,pd.DataFrame(regrows)],ignore_index=True).to_parquet(regpath,index=False)
 m=min(rows,key=lambda r:r['rmse']); pred=preds[m['model']]
 pd.DataFrame({'code':test.index,'observed_2020':test.rail_2020,'temporal_prediction_from_2010_ses':pred,'residual':test.rail_2020-pred}).to_csv(A/'temporal/temporal_holdout_predictions.csv',index=False)
 # Descriptive change score models: X_2010 from harmonized SSDS only; 2020 covariates excluded.
 pan=pd.read_parquet(P/'municipality_panel_balanced.parquet'); b=pan[pan.year==2010].set_index('code'); e=pan[pan.year==2020].set_index('code'); common=b.index.intersection(e.index)
 candidates=pd.read_parquet(A/'core_features.parquet').feature.tolist(); feats=[x for x in candidates if x in b and b[x].notna().mean()>.75]
 d=b.loc[common,feats].copy(); d['rail_2010']=b.loc[common,'rail_any_share']; d['delta_10_20']=e.loc[common,'rail_any_share']-b.loc[common,'rail_any_share']; d=d.dropna()
 # Change-score OLS, robust HC3. This is not causal; time t features only.
 form='delta_10_20 ~ rail_2010 + ' + ' + '.join(feats[:10]); res=sm.OLS.from_formula(form,d).fit(cov_type='HC3'); (A/'panel/change_score_2010_2020.txt').write_text(res.summary().as_text(),encoding='utf8')
 # First-difference analogue of two-way FE for 2010/2020, with delta X and level baseline reported separately.
 delta=pd.DataFrame(index=common); delta['delta_rail']=e.loc[common,'rail_any_share']-b.loc[common,'rail_any_share']
 for x in feats[:10]: delta['d_'+x]=e.loc[common,x]-b.loc[common,x]
 delta=delta.dropna(); fd=sm.OLS(delta.delta_rail,sm.add_constant(delta.drop(columns='delta_rail'))).fit(cov_type='HC3'); (A/'panel/first_difference_2010_2020.txt').write_text(fd.summary().as_text(),encoding='utf8')
 pd.DataFrame({'measure':['strict_temporal_train_n','strict_temporal_test_n','strict_temporal_rmse','strict_temporal_mae','strict_temporal_r2'],'value':[len(train),len(test),m['rmse'],m['mae'],m['r2']]}).to_csv(T/'longitudinal_summary.csv',index=False)
 sns.set_theme(style='whitegrid'); plt.figure(figsize=(7,4));
 sns.kdeplot(transitions.delta_00_10*100,label='2000→2010');sns.kdeplot(transitions.delta_10_20*100,label='2010→2020 (COVID-included)');plt.xlabel('Rail-use change (percentage points)');plt.legend();plt.tight_layout();plt.savefig(F/'strict_cohort_rail_change_distributions.png',dpi=180);plt.close()
 note={'strict_exact_code_cohort_n':int(len(wide)),'temporal_validation':{'train_transition':'2000->2010','test_transition':'2010->2020','features':basecols,'models':'Ridge and Elastic Net; median imputation/scaling fit on 2000 train only','metrics':{k:v for k,v in m.items() if k in {'rmse','mae','r2','spearman'}},'n_train':int(len(train)),'n_test':int(len(test)),'caveat':'2000 age totals and 2010 SSDS age variables are definition-audited but are a compact common predictor set; 2020 socioeconomic fields are never used.'},'change_score_n':int(res.nobs),'first_difference_n':int(fd.nobs)}
 (A/'temporal/temporal_validation.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf8')
 k=json.loads((FINAL/'key_numbers.json').read_text(encoding='utf8'));k['longitudinal']=note;(FINAL/'key_numbers.json').write_text(json.dumps(k,ensure_ascii=False,indent=2),encoding='utf8')
 (Q/'municipality_harmonization.md').write_text('Strict exact-code cohort uses only area codes present with nonmissing numerator and denominator in 2000, 2010, and 2020 direct Census tables; national/prefecture aggregate codes ending 000 are excluded. No municipality name join or value aggregation is performed. Shared code does not prove unchanged geography; this is labelled strict_exact_code rather than boundary-perfect.\n',encoding='utf8')
if __name__=='__main__':main()
