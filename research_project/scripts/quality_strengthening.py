#!/usr/bin/env python3
"""Validation-first additions: temporal baselines, bounded outcome, OOF, bootstrap."""
from pathlib import Path
import gzip, glob, hashlib, json
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

R=Path(__file__).resolve().parents[1]; P=R/'processed'; A=R/'analysis'; T=R/'tables'; Q=R/'qa'; RUN=R/'runs'
for x in [A/'robustness',A/'temporal',A/'residual',A/'bounded',A/'bootstrap',T,Q,RUN]:x.mkdir(parents=True,exist_ok=True)
SEED=2026
def met(y,p):
 y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); e=p-y
 return {'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(np.mean(np.abs(e))),'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),'spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman')),'mean_bias':float(e.mean()),'observed_mean':float(y.mean()),'predicted_mean':float(p.mean()),'observed_sd':float(y.std(ddof=1)),'predicted_sd':float(p.std(ddof=1))}
def age00():
 rows=[]
 for path in glob.glob(str(R/'data/raw/census_2000_ses_age/*jsonl.gz')):
  with gzip.open(path,'rt',encoding='utf8') as f:
   for line in f:
    x=json.loads(line);rows.append((str(x.get('@area')),str(x.get('@cat02')),pd.to_numeric(x.get('$'),errors='coerce')))
 z=pd.DataFrame(rows,columns=['code','age','v']).drop_duplicates(['code','age']).pivot(index='code',columns='age',values='v'); z['population_total']=z['T01'];z['aging_share_65plus']=z['565']/z['T01'];z['working_age_share_15_64']=(z['415']-z['565'])/z['T01'];return z[['population_total','aging_share_65plus','working_age_share_15_64']]
def register(rows):
 rp=RUN/'experiment_registry.parquet'; old=pd.read_parquet(rp) if rp.exists() else pd.DataFrame(); ids=[x['run_id'] for x in rows]
 if not old.empty: old=old[~old.run_id.isin(ids)]
 pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_parquet(rp,index=False)
def temporal():
 s=pd.read_parquet(P/'municipality_panel_stable_boundary.parquet'); w=s.pivot(index='code',columns='year',values='rail_any_share').dropna(); tr=pd.DataFrame({'rail_t':w[2000],'y_next':w[2010]}).join(age00(),how='inner').dropna(); p=pd.read_parquet(P/'municipality_panel_balanced.parquet'); x=p[p.year==2010].set_index('code'); te=pd.DataFrame({'rail_t':w[2010],'y_next':w[2020]}).join(pd.DataFrame({'population_total':x.A1101,'aging_share_65plus':x.aging_share_65plus,'working_age_share_15_64':x.working_age_share_15_64}),how='inner').dropna()
 f=['population_total','aging_share_65plus','working_age_share_15_64','rail_t']; rows=[]; pred={}
 # Baselines are defined from training transition only.
 train_change=(tr.y_next-tr.rail_t).mean(); pred['persistence']=te.rail_t.to_numpy(); pred['mean_historical_change']=te.rail_t.to_numpy()+train_change
 models={'ridge':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',Ridge(alpha=5.0))]),'elastic_net':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',ElasticNet(alpha=.001,l1_ratio=.25,max_iter=20000,random_state=SEED))]),'extra_trees':Pipeline([('impute',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,n_jobs=-1,random_state=SEED))])}
 for n,m in models.items():m.fit(tr[f],tr.y_next);pred[n]=m.predict(te[f])
 # Direct delta learning; add prior rail level only as t information.
 delta_models={'delta_ridge':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',Ridge(alpha=5.0))]),'delta_elastic_net':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',ElasticNet(alpha=.001,l1_ratio=.25,max_iter=20000,random_state=SEED))]),'delta_extra_trees':Pipeline([('impute',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,n_jobs=-1,random_state=SEED))])}
 for n,m in delta_models.items():m.fit(tr[f],tr.y_next-tr.rail_t);pred[n]=te.rail_t.to_numpy()+m.predict(te[f])
 for n,v in pred.items(): rows.append({'model':n,'task':'level' if not n.startswith('delta_') else 'delta_direct','train_transition':'2000->2010','test_transition':'2010->2020','n_train':len(tr),'n_test':len(te),**met(te.y_next,v),'features':';'.join(f) if n not in ['persistence','mean_historical_change'] else 'rail_t'})
 out=pd.DataFrame(rows);out.to_csv(A/'temporal/temporal_validation_enhanced.csv',index=False);pd.DataFrame({'code':te.index,'observed_2020':te.y_next,**pred}).to_parquet(A/'temporal/temporal_enhanced_predictions.parquet',index=False)
 # concise machine-readable shift decomposition
 (A/'temporal/temporal_distribution_shift.json').write_text(json.dumps({'train_y_2010':{'mean':float(tr.y_next.mean()),'sd':float(tr.y_next.std())},'test_y_2020':{'mean':float(te.y_next.mean()),'sd':float(te.y_next.std())},'train_change_00_10':float(train_change),'test_change_10_20':float((te.y_next-te.rail_t).mean())},indent=2),encoding='utf8')
 return out
def oof():
 d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').dropna(subset=['rail_any_share']).reset_index(drop=True);f=pd.read_parquet(A/'ml_features.parquet').feature.tolist();f=[x for x in f if x in d and d[x].notna().mean()>=.55]; g=d.code.astype(str).str[:2]; pred={}
 specs={'extra_trees':Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,n_jobs=-1,random_state=SEED))]),'hist_gbdt':Pipeline([('i',SimpleImputer(strategy='median')),('m',HistGradientBoostingRegressor(max_iter=300,l2_regularization=1,random_state=SEED))]),'elastic_net':Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',ElasticNet(alpha=.002,l1_ratio=.5,max_iter=20000,random_state=SEED))])}
 for name,model in specs.items():
  z=np.empty(len(d));
  for a,b in GroupKFold(5).split(d[f],d.rail_any_share,g):model.fit(d.loc[a,f],d.loc[a,'rail_any_share']);z[b]=model.predict(d.loc[b,f])
  pred[name]=z
 for name in ['xgboost','catboost']:
  z=np.empty(len(d));
  for a,b in GroupKFold(5).split(d[f],d.rail_any_share,g):
   imp=SimpleImputer(strategy='median').fit(d.loc[a,f]);xa,xb=imp.transform(d.loc[a,f]),imp.transform(d.loc[b,f])
   model=XGBRegressor(n_estimators=500,max_depth=4,learning_rate=.04,subsample=.8,colsample_bytree=.8,reg_lambda=4,objective='reg:squarederror',tree_method='hist',random_state=SEED,n_jobs=-1) if name=='xgboost' else CatBoostRegressor(iterations=600,depth=6,learning_rate=.04,l2_leaf_reg=5,loss_function='RMSE',random_seed=SEED,verbose=False,thread_count=-1,allow_writing_files=False)
   model.fit(xa,d.loc[a,'rail_any_share']);z[b]=model.predict(xb)
  pred[name]=z
 out=pd.DataFrame({'code':d.code,'prefecture_group':g,'observed':d.rail_any_share}); rec=[]
 for n,z in pred.items():out[n+'_prediction']=z;out[n+'_residual']=d.rail_any_share-z;rec.append({'model':n,'split':'prefecture_group_kfold','n':len(d),**met(d.rail_any_share,z)})
 rs=np.column_stack([out[n+'_residual'] for n in pred]);out['consensus_residual']=np.median(rs,axis=1);out['model_disagreement_sd']=np.std(rs,axis=1);out['residual_sign_agreement']=np.abs(np.sign(rs).sum(axis=1))/len(pred);out.to_parquet(A/'robustness/oof_predictions_all_models.parquet',index=False);out[(out.residual_sign_agreement==1)].sort_values('consensus_residual').to_csv(A/'residual/consensus_residual_consistent_outliers.csv',index=False);pd.DataFrame(rec).to_csv(A/'robustness/oof_model_metrics.csv',index=False);return out,rec
def fractional_and_bootstrap(oofdf,metrics_rows):
 d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').merge(oofdf[['code']],on='code').drop_duplicates('code').copy(); fs=['population_density','aging_share_65plus','working_age_share_15_64','youth_share_0_14']; z=d.dropna(subset=fs+['rail_any_share','commuters']).copy(); X=(z[fs]-z[fs].mean())/z[fs].std();X=sm.add_constant(X); rows=[]
 for label,w in [('municipality_equal',np.ones(len(z))),('commuters_weighted',z.commuters.to_numpy())]:
  fit=sm.GLM(z.rail_any_share,X,family=sm.families.Binomial(),freq_weights=w).fit(cov_type='HC3')
  for term in fit.params.index:rows.append({'specification':label,'term':term,'coefficient':fit.params[term],'se_hc3':fit.bse[term],'pvalue':fit.pvalues[term],'n':len(z)})
 pd.DataFrame(rows).to_csv(A/'bounded/fractional_logit_coefficients.csv',index=False)
 # Cluster bootstrap: resample prefectures, then evaluate OOF metrics and refit equal-weight fractional logit.
 rng=np.random.default_rng(SEED); groups=oofdf.prefecture_group.unique(); bmet=[]; bcoef=[]; top=pd.read_csv(A/'interpretation/oof_permutation_importance.csv').head(10).feature.tolist(); foldimp=pd.read_csv(A/'interpretation/oof_permutation_importance.csv')
 for b in range(500):
  picked=rng.choice(groups,len(groups),replace=True); ix=np.concatenate([np.where(oofdf.prefecture_group.to_numpy()==q)[0] for q in picked])
  for n in ['extra_trees','hist_gbdt','elastic_net','xgboost','catboost']:bmet.append({'replicate':b,'model':n,**met(oofdf.observed.to_numpy()[ix],oofdf[n+'_prediction'].to_numpy()[ix])})
  zz=pd.concat([z[z.code.astype(str).str[:2]==q] for q in picked],ignore_index=True); xx=(zz[fs]-z[fs].mean())/z[fs].std();
  try:
   fit=sm.GLM(zz.rail_any_share,sm.add_constant(xx),family=sm.families.Binomial()).fit()
   bcoef.extend({'replicate':b,'term':k,'coefficient':v} for k,v in fit.params.items())
  except Exception: pass
 # Bootstrap folded importance summary uses fold-level uncertainty retained in its source summary SD.
 imp=pd.read_csv(A/'interpretation/oof_permutation_importance.csv'); bi=[]
 for b in range(500):
  draw=rng.normal(imp.mean_importance,imp.sd_across_folds.fillna(0)); rank=pd.Series(draw,index=imp.feature).rank(ascending=False,method='min')
  bi.extend({'replicate':b,'feature':k,'importance':float(v),'rank':int(rank[k])} for k,v in zip(imp.feature,draw))
 bm=pd.DataFrame(bmet);bc=pd.DataFrame(bcoef);bi=pd.DataFrame(bi);bm.to_parquet(A/'bootstrap/cluster_bootstrap_metrics.parquet',index=False);bc.to_parquet(A/'bootstrap/cluster_bootstrap_fractional_logit.parquet',index=False);bi.to_parquet(A/'bootstrap/cluster_bootstrap_importance.parquet',index=False)
 summ=[]
 for n,g in bm.groupby('model'):summ.append({'artifact':'metric','name':n,'rmse_p025':g.rmse.quantile(.025),'rmse_p975':g.rmse.quantile(.975),'mae_p025':g.mae.quantile(.025),'mae_p975':g.mae.quantile(.975)})
 for n,g in bc.groupby('term'):summ.append({'artifact':'fractional_logit','name':n,'coef_p025':g.coefficient.quantile(.025),'coef_p975':g.coefficient.quantile(.975)})
 for n,g in bi.groupby('feature'):summ.append({'artifact':'permutation_importance','name':n,'importance_p025':g.importance.quantile(.025),'importance_p975':g.importance.quantile(.975),'top5_stability':float((g['rank']<=5).mean())})
 pd.DataFrame(summ).to_csv(A/'bootstrap/bootstrap_summary.csv',index=False)
def main():
 t=temporal();o,mm=oof();fractional_and_bootstrap(o,mm)
 # Register every pre-specified validation result, including negative R2 values.
 dh=hashlib.sha256(o[['code','observed']].to_csv(index=False).encode()).hexdigest(); rows=[]
 for r in mm:
  rid='robust_oof_'+r['model']+'_'+dh[:12]; rows.append({'run_id':rid,'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_hash':dh,'feature_set':'analysis/ml_features.parquet','outcome':'rail_any_share','period':'2020','model':r['model'],'parameters':json.dumps({'random_state':SEED}),'split_method':'GroupKFold(5) by prefecture','metrics':json.dumps({k:r[k] for k in ['rmse','mae','r2','spearman','mean_bias']}),'selected':False,'selection_reason':'fold-separated OOF robustness comparison'})
 for _,r in t.iterrows():
  rid='temporal_enhanced_'+r.model+'_'+dh[:12]; rows.append({'run_id':rid,'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_hash':dh,'feature_set':r.features,'outcome':'rail_any_share_t_plus_10' if r.task=='level' else 'delta_rail_share_t_to_t_plus_10','period':'train 2000->2010; test 2010->2020','model':r.model,'parameters':json.dumps({'random_state':SEED}),'split_method':'strict chronological holdout','metrics':json.dumps({k:r[k] for k in ['rmse','mae','r2','spearman','mean_bias']}),'selected':False,'selection_reason':'pre-specified temporal baseline/level/direct-change comparison; negative results retained'})
 register(rows)
if __name__=='__main__':main()
