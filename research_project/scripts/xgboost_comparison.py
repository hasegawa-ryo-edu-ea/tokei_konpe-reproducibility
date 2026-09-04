#!/usr/bin/env python3
"""Reproducible XGBoost comparison, with geographic out-of-fold evaluation."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor

R=Path(__file__).resolve().parents[1];P=R/'processed';A=R/'analysis';RUN=R/'runs';T=R/'tables'
for d in [A/'ml',RUN,T]:d.mkdir(parents=True,exist_ok=True)
def main():
 d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').dropna(subset=['rail_any_share']).copy();f=pd.read_parquet(A/'ml_features.parquet').feature.tolist();f=[x for x in f if x in d and d[x].notna().mean()>=.55]
 groups=d.code.astype(str).str[:2]; pred=np.full(len(d),np.nan); pars={'n_estimators':500,'max_depth':4,'learning_rate':.04,'subsample':.8,'colsample_bytree':.8,'reg_lambda':4.0,'objective':'reg:squarederror','tree_method':'hist','random_state':2026,'n_jobs':-1}
 for tr,te in GroupKFold(5).split(d[f],d.rail_any_share,groups):
  m=Pipeline([('impute',SimpleImputer(strategy='median')),('xgb',XGBRegressor(**pars))]);m.fit(d.iloc[tr][f],d.iloc[tr].rail_any_share);pred[te]=m.predict(d.iloc[te][f])
 metrics={'rmse':float(mean_squared_error(d.rail_any_share,pred)**.5),'mae':float(mean_absolute_error(d.rail_any_share,pred)),'r2':float(r2_score(d.rail_any_share,pred)),'spearman':float(pd.Series(d.rail_any_share).corr(pd.Series(pred),method='spearman'))}
 pd.DataFrame({'code':d.code,'observed':d.rail_any_share,'group_oof_prediction':pred,'residual':d.rail_any_share-pred}).to_csv(A/'ml/xgboost_group_oof_predictions.csv',index=False)
 dh=hashlib.sha256(d[['code','year','rail_any_share',*f]].to_csv(index=False).encode()).hexdigest();rid='xgboost_2020_groupcv_'+hashlib.sha256((dh+json.dumps(pars,sort_keys=True)).encode()).hexdigest()[:12]
 row={'run_id':rid,'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_hash':dh,'feature_set':json.dumps(f),'outcome':'rail_any_share','period':'2020','model':'XGBRegressor','parameters':json.dumps(pars,sort_keys=True),'split_method':'GroupKFold(5) by prefecture code','metrics':json.dumps(metrics),'selected':False,'selection_reason':'nonlinear benchmark; compared with ExtraTrees and HistGradientBoosting'}
 rp=RUN/'experiment_registry.parquet';old=pd.read_parquet(rp) if rp.exists() else pd.DataFrame();old=old[old.run_id!=rid] if not old.empty else old;pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_parquet(rp,index=False)
 pd.DataFrame([{'model':'xgboost','feature_set':'all_features','split':'prefecture_group_kfold','n_features':len(f),**metrics}]).to_csv(A/'ml/xgboost_metrics.csv',index=False)
if __name__=='__main__':main()
