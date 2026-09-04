#!/usr/bin/env python3
"""CatBoost benchmark under the same prefecture-grouped validation as XGBoost."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from catboost import CatBoostRegressor

R=Path(__file__).resolve().parents[1];P=R/'processed';A=R/'analysis';RUN=R/'runs'
for d in [A/'ml',RUN]:d.mkdir(parents=True,exist_ok=True)
def main():
 d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').dropna(subset=['rail_any_share']).copy(); f=pd.read_parquet(A/'ml_features.parquet').feature.tolist(); f=[x for x in f if x in d and d[x].notna().mean()>=.55]
 groups=d.code.astype(str).str[:2]; pred=np.full(len(d),np.nan)
 pars={'iterations':600,'depth':6,'learning_rate':.04,'l2_leaf_reg':5.0,'loss_function':'RMSE','random_seed':2026,'verbose':False,'thread_count':-1,'allow_writing_files':False}
 for tr,te in GroupKFold(5).split(d[f],d.rail_any_share,groups):
  imp=SimpleImputer(strategy='median').fit(d.iloc[tr][f]); xt=imp.transform(d.iloc[tr][f]); xe=imp.transform(d.iloc[te][f])
  m=CatBoostRegressor(**pars);m.fit(xt,d.iloc[tr].rail_any_share);pred[te]=m.predict(xe)
 metrics={'rmse':float(mean_squared_error(d.rail_any_share,pred)**.5),'mae':float(mean_absolute_error(d.rail_any_share,pred)),'r2':float(r2_score(d.rail_any_share,pred)),'spearman':float(pd.Series(d.rail_any_share).corr(pd.Series(pred),method='spearman'))}
 pd.DataFrame({'code':d.code,'observed':d.rail_any_share,'group_oof_prediction':pred,'residual':d.rail_any_share-pred}).to_csv(A/'ml/catboost_group_oof_predictions.csv',index=False)
 dh=hashlib.sha256(d[['code','year','rail_any_share',*f]].to_csv(index=False).encode()).hexdigest();rid='catboost_2020_groupcv_'+hashlib.sha256((dh+json.dumps(pars,sort_keys=True)).encode()).hexdigest()[:12]
 row={'run_id':rid,'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_hash':dh,'feature_set':json.dumps(f),'outcome':'rail_any_share','period':'2020','model':'CatBoostRegressor','parameters':json.dumps(pars,sort_keys=True),'split_method':'GroupKFold(5) by prefecture code','metrics':json.dumps(metrics),'selected':False,'selection_reason':'nonlinear benchmark; same grouped CV and feature set as XGBoost'}
 rp=RUN/'experiment_registry.parquet';old=pd.read_parquet(rp) if rp.exists() else pd.DataFrame();old=old[old.run_id!=rid] if not old.empty else old;pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_parquet(rp,index=False)
 pd.DataFrame([{'model':'catboost','feature_set':'all_features','split':'prefecture_group_kfold','n_features':len(f),**metrics}]).to_csv(A/'ml/catboost_metrics.csv',index=False)
if __name__=='__main__':main()
