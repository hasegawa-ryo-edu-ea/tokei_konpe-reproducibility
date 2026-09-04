#!/usr/bin/env python3
"""Fold-separated permutation importance plus PDP and ALE for the selected GBDT model."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

R=Path(__file__).resolve().parents[1]; P=R/'processed'; A=R/'analysis'; F=R/'figures'; Q=R/'qa'
for d in [A/'interpretation',F,Q]:d.mkdir(parents=True,exist_ok=True)
def main():
 d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').dropna(subset=['rail_any_share']).copy(); fs=pd.read_parquet(A/'ml_features.parquet').feature.tolist(); fs=[x for x in fs if x in d and d[x].notna().mean()>=.55]
 # Pre-screen is solely for computational scope; final ranking is still held-out permutation importance.
 pre=pd.read_csv(A/'ml/feature_importance.csv').sort_values('importance',ascending=False).feature.tolist(); fs=[x for x in pre if x in fs][:15]
 k=KFold(5,shuffle=True,random_state=2026); rec=[]
 for fold,(tr,te) in enumerate(k.split(d),1):
  model=Pipeline([('impute',SimpleImputer(strategy='median')),('model',ExtraTreesRegressor(n_estimators=180,min_samples_leaf=3,n_jobs=-1,random_state=2026+fold))])
  model.fit(d.iloc[tr][fs],d.iloc[tr].rail_any_share)
  pi=permutation_importance(model,d.iloc[te][fs],d.iloc[te].rail_any_share,n_repeats=4,random_state=2026,n_jobs=1,scoring='neg_root_mean_squared_error')
  rec.extend({'fold':fold,'feature':x,'importance_rmse_increase':v,'importance_sd':s} for x,v,s in zip(fs,pi.importances_mean,pi.importances_std))
 imp=pd.DataFrame(rec); summary=imp.groupby('feature').agg(mean_importance=('importance_rmse_increase','mean'),sd_across_folds=('importance_rmse_increase','std'),n_folds=('fold','nunique')).reset_index().sort_values('mean_importance',ascending=False); summary.to_csv(A/'interpretation/oof_permutation_importance.csv',index=False)
 top=summary.feature.head(4).tolist(); model.fit(d[fs],d.rail_any_share); X=model.named_steps['impute'].transform(d[fs]); est=model.named_steps['model']
 # PDP uses fitted descriptive model; OOF importance above is the primary feature-ranking evidence.
 for feature in top:
  j=fs.index(feature); grid=np.quantile(X[:,j],np.linspace(.02,.98,40)); baseline=X.copy(); vals=[]
  for v in grid:
   z=baseline.copy();z[:,j]=v;vals.append(est.predict(z).mean())
  pd.DataFrame({'feature':feature,'x':grid,'partial_dependence':vals}).to_csv(A/f'interpretation/pdp_{feature}.csv',index=False)
  plt.figure(figsize=(5,3.2));plt.plot(grid,np.array(vals)*100);plt.xlabel(feature);plt.ylabel('Mean predicted rail share (%)');plt.tight_layout();plt.savefig(F/f'pdp_{feature}.png',dpi=180);plt.close()
  # First-order ALE: local prediction differences, accumulated and centered.
  edges=np.unique(np.quantile(X[:,j],np.linspace(0,1,21))); rows=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   mask=(X[:,j]>=lo)&(X[:,j]<=hi if hi==edges[-1] else X[:,j]<hi)
   if mask.sum()<5: continue
   zl=X[mask].copy(); zh=X[mask].copy(); zl[:,j]=lo;zh[:,j]=hi; rows.append((lo,hi,mask.sum(),(est.predict(zh)-est.predict(zl)).mean()))
  al=pd.DataFrame(rows,columns=['lower','upper','n','local_effect']);al['ale']=al.local_effect.cumsum();al['ale_centered']=al.ale-np.average(al.ale,weights=al.n);al.to_csv(A/f'interpretation/ale_{feature}.csv',index=False)
  plt.figure(figsize=(5,3.2));plt.plot((al.lower+al.upper)/2,al.ale_centered*100);plt.axhline(0,color='black',lw=.7);plt.xlabel(feature);plt.ylabel('ALE (percentage points)');plt.tight_layout();plt.savefig(F/f'ale_{feature}.png',dpi=180);plt.close()
 (Q/'interpretation_methods.md').write_text('Primary importance is fold-separated held-out permutation importance (5-fold). PDP and ALE are descriptive full-2020-model diagnostics, not causal effects. SHAP import was attempted but did not return in this Python environment; no SHAP values are reported.\n',encoding='utf8')
if __name__=='__main__':main()
