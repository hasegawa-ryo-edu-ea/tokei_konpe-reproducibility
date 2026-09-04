#!/usr/bin/env python3
"""Independent robustness, validation, and presentation artifacts from the panel."""
from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

R=Path(__file__).resolve().parents[1]; P=R/'processed'; A=R/'analysis'; Q=R/'qa'; F=R/'figures'; T=R/'tables'; RUN=R/'runs'; FINAL=R/'final'
for x in [A/'sensitivity',A/'ml',F,T]: x.mkdir(parents=True,exist_ok=True)
def score(y,p): return {'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)),'spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman'))}
def cv(d, feats, model, splitter, groups=None):
    pr=np.full(len(d),np.nan)
    for tr,te in splitter.split(d[feats],d.rail_any_share,groups):
        model.fit(d.iloc[tr][feats],d.iloc[tr].rail_any_share); pr[te]=model.predict(d.iloc[te][feats])
    return pr,score(d.rail_any_share,pr)
def main():
    d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').dropna(subset=['rail_any_share']).copy()
    f=pd.read_parquet(A/'ml_features.parquet').feature.tolist(); f=[x for x in f if x in d and d[x].notna().mean()>.55]
    # QA: distribution, correlation, NZV.
    audit=pd.DataFrame({'feature':f,'missing_rate':[d[x].isna().mean() for x in f],'n_unique':[d[x].nunique(dropna=True) for x in f],'variance':[d[x].var() for x in f]})
    audit['near_zero_variance']=(audit.n_unique<=1)|(audit.variance<=1e-12); audit.to_csv(Q/'feature_audit.csv',index=False)
    corr=d[f].corr(); pairs=(corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack().rename('correlation').reset_index().query('abs(correlation)>=0.95').sort_values('correlation',key=np.abs,ascending=False)); pairs.to_csv(Q/'high_correlation_pairs.csv',index=False)
    keep=[x for x in f if x not in set(pairs.level_1)]
    models={
      'extra_trees':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,n_jobs=-1,random_state=2026))]),
      'hist_gbdt':Pipeline([('impute',SimpleImputer(strategy='median')),('model',HistGradientBoostingRegressor(max_iter=300,l2_regularization=1,random_state=2026))]),
      'elastic_net':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',ElasticNet(alpha=.002,l1_ratio=.5,max_iter=20000,random_state=2026))])}
    rows=[]; k=KFold(5,shuffle=True,random_state=2026); g=GroupKFold(5); pref=d.code.astype(str).str[:2]
    for name,model in models.items():
      for fs,label in [(f,'all_features'),(keep,'corr_reduced')]:
       for sp,split,groups in [('random_kfold',k,None),('prefecture_group_kfold',g,pref)]:
        pred,m=cv(d,fs,model,split,groups); rows.append({'model':name,'feature_set':label,'split':sp,'n_features':len(fs),**m})
        if name=='extra_trees' and label=='all_features' and sp=='prefecture_group_kfold': d['group_oof_prediction']=pred
    # Missing-data sensitivity: same tree specification and geographic split,
    # changing only the imputation estimator fitted in each training fold.
    for strategy in ['median','mean']:
      model=Pipeline([('impute',SimpleImputer(strategy=strategy)),('scale',StandardScaler()),('model',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,n_jobs=-1,random_state=2026))])
      pred,m=cv(d,f,model,g,pref); rows.append({'model':'extra_trees','feature_set':'all_features','split':'prefecture_group_kfold','n_features':len(f),**m,'sensitivity_axis':'imputation','imputation_strategy':strategy})
    pd.DataFrame(rows).to_csv(A/'sensitivity/model_performance.csv',index=False)
    # Cluster method/grid comparison, no target input.
    # Clustering is descriptive rather than the prediction comparison.  Limit
    # its dimensionality deterministically to keep the GMM grid tractable when
    # the time-safe Expanded feature set is broad; supervised CV above still
    # uses every eligible Expanded feature.
    cluster_f=f[:min(30,len(f))]
    z=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]).fit_transform(d[cluster_f]); cr=[]
    for n in range(2,9):
      for method,obj in [('kmeans',KMeans(n_clusters=n,n_init=30,random_state=2026)),('hierarchical_ward',AgglomerativeClustering(n_clusters=n,linkage='ward'))]:
       lab=obj.fit_predict(z); cr.append({'method':method,'n_clusters':n,'silhouette':silhouette_score(z,lab),'calinski_harabasz':calinski_harabasz_score(z,lab),'davies_bouldin':davies_bouldin_score(z,lab)})
    pd.DataFrame(cr).to_csv(A/'clustering_comparison.csv',index=False)
    # Residual results use stricter geographic OOF predictions.
    d['group_oof_residual']=d.rail_any_share-d.group_oof_prediction
    d.sort_values('group_oof_residual').to_csv(A/'residual/residual_municipalities_group_oof.csv',index=False)
    # Close social-structure pairs with divergent rail use.
    from sklearn.neighbors import NearestNeighbors
    nn=NearestNeighbors(n_neighbors=6).fit(z); dist,idx=nn.kneighbors(z); pair=[]
    for i in range(len(d)):
      for dd,j in zip(dist[i,1:],idx[i,1:]):
       if i<j: pair.append({'code_a':d.code.iloc[i],'code_b':d.code.iloc[j],'feature_distance':dd,'rail_share_gap_pp':abs(d.rail_any_share.iloc[i]-d.rail_any_share.iloc[j])*100})
    pd.DataFrame(pair).query('feature_distance<=feature_distance.quantile(.25)').sort_values('rail_share_gap_pp',ascending=False).head(200).to_csv(A/'residual/similar_structure_divergent_rail_pairs.csv',index=False)
    # Figures/table files.
    sns.set_theme(style='whitegrid');
    plt.figure(figsize=(6,5)); plt.scatter(d.group_oof_prediction*100,d.rail_any_share*100,s=8,alpha=.35); lo,hi=0,max(d.group_oof_prediction.max(),d.rail_any_share.max())*100; plt.plot([lo,hi],[lo,hi],color='black'); plt.xlabel('Group-OOF predicted rail share (%)');plt.ylabel('Observed rail share (%)');plt.tight_layout();plt.savefig(F/'group_oof_predicted_vs_observed.png',dpi=180);plt.close()
    plt.figure(figsize=(7,4));sns.histplot(d.group_oof_residual*100,bins=45);plt.xlabel('Group-OOF residual (percentage points)');plt.tight_layout();plt.savefig(F/'group_oof_residual_distribution.png',dpi=180);plt.close()
    pd.DataFrame(rows).to_csv(T/'model_performance.csv',index=False); pd.DataFrame(cr).to_csv(T/'cluster_method_comparison.csv',index=False)
    k=json.loads((FINAL/'key_numbers.json').read_text(encoding='utf8')); k['group_cv_sensitivity']=rows; k['strict_three_wave_cohort_n']=int(pd.read_parquet(P/'municipality_panel_stable_boundary.parquet').code.nunique()); (FINAL/'key_numbers.json').write_text(json.dumps(k,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__': main()
