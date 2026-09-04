#!/usr/bin/env python3
"""End-to-end, non-narrative research pipeline for the railway-use project."""
from __future__ import annotations
import gzip, hashlib, json, math, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
from pygam import LinearGAM, s

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; PROC=ROOT/'processed'; ANA=ROOT/'analysis'; FIG=ROOT/'figures'; TAB=ROOT/'tables'; QA=ROOT/'qa'; FINAL=ROOT/'final'; RUNS=ROOT/'runs'; CAT=ROOT/'catalog'
for d in [PROC,ANA,FIG,TAB,QA,FINAL,RUNS,CAT]: d.mkdir(parents=True,exist_ok=True)
REGISTRY_ROWS=[]

def gzrows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def firstglob(folder):
    xs=list(folder.glob('values/*/*.jsonl.gz')) or list(folder.glob('*.jsonl.gz'))
    if not xs: raise FileNotFoundError(folder)
    return xs[0]
def meta_classes(path):
    m=json.loads(path.read_text(encoding='utf-8')); out={}
    for c in m['GET_META_INFO']['METADATA_INF']['CLASS_INF']['CLASS_OBJ']:
        items=c.get('CLASS',[])
        if isinstance(items,dict): items=[items]
        out[c['@id']]={str(x['@code']):str(x.get('@name','')) for x in items}
    return out
def transport_ssds():
    p=firstglob(RAW/'estat_target_transport'); rows=[]
    for x in gzrows(p):
        a=x['attributes'];
        if a.get('cat01') in {'H7301','H730102','H730105','H730111'}:
            rows.append({'code':a.get('area'),'year':int(a.get('time','')[:4]),'item':a.get('cat01'),'value':pd.to_numeric(x.get('value'),errors='coerce')})
    z=pd.DataFrame(rows).pivot_table(index=['code','year'],columns='item',values='value',aggfunc='first').reset_index()
    z=z.rename(columns={'H7301':'commuters','H730102':'rail_users','H730105':'car_users','H730111':'bike_users'})
    z['rail_any_share']=z.rail_users/z.commuters; z['car_any_share']=z.car_users/z.commuters; z['bike_share']=z.bike_users/z.commuters
    return z
def ssds_features():
    rows=[]; dictionary=[]
    for p in (RAW/'ssds_core').glob('*.jsonl.gz'):
        table=p.name.split('_')[0]; mp=p.with_name(table+'_metadata.json')
        if not mp.exists(): continue
        cls=meta_classes(mp); names=cls.get('cat01',{})
        for x in gzrows(p):
            if not isinstance(x,dict) or '@cat01' not in x: continue
            year=str(x.get('@time',''))[:4]
            if year not in {'2010','2020'}: continue
            val=pd.to_numeric(x.get('$'),errors='coerce')
            rows.append({'code':x.get('@area'),'year':int(year),'feature':x.get('@cat01'),'value':val})
        for code,name in names.items(): dictionary.append({'feature':code,'item_name':name,'stat_table_id':table,'source_stat':'e-Stat SSDS municipal data','raw_or_derived':'raw'})
    long=pd.DataFrame(rows); wide=long.pivot_table(index=['code','year'],columns='feature',values='value',aggfunc='first').reset_index()
    pd.DataFrame(dictionary).drop_duplicates().to_parquet(CAT/'data_dictionary.parquet',index=False)
    return wide
def direct_year(folder,year,rail_codes,total_code,cat2):
    p=firstglob(RAW/folder); rows=[]
    for x in gzrows(p):
        # Rust-captured files use attributes; the compact 2020 table-17-2
        # extract is the e-Stat API VALUE shape with @-prefixed dimensions.
        a=x.get('attributes',x); c1=a.get('cat01',a.get('@cat01')); c2=a.get('cat02',a.get('@cat02'))
        if c1 in set(rail_codes+[total_code]) and (cat2 is None or c2==cat2):
            rows.append((a.get('area',a.get('@area')),c1,pd.to_numeric(x.get('value',x.get('$')),errors='coerce')))
    d=pd.DataFrame(rows,columns=['code','item','value']); rail=d[d.item.isin(rail_codes)].groupby('code').value.sum(min_count=1); den=d[d.item==total_code].groupby('code').value.first()
    idx=rail.index.union(den.index)
    return pd.DataFrame({'code':idx.to_numpy(),'year':year,'rail_users':rail.reindex(idx).to_numpy(),'commuters':den.reindex(idx).to_numpy(),'rail_any_share':(rail.reindex(idx)/den.reindex(idx)).to_numpy()})
def registry(run, dataset, feats, model, split, metrics, selected, note):
    dh=hashlib.sha256(dataset.to_csv(index=False).encode()).hexdigest()
    canonical=hashlib.sha256(json.dumps({'base':run,'dataset_hash':dh,'features':feats,'model':model,'split':split},sort_keys=True).encode()).hexdigest()[:12]
    clean_metrics={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in metrics.items()}
    row={'run_id':f'{run}_{canonical}','timestamp':datetime.now(timezone.utc).isoformat(),'dataset_hash':dh,'feature_set':json.dumps(feats),'outcome':'rail_any_share','period':'2010-2020','model':model,'parameters':json.dumps({'random_state':2026},sort_keys=True),'split_method':split,'metrics':json.dumps(clean_metrics,allow_nan=False),'selected':selected,'selection_reason':note}
    REGISTRY_ROWS.append(row)
def write_registry():
    p=RUNS/'experiment_registry.parquet'; old=pd.read_parquet(p) if p.exists() else pd.DataFrame(); run_ids={row['run_id'] for row in REGISTRY_ROWS}; old=old[~old.run_id.isin(run_ids)] if not old.empty else old; pd.concat([old,pd.DataFrame(REGISTRY_ROWS)],ignore_index=True).to_parquet(p,index=False)
def metrics(y,p): return {'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)),'spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman'))}
def main():
    y=transport_ssds(); x=ssds_features(); panel=y.merge(x,on=['code','year'],how='left')
    # Derived, interpretable socioeconomic rates. Division guards preserve missingness rather than inventing values.
    panel['population_density'] = panel['A1101'] / panel['B1101'].where(panel['B1101']>0)
    panel['aging_share_65plus'] = panel['A1303'] / panel['A1101'].where(panel['A1101']>0)
    panel['working_age_share_15_64'] = panel['A1302'] / panel['A1101'].where(panel['A1101']>0)
    panel['youth_share_0_14'] = panel['A1301'] / panel['A1101'].where(panel['A1101']>0)
    # Exclude outcome/transport features plus excessively missing variables.
    idcols={'code','year','commuters','rail_users','car_users','bike_users','rail_any_share','car_any_share','bike_share'}
    feats=[c for c in panel.columns if c not in idcols and panel[c].notna().mean()>=.55 and not c.startswith('H730')]
    panel['log_commuters']=np.log1p(panel.commuters); panel['rail_pp']=panel.rail_any_share*100
    panel.to_parquet(PROC/'municipality_panel.parquet',index=False); panel.dropna(subset=['rail_any_share']).query('year==2020').to_parquet(PROC/'municipality_cross_section_2020.parquet',index=False)
    balanced=panel.groupby('code').filter(lambda d:set(d.year)=={2010,2020}).copy(); balanced.to_parquet(PROC/'municipality_panel_balanced.parquet',index=False)
    # Direct-census strict code intersection: 2000, 2010 and 2020 definitions recorded separately.
    d00=direct_year('census_2000_rail',2000,['003','012','013','014','015','016'],'000','000').merge(direct_year('census_2000_total',2000,[],'000','000')[['code','commuters']],on='code',suffixes=('','_den'))
    d00['commuters']=d00.commuters_den; d00['rail_any_share']=d00.rail_users/d00.commuters; d00=d00.drop(columns='commuters_den')
    d10=direct_year('census_2010_rail',2010,['003','012','013','014','016','017'],'000','017').merge(direct_year('census_2010_total',2010,[],'000','017')[['code','commuters']],on='code',suffixes=('','_den')); d10['commuters']=d10.commuters_den; d10['rail_any_share']=d10.rail_users/d10.commuters; d10=d10.drop(columns='commuters_den')
    # 2020 table 17-2 is required for the same observable definition as 2000/2010:
    # rail-only plus each rail-containing two-mode category; three+ modes excluded.
    # cat02=0 is the table's resident-based commuting/schooling target population.
    d20=direct_year('census_2020_transport_detail',2020,['21','31','32','33','34','35'],'0','0'); d20['rail_any_share']=d20.rail_users/d20.commuters
    strict=pd.concat([d00,d10,d20],ignore_index=True).dropna(subset=['rail_users','commuters','rail_any_share'])
    strict=strict[~strict.code.astype(str).str.endswith('000')]
    strict=strict.groupby('code').filter(lambda d:set(d.year)=={2000,2010,2020})
    strict.to_parquet(PROC/'municipality_panel_stable_boundary.parquet',index=False)
    trans=balanced.pivot(index='code',columns='year',values='rail_any_share'); trans['delta_rail_10_20']=trans[2020]-trans[2010]; trans.reset_index().to_parquet(PROC/'municipality_transitions.parquet',index=False)
    # QA
    miss=pd.DataFrame({'feature':panel.columns,'missing_rate':[panel[c].isna().mean() for c in panel.columns]}); miss.to_csv(QA/'missingness.csv',index=False)
    (QA/'leakage.md').write_text('Excluded target, raw transport counts and H730* transport variables from feature matrices. Temporal task uses X_2010 only. All imputation/scaling occurs inside sklearn Pipelines.\n',encoding='utf-8')
    temporal_def={'status':'pass','definition':'Resident-based age-15+ out-of-home worker/student transport universe; numerator is rail-only plus explicitly rail-containing two-mode categories; 3+ modes excluded in every year because rail membership is not identifiable in every wave.','years':{'2000':{'table':'0000033362','denominator':'cat01=000, cat02=000','rail_categories':['003','012','013','014','015','016']},'2010':{'table':'0003063775','denominator':'cat01=000, cat02=017','rail_categories':['003','012','013','014','016','017']},'2020':{'table':'0003454513','denominator':'cat01=0, cat02=0','rail_categories':['21','31','32','33','34','35']}},'excluded_all_years':'three or more transport modes; rail inclusion is not uniformly observable'}
    (QA/'temporal_target_definition.json').write_text(json.dumps(temporal_def,ensure_ascii=False,indent=2),encoding='utf8')
    (QA/'source_qa.md').write_text('Raw API responses retained below data/raw. SSDS target H730102/H7301 is used only for 2010/2020 cross-sectional work. Strict temporal target is separately harmonized from direct Census tables; see qa/temporal_target_definition.json. 2000/2010/2020 category definitions are not silently pooled.\n',encoding='utf-8')
    # EDA
    sns.set_theme(); plt.figure(figsize=(8,4)); sns.boxplot(data=balanced,x='year',y='rail_pp'); plt.tight_layout(); plt.savefig(FIG/'eda_rail_distribution.png',dpi=180); plt.close()
    # Core cross-sectional analysis
    cs=balanced.query('year==2020').dropna(subset=['rail_any_share']); F=[c for c in feats if cs[c].notna().mean()>=.7][:min(35,len(feats))]
    pd.DataFrame({'feature':F}).to_parquet(ANA/'core_features.parquet',index=False); pd.DataFrame({'feature':feats}).to_parquet(ANA/'ml_features.parquet',index=False)
    pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',ExtraTreesRegressor(n_estimators=400,min_samples_leaf=3,n_jobs=-1,random_state=2026))])
    oof=np.full(len(cs),np.nan); kf=KFold(5,shuffle=True,random_state=2026)
    for tr,te in kf.split(cs): pipe.fit(cs.iloc[tr][F],cs.iloc[tr].rail_any_share); oof[te]=pipe.predict(cs.iloc[te][F])
    m=metrics(cs.rail_any_share,oof); registry('cross_2020_extratrees_oof',cs,F,'ExtraTrees','KFold(5)',m,True,'OOF residual baseline')
    cs=cs.assign(oof_prediction=oof,residual=cs.rail_any_share-oof); cs[['code','rail_any_share','oof_prediction','residual']].sort_values('residual').to_csv(ANA/'residual/residual_municipalities.csv',index=False) if (ANA/'residual').mkdir(exist_ok=True) is None else None
    pipe.fit(cs[F],cs.rail_any_share); imp=pipe.named_steps.model.feature_importances_; pd.DataFrame({'feature':F,'importance':imp}).sort_values('importance',ascending=False).to_csv(ANA/'ml/feature_importance.csv',index=False) if (ANA/'ml').mkdir(exist_ok=True) is None else None
    # PCA / KMeans without target
    Z=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]).fit_transform(cs[F]); pca=PCA(random_state=2026).fit(Z); scores=pca.transform(Z); load=pd.DataFrame(pca.components_.T,index=F); load.to_csv(ANA/'pca_loadings.csv'); pd.DataFrame({'component':range(1,len(pca.explained_variance_ratio_)+1),'explained_variance_ratio':pca.explained_variance_ratio_}).to_csv(ANA/'pca_variance.csv',index=False)
    km=KMeans(n_clusters=4,n_init=30,random_state=2026).fit(scores[:,:min(8,scores.shape[1])]); pd.DataFrame({'code':cs.code,'cluster':km.labels_,'rail_any_share':cs.rail_any_share}).to_csv(ANA/'clusters.csv',index=False); registry('cluster_2020_kmeans4',cs,F,'KMeans','n/a',{'silhouette':float(silhouette_score(scores[:,:8],km.labels_))},True,'best interpretability candidate')
    # FE and GAM with named available features
    f2=F[:min(10,len(F))]; dat=balanced.dropna(subset=['rail_any_share',*f2]).copy()
    # Within transformation avoids an enormous dummy matrix and its rank defect.
    within=dat[['code','year','rail_any_share',*f2]].copy()
    within['y_within']=within.rail_any_share-within.groupby('code').rail_any_share.transform('mean')
    for col in f2: within[col]=within[col]-within.groupby('code')[col].transform('mean')
    within['year_2020']=(within.year==2020).astype(float)-.5
    fe=smf.ols('y_within ~ year_2020 + '+ ' + '.join(f2),data=within).fit(cov_type='cluster',cov_kwds={'groups':within.code})
    (ANA/'panel').mkdir(exist_ok=True); (ANA/'panel/fixed_effects.txt').write_text(fe.summary().as_text(),encoding='utf-8'); registry('panel_fe_2010_2020_within',within,f2,'OLS within FE','municipality-clustered SE',{'n':int(fe.nobs),'r2':float(fe.rsquared)},True,'within-municipality association; 2-wave panel')
    gfeat='population_density' if 'population_density' in cs else F[0]; gd=cs.dropna(subset=[gfeat]); gam=LinearGAM(s(0)).fit(gd[[gfeat]],gd.rail_any_share); grid=gam.generate_X_grid(term=0); pd.DataFrame({gfeat:grid[:,0],'prediction':gam.predict(grid),'lower':gam.confidence_intervals(grid)[:,0],'upper':gam.confidence_intervals(grid)[:,1]}).to_csv(ANA/'gam_curve.csv',index=False)
    # temporal 2010 -> 2020
    a=balanced.pivot(index='code',columns='year',values='rail_any_share').dropna(); base=balanced[balanced.year==2010].set_index('code'); future=balanced[balanced.year==2020].set_index('code'); common=a.index; tf=[c for c in F if c in base]
    train=base.loc[common].dropna(subset=tf); yy=future.loc[train.index].rail_any_share; model=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',ExtraTreesRegressor(n_estimators=400,min_samples_leaf=3,n_jobs=-1,random_state=2026))]); model.fit(train[tf],yy); pred=model.predict(train[tf]); tm=metrics(yy,pred); registry('temporal_2010_2020_extratrees',train,tf,'ExtraTrees','train-2010 / test-2020 aligned municipalities',tm,False,'in-sample diagnostic only; no earlier labelled wave for honest model fitting')
    write_registry()
    # factual deliverables
    key={'n_cross_section_2020':int(len(cs)),'n_balanced_2010_2020':int(balanced.code.nunique()),'n_features_ml':len(feats),'cross_oof_metrics':m,'temporal_diagnostic_metrics':tm,'note':'Temporal model is labelled diagnostic, not a valid held-out performance estimate because only two harmonized target waves are available.'}; (FINAL/'key_numbers.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf-8')
    (FINAL/'results_factual.md').write_text('\n'.join([f'- 2020 cross-section municipalities: {len(cs)}.',f'- Balanced SSDS panel (2010/2020): {balanced.code.nunique()} municipalities.',f'- Cross-sectional ExtraTrees 5-fold OOF RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}, R2={m["r2"]:.4f}.','- 2010→2020 is treated as COVID-included; the temporal diagnostic is explicitly not a held-out score.']),encoding='utf-8')
    (FINAL/'methodology_technical.md').write_text('API raw files are immutable. Municipality code is the join key. Target is H730102/H7301; share is non-exclusive because transport means can be multiple. PCA/clustering exclude all H730 transport variables. Pipelines fit imputation/scaling inside folds. Panel uses municipality and year fixed effects with municipality-clustered SE.\n',encoding='utf-8')
    (FINAL/'limitations_factual.md').write_text('SSDS H730 transport availability is 2010/2020 only. 2000 direct census extraction is retained separately because boundary/definition harmonization requires explicit validation. 2020 includes COVID-era behavior. Supply-side railway variables are not present. Associations are non-causal.\n',encoding='utf-8')
    (QA/'reproducibility.md').write_text(f'Python {sys.version}; platform {platform.platform()}. Re-run: python scripts/run_analysis.py. Raw inputs are not overwritten.\n',encoding='utf-8')
if __name__=='__main__': main()
