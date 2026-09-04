#!/usr/bin/env python3
"""GAM curves and factual EDA for density and demographic composition."""
from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt,seaborn as sns
from pygam import LinearGAM,s
R=Path(__file__).resolve().parents[1];P=R/'processed';A=R/'analysis';F=R/'figures';T=R/'tables'
for d in [A/'gam',F,T]:d.mkdir(parents=True,exist_ok=True)
def main():
 d=pd.read_parquet(P/'municipality_cross_section_2020.parquet').dropna(subset=['rail_any_share'])
 rows=[]
 for feat in ['population_density','aging_share_65plus','working_age_share_15_64','youth_share_0_14']:
  z=d[[feat,'rail_any_share']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)<100:continue
  g=LinearGAM(s(0)).fit(z[[feat]],z.rail_any_share);grid=g.generate_X_grid(term=0);ci=g.confidence_intervals(grid); out=pd.DataFrame({'feature':feat,'x':grid[:,0],'prediction':g.predict(grid),'lower':ci[:,0],'upper':ci[:,1]});out.to_csv(A/f'gam/{feat}_curve.csv',index=False)
  rows.append({'feature':feat,'n':len(z),'edof':float(g.statistics_['edof']),'pseudo_r2':float(g.statistics_['pseudo_r2']['explained_deviance'])})
  plt.figure(figsize=(5.2,3.3));plt.plot(out.x,out.prediction*100);plt.fill_between(out.x,out.lower*100,out.upper*100,alpha=.2);plt.xlabel(feat);plt.ylabel('Predicted rail share (%)');plt.tight_layout();plt.savefig(F/f'gam_{feat}.png',dpi=180);plt.close()
 pd.DataFrame(rows).to_csv(T/'gam_summary.csv',index=False)
 sns.set_theme(style='whitegrid');
 for feat in ['population_density','aging_share_65plus']:
  z=d[[feat,'rail_any_share']].replace([np.inf,-np.inf],np.nan).dropna();plt.figure(figsize=(5.2,3.3));plt.scatter(z[feat],z.rail_any_share*100,s=5,alpha=.25);plt.xlabel(feat);plt.ylabel('Rail share (%)');plt.tight_layout();plt.savefig(F/f'eda_{feat}_scatter.png',dpi=180);plt.close()
if __name__=='__main__':main()
