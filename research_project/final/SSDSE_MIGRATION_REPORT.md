# SSDSE-A-2026 Migration Report

This is the single generated fact source for paper revision. It does not modify the paper DOCX.

## 1. Final dataset provenance

- SSDSE-A official dataset: SSDSE-市区町村（SSDSE-A） (SSDSE-A-2026).
- Official source: https://www.nstac.go.jp/files/SSDSE-A-2026.csv; raw SHA-256: `1968d5c03de4db24c1ad7df20d06e7cb04bc46cb99a862364192d8d51c196409`.
- Raw municipalities: 1741; 2020 analysis municipalities: 1740.
- Exclusion detail is machine-readable in `qa/ssdse_analysis_exclusions.csv`; one canonical municipality has no saved H730 outcome.

## 2. Municipality universe migration

```csv
old_N,new_canonical_N,common,old_only,administrative_wards,other_noncanonical,SSDSE_only,final_analysis_N
1910,1740,1740,170,169,1,1,1740
```

## 3. Bugs corrected during SSDSE migration

- Corrected `foreign_population_share`, `single_household_share`, and `elderly_single_household_share` numerator/denominator definitions.
- Added derived-feature temporal leakage checks, year-compatible e-Stat crosschecks, full model-feature provenance verification, and independent derived-formula value checks.
- Removed stale legacy balanced-N and feature-count claims and fixed Core-membership generation in the paper fact package.

## 4. Final feature architecture

- Old reported feature count: 41; SSDSE Core: 29; Expanded: 69.
- SSDSE Core features: A1101, A1301, A1302, A1303, A1419, A1700, A7101, A710101, A810105, A8301, F1102, F1107, F1108, F2201, F2211, F2221, log_population, aging_share_65plus, share_75plus, youth_share_0_14, working_age_share_15_64, foreign_population_share, single_household_share, elderly_single_household_share, unemployment_rate, non_labor_force_share, primary_industry_share, secondary_industry_share, tertiary_industry_share.
- Expanded additions beyond Core: A141902, A130301, A710201, F110202, F110801, A141901, A130101, A130201, C310202, A8201, A110202, F110702, A1102, A110101, F110701, A130202, C310201, F110802, A110102, A110201, A130102, A130302, F110201, A810102, A811102, A1401, A1402, B1104, C120110, C120120, D2201, D2202, E9106, F1101, F1301, F1307, F1311, I5101, I5211, population_density.

## 5. SSDSE Core vs Expanded

```csv
model,split,n,n_features,rmse,mae,r2,spearman
SSDSE-A Core,prefecture_group_kfold,1740,29,0.050508,0.030580,0.758059,0.704322
SSDSE-A + e-Stat SSDS Expanded,prefecture_group_kfold,1740,69,0.049357,0.029074,0.768960,0.730601
```

## 6. Final model comparison

```csv
model,n,rmse,mae,r2,spearman
extra_trees,1740,0.049358,0.029075,0.768956,0.730580
hist_gbdt,1740,0.046839,0.028388,0.791936,0.713008
elastic_net,1740,0.048950,0.032153,0.772758,0.675804
xgboost,1740,0.045734,0.027568,0.801637,0.729840
catboost,1740,0.048636,0.028491,0.775661,0.731465
```

- Best R²/RMSE model: xgboost (R²=0.801637, RMSE=0.045734).

## 7. Permutation importance

```csv
feature,mean_importance,sd_across_folds,n_folds
E9106,0.056413,0.001995,5
F1301,0.021779,0.004887,5
D2201,0.008296,0.003096,5
C120110,0.005224,0.001339,5
F1307,0.004430,0.001400,5
F1102,0.002659,0.000610,5
A1302,0.001268,0.000538,5
A1401,0.001146,0.000317,5
A1301,0.001046,0.000514,5
A1402,0.000974,0.000245,5
```

## 8. Fractional logit

```csv
term,coefficient,pvalue,coef_p025,coef_p975,CI crosses zero,interpretation_status
const,-2.960938,0.000000,-3.238745,-2.726080,False,robust_negative
population_density,0.448015,0.000000,0.307909,0.909288,False,robust_positive
aging_share_65plus,-0.444369,0.005526,-0.711594,0.101633,True,point_estimate_negative_but_bootstrap_uncertain
working_age_share_15_64,0.105534,0.464568,-0.016727,0.467558,True,point_estimate_positive_but_bootstrap_uncertain
youth_share_0_14,-0.156713,0.024854,-0.505626,0.058789,True,point_estimate_negative_but_bootstrap_uncertain
```

- Population density: robust_positive; aging share: point_estimate_negative_but_bootstrap_uncertain.

## 9. GAM / nonlinear findings

The following saved GAM curve records are the numeric basis for any nonlinear wording:

```csv
population_density,prediction,lower,upper
0.012908,0.014079,0.010212,0.017947
2.354403,0.032547,0.029580,0.035514
4.695898,0.049800,0.046029,0.053571
7.037393,0.065992,0.061282,0.070703
9.378888,0.081278,0.075766,0.086790
11.720383,0.095810,0.089658,0.101962
14.061877,0.109742,0.103175,0.116308
16.403372,0.123115,0.116298,0.129932
18.744867,0.135706,0.128503,0.142909
21.086362,0.147252,0.139373,0.155131
23.427857,0.157491,0.148774,0.166208
25.769352,0.166161,0.156723,0.175598
28.110847,0.173005,0.163203,0.182807
30.452342,0.178159,0.168260,0.188057
32.793837,0.182328,0.172214,0.192442
35.135332,0.186262,0.175598,0.196927
37.476827,0.190715,0.179269,0.202162
39.818322,0.196437,0.184301,0.208574
42.159817,0.204145,0.191729,0.216561
44.501312,0.213842,0.201499,0.226186
```

## 10. Longitudinal analysis

- Strict exact-code universe N=1445; train N=1445; test N=1316.
- Test N is smaller because held-out transition rows with a missing outcome or required predictor are not evaluable; do not call all strict-universe municipalities test observations.
- SSDSE-A-2026 is not retrojected into historical years.

```csv
model,task,n_train,n_test,rmse,mae,r2,spearman
persistence,level,1445,1316,0.020677,0.013796,0.980806,0.978706
mean_historical_change,level,1445,1316,0.019908,0.013079,0.982205,0.978706
ridge,level,1445,1316,0.163941,0.162920,-0.206689,0.977917
elastic_net,level,1445,1316,0.053647,0.050891,0.870786,0.980643
extra_trees,level,1445,1316,0.017737,0.012096,0.985876,0.978626
delta_ridge,delta_direct,1445,1316,0.143529,0.142389,0.075087,0.979397
delta_elastic_net,delta_direct,1445,1316,0.040203,0.037080,0.927434,0.980231
delta_extra_trees,delta_direct,1445,1316,0.017624,0.012317,0.986054,0.978955
```

- Change-score N=1740; first-difference N=1740.

## 11. Residual analysis

- Analysis N=1740; residual records=1163.
- Largest negative and positive consensus residual records (municipality codes; use names only if joined to an authoritative name artifact):

```csv
code,consensus_residual,residual_sign_agreement
47205,-0.251956,1.000000
47208,-0.240004,1.000000
47201,-0.239897,1.000000
14301,-0.219354,1.000000
47211,-0.197465,1.000000
13101,0.305378,1.000000
27301,0.260352,1.000000
29342,0.244620,1.000000
13113,0.198080,1.000000
27221,0.168763,1.000000
```

## 12. Old paper -> new paper update matrix

```csv
paper_item,old_value_or_claim,new_value_or_claim,source_artifact,mandatory_change,reason
"全国1,910市区町村","1,910 municipalities",1740,qa/ssdse_municipality_universe_summary.json,True,Canonical SSDSE municipality universe and observed target exclusion
"2020横断1,910","N=1,910",1740,final/key_numbers.json,True,Regenerated canonical analysis frame
"均衡1,906","N=1,906",1740,final/key_numbers.json,True,Canonical-code 2010/2020 intersection
41特徴量,41,69,qa/model_feature_provenance_audit.csv,True,Expanded model now has fully audited features
旧GroupKFold指標,legacy values,Use regenerated OOF model comparison,analysis/robustness/oof_model_metrics.csv,True,Canonical outcome universe
旧重要度順位,legacy ranking,Use regenerated top-10 permutation ranking,analysis/interpretation/oof_permutation_importance.csv,True,Canonical model rerun
旧fractional logit係数,legacy coefficients,Use regenerated municipality-equal coefficients,analysis/bounded/fractional_logit_coefficients.csv,True,Canonical analysis frame
旧bootstrap CI,legacy intervals,Use regenerated prefecture-cluster bootstrap intervals,analysis/bootstrap/bootstrap_summary.csv,True,Canonical analysis frame
旧temporal N,legacy temporal N,strict=1445; train=1445; test=1316,final/key_numbers.json,True,Strict cohort and evaluable holdout rows differ
旧temporal metrics,legacy metrics,Use regenerated temporal validation table,analysis/temporal/temporal_validation_enhanced.csv,True,Retained exact-code historical panel
高齢化率の「頑健な負」主張,robust negative,point_estimate_negative_but_bootstrap_uncertain,analysis/bootstrap/bootstrap_summary.csv,True,Bootstrap CI must determine robustness wording
```

Machine-readable companions: `final/paper_update_matrix.csv` and `final/paper_update_facts.json`.

## QA status

- SSDSE-A is the canonical 2020 base; e-Stat SSDS is an extension; Census provides the outcome/historical source.
- Reproduce with `python scripts/run_all.py` followed by `python scripts/verify_artifacts.py`.
