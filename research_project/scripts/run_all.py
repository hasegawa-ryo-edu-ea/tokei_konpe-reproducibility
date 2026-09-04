#!/usr/bin/env python3
"""One-command regeneration of all non-raw research artifacts."""
from pathlib import Path
import os,subprocess,sys
R=Path(__file__).resolve().parents[1]
ORDER=['clean_generated.py','run_analysis.py','ssdse_integration.py','nonlinear_eda.py','post_analysis.py','longitudinal_analysis.py','interpret_models.py','xgboost_comparison.py','catboost_comparison.py','quality_strengthening.py','build_provenance.py','enhance_dictionary.py','semantic_audit.py','audit_model_feature_provenance.py','normalize_registry.py','build_final_records.py','final_wording_fix.py','consistency_check.py','build_ssdse_migration_report.py']
def main():
 env=os.environ.copy();env['MPLBACKEND']='Agg'
 for script in ORDER:
  print(f'== {script} ==',flush=True)
  subprocess.run([sys.executable,str(R/'scripts'/script)],cwd=R,check=True,env=env)
if __name__=='__main__':main()
