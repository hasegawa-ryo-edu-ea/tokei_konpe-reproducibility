#!/usr/bin/env python3
"""Resume regeneration after the SSDSE base frame has been rebuilt."""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[1]
ORDER=['longitudinal_analysis.py','interpret_models.py','xgboost_comparison.py','catboost_comparison.py','quality_strengthening.py','build_provenance.py','enhance_dictionary.py','semantic_audit.py','normalize_registry.py','build_final_records.py','consistency_check.py','build_ssdse_migration_report.py']
for script in ORDER:
 print(f'== {script} ==',flush=True)
 subprocess.run([sys.executable,str(R/'scripts'/script)],cwd=R,check=True)
