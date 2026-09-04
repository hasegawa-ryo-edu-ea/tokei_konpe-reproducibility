#!/usr/bin/env python3
"""Remove only regenerable outputs; immutable e-Stat raw inputs are never touched."""
from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[1]
for name in ['processed','analysis','tables','figures','final','qa','runs']:
 p=R/name
 if p.exists(): shutil.rmtree(p)
for p in R.glob('*_stdout.log'): p.unlink()
for p in R.glob('*_stderr.log'): p.unlink()
print('removed regenerable artifacts; data/raw and source code preserved')
