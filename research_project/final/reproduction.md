# Reproduction record

Raw e-Stat API responses and request manifests are preserved under `data/raw/`; analysis scripts do not overwrite them.

## Clean-environment execution

1. Create and activate a Python 3.11+ environment.
2. Run `pip install -r requirements.txt`.
3. From `research_project`, run `python scripts/run_all.py`.
4. Run `python scripts/verify_artifacts.py`. It must emit the invariant JSON and write `qa/final_verification.json` with `status: pass`.

`run_all.py` rebuilds processed panels, figures, model records, QA, factual indexes, enhanced feature dictionary, and the de-duplicated registry from preserved raw inputs. Randomized estimators use `random_state=2026`; hardware-dependent floating-point variation can cause minor metric differences. No API retrieval is needed for this replay.
