# Reproducibility materials

This repository is the public, self-contained reproduction package for the
municipal railway-use research results. SSDSE-A-2026 is the canonical base
dataset for the 2020 cross-section; e-Stat SSDS is an explicitly labelled
extension, and Census extracts provide the railway outcome and historical
analysis inputs. The original research-project directory and filenames are
preserved under `research_project/`.

To reproduce the analyses, follow the instructions in
[`research_project/README.md`](research_project/README.md). The package contains
the immutable official SSDSE-A-2026 distribution, e-Stat source responses,
processing and analysis code, configuration,
and the generated factual research artifacts required for verification.

The frozen verification workflow is read-only: it rebuilds the package and checks
research invariants with `verify_artifacts.py`. It does not push generated files or
require byte-identical floating-point outputs across platforms.

This publication repository deliberately excludes development-only materials,
retrieval tooling that is not needed to replay the included data, local settings,
and Git history.
