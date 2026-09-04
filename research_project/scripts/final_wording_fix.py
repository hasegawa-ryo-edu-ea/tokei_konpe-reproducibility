#!/usr/bin/env python3
"""Apply final factual wording clarifications after artifact generation."""
from pathlib import Path
import json

R = Path(__file__).resolve().parents[1]
F = R / "final"
Q = R / "qa"

# Clarify what the temporal evaluation can and cannot establish.
key_path = F / "key_numbers.json"
key = json.loads(key_path.read_text(encoding="utf-8"))
key["note"] = (
    "Strict temporal analysis uses a single chronological future-period holdout: "
    "fit on 2000->2010 and evaluate on 2010->2020 for the same harmonized municipality cohort. "
    "This supports one-period temporal validation, but not generalization across multiple independent periods or unseen municipalities."
)
key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

# Clarify that 3+ mode cases remain in the official denominator and are omitted only from the rail numerator.
target_path = Q / "temporal_target_definition.json"
target = json.loads(target_path.read_text(encoding="utf-8"))
target["definition"] = (
    "Resident-based age-15+ out-of-home worker/student transport universe; numerator is rail-only plus explicitly "
    "rail-containing two-mode categories. Three-or-more-mode cases are not counted in the rail numerator because "
    "rail membership is not uniformly identifiable across waves; the official total remains the denominator."
)
target.pop("excluded_all_years", None)
target["numerator_exclusion_all_years"] = (
    "Three or more transport modes are excluded from the rail numerator only; they remain in the official denominator total."
)
target_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")

print("final factual wording clarified")
