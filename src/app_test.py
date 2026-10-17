"""Run the 30 application test cases through the same logic the web app uses.

Run from the project root (venv active):
    python src/app_test.py
Reads data/raw/app_tests.csv (id,group,response_text,expected,notes) and writes results/app_tests_results.csv.

Expected values: one of the six labels, "out_of_scope" (app should decline), or "not_sure" (confidence below 0.5).

Columns written: id, group, response_text, expected, predicted, confidence, shown_as, outcome_matches_expected,
high_conf_error, feedback_shown, feedback_appropriate (YOU fill this in after reading feedback_shown).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import (CONFIDENCE_THRESHOLD, DISPLAY_NAMES, FEEDBACK, LOW_CONFIDENCE_FEEDBACK,  # noqa: E402
                      OUT_OF_SCOPE_FEEDBACK, classifier, looks_in_scope, prepare)

HIGH_CONF = 0.80
SRC = ROOT / "data" / "raw" / "app_tests.csv"
OUT = ROOT / "results" / "app_tests_results.csv"


def diagnose(text):
    """Mirror of /predict in app/main.py: returns (predicted, confidence, shown_as, feedback)."""
    if not looks_in_scope(text):
        return "out_of_scope", 0.0, "out_of_scope", OUT_OF_SCOPE_FEEDBACK
    scores = classifier.scores(prepare(text))
    label, conf = max(scores.items(), key=lambda kv: kv[1])
    if conf >= CONFIDENCE_THRESHOLD:
        return label, round(conf, 3), label, FEEDBACK.get(label, "")
    return label, round(conf, 3), "not_sure", LOW_CONFIDENCE_FEEDBACK


def main():
    if not SRC.exists():
        sys.exit(f"Not found: {SRC}")
    df = pd.read_csv(SRC).fillna("")
    classifier.load()
    rows = []
    for r in df.itertuples():
        pred, conf, shown, fb = diagnose(r.response_text)
        matches = shown == r.expected
        # a confident, wrong diagnosis (or a confident answer to an ambiguous/off-topic input) is the worst case
        high_conf_error = (not matches) and shown not in ("not_sure", "out_of_scope") and conf >= HIGH_CONF
        rows.append({
            "id": r.id, "group": r.group, "response_text": r.response_text, "expected": r.expected,
            "predicted": pred, "confidence": conf, "shown_as": shown,
            "outcome_matches_expected": "yes" if matches else "no",
            "high_conf_error": "YES" if high_conf_error else "",
            "feedback_shown": fb, "feedback_appropriate": "", "notes": r.notes,
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    print(out[["id", "expected", "shown_as", "confidence", "outcome_matches_expected", "high_conf_error"]]
          .to_string(index=False))
    print("\nOutcome matches expected, by group:")
    print(out.groupby("group")["outcome_matches_expected"].apply(lambda s: f"{(s == 'yes').sum()}/{len(s)}")
          .to_string())
    print(f"\nOverall: {(out['outcome_matches_expected'] == 'yes').sum()}/{len(out)}")
    hc = out[out["high_conf_error"] == "YES"]
    print(f"Wrong answers shown with confidence >= {HIGH_CONF}: {len(hc)}")
    for r in hc.itertuples():
        print(f"   {r.id}: expected {r.expected}, app said {r.shown_as} ({r.confidence})  <- {r.response_text}")
    print(f"\nSaved {OUT}. Open it and fill in feedback_appropriate (yes/no) after reading feedback_shown.")


if __name__ == "__main__":
    main()
