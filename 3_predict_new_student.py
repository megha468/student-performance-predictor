"""
STEP 9: Real-Time Inference & AI Recommendations
------------------------------------------------
Demonstrates command-line inference for student profiles with
prediction probabilities, risk assessment, and personalized advisory insights.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.student_predictor.advisor import generate_recommendations
from src.student_predictor.models import load_model_suite

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 75)
    print("[*] STUDENT PERFORMANCE PREDICTOR -- INFERENCE & ADVISORY")
    print("=" * 75)

    suite = load_model_suite(BASE_DIR)

    # Example student cohort profiles
    cohort = [
        {
            "name": "Alex Taylor (Dedicated Scholar)",
            "study_hours_per_week": 22.0,
            "attendance_percent": 94.0,
            "previous_score": 82.0,
            "sleep_hours": 7.5,
            "parental_education": "bachelors",
            "internet_access": "yes",
            "extracurricular": "yes",
        },
        {
            "name": "Jordan Lee (At-Risk Intervention Needed)",
            "study_hours_per_week": 4.5,
            "attendance_percent": 52.0,
            "previous_score": 38.0,
            "sleep_hours": 4.5,
            "parental_education": "high_school",
            "internet_access": "no",
            "extracurricular": "no",
        },
    ]

    for idx, student in enumerate(cohort, 1):
        name = student.pop("name")
        pred = suite.predict_single(student)
        recs = generate_recommendations(student, pred)

        print(f"\n[{idx}] Profile: {name}")
        print("-" * 65)
        print(
            f"  - Inputs             : {student['study_hours_per_week']}h/wk study | "
            f"{student['attendance_percent']}% att. | "
            f"{student['previous_score']} prior score | "
            f"{student['sleep_hours']}h sleep"
        )
        print(f"  - Predicted Score    : {pred['predicted_score']:.1f} / 100")
        print(
            f"  - Predicted Grade    : {pred['predicted_grade']} "
            f"(Probabilities: {pred['grade_probabilities']})"
        )
        print(
            f"  - Outcome            : {pred['predicted_passfail']} "
            f"(Confidence: {pred['passfail_probabilities'].get(pred['predicted_passfail'], 0.0):.1%})"
        )
        print(f"  - Risk Tier          : {pred['risk_tier']}")

        print("\n  >> AI Academic Action Plan:")
        for r in recs:
            print(f"     [{r['category']}] {r['action']}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
