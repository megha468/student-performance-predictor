"""
STEP 1: Generate Realistic Student Performance Dataset
------------------------------------------------------
Generates a realistic, synthetic student performance dataset with behavioral,
lifestyle, and demographic factors, then exports to `student_data.csv`.

Columns:
  - study_hours_per_week: Weekly study volume (0-40)
  - attendance_percent:   Class attendance rate (30-100%)
  - previous_score:       Previous exam score (0-100)
  - sleep_hours:          Average nightly sleep (3-10)
  - parental_education:   high_school / bachelors / masters / phd
  - internet_access:      yes / no
  - extracurricular:      yes / no
  - final_score:          Target continuous score (0-100)
  - pass_fail:            Pass (>=40) / Fail (<40)
  - grade:                A (>=85), B (70-84.9), C (50-69.9), D (<50)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.student_predictor.data import generate_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "student_data.csv")


def main():
    print("=" * 70)
    print("🎓 STUDENT PERFORMANCE PREDICTOR — DATASET GENERATOR")
    print("=" * 70)

    print("\n[INFO] Synthesizing 1,000 student behavioral records...")
    df = generate_dataset(n_samples=1000, random_state=42, output_path=OUTPUT_FILE)

    print(f"[SUCCESS] Saved dataset -> {OUTPUT_FILE}")
    print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("-" * 50)
    print("Dataset Sample (First 5 Rows):")
    print("-" * 50)
    print(df.head())

    print("\n" + "-" * 50)
    print("Letter Grade Distribution:")
    print("-" * 50)
    print(df["grade"].value_counts())

    print("\n" + "-" * 50)
    print("Pass/Fail Outcome Distribution:")
    print("-" * 50)
    print(df["pass_fail"].value_counts())

    print("\n[DONE] Dataset generation completed successfully.")


if __name__ == "__main__":
    main()
