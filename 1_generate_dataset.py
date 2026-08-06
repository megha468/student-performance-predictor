"""
STEP 1: Generate the dataset
-----------------------------
This creates a realistic, synthetic "student performance" dataset with the
same kind of columns you'd find in the real UCI Student Performance dataset
or the Kaggle "Students Performance in Exams" dataset:

    study_hours_per_week   - how many hours/week the student studies
    attendance_percent     - % of classes attended
    previous_score         - score in the last exam (0-100)
    parental_education     - highest education level of parents
    internet_access        - yes/no, has internet at home
    extracurricular        - yes/no, participates in extracurricular activities
    sleep_hours            - average hours of sleep per night
    final_score             - TARGET (0-100), what we want to predict

We build final_score from a realistic formula (study time, attendance and
past performance matter most) plus random noise, so the patterns a model
finds are believable, not random.
"""

import os
import numpy as np
import pandas as pd

# Save files in the SAME folder this script is in (works on any computer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
N = 1000  # number of students

# --- Features ---
study_hours_per_week = np.clip(np.random.normal(15, 6, N), 0, 40)
attendance_percent = np.clip(np.random.normal(80, 12, N), 30, 100)
previous_score = np.clip(np.random.normal(65, 15, N), 0, 100)

parental_education = np.random.choice(
    ["high_school", "bachelors", "masters", "phd"],
    size=N, p=[0.4, 0.35, 0.2, 0.05]
)
internet_access = np.random.choice(["yes", "no"], size=N, p=[0.85, 0.15])
extracurricular = np.random.choice(["yes", "no"], size=N, p=[0.5, 0.5])
sleep_hours = np.clip(np.random.normal(7, 1.3, N), 3, 10)

# Map parental education to a numeric "boost" for score formula
edu_boost = pd.Series(parental_education).map(
    {"high_school": 0, "bachelors": 2, "masters": 4, "phd": 6}
).values
internet_boost = pd.Series(internet_access).map({"yes": 3, "no": 0}).values
extracurricular_boost = pd.Series(extracurricular).map({"yes": 1, "no": 0}).values

# --- Target: final_score ---
# Study hours and previous score matter most, attendance matters a lot too.
final_score = (
    0.9 * previous_score * 0.4
    + study_hours_per_week * 1.3
    + attendance_percent * 0.25
    + edu_boost
    + internet_boost
    + extracurricular_boost
    - np.abs(sleep_hours - 7.5) * 2       # too little/too much sleep hurts slightly
    + np.random.normal(0, 6, N)           # random noise (life happens)
)
final_score = np.clip(final_score, 0, 100).round(1)

df = pd.DataFrame({
    "study_hours_per_week": study_hours_per_week.round(1),
    "attendance_percent": attendance_percent.round(1),
    "previous_score": previous_score.round(1),
    "parental_education": parental_education,
    "internet_access": internet_access,
    "extracurricular": extracurricular,
    "sleep_hours": sleep_hours.round(1),
    "final_score": final_score,
})

# Derived labels used later:
# Pass/Fail -> pass if final_score >= 40
df["pass_fail"] = np.where(df["final_score"] >= 40, "Pass", "Fail")

# Letter grade -> A/B/C/D based on common thresholds
def grade(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"

df["grade"] = df["final_score"].apply(grade)

df.to_csv(os.path.join(BASE_DIR, "student_data.csv"), index=False)
print("Dataset created: student_data.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nGrade distribution:\n", df["grade"].value_counts())
print("\nPass/Fail distribution:\n", df["pass_fail"].value_counts())
