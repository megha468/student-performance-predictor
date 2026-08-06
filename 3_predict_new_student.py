"""
STEP 9: PREDICT FOR A NEW STUDENT
-----------------------------------
Loads the trained models and predicts:
  - Final Score (number)
  - Pass / Fail
  - Letter Grade (A/B/C/D)

for one or more new students you describe below.

HOW TO USE:
Just edit the `new_students` list below with real values, then run:
    python3 3_predict_new_student.py
"""

import os
import pandas as pd
import joblib

# Use the SAME folder this script is in (works on any computer)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load everything we saved during training
reg_model = joblib.load(f"{OUT_DIR}/model_score_regressor.pkl")
clf_passfail = joblib.load(f"{OUT_DIR}/model_passfail_classifier.pkl")
clf_grade = joblib.load(f"{OUT_DIR}/model_grade_classifier.pkl")
le_pf = joblib.load(f"{OUT_DIR}/encoder_passfail.pkl")
le_grade = joblib.load(f"{OUT_DIR}/encoder_grade.pkl")
feature_cols = joblib.load(f"{OUT_DIR}/feature_columns.pkl")

# ---- EDIT THIS: describe the new student(s) you want to predict for ----
new_students = [
    {
        "study_hours_per_week": 20,
        "attendance_percent": 90,
        "previous_score": 75,
        "sleep_hours": 7.5,
        "parental_education": "bachelors",   # high_school / bachelors / masters / phd
        "internet_access": "yes",             # yes / no
        "extracurricular": "no",              # yes / no
    },
    {
        "study_hours_per_week": 5,
        "attendance_percent": 55,
        "previous_score": 40,
        "sleep_hours": 5.0,
        "parental_education": "high_school",
        "internet_access": "no",
        "extracurricular": "no",
    },
]
# --------------------------------------------------------------------

new_df = pd.DataFrame(new_students)

# Encode the same way training data was encoded
new_encoded = pd.get_dummies(
    new_df, columns=["parental_education", "internet_access", "extracurricular"]
)

# Make sure it has exactly the same columns as training data (fill missing with 0)
new_encoded = new_encoded.reindex(columns=feature_cols, fill_value=0)

# Predict
scores = reg_model.predict(new_encoded)
passfail = le_pf.inverse_transform(clf_passfail.predict(new_encoded))
grades = le_grade.inverse_transform(clf_grade.predict(new_encoded))

for i, student in enumerate(new_students):
    print(f"\nStudent {i + 1}: {student}")
    print(f"  -> Predicted Final Score : {scores[i]:.1f} / 100")
    print(f"  -> Predicted Pass/Fail   : {passfail[i]}")
    print(f"  -> Predicted Grade       : {grades[i]}")
