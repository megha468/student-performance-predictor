"""
STEPS 2-8: Explore -> Clean -> Prepare -> Split -> Train -> Evaluate
---------------------------------------------------------------------
We train THREE models on the same student data:

  1. Regression model      -> predicts the numeric final_score (0-100)
  2. Classification model  -> predicts Pass / Fail
  3. Classification model  -> predicts letter Grade (A/B/C/D)

All three share the same input features (study hours, attendance, etc.)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file instead of popping up a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# Use the SAME folder this script is in (works on any computer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
OUT_DIR = BASE_DIR

# ============================================================
# STEP 2: LOAD THE DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("STEP 2: LOAD DATA")
print("=" * 60)
print(df.head())
print("\nInfo:")
print(df.info())

# ============================================================
# STEP 3: EXPLORE THE DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: EXPLORE DATA")
print("=" * 60)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# Correlation heatmap (numeric columns only) -> shows what relates to final_score
numeric_cols = ["study_hours_per_week", "attendance_percent", "previous_score",
                 "sleep_hours", "final_score"]
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between features and final score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_correlation_heatmap.png", dpi=120)
plt.close()

# Distribution of final scores
plt.figure(figsize=(7, 5))
sns.histplot(df["final_score"], bins=25, kde=True)
plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_score_distribution.png", dpi=120)
plt.close()

# Study hours vs final score
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="study_hours_per_week", y="final_score", hue="pass_fail")
plt.title("Study Hours vs Final Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_study_vs_score.png", dpi=120)
plt.close()

print("\nSaved 3 exploration plots (correlation heatmap, score distribution, study vs score).")

# ============================================================
# STEP 4: CLEAN & PREPARE THE DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CLEAN & PREPARE DATA")
print("=" * 60)

# No missing values in this dataset, but here's how you WOULD handle them:
# df = df.dropna()                              # drop rows with missing data, OR
# df["study_hours_per_week"] = df["study_hours_per_week"].fillna(df["study_hours_per_week"].mean())

# Encode categorical columns into numbers (one-hot encoding)
categorical_cols = ["parental_education", "internet_access", "extracurricular"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in df_encoded.columns
                 if c not in ["final_score", "pass_fail", "grade"]]

print("Final feature columns used for prediction:")
print(feature_cols)

X = df_encoded[feature_cols]

# ============================================================
# STEP 5 & 6: SPLIT INTO TRAIN / TEST SETS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5-6: TRAIN/TEST SPLIT")
print("=" * 60)

y_score = df_encoded["final_score"]
y_passfail = df_encoded["pass_fail"]
y_grade = df_encoded["grade"]

X_train, X_test, yscore_train, yscore_test, ypf_train, ypf_test, ygrade_train, ygrade_test = \
    train_test_split(X, y_score, y_passfail, y_grade, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} students")
print(f"Test set:     {X_test.shape[0]} students")

# ============================================================
# STEP 7: TRAIN THE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: TRAIN MODELS")
print("=" * 60)

# --- Model 1: Regression -> predict numeric final_score ---
reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
reg_model.fit(X_train, yscore_train)

# --- Model 2: Classification -> predict Pass/Fail ---
le_pf = LabelEncoder()
ypf_train_enc = le_pf.fit_transform(ypf_train)
ypf_test_enc = le_pf.transform(ypf_test)

clf_passfail = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf_passfail.fit(X_train, ypf_train_enc)

# --- Model 3: Classification -> predict Grade (A/B/C/D) ---
le_grade = LabelEncoder()
ygrade_train_enc = le_grade.fit_transform(ygrade_train)
ygrade_test_enc = le_grade.transform(ygrade_test)

clf_grade = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf_grade.fit(X_train, ygrade_train_enc)

print("All 3 models trained: score regressor, pass/fail classifier, grade classifier.")

# ============================================================
# STEP 8: EVALUATE THE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: EVALUATE MODELS")
print("=" * 60)

# --- Evaluate regression model ---
score_pred = reg_model.predict(X_test)
mae = mean_absolute_error(yscore_test, score_pred)
rmse = np.sqrt(mean_squared_error(yscore_test, score_pred))
r2 = r2_score(yscore_test, score_pred)

print("\n--- Score Prediction (Regression) ---")
print(f"MAE  (avg error in points): {mae:.2f}")
print(f"RMSE                      : {rmse:.2f}")
print(f"R^2 (how well it fits, 1=perfect): {r2:.3f}")

# --- Evaluate pass/fail model ---
pf_pred = clf_passfail.predict(X_test)
pf_acc = accuracy_score(ypf_test_enc, pf_pred)
print("\n--- Pass/Fail Prediction (Classification) ---")
print(f"Accuracy: {pf_acc:.2%}")
print(classification_report(ypf_test_enc, pf_pred, target_names=le_pf.classes_))

# --- Evaluate grade model ---
grade_pred = clf_grade.predict(X_test)
grade_acc = accuracy_score(ygrade_test_enc, grade_pred)
print("\n--- Grade Prediction (A/B/C/D Classification) ---")
print(f"Accuracy: {grade_acc:.2%}")
print(classification_report(ygrade_test_enc, grade_pred, target_names=le_grade.classes_))

# Confusion matrix plot for grade prediction
cm = confusion_matrix(ygrade_test_enc, grade_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_grade.classes_, yticklabels=le_grade.classes_)
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.title("Grade Prediction: Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_grade_confusion_matrix.png", dpi=120)
plt.close()

# Feature importance (what actually drives the final score)
importances = pd.Series(reg_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("What Predicts Final Score Most? (Feature Importance)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot_feature_importance.png", dpi=120)
plt.close()

print("\nSaved confusion matrix and feature importance plots.")

# ============================================================
# SAVE EVERYTHING NEEDED FOR PREDICTING NEW STUDENTS
# ============================================================
joblib.dump(reg_model, f"{OUT_DIR}/model_score_regressor.pkl")
joblib.dump(clf_passfail, f"{OUT_DIR}/model_passfail_classifier.pkl")
joblib.dump(clf_grade, f"{OUT_DIR}/model_grade_classifier.pkl")
joblib.dump(le_pf, f"{OUT_DIR}/encoder_passfail.pkl")
joblib.dump(le_grade, f"{OUT_DIR}/encoder_grade.pkl")
joblib.dump(feature_cols, f"{OUT_DIR}/feature_columns.pkl")

print("\nAll models and encoders saved to disk. Ready for predictions on new students!")
