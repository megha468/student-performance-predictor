"""
STEPS 2-8: Machine Learning Training & Evaluation Pipeline
---------------------------------------------------------
Loads student dataset, generates exploratory visualizations, trains
regression and multi-class classification models, generates performance
diagnostics, and saves serialized model artifacts.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
OUT_DIR = BASE_DIR

# Set clean aesthetic plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def main():
    print("=" * 70)
    print("🚀 STUDENT PERFORMANCE PREDICTOR — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------
    # STEP 2: LOAD DATA
    # ------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        from src.student_predictor.data import generate_dataset
        print("[INFO] student_data.csv not found. Auto-generating dataset...")
        generate_dataset(n_samples=1000, random_state=42, output_path=DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print(f"\n[STEP 2] Loaded dataset from {DATA_PATH} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # ------------------------------------------------------------
    # STEP 3: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
    # ------------------------------------------------------------
    print("\n[STEP 3] Generating Exploratory Visualizations...")

    # 1. Correlation Heatmap
    numeric_cols = [
        "study_hours_per_week",
        "attendance_percent",
        "previous_score",
        "sleep_hours",
        "final_score",
    ]
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix: Behavioral Factors vs Final Score", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_correlation_heatmap.png"), dpi=150)
    plt.close()

    # 2. Final Score Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["final_score"], bins=25, kde=True, color="#4F46E5", edgecolor="white", alpha=0.7)
    plt.axvline(df["final_score"].mean(), color="#EF4444", linestyle="--", label=f"Mean: {df['final_score'].mean():.1f}")
    plt.axvline(85, color="#10B981", linestyle=":", label="Grade A Threshold (85)")
    plt.axvline(50, color="#F59E0B", linestyle=":", label="Passing Threshold (50)")
    plt.title("Distribution of Final Student Scores", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Final Score (0-100)")
    plt.ylabel("Student Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_score_distribution.png"), dpi=150)
    plt.close()

    # 3. Study Hours vs Final Score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="study_hours_per_week",
        y="final_score",
        hue="pass_fail",
        palette={"Pass": "#10B981", "Fail": "#EF4444"},
        alpha=0.8,
        s=50,
    )
    plt.title("Weekly Study Hours vs. Final Exam Performance", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Study Hours Per Week")
    plt.ylabel("Final Score (0-100)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_study_vs_score.png"), dpi=150)
    plt.close()

    print("  -> Saved correlation heatmap, score distribution, and study vs score plots.")

    # ------------------------------------------------------------
    # STEP 4: DATA PREPARATION & ENCODING
    # ------------------------------------------------------------
    print("\n[STEP 4] Preprocessing & Encoding Features...")
    categorical_cols = ["parental_education", "internet_access", "extracurricular"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    feature_cols = [
        c for c in df_encoded.columns if c not in ["final_score", "pass_fail", "grade"]
    ]
    X = df_encoded[feature_cols]
    y_score = df_encoded["final_score"]
    y_passfail = df_encoded["pass_fail"]
    y_grade = df_encoded["grade"]

    # ------------------------------------------------------------
    # STEP 5 & 6: TRAIN / TEST SPLIT
    # ------------------------------------------------------------
    X_train, X_test, yscore_train, yscore_test, ypf_train, ypf_test, ygrade_train, ygrade_test = (
        train_test_split(X, y_score, y_passfail, y_grade, test_size=0.2, random_state=42)
    )
    print(f"  -> Split: Training={len(X_train)} samples, Testing={len(X_test)} samples")

    # ------------------------------------------------------------
    # STEP 7: TRAIN MODELS
    # ------------------------------------------------------------
    print("\n[STEP 7] Training Machine Learning Models...")

    # Model 1: Score Regressor
    reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_model.fit(X_train, yscore_train)

    # Model 2: Pass/Fail Classifier
    le_pf = LabelEncoder()
    ypf_train_enc = le_pf.fit_transform(ypf_train)
    ypf_test_enc = le_pf.transform(ypf_test)
    clf_passfail = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf_passfail.fit(X_train, ypf_train_enc)

    # Model 3: Grade Classifier
    le_grade = LabelEncoder()
    ygrade_train_enc = le_grade.fit_transform(ygrade_train)
    ygrade_test_enc = le_grade.transform(ygrade_test)
    clf_grade = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf_grade.fit(X_train, ygrade_train_enc)

    print("  -> Trained Random Forest Regressor, Pass/Fail Classifier, and Grade Classifier.")

    # ------------------------------------------------------------
    # STEP 8: EVALUATION & DIAGNOSTICS
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STEP 8] MODEL EVALUATION REPORT")
    print("=" * 70)

    # Regression Metrics
    score_pred = reg_model.predict(X_test)
    mae = mean_absolute_error(yscore_test, score_pred)
    rmse = np.sqrt(mean_squared_error(yscore_test, score_pred))
    r2 = r2_score(yscore_test, score_pred)
    cv_r2 = cross_val_score(reg_model, X_train, yscore_train, cv=5, scoring="r2").mean()

    print(f"\n📊 1. Score Prediction (Regression):")
    print(f"   • Mean Absolute Error (MAE) : {mae:.2f} points")
    print(f"   • Root Mean Squared Error   : {rmse:.2f} points")
    print(f"   • Test R² Score             : {r2:.3f}")
    print(f"   • 5-Fold Cross-Val R²       : {cv_r2:.3f}")

    # Pass/Fail Metrics
    pf_pred = clf_passfail.predict(X_test)
    pf_acc = accuracy_score(ypf_test_enc, pf_pred)
    pf_f1 = f1_score(ypf_test_enc, pf_pred, average="weighted")
    print(f"\n🎯 2. Pass / Fail Prediction (Binary Classification):")
    print(f"   • Accuracy                  : {pf_acc:.2%}")
    print(f"   • Weighted F1-Score         : {pf_f1:.3f}")

    # Grade Metrics
    grade_pred = clf_grade.predict(X_test)
    grade_acc = accuracy_score(ygrade_test_enc, grade_pred)
    grade_f1 = f1_score(ygrade_test_enc, grade_pred, average="weighted")
    print(f"\n🏆 3. Grade Prediction (Multi-Class Classification A/B/C/D):")
    print(f"   • Accuracy                  : {grade_acc:.2%}")
    print(f"   • Weighted F1-Score         : {grade_f1:.3f}")
    print("\nDetailed Grade Classification Report:")
    print(classification_report(ygrade_test_enc, grade_pred, target_names=le_grade.classes_))

    # Confusion Matrix Plot
    cm = confusion_matrix(ygrade_test_enc, grade_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le_grade.classes_,
        yticklabels=le_grade.classes_,
    )
    plt.xlabel("Predicted Grade", fontsize=11, weight="bold")
    plt.ylabel("Actual Grade", fontsize=11, weight="bold")
    plt.title("Grade Prediction: Confusion Matrix", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_grade_confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature Importance Plot
    importances = pd.Series(reg_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    colors = ["#6366F1" if i >= 0.1 else "#94A3B8" for i in importances.values]
    plt.barh(importances.index, importances.values, color=colors)
    plt.title("Feature Importance Weights (Predicting Final Exam Score)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Relative Importance Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_feature_importance.png"), dpi=150)
    plt.close()

    print("  -> Saved confusion matrix and feature importance plots.")

    # ------------------------------------------------------------
    # SAVE MODEL ARTIFACTS
    # ------------------------------------------------------------
    joblib.dump(reg_model, os.path.join(OUT_DIR, "model_score_regressor.pkl"))
    joblib.dump(clf_passfail, os.path.join(OUT_DIR, "model_passfail_classifier.pkl"))
    joblib.dump(clf_grade, os.path.join(OUT_DIR, "model_grade_classifier.pkl"))
    joblib.dump(le_pf, os.path.join(OUT_DIR, "encoder_passfail.pkl"))
    joblib.dump(le_grade, os.path.join(OUT_DIR, "encoder_grade.pkl"))
    joblib.dump(feature_cols, os.path.join(OUT_DIR, "feature_columns.pkl"))

    print("\n" + "=" * 70)
    print("✅ All model artifacts serialized and saved successfully to project root.")
    print("=" * 70)


if __name__ == "__main__":
    main()
