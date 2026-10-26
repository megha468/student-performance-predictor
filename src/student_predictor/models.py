"""
Machine Learning Pipeline & Inference Engine
---------------------------------------------
Manages regression and classification models, feature transformation,
multi-class probability estimation, model benchmarking, and artifact persistence.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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

CATEGORICAL_COLS = ["parental_education", "internet_access", "extracurricular"]


DEFAULT_INPUTS = {
    "study_hours_per_week": 15.0,
    "attendance_percent": 80.0,
    "previous_score": 65.0,
    "sleep_hours": 7.0,
    "parental_education": "high_school",
    "internet_access": "yes",
    "extracurricular": "no",
}

DEFAULT_SUBJECTS: Dict[str, float] = {
    "Mathematics": 75.0,
    "Science": 70.0,
    "English": 80.0,
    "Social Science": 72.0,
    "Computer Science": 78.0,
}

SUBJECT_PROFILES: Dict[str, Dict[str, Any]] = {
    "Mathematics": {
        "icon": "📐",
        "weight": 1.15,
        "study_sensitivity": 0.22,
        "advice_strong": "Exemplary analytical reasoning! Continue solving advanced problem sets to maintain top percentile marks.",
        "advice_average": "Solid foundation. Increase daily targeted problem-solving by 20-30 mins to avoid exam calculation slips.",
        "advice_weak": "Prioritize algebra and geometry fundamentals with step-by-step worked examples before timed tests.",
    },
    "Science": {
        "icon": "🔬",
        "weight": 1.10,
        "study_sensitivity": 0.18,
        "advice_strong": "Outstanding conceptual mastery in scientific principles! Maintain lab experiment curiosity.",
        "advice_average": "Good scientific intuition. Review reaction formulas, physics derivations, and biology diagrams weekly.",
        "advice_weak": "Focus on core scientific definitions and practice chapter-end numerical questions.",
    },
    "English": {
        "icon": "📚",
        "weight": 0.85,
        "study_sensitivity": 0.10,
        "advice_strong": "Exceptional verbal and essay articulation! Excellent reading comprehension scores.",
        "advice_average": "Consistent comprehension. Read high-level literature and practice timed structured essays.",
        "advice_weak": "Dedicate 15 minutes each evening to vocabulary enrichment and grammar correction drills.",
    },
    "Social Science": {
        "icon": "🌍",
        "weight": 0.90,
        "study_sensitivity": 0.12,
        "advice_strong": "Remarkable grasp of historical chronologies, civics, and geographic systems.",
        "advice_average": "Steady recall. Use concept maps and timeline charts to retain historical events and case studies.",
        "advice_weak": "Build visual flashcards for key historical dates, constitutional terms, and geographic maps.",
    },
    "Computer Science": {
        "icon": "💻",
        "weight": 1.05,
        "study_sensitivity": 0.20,
        "advice_strong": "Superb algorithmic thinking and coding aptitude! Explore open-source projects or data structures.",
        "advice_average": "Strong theoretical understanding. Write and debug working code regularly instead of just reading theory.",
        "advice_weak": "Focus on basic algorithmic logic, conditional structures, and pseudocode debugging exercises.",
    },
}


def predict_subject_wise(
    overall_prediction: Dict[str, Any],
    subject_marks: Dict[str, float],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict upcoming marks for each individual subject based on previous year subject marks,
    overall ML model trajectory, study volume, attendance, and sleep habits.
    
    Returns a dictionary containing:
        - "subjects": Dict[subject_name, subject_result_dict]
        - "summary": Dict with strongest subject, weakest subject, average improvement, etc.
    """
    if not subject_marks:
        subject_marks = DEFAULT_SUBJECTS.copy()

    overall_pred_score = float(overall_prediction.get("predicted_score", 65.0))
    prev_scores = [float(v) for v in subject_marks.values()]
    prev_avg = sum(prev_scores) / len(prev_scores) if prev_scores else 65.0
    overall_delta = overall_pred_score - prev_avg

    study_hours = float(inputs.get("study_hours_per_week", 15.0))
    attendance = float(inputs.get("attendance_percent", 80.0))
    sleep = float(inputs.get("sleep_hours", 7.0))

    study_delta = (study_hours - 15.0) * 0.15
    att_delta = (attendance - 80.0) * 0.08
    sleep_penalty = -2.0 if sleep < 6.0 else (0.5 if 7.0 <= sleep <= 8.5 else 0.0)

    results: Dict[str, Dict[str, Any]] = {}

    for sub_name, prev_val in subject_marks.items():
        prev_s = float(prev_val)
        profile = SUBJECT_PROFILES.get(sub_name, {
            "icon": "📝",
            "weight": 1.0,
            "study_sensitivity": 0.15,
            "advice_strong": "Great job! Keep up your steady performance.",
            "advice_average": "Consistent effort. Practice more to reach top tier.",
            "advice_weak": "Needs focused attention and structured review.",
        })

        subj_momentum = overall_delta * profile["weight"]
        habit_effect = study_delta * (profile["study_sensitivity"] / 0.15) + att_delta + sleep_penalty

        raw_upcoming = prev_s + subj_momentum + habit_effect
        upcoming_score = round(max(0.0, min(100.0, raw_upcoming)), 1)
        diff = round(upcoming_score - prev_s, 1)

        if upcoming_score >= 85.0:
            grade = "A"
            advice = profile["advice_strong"]
        elif upcoming_score >= 70.0:
            grade = "B"
            advice = profile["advice_average"]
        elif upcoming_score >= 50.0:
            grade = "C"
            advice = profile["advice_average"]
        else:
            grade = "D"
            advice = profile["advice_weak"]

        passfail = "PASS" if upcoming_score >= 40.0 else "FAIL"

        if diff >= 2.0:
            trend = "Improving 🚀"
        elif diff <= -2.0:
            trend = "Declining ⚠️"
        else:
            trend = "Steady ⚖️"

        results[sub_name] = {
            "icon": profile["icon"],
            "previous_score": prev_s,
            "predicted_score": upcoming_score,
            "score_diff": diff,
            "predicted_grade": grade,
            "predicted_passfail": passfail,
            "trend": trend,
            "advice": advice,
        }

    # Smooth normalization so the mean of subjects tracks overall_pred_score
    subject_predicted_scores = [v["predicted_score"] for v in results.values()]
    mean_pred = sum(subject_predicted_scores) / len(subject_predicted_scores)
    discrepancy = overall_pred_score - mean_pred

    for sub_name in results:
        adjusted_score = round(max(0.0, min(100.0, results[sub_name]["predicted_score"] + discrepancy * 0.4)), 1)
        prev_s = results[sub_name]["previous_score"]
        diff = round(adjusted_score - prev_s, 1)
        results[sub_name]["predicted_score"] = adjusted_score
        results[sub_name]["score_diff"] = diff
        if adjusted_score >= 85.0:
            results[sub_name]["predicted_grade"] = "A"
        elif adjusted_score >= 70.0:
            results[sub_name]["predicted_grade"] = "B"
        elif adjusted_score >= 50.0:
            results[sub_name]["predicted_grade"] = "C"
        else:
            results[sub_name]["predicted_grade"] = "D"
        results[sub_name]["predicted_passfail"] = "PASS" if adjusted_score >= 40.0 else "FAIL"

    sorted_by_pred = sorted(results.items(), key=lambda item: item[1]["predicted_score"], reverse=True)
    strongest = sorted_by_pred[0]
    weakest = sorted_by_pred[-1]

    summary = {
        "strongest_subject": strongest[0],
        "strongest_score": strongest[1]["predicted_score"],
        "weakest_subject": weakest[0],
        "weakest_score": weakest[1]["predicted_score"],
        "average_previous": round(prev_avg, 1),
        "average_predicted": round(sum(v["predicted_score"] for v in results.values()) / len(results), 1),
    }

    return {
        "subjects": results,
        "summary": summary,
    }


class StudentModelSuite:
    """Encapsulates all trained models, encoders, and feature column schemas."""

    def __init__(
        self,
        reg_model: Any,
        clf_passfail: Any,
        clf_grade: Any,
        le_pf: LabelEncoder,
        le_grade: LabelEncoder,
        feature_cols: List[str],
    ):
        self.reg_model = reg_model
        self.clf_passfail = clf_passfail
        self.clf_grade = clf_grade
        self.le_pf = le_pf
        self.le_grade = le_grade
        self.feature_cols = feature_cols

    def preprocess_input(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """Transform raw student dictionary into aligned model feature vector with safe defaults."""
        full_inputs = {**DEFAULT_INPUTS, **inputs}
        df_single = pd.DataFrame([full_inputs])
        cats = [c for c in CATEGORICAL_COLS if c in df_single.columns]
        encoded = pd.get_dummies(df_single, columns=cats)
        aligned = encoded.reindex(columns=self.feature_cols, fill_value=0)
        return aligned

    def preprocess_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Transform batch dataframe into aligned model feature matrix with safe defaults."""
        df_copy = df_batch.copy()
        for k, v in DEFAULT_INPUTS.items():
            if k not in df_copy.columns:
                df_copy[k] = v
        cats = [c for c in CATEGORICAL_COLS if c in df_copy.columns]
        encoded = pd.get_dummies(df_copy, columns=cats)
        aligned = encoded.reindex(columns=self.feature_cols, fill_value=0)
        return aligned

    def predict_single(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce comprehensive prediction bundle for an individual student:
        predicted score, pass/fail status, letter grade, class probabilities, and risk tier.
        """
        features = self.preprocess_input(inputs)

        # Regression: final score (clipped to 0-100 range)
        raw_score = float(self.reg_model.predict(features)[0])
        score = round(max(0.0, min(100.0, raw_score)), 1)

        # Classification: Pass / Fail
        pf_pred_idx = self.clf_passfail.predict(features)[0]
        passfail = self.le_pf.inverse_transform([pf_pred_idx])[0]
        pf_proba = {}
        if hasattr(self.clf_passfail, "predict_proba"):
            probs = self.clf_passfail.predict_proba(features)[0]
            for cls_name, prob in zip(self.le_pf.classes_, probs):
                pf_proba[cls_name] = round(float(prob), 3)

        # Classification: Letter Grade (A/B/C/D)
        grade_pred_idx = self.clf_grade.predict(features)[0]
        grade = self.le_grade.inverse_transform([grade_pred_idx])[0]
        grade_proba = {}
        if hasattr(self.clf_grade, "predict_proba"):
            g_probs = self.clf_grade.predict_proba(features)[0]
            for cls_name, prob in zip(self.le_grade.classes_, g_probs):
                grade_proba[cls_name] = round(float(prob), 3)

        # Risk Tier calculation
        if score < 50.0 or passfail == "Fail":
            risk_tier = "High Risk"
            risk_color = "#EF4444"  # Red
        elif score < 70.0:
            risk_tier = "Moderate Risk"
            risk_color = "#F59E0B"  # Amber
        else:
            risk_tier = "Low Risk"
            risk_color = "#10B981"  # Emerald

        return {
            "predicted_score": score,
            "predicted_passfail": passfail,
            "predicted_grade": grade,
            "passfail_probabilities": pf_proba,
            "grade_probabilities": grade_proba,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
        }

    def predict_with_subjects(
        self,
        inputs: Dict[str, Any],
        subject_marks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive prediction bundle including overall metrics
        and fine-grained subject-wise predicted upcoming marks.
        """
        overall = self.predict_single(inputs)
        sub_bundle = predict_subject_wise(overall, subject_marks or {}, inputs)
        return {
            **overall,
            "subject_breakdown": sub_bundle,
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Run predictions across a dataframe of multiple students."""
        features = self.preprocess_batch(df_batch)
        scores = np.clip(self.reg_model.predict(features), 0.0, 100.0).round(1)
        pf_indices = self.clf_passfail.predict(features)
        passfail = self.le_pf.inverse_transform(pf_indices)
        grade_indices = self.clf_grade.predict(features)
        grades = self.le_grade.inverse_transform(grade_indices)

        result_df = df_batch.copy()
        result_df["predicted_score"] = scores
        result_df["predicted_passfail"] = passfail
        result_df["predicted_grade"] = grades
        result_df["risk_tier"] = np.where(
            scores < 50.0, "High Risk", np.where(scores < 70.0, "Moderate Risk", "Low Risk")
        )
        return result_df

    def get_feature_importances(self) -> Dict[str, float]:
        """Extract relative feature importance weights from the regression forest."""
        if hasattr(self.reg_model, "feature_importances_"):
            weights = self.reg_model.feature_importances_
            return dict(sorted(zip(self.feature_cols, weights), key=lambda x: x[1], reverse=True))
        return {}


def load_model_suite(models_dir: str) -> StudentModelSuite:
    """Load serialized models and encoders from directory."""
    reg_path = os.path.join(models_dir, "model_score_regressor.pkl")
    pf_path = os.path.join(models_dir, "model_passfail_classifier.pkl")
    grade_path = os.path.join(models_dir, "model_grade_classifier.pkl")
    le_pf_path = os.path.join(models_dir, "encoder_passfail.pkl")
    le_grade_path = os.path.join(models_dir, "encoder_grade.pkl")
    cols_path = os.path.join(models_dir, "feature_columns.pkl")

    for path in [reg_path, pf_path, grade_path, le_pf_path, le_grade_path, cols_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required model artifact at: {path}")

    reg_model = joblib.load(reg_path)
    clf_passfail = joblib.load(pf_path)
    clf_grade = joblib.load(grade_path)
    le_pf = joblib.load(le_pf_path)
    le_grade = joblib.load(le_grade_path)
    feature_cols = joblib.load(cols_path)

    return StudentModelSuite(
        reg_model=reg_model,
        clf_passfail=clf_passfail,
        clf_grade=clf_grade,
        le_pf=le_pf,
        le_grade=le_grade,
        feature_cols=feature_cols,
    )


def train_models(
    df: pd.DataFrame,
    out_dir: str,
    random_state: int = 42,
) -> Tuple[StudentModelSuite, Dict[str, Any]]:
    """
    Train regression, pass/fail classification, and grade classification models.
    Compares candidate architectures and exports serialized artifacts.
    """
    os.makedirs(out_dir, exist_ok=True)

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    feature_cols = [
        c for c in df_encoded.columns if c not in ["final_score", "pass_fail", "grade"]
    ]

    X = df_encoded[feature_cols]
    y_score = df_encoded["final_score"]
    y_pf = df_encoded["pass_fail"]
    y_grade = df_encoded["grade"]

    # Train / Test split
    X_train, X_test, yscore_train, yscore_test, ypf_train, ypf_test, ygrade_train, ygrade_test = (
        train_test_split(X, y_score, y_pf, y_grade, test_size=0.2, random_state=random_state)
    )

    # 1. Regressor (Score 0-100)
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf_reg.fit(X_train, yscore_train)

    score_pred = rf_reg.predict(X_test)
    mae = mean_absolute_error(yscore_test, score_pred)
    rmse = np.sqrt(mean_squared_error(yscore_test, score_pred))
    r2 = r2_score(yscore_test, score_pred)
    cv_r2 = cross_val_score(rf_reg, X_train, yscore_train, cv=5, scoring="r2").mean()

    # 2. Pass / Fail Classifier
    le_pf = LabelEncoder()
    ypf_train_enc = le_pf.fit_transform(ypf_train)
    ypf_test_enc = le_pf.transform(ypf_test)

    clf_passfail = RandomForestClassifier(
        n_estimators=200, random_state=random_state, class_weight="balanced"
    )
    clf_passfail.fit(X_train, ypf_train_enc)
    pf_pred = clf_passfail.predict(X_test)
    pf_acc = accuracy_score(ypf_test_enc, pf_pred)
    pf_f1 = f1_score(ypf_test_enc, pf_pred, average="weighted")

    # 3. Grade Classifier
    le_grade = LabelEncoder()
    ygrade_train_enc = le_grade.fit_transform(ygrade_train)
    ygrade_test_enc = le_grade.transform(ygrade_test)

    clf_grade = RandomForestClassifier(
        n_estimators=200, random_state=random_state, class_weight="balanced"
    )
    clf_grade.fit(X_train, ygrade_train_enc)
    grade_pred = clf_grade.predict(X_test)
    grade_acc = accuracy_score(ygrade_test_enc, grade_pred)
    grade_f1 = f1_score(ygrade_test_enc, grade_pred, average="weighted")

    # Serialize artifacts
    joblib.dump(rf_reg, os.path.join(out_dir, "model_score_regressor.pkl"))
    joblib.dump(clf_passfail, os.path.join(out_dir, "model_passfail_classifier.pkl"))
    joblib.dump(clf_grade, os.path.join(out_dir, "model_grade_classifier.pkl"))
    joblib.dump(le_pf, os.path.join(out_dir, "encoder_passfail.pkl"))
    joblib.dump(le_grade, os.path.join(out_dir, "encoder_grade.pkl"))
    joblib.dump(feature_cols, os.path.join(out_dir, "feature_columns.pkl"))

    suite = StudentModelSuite(
        reg_model=rf_reg,
        clf_passfail=clf_passfail,
        clf_grade=clf_grade,
        le_pf=le_pf,
        le_grade=le_grade,
        feature_cols=feature_cols,
    )

    metrics = {
        "regression": {"mae": mae, "rmse": rmse, "r2": r2, "cv_r2": cv_r2},
        "pass_fail": {"accuracy": pf_acc, "f1_weighted": pf_f1},
        "grade": {"accuracy": grade_acc, "f1_weighted": grade_f1},
    }

    return suite, metrics
