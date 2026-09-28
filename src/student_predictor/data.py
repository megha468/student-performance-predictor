"""
Data Generation, Cleaning, and Preparation Module
-------------------------------------------------
Provides utilities to generate realistic synthetic student datasets,
load and validate data, and split into train/test sets.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLS = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "parental_education",
    "internet_access",
    "extracurricular",
]

NUMERIC_COLS = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
]

CATEGORICAL_COLS = [
    "parental_education",
    "internet_access",
    "extracurricular",
]


def assign_grade(score: float) -> str:
    """Map a continuous score (0-100) to an academic letter grade."""
    if score >= 85.0:
        return "A"
    elif score >= 70.0:
        return "B"
    elif score >= 50.0:
        return "C"
    else:
        return "D"


def assign_pass_fail(score: float, pass_threshold: float = 40.0) -> str:
    """Map a continuous score to Pass or Fail."""
    return "Pass" if score >= pass_threshold else "Fail"


def generate_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a realistic, synthetic student performance dataset.

    Features are modeled after empirical educational research:
    - Study hours and previous score are primary determinants.
    - Attendance and parental education provide steady positive boost.
    - Extreme sleep deprivation or excess slightly diminishes performance.
    - Random environmental noise accounts for exam-day variance.
    """
    np.random.seed(random_state)
    N = n_samples

    # Primary behavioral features
    study_hours_per_week = np.clip(np.random.normal(15, 6, N), 0, 40)
    attendance_percent = np.clip(np.random.normal(80, 12, N), 30, 100)
    previous_score = np.clip(np.random.normal(65, 15, N), 0, 100)
    sleep_hours = np.clip(np.random.normal(7, 1.3, N), 3, 10)

    # Demographic & environment features
    parental_education = np.random.choice(
        ["high_school", "bachelors", "masters", "phd"],
        size=N,
        p=[0.40, 0.35, 0.20, 0.05],
    )
    internet_access = np.random.choice(["yes", "no"], size=N, p=[0.85, 0.15])
    extracurricular = np.random.choice(["yes", "no"], size=N, p=[0.50, 0.50])

    # Feature contribution multipliers
    edu_boost = pd.Series(parental_education).map(
        {"high_school": 0.0, "bachelors": 2.0, "masters": 4.0, "phd": 6.0}
    ).values
    internet_boost = pd.Series(internet_access).map({"yes": 3.0, "no": 0.0}).values
    extracurricular_boost = pd.Series(extracurricular).map({"yes": 1.0, "no": 0.0}).values

    # Target calculation
    final_score = (
        0.9 * previous_score * 0.4
        + study_hours_per_week * 1.3
        + attendance_percent * 0.25
        + edu_boost
        + internet_boost
        + extracurricular_boost
        - np.abs(sleep_hours - 7.5) * 2.0
        + np.random.normal(0, 6.0, N)
    )
    final_score = np.clip(final_score, 0.0, 100.0).round(1)

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

    df["pass_fail"] = df["final_score"].apply(assign_pass_fail)
    df["grade"] = df["final_score"].apply(assign_grade)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load and validate student dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df
