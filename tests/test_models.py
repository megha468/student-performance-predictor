"""
Unit tests for Machine Learning Models and Advisor Module.
"""

import os
import unittest
from src.student_predictor.models import load_model_suite
from src.student_predictor.advisor import (
    generate_recommendations,
    simulate_what_if,
    target_goal_solver,
)


class TestModelsAndAdvisor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.suite = load_model_suite(project_dir)

    def test_predict_single_schema(self):
        sample = {
            "study_hours_per_week": 16.0,
            "attendance_percent": 85.0,
            "previous_score": 70.0,
            "sleep_hours": 7.0,
            "parental_education": "bachelors",
            "internet_access": "yes",
            "extracurricular": "yes",
        }
        res = self.suite.predict_single(sample)
        self.assertIn("predicted_score", res)
        self.assertIn("predicted_passfail", res)
        self.assertIn("predicted_grade", res)
        self.assertIn("grade_probabilities", res)
        self.assertIn("risk_tier", res)

        self.assertGreaterEqual(res["predicted_score"], 0.0)
        self.assertLessEqual(res["predicted_score"], 100.0)
        self.assertIn(res["predicted_passfail"], ["Pass", "Fail"])
        self.assertIn(res["predicted_grade"], ["A", "B", "C", "D"])

    def test_recommendations_generation(self):
        sample = {
            "study_hours_per_week": 6.0,
            "attendance_percent": 65.0,
            "previous_score": 50.0,
            "sleep_hours": 5.0,
        }
        pred = self.suite.predict_single(sample)
        recs = generate_recommendations(sample, pred)
        self.assertGreater(len(recs), 0)
        for rec in recs:
            self.assertIn("category", rec)
            self.assertIn("title", rec)
            self.assertIn("action", rec)

    def test_simulate_what_if(self):
        base = {
            "study_hours_per_week": 10.0,
            "attendance_percent": 75.0,
            "previous_score": 60.0,
            "sleep_hours": 7.0,
            "parental_education": "high_school",
            "internet_access": "yes",
            "extracurricular": "no",
        }
        sim = simulate_what_if(self.suite, base, {"study_hours_per_week": 25.0, "attendance_percent": 95.0})
        self.assertGreater(sim["simulated_score"], sim["base_score"])
        self.assertGreater(sim["score_delta"], 0)

    def test_target_goal_solver(self):
        base = {
            "study_hours_per_week": 12.0,
            "attendance_percent": 80.0,
            "previous_score": 65.0,
            "sleep_hours": 7.0,
            "parental_education": "high_school",
            "internet_access": "yes",
            "extracurricular": "no",
        }
        goal = target_goal_solver(self.suite, base, target_score=85.0)
        self.assertIn("target_score", goal)
        self.assertIn("current_score", goal)


if __name__ == "__main__":
    unittest.main()
