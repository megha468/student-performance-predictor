"""
Unit tests for Database and Authentication Module.
"""

import os
import tempfile
import unittest
from src.student_predictor.database import (
    create_user,
    delete_student,
    get_all_predictions,
    get_all_students,
    get_predictions_for_user,
    get_summary_stats,
    hash_password,
    init_db,
    save_batch_predictions,
    save_prediction,
    update_password,
    change_user_password,
    update_student,
    verify_password,
    verify_user,
)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_app.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_admin_created(self):
        user = verify_user("admin", "admin123", db_path=self.db_path)
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "admin")

    def test_create_and_verify_student(self):
        success, msg = create_user("alice", "secret123", "Alice Smith", role="student", db_path=self.db_path)
        self.assertTrue(success)

        # Correct login
        user = verify_user("alice", "secret123", db_path=self.db_path)
        self.assertIsNotNone(user)
        self.assertEqual(user["full_name"], "Alice Smith")

        # Wrong password
        wrong = verify_user("alice", "wrongpw", db_path=self.db_path)
        self.assertIsNone(wrong)

        # Duplicate username
        dup_success, dup_msg = create_user("alice", "otherpw", "Alice 2", role="student", db_path=self.db_path)
        self.assertFalse(dup_success)

    def test_update_password(self):
        create_user("bob", "password123", "Bob Jones", role="student", db_path=self.db_path)
        update_password("bob", "newpassword456", db_path=self.db_path)
        self.assertIsNone(verify_user("bob", "password123", db_path=self.db_path))
        self.assertIsNotNone(verify_user("bob", "newpassword456", db_path=self.db_path))

    def test_prediction_storage_and_retrieval(self):
        create_user("charlie", "password123", "Charlie Brown", role="student", db_path=self.db_path)
        inputs = {
            "study_hours_per_week": 18.0,
            "attendance_percent": 92.0,
            "previous_score": 78.0,
            "sleep_hours": 7.5,
            "parental_education": "bachelors",
            "internet_access": "yes",
            "extracurricular": "yes",
        }
        pred_id = save_prediction("charlie", inputs, 82.5, "Pass", "B", db_path=self.db_path)
        self.assertGreater(pred_id, 0)

        preds = get_predictions_for_user("charlie", db_path=self.db_path)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["predicted_score"], 82.5)
        self.assertEqual(preds[0]["predicted_grade"], "B")

    def test_summary_stats(self):
        init_stats = get_summary_stats(db_path=self.db_path)
        base_students = init_stats["total_students"]
        base_preds = init_stats["total_predictions"]

        create_user("diana", "password123", "Diana Prince", role="student", db_path=self.db_path)
        inputs = {
            "study_hours_per_week": 25.0,
            "attendance_percent": 98.0,
            "previous_score": 90.0,
            "sleep_hours": 8.0,
        }
        save_prediction("diana", inputs, 91.0, "Pass", "A", db_path=self.db_path)

        stats = get_summary_stats(db_path=self.db_path)
        self.assertEqual(stats["total_students"], base_students + 1)
        self.assertEqual(stats["total_predictions"], base_preds + 1)
        self.assertGreater(stats["avg_score"], 0)

    def test_delete_student(self):
        create_user("eve", "password123", "Eve", role="student", db_path=self.db_path)
        inputs = {"study_hours_per_week": 10.0, "attendance_percent": 60.0, "previous_score": 40.0}
        save_prediction("eve", inputs, 38.0, "Fail", "D", db_path=self.db_path)

        delete_student("eve", db_path=self.db_path)
        self.assertIsNone(verify_user("eve", "password123", db_path=self.db_path))
        self.assertEqual(len(get_predictions_for_user("eve", db_path=self.db_path)), 0)

    def test_update_student(self):
        create_user("frank", "password123", "Frank Miller", role="student", db_path=self.db_path)
        ok, msg = update_student("frank", full_name="Frankie Miller", new_password="newsecretpassword", db_path=self.db_path)
        self.assertTrue(ok)
        user = verify_user("frank", "newsecretpassword", db_path=self.db_path)
        self.assertIsNotNone(user)
        self.assertEqual(user["full_name"], "Frankie Miller")

    def test_change_user_password(self):
        create_user("grace", "initialpass123", "Grace Hopper", role="student", db_path=self.db_path)
        # Fail on wrong current password
        ok, msg = change_user_password("grace", "wrongcurrentpass", "newpass123", db_path=self.db_path)
        self.assertFalse(ok)
        # Succeed on correct current password
        ok, msg = change_user_password("grace", "initialpass123", "newpass123", db_path=self.db_path)
        self.assertTrue(ok)
        self.assertIsNotNone(verify_user("grace", "newpass123", db_path=self.db_path))

    def test_save_prediction_with_subjects(self):
        create_user("helen", "password123", "Helen Keller", role="student", db_path=self.db_path)
        inputs = {"study_hours_per_week": 20.0, "attendance_percent": 90.0, "previous_score": 75.0}
        sub_scores = {"subjects": {"Mathematics": {"predicted_score": 85.0}}, "summary": {"strongest_subject": "Mathematics"}}
        pred_id = save_prediction("helen", inputs, 85.0, "Pass", "A", db_path=self.db_path, subject_scores=sub_scores)
        self.assertGreater(pred_id, 0)
        preds = get_predictions_for_user("helen", db_path=self.db_path)
        self.assertEqual(len(preds), 1)
        self.assertIn("subject_scores", preds[0].keys())
        self.assertIsNotNone(preds[0]["subject_scores"])


if __name__ == "__main__":
    unittest.main()
