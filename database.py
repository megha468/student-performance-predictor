"""
DATABASE MODULE
------------------
Handles everything related to storing data:
  - User accounts (students + admin), with hashed passwords
  - Each student's submitted info + their prediction results

Uses SQLite, which comes built into Python (no extra install needed).
Data is saved in a file called app_data.db in the same folder.
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data.db")


def hash_password(password: str) -> str:
    """Turn a plain password into a scrambled hash so we never store it in plain text."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the tables if they don't exist yet, and a default admin account."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'student')),
            full_name TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            study_hours_per_week REAL,
            attendance_percent REAL,
            previous_score REAL,
            sleep_hours REAL,
            parental_education TEXT,
            internet_access TEXT,
            extracurricular TEXT,
            predicted_score REAL,
            predicted_passfail TEXT,
            predicted_grade TEXT,
            created_at TEXT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)

    # Create a default admin account if one doesn't exist yet.
    cur.execute("SELECT * FROM users WHERE role = 'admin'")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "admin", "Administrator"),
        )

    conn.commit()
    conn.close()


def create_user(username: str, password: str, full_name: str, role: str = "student") -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), role, full_name),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def verify_user(username: str, password: str):
    """Check login credentials. Returns the user row if valid, else None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    if user and user["password_hash"] == hash_password(password):
        return user
    return None


def save_prediction(username, inputs: dict, predicted_score, predicted_passfail, predicted_grade):
    """Save a student's submitted info and their prediction result."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (
            username, study_hours_per_week, attendance_percent, previous_score,
            sleep_hours, parental_education, internet_access, extracurricular,
            predicted_score, predicted_passfail, predicted_grade, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        inputs["study_hours_per_week"], inputs["attendance_percent"], inputs["previous_score"],
        inputs["sleep_hours"], inputs["parental_education"], inputs["internet_access"],
        inputs["extracurricular"], predicted_score, predicted_passfail, predicted_grade,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    ))
    conn.commit()
    conn.close()


def get_predictions_for_user(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions WHERE username = ? ORDER BY created_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_predictions():
    """Used by the admin dashboard to see every student's predictions."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT predictions.*, users.full_name
        FROM predictions
        JOIN users ON predictions.username = users.username
        ORDER BY predictions.created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_students():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, full_name FROM users WHERE role = 'student'")
    rows = cur.fetchall()
    conn.close()
    return rows


def update_password(username: str, new_password: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (hash_password(new_password), username),
    )
    conn.commit()
    conn.close()


def delete_student(username: str):
    """Removes a student account and all of their saved predictions."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE username = ?", (username,))
    cur.execute("DELETE FROM users WHERE username = ? AND role = 'student'", (username,))
    conn.commit()
    conn.close()


def get_summary_stats():
    """Quick numbers for the admin overview page."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as c FROM users WHERE role = 'student'")
    total_students = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM predictions")
    total_predictions = cur.fetchone()["c"]

    cur.execute("SELECT AVG(predicted_score) as a FROM predictions")
    avg_score = cur.fetchone()["a"]

    cur.execute("SELECT COUNT(*) as c FROM predictions WHERE predicted_passfail = 'Pass'")
    pass_count = cur.fetchone()["c"]

    conn.close()
    pass_rate = (pass_count / total_predictions * 100) if total_predictions else 0
    return {
        "total_students": total_students,
        "total_predictions": total_predictions,
        "avg_score": avg_score or 0,
        "pass_rate": pass_rate,
    }
