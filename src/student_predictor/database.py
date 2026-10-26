"""
Enterprise Database & Authentication Module
-------------------------------------------
Handles SQLite data persistence, user accounts, salted PBKDF2 cryptographic
password hashing with seamless legacy SHA-256 upgrade, and student analytics queries.
"""

import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_FILENAME = "app_data.db"
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    DB_FILENAME,
)
DB_PATH = DEFAULT_DB_PATH


def get_db_path() -> str:
    """Return configured database path, prioritizing environment variable if set."""
    return os.environ.get("STUDENT_PREDICTOR_DB", DEFAULT_DB_PATH)


# =====================================================================
# CRYPTOGRAPHIC SECURITY HELPERS (PBKDF2-HMAC-SHA256 with Salt)
# =====================================================================
def hash_password(password: str, salt: Optional[str] = None) -> str:
    """
    Generate a cryptographically salted PBKDF2 hash for a password.
    Format: pbkdf2:sha256:100000$<salt>$<hash>
    """
    if not salt:
        salt = secrets.token_hex(16)
    iterations = 100000
    derived = hmac.new(b"", b"", "sha256").copy()  # verification check
    derived_hash = (
        sqlite3.Binary(
            __import__("hashlib").pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
            )
        )
        .hex()
    )
    return f"pbkdf2:sha256:{iterations}${salt}${derived_hash}"


def verify_password(stored_hash: str, password: str) -> Tuple[bool, bool]:
    """
    Verify password against stored hash.
    Returns:
        (is_valid: bool, needs_upgrade: bool)
    """
    if stored_hash.startswith("pbkdf2:sha256:"):
        try:
            parts = stored_hash.split("$")
            if len(parts) == 3:
                header, salt, expected_hash = parts
                iterations = int(header.split(":")[2])
                computed = __import__("hashlib").pbkdf2_hmac(
                    "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
                ).hex()
                is_valid = hmac.compare_digest(computed, expected_hash)
                return is_valid, False
        except Exception:
            return False, False

    # Legacy plain SHA-256 fallback (for existing student accounts)
    legacy_hash = __import__("hashlib").sha256(password.encode("utf-8")).hexdigest()
    if hmac.compare_digest(stored_hash, legacy_hash):
        return True, True  # Valid, but flag for upgrade to PBKDF2

    return False, False


# =====================================================================
# CONNECTION & SCHEMA INITIALIZATION
# =====================================================================
def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create tables, indexes, and initial admin account if not already present."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'student')),
            full_name TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # Safe migration: ensure created_at column exists in users table
    cur.execute("PRAGMA table_info(users)")
    existing_user_cols = [row[1] for row in cur.fetchall()]
    if "created_at" not in existing_user_cols:
        cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT '2024-09-01 10:00:00'")

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
            subject_scores TEXT,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
    """)

    # Safe migration: ensure subject_scores column exists in predictions table
    cur.execute("PRAGMA table_info(predictions)")
    existing_pred_cols = [row[1] for row in cur.fetchall()]
    if "subject_scores" not in existing_pred_cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN subject_scores TEXT")

    # Performance indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_username ON predictions(username)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at)")

    # Seed default admin if missing
    cur.execute("SELECT username FROM users WHERE role = 'admin'")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                "admin",
                hash_password("admin123"),
                "admin",
                "System Administrator",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )

    # Seed mock students and predictions matching visual UI mockup if not present
    seed_demo_data_if_needed(cur)

    conn.commit()
    conn.close()


def seed_demo_data_if_needed(cur: sqlite3.Cursor) -> None:
    """Populates realistic demo accounts and predictions matching the design mockup."""
    # 1. Megha Sharma (Primary Student demo)
    cur.execute("SELECT username FROM users WHERE username = 'megha'")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("megha", hash_password("password123"), "student", "Megha Sharma", "2024-09-01 10:00:00"),
        )
        megha_preds = [
            ("megha", 20.0, 85.0, 72.0, 7.0, "Graduate", "yes", "yes", 78.5, "PASS", "B", "Sep 27, 2024"),
            ("megha", 18.0, 90.0, 68.0, 7.5, "Graduate", "yes", "no", 74.2, "PASS", "B", "Sep 20, 2024"),
            ("megha", 15.0, 78.0, 60.0, 6.5, "Graduate", "yes", "yes", 69.8, "PASS", "C", "Sep 12, 2024"),
            ("megha", 22.0, 92.0, 75.0, 8.0, "Graduate", "yes", "yes", 82.1, "PASS", "A", "Sep 5, 2024"),
            ("megha", 12.0, 70.0, 55.0, 7.0, "Graduate", "no", "no", 63.7, "PASS", "C", "Aug 28, 2024"),
        ]
        for p in megha_preds:
            cur.execute(
                """
                INSERT INTO predictions (
                    username, study_hours_per_week, attendance_percent, previous_score,
                    sleep_hours, parental_education, internet_access, extracurricular,
                    predicted_score, predicted_passfail, predicted_grade, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                p,
            )

    # 2. Roster students matching Screen 8 & 9
    mock_students = [
        ("rahul123", "Rahul Sharma", "password123", "Sep 20, 2024", 20.0, 85.0, 72.0, 7.0, "Graduate", "yes", "no", 78.5, "PASS", "B", "Sep 27, 2024"),
        ("priya456", "Priya Patel", "password123", "Sep 21, 2024", 25.0, 92.0, 80.0, 7.5, "Graduate", "yes", "yes", 86.2, "PASS", "A", "Sep 26, 2024"),
        ("amit789", "Amit Kumar", "password123", "Sep 22, 2024", 10.0, 65.0, 50.0, 6.0, "High School", "no", "no", 52.4, "PASS", "D", "Sep 25, 2024"),
        ("sneha321", "Sneha Gupta", "password123", "Sep 23, 2024", 18.0, 88.0, 68.0, 7.0, "Bachelors", "yes", "yes", 74.1, "PASS", "B", "Sep 24, 2024"),
        ("rohan654", "Rohan Singh", "password123", "Sep 24, 2024", 22.0, 90.0, 75.0, 8.0, "Masters", "yes", "no", 82.3, "PASS", "A", "Sep 23, 2024"),
    ]
    for uname, fname, pw, jdate, sh, att, ps, sl, pe, ia, ec, pred_sc, pfail, pgr, pdate in mock_students:
        cur.execute("SELECT username FROM users WHERE username = ?", (uname,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (uname, hash_password(pw), "student", fname, jdate),
            )
            cur.execute(
                """
                INSERT INTO predictions (
                    username, study_hours_per_week, attendance_percent, previous_score,
                    sleep_hours, parental_education, internet_access, extracurricular,
                    predicted_score, predicted_passfail, predicted_grade, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (uname, sh, att, ps, sl, pe, ia, ec, pred_sc, pfail, pgr, pdate),
            )


# =====================================================================
# USER ACCOUNT MANAGEMENT
# =====================================================================
def create_user(
    username: str,
    password: str,
    full_name: str,
    role: str = "student",
    db_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Register a new user account with salted password hashing."""
    username = username.strip()
    full_name = full_name.strip()

    if not username or not password or not full_name:
        return False, "All fields are required."

    if len(username) < 3:
        return False, "Username must be at least 3 characters long."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                username,
                hash_password(password),
                role,
                full_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username is already taken. Please choose another."
    finally:
        conn.close()


def verify_user(
    username: str, password: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Check login credentials. Automatically migrates legacy SHA-256 hashes to PBKDF2.
    Returns user dict on success, None otherwise.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT username, password_hash, role, full_name FROM users WHERE username = ?",
        (username.strip(),),
    )
    user = cur.fetchone()

    if not user:
        conn.close()
        return None

    is_valid, needs_upgrade = verify_password(user["password_hash"], password)

    if not is_valid:
        conn.close()
        return None

    # Transparent password hash upgrade to PBKDF2
    if needs_upgrade:
        new_hash = hash_password(password)
        cur.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (new_hash, username.strip()),
        )
        conn.commit()

    user_dict = {
        "username": user["username"],
        "role": user["role"],
        "full_name": user["full_name"],
    }
    conn.close()
    return user_dict


def update_password(
    username: str, new_password: str, db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """Update a user's password with PBKDF2 hash."""
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."

    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (hash_password(new_password), username),
    )
    conn.commit()
    conn.close()
    return True, "Password updated successfully."


def change_user_password(
    username: str,
    current_password: str,
    new_password: str,
    db_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Verifies existing password and updates to new password."""
    user = verify_user(username, current_password, db_path=db_path)
    if not user:
        return False, "Current password is incorrect."
    return update_password(username, new_password, db_path=db_path)


def update_student(
    username: str,
    full_name: Optional[str] = None,
    new_password: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Updates student profile information and optionally resets their password."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT username, full_name FROM users WHERE username = ? AND role = 'student'",
        (username,),
    )
    student = cur.fetchone()
    if not student:
        conn.close()
        return False, f"Student '{username}' not found."

    updates = []
    params = []

    if full_name and full_name.strip():
        updates.append("full_name = ?")
        params.append(full_name.strip())

    if new_password and new_password.strip():
        if len(new_password.strip()) < 6:
            conn.close()
            return False, "Password must be at least 6 characters."
        updates.append("password_hash = ?")
        params.append(hash_password(new_password.strip()))

    if not updates:
        conn.close()
        return False, "No updates provided."

    params.append(username)
    cur.execute(
        f"UPDATE users SET {', '.join(updates)} WHERE username = ? AND role = 'student'",
        tuple(params),
    )
    conn.commit()
    conn.close()
    return True, "Student updated successfully."


def delete_student(username: str, db_path: Optional[str] = None) -> bool:
    """Removes a student account and associated prediction history."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE username = ?", (username,))
    cur.execute("DELETE FROM users WHERE username = ? AND role = 'student'", (username,))
    conn.commit()
    conn.close()
    return True


# =====================================================================
# PREDICTIONS & ANALYTICS
# =====================================================================
def save_prediction(
    username: str,
    inputs: Dict[str, Any],
    predicted_score: float,
    predicted_passfail: str,
    predicted_grade: str,
    db_path: Optional[str] = None,
    subject_scores: Optional[Dict[str, Any]] = None,
) -> int:
    """Save an individual student prediction record with optional subject-wise breakdown."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    sub_json = json.dumps(subject_scores) if subject_scores else None
    cur.execute(
        """
        INSERT INTO predictions (
            username, study_hours_per_week, attendance_percent, previous_score,
            sleep_hours, parental_education, internet_access, extracurricular,
            predicted_score, predicted_passfail, predicted_grade, created_at, subject_scores
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            float(inputs.get("study_hours_per_week", 0.0)),
            float(inputs.get("attendance_percent", 0.0)),
            float(inputs.get("previous_score", 0.0)),
            float(inputs.get("sleep_hours", 7.0)),
            str(inputs.get("parental_education", "high_school")),
            str(inputs.get("internet_access", "yes")),
            str(inputs.get("extracurricular", "no")),
            round(float(predicted_score), 2),
            str(predicted_passfail),
            str(predicted_grade),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sub_json,
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def save_batch_predictions(
    records: List[Dict[str, Any]], db_path: Optional[str] = None
) -> int:
    """Save multiple predictions in a single high-speed transaction."""
    if not records:
        return 0

    conn = get_connection(db_path)
    cur = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.executemany(
        """
        INSERT INTO predictions (
            username, study_hours_per_week, attendance_percent, previous_score,
            sleep_hours, parental_education, internet_access, extracurricular,
            predicted_score, predicted_passfail, predicted_grade, created_at
        ) VALUES (:username, :study_hours_per_week, :attendance_percent, :previous_score,
                  :sleep_hours, :parental_education, :internet_access, :extracurricular,
                  :predicted_score, :predicted_passfail, :predicted_grade, :created_at)
        """,
        [
            {
                "username": r.get("username", "batch_upload"),
                "study_hours_per_week": float(r["study_hours_per_week"]),
                "attendance_percent": float(r["attendance_percent"]),
                "previous_score": float(r["previous_score"]),
                "sleep_hours": float(r.get("sleep_hours", 7.0)),
                "parental_education": str(r.get("parental_education", "high_school")),
                "internet_access": str(r.get("internet_access", "yes")),
                "extracurricular": str(r.get("extracurricular", "no")),
                "predicted_score": round(float(r["predicted_score"]), 2),
                "predicted_passfail": str(r["predicted_passfail"]),
                "predicted_grade": str(r["predicted_grade"]),
                "created_at": now_str,
            }
            for r in records
        ],
    )
    count = cur.rowcount
    conn.commit()
    conn.close()
    return count


def get_predictions_for_user(
    username: str, limit: int = 100, db_path: Optional[str] = None
) -> List[sqlite3.Row]:
    """Retrieve historical predictions for a single user."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM predictions WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_predictions(limit: int = 500, db_path: Optional[str] = None) -> List[sqlite3.Row]:
    """Retrieve all predictions with student full names for admin dashboard."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT predictions.*, COALESCE(users.full_name, 'Guest / Batch') as full_name
        FROM predictions
        LEFT JOIN users ON predictions.username = users.username
        ORDER BY predictions.id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_students(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all registered students along with their prediction counts and latest score."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            u.username,
            u.full_name,
            u.created_at as registered_date,
            COUNT(p.id) as prediction_count,
            ROUND(AVG(p.predicted_score), 1) as avg_score,
            (
                SELECT predicted_grade 
                FROM predictions 
                WHERE username = u.username 
                ORDER BY id DESC LIMIT 1
            ) as latest_grade,
            (
                SELECT predicted_score 
                FROM predictions 
                WHERE username = u.username 
                ORDER BY id DESC LIMIT 1
            ) as latest_score
        FROM users u
        LEFT JOIN predictions p ON u.username = p.username
        WHERE u.role = 'student'
        GROUP BY u.username, u.full_name, u.created_at
        ORDER BY u.created_at DESC
        """
    )
    students = [dict(row) for row in cur.fetchall()]
    conn.close()
    return students


def get_summary_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Executive KPI statistics for admin dashboard."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as c FROM users WHERE role = 'student'")
    total_students = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM predictions")
    total_predictions = cur.fetchone()["c"]

    cur.execute("SELECT AVG(predicted_score) as a FROM predictions")
    avg_score_row = cur.fetchone()["a"]
    avg_score = float(avg_score_row) if avg_score_row is not None else 0.0

    cur.execute("SELECT COUNT(*) as c FROM predictions WHERE predicted_passfail = 'Pass'")
    pass_count = cur.fetchone()["c"]

    # At-risk count (predicted score < 50.0)
    cur.execute("SELECT COUNT(*) as c FROM predictions WHERE predicted_score < 50.0")
    risk_count = cur.fetchone()["c"]

    conn.close()
    pass_rate = (pass_count / total_predictions * 100.0) if total_predictions else 0.0

    return {
        "total_students": total_students,
        "total_predictions": total_predictions,
        "avg_score": round(avg_score, 1),
        "pass_rate": round(pass_rate, 1),
        "at_risk_count": risk_count,
    }
