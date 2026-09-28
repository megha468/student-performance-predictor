"""
DATABASE MODULE (BRIDGE)
-----------------------
Maintains backward compatibility with legacy scripts while directing
all persistence logic to the robust src.student_predictor.database module.
"""

from src.student_predictor.database import (
    DB_PATH,
    create_user,
    delete_student,
    get_all_predictions,
    get_all_students,
    get_connection,
    get_predictions_for_user,
    get_summary_stats,
    hash_password,
    init_db,
    save_batch_predictions,
    save_prediction,
    update_password,
    change_user_password,
    update_student,
    verify_user,
)

__all__ = [
    "DB_PATH",
    "init_db",
    "create_user",
    "verify_user",
    "save_prediction",
    "save_batch_predictions",
    "get_predictions_for_user",
    "get_all_predictions",
    "get_all_students",
    "update_password",
    "change_user_password",
    "update_student",
    "delete_student",
    "get_summary_stats",
    "hash_password",
    "get_connection",
]
