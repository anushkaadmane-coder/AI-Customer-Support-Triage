import sqlite3
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------
# DATABASE LOCATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "tickets.db"


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_connection():
    """
    Create and return a SQLite database connection.
    """
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ---------------------------------------------------------
# INITIALIZE DATABASE
# ---------------------------------------------------------

def initialize_database():
    """
    Create the predictions table if it does not already exist.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                ticket_text TEXT NOT NULL,

                category TEXT NOT NULL,
                urgency TEXT NOT NULL,

                category_confidence REAL,
                urgency_confidence REAL,
                overall_confidence REAL,

                assigned_queue TEXT,

                review_required INTEGER DEFAULT 0,
                review_status TEXT,

                urgency_source TEXT,

                critical_signal_detected INTEGER DEFAULT 0,
                matched_critical_pattern TEXT,

                created_at TEXT NOT NULL
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


# ---------------------------------------------------------
# SAVE PREDICTION
# ---------------------------------------------------------

def save_prediction(
    ticket_text: str,
    category: str,
    urgency: str,
    category_confidence: float,
    urgency_confidence: float,
    overall_confidence: float,
    assigned_queue: str,
    review_required: bool,
    review_status: str,
    urgency_source: str,
    critical_signal_detected: bool,
    matched_critical_pattern: Optional[str],
    created_at: str,
):
    """
    Save one prediction into the database.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                ticket_text,
                category,
                urgency,
                category_confidence,
                urgency_confidence,
                overall_confidence,
                assigned_queue,
                review_required,
                review_status,
                urgency_source,
                critical_signal_detected,
                matched_critical_pattern,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_text,
                category,
                urgency,
                category_confidence,
                urgency_confidence,
                overall_confidence,
                assigned_queue,
                int(review_required),
                review_status,
                urgency_source,
                int(critical_signal_detected),
                matched_critical_pattern,
                created_at,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    finally:
        connection.close()


# ---------------------------------------------------------
# GET PREDICTIONS
# ---------------------------------------------------------

def get_predictions(limit: int = 50):
    """
    Get the latest predictions from the database.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                ticket_text,
                category,
                urgency,
                category_confidence,
                urgency_confidence,
                overall_confidence,
                assigned_queue,
                review_required,
                review_status,
                urgency_source,
                critical_signal_detected,
                matched_critical_pattern,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        predictions = []

        for row in rows:
            predictions.append(
                {
                    "id": row["id"],
                    "ticket_text": row["ticket_text"],
                    "category": row["category"],
                    "urgency": row["urgency"],
                    "category_confidence": row["category_confidence"],
                    "urgency_confidence": row["urgency_confidence"],
                    "overall_confidence": row["overall_confidence"],
                    "assigned_queue": row["assigned_queue"],
                    "review_required": bool(row["review_required"]),
                    "review_status": row["review_status"],
                    "urgency_source": row["urgency_source"],
                    "critical_signal_detected": bool(
                        row["critical_signal_detected"]
                    ),
                    "matched_critical_pattern": row[
                        "matched_critical_pattern"
                    ],
                    "timestamp": row["created_at"],
                }
            )

        # Return oldest → newest so the frontend history
        # appears in the same natural order as before.
        predictions.reverse()

        return predictions

    finally:
        connection.close()


# ---------------------------------------------------------
# CLEAR ALL PREDICTIONS
# ---------------------------------------------------------

def clear_predictions():
    """
    Delete all saved predictions from the database.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("DELETE FROM predictions")

        connection.commit()

    finally:
        connection.close()