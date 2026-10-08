from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# DATABASE IMPORT
# ============================================================

try:
    # Works when running:
    # uvicorn backend.main:app --reload
    from .database import (
        initialize_database,
        save_prediction,
        get_predictions,
        clear_predictions
    )
except ImportError:
    # Works when running from backend folder:
    # uvicorn main:app --reload
    from database import (
        initialize_database,
        save_prediction,
        get_predictions,
        clear_predictions
    )


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="AI Customer Support Ticket Triage API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml" / "models"


# ============================================================
# LOAD ML MODELS
# ============================================================

category_model = joblib.load(
    MODEL_DIR / "category_model.pkl"
)

urgency_model = joblib.load(
    MODEL_DIR / "urgency_model.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

REVIEW_THRESHOLD = 70.0


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# CRITICAL BUSINESS SIGNALS
# ============================================================

CRITICAL_PATTERNS = [
    "critical issue",
    "critical outage",
    "complete outage",
    "major outage",
    "production outage",

    "production system is down",
    "production system down",
    "production environment is down",
    "production server is down",

    "entire service is unavailable",
    "entire system is unavailable",

    "completely unavailable",
    "completely down",
    "completely offline",
    "system is completely down",

    "all users are unable to access",
    "all customers are affected",
    "no users can access",
    "customers cannot access",

    "all production services are down",
    "entire production system has failed",
    "service is completely unavailable",

    "immediate emergency",
    "emergency assistance",
    "emergency action",
    "emergency intervention",
    "requires immediate emergency",
    "urgent emergency"
]


# ============================================================
# HIGH URGENCY BUSINESS SIGNALS
# ============================================================

HIGH_URGENCY_PATTERNS = [

    # Billing / payment
    "charged twice",
    "charged multiple times",
    "double charged",
    "charged more than once",
    "duplicate charge",
    "duplicate payment",
    "money deducted",

    "payment failed",
    "payment failure",
    "payment issue",

    "billing error",
    "incorrect charge",
    "wrong charge",

    # Refund
    "refund immediately",
    "need a refund immediately",
    "need refund immediately",
    "refund urgently",
    "need a refund urgently",
    "urgent refund",

    # Account access
    "cannot log in",
    "can't log in",
    "unable to login",
    "unable to log in",
    "cannot access my account",
    "can't access my account",
    "account is locked",

    # Service access
    "service is unavailable",
    "application is unavailable",
    "app is unavailable",
    "cannot access the application",
    "cannot access the service",
    "unable to access the application",
    "unable to access the service",

    # System problems
    "system failure",
    "application failure",
    "system error",
    "blocking my work",
    "cannot complete my work",
    "unable to complete my work",
    "business is blocked",
    "business operations are affected"
]


# ============================================================
# LOW URGENCY / INFORMATIONAL SIGNALS
# ============================================================

LOW_URGENCY_PATTERNS = [

    "want to know",
    "would like to know",
    "can you tell me",
    "tell me whether",
    "tell me if",

    "does the product support",
    "does this product support",
    "does this support",
    "is this supported",

    "what features",
    "what are the features",
    "what is the feature",

    "how does this work",
    "how does the product work",

    "information about",
    "more information",
    "general question",
    "just a question",

    "i have a question",
    "i was wondering",
    "wondering if",

    "can i know",
    "could you tell me"
]


# ============================================================
# QUEUE MAPPING
# ============================================================

QUEUE_MAPPING = {
    "Billing": "Billing Support",
    "Account": "Account Support",
    "Product": "Product Support",
    "Technical": "Technical Support"
}


# ============================================================
# REQUEST MODEL
# ============================================================

class TicketRequest(BaseModel):
    ticket_text: str


# ============================================================
# DETECT CRITICAL SIGNAL
# ============================================================

def detect_critical_signal(ticket_text: str):

    text = ticket_text.lower().strip()

    for pattern in CRITICAL_PATTERNS:

        if pattern in text:
            return True, pattern

    return False, None


# ============================================================
# DETECT HIGH SIGNAL
# ============================================================

def detect_high_urgency_signal(ticket_text: str):

    text = ticket_text.lower().strip()

    for pattern in HIGH_URGENCY_PATTERNS:

        if pattern in text:
            return True, pattern

    return False, None


# ============================================================
# DETECT LOW SIGNAL
# ============================================================

def detect_low_urgency_signal(ticket_text: str):

    text = ticket_text.lower().strip()

    for pattern in LOW_URGENCY_PATTERNS:

        if pattern in text:
            return True, pattern

    return False, None


# ============================================================
# REVIEW STATUS
# ============================================================

def calculate_review_status(confidence: float):

    if confidence < REVIEW_THRESHOLD:
        return True, "Human Review"

    return False, "Auto Processed"


# ============================================================
# QUEUE
# ============================================================

def get_queue(category: str):

    return QUEUE_MAPPING.get(
        category,
        "General Support"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "success",
        "message": "AI Customer Support Triage API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": True
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict_ticket(request: TicketRequest):

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    ticket_text = request.ticket_text.strip()

    if not ticket_text:

        return {
            "success": False,
            "error": "Ticket text cannot be empty"
        }


    # --------------------------------------------------------
    # TF-IDF TRANSFORMATION
    # --------------------------------------------------------

    X = vectorizer.transform(
        [ticket_text]
    )


    # ========================================================
    # CATEGORY MODEL
    # ========================================================

    category_prediction = category_model.predict(X)[0]

    category_probabilities = (
        category_model.predict_proba(X)[0]
    )

    category_confidence = float(
        max(category_probabilities) * 100
    )


    # ========================================================
    # URGENCY MODEL
    # ========================================================

    ml_urgency_prediction = urgency_model.predict(X)[0]

    urgency_probabilities = (
        urgency_model.predict_proba(X)[0]
    )

    urgency_confidence = float(
        max(urgency_probabilities) * 100
    )


    # ========================================================
    # BUSINESS SIGNAL DETECTION
    # ========================================================

    critical_signal, critical_pattern = (
        detect_critical_signal(ticket_text)
    )

    high_signal, high_pattern = (
        detect_high_urgency_signal(ticket_text)
    )

    low_signal, low_pattern = (
        detect_low_urgency_signal(ticket_text)
    )


    # ========================================================
    # DEFAULT TO ML
    # ========================================================

    final_urgency = str(
        ml_urgency_prediction
    )

    final_urgency_confidence = float(
        urgency_confidence
    )

    urgency_source = "ML Model"

    matched_business_pattern = None


    # ========================================================
    # PRIORITY 1 — CRITICAL
    # ========================================================

    if critical_signal:

        final_urgency = "Critical"

        final_urgency_confidence = max(
            float(urgency_confidence),
            90.0
        )

        urgency_source = (
            "Hybrid Rule + ML Model"
        )

        matched_business_pattern = (
            critical_pattern
        )


    # ========================================================
    # PRIORITY 2 — HIGH
    # ========================================================

    elif high_signal:

        # Never downgrade an ML Critical prediction.

        if ml_urgency_prediction == "Critical":

            final_urgency = "Critical"

            final_urgency_confidence = float(
                urgency_confidence
            )

            urgency_source = "ML Model"

        else:

            final_urgency = "High"

            final_urgency_confidence = max(
                float(urgency_confidence),
                85.0
            )

            urgency_source = (
                "Hybrid Rule + ML Model"
            )

            matched_business_pattern = (
                high_pattern
            )


    # ========================================================
    # PRIORITY 3 — LOW / INFORMATIONAL
    # ========================================================

    elif low_signal:

        # Informational questions should not be
        # classified as High or Critical.

        final_urgency = "Low"

        final_urgency_confidence = max(
            float(urgency_confidence),
            85.0
        )

        urgency_source = (
            "Hybrid Rule + ML Model"
        )

        matched_business_pattern = (
            low_pattern
        )


    # ========================================================
    # OVERALL CONFIDENCE
    # ========================================================

    overall_confidence = (

        float(category_confidence)

        +

        float(final_urgency_confidence)

    ) / 2


    # ========================================================
    # REVIEW
    # ========================================================

    review_required, review_status = (
        calculate_review_status(
            overall_confidence
        )
    )


    # ========================================================
    # QUEUE
    # ========================================================

    assigned_queue = get_queue(
        str(category_prediction)
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = datetime.now().isoformat()


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "category": str(
            category_prediction
        ),

        "category_confidence": float(
            round(
                category_confidence,
                2
            )
        ),

        "urgency": str(
            final_urgency
        ),

        "urgency_confidence": float(
            round(
                final_urgency_confidence,
                2
            )
        ),

        "overall_confidence": float(
            round(
                overall_confidence,
                2
            )
        ),

        "assigned_queue": str(
            assigned_queue
        ),

        "review_required": bool(
            review_required
        ),

        "review_status": str(
            review_status
        ),

        "urgency_source": str(
            urgency_source
        ),

        "critical_signal_detected": bool(
            critical_signal
        ),

        "matched_critical_pattern": (
            str(critical_pattern)
            if critical_pattern
            else None
        ),

        "timestamp": timestamp
    }


    # ========================================================
    # SAVE TO SQLITE DATABASE
    # ========================================================

    try:

        save_prediction(
            ticket_text=ticket_text,

            category=str(
                category_prediction
            ),

            urgency=str(
                final_urgency
            ),

            category_confidence=float(
                round(
                    category_confidence,
                    2
                )
            ),

            urgency_confidence=float(
                round(
                    final_urgency_confidence,
                    2
                )
            ),

            overall_confidence=float(
                round(
                    overall_confidence,
                    2
                )
            ),

            assigned_queue=str(
                assigned_queue
            ),

            review_required=bool(
                review_required
            ),

            review_status=str(
                review_status
            ),

            urgency_source=str(
                urgency_source
            ),

            critical_signal_detected=bool(
                critical_signal
            ),

            matched_critical_pattern=(
                str(critical_pattern)
                if critical_pattern
                else None
            ),

            created_at=timestamp
        )

    except Exception as e:

        # Prediction itself should still work even if
        # database storage encounters an error.
        print(
            f"Database save error: {e}"
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success": True,

        "prediction": result

    }


# ============================================================
# HISTORY
# ============================================================

@app.get("/history")
def get_history():

    try:

        history = get_predictions(
            limit=50
        )

        return {

            "success": True,

            "total_predictions": len(
                history
            ),

            "history": history
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e),

            "history": []
        }


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.delete("/history")
def clear_history():

    try:

        clear_predictions()

        return {

            "success": True,

            "message": "Prediction history cleared"
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }


# ============================================================
# MODEL EVALUATION
# ============================================================

@app.get("/evaluation")
def evaluate_models():

    try:

        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        dataset_path = (
            BASE_DIR
            / "dataset"
            / "tickets.csv"
        )

        df = pd.read_csv(
            dataset_path
        )

        required_columns = [
            "ticket_text",
            "category",
            "urgency"
        ]

        for column in required_columns:

            if column not in df.columns:

                return {

                    "success": False,

                    "error": (
                        f"Missing column: {column}"
                    )
                }


        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        df = df.dropna(
            subset=[
                "ticket_text",
                "category",
                "urgency"
            ]
        ).reset_index(
            drop=True
        )

        df["ticket_text"] = (
            df["ticket_text"]
            .astype(str)
        )

        df["category"] = (
            df["category"]
            .astype(str)
        )

        df["urgency"] = (
            df["urgency"]
            .astype(str)
        )


        # ----------------------------------------------------
        # CREATE THE SAME TEST SPLIT
        # USED DURING MODEL TRAINING
        # ----------------------------------------------------

        from sklearn.model_selection import (
            train_test_split
        )

        train_indices, test_indices = (
            train_test_split(
                range(len(df)),
                test_size=0.20,
                random_state=42,
                stratify=df["category"]
            )
        )

        test_indices = list(
            test_indices
        )


        # ----------------------------------------------------
        # GET TEST DATA ONLY
        # ----------------------------------------------------

        test_df = df.iloc[
            test_indices
        ]

        test_texts = (
            test_df["ticket_text"]
            .tolist()
        )

        actual_categories = (
            test_df["category"]
            .tolist()
        )

        actual_urgencies = (
            test_df["urgency"]
            .tolist()
        )


        # ----------------------------------------------------
        # TF-IDF TRANSFORMATION
        # ----------------------------------------------------

        X_test = vectorizer.transform(
            test_texts
        )


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predicted_categories = (
            category_model
            .predict(X_test)
            .astype(str)
            .tolist()
        )

        predicted_urgencies = (
            urgency_model
            .predict(X_test)
            .astype(str)
            .tolist()
        )


        # ----------------------------------------------------
        # CATEGORY METRICS
        # ----------------------------------------------------

        category_accuracy = accuracy_score(
            actual_categories,
            predicted_categories
        )

        category_precision = precision_score(
            actual_categories,
            predicted_categories,
            average="weighted",
            zero_division=0
        )

        category_recall = recall_score(
            actual_categories,
            predicted_categories,
            average="weighted",
            zero_division=0
        )

        category_f1 = f1_score(
            actual_categories,
            predicted_categories,
            average="weighted",
            zero_division=0
        )


        # ----------------------------------------------------
        # URGENCY METRICS
        # ----------------------------------------------------

        urgency_accuracy = accuracy_score(
            actual_urgencies,
            predicted_urgencies
        )

        urgency_precision = precision_score(
            actual_urgencies,
            predicted_urgencies,
            average="weighted",
            zero_division=0
        )

        urgency_recall = recall_score(
            actual_urgencies,
            predicted_urgencies,
            average="weighted",
            zero_division=0
        )

        urgency_f1 = f1_score(
            actual_urgencies,
            predicted_urgencies,
            average="weighted",
            zero_division=0
        )


        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

        return {

            "success": True,

            "dataset_size": int(
                len(df)
            ),

            "training_samples": int(
                len(df) - len(test_df)
            ),

            "testing_samples": int(
                len(test_df)
            ),

            "category": {

                "accuracy": round(
                    float(
                        category_accuracy
                    ) * 100,
                    2
                ),

                "precision": round(
                    float(
                        category_precision
                    ) * 100,
                    2
                ),

                "recall": round(
                    float(
                        category_recall
                    ) * 100,
                    2
                ),

                "f1_score": round(
                    float(
                        category_f1
                    ) * 100,
                    2
                )
            },

            "urgency": {

                "accuracy": round(
                    float(
                        urgency_accuracy
                    ) * 100,
                    2
                ),

                "precision": round(
                    float(
                        urgency_precision
                    ) * 100,
                    2
                ),

                "recall": round(
                    float(
                        urgency_recall
                    ) * 100,
                    2
                ),

                "f1_score": round(
                    float(
                        urgency_f1
                    ) * 100,
                    2
                )
            }
        }


    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }