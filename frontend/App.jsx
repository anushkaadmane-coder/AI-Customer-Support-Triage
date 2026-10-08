import { useEffect, useMemo, useState } from "react";

const API_URL = "http://127.0.0.1:8000";
const REVIEW_THRESHOLD = 70;

/* =========================================================
   ICON COMPONENT
========================================================= */

function Icon({ type, size = 20 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.8",
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };

  if (type === "category") {
    return (
      <svg {...common}>
        <rect x="4" y="4" width="6" height="6" rx="1.5" />
        <rect x="14" y="4" width="6" height="6" rx="1.5" />
        <rect x="4" y="14" width="6" height="6" rx="1.5" />
        <rect x="14" y="14" width="6" height="6" rx="1.5" />
      </svg>
    );
  }

  if (type === "urgency") {
    return (
      <svg {...common}>
        <path d="M12 3.5 21 20H3L12 3.5Z" />
        <path d="M12 9v5" />
        <circle cx="12" cy="17" r=".7" fill="currentColor" />
      </svg>
    );
  }

  if (type === "queue") {
    return (
      <svg {...common}>
        <path d="M4 6h16" />
        <path d="M4 12h16" />
        <path d="M4 18h16" />
        <circle cx="8" cy="6" r="1.5" fill="currentColor" />
        <circle cx="15" cy="12" r="1.5" fill="currentColor" />
        <circle cx="10" cy="18" r="1.5" fill="currentColor" />
      </svg>
    );
  }

  if (type === "check") {
    return (
      <svg {...common}>
        <path d="m5 12 4 4L19 6" />
      </svg>
    );
  }

  if (type === "warning") {
    return (
      <svg {...common}>
        <path d="M12 3.5 21 20H3L12 3.5Z" />
        <path d="M12 9v5" />
        <circle cx="12" cy="17" r=".7" fill="currentColor" />
      </svg>
    );
  }

  if (type === "spark") {
    return (
      <svg {...common}>
        <path d="m12 3 1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3Z" />
      </svg>
    );
  }

  if (type === "ticket") {
    return (
      <svg {...common}>
        <path d="M5 5h14v4a2 2 0 0 0 0 4v6H5v-6a2 2 0 0 0 0-4V5Z" />
        <path d="M9 8h6" />
        <path d="M9 16h6" />
      </svg>
    );
  }

  if (type === "review") {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8v5" />
        <circle cx="12" cy="16.5" r=".7" fill="currentColor" />
      </svg>
    );
  }

  return null;
}

/* =========================================================
   APP
========================================================= */

function App() {
  const [ticketText, setTicketText] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [error, setError] = useState("");

  /* =======================================================
     MODEL EVALUATION STATE
  ======================================================= */

  const [evaluation, setEvaluation] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");

  /* =======================================================
     LOAD HISTORY
  ======================================================= */

  const loadHistory = async () => {
    try {
      setHistoryLoading(true);

      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        throw new Error("Unable to load prediction history.");
      }

      const data = await response.json();

      setHistory(
        Array.isArray(data.predictions)
          ? data.predictions
          : Array.isArray(data.history)
          ? data.history
          : []
      );
    } catch (err) {
      console.error("History error:", err);
    } finally {
      setHistoryLoading(false);
    }
  };

  /* =======================================================
     LOAD MODEL EVALUATION
  ======================================================= */

  const loadEvaluation = async () => {
    try {
      setEvaluationLoading(true);
      setEvaluationError("");

      const response = await fetch(`${API_URL}/evaluation`);

      if (!response.ok) {
        throw new Error("Unable to load model evaluation.");
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(
          data.error ||
            data.message ||
            "Model evaluation could not be loaded."
        );
      }

      setEvaluation(data);
    } catch (err) {
      console.error("Evaluation error:", err);

      setEvaluationError(
        err.message ||
          "Unable to connect to the evaluation API."
      );
    } finally {
      setEvaluationLoading(false);
    }
  };

  /* =======================================================
     INITIAL LOAD
  ======================================================= */

  useEffect(() => {
    loadHistory();
    loadEvaluation();
  }, []);

  /* =======================================================
     ANALYZE TICKET
  ======================================================= */

  const predictTicket = async () => {
    if (!ticketText.trim()) {
      setError("Please enter a customer support ticket.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ticket_text: ticketText.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.message ||
            data.detail ||
            "Unable to analyze the ticket."
        );
      }

      setPrediction(data.prediction);

      await loadHistory();
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message ||
          "Unable to connect to the FastAPI backend."
      );
    } finally {
      setLoading(false);
    }
  };

  /* =======================================================
     CLEAR HISTORY
  ======================================================= */

  const clearHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/history`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Unable to clear history.");
      }

      setHistory([]);
    } catch (err) {
      console.error("Clear history error:", err);

      setError("Unable to clear prediction history.");
    }
  };

  /* =======================================================
     EXAMPLES
  ======================================================= */

  const useExample = (text) => {
    setTicketText(text);
    setPrediction(null);
    setError("");
  };

  /* =======================================================
     STATISTICS
  ======================================================= */

  const stats = useMemo(() => {
    const total = history.length;

    const highUrgency = history.filter(
      (item) =>
        item.urgency === "High" ||
        item.urgency === "Critical"
    ).length;

    const reviewRequired = history.filter(
      (item) => item.review_required === true
    ).length;

    const averageConfidence =
      total > 0
        ? history.reduce(
            (sum, item) =>
              sum +
              Number(item.overall_confidence || 0),
            0
          ) / total
        : 0;

    return {
      total,
      highUrgency,
      reviewRequired,
      averageConfidence,
    };
  }, [history]);

  /* =======================================================
     HELPERS
  ======================================================= */

  const formatConfidence = (value) => {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "0.00%";
    }

    return `${number.toFixed(2)}%`;
  };

  const getConfidenceLevel = (value) => {
    const confidence = Number(value || 0);

    if (confidence >= 85) return "high";
    if (confidence >= REVIEW_THRESHOLD) return "medium";

    return "low";
  };

  const getUrgencyClass = (urgency) => {
    switch (urgency) {
      case "Critical":
        return "critical";

      case "High":
        return "high";

      case "Medium":
        return "medium";

      default:
        return "low";
    }
  };

  /* =======================================================
     EVALUATION HELPERS
  ======================================================= */

  const evaluationValue = (value) => {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "—";
    }

    return `${number.toFixed(2)}%`;
  };

  const getEvaluationMetric = (model, metric) => {
    if (!evaluation) {
      return null;
    }

    const modelData = evaluation[model];

    if (!modelData) {
      return null;
    }

    return (
      modelData[metric] ??
      modelData[metric.replace("_score", "")] ??
      null
    );
  };

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="app">

      {/* ===================================================
          NAVBAR
      =================================================== */}

      <header className="navbar">

        <div className="brand">

          <div className="brand-mark">
            <Icon type="spark" size={22} />
          </div>

          <div className="brand-copy">
            <h1>TicketIQ</h1>
            <span>
              Customer Support Intelligence
            </span>
          </div>

        </div>

        <div className="engine-status">
          <span className="engine-dot"></span>
          <span>AI Engine Online</span>
        </div>

      </header>

      <main className="container">

        {/* =================================================
            HERO
        ================================================= */}

        <section className="hero">

          <div className="hero-badge">
            <span></span>
            AI-POWERED TICKET TRIAGE
          </div>

          <h2>
            Turn support tickets into
            <span> intelligent decisions.</span>
          </h2>

          <p>
            Automatically classify customer tickets,
            detect urgency, route them to the right team,
            and identify cases that need human review.
          </p>

        </section>

        {/* =================================================
            STATS
        ================================================= */}

        <section className="stats-grid">

          <div className="stat-card">

            <div className="stat-icon stat-blue">
              <Icon type="ticket" size={20} />
            </div>

            <div className="stat-content">
              <span>Total Predictions</span>
              <strong>{stats.total}</strong>
            </div>

          </div>

          <div className="stat-card">

            <div className="stat-icon stat-red">
              <Icon type="urgency" size={20} />
            </div>

            <div className="stat-content">
              <span>High Priority</span>
              <strong>{stats.highUrgency}</strong>
            </div>

          </div>

          <div className="stat-card">

            <div className="stat-icon stat-orange">
              <Icon type="review" size={20} />
            </div>

            <div className="stat-content">
              <span>Human Review</span>
              <strong>{stats.reviewRequired}</strong>
            </div>

          </div>

          <div className="stat-card">

            <div className="stat-icon stat-green">
              <span>%</span>
            </div>

            <div className="stat-content">
              <span>Average Confidence</span>

              <strong>
                {stats.averageConfidence.toFixed(2)}%
              </strong>

            </div>

          </div>

        </section>

        {/* =================================================
            MAIN WORKSPACE
        ================================================= */}

        <section className="workspace">

          {/* =================================================
              INPUT
          ================================================= */}

          <div className="panel input-panel">

            <div className="panel-top">

              <div>

                <span className="section-number">
                  01 / INPUT
                </span>

                <h3>
                  Analyze a Support Ticket
                </h3>

              </div>

              <div className="ready-badge">

                <Icon
                  type="spark"
                  size={14}
                />

                AI Ready

              </div>

            </div>

            <p className="panel-description">
              Paste a customer support message below.
              The AI will classify the ticket, detect
              urgency, and recommend the correct queue.
            </p>

            <div className="textarea-wrapper">

              <textarea
                value={ticketText}
                onChange={(event) => {
                  setTicketText(event.target.value);
                  setError("");
                }}
                placeholder="Example: I was charged twice for my subscription and need a refund..."
              />

              <span className="character-count">
                {ticketText.length} characters
              </span>

            </div>

            <div className="input-actions">

              <button
                className="analyze-btn"
                onClick={predictTicket}
                disabled={loading}
              >

                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing
                  </>
                ) : (
                  <>
                    Analyze Ticket
                    <span className="button-arrow">
                      →
                    </span>
                  </>
                )}

              </button>

            </div>

            {error && (
              <div className="error-message">

                <Icon
                  type="warning"
                  size={18}
                />

                <span>{error}</span>

              </div>
            )}

            <div className="examples">

              <span>Try an example</span>

              <button
                onClick={() =>
                  useExample(
                    "I was charged twice for my subscription and need a refund immediately."
                  )
                }
              >
                Billing
              </button>

              <button
                onClick={() =>
                  useExample(
                    "I cannot log into my account because I am not receiving the verification code."
                  )
                }
              >
                Account
              </button>

              <button
                onClick={() =>
                  useExample(
                    "The application crashes every time I try to upload a file."
                  )
                }
              >
                Technical
              </button>

              <button
                onClick={() =>
                  useExample(
                    "I want to know whether the new product supports dark mode."
                  )
                }
              >
                Product
              </button>

            </div>

          </div>

          {/* =================================================
              OUTPUT
          ================================================= */}

          <div className="panel output-panel">

            <div className="panel-top">

              <div>

                <span className="section-number">
                  02 / AI OUTPUT
                </span>

                <h3>
                  Prediction Results
                </h3>

              </div>

              {prediction && (
                <div className="processed-badge">
                  <span></span>
                  Processed
                </div>
              )}

            </div>

            {!prediction ? (

              <div className="empty-result">

                <div className="empty-result-icon">
                  <Icon
                    type="spark"
                    size={24}
                  />
                </div>

                <h4>
                  Waiting for a ticket
                </h4>

                <p>
                  Your classification results will
                  appear here after analysis.
                </p>

              </div>

            ) : (

              <div className="results">

                {/* CATEGORY */}

                <div className="result-item">

                  <div className="result-icon category-icon">

                    <Icon
                      type="category"
                      size={21}
                    />

                  </div>

                  <div className="result-info">

                    <span className="result-label">
                      Category
                    </span>

                    <strong>
                      {prediction.category}
                    </strong>

                  </div>

                  <div
                    className={`result-confidence ${getConfidenceLevel(
                      prediction.category_confidence
                    )}`}
                  >
                    {formatConfidence(
                      prediction.category_confidence
                    )}
                  </div>

                </div>

                {/* URGENCY */}

                <div className="result-item">

                  <div className="result-icon urgency-icon">

                    <Icon
                      type="urgency"
                      size={21}
                    />

                  </div>

                  <div className="result-info">

                    <span className="result-label">
                      Urgency
                    </span>

                    <strong>

                      <span
                        className={`urgency-badge ${getUrgencyClass(
                          prediction.urgency
                        )}`}
                      >
                        {prediction.urgency}
                      </span>

                    </strong>

                  </div>

                  <div
                    className={`result-confidence ${getConfidenceLevel(
                      prediction.urgency_confidence
                    )}`}
                  >
                    {formatConfidence(
                      prediction.urgency_confidence
                    )}
                  </div>

                </div>

                {/* QUEUE */}

                <div className="result-item">

                  <div className="result-icon queue-icon">

                    <Icon
                      type="queue"
                      size={21}
                    />

                  </div>

                  <div className="result-info">

                    <span className="result-label">
                      Assigned Queue
                    </span>

                    <strong>
                      {prediction.assigned_queue}
                    </strong>

                  </div>

                  <span
                    className={
                      prediction.review_required
                        ? "route-badge review"
                        : "route-badge auto"
                    }
                  >
                    {prediction.review_required
                      ? "Review Required"
                      : "Auto Routed"}
                  </span>

                </div>

                {/* CONFIDENCE */}

                <div className="confidence-card">

                  <div className="confidence-header">

                    <div>

                      <span>
                        Overall AI Confidence
                      </span>

                      <small>
                        Based on the model prediction
                      </small>

                    </div>

                    <strong>
                      {formatConfidence(
                        prediction.overall_confidence
                      )}
                    </strong>

                  </div>

                  <div className="confidence-track">

                    <div
                      className={`confidence-fill ${getConfidenceLevel(
                        prediction.overall_confidence
                      )}`}
                      style={{
                        width: `${Math.min(
                          Number(
                            prediction.overall_confidence || 0
                          ),
                          100
                        )}%`,
                      }}
                    ></div>

                  </div>

                  <div className="threshold-label">

                    <span>
                      Review threshold:{" "}
                      {REVIEW_THRESHOLD}%
                    </span>

                    <span>
                      {Number(
                        prediction.overall_confidence || 0
                      ) >= REVIEW_THRESHOLD
                        ? "Above threshold"
                        : "Below threshold"}
                    </span>

                  </div>

                </div>

                {/* REVIEW STATUS */}

                {prediction.review_required ? (

                  <div className="review-card warning">

                    <div className="review-card-icon">

                      <Icon
                        type="review"
                        size={20}
                      />

                    </div>

                    <div>

                      <strong>
                        Human Review Required
                      </strong>

                      <p>
                        Confidence is below the 70%
                        automation threshold. A support
                        agent should verify this ticket
                        before final routing.
                      </p>

                    </div>

                  </div>

                ) : (

                  <div className="review-card success">

                    <div className="review-card-icon">

                      <Icon
                        type="check"
                        size={20}
                      />

                    </div>

                    <div>

                      <strong>
                        Ready for Automatic Routing
                      </strong>

                      <p>
                        Confidence is above the 70%
                        automation threshold.
                      </p>

                    </div>

                  </div>

                )}

              </div>

            )}

          </div>

        </section>

        {/* =================================================
            HISTORY
        ================================================= */}

        <section className="panel history-panel">

          <div className="history-top">

            <div>

              <span className="section-number">
                03 / HISTORY
              </span>

              <h3>
                Recent Predictions
              </h3>

            </div>

            <button
              className="clear-btn"
              onClick={clearHistory}
              disabled={history.length === 0}
            >
              Clear History
            </button>

          </div>

          {historyLoading ? (

            <div className="history-empty">
              Loading prediction history...
            </div>

          ) : history.length === 0 ? (

            <div className="history-empty">

              <Icon
                type="ticket"
                size={24}
              />

              <strong>
                No predictions yet
              </strong>

              <span>
                Analyze your first ticket to see it here.
              </span>

            </div>

          ) : (

            <div className="table-container">

              <table>

                <thead>

                  <tr>
                    <th>Ticket</th>
                    <th>Category</th>
                    <th>Urgency</th>
                    <th>Queue</th>
                    <th>Confidence</th>
                    <th>Status</th>
                  </tr>

                </thead>

                <tbody>

                  {history.map((item, index) => (

                    <tr
                      key={
                        item.timestamp ||
                        `${item.ticket_text}-${index}`
                      }
                    >

                      <td>

                        <div className="ticket-cell">

                          <span className="ticket-index">
                            #{history.length - index}
                          </span>

                          <span className="ticket-text">
                            {item.ticket_text}
                          </span>

                        </div>

                      </td>

                      <td>

                        <span className="category-pill">
                          {item.category}
                        </span>

                      </td>

                      <td>

                        <span
                          className={`urgency-badge ${getUrgencyClass(
                            item.urgency
                          )}`}
                        >
                          {item.urgency}
                        </span>

                      </td>

                      <td>

                        <span className="queue-text">
                          {item.assigned_queue}
                        </span>

                      </td>

                      <td>

                        <span
                          className={`table-confidence ${getConfidenceLevel(
                            item.overall_confidence
                          )}`}
                        >
                          {formatConfidence(
                            item.overall_confidence
                          )}
                        </span>

                      </td>

                      <td>

                        {item.review_required ? (

                          <span className="status-pill review">
                            Review
                          </span>

                        ) : (

                          <span className="status-pill auto">
                            Auto
                          </span>

                        )}

                      </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          )}

        </section>

        {/* =================================================
            MODEL EVALUATION
        ================================================= */}

        <section className="panel evaluation-panel">

          <div className="history-top">

            <div>

              <span className="section-number">
                04 / MODEL EVALUATION
              </span>

              <h3>
                Model Evaluation
              </h3>

            </div>

            <div className="ready-badge">
              <Icon
                type="check"
                size={14}
              />
              Performance
            </div>

          </div>

          {evaluationLoading ? (

            <div className="history-empty">
              Loading model evaluation...
            </div>

          ) : evaluationError ? (

            <div className="error-message">

              <Icon
                type="warning"
                size={18}
              />

              <span>
                {evaluationError}
              </span>

            </div>

          ) : evaluation ? (

            <>

              {/* =========================================
                  DATASET SUMMARY
              ========================================= */}

              <div className="evaluation-summary">

                <div className="evaluation-summary-card">

                  <span>
                    Dataset Size
                  </span>

                  <strong>
                    {evaluation.dataset_size ?? "—"}
                  </strong>

                </div>

                <div className="evaluation-summary-card">

                  <span>
                    Training Samples
                  </span>

                  <strong>
                    {evaluation.training_samples ??
                      evaluation.train_samples ??
                      "—"}
                  </strong>

                </div>

                <div className="evaluation-summary-card">

                  <span>
                    Testing Samples
                  </span>

                  <strong>
                    {evaluation.testing_samples ??
                      evaluation.test_samples ??
                      "—"}
                  </strong>

                </div>

              </div>

              {/* =========================================
                  CATEGORY MODEL
              ========================================= */}

              <div className="evaluation-block">

                <div className="evaluation-block-header">

                  <div>

                    <span className="result-label">
                      Model 01
                    </span>

                    <h3>
                      Category Classification
                    </h3>

                  </div>

                  <span className="confusion-model">
                    Logistic Regression
                  </span>

                </div>

                <div className="evaluation-metrics">

                  <div className="evaluation-metric">

                    <span>
                      Accuracy
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "category",
                          "accuracy"
                        )
                      )}
                    </strong>

                  </div>

                  <div className="evaluation-metric">

                    <span>
                      Precision
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "category",
                          "precision"
                        )
                      )}
                    </strong>

                  </div>

                  <div className="evaluation-metric">

                    <span>
                      Recall
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "category",
                          "recall"
                        )
                      )}
                    </strong>

                  </div>

                  <div className="evaluation-metric">

                    <span>
                      F1 Score
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "category",
                          "f1_score"
                        )
                      )}
                    </strong>

                  </div>

                </div>

              </div>

              {/* =========================================
                  URGENCY MODEL
              ========================================= */}

              <div className="evaluation-block">

                <div className="evaluation-block-header">

                  <div>

                    <span className="result-label">
                      Model 02
                    </span>

                    <h3>
                      Urgency Classification
                    </h3>

                  </div>

                  <span className="confusion-model">
                    Logistic Regression
                  </span>

                </div>

                <div className="evaluation-metrics">

                  <div className="evaluation-metric">

                    <span>
                      Accuracy
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "urgency",
                          "accuracy"
                        )
                      )}
                    </strong>

                  </div>

                  <div className="evaluation-metric">

                    <span>
                      Precision
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "urgency",
                          "precision"
                        )
                      )}
                    </strong>

                  </div>

                  <div className="evaluation-metric">

                    <span>
                      Recall
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "urgency",
                          "recall"
                        )
                      )}
                    </strong>

                  </div>

                  <div className="evaluation-metric">

                    <span>
                      F1 Score
                    </span>

                    <strong>
                      {evaluationValue(
                        getEvaluationMetric(
                          "urgency",
                          "f1_score"
                        )
                      )}
                    </strong>

                  </div>

                </div>

              </div>

              {/* =========================================
                  CONFUSION MATRICES
              ========================================= */}

              <div className="confusion-matrices">

                {/* CATEGORY MATRIX */}

                <div className="confusion-card">

                  <div className="confusion-card-header">

                    <div>

                      <span className="confusion-label">
                        MODEL 01
                      </span>

                      <h3>
                        Category Confusion Matrix
                      </h3>

                      <p>
                        Actual vs predicted ticket
                        categories on the test set.
                      </p>

                    </div>

                    <span className="confusion-model">
                      Logistic Regression
                    </span>

                  </div>

                  <div className="confusion-image-wrapper">

                    <img
                      src="/evaluation/category_confusion_matrix.png"
                      alt="Category Classification Confusion Matrix"
                      className="confusion-image"
                    />

                  </div>

                </div>

                {/* URGENCY MATRIX */}

                <div className="confusion-card">

                  <div className="confusion-card-header">

                    <div>

                      <span className="confusion-label">
                        MODEL 02
                      </span>

                      <h3>
                        Urgency Confusion Matrix
                      </h3>

                      <p>
                        Actual vs predicted urgency
                        levels on the test set.
                      </p>

                    </div>

                    <span className="confusion-model">
                      Logistic Regression
                    </span>

                  </div>

                  <div className="confusion-image-wrapper">

                    <img
                      src="/evaluation/urgency_confusion_matrix.png"
                      alt="Urgency Classification Confusion Matrix"
                      className="confusion-image"
                    />

                  </div>

                </div>

              </div>

              {/* =========================================
                  EVALUATION METHOD
              ========================================= */}

              <div className="evaluation-note">

                <strong>
                  Evaluation Method
                </strong>

                <p>
                  Train / Test Split · TF-IDF +
                  Logistic Regression
                </p>

                <p>
                  The evaluation reports the performance
                  of the trained category and urgency
                  classifiers using accuracy, precision,
                  recall, and F1 score on the test set.
                </p>

              </div>

            </>

          ) : (

            <div className="history-empty">

              <Icon
                type="warning"
                size={24}
              />

              <strong>
                Evaluation data unavailable
              </strong>

              <span>
                Make sure the FastAPI backend is running.
              </span>

            </div>

          )}

        </section>

        {/* =================================================
            FOOTER
        ================================================= */}

        <footer>

          <span>
            TicketIQ · AI Customer Support Ticket Triage
          </span>

          <span>
            TF-IDF + Logistic Regression · FastAPI
          </span>

        </footer>

      </main>

    </div>
  );
}

export default App;