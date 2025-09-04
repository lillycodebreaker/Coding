Love this—here’s a crisp concept for the **Evaluation API Toolkit** and a working **Python FastAPI microservice** you can run today.

# Concept: Evaluation API Toolkit (plug-and-play)

* **API-first** service that any team (predictive, RAG/chatbot, vendor tools) can POST metrics to—no repo reshuffles required.
* **Three domains, one interface**

  1. **Predictive models** → ROC-AUC, PR-AUC, precision/recall/F1 at threshold, lift\@k, calibration, drift (PSI), fairness slices.
  2. **Retrieval** → precision\@k, recall\@k, MAP\@k, nDCG\@k across queries.
  3. **Generation** → answer relevance (to the question), faithfulness to sources (citation overlap proxy), hallucination flag.
* **Governance & observability**: JSON scorecards, per-slice fairness, simple drift, and natural-language summaries for business partners.
* **Adapters**: because it’s just HTTP+JSON, you can wrap legacy scripts, vendor responses, or internal models the same way.
* **Extensible**: drop-in custom metrics per team/domain without changing callers.

---

# Code: FastAPI microservice (single file)

> Save as `evaluation_api.py`, then run with `uvicorn evaluation_api:app --reload`

```python
# evaluation_api.py
# Minimal Evaluation API Toolkit for predictive, retrieval, and generation systems.
# Run: pip install fastapi uvicorn pydantic numpy
# Start: uvicorn evaluation_api:app --reload

from typing import List, Dict, Optional, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="Evaluation API Toolkit", version="0.1.0")

# --------------------
# Utility: metrics math
# --------------------

def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    P = y_true.sum()
    N = len(y_true) - P
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    # prepend (0,0)
    tpr = np.concatenate(([0.0], tps / (P if P > 0 else 1)))
    fpr = np.concatenate(([0.0], fps / (N if N > 0 else 1)))
    return fpr, tpr

def roc_auc(y_true: List[int], y_score: List[float]) -> float:
    fpr, tpr = _roc_curve(np.array(y_true), np.array(y_score))
    return float(np.trapz(tpr, fpr))

def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(y_true.sum(), 1)
    # prepend start points
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return precision, recall

def pr_auc(y_true: List[int], y_score: List[float]) -> float:
    precision, recall = _precision_recall_curve(np.array(y_true), np.array(y_score))
    # integrate precision-recall (recall asc)
    return float(np.trapz(precision, recall))

def confusion_at_threshold(y_true: List[int], y_score: List[float], threshold: float):
    y_pred = (np.array(y_score) >= threshold).astype(int)
    y_true = np.array(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

def lift_at_k(y_true: List[int], y_score: List[float], k: float = 0.1) -> float:
    assert 0 < k <= 1.0
    n = len(y_true)
    top = int(max(1, np.floor(n * k)))
    order = np.argsort(-np.array(y_score))
    y_top = np.array(y_true)[order][:top]
    base_rate = np.mean(y_true)
    top_rate = float(np.mean(y_top))
    return (top_rate / base_rate) if base_rate > 0 else 0.0

def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    k = min(k, len(retrieved))
    if k == 0: return 0.0
    hits = sum(1 for x in retrieved[:k] if x in set(relevant))
    return hits / k

def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    rel_set = set(relevant)
    if not rel_set: return 0.0
    hits = sum(1 for x in retrieved[:k] if x in rel_set)
    return hits / len(rel_set)

def average_precision(relevant: List[str], retrieved: List[str], k: Optional[int] = None) -> float:
    if k is None: k = len(retrieved)
    rel_set = set(relevant)
    if not rel_set: return 0.0
    score = 0.0
    hits = 0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in rel_set:
            hits += 1
            score += hits / i
    return score / len(rel_set)

def ndcg_at_k(gains: List[int], k: int) -> float:
    # gains: list of relevance levels for ranked items (retrieved order)
    k = min(k, len(gains))
    dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains[:k]))
    ideal = sorted(gains, reverse=True)
    idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return float(dcg / idcg) if idcg > 0 else 0.0

def psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    e, edges = np.histogram(expected, bins=bins)
    a, _ = np.histogram(actual, bins=edges)
    e = e / np.maximum(e.sum(), 1)
    a = a / np.maximum(a.sum(), 1)
    # add tiny to avoid div/0
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def token_set(s: str) -> set:
    import re
    return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

def relevancy_to_question(answer: str, question: str) -> float:
    a, q = token_set(answer), token_set(question)
    return len(a & q) / max(len(q), 1)

def faithfulness_to_sources(answer: str, sources: List[str]) -> float:
    a = token_set(answer)
    src = set().union(*[token_set(s) for s in sources]) if sources else set()
    return len(a & src) / max(len(a), 1)

# --------------------
# Pydantic Schemas
# --------------------

class PredictiveEvalRequest(BaseModel):
    y_true: List[int]
    y_scores: List[float]
    threshold: float = 0.5
    top_k_fraction: float = Field(0.1, gt=0, le=1)
    protected_attr: Optional[List[str]] = None  # e.g., ["groupA","groupB",...]

class PredictiveEvalResponse(BaseModel):
    roc_auc: float
    pr_auc: float
    at_threshold: Dict[str, float]
    lift_at_top_k: float
    drift_psi: Optional[float] = None
    fairness_by_group: Optional[Dict[str, Dict[str, float]]] = None

class RetrievalQuery(BaseModel):
    relevant_ids: List[str]
    retrieved_ids: List[str]  # ranked
    gains: Optional[List[int]] = None  # optional graded relevance for nDCG

class RetrievalEvalRequest(BaseModel):
    k: int = 5
    queries: List[RetrievalQuery]

class RetrievalEvalResponse(BaseModel):
    macro_precision_at_k: float
    macro_recall_at_k: float
    macro_map_at_k: float
    macro_ndcg_at_k: Optional[float] = None

class GenerationEvalRequest(BaseModel):
    question: str
    answer: str
    sources: List[str] = []

class GenerationEvalResponse(BaseModel):
    answer_relevance: float
    faithfulness: float
    hallucination_flag: bool

class ReportRequest(BaseModel):
    predictive: Optional[PredictiveEvalResponse] = None
    retrieval: Optional[RetrievalEvalResponse] = None
    generation: Optional[GenerationEvalResponse] = None

# --------------------
# Endpoints
# --------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/evaluate/predictive", response_model=PredictiveEvalResponse)
def evaluate_predictive(req: PredictiveEvalRequest):
    y_true = req.y_true
    y_scores = req.y_scores
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must be the same length")

    out = {
        "roc_auc": roc_auc(y_true, y_scores),
        "pr_auc": pr_auc(y_true, y_scores),
        "at_threshold": confusion_at_threshold(y_true, y_scores, req.threshold),
        "lift_at_top_k": lift_at_k(y_true, y_scores, req.top_k_fraction),
        "drift_psi": None,
        "fairness_by_group": None,
    }

    # Optional drift vs. training proxy: use scores as distribution
    # (Caller can pass training_scores in protected_attr as a hack? Better: add separate field.)
    # Here we skip unless caller sends 'protected_attr' == "__drift__:<comma-separated-training-scores>"
    if req.protected_attr and len(req.protected_attr) == 1 and req.protected_attr[0].startswith("__drift__:"):
        try:
            train_scores = [float(x) for x in req.protected_attr[0].split(":",1)[1].split(",")]
            out["drift_psi"] = psi(train_scores, y_scores)
        except Exception:
            out["drift_psi"] = None

    # Optional fairness by group
    if req.protected_attr and len(req.protected_attr) == len(y_true):
        groups = {}
        arr_true = np.array(y_true)
        arr_scores = np.array(y_scores)
        for g in sorted(set(req.protected_attr)):
            mask = np.array([x == g for x in req.protected_attr])
            if mask.sum() > 0:
                groups[g] = {
                    "roc_auc": roc_auc(arr_true[mask].tolist(), arr_scores[mask].tolist()),
                    "positive_rate@thr": float(((arr_scores[mask] >= req.threshold).astype(int)).mean()),
                }
        out["fairness_by_group"] = groups

    return out

@app.post("/evaluate/retrieval", response_model=RetrievalEvalResponse)
def evaluate_retrieval(req: RetrievalEvalRequest):
    p_list, r_list, ap_list, ndcg_list = [], [], [], []
    for q in req.queries:
        p_list.append(precision_at_k(q.relevant_ids, q.retrieved_ids, req.k))
        r_list.append(recall_at_k(q.relevant_ids, q.retrieved_ids, req.k))
        ap_list.append(average_precision(q.relevant_ids, q.retrieved_ids, req.k))
        if q.gains:
            ndcg_list.append(ndcg_at_k(q.gains, req.k))
    resp = {
        "macro_precision_at_k": float(np.mean(p_list)) if p_list else 0.0,
        "macro_recall_at_k": float(np.mean(r_list)) if r_list else 0.0,
        "macro_map_at_k": float(np.mean(ap_list)) if ap_list else 0.0,
        "macro_ndcg_at_k": float(np.mean(ndcg_list)) if ndcg_list else None
    }
    return resp

@app.post("/evaluate/generation", response_model=GenerationEvalResponse)
def evaluate_generation(req: GenerationEvalRequest):
    rel = relevancy_to_question(req.answer, req.question)
    faith = faithfulness_to_sources(req.answer, req.sources)
    halluc = bool(faith < 0.4)  # simple proxy; tune for your domain
    return {"answer_relevance": rel, "faithfulness": faith, "hallucination_flag": halluc}

@app.post("/report/summary")
def generate_summary(req: ReportRequest) -> Dict[str, Any]:
    lines = []
    if req.predictive:
        thr = req.predictive.at_threshold
        lines.append(
            f"Predictive — ROC-AUC: {req.predictive.roc_auc:.3f}, PR-AUC: {req.predictive.pr_auc:.3f}, "
            f"Precision/Recall@thr: {thr['precision']:.3f}/{thr['recall']:.3f}, F1: {thr['f1']:.3f}, "
            f"Lift@topK: {req.predictive.lift_at_top_k:.2f}."
        )
        if req.predictive.drift_psi is not None:
            lines.append(f"  Drift PSI: {req.predictive.drift_psi:.3f} (>=0.2 suggests shift).")
        if req.predictive.fairness_by_group:
            for g, m in req.predictive.fairness_by_group.items():
                lines.append(f"  Group {g}: AUC={m['roc_auc']:.3f}, PosRate@thr={m['positive_rate@thr']:.3f}")
    if req.retrieval:
        lines.append(
            f"Retrieval — P@K: {req.retrieval.macro_precision_at_k:.3f}, "
            f"R@K: {req.retrieval.macro_recall_at_k:.3f}, MAP@K: {req.retrieval.macro_map_at_k:.3f}"
            + (f", nDCG@K: {req.retrieval.macro_ndcg_at_k:.3f}" if req.retrieval.macro_ndcg_at_k is not None else "")
        )
    if req.generation:
        lines.append(
            f"Generation — Relevance: {req.generation.answer_relevance:.3f}, "
            f"Faithfulness: {req.generation.faithfulness:.3f}, "
            f"Hallucination: {'YES' if req.generation.hallucination_flag else 'no'}."
        )
    return {"summary": "\n".join(lines) or "No sections provided."}
```

---

## How to run (local)

```bash
pip install fastapi uvicorn pydantic numpy
uvicorn evaluation_api:app --reload
# Service at http://127.0.0.1:8000/docs (interactive Swagger UI)
```

## Example requests (copy-paste into Swagger UI or curl)

**Predictive**

```json
POST /evaluate/predictive
{
  "y_true": [1,0,1,0,1,0,0,1,0,1],
  "y_scores": [0.9,0.2,0.7,0.4,0.8,0.3,0.1,0.65,0.5,0.85],
  "threshold": 0.6,
  "top_k_fraction": 0.2,
  "protected_attr": ["A","B","A","B","A","B","A","B","A","B"]
}
```

**Retrieval**

```json
POST /evaluate/retrieval
{
  "k": 5,
  "queries": [
    {"relevant_ids": ["d1","d3","d4"], "retrieved_ids": ["d1","d2","d3","d5","d4"], "gains":[3,0,2,0,1]},
    {"relevant_ids": ["x2","x9"], "retrieved_ids": ["x9","x1","x7","x2","x5"]}
  ]
}
```

**Generation**

```json
POST /evaluate/generation
{
  "question": "What reduces policy lapse risk?",
  "answer": "Auto-pay enrollment and tenure reduce lapse risk, per retention policy guide.",
  "sources": ["Retention policy guide recommends auto-pay and tenure as key drivers."]
}
```

**Natural-language summary**

```json
POST /report/summary
{
  "predictive": {...response from /evaluate/predictive...},
  "retrieval":  {...response from /evaluate/retrieval...},
  "generation": {...response from /evaluate/generation...}
}
```

---

## Notes & next steps

* This is intentionally **dependency-light** and fast to adopt. You can swap in your own metrics (e.g., SHAP summaries, calibration curves, Brier score, advanced hallucination detection) behind the same endpoints.
* Wire this service into CI/CD to auto-emit scorecards on every model build; publish JSON to your dashboards.
* Add **auth**, **rate-limits**, and **PII scrubbing** before production; keep a model card per version.

If you’d like, I can package this into a **Dockerfile + sample scorecard notebook** or add **fairness slices** (e.g., demographic parity ratio) as first-class fields.
