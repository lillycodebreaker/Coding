#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAA Life - Lapse Prediction Mock Case
-------------------------------------
Usage:
  python aaa_lapse_model.py --csv path/to/data.csv  # if you have a dataset
  python aaa_lapse_model.py                          # uses synthetic data

The script:
- Loads a CSV (or generates synthetic data)
- Preprocesses (encoding, imputation, scaling where needed)
- Trains Logistic Regression and HistGradientBoostingClassifier
- Evaluates with ROC-AUC, PR-AUC, and cost-based threshold selection
- Produces feature importance and example local "explanations"
- Generates templated "agent brief" and "customer message" for top-risk cases
"""

import argparse
import os
import json
import math
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore", category=UserWarning)


SEED = 42
rng = np.random.default_rng(SEED)


def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """Generate a synthetic dataset roughly matching the schema in the case."""
    product_type = rng.choice(["term_10", "term_20", "whole_life", "ul", "iul"], size=n, p=[0.25, 0.35, 0.15, 0.15, 0.10])
    premium_mode = rng.choice(["annual", "semiannual", "quarterly", "monthly"], size=n, p=[0.10, 0.15, 0.25, 0.50])
    payment_method = rng.choice(["ach", "credit_card", "check"], size=n, p=[0.55, 0.35, 0.10])
    agent_channel = rng.choice(["captive", "independent", "online"], size=n, p=[0.55, 0.35, 0.10])
    underwriting_class = rng.choice(["pref_plus", "preferred", "standard", "substandard"], size=n, p=[0.15, 0.40, 0.35, 0.10])
    e_bill_enrolled = rng.integers(0, 2, size=n)
    portal_account_created = (e_bill_enrolled | (rng.random(size=n) < 0.6)).astype(int)
    customer_service_contacts_30d = rng.poisson(0.3, size=n)
    email_open_rate_30d = np.clip(rng.normal(0.45, 0.2, size=n), 0, 1)

    # Numerics
    age_at_issue = rng.integers(20, 70, size=n)
    credit_proxy_score = np.clip(rng.normal(0.6, 0.18, size=n), 0, 1)
    has_prior_aaa_membership = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    tenure_aaa_membership_months = (has_prior_aaa_membership * rng.integers(1, 120, size=n)).astype(int)
    uw_decision_days = np.maximum(0, (rng.normal(12, 6, size=n))).astype(int)
    first_premium_paid_days = np.maximum(0, (rng.normal(3, 4, size=n))).astype(int)
    face_amount = np.maximum(10000, rng.normal(250000, 80000, size=n)).astype(int)

    # Base lapse propensity from some drivers
    # Higher risk: monthly, check, no e-bill, low credit score, lower engagement, substandard UW
    base = (
        0.15
        + 0.10 * (premium_mode == "monthly")
        + 0.06 * (payment_method == "check")
        - 0.08 * (payment_method == "ach")
        - 0.08 * e_bill_enrolled
        - 0.05 * portal_account_created
        + 0.10 * (underwriting_class == "substandard")
        + 0.05 * (underwriting_class == "standard")
        - 0.04 * (underwriting_class == "pref_plus")
        - 0.12 * has_prior_aaa_membership
        - 0.10 * (tenure_aaa_membership_months > 12)
        - 0.10 * email_open_rate_30d
        - 0.12 * credit_proxy_score
        + 0.03 * (customer_service_contacts_30d > 1)
        + 0.02 * (uw_decision_days > 20)
        + 0.01 * (first_premium_paid_days > 7)
    )

    logits = base + rng.normal(0, 0.15, size=n)
    prob = 1 / (1 + np.exp(-logits))
    lapse_12m = (rng.random(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "product_type": product_type,
        "premium_mode": premium_mode,
        "payment_method": payment_method,
        "agent_channel": agent_channel,
        "underwriting_class": underwriting_class,
        "e_bill_enrolled": e_bill_enrolled,
        "portal_account_created": portal_account_created,
        "customer_service_contacts_30d": customer_service_contacts_30d,
        "email_open_rate_30d": email_open_rate_30d,
        "age_at_issue": age_at_issue,
        "credit_proxy_score": credit_proxy_score,
        "has_prior_aaa_membership": has_prior_aaa_membership,
        "tenure_aaa_membership_months": tenure_aaa_membership_months,
        "uw_decision_days": uw_decision_days,
        "first_premium_paid_days": first_premium_paid_days,
        "face_amount": face_amount,
        "lapse_12m": lapse_12m,
    })
    return df


def load_or_generate(csv_path: str = None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    return generate_synthetic_data()


def build_pipelines(cat_cols: List[str], num_cols: List[str]) -> Tuple[Pipeline, Pipeline]:
    """Return (logistic_pipeline, hgb_pipeline)."""
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ]
    )

    logreg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=SEED))
    ])

    hgb = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=None,
            max_iter=300,
            l2_regularization=0.0,
            random_state=SEED
        ))
    ])

    return logreg, hgb


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    return {"roc_auc": roc, "pr_auc": pr}


def choose_threshold_by_utility(y_true: np.ndarray, y_prob: np.ndarray,
                                benefit_retain: float = 350.0,
                                cost_outreach: float = 7.0) -> Dict[str, float]:
    """
    Choose a threshold that maximizes expected utility:
    Utility = TP * benefit_retain - (TP + FP) * cost_outreach
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)  # align lengths
    tp = recall * y_true.sum()
    fp = (tp / (precision + 1e-12)) - tp
    utility = tp * benefit_retain - (tp + fp) * cost_outreach

    best_idx = int(np.argmax(utility))
    return {
        "best_threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "utility": float(utility[best_idx])
    }


def top_feature_importances(pipe: Pipeline, feature_names: List[str], top_k: int = 15) -> pd.DataFrame:
    """Extract feature importances for HistGradientBoostingClassifier or coefficients for LogisticRegression."""
    model = pipe.named_steps["clf"]
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    else:
        # For logistic regression, approximate by absolute coefficient values
        if hasattr(model, "coef_"):
            imps = np.abs(model.coef_).ravel()
        else:
            imps = np.zeros(len(feature_names))
    df = pd.DataFrame({"feature": feature_names, "importance": imps})
    df = df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    return df


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve transformed feature names from the ColumnTransformer + OneHotEncoder."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "named_steps"):
            # Pipeline
            last = list(trans.named_steps.values())[-1]
            if isinstance(last, OneHotEncoder):
                # expand categories
                ohe = last
                cats = ohe.categories_
                ohe_names = []
                for col, cat_list in zip(cols, cats):
                    ohe_names.extend([f"{col}={cat}" for cat in cat_list])
                feature_names.extend(ohe_names)
            else:
                feature_names.extend(cols)
        else:
            # direct transformer (e.g., 'drop' or passthrough)
            if isinstance(trans, OneHotEncoder):
                cats = trans.categories_
                ohe_names = []
                for col, cat_list in zip(cols, cats):
                    ohe_names.extend([f"{col}={cat}" for cat in cat_list])
                feature_names.extend(ohe_names)
            elif trans == "drop":
                continue
            else:
                feature_names.extend(cols)
    return feature_names


def local_explanations(sample_row: pd.Series, feature_importance_df: pd.DataFrame) -> List[str]:
    """
    Very simple heuristic "local explanation":
    - Report if sample_row has high-risk patterns among top global features (e.g., monthly, check, no e-bill).
    """
    notes = []
    s = sample_row
    # Example heuristics matching synthetic generation
    if "premium_mode" in s and s["premium_mode"] == "monthly":
        notes.append("Monthly premium mode increases lapse risk.")
    if "payment_method" in s and s["payment_method"] == "check":
        notes.append("Check payment method correlates with higher lapse risk; consider switching to ACH.")
    if "e_bill_enrolled" in s and int(s["e_bill_enrolled"]) == 0:
        notes.append("Not enrolled in e-bill; enrollment improves payment stability.")
    if "portal_account_created" in s and int(s["portal_account_created"]) == 0:
        notes.append("No portal account created; engagement is low.")
    if "credit_proxy_score" in s and float(s["credit_proxy_score"]) < 0.45:
        notes.append("Lower credit proxy score indicates higher risk.")
    if "tenure_aaa_membership_months" in s and int(s["tenure_aaa_membership_months"]) < 6:
        notes.append("Short AAA membership tenure; weaker relationship signal.")
    if not notes:
        notes.append("Primary drivers include payment mode/method and engagement signals.")
    return notes[:3]  # top 3 reasons


def agent_brief(policy_id: str, risk: float, reasons: List[str]) -> str:
    return (
        f"Policy {policy_id or 'N/A'} is high risk ({risk:.2f}). "
        f"Drivers: {', '.join(reasons)}. "
        f"Recommended actions: offer ACH incentive, guide e-bill enrollment, schedule agent follow-up within 7 days."
    )


def customer_message(risk: float) -> str:
    return (
        "Hi there — we want to make managing your policy easy. "
        "You can set up automatic payments (ACH) and paperless e‑billing in minutes. "
        "If you have any questions, we’re here to help."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV file (optional).")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=10, help="Top K feature importances to display.")
    ap.add_argument("--benefit_retain", type=float, default=350.0, help="Benefit value for a true positive (retained policy).")
    ap.add_argument("--cost_outreach", type=float, default=7.0, help="Cost per outreach.")
    args = ap.parse_args()

    df = load_or_generate(args.csv)
    if "lapse_12m" not in df.columns:
        raise ValueError("Dataset must contain a 'lapse_12m' (0/1) target column.")

    target = "lapse_12m"
    y = df[target].astype(int).values
    X = df.drop(columns=[target])

    # Basic schema guess
    cat_cols = [c for c in X.columns if X[c].dtype == "object"] + \
               [c for c in X.columns if str(X[c].dtype).startswith("int") and X[c].nunique() < 10]
    # Keep small-cardinality ints as cats (e.g., 0/1 flags)
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y
    )

    logreg_pipe, hgb_pipe = build_pipelines(cat_cols, num_cols)

    # Fit
    logreg_pipe.fit(X_train, y_train)
    hgb_pipe.fit(X_train, y_train)

    # Predict
    logreg_prob = logreg_pipe.predict_proba(X_test)[:, 1]
    hgb_prob = hgb_pipe.predict_proba(X_test)[:, 1]

    # Evaluate
    logreg_metrics = evaluate(y_test, logreg_prob)
    hgb_metrics = evaluate(y_test, hgb_prob)

    # Select best by PR-AUC
    best_name = "HistGradientBoosting" if hgb_metrics["pr_auc"] >= logreg_metrics["pr_auc"] else "LogisticRegression"
    best_pipe = hgb_pipe if best_name == "HistGradientBoosting" else logreg_pipe
    best_prob = hgb_prob if best_name == "HistGradientBoosting" else logreg_prob
    best_metrics = hgb_metrics if best_name == "HistGradientBoosting" else logreg_metrics

    # Threshold by utility
    thr_info = choose_threshold_by_utility(
        y_test, best_prob, benefit_retain=args.benefit_retain, cost_outreach=args.cost_outreach
    )

    # Feature importances
    pre = best_pipe.named_steps["preprocess"]
    feature_names = get_feature_names(pre)
    # Refit on entire training set to ensure attributes are populated for importance
    # (Already fitted; retrieving is fine.)
    # Need access to internal model features
    # For HistGB, feature_importances_ exist; for LogReg, coef_ exist post-fit
    best_clf = best_pipe.named_steps["clf"]
    top_imp = top_feature_importances(best_pipe, feature_names, top_k=args.top_k)

    # Pick top risk cases and generate briefs/messages
    test_df = X_test.copy().reset_index(drop=True)
    test_df["prob"] = best_prob
    test_df["y"] = y_test
    test_df_sorted = test_df.sort_values("prob", ascending=False).head(5).reset_index(drop=True)

    briefs = []
    for i in range(min(5, len(test_df_sorted))):
        row = test_df_sorted.loc[i]
        reasons = local_explanations(row, top_imp)
        briefs.append({
            "rank": int(i+1),
            "risk_prob": float(row["prob"]),
            "reasons": reasons,
            "agent_brief": agent_brief(policy_id=row.get("policy_id", f"TEST_{i+1}"), risk=row["prob"], reasons=reasons),
            "customer_message": customer_message(risk=row["prob"]),
        })

    # Results summary
    summary = {
        "model_selected": best_name,
        "metrics": best_metrics,
        "threshold_selection": thr_info,
        "top_feature_importances": top_imp.to_dict(orient="records"),
        "top_risk_cases": briefs
    }

    out_dir = "aaa_outputs"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    top_imp.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)

    # Also export the top risk briefs as a CSV
    briefs_df = pd.DataFrame([{
        "rank": b["rank"],
        "risk_prob": b["risk_prob"],
        "reasons": " | ".join(b["reasons"]),
        "agent_brief": b["agent_brief"],
        "customer_message": b["customer_message"]
    } for b in briefs])
    briefs_df.to_csv(os.path.join(out_dir, "top_risk_briefs.csv"), index=False)

    print("\n=== AAA Life Lapse Prediction - Summary ===")
    print(f"Selected model: {best_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.3f} | PR-AUC: {best_metrics['pr_auc']:.3f}")
    print("Threshold (by utility): "
          f"thr={thr_info['best_threshold']:.3f}, precision={thr_info['precision']:.3f}, "
          f"recall={thr_info['recall']:.3f}, utility={thr_info['utility']:.1f}")
    print("\nTop feature importances:")
    print(top_imp)

    print("\nTop 5 high-risk cases (agent briefs):")
    for b in briefs:
        print(f"  - {b['agent_brief']}")

    print(f"\nArtifacts written to: {os.path.abspath(out_dir)}")
    print(" - summary.json")
    print(" - feature_importances.csv")
    print(" - top_risk_briefs.csv")
    print("\nTip: pass --csv yourfile.csv to run on a real dataset.\n")


if __name__ == "__main__":
    main()
