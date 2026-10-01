"""
Unit tests for Heuristic Baseline, LightGBM Ranker, and SHAP Explainability.
"""

import pandas as pd
import numpy as np
from src.models.baseline_rules import BaselineHeuristicScorer
from src.models.anomaly_ranker import AnomalyRanker
from src.models.explainability import CollusionExplainer


def test_baseline_heuristic_scorer():
    scorer = BaselineHeuristicScorer()
    df = pd.DataFrame([
        {"Tender_ID": "T1", "winner_to_second_gap": 0.20, "max_pair_cobids": 20, "bid_count": 4, "winning_bid_ratio": 0.97, "bid_spread": 0.25},
        {"Tender_ID": "T2", "winner_to_second_gap": 0.02, "max_pair_cobids": 1, "bid_count": 5, "winning_bid_ratio": 0.90, "bid_spread": 0.05},
    ])

    df_scored = scorer.score(df)
    assert df_scored.iloc[0]["baseline_anomaly_score"] >= 50
    assert df_scored.iloc[1]["baseline_anomaly_score"] < 20


def test_anomaly_ranker_and_shap():
    ranker = AnomalyRanker(random_state=42)

    # Synthetic training records
    X = pd.DataFrame({
        "winner_to_second_gap": [0.25, 0.22, 0.01, 0.02, 0.03, 0.28],
        "max_pair_cobids": [15, 18, 1, 2, 1, 20],
        "winning_bid_ratio": [0.98, 0.97, 0.88, 0.91, 0.89, 0.96],
        "bid_count": [4, 4, 6, 5, 7, 3],
    })
    y = pd.Series([1, 1, 0, 0, 0, 1])

    ranker.fit(X, y)
    probs, ranks = ranker.predict_risk(X)

    assert len(probs) == 6
    assert ranks[0] <= 3  # suspicious should be ranked high

    # Test SHAP explanation
    explainer = CollusionExplainer(ranker.model, list(X.columns))
    case_exp = explainer.explain_case(X.iloc[[0]])
    assert "top_factors" in case_exp
    assert "explanation_text" in case_exp
    assert len(case_exp["top_factors"]) > 0
