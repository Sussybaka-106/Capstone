"""
Machine Learning Anomaly Risk Ranking Model.
Compliant with Stage 13, 15, and 22 of the Project Blueprint.
"""

from typing import Dict, List, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score


FEATURE_COLUMNS = [
    "bid_count",
    "winning_bid_ratio",
    "mean_bid_ratio",
    "std_bid_ratio",
    "bid_spread",
    "bid_cv",
    "winner_to_second_gap",
    "max_pair_cobids",
    "mean_pair_cobids",
    "vendor_total_bids",
    "vendor_total_wins",
    "vendor_win_rate",
    "vendor_avg_bid_ratio",
    "vendor_std_bid_ratio",
    "vendor_category_concentration",
    "graph_degree_centrality",
    "graph_weighted_degree",
    "graph_betweenness",
    "graph_clustering_coeff",
    "graph_pagerank",
]


class AnomalyRanker:
    """
    Supervised LightGBM model trained to prioritize and rank suspicious procurement cases.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            random_state=random_state,
            verbose=-1,
        )
        self.fitted_features = []

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extracts available numerical features and imputes zeros for missing ones."""
        avail_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[avail_cols].copy().fillna(0)
        return X, avail_cols

    def evaluate_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """Runs Stratified K-Fold cross validation and computes Precision@K, ROC-AUC, and PR-AUC."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        auc_scores = []
        pr_auc_scores = []
        p_at_10 = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            clf = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                random_state=self.random_state,
                verbose=-1,
            )
            clf.fit(X_train, y_train)

            preds = clf.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, preds))
            pr_auc_scores.append(average_precision_score(y_val, preds))

            # Precision at Top 10 in validation set
            top_10_indices = np.argsort(preds)[::-1][:10]
            p_at_10.append(precision_score(y_val.iloc[top_10_indices], [1]*len(top_10_indices), zero_division=0))

        metrics = {
            "mean_roc_auc": round(float(np.mean(auc_scores)), 4),
            "mean_pr_auc": round(float(np.mean(pr_auc_scores)), 4),
            "mean_precision_at_10": round(float(np.mean(p_at_10)), 4),
        }
        return metrics

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the final model on the complete training set."""
        X_clean, features = self.prepare_features(X)
        self.fitted_features = features
        self.model.fit(X_clean, y)

    def predict_risk(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            1. Anomaly Probability [0, 1]
            2. Priority Rank (1 = highest risk)
        """
        X_clean = X[self.fitted_features].fillna(0)
        probs = self.model.predict_proba(X_clean)[:, 1]
        ranks = np.argsort(np.argsort(-probs)) + 1
        return probs, ranks

    def save(self, filepath: str):
        """Persists trained model and feature list."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "features": self.fitted_features}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AnomalyRanker":
        """Loads serialized model."""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.fitted_features = data["features"]
        return instance
