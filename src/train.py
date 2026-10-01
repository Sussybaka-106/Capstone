"""
Model Training, Graph Analytics, SHAP Explanation, and Case Ranking Pipeline.
Executes Stages 10, 11, 12, 13, 14, 15, and 22 of the Project Blueprint.
"""

import os
import pandas as pd
import numpy as np

from src.graph.bipartite_graph import ProcurementGraphBuilder
from src.models.baseline_rules import BaselineHeuristicScorer
from src.models.anomaly_ranker import AnomalyRanker
from src.models.explainability import CollusionExplainer


def train_and_evaluate_pipeline(
    data_dir: str = "data",
    model_output_dir: str = "models",
) -> dict:
    proc_dir = os.path.join(data_dir, "processed")
    gt_dir = os.path.join(data_dir, "ground_truth")
    os.makedirs(model_output_dir, exist_ok=True)

    print("--> 1. Loading processed data & ground truth...")
    df_features = pd.read_csv(os.path.join(proc_dir, "tender_features.csv"))
    df_bids = pd.read_csv(os.path.join(proc_dir, "clean_bids.csv"))
    df_gt = pd.read_csv(os.path.join(gt_dir, "ground_truth_collusion.csv"))

    # Stage 10 & 11: NetworkX Graph Analytics
    print("--> 2. Constructing Bipartite Graph & Vendor Co-Bidding Network...")
    graph_builder = ProcurementGraphBuilder()
    graph_builder.build_bipartite_graph(df_bids)
    graph_builder.build_vendor_cobidding_network(df_bids)

    df_graph_metrics = graph_builder.compute_graph_metrics()
    df_communities, candidate_rings = graph_builder.detect_collusion_communities()

    print(f"    Vendor Network contains {len(graph_builder.vendor_network.nodes())} vendors and {len(graph_builder.vendor_network.edges())} co-bidding edges.")
    print(f"    Detected {len(candidate_rings)} dense candidate rings through community detection.")

    # Merge graph metrics into tender features via Winning_Vendor_ID
    df_features = df_features.merge(
        df_graph_metrics,
        left_on="Winning_Vendor_ID",
        right_on="Vendor_ID",
        how="left"
    ).drop(columns=["Vendor_ID"], errors="ignore")

    # Fill graph NaNs
    graph_cols = [c for c in df_graph_metrics.columns if c != "Vendor_ID"]
    df_features[graph_cols] = df_features[graph_cols].fillna(0)

    # Stage 12: Baseline Heuristic Scorer
    print("--> 3. Scoring cases with Baseline Heuristic Engine (Stage 12)...")
    heuristic_scorer = BaselineHeuristicScorer()
    df_features = heuristic_scorer.score(df_features)

    # Merge ground-truth target
    df_merged = df_features.merge(
        df_gt[["Tender_ID", "Is_Collusion", "Collusion_Type", "Ring_ID"]],
        on="Tender_ID",
        how="left"
    )
    y = df_merged["Is_Collusion"].fillna(0).astype(int)

    # Stage 13: LightGBM Anomaly Ranker
    print("--> 4. Training LightGBM Risk Ranking Model (Stage 13)...")
    ranker = AnomalyRanker(random_state=42)
    X, feature_cols = ranker.prepare_features(df_merged)

    print("    Running 5-Fold Stratified Cross-Validation...")
    cv_metrics = ranker.evaluate_cv(X, y, n_splits=5)
    print(f"    CV ROC-AUC:            {cv_metrics['mean_roc_auc']:.4f}")
    print(f"    CV PR-AUC:             {cv_metrics['mean_pr_auc']:.4f}")
    print(f"    CV Precision@10:       {cv_metrics['mean_precision_at_10']:.4f}")

    # Fit final model
    ranker.fit(X, y)
    probs, ranks = ranker.predict_risk(X)
    df_merged["ml_anomaly_prob"] = np.round(probs, 4)
    df_merged["priority_rank"] = ranks

    # Stage 14: SHAP Explainability
    print("--> 5. Generating SHAP Explainability Annotations (Stage 14)...")
    explainer = CollusionExplainer(ranker.model, feature_cols)
    explanations = explainer.explain_dataset(X)
    df_merged["shap_explanation"] = explanations

    # Stage 15: Prioritized Investigation Queue
    print("--> 6. Building Prioritized Investigation Queue (Stage 15)...")
    df_queue = df_merged.sort_values("priority_rank").reset_index(drop=True)

    # Stage 22: Evaluation Metrics Calculation
    print("--> 7. Measuring Stage 22 Evaluation Metrics...")
    top_50 = df_queue.head(50)
    top_80 = df_queue.head(80)

    precision_at_50 = float(top_50["Is_Collusion"].mean())
    precision_at_80 = float(top_80["Is_Collusion"].mean())
    ring_recovery = float(df_queue[df_queue["Is_Collusion"] == 1]["priority_rank"].le(100).mean())

    print(f"    Evaluation Report:")
    print(f"    - Precision@50:                  {precision_at_50*100:.1f}%")
    print(f"    - Precision@80 (Target ring size): {precision_at_80*100:.1f}%")
    print(f"    - Injected-Ring Recovery in Top 100: {ring_recovery*100:.1f}%")

    # Persist artifacts
    model_path = os.path.join(model_output_dir, "anomaly_ranker.joblib")
    ranker.save(model_path)

    queue_path = os.path.join(proc_dir, "investigation_queue.csv")
    df_queue.to_csv(queue_path, index=False)

    communities_path = os.path.join(proc_dir, "vendor_communities.csv")
    df_communities.to_csv(communities_path, index=False)

    print(f"--> Saved trained model to {model_path}")
    print(f"--> Saved investigation queue to {queue_path}")

    return {
        "cv_roc_auc": cv_metrics["mean_roc_auc"],
        "precision_at_50": precision_at_50,
        "precision_at_80": precision_at_80,
        "ring_recovery": ring_recovery,
        "model_path": model_path,
        "queue_path": queue_path,
    }


if __name__ == "__main__":
    train_and_evaluate_pipeline()
