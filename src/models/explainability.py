"""
Explainability Module using SHAP (SHapley Additive exPlanations).
Compliant with Stage 14 of the Project Blueprint.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import shap


class CollusionExplainer:
    """
    Computes local and global feature attribution explanations using SHAP TreeExplainer,
    translating model predictions into plain English evidence for fraud investigators.
    """

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_case(self, X_case: pd.DataFrame, top_k: int = 3) -> Dict[str, Any]:
        """
        Explains an individual tender case prediction.
        Returns:
            - top_features: list of dicts with feature name, actual value, and SHAP attribution value
            - text_summary: readable natural language explanation
        """
        X_case_clean = X_case[self.feature_names].fillna(0)
        shap_values = self.explainer.shap_values(X_case_clean)

        # In binary classification, shap_values can be a list [class_0, class_1] or single array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        feature_contribs = []
        for name, val, s in zip(self.feature_names, X_case_clean.iloc[0], sv):
            feature_contribs.append({
                "feature": name,
                "value": round(float(val), 4),
                "shap_impact": round(float(s), 4),
            })

        # Sort by highest positive impact (factors increasing suspicion)
        feature_contribs.sort(key=lambda x: x["shap_impact"], reverse=True)
        top_factors = feature_contribs[:top_k]

        # Build readable summary
        bullet_points = []
        for f in top_factors:
            if f["shap_impact"] > 0:
                name_clean = f["feature"].replace("_", " ").title()
                bullet_points.append(f"{name_clean} was {f['value']} (boosted suspicion by {f['shap_impact']:+.3f})")

        explanation_text = (
            "Primary drivers for investigation flag: " + "; ".join(bullet_points)
            if bullet_points
            else "Low suspicion — standard market competition characteristics."
        )

        return {
            "top_factors": top_factors,
            "explanation_text": explanation_text,
            "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value),
        }

    def explain_dataset(self, X: pd.DataFrame) -> List[str]:
        """Generates natural language explanations for an entire dataset of cases."""
        X_clean = X[self.feature_names].fillna(0)
        shap_values = self.explainer.shap_values(X_clean)

        if isinstance(shap_values, list):
            sv_matrix = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv_matrix = shap_values[:, :, 1]
        else:
            sv_matrix = shap_values

        explanations = []
        for i in range(len(X_clean)):
            row_sv = sv_matrix[i]
            # Find top 2 positive drivers
            sorted_indices = np.argsort(row_sv)[::-1][:2]
            drivers = []
            for idx in sorted_indices:
                if row_sv[idx] > 0.01:
                    fname = self.feature_names[idx].replace("_", " ")
                    fval = round(float(X_clean.iloc[i, idx]), 2)
                    drivers.append(f"{fname} = {fval}")

            if drivers:
                explanations.append(f"Flagged due to: {', '.join(drivers)}")
            else:
                explanations.append("Normal competitive bidding behaviour")

        return explanations
