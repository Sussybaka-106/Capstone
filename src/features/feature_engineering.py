"""
Feature Engineering Module for Collusion Risk Scoring.
Compliant with Stage 8 of the Project Blueprint.
"""

from typing import Tuple
import pandas as pd
import numpy as np


class ProcurementFeatureEngineer:
    """
    Computes pricing, behavioral, temporal, and interaction features
    across tenders, bids, and vendors to expose collusion signatures.
    """

    def compute_tender_pricing_features(
        self,
        df_tenders: pd.DataFrame,
        df_bids: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes tender-level pricing features (spread, CV, 1st-to-2nd gap, win ratio).
        """
        # Merge bid amounts with tender estimated value
        merged = df_bids.merge(
            df_tenders[["Tender_ID", "Estimated_Value"]],
            on="Tender_ID",
            how="inner",
        )
        merged["bid_ratio"] = merged["Bid_Amount"] / merged["Estimated_Value"]

        features = []
        for tender_id, group in merged.groupby("Tender_ID"):
            sorted_bids = group.sort_values("Bid_Amount").reset_index(drop=True)
            n_bids = len(sorted_bids)

            min_bid = sorted_bids["Bid_Amount"].iloc[0]
            max_bid = sorted_bids["Bid_Amount"].iloc[-1]
            est_val = sorted_bids["Estimated_Value"].iloc[0]

            winning_bid_ratio = min_bid / est_val
            mean_bid_ratio = sorted_bids["bid_ratio"].mean()
            std_bid_ratio = sorted_bids["bid_ratio"].std() if n_bids > 1 else 0.0
            bid_spread = (max_bid - min_bid) / est_val

            # Winner to 2nd lowest bid gap (key indicator for cover bidding)
            if n_bids > 1:
                second_min_bid = sorted_bids["Bid_Amount"].iloc[1]
                winner_to_second_gap = (second_min_bid - min_bid) / est_val
            else:
                winner_to_second_gap = 0.0

            # Winning vendor ID
            winner_row = sorted_bids[sorted_bids["Winner"] == 1]
            winning_vendor = (
                winner_row["Canonical_Vendor_ID"].iloc[0]
                if "Canonical_Vendor_ID" in winner_row.columns
                else (winner_row["Vendor_ID"].iloc[0] if len(winner_row) > 0 else "")
            )

            features.append({
                "Tender_ID": tender_id,
                "Winning_Vendor_ID": winning_vendor,
                "bid_count": n_bids,
                "winning_bid_ratio": winning_bid_ratio,
                "mean_bid_ratio": mean_bid_ratio,
                "std_bid_ratio": std_bid_ratio if not np.isnan(std_bid_ratio) else 0.0,
                "bid_spread": bid_spread,
                "bid_cv": (std_bid_ratio / mean_bid_ratio) if mean_bid_ratio > 0 and not np.isnan(std_bid_ratio) else 0.0,
                "winner_to_second_gap": winner_to_second_gap,
            })

        df_pricing = pd.DataFrame(features)
        return df_pricing

    def compute_vendor_behavioral_features(
        self,
        df_bids: pd.DataFrame,
        df_tenders: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes historical vendor metrics: participation counts, win rates, and average bid margins.
        """
        vendor_col = "Canonical_Vendor_ID" if "Canonical_Vendor_ID" in df_bids.columns else "Vendor_ID"

        merged = df_bids.merge(
            df_tenders[["Tender_ID", "Estimated_Value", "Category"]],
            on="Tender_ID",
            how="inner",
        )
        merged["bid_ratio"] = merged["Bid_Amount"] / merged["Estimated_Value"]

        vendor_stats = []
        for vendor_id, group in merged.groupby(vendor_col):
            tenders_entered = len(group)
            tenders_won = group["Winner"].sum()
            win_rate = tenders_won / tenders_entered if tenders_entered > 0 else 0.0
            avg_bid_ratio = group["bid_ratio"].mean()
            std_bid_ratio = group["bid_ratio"].std() if tenders_entered > 1 else 0.0

            # Primary category concentration
            cat_counts = group["Category"].value_counts()
            top_cat_share = cat_counts.iloc[0] / tenders_entered if len(cat_counts) > 0 else 1.0

            vendor_stats.append({
                "Vendor_ID": vendor_id,
                "vendor_total_bids": tenders_entered,
                "vendor_total_wins": tenders_won,
                "vendor_win_rate": win_rate,
                "vendor_avg_bid_ratio": avg_bid_ratio,
                "vendor_std_bid_ratio": std_bid_ratio if not np.isnan(std_bid_ratio) else 0.0,
                "vendor_category_concentration": top_cat_share,
            })

        return pd.DataFrame(vendor_stats)

    def compute_co_bidding_features(
        self,
        df_bids: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes co-bidding matrices and tender-level repeated partner metrics.
        High co-bidding frequency among a closed group of vendors is a classic cartel signature.
        """
        vendor_col = "Canonical_Vendor_ID" if "Canonical_Vendor_ID" in df_bids.columns else "Vendor_ID"

        # Find all pairs of vendors bidding on each tender
        tender_vendors = df_bids.groupby("Tender_ID")[vendor_col].unique().to_dict()

        pair_counts = {}
        for tender_id, vendors in tender_vendors.items():
            v_list = sorted(list(vendors))
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    pair = (v_list[i], v_list[j])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        pair_records = [
            {"Vendor_A": k[0], "Vendor_B": k[1], "co_bidding_count": v}
            for k, v in pair_counts.items()
        ]
        df_pairs = pd.DataFrame(pair_records)

        # Tender-level co-bidding intensity
        tender_cobid_features = []
        for tender_id, vendors in tender_vendors.items():
            v_list = sorted(list(vendors))
            if len(v_list) < 2:
                tender_cobid_features.append({
                    "Tender_ID": tender_id,
                    "max_pair_cobids": 0,
                    "mean_pair_cobids": 0.0,
                })
                continue

            cobid_values = []
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    pair = (v_list[i], v_list[j])
                    cobid_values.append(pair_counts.get(pair, 0))

            tender_cobid_features.append({
                "Tender_ID": tender_id,
                "max_pair_cobids": max(cobid_values),
                "mean_pair_cobids": float(np.mean(cobid_values)),
            })

        df_tender_cobid = pd.DataFrame(tender_cobid_features)
        return df_tender_cobid, df_pairs

    def build_feature_dataset(
        self,
        df_tenders: pd.DataFrame,
        df_bids: pd.DataFrame,
        df_vendors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combines tender pricing, co-bidding, and winning vendor behavioral features
        into a unified tabular dataset ready for risk modeling.
        """
        df_pricing = self.compute_tender_pricing_features(df_tenders, df_bids)
        df_vendor_stats = self.compute_vendor_behavioral_features(df_bids, df_tenders)
        df_tender_cobid, _ = self.compute_co_bidding_features(df_bids)

        # Merge pricing + co-bidding features
        df_features = df_tenders[["Tender_ID", "Department", "Category", "Location", "Estimated_Value", "Tender_Duration_Days"]].merge(
            df_pricing, on="Tender_ID", how="left"
        )
        df_features = df_features.merge(df_tender_cobid, on="Tender_ID", how="left")

        # Merge winning vendor behavioral stats
        df_features = df_features.merge(
            df_vendor_stats,
            left_on="Winning_Vendor_ID",
            right_on="Vendor_ID",
            how="left",
        )

        df_features = df_features.drop(columns=["Vendor_ID"], errors="ignore")

        # Fill potential NaNs in stats
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)

        return df_features
