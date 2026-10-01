"""
Baseline Rule-Based Collusion Anomaly Scorer.
Compliant with Stage 12 of the Project Blueprint.
"""

import pandas as pd
import numpy as np


class BaselineHeuristicScorer:
    """
    Expert heuristic rule engine that computes a weighted collusion risk score (0-100)
    based on procurement red flags established by competition authorities (OECD / CCI).
    """

    def score(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates heuristic collusion risk score and generates triggered red flag descriptions.
        """
        df = df_features.copy()

        scores = []
        flags_list = []

        for _, row in df.iterrows():
            score = 0.0
            flags = []

            # Red Flag 1: Large gap between 1st (winning) bid and 2nd bid (Cover Bidding signature)
            gap = row.get("winner_to_second_gap", 0.0)
            if gap >= 0.15:
                score += 30.0
                flags.append(f"Severe 1st-to-2nd bid gap ({gap*100:.1f}%) suggesting cover bids")
            elif gap >= 0.08:
                score += 15.0
                flags.append(f"Moderate 1st-to-2nd bid gap ({gap*100:.1f}%)")

            # Red Flag 2: High repeated co-bidding frequency
            max_cobids = row.get("max_pair_cobids", 0)
            if max_cobids >= 15:
                score += 25.0
                flags.append(f"Repeated co-bidding ring (paired {max_cobids} times)")
            elif max_cobids >= 8:
                score += 12.0
                flags.append(f"Frequent co-bidding ({max_cobids} times)")

            # Red Flag 3: Low bid count (restricted competition)
            bid_cnt = row.get("bid_count", 0)
            if bid_cnt <= 2:
                score += 15.0
                flags.append(f"Low bidder turnout ({bid_cnt} bids)")

            # Red Flag 4: Winning bid close to budget ceiling while spread is high
            win_ratio = row.get("winning_bid_ratio", 0.0)
            spread = row.get("bid_spread", 0.0)
            if win_ratio >= 0.95 and spread >= 0.15:
                score += 15.0
                flags.append(f"Winning bid near ceiling ({win_ratio*100:.1f}%) with wide bid spread ({spread*100:.1f}%)")

            # Red Flag 5: Winning vendor has high win rate
            win_rate = row.get("vendor_win_rate", 0.0)
            total_bids = row.get("vendor_total_bids", 0)
            if total_bids >= 5 and win_rate >= 0.50:
                score += 15.0
                flags.append(f"Dominant winner win-rate ({win_rate*100:.1f}% across {total_bids} tenders)")

            # Cap score at 100
            score = min(100.0, score)
            scores.append(round(score, 1))
            flags_list.append(" | ".join(flags) if flags else "Normal bidding characteristics")

        df["baseline_anomaly_score"] = scores
        df["baseline_red_flags"] = flags_list

        return df
