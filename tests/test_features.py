"""
Unit tests for Feature Engineering.
"""

import pandas as pd
from src.features.feature_engineering import ProcurementFeatureEngineer


def test_pricing_features_calculation():
    fe = ProcurementFeatureEngineer()

    df_tenders = pd.DataFrame([
        {"Tender_ID": "T1", "Estimated_Value": 1000.0, "Category": "IT"},
    ])
    df_bids = pd.DataFrame([
        {"Tender_ID": "T1", "Vendor_ID": "V1", "Canonical_Vendor_ID": "V1", "Bid_Amount": 900.0, "Winner": 1},
        {"Tender_ID": "T1", "Vendor_ID": "V2", "Canonical_Vendor_ID": "V2", "Bid_Amount": 1100.0, "Winner": 0},
    ])

    df_pricing = fe.compute_tender_pricing_features(df_tenders, df_bids)

    assert len(df_pricing) == 1
    row = df_pricing.iloc[0]
    assert row["Tender_ID"] == "T1"
    assert row["Winning_Vendor_ID"] == "V1"
    assert row["winning_bid_ratio"] == 0.90
    assert row["bid_spread"] == 0.20  # (1100 - 900) / 1000
    assert row["winner_to_second_gap"] == 0.20  # (1100 - 900) / 1000


def test_co_bidding_calculation():
    fe = ProcurementFeatureEngineer()

    df_bids = pd.DataFrame([
        {"Tender_ID": "T1", "Vendor_ID": "V1"},
        {"Tender_ID": "T1", "Vendor_ID": "V2"},
        {"Tender_ID": "T2", "Vendor_ID": "V1"},
        {"Tender_ID": "T2", "Vendor_ID": "V2"},
        {"Tender_ID": "T2", "Vendor_ID": "V3"},
    ])

    df_tender_cobid, df_pairs = fe.compute_co_bidding_features(df_bids)

    pair_v1_v2 = df_pairs[
        ((df_pairs["Vendor_A"] == "V1") & (df_pairs["Vendor_B"] == "V2"))
        | ((df_pairs["Vendor_A"] == "V2") & (df_pairs["Vendor_B"] == "V1"))
    ]
    assert len(pair_v1_v2) == 1
    assert pair_v1_v2.iloc[0]["co_bidding_count"] == 2
