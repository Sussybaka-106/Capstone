"""
Unit tests for Synthetic Procurement Data Generation & Collusion Injection.
"""

import os
import pandas as pd
import pytest
from src.data_generation.generator import ProcurementDataGenerator, generate_and_save_dataset


def test_vendor_generation():
    generator = ProcurementDataGenerator(seed=123)
    df_vendors = generator.generate_vendors(num_vendors=50)

    assert len(df_vendors) == 50
    assert "Vendor_ID" in df_vendors.columns
    assert "Vendor_Name" in df_vendors.columns
    assert "Registration_ID" in df_vendors.columns
    assert df_vendors["Vendor_ID"].nunique() == 50


def test_alias_injection():
    generator = ProcurementDataGenerator(seed=123)
    df_vendors = generator.generate_vendors(num_vendors=20)
    df_with_aliases, alias_log = generator.inject_entity_aliases(df_vendors, num_alias_pairs=3)

    assert len(df_with_aliases) == 23
    assert len(alias_log) == 3
    assert df_with_aliases["Vendor_ID"].nunique() == 23


def test_tender_and_bid_generation():
    generator = ProcurementDataGenerator(seed=123)
    df_vendors = generator.generate_vendors(num_vendors=30)
    df_tenders = generator.generate_tenders(num_tenders=50)

    assert len(df_tenders) == 50
    assert "Estimated_Value" in df_tenders.columns
    assert (df_tenders["Estimated_Value"] > 0).all()

    df_bids, df_gt = generator.generate_bids_and_collusion(df_tenders, df_vendors, collusion_rate=0.20)

    assert len(df_gt) == 50
    assert df_gt["Is_Collusion"].sum() == 10
    assert len(df_bids) > 50

    # Ensure every tender has exactly one winner (Rank 1)
    winners_per_tender = df_bids.groupby("Tender_ID")["Winner"].sum()
    assert (winners_per_tender == 1).all()

    # Ensure all bids have valid positive amounts
    assert (df_bids["Bid_Amount"] > 0).all()
