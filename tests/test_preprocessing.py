"""
Unit tests for Data Cleaning and Entity Resolution.
"""

import pandas as pd
from src.preprocessing.cleaner import ProcurementDataCleaner
from src.preprocessing.entity_resolution import EntityResolver


def test_cleaner_removes_invalid_records():
    cleaner = ProcurementDataCleaner()

    # Raw sample tenders
    df_tenders = pd.DataFrame([
        {"Tender_ID": "T1", "Department": " PWD ", "Category": "Roads", "Location": "Pune", "Estimated_Value": 100000, "Tender_Date": "2025-01-01", "Deadline": "2025-01-20"},
        {"Tender_ID": "T1", "Department": "PWD", "Category": "Roads", "Location": "Pune", "Estimated_Value": 100000, "Tender_Date": "2025-01-01", "Deadline": "2025-01-20"},  # duplicate
        {"Tender_ID": "T2", "Department": "Health", "Category": "Meds", "Location": "Mumbai", "Estimated_Value": -500, "Tender_Date": "2025-01-01", "Deadline": "2025-01-10"},  # negative
    ])

    cleaned_t = cleaner.clean_tenders(df_tenders)
    assert len(cleaned_t) == 1
    assert cleaned_t.iloc[0]["Tender_ID"] == "T1"
    assert cleaned_t.iloc[0]["Department"] == "PWD"
    assert cleaned_t.iloc[0]["Tender_Duration_Days"] == 19


def test_entity_resolution_clusters_aliases():
    df_vendors = pd.DataFrame([
        {"Vendor_ID": "V1", "Vendor_Name": "Apex Infra Pvt Ltd", "Address": "100 MG Road", "City": "Mumbai"},
        {"Vendor_ID": "V2", "Vendor_Name": "Apex Infrastructure Private Limited", "Address": "100 MG Road", "City": "Mumbai"},
        {"Vendor_ID": "V3", "Vendor_Name": "Zenith Solutions", "Address": "200 Station Road", "City": "Pune"},
    ])

    resolver = EntityResolver(name_threshold=80.0)
    resolved_v, matches = resolver.resolve_entities(df_vendors)

    assert len(matches) >= 1
    v1_row = resolved_v[resolved_v["Vendor_ID"] == "V1"].iloc[0]
    v2_row = resolved_v[resolved_v["Vendor_ID"] == "V2"].iloc[0]
    v3_row = resolved_v[resolved_v["Vendor_ID"] == "V3"].iloc[0]

    # V1 and V2 should share the same canonical ID
    assert v1_row["Canonical_Vendor_ID"] == v2_row["Canonical_Vendor_ID"]
    assert v2_row["Is_Alias"] == 1
    # V3 is separate
    assert v3_row["Canonical_Vendor_ID"] == "V3"
    assert v3_row["Is_Alias"] == 0
