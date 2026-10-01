"""
End-to-End Preprocessing & Feature Engineering Pipeline.
Executes Stages 6, 7, and 8 of the Project Blueprint.
"""

import os
import pandas as pd
from src.preprocessing.cleaner import ProcurementDataCleaner
from src.preprocessing.entity_resolution import EntityResolver
from src.features.feature_engineering import ProcurementFeatureEngineer


def run_preprocessing_and_features(
    data_dir: str = "data",
) -> dict:
    """Runs data cleaning, entity resolution, and feature engineering."""
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)

    print("--> Loading raw procurement data...")
    df_tenders = pd.read_csv(os.path.join(raw_dir, "tenders.csv"))
    df_vendors = pd.read_csv(os.path.join(raw_dir, "vendors.csv"))
    df_bids = pd.read_csv(os.path.join(raw_dir, "bids.csv"))

    # Stage 6: Data Cleaning
    print("--> Stage 6: Cleaning tenders, vendors, and bids...")
    cleaner = ProcurementDataCleaner()
    clean_t, clean_v, clean_b = cleaner.clean_all(df_tenders, df_vendors, df_bids)

    # Stage 7: Entity Resolution
    print("--> Stage 7: Executing fuzzy entity resolution...")
    resolver = EntityResolver(name_threshold=85.0, combined_threshold=75.0)
    resolved_v, match_log = resolver.resolve_entities(clean_v)
    resolved_b = resolver.map_bids_to_canonical(clean_b, resolved_v)

    aliases_found = resolved_v["Is_Alias"].sum()
    print(f"    Discovered {aliases_found} alias/duplicate vendor profiles clustered into canonical IDs.")

    # Stage 8: Feature Engineering
    print("--> Stage 8: Computing pricing anomalies and co-bidding features...")
    fe = ProcurementFeatureEngineer()
    df_features = fe.build_feature_dataset(clean_t, resolved_b, resolved_v)
    _, df_pairs = fe.compute_co_bidding_features(resolved_b)

    # Persist processed artifacts
    clean_t.to_csv(os.path.join(proc_dir, "clean_tenders.csv"), index=False)
    resolved_v.to_csv(os.path.join(proc_dir, "clean_vendors.csv"), index=False)
    resolved_b.to_csv(os.path.join(proc_dir, "clean_bids.csv"), index=False)
    match_log.to_csv(os.path.join(proc_dir, "entity_resolution_matches.csv"), index=False)
    df_features.to_csv(os.path.join(proc_dir, "tender_features.csv"), index=False)
    df_pairs.to_csv(os.path.join(proc_dir, "co_bidding_pairs.csv"), index=False)

    print(f"--> Successfully generated feature matrix: {df_features.shape[0]} tenders x {df_features.shape[1]} features.")
    print(f"--> Saved all processed artifacts to {proc_dir}/")

    return {
        "features_shape": df_features.shape,
        "aliases_detected": aliases_found,
    }


if __name__ == "__main__":
    run_preprocessing_and_features()
