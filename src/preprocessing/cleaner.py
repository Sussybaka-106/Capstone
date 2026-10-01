"""
Data Cleaning Module for Public Procurement Records.
Compliant with Stage 6 of the Project Blueprint.
"""

from typing import Tuple
import pandas as pd
import numpy as np


class ProcurementDataCleaner:
    """Cleans and validates Tender, Vendor, and Bid datasets."""

    @staticmethod
    def clean_tenders(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans tender records: parses dates, strips whitespace, validates values."""
        df = df.copy()

        # Deduplicate on Tender_ID
        df = df.drop_duplicates(subset=["Tender_ID"])

        # String stripping and standardization
        string_cols = ["Department", "Category", "Location"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Parse dates
        df["Tender_Date"] = pd.to_datetime(df["Tender_Date"], errors="coerce")
        df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")

        # Validate Estimated Value
        df["Estimated_Value"] = pd.to_numeric(df["Estimated_Value"], errors="coerce")
        df = df[df["Estimated_Value"] > 0]

        # Calculate Tender Duration in days
        df["Tender_Duration_Days"] = (df["Deadline"] - df["Tender_Date"]).dt.days

        return df.reset_index(drop=True)

    @staticmethod
    def clean_vendors(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans vendor records: standardizes text and strips whitespace."""
        df = df.copy()

        df = df.drop_duplicates(subset=["Vendor_ID"])

        string_cols = ["Vendor_Name", "Address", "City", "Registration_ID", "Industry"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Normalized Name for comparison
        df["Normalized_Name"] = (
            df["Vendor_Name"]
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.strip()
        )

        return df.reset_index(drop=True)

    @staticmethod
    def clean_bids(df: pd.DataFrame, valid_tenders: pd.DataFrame = None, valid_vendors: pd.DataFrame = None) -> pd.DataFrame:
        """Cleans bid records: removes invalid bids and ensures integrity."""
        df = df.copy()

        # Deduplicate on Tender_ID + Vendor_ID
        df = df.drop_duplicates(subset=["Tender_ID", "Vendor_ID"])

        # Validate numeric bid amounts
        df["Bid_Amount"] = pd.to_numeric(df["Bid_Amount"], errors="coerce")
        df = df[df["Bid_Amount"] > 0]

        # Ensure Winner flag is binary
        df["Winner"] = df["Winner"].fillna(0).astype(int)

        # Refer integrity if reference DataFrames are provided
        if valid_tenders is not None:
            df = df[df["Tender_ID"].isin(valid_tenders["Tender_ID"])]

        if valid_vendors is not None:
            df = df[df["Vendor_ID"].isin(valid_vendors["Vendor_ID"])]

        return df.reset_index(drop=True)

    def clean_all(
        self,
        df_tenders: pd.DataFrame,
        df_vendors: pd.DataFrame,
        df_bids: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cleans all 3 datasets together maintaining relational integrity."""
        clean_t = self.clean_tenders(df_tenders)
        clean_v = self.clean_vendors(df_vendors)
        clean_b = self.clean_bids(df_bids, valid_tenders=clean_t, valid_vendors=clean_v)
        return clean_t, clean_v, clean_b
