"""
FastAPI Backend for Public Procurement Collusion Analytics.
Compliant with Stage 16 of the Project Blueprint.
"""

from typing import Dict, List, Optional
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Public Procurement Collusion Analytics API",
    description="Backend API for tender anomaly ranking, vendor network analysis, and explainable collusion scoring.",
    version="1.0.0",
)

# Enable CORS for dashboard/web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_data():
    queue_path = os.path.join(DATA_DIR, "investigation_queue.csv")
    bids_path = os.path.join(DATA_DIR, "clean_bids.csv")
    vendors_path = os.path.join(DATA_DIR, "clean_vendors.csv")
    cobids_path = os.path.join(DATA_DIR, "co_bidding_pairs.csv")

    df_queue = pd.read_csv(queue_path) if os.path.exists(queue_path) else pd.DataFrame()
    df_bids = pd.read_csv(bids_path) if os.path.exists(bids_path) else pd.DataFrame()
    df_vendors = pd.read_csv(vendors_path) if os.path.exists(vendors_path) else pd.DataFrame()
    df_cobids = pd.read_csv(cobids_path) if os.path.exists(cobids_path) else pd.DataFrame()

    return df_queue, df_bids, df_vendors, df_cobids


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "collusion-analytics-api"}


@app.get("/kpis")
def get_kpis():
    """Returns high-level procurement analytics KPIs."""
    df_queue, _, df_vendors, _ = load_data()
    if df_queue.empty:
        return {"error": "Pipeline has not been executed yet."}

    total_tenders = len(df_queue)
    flagged_high_risk = int((df_queue["ml_anomaly_prob"] >= 0.70).sum())
    flagged_moderate_risk = int(((df_queue["ml_anomaly_prob"] >= 0.40) & (df_queue["ml_anomaly_prob"] < 0.70)).sum())
    aliases_resolved = int(df_vendors["Is_Alias"].sum()) if not df_vendors.empty else 0

    return {
        "total_tenders_analyzed": total_tenders,
        "high_risk_cases": flagged_high_risk,
        "moderate_risk_cases": flagged_moderate_risk,
        "total_vendors": len(df_vendors),
        "aliases_resolved": aliases_resolved,
    }


@app.get("/cases")
def list_cases(
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
    department: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    """Returns the prioritized investigation queue with optional filtering."""
    df_queue, _, _, _ = load_data()
    if df_queue.empty:
        return []

    filtered = df_queue[df_queue["ml_anomaly_prob"] >= min_risk]
    if department:
        filtered = filtered[filtered["Department"].str.contains(department, case=False, na=False)]

    result = filtered.head(limit).to_dict(orient="records")
    return result


@app.get("/cases/{tender_id}")
def get_case_detail(tender_id: str):
    """Retrieves full forensic details, submitted bids, and explanations for a specific tender."""
    df_queue, df_bids, df_vendors, _ = load_data()

    case_row = df_queue[df_queue["Tender_ID"] == tender_id]
    if case_row.empty:
        raise HTTPException(status_code=404, detail=f"Tender {tender_id} not found.")

    case_dict = case_row.iloc[0].to_dict()

    # Get all submitted bids for this tender
    bids = df_bids[df_bids["Tender_ID"] == tender_id].sort_values("Rank").to_dict(orient="records")

    return {
        "case_overview": case_dict,
        "submitted_bids": bids,
    }


@app.get("/vendors/{vendor_id}")
def get_vendor_profile(vendor_id: str):
    """Retrieves vendor profile, known aliases, win stats, and top co-bidders."""
    df_queue, df_bids, df_vendors, df_cobids = load_data()

    v_row = df_vendors[df_vendors["Vendor_ID"] == vendor_id]
    if v_row.empty:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found.")

    vendor_info = v_row.iloc[0].to_dict()

    # Find aliases sharing the same canonical ID
    canonical_id = vendor_info.get("Canonical_Vendor_ID", vendor_id)
    aliases = df_vendors[
        (df_vendors["Canonical_Vendor_ID"] == canonical_id) & (df_vendors["Vendor_ID"] != vendor_id)
    ].to_dict(orient="records")

    # Find frequent co-bidding partners
    partner_records = df_cobids[
        (df_cobids["Vendor_A"] == canonical_id) | (df_cobids["Vendor_B"] == canonical_id)
    ].sort_values("co_bidding_count", ascending=False).head(10).to_dict(orient="records")

    return {
        "vendor_info": vendor_info,
        "known_aliases": aliases,
        "top_cobidders": partner_records,
    }


@app.get("/graph/network")
def get_network_graph(min_cobids: int = 5):
    """Returns nodes and edges formatted for visual network graphs."""
    _, _, df_vendors, df_cobids = load_data()
    if df_cobids.empty:
        return {"nodes": [], "edges": []}

    filtered_edges = df_cobids[df_cobids["co_bidding_count"] >= min_cobids]
    nodes_set = set(filtered_edges["Vendor_A"]).union(set(filtered_edges["Vendor_B"]))

    nodes = [{"id": n, "label": n} for n in nodes_set]
    edges = [
        {"from": r["Vendor_A"], "to": r["Vendor_B"], "weight": int(r["co_bidding_count"])}
        for _, r in filtered_edges.iterrows()
    ]

    return {"nodes": nodes, "edges": edges}
