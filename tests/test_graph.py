"""
Unit tests for NetworkX Bipartite Graph and Co-Bidding Analytics.
"""

import pandas as pd
from src.graph.bipartite_graph import ProcurementGraphBuilder


def test_bipartite_and_cobidding_graphs():
    builder = ProcurementGraphBuilder()

    df_bids = pd.DataFrame([
        {"Tender_ID": "T1", "Vendor_ID": "V1", "Canonical_Vendor_ID": "V1", "Bid_Amount": 100, "Winner": 1, "Rank": 1},
        {"Tender_ID": "T1", "Vendor_ID": "V2", "Canonical_Vendor_ID": "V2", "Bid_Amount": 120, "Winner": 0, "Rank": 2},
        {"Tender_ID": "T2", "Vendor_ID": "V1", "Canonical_Vendor_ID": "V1", "Bid_Amount": 200, "Winner": 1, "Rank": 1},
        {"Tender_ID": "T2", "Vendor_ID": "V2", "Canonical_Vendor_ID": "V2", "Bid_Amount": 220, "Winner": 0, "Rank": 2},
        {"Tender_ID": "T2", "Vendor_ID": "V3", "Canonical_Vendor_ID": "V3", "Bid_Amount": 250, "Winner": 0, "Rank": 3},
    ])

    G_bi = builder.build_bipartite_graph(df_bids)
    assert len(G_bi.nodes()) == 5  # 2 tenders + 3 vendors

    G_co = builder.build_vendor_cobidding_network(df_bids)
    assert len(G_co.nodes()) == 3  # 3 vendors
    assert G_co.has_edge("V1", "V2")
    assert G_co["V1"]["V2"]["weight"] == 2

    metrics = builder.compute_graph_metrics()
    assert len(metrics) == 3
    assert "graph_degree_centrality" in metrics.columns
    assert "graph_weighted_degree" in metrics.columns
