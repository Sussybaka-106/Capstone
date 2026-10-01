"""
Graph Analytics and Network Modeling for Public Procurement Collusion.
Compliant with Stages 10 and 11 of the Project Blueprint.
"""

from typing import Dict, List, Tuple
import networkx as nx
import pandas as pd
import numpy as np


class ProcurementGraphBuilder:
    """
    Constructs and analyzes:
    1. Bipartite Tender–Vendor Graph (Tenders and Vendors as nodes).
    2. Projected Co-Bidding Vendor Network (Vendors as nodes, co-bidding frequency as edge weights).
    """

    def __init__(self):
        self.bipartite_graph = nx.Graph()
        self.vendor_network = nx.Graph()

    def build_bipartite_graph(self, df_bids: pd.DataFrame) -> nx.Graph:
        """
        Creates a bipartite graph where Tenders and Vendors are disjoint sets of nodes.
        Edges represent bids submitted, weighted by bid amount or win status.
        """
        G = nx.Graph()
        vendor_col = "Canonical_Vendor_ID" if "Canonical_Vendor_ID" in df_bids.columns else "Vendor_ID"

        for _, row in df_bids.iterrows():
            tender_node = f"TND_{row['Tender_ID']}"
            vendor_node = f"VND_{row[vendor_col]}"

            # Add nodes with bipartite attribute
            G.add_node(tender_node, bipartite=0, node_type="tender")
            G.add_node(vendor_node, bipartite=1, node_type="vendor")

            # Add bid edge
            G.add_edge(
                tender_node,
                vendor_node,
                bid_amount=float(row.get("Bid_Amount", 0.0)),
                is_winner=int(row.get("Winner", 0)),
                rank=int(row.get("Rank", 1)),
            )

        self.bipartite_graph = G
        return G

    def build_vendor_cobidding_network(
        self,
        df_bids: pd.DataFrame,
        min_cobid_threshold: int = 1
    ) -> nx.Graph:
        """
        Projects bipartite graph into a single-mode Vendor Co-Bidding Network.
        Nodes = Vendors.
        Edge Weight = Number of tenders co-bid together.
        """
        vendor_col = "Canonical_Vendor_ID" if "Canonical_Vendor_ID" in df_bids.columns else "Vendor_ID"
        tender_vendors = df_bids.groupby("Tender_ID")[vendor_col].unique().to_dict()

        co_bids = {}
        for tender_id, vendors in tender_vendors.items():
            v_list = sorted(list(set(vendors)))
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    pair = (v_list[i], v_list[j])
                    co_bids[pair] = co_bids.get(pair, 0) + 1

        G_vendors = nx.Graph()
        # Add all vendors as nodes
        unique_vendors = df_bids[vendor_col].unique()
        for v in unique_vendors:
            G_vendors.add_node(v, label=v)

        for (v1, v2), count in co_bids.items():
            if count >= min_cobid_threshold:
                G_vendors.add_edge(v1, v2, weight=count)

        self.vendor_network = G_vendors
        return G_vendors

    def compute_graph_metrics(self) -> pd.DataFrame:
        """
        Computes network centrality and structural metrics for each vendor:
        - Degree Centrality
        - Weighted Degree (total co-bids)
        - Betweenness Centrality
        - Clustering Coefficient
        - PageRank
        """
        G = self.vendor_network
        if len(G) == 0:
            return pd.DataFrame()

        deg_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight="weight")
        clustering = nx.clustering(G, weight="weight")
        pagerank = nx.pagerank(G, weight="weight")

        # Weighted degree (strength)
        weighted_degree = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}

        metrics = []
        for node in G.nodes():
            metrics.append({
                "Vendor_ID": node,
                "graph_degree_centrality": deg_centrality.get(node, 0.0),
                "graph_weighted_degree": weighted_degree.get(node, 0),
                "graph_betweenness": betweenness.get(node, 0.0),
                "graph_clustering_coeff": clustering.get(node, 0.0),
                "graph_pagerank": pagerank.get(node, 0.0),
            })

        return pd.DataFrame(metrics)

    def detect_collusion_communities(self, min_community_size: int = 3) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Uses greedy modularity maximization community detection to discover suspicious vendor clusters/cliques.
        Returns:
            1. df_communities: Mapping of Vendor_ID to Community_ID and cluster size.
            2. candidate_rings: List of vendor rings with abnormally high co-bidding density.
        """
        G = self.vendor_network
        if len(G) == 0:
            return pd.DataFrame(), []

        # Find communities using Louvain / modularity
        communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))

        comm_records = []
        candidate_rings = []

        for comm_id, members in enumerate(communities, start=1):
            m_list = list(members)
            subgraph = G.subgraph(m_list)
            density = nx.density(subgraph)

            # Abnormally high density and size >= 3 strongly suggests a closed bidding ring
            is_suspicious_ring = (len(m_list) >= min_community_size and density >= 0.70)
            if is_suspicious_ring:
                candidate_rings.append(m_list)

            for vendor in m_list:
                comm_records.append({
                    "Vendor_ID": vendor,
                    "Community_ID": f"COMM_{comm_id:03d}",
                    "Community_Size": len(m_list),
                    "Community_Density": round(density, 4),
                    "Is_Suspicious_Ring": 1 if is_suspicious_ring else 0,
                })

        return pd.DataFrame(comm_records), candidate_rings
