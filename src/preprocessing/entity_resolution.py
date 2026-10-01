"""
Entity Resolution Module for Public Procurement Vendors.
Compliant with Stage 7 of the Project Blueprint.
"""

from typing import Dict, List, Tuple
import pandas as pd
from rapidfuzz import fuzz
import networkx as nx


class EntityResolver:
    """
    Identifies and resolves duplicate or alias vendor entities using fuzzy string matching
    on company names and registered addresses.
    """

    def __init__(self, name_threshold: float = 85.0, combined_threshold: float = 75.0):
        self.name_threshold = name_threshold
        self.combined_threshold = combined_threshold

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalizes company suffix variations for accurate fuzzy matching."""
        n = str(name).lower().strip()
        replacements = {
            "private limited": "pvt ltd",
            "limited": "ltd",
            "infrastructure": "infra",
            "corporation": "corp",
            "company": "co",
            "associates": "assoc",
            "solutions": "sol",
            "enterprises": "ent",
        }
        for k, v in replacements.items():
            n = n.replace(k, v)
        # remove punctuation
        return "".join([c for c in n if c.isalnum() or c.isspace()])

    def find_matches(self, df_vendors: pd.DataFrame) -> List[Dict]:
        """
        Computes pairwise similarity scores across vendors to discover aliases/duplicates.
        """
        matches = []
        n = len(df_vendors)
        records = df_vendors.to_dict(orient="records")

        for i in range(n):
            v1 = records[i]
            n1 = self._normalize_name(v1["Vendor_Name"])
            addr1 = str(v1.get("Address", "")).lower()

            for j in range(i + 1, n):
                v2 = records[j]
                n2 = self._normalize_name(v2["Vendor_Name"])
                addr2 = str(v2.get("Address", "")).lower()

                name_sim = fuzz.token_sort_ratio(n1, n2)
                addr_sim = fuzz.token_set_ratio(addr1, addr2) if (addr1 and addr2) else 0.0

                # High name similarity OR moderate name similarity with shared address
                is_match = (
                    name_sim >= self.name_threshold
                    or (name_sim >= self.combined_threshold and addr_sim >= 85.0)
                )

                if is_match:
                    matches.append({
                        "Vendor_ID_1": v1["Vendor_ID"],
                        "Vendor_Name_1": v1["Vendor_Name"],
                        "Vendor_ID_2": v2["Vendor_ID"],
                        "Vendor_Name_2": v2["Vendor_Name"],
                        "Name_Similarity": name_sim,
                        "Address_Similarity": addr_sim,
                    })

        return matches

    def resolve_entities(self, df_vendors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clusters matched vendors into canonical entities using graph connected components.
        Returns:
            1. df_resolved_vendors: vendors DataFrame with Canonical_Vendor_ID and Is_Alias flag
            2. df_matches: pairwise match audit log
        """
        matches = self.find_matches(df_vendors)
        df_matches = pd.DataFrame(matches)

        # Graph of matched entities
        G = nx.Graph()
        for v_id in df_vendors["Vendor_ID"]:
            G.add_node(v_id)

        for m in matches:
            G.add_edge(m["Vendor_ID_1"], m["Vendor_ID_2"], weight=m["Name_Similarity"])

        # Find connected components (each component is one real-world entity)
        vendor_to_canonical = {}
        for component in nx.connected_components(G):
            # Canonical ID is the minimum/first Vendor_ID in the cluster
            canonical_id = sorted(list(component))[0]
            for v_id in component:
                vendor_to_canonical[v_id] = {
                    "Canonical_Vendor_ID": canonical_id,
                    "Cluster_Size": len(component),
                    "Is_Alias": 1 if v_id != canonical_id else 0,
                }

        df_resolved = df_vendors.copy()
        df_resolved["Canonical_Vendor_ID"] = df_resolved["Vendor_ID"].map(
            lambda x: vendor_to_canonical.get(x, {}).get("Canonical_Vendor_ID", x)
        )
        df_resolved["Is_Alias"] = df_resolved["Vendor_ID"].map(
            lambda x: vendor_to_canonical.get(x, {}).get("Is_Alias", 0)
        )
        df_resolved["Cluster_Size"] = df_resolved["Vendor_ID"].map(
            lambda x: vendor_to_canonical.get(x, {}).get("Cluster_Size", 1)
        )

        return df_resolved, df_matches

    @staticmethod
    def map_bids_to_canonical(df_bids: pd.DataFrame, df_resolved_vendors: pd.DataFrame) -> pd.DataFrame:
        """Maps bids from raw Vendor_ID to Canonical_Vendor_ID."""
        mapping = dict(zip(df_resolved_vendors["Vendor_ID"], df_resolved_vendors["Canonical_Vendor_ID"]))
        df_mapped = df_bids.copy()
        df_mapped["Canonical_Vendor_ID"] = df_mapped["Vendor_ID"].map(mapping).fillna(df_mapped["Vendor_ID"])
        return df_mapped
