"""
Synthetic Public Procurement Data Generator with Collusion Injection.
Compliant with Stages 2, 3, 4, 5, and 9 of the Project Blueprint.
"""

import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


DEPARTMENTS = [
    "Public Works Department",
    "Department of Health & Family Welfare",
    "Municipal Corporation",
    "Department of Education & Literacy",
    "Water Supply & Sanitation Board",
    "Transport Infrastructure Authority",
    "Information Technology Services",
]

CATEGORIES = [
    "Road Construction & Repair",
    "Medical Equipment Supply",
    "IT Hardware & Networking",
    "School Infrastructure & Furniture",
    "Water Pipeline Maintenance",
    "Waste Management & Sanitation",
    "Civil Structural Engineering",
]

CITIES = [
    "Mumbai",
    "Pune",
    "Nagpur",
    "Nashik",
    "Thane",
    "Aurangabad",
    "Solapur",
    "Amravati",
]

STREETS = [
    "MG Road, Industrial Area",
    "MIDC Phase II",
    "Station Road, Commercial Complex",
    "Linking Road, Tech Park",
    "Ring Road, Sector 4",
    "Apex Business District",
    "SV Road, Logistics Center",
]

COMPANY_SUFFIXES = ["Pvt Ltd", "Limited", "Enterprises", "Infra & Co", "Solutions", "Associates"]

NAME_SEEDS = [
    "Apex", "Prime", "Universal", "Sterling", "Bharat", "Navkar", "Pinnacle",
    "Vanguard", "Delta", "Regal", "Zenith", "Sunrise", "Royal", "Falcon",
    "Everest", "Matrix", "Kestrel", "Horizon", "Paramount", "Titan"
]


class ProcurementDataGenerator:
    """
    Generates realistic public procurement records (Tenders, Vendors, Bids)
    with controlled synthetic collusion rings (Cover Bidding, Bid Rotation,
    Market Allocation, and Entity Duplication).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_vendors(self, num_vendors: int = 80) -> pd.DataFrame:
        """Generates a realistic pool of commercial vendors."""
        vendors = []
        used_names = set()

        for i in range(1, num_vendors + 1):
            vendor_id = f"VND-{i:04d}"
            base_name = f"{random.choice(NAME_SEEDS)} {random.choice(NAME_SEEDS)}"
            suffix = random.choice(COMPANY_SUFFIXES)
            vendor_name = f"{base_name} {suffix}"

            # Ensure unique names in baseline pool
            while vendor_name in used_names:
                base_name = f"{random.choice(NAME_SEEDS)} {random.choice(NAME_SEEDS)}"
                vendor_name = f"{base_name} {suffix}"
            used_names.add(vendor_name)

            city = random.choice(CITIES)
            address = f"{random.randint(101, 999)}, {random.choice(STREETS)}, {city}"
            reg_id = f"REG-{random.randint(100000, 999999)}-{random.choice(['MH', 'DL', 'KA'])}"
            industry = random.choice(CATEGORIES)

            vendors.append({
                "Vendor_ID": vendor_id,
                "Vendor_Name": vendor_name,
                "Address": address,
                "City": city,
                "Registration_ID": reg_id,
                "Industry": industry,
            })

        return pd.DataFrame(vendors)

    def inject_entity_aliases(self, df_vendors: pd.DataFrame, num_alias_pairs: int = 4) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Injects alias/shell entities with slight typographical variations or shared addresses
        to test Entity Resolution (Stage 7).
        """
        alias_log = []
        new_records = []
        vendor_ids = df_vendors["Vendor_ID"].tolist()

        for i in range(num_alias_pairs):
            parent_idx = i * 2
            parent_vendor = df_vendors.iloc[parent_idx]
            new_id = f"VND-{len(df_vendors) + i + 1:04d}"

            # Create alias name variations
            orig_name = parent_vendor["Vendor_Name"]
            if "Pvt Ltd" in orig_name:
                alias_name = orig_name.replace("Pvt Ltd", "Private Limited")
            elif "Limited" in orig_name:
                alias_name = orig_name.replace("Limited", "Ltd.")
            elif "Infra & Co" in orig_name:
                alias_name = orig_name.replace("Infra & Co", "Infrastructure Company")
            else:
                alias_name = f"{orig_name} Group"

            # 50% chance of identical address (shared shell office)
            same_address = (i % 2 == 0)
            address = parent_vendor["Address"] if same_address else f"{random.randint(10, 99)}, {parent_vendor['Address']}"

            alias_record = {
                "Vendor_ID": new_id,
                "Vendor_Name": alias_name,
                "Address": address,
                "City": parent_vendor["City"],
                "Registration_ID": f"{parent_vendor['Registration_ID'][:10]}A",
                "Industry": parent_vendor["Industry"],
            }
            new_records.append(alias_record)
            alias_log.append({
                "Parent_Vendor_ID": parent_vendor["Vendor_ID"],
                "Alias_Vendor_ID": new_id,
                "Parent_Name": orig_name,
                "Alias_Name": alias_name,
                "Shared_Address": same_address
            })

        df_updated = pd.concat([df_vendors, pd.DataFrame(new_records)], ignore_index=True)
        return df_updated, alias_log

    def generate_tenders(
        self,
        num_tenders: int = 300,
        start_date: datetime = datetime(2025, 1, 1),
    ) -> pd.DataFrame:
        """Generates realistic procurement tenders over a time horizon."""
        tenders = []

        for i in range(1, num_tenders + 1):
            tender_id = f"TND-{i:05d}"
            department = random.choice(DEPARTMENTS)
            category = random.choice(CATEGORIES)
            location = random.choice(CITIES)

            # Realistic budget distribution: between 500,000 and 50,000,000 INR
            estimated_value = round(float(np.random.lognormal(mean=14.5, sigma=0.8)), -3)
            estimated_value = max(300_000.0, min(80_000_000.0, estimated_value))

            tender_offset = random.randint(0, 365)
            tender_date = start_date + timedelta(days=tender_offset)
            deadline = tender_date + timedelta(days=random.randint(14, 45))

            tenders.append({
                "Tender_ID": tender_id,
                "Tender_Date": tender_date.strftime("%Y-%m-%d"),
                "Department": department,
                "Category": category,
                "Location": location,
                "Estimated_Value": estimated_value,
                "Deadline": deadline.strftime("%Y-%m-%d"),
            })

        df_tenders = pd.DataFrame(tenders)
        df_tenders = df_tenders.sort_values("Tender_Date").reset_index(drop=True)
        return df_tenders

    def generate_bids_and_collusion(
        self,
        df_tenders: pd.DataFrame,
        df_vendors: pd.DataFrame,
        collusion_rate: float = 0.20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates realistic bids for each tender and injects controlled collusion rings:
        1. Cover Bidding (Ring A)
        2. Bid Rotation (Ring B)
        3. Market/Geographic Allocation (Ring C)
        """
        all_bids = []
        ground_truth = []

        vendor_ids = df_vendors["Vendor_ID"].tolist()

        # Define 3 distinct Collusion Rings
        ring_cover = vendor_ids[2:6]        # 4 vendors (VND-0003 to VND-0006)
        ring_rotation = vendor_ids[10:14]    # 4 vendors (VND-0011 to VND-0014)
        ring_market = vendor_ids[20:23]      # 3 vendors (VND-0021 to VND-0023)

        rotation_counter = 0

        num_tenders = len(df_tenders)
        collusion_target_count = int(num_tenders * collusion_rate)

        # Randomly choose which tenders will be collusive
        collusive_indices = set(random.sample(range(num_tenders), collusion_target_count))

        for idx, tender in df_tenders.iterrows():
            tender_id = tender["Tender_ID"]
            est_value = tender["Estimated_Value"]
            category = tender["Category"]
            location = tender["Location"]

            is_collusion_tender = idx in collusive_indices
            tender_bids = []

            if is_collusion_tender:
                collusion_type = random.choice(["Cover Bidding", "Bid Rotation", "Market Allocation"])

                if collusion_type == "Cover Bidding":
                    # Designated winner bids just below estimate (e.g. 96-98%)
                    # Cartel members submit inflated cover bids (115% - 130%)
                    designated_winner = ring_cover[0]
                    cover_bidders = ring_cover[1:]

                    winner_bid = round(est_value * np.random.uniform(0.96, 0.985), 2)
                    tender_bids.append({
                        "Tender_ID": tender_id,
                        "Vendor_ID": designated_winner,
                        "Bid_Amount": winner_bid,
                        "Bid_Status": "Submitted",
                    })

                    for cb in cover_bidders:
                        cover_bid = round(est_value * np.random.uniform(1.15, 1.30), 2)
                        tender_bids.append({
                            "Tender_ID": tender_id,
                            "Vendor_ID": cb,
                            "Bid_Amount": cover_bid,
                            "Bid_Status": "Submitted",
                        })

                    # Add 1-2 innocent competitors bidding high or just participating
                    remaining_pool = [v for v in vendor_ids if v not in ring_cover]
                    extra_vendors = random.sample(remaining_pool, random.randint(1, 2))
                    for ev in extra_vendors:
                        extra_bid = round(est_value * np.random.uniform(1.02, 1.25), 2)
                        tender_bids.append({
                            "Tender_ID": tender_id,
                            "Vendor_ID": ev,
                            "Bid_Amount": extra_bid,
                            "Bid_Status": "Submitted",
                        })

                    ground_truth.append({
                        "Tender_ID": tender_id,
                        "Is_Collusion": 1,
                        "Collusion_Type": "Cover Bidding",
                        "Ring_ID": "RING_COVER_A",
                        "Involved_Vendors": ",".join(ring_cover),
                        "Winning_Vendor": designated_winner,
                        "Description": "Designated winner bidding ~97% with cartel members submitting non-competitive cover bids >115%.",
                    })

                elif collusion_type == "Bid Rotation":
                    # Cartel members rotate who wins sequentially
                    winning_idx = rotation_counter % len(ring_rotation)
                    rotation_counter += 1
                    designated_winner = ring_rotation[winning_idx]
                    other_rotators = [v for v in ring_rotation if v != designated_winner]

                    winner_bid = round(est_value * np.random.uniform(0.92, 0.95), 2)
                    tender_bids.append({
                        "Tender_ID": tender_id,
                        "Vendor_ID": designated_winner,
                        "Bid_Amount": winner_bid,
                        "Bid_Status": "Submitted",
                    })

                    for orb in other_rotators:
                        rotating_cover_bid = round(est_value * np.random.uniform(1.08, 1.22), 2)
                        tender_bids.append({
                            "Tender_ID": tender_id,
                            "Vendor_ID": orb,
                            "Bid_Amount": rotating_cover_bid,
                            "Bid_Status": "Submitted",
                        })

                    ground_truth.append({
                        "Tender_ID": tender_id,
                        "Is_Collusion": 1,
                        "Collusion_Type": "Bid Rotation",
                        "Ring_ID": "RING_ROTATION_B",
                        "Involved_Vendors": ",".join(ring_rotation),
                        "Winning_Vendor": designated_winner,
                        "Description": f"Pre-arranged cyclical rotation; vendor {designated_winner} wins this round.",
                    })

                else:  # Market Allocation
                    # In market allocation, only the designated regional cartel member bids competitively
                    designated_winner = ring_market[idx % len(ring_market)]
                    winner_bid = round(est_value * np.random.uniform(0.95, 0.99), 2)
                    tender_bids.append({
                        "Tender_ID": tender_id,
                        "Vendor_ID": designated_winner,
                        "Bid_Amount": winner_bid,
                        "Bid_Status": "Submitted",
                    })

                    ground_truth.append({
                        "Tender_ID": tender_id,
                        "Is_Collusion": 1,
                        "Collusion_Type": "Market Allocation",
                        "Ring_ID": "RING_MARKET_C",
                        "Involved_Vendors": designated_winner,
                        "Winning_Vendor": designated_winner,
                        "Description": f"Sole uncontested bid by cartel member {designated_winner} allocated to {location}.",
                    })

            else:
                # Genuine, competitive bidding:
                # 3 to 7 independent vendors submit bids around market rate (88% to 110% of estimate)
                num_bidders = random.randint(3, 7)
                selected_vendors = random.sample(vendor_ids, num_bidders)

                for v in selected_vendors:
                    # Normal distribution around 98% with 5% standard deviation
                    ratio = np.clip(np.random.normal(0.98, 0.05), 0.85, 1.15)
                    bid_amount = round(est_value * ratio, 2)
                    tender_bids.append({
                        "Tender_ID": tender_id,
                        "Vendor_ID": v,
                        "Bid_Amount": bid_amount,
                        "Bid_Status": "Submitted",
                    })

                ground_truth.append({
                    "Tender_ID": tender_id,
                    "Is_Collusion": 0,
                    "Collusion_Type": "None",
                    "Ring_ID": "NONE",
                    "Involved_Vendors": "",
                    "Winning_Vendor": "",
                    "Description": "Normal competitive bidding.",
                })

            # Determine rank and winner (lowest submitted bid = rank 1 = Winner)
            tender_bids_sorted = sorted(tender_bids, key=lambda x: x["Bid_Amount"])
            for rank_idx, b in enumerate(tender_bids_sorted, start=1):
                b["Rank"] = rank_idx
                b["Winner"] = 1 if rank_idx == 1 else 0

            # Record winning vendor in ground truth for competitive tenders as well
            if not is_collusion_tender:
                ground_truth[-1]["Winning_Vendor"] = tender_bids_sorted[0]["Vendor_ID"]

            all_bids.extend(tender_bids_sorted)

        df_bids = pd.DataFrame(all_bids)
        df_ground_truth = pd.DataFrame(ground_truth)

        return df_bids, df_ground_truth


def generate_and_save_dataset(
    output_dir: str = "data",
    num_tenders: int = 400,
    num_vendors: int = 100,
    collusion_rate: float = 0.20,
    seed: int = 42,
) -> Dict[str, str]:
    """Generates the full capstone procurement benchmark and persists to disk."""
    generator = ProcurementDataGenerator(seed=seed)

    print("1. Generating vendors...")
    df_vendors = generator.generate_vendors(num_vendors=num_vendors)

    print("2. Injecting entity resolution variations...")
    df_vendors, alias_log = generator.inject_entity_aliases(df_vendors)

    print("3. Generating tenders...")
    df_tenders = generator.generate_tenders(num_tenders=num_tenders)

    print("4. Generating bids and injecting collusion patterns...")
    df_bids, df_ground_truth = generator.generate_bids_and_collusion(
        df_tenders=df_tenders,
        df_vendors=df_vendors,
        collusion_rate=collusion_rate,
    )

    # Save to data directories
    raw_dir = os.path.join(output_dir, "raw")
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    tenders_path = os.path.join(raw_dir, "tenders.csv")
    vendors_path = os.path.join(raw_dir, "vendors.csv")
    bids_path = os.path.join(raw_dir, "bids.csv")
    gt_path = os.path.join(gt_dir, "ground_truth_collusion.csv")
    aliases_path = os.path.join(gt_dir, "ground_truth_aliases.csv")

    df_tenders.to_csv(tenders_path, index=False)
    df_vendors.to_csv(vendors_path, index=False)
    df_bids.to_csv(bids_path, index=False)
    df_ground_truth.to_csv(gt_path, index=False)
    pd.DataFrame(alias_log).to_csv(aliases_path, index=False)

    print(f"Generated {len(df_tenders)} tenders, {len(df_vendors)} vendors, and {len(df_bids)} bids.")
    print(f"Collusive tenders: {df_ground_truth['Is_Collusion'].sum()} ({(df_ground_truth['Is_Collusion'].mean()*100):.1f}%)")

    return {
        "tenders": tenders_path,
        "vendors": vendors_path,
        "bids": bids_path,
        "ground_truth": gt_path,
        "aliases": aliases_path,
    }


if __name__ == "__main__":
    generate_and_save_dataset()
