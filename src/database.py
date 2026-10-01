"""
Database Storage and Persistence Module.
Compliant with Stage 18 of the Project Blueprint.
Supports SQLite (zero-setup default) and PostgreSQL.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/procurement.db")


class ProcurementDatabase:
    """Manages relational persistence of tenders, vendors, bids, features, and case queue."""

    def __init__(self, db_url: str = DEFAULT_DB_URL):
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)

    def init_db(self, data_dir: str = "data/processed"):
        """Loads processed CSV artifacts into relational tables."""
        tenders_path = os.path.join(data_dir, "clean_tenders.csv")
        vendors_path = os.path.join(data_dir, "clean_vendors.csv")
        bids_path = os.path.join(data_dir, "clean_bids.csv")
        queue_path = os.path.join(data_dir, "investigation_queue.csv")

        print(f"--> Initializing database at: {self.db_url}")

        if os.path.exists(tenders_path):
            df_t = pd.read_csv(tenders_path)
            df_t.to_sql("tenders", self.engine, if_exists="replace", index=False)
            print(f"    Loaded {len(df_t)} records into 'tenders' table.")

        if os.path.exists(vendors_path):
            df_v = pd.read_csv(vendors_path)
            df_v.to_sql("vendors", self.engine, if_exists="replace", index=False)
            print(f"    Loaded {len(df_v)} records into 'vendors' table.")

        if os.path.exists(bids_path):
            df_b = pd.read_csv(bids_path)
            df_b.to_sql("bids", self.engine, if_exists="replace", index=False)
            print(f"    Loaded {len(df_b)} records into 'bids' table.")

        if os.path.exists(queue_path):
            df_q = pd.read_csv(queue_path)
            df_q.to_sql("investigation_cases", self.engine, if_exists="replace", index=False)
            print(f"    Loaded {len(df_q)} records into 'investigation_cases' table.")

        print("--> Database initialization complete.")

    def query(self, sql_query: str) -> pd.DataFrame:
        """Executes a SQL query and returns results as a DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn)


if __name__ == "__main__":
    db = ProcurementDatabase()
    db.init_db()
