"""
Unit tests for Database Persistence (Stage 18).
"""

from src.database import ProcurementDatabase


def test_database_init_and_query():
    # Use in-memory SQLite database for zero-lock testing
    db = ProcurementDatabase(db_url="sqlite:///:memory:")
    db.init_db(data_dir="data/processed")

    df_cases = db.query("SELECT * FROM investigation_cases LIMIT 5")
    assert len(df_cases) == 5
    assert "Tender_ID" in df_cases.columns
    assert "ml_anomaly_prob" in df_cases.columns

    db.engine.dispose()
