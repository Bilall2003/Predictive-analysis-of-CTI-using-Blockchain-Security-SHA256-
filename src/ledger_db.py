
import re
import pandas as pd
from sqlalchemy import MetaData, Table, Column, BigInteger, String, Text, Float, DateTime, func, select

try:
    from create_engine import ENGINE          # MySQL connection (your file with the credentials)
    ENGINE_ERROR = None
except Exception as e:
    ENGINE = None
    ENGINE_ERROR = str(e)


def build_table(table_name):
    """Describe the table (its columns). The table name changes for every run."""
    return Table(
        table_name, MetaData(),
        Column("log_index",            BigInteger, primary_key=True, autoincrement=False),
        Column("source_ip",            String(64)),
        Column("destination_ip",       String(64)),
        Column("dst_port",             String(16)),
        Column("true_label",           String(100)),
        Column("xgboost_prediction",   String(100)),
        Column("dae_alert",            String(32)),
        Column("reconstruction_error", Float(53)),
        Column("hybrid_decision",      String(48)),
        Column("nlp_summary",          Text),
        Column("hash_sha256",          String(64)),
        Column("logged_at",            DateTime, server_default=func.now()),
    )


def clean_text(series):
    """Remove emoji icons and extra spaces so MySQL stores plain text."""
    return series.astype(str).str.replace(r"[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f\u200d]", "", regex=True).str.strip()


def to_records(df):
    """Rename the on-screen ledger columns to the table columns."""
    out = pd.DataFrame({
        "log_index":            pd.to_numeric(df["Log Index"]).astype("int64"),
        "source_ip":            clean_text(df["Source IP"]),
        "destination_ip":       clean_text(df["Destination IP"]),
        "dst_port":             clean_text(df["Dst Port"]),
        "true_label":           clean_text(df["True Label"]),
        "xgboost_prediction":   clean_text(df["XGBoost Prediction"]),
        "dae_alert":            clean_text(df["DAE Alert"]),
        "reconstruction_error": pd.to_numeric(df["Reconstruction Error"], errors="coerce"),
        "hybrid_decision":      clean_text(df["Hybrid Decision"]),
        "nlp_summary":          clean_text(df["NLP Summary"]),
        "hash_sha256":          clean_text(df["Hash"]),
    })
    return out.astype(object).where(out.notna(), None).to_dict("records")     # NaN -> NULL


def save_ledger_rows(df, table_name):
    """Create the table if needed, then insert the rows. Returns (ok, message, rows_saved)."""
    if ENGINE is None:
        return False, f"create_engine.py could not be imported: {ENGINE_ERROR}", 0
    if len(df) == 0:
        return True, "", 0
    try:
        table = build_table(table_name)
        table.create(ENGINE, checkfirst=True)          # CREATE TABLE IF NOT EXISTS
        with ENGINE.begin() as conn:                   # one transaction = the whole batch or nothing
            conn.execute(table.insert(), to_records(df))
        return True, "", len(df)
    except Exception as e:
        return False, f"Database write failed: {str(e)[:300]}", 0


def count_rows(table_name):
    """How many rows are stored in this run's table (None if it cannot be read)."""
    try:
        table = build_table(table_name)
        with ENGINE.connect() as conn:
            return conn.execute(select(func.count()).select_from(table)).scalar()
    except Exception:
        return None