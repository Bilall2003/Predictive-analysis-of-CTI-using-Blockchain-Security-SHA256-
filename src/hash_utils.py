
import hashlib
import pandas as pd


def generate_sha256(data_string: str) -> str:
    """Return the SHA-256 hex digest of a string ('' if the input is empty/NaN)."""
    if not data_string or pd.isna(data_string):
        return ""
    return hashlib.sha256(str(data_string).encode('utf-8')).hexdigest()


def add_hash_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Hash' column: a SHA-256 signature for threat rows, '' for benign rows."""
    if df.empty:
        return df

    df_copy = df.copy()

    def row_hasher(row):
        decision = str(row.get('Hybrid Decision', '')).upper()
        # Hash everything that is NOT plain benign (attacks, XGB-only threats, zero-days)
        if 'BENIGN' not in decision or 'ZERO-DAY' in decision:
            signature_raw = f"{row.get('Source IP')}-{row.get('Destination IP')}-{row.get('Dst Port')}-{row.get('True Label')}"
            return generate_sha256(signature_raw)
        return ""

    df_copy['Hash'] = df_copy.apply(row_hasher, axis=1)
    return df_copy


def get_hash_statistics(df: pd.DataFrame) -> dict:
    """Counts shown in the ledger header: hashed threats, benign rows, unique hashes, threat %."""
    if df.empty or 'Hash' not in df.columns:
        return {}

    hashed_rows = df[df['Hash'] != '']
    total_rows = len(df)
    total_hashed = len(hashed_rows)
    unique_hashes = hashed_rows['Hash'].nunique()

    return {
        'hashed_attacks': total_hashed,
        'benign_samples': total_rows - total_hashed,
        'unique_hashes': unique_hashes,
        'hash_percentage': round((total_hashed / total_rows * 100), 2) if total_rows > 0 else 0.0
    }
