
import pandas as pd


def generate_threat_summary(log_row: pd.Series) -> str:
    """Return a one-sentence alert for a ledger row, based on its Hybrid Decision."""
    src_ip = log_row.get('Source IP', 'Unknown IP')
    dst_ip = log_row.get('Destination IP', 'Unknown IP')
    port = log_row.get('Dst Port', 'N/A')
    label = log_row.get('True Label', 'Unspecified')
    decision = log_row.get('Hybrid Decision', 'Analysis Pending')

    if 'ZERO-DAY' in str(decision).upper():
        return f"CRITICAL NLP ALERT: Potential Zero-Day exploit targeted from {src_ip} towards {dst_ip}:{port}. XGBoost misclassified, Autoencoder flagged high variance."
    elif 'ATTACK' in str(decision).upper() or 'THREAT' in str(decision).upper():
        return f"HIGH WARNING: {label} activity detected between {src_ip} and {dst_ip}:{port} matching known signatures."
    else:
        return f"INFO: Standard benign network communication observed from {src_ip}."
