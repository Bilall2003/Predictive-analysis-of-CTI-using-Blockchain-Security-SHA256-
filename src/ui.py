import time
import warnings
import os
import uuid
import re
import datetime

# --- third-party ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

# --- project modules ---
# Each one is imported inside try/except so the dashboard still opens if one of them cannot
# be loaded (for example SHAP or the MySQL driver is not installed). The matching feature
# then shows the error instead of crashing the whole app.
try:
    import module_1 as xai          # SHAP explanations (Explainable AI tab)
    XAI_IMPORT_ERROR = None
except Exception as _xai_err:
    xai = None
    XAI_IMPORT_ERROR = str(_xai_err)

try:
    import hash_utils               # SHA-256 threat signatures
except Exception as _hash_err:
    hash_utils = None

try:
    import nlp_utils                # plain-English alert sentences
except Exception as _nlp_err:
    nlp_utils = None

try:
    import hash_validator           # hash format check (Hash Validator tab)
    VALIDATOR_IMPORT_ERROR = None
except Exception as _val_err:
    hash_validator = None
    VALIDATOR_IMPORT_ERROR = str(_val_err)

try:
    import ledger_db                # mirrors the ledger into MySQL through create_engine.py
    LEDGER_DB_IMPORT_ERROR = None
except Exception as _ldb_err:
    ledger_db = None
    LEDGER_DB_IMPORT_ERROR = str(_ldb_err)

warnings.filterwarnings('ignore')

# ==============================================================================
# PER-LOG PDF SUMMARY (used by the Result Insights ledger)
# ==============================================================================
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    PDF_IMPORT_ERROR = None
except Exception as _pdf_err:
    PDF_IMPORT_ERROR = str(_pdf_err)


def build_log_summary_pdf(row, threshold):
    """Return a half-page PDF (bytes) summarising ONE hybrid-ledger entry."""
    import io
    import datetime
    from xml.sax.saxutils import escape

    def clean(v):
        # Built-in PDF fonts cannot draw emoji, so keep Latin-1 characters only.
        s = "" if v is None else str(v)
        for a, b in (("\u2014", "-"), ("\u2013", "-"), ("\u2018", "'"), ("\u2019", "'"),
                     ("\u201c", '"'), ("\u201d", '"'), ("\u2026", "...")):
            s = s.replace(a, b)
        return "".join(ch for ch in s if ord(ch) < 256).strip()

    def esc(v):
        return escape(clean(v))

    log_idx  = clean(row.get('Log Index', ''))
    src_ip   = clean(row.get('Source IP', ''))
    dst_ip   = clean(row.get('Destination IP', ''))
    dst_port = clean(row.get('Dst Port', ''))
    true_lbl = clean(row.get('True Label', ''))
    xgb_pred = clean(row.get('XGBoost Prediction', ''))
    dae_alert = clean(row.get('DAE Alert', ''))
    decision = clean(row.get('Hybrid Decision', ''))
    nlp_text = clean(row.get('NLP Summary', '')) or "No NLP summary was generated for this entry."
    sha      = clean(row.get('Hash', ''))
    try:
        err = float(row.get('Reconstruction Error'))
    except Exception:
        err = None

    thr_txt = f"{float(threshold):.6f}"
    err_txt = f"{err:.6f}" if err is not None else "n/a"

    # Flow duration (CICIDS2017 stores it in microseconds)
    fd_txt = "not recorded for this entry"
    try:
        _fd = float(row.get('Flow Duration'))
        if _fd >= 0:
            fd_txt = f"{_fd:,.0f} \u00b5s ({_fd / 1e6:,.3f} s)"
        else:
            fd_txt = "not available (missing or negative value in the source capture)"
    except Exception:
        pass

    dec_up = decision.upper()
    if "ZERO-DAY" in dec_up:
        kind, accent = "zero", rl_colors.HexColor("#B91C1C")
    elif "XGB" in dec_up:
        kind, accent = "xgb", rl_colors.HexColor("#C2410C")
    elif "ATTACK" in dec_up:
        kind, accent = "attack", rl_colors.HexColor("#C2410C")
    else:
        kind, accent = "benign", rl_colors.HexColor("#15803D")

    if kind == "zero":
        reasoning = (f"XGBoost classified this flow as {esc(xgb_pred)}, so no known attack signature matched. "
                     f"The denoising autoencoder, trained only on benign traffic, nevertheless reconstructed it poorly "
                     f"(error {err_txt} against a threshold of {thr_txt}). The flow behaves unlike normal traffic "
                     f"without resembling any known attack class, which is the pattern expected of a zero-day or "
                     f"previously unseen attack.")
        actions = ["Isolate or closely monitor the source host while the flow is reviewed.",
                   "Inspect captures and related flows from the same source for the same behaviour.",
                   "Open the Explainable AI tab and generate the zero-day proof for this log index.",
                   "If confirmed malicious, add an indicator to the blocklist and retain this record as evidence."]
    elif kind == "attack":
        reasoning = (f"XGBoost classified this flow as {esc(xgb_pred)} and the autoencoder independently flagged it as "
                     f"anomalous (error {err_txt} above the threshold of {thr_txt}). Agreement between the supervised "
                     f"and unsupervised detectors gives this entry the highest confidence of the four hybrid outcomes.")
        actions = ["Block or rate-limit the source address at the perimeter pending review.",
                   "Check the destination service for signs of compromise or degradation.",
                   "Correlate with other flows from the same source in the ledger."]
    elif kind == "xgb":
        reasoning = (f"XGBoost classified this flow as {esc(xgb_pred)}, but its reconstruction error ({err_txt}) stayed "
                     f"below the anomaly threshold ({thr_txt}). The flow matches a known attack signature while looking "
                     f"statistically normal to the autoencoder, so it is reported on the strength of the supervised "
                     f"model alone.")
        actions = ["Verify the XGBoost class against related flows from the same source.",
                   "Review destination service logs and treat the flow as a threat unless disproved."]
    else:
        reasoning = (f"XGBoost classified this flow as {esc(xgb_pred)} and the autoencoder reconstructed it within normal "
                     f"limits (error {err_txt}, threshold {thr_txt}). Neither detector raised an alert.")
        actions = ["No action required. The entry is retained in the ledger for audit purposes."]

    truth_is_attack = true_lbl.upper() not in ("BENIGN", "0", "")
    flagged = kind != "benign"
    if truth_is_attack and flagged:
        truth_note = f"The dataset labels this flow {esc(true_lbl)}, so the alert is a correct detection."
    elif (not truth_is_attack) and flagged:
        truth_note = "The dataset labels this flow BENIGN, so this alert is a false-positive candidate."
    elif truth_is_attack and not flagged:
        truth_note = f"The dataset labels this flow {esc(true_lbl)}, but the hybrid system did not flag it (missed detection)."
    else:
        truth_note = "The dataset labels this flow BENIGN, so passing it as benign is correct."

    ss = getSampleStyleSheet()
    title = ParagraphStyle('t', parent=ss['Title'], fontName='Helvetica-Bold', fontSize=15, leading=18,
                           alignment=0, textColor=accent, spaceAfter=1)
    sub   = ParagraphStyle('s', parent=ss['Normal'], fontName='Helvetica', fontSize=8.5, leading=11,
                           textColor=rl_colors.HexColor("#4B5563"), spaceAfter=6)
    h     = ParagraphStyle('h', parent=ss['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13,
                           textColor=rl_colors.HexColor("#111827"), spaceBefore=7, spaceAfter=2)
    body  = ParagraphStyle('b', parent=ss['Normal'], fontName='Helvetica', fontSize=9, leading=12.2, alignment=4)
    cell  = ParagraphStyle('c', parent=ss['Normal'], fontName='Helvetica', fontSize=8.8, leading=11)
    cellb = ParagraphStyle('cb', parent=cell, fontName='Helvetica-Bold')
    bul   = ParagraphStyle('bl', parent=body, leftIndent=10, bulletIndent=1, spaceAfter=1, alignment=0)
    mono  = ParagraphStyle('m', parent=ss['Normal'], fontName='Courier', fontSize=8, leading=10.5,
                           textColor=rl_colors.HexColor("#1F2937"))
    foot  = ParagraphStyle('f', parent=ss['Normal'], fontName='Helvetica-Oblique', fontSize=7.5, leading=9.5,
                           textColor=rl_colors.HexColor("#6B7280"), spaceBefore=8)

    def L(t): return Paragraph(t, cellb)
    def V(t): return Paragraph(esc(t) if t != "" else "-", cell)

    grid = Table([
        [L("Log Index"),            V(log_idx),   L("Hybrid Decision"), V(decision)],
        [L("Source IP"),            V(src_ip),    L("Destination IP"),  V(dst_ip)],
        [L("Destination Port"),     V(dst_port),  L("True Label"),      V(true_lbl)],
        [L("XGBoost Prediction"),   V(xgb_pred),  L("DAE Alert"),       V(dae_alert)],
        [L("Reconstruction Error"), V(err_txt),   L("Anomaly Threshold"), V(thr_txt)],
    ], colWidths=[36*mm, 42*mm, 36*mm, 58*mm])
    grid.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#D1D5DB")),
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor("#F3F4F6")),
        ('BACKGROUND', (2, 0), (2, -1), rl_colors.HexColor("#F3F4F6")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))

    story = [
        Paragraph("Threat Log Summary", title),
        Paragraph(f"Hybrid NIDS detection ledger &nbsp;|&nbsp; Entry #{esc(log_idx)} &nbsp;|&nbsp; "
                  f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sub),
        grid,
        Paragraph("Threat Narrative", h),
        Paragraph(esc(nlp_text), body),
        Spacer(1, 2),
        Paragraph(f"<b>Flow duration:</b> {esc(fd_txt)}", body),
        Paragraph("Detection Reasoning", h),
        Paragraph(reasoning, body),
        Spacer(1, 2),
        Paragraph(truth_note, body),
        Paragraph("Recommended Analyst Action", h),
    ]
    for a in actions:
        story.append(Paragraph(esc(a), bul, bulletText="\u2022"))
    story.append(Paragraph("Record Integrity", h))
    if sha:
        story.append(Paragraph("SHA-256 hash recorded for this entry:", body))
        story.append(Paragraph(esc(sha), mono))
    else:
        story.append(Paragraph("No hash is recorded for this entry in the ledger.", body))
    story.append(Paragraph("Automatically generated by the Cyber Intrusion Detection Console from the Hybrid "
                           "Model detection ledger. Verify against the source logs before acting.", foot))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=15*mm, bottomMargin=15*mm,
                            title=f"Threat Log Summary - Entry {log_idx}", author="Cyber Intrusion Detection Console")
    doc.build(story)
    return buf.getvalue()


def _on_ledger_download_click():
    """Runs when a Download button inside the lower ledger is clicked."""
    click = st.session_state.get("ledger_dl_click")
    if click is not None:
        st.session_state["pdf_last"] = int(click["row"])      # row position in hybrid_logs
        st.session_state["pdf_autodl"] = int(click["row"])    # start the download once


def render_lower_ledger(ledger_df, cols, height=200):
    """Lower ledger (Log Index | NLP Summary | Hash | Download).
    The Download cell of every row is a button that downloads that log's half-page PDF."""
    import base64
    import streamlit.components.v1 as components

    pdf_ok = PDF_IMPORT_ERROR is None
    has_btn = hasattr(st.column_config, "ButtonColumn")
    thr = float(st.session_state.get('active_thresh', 0.616377))

    view = ledger_df[cols].copy()
    if 'Download' in view.columns:
        if not pdf_ok:
            view['Download'] = "pip install reportlab"
        elif has_btn:
            view['Download'] = "Download PDF"
        else:
            view['Download'] = "Tick row, then download below"

    pos = None
    if has_btn and pdf_ok:
        st.dataframe(
            view, use_container_width=True, hide_index=True, height=height,
            column_config={"Download": st.column_config.ButtonColumn(
                "Download", on_click=_on_ledger_download_click, key="ledger_dl_click")})
        pos = st.session_state.get("pdf_last")
    else:
        # Older Streamlit without in-cell buttons: select a row, then use the button below the table.
        try:
            ev = st.dataframe(view, use_container_width=True, hide_index=True, height=height,
                              on_select="rerun", selection_mode="single-row", key="ledger_lower_sel")
            rows = list(ev.selection.rows) if ev is not None else []
            pos = rows[0] if rows else None
            if not has_btn:
                st.info(f"Streamlit {st.__version__} cannot put buttons inside a table (needs 1.59 or newer). "
                        "Tick the box at the left of a row, then press the download button that appears "
                        "under the table. For buttons inside the table run: pip install -U streamlit")
        except TypeError:
            st.dataframe(view, use_container_width=True, hide_index=True, height=height)

    if not pdf_ok:
        st.warning(f"PDF export unavailable: {PDF_IMPORT_ERROR}. Run: pip install reportlab")
        return
    if pos is None or not (0 <= int(pos) < len(ledger_df)):
        return

    row = ledger_df.iloc[int(pos)]
    log_idx = row.get('Log Index', int(pos) + 1)
    pdf_bytes = build_log_summary_pdf(row, thr)
    fname = f"log_{log_idx}_summary.pdf"

    # Start the browser download right after the click (one time per click)
    if st.session_state.pop("pdf_autodl", None) is not None:
        b64 = base64.b64encode(pdf_bytes).decode()
        _dl_js = f"""
            <script>
            (function() {{
              var bin = atob("{b64}"), arr = new Uint8Array(bin.length);
              for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
              var url = URL.createObjectURL(new Blob([arr], {{type: "application/pdf"}}));
              var a = document.createElement("a"); a.href = url; a.download = "{fname}";
              try {{ var d = window.parent.document; d.body.appendChild(a); a.click(); d.body.removeChild(a); }}
              catch (e) {{ document.body.appendChild(a); a.click(); }}
              setTimeout(function() {{ URL.revokeObjectURL(url); }}, 10000);
            }})();
            </script>"""
        if hasattr(st, "iframe"):
            st.iframe(_dl_js, height=1)
        else:
            components.html(_dl_js, height=0)

    st.download_button(f"Log #{log_idx} PDF (use this if the download did not start)",
                       data=pdf_bytes, file_name=fname, mime="application/pdf", key="ledger_dl_fallback")

# PAGE SETUP AND STYLING
st.set_page_config(
    layout="wide",
    page_title="Cyber Intrusion Detection Console"
)

# Dark theme: colours, metric cards and the animated status banner (CSS)
st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1.5rem !important;
        }
        .stApp {
            background-color: #0b0f14 !important;
            color: #adbac7 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #539bf5 !important;
            font-family: monospace !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #768390 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            font-size: 0.75rem !important;
        }
        h1, h2, h3, h4 {
            color: #f0f6fc !important;
            font-family: monospace !important;
            font-weight: 600 !important;
            margin-bottom: 5px !important;
        }
        .stButton>button {
            width: 100%;
            background-color: #1c2128 !important;
            color: #adbac7 !important;
            border: 1px solid #444c56 !important;
            border-radius: 4px !important;
            font-family: monospace !important;
        }
        .stButton>button:hover {
            border-color: #539bf5 !important;
            color: #539bf5 !important;
            background-color: #22272e !important;
        }
        .status-container {
            background: #1c2128;
            border: 1px solid #444c56;
            border-left: 5px solid #539bf5;
            padding: 14px 20px;
            border-radius: 6px;
            color: #adbac7;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);
            margin-bottom: 20px;
        }
        .status-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .status-dot {
            position: relative;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .status-dot::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            border-radius: 50%;
            background: inherit;
        }
        .status-dot.idle {
            background: #3fb950;
            animation: status-pulse-soft 1.2s ease-in-out infinite;
        }
        .status-dot.active {
            background: #f5a623;
            animation: status-pulse-fast 1s ease-in-out infinite;
        }
        .status-dot.done {
            background: #3fb950;
            animation: status-pulse-fast 1s ease-in-out infinite;
        }
        @keyframes status-pulse-soft {
            0%   { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.55); }
            70%  { box-shadow: 0 0 0 7px rgba(63, 185, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
        }
        @keyframes status-pulse-fast {
            0%   { box-shadow: 0 0 0 0 rgba(245, 166, 35, 0.6); }
            70%  { box-shadow: 0 0 0 9px rgba(245, 166, 35, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 166, 35, 0); }
        }
        .status-loading-track {
            margin-top: 10px;
            height: 4px;
            width: 100%;
            border-radius: 2px;
            background: #0b0f14;
            overflow: hidden;
        }
        .status-loading-fill {
            height: 100%;
            width: 40%;
            border-radius: 2px;
            background: linear-gradient(90deg, transparent, #f5a623, transparent);
            background-size: 60% 100%;
            animation: status-shimmer 1.1s linear infinite;
        }
        @keyframes status-shimmer {
            0%   { transform: translateX(-120%); }
            100% { transform: translateX(340%); }
        }
    </style>
""", unsafe_allow_html=True)


# AUTOENCODER MODEL (layer sizes must match training exactly, see final_dae1 (2) (1).py)

# ==============================================================================
# K-MEANS SETTINGS
# K-Means groups flows by how similar their autoencoder latent codes are. We use a SMALL, fixed
# number of groups ("behaviour groups"), so similar attacks (for example the DoS family, or Bot
# and web attacks) can share a group. K-Means cannot be told which attacks belong together;
# we only choose how many groups there are, and the algorithm decides who is close to whom.
# ==============================================================================
KMEANS_CLUSTERS    = 6      # number of behaviour groups
RECLUSTER_INTERVAL = 5      # re-fit K-Means on all rows every 5th batch

# Helper function to name K-Means clusters with parent family or attack names without mentioning group indices
def get_cluster_display_name(g_id, cluster_ids, true_labels):
    if not cluster_ids or not true_labels:
        return f"Cluster {g_id}"
    df_cluster = pd.DataFrame({'cluster': cluster_ids, 'label': true_labels})
    labels_in_g = df_cluster[df_cluster['cluster'] == g_id]['label'].value_counts()
    if len(labels_in_g) == 0:
        return f"Cluster {g_id}"
    top_labels = labels_in_g.head(3).index.tolist()
    lower_labels = [str(l).lower() for l in top_labels]
    
    if all('dos' in l for l in lower_labels) or (any(l.startswith('dos') for l in lower_labels) and len(top_labels) <= 2):
        return "DoS Attacks"
    elif all('web attack' in l for l in lower_labels) or sum('web attack' in l for l in lower_labels) >= 2:
        return "Web Attacks"
    elif all('patator' in l for l in lower_labels) or sum('patator' in l for l in lower_labels) >= 2:
        return "Patator Attacks"
    else:
        clean_names = [str(l).strip() for l in top_labels]
        return '/'.join(clean_names)

# ==============================================================================
# DATABASE TABLE NAME: every run is saved in its own table
# ==============================================================================
def make_table_name(file_name):
    """e.g. Friday.csv -> hybrid_ledger_Friday_20261015_183700 (dataset name + date/time of the run)"""
    dataset = os.path.splitext(os.path.basename(str(file_name)))[0]
    dataset = re.sub(r'[^A-Za-z0-9]+', '_', dataset).strip('_')[:30] or "dataset"
    return f"hybrid_ledger_{dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

class DAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# MODEL AND DATA LOADERS (cached: loaded once, not on every Streamlit rerun)
@st.cache_resource
def load_xgb_pipeline_components():
    try:
        model = joblib.load("XGBoost_model(smote)_3rd_milestone.joblib")
        encoder = joblib.load("preprocessed_encoder.joblib")
        return model, encoder
    except Exception as e:
        st.error(f"XGBoost Loader Failure: {e}")
        return None, None


@st.cache_resource
def load_ae_pipeline_components():
    try:
        config = joblib.load("ae_config.pkl")
        scaler = joblib.load("ae_scaler.pkl")

        ae_target_features = config.get("feature_cols", [])
        input_dim = len(ae_target_features) if ae_target_features else 117

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pytorch_engine = DAE(input_dim=input_dim).to(device)

        if os.path.exists("ae_model.pth"):
            state_dict = torch.load("ae_model.pth", map_location=device)
        elif os.path.exists("best_dae_final.pth"):
            state_dict = torch.load("best_dae_final.pth", map_location=device)
        else:
            internal_path = os.path.join("best_dae (3).pth", "best_dae", "data.pkl")
            if os.path.exists(internal_path):
                state_dict = torch.load(internal_path, map_location=device)
            else:
                state_dict = torch.load("best_dae (3).pth", map_location=device)

        pytorch_engine.load_state_dict(state_dict)
        pytorch_engine.eval()

        for module in pytorch_engine.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.track_running_stats = False

        return config, scaler, pytorch_engine
    except Exception as e:
        st.error(f"Autoencoder Setup Failure: {e}")
        return None, None, None


@st.cache_data(show_spinner=False)
def clean_and_parse_input_stream(file_data):
    df = pd.read_csv(file_data, low_memory=False)
    
    df.columns = df.columns.str.strip()
    
    label_cols = [c for c in df.columns if 'label' in c.lower()]
    if label_cols:
        df['target_label'] = df[label_cols[0]].astype(str).str.strip()
        df['target_label'] = df['target_label'].str.replace('- Attempted', '', regex=False).str.strip()
    else:
        df['target_label'] = 'BENIGN'

    src_ip_col   = [c for c in df.columns if 'src ip' in c.lower() or 'source ip' in c.lower()]
    dst_ip_col   = [c for c in df.columns if 'dst ip' in c.lower() or 'destination ip' in c.lower()]
    dst_port_col = [c for c in df.columns if 'dst port' in c.lower() or 'destination port' in c.lower()]

    df['display_src_ip']   = df[src_ip_col[0]]   if src_ip_col   else '192.168.1.100'
    df['display_dst_ip']   = df[dst_ip_col[0]]   if dst_ip_col   else '10.0.0.5'
    df['display_dst_port'] = df[dst_port_col[0]] if dst_port_col else '443'

    eda_drop_targets = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Timestamp', 'Unnamed: 0'
    ]
    drop_list = []
    for col in df.columns:
        if any(target.lower() in col.lower() for target in eda_drop_targets):
            if 'display_' not in col and 'target_label' not in col:
                drop_list.append(col)
    df.drop(columns=drop_list, errors='ignore', inplace=True)

    meta_cols = {'target_label', 'display_src_ip', 'display_dst_ip', 'display_dst_port'}
    for col in df.columns:
        if col not in meta_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())

    # FIX: fill only the NUMBER columns with 0.0. The text columns (target_label and the
    # display_* columns) are left alone: newer pandas refuses to put 0.0 into a text column,
    # and the processing loop already gives them safe defaults ('0.0.0.0', '0', 'BENIGN').
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0.0)

    return df

# MAIN WORKSPACE: session state, SHAP tab, dashboard UI and the batch-processing loop
def _opt_kwargs(fn, **kw):
    """Keep only the keyword arguments that `fn` really accepts (works with older module_1.py files)."""
    import inspect
    try:
        accepted = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in kw.items() if k in accepted}


class ProductionNIDSWorkspace:

    def __init__(self):
        self.xgb_model, self.label_encoder = load_xgb_pipeline_components()
        self.ae_config, self.ae_scaler, self.pytorch_engine = load_ae_pipeline_components()

        if 'active_thresh' not in st.session_state:
            if self.ae_config and "threshold" in self.ae_config:
                st.session_state.active_thresh = float(self.ae_config["threshold"])
            else:
                st.session_state.active_thresh = 0.616377

        if 'stream_active'           not in st.session_state: st.session_state.stream_active           = False
        if 'current_index'           not in st.session_state: st.session_state.current_index           = 0
        if 'simulation_finished'     not in st.session_state: st.session_state.simulation_finished     = False

        if 'total_attacks_injected'   not in st.session_state: st.session_state.total_attacks_injected   = 0
        if 'correctly_predicted'      not in st.session_state: st.session_state.correctly_predicted      = 0
        if 'model_misclassifications' not in st.session_state: st.session_state.model_misclassifications = 0
        if 'completely_missed'        not in st.session_state: st.session_state.completely_missed        = 0
        if 'latest_inference_latency' not in st.session_state: st.session_state.latest_inference_latency = 0.0

        if 'ae_total_anomalies_detected' not in st.session_state: st.session_state.ae_total_anomalies_detected = 0
        if 'ae_correct_alerts'           not in st.session_state: st.session_state.ae_correct_alerts           = 0
        if 'ae_false_alarms'             not in st.session_state: st.session_state.ae_false_alarms             = 0
        if 'ae_missed_attacks'           not in st.session_state: st.session_state.ae_missed_attacks           = 0

        # K-Means clustering of the autoencoder's latent codes
        if 'latent_space_history' not in st.session_state: st.session_state.latent_space_history = []
        if 'cluster_id_history'   not in st.session_state: st.session_state.cluster_id_history   = []

        if 'kmeans_batch_counter' not in st.session_state: st.session_state.kmeans_batch_counter = 0
        if 'kmeans_model'         not in st.session_state: st.session_state.kmeans_model         = None   # last fitted K-Means

        if 'ae_true_labels_history' not in st.session_state: st.session_state.ae_true_labels_history = []
        if 'ae_pred_labels_history' not in st.session_state: st.session_state.ae_pred_labels_history = []
        if 'y_true_list'            not in st.session_state: st.session_state.y_true_list            = []
        if 'y_pred_list'            not in st.session_state: st.session_state.y_pred_list            = []

        if 'unique_labels_seen'              not in st.session_state: st.session_state.unique_labels_seen              = set()
        if 'benign_sse_scores'               not in st.session_state: st.session_state.benign_sse_scores               = []
        if 'attack_sse_scores'               not in st.session_state: st.session_state.attack_sse_scores               = []

        if 'flow_duration_history'         not in st.session_state: st.session_state.flow_duration_history         = []
        if 'fwd_packets_history'           not in st.session_state: st.session_state.fwd_packets_history           = []
        if 'anomaly_reconstruction_scores' not in st.session_state: st.session_state.anomaly_reconstruction_scores = []
        if 'threat_distribution'           not in st.session_state: st.session_state.threat_distribution          = {}

        # The detection ledger: one row per processed flow
        if 'hybrid_logs' not in st.session_state:
            st.session_state.hybrid_logs = pd.DataFrame(
                columns=['Log Index', 'Source IP', 'Destination IP', 'Dst Port',
                         'True Label', 'XGBoost Prediction', 'DAE Alert',
                         'Reconstruction Error', 'Hybrid Decision', 'NLP Summary', 'Hash', 'Download']
            )
        # Database: name of this run's table, and how many ledger rows are already saved in it
        if 'ledger_table'    not in st.session_state: st.session_state.ledger_table    = None
        if 'ledger_db_saved' not in st.session_state: st.session_state.ledger_db_saved = 0
        if 'ledger_db_error' not in st.session_state: st.session_state.ledger_db_error = ""

        if 'hybrid_true_list'  not in st.session_state: st.session_state.hybrid_true_list  = []
        if 'hybrid_pred_list'  not in st.session_state: st.session_state.hybrid_pred_list  = []

        if 'visible_logs' not in st.session_state:
            st.session_state.visible_logs = pd.DataFrame(
                columns=['Log Index', 'Source IP', 'Destination IP', 'Dst Port', 'True Label', 'XGBoost Prediction']
            )
        if 'ae_visible_logs' not in st.session_state:
            st.session_state.ae_visible_logs = pd.DataFrame(
                columns=['Log Index', 'Source IP', 'Destination IP', 'Dst Port', 'True Label', 'Reconstruction Error', 'DAE Guard Alert']
            )

    def _build_xai_frames(self, chunk):
        chunk = chunk.reset_index(drop=True)
        y_arr = chunk['target_label'].astype(str).str.strip().values

        meta = {'target_label', 'display_src_ip', 'display_dst_ip', 'display_dst_port'}

        xs = chunk.copy()
        xs.columns = xs.columns.str.strip().str.lower()
        xs = xs.drop(columns=[c for c in meta if c in xs.columns], errors='ignore').select_dtypes(include=[np.number])
        if hasattr(self.xgb_model, "feature_names_in_"):
            feats = list(self.xgb_model.feature_names_in_)
            for c in feats:
                if c not in xs.columns:
                    xs[c] = 0.0
            xs = xs[feats]
        X_xgb = xs.reset_index(drop=True)

        cols = self.ae_config.get("feature_cols", []) if self.ae_config else []
        if not cols:
            raise ValueError("ae_config has no 'feature_cols' — cannot align autoencoder features.")
        aligned = pd.DataFrame({c: (chunk[c] if c in chunk.columns else 0.0) for c in cols})
        scaled  = np.clip(self.ae_scaler.transform(aligned.values.astype(np.float32)), -10, 10)
        X_ae    = pd.DataFrame(scaled, columns=cols)
        return X_xgb, X_ae, y_arr

    def _prepare_xai_frames(self, dataset_workspace, start, end):
        # The chosen batch = dataset rows [start, end). Only already-processed rows are ever offered.
        chunk = dataset_workspace.iloc[start:end].reset_index(drop=True)
        X_xgb, X_ae, y_arr = self._build_xai_frames(chunk)

        benign_mask = pd.Series(y_arr).str.upper().eq('BENIGN').values
        X_bg = X_ae[benign_mask] if benign_mask.sum() >= 50 else X_ae
        return X_xgb, X_ae, X_bg, y_arr

    def render_xai_tab(self, dataset_workspace):
        st.markdown("## Explainable AI (SHAP)")

        if xai is None:
            st.error(f"Could not import xai_shap_module.py: {XAI_IMPORT_ERROR}")
            return
        if dataset_workspace is None:
            st.info("Upload a dataset from the sidebar first.")
            return
        if self.xgb_model is None or self.label_encoder is None or self.pytorch_engine is None or self.ae_scaler is None:
            st.error("Models failed to load — SHAP needs both XGBoost and the Autoencoder.")
            return

        busy = st.session_state.stream_active
        processed = int(min(st.session_state.current_index, len(dataset_workspace)))
        if busy:
            st.warning("Evaluation is running — SHAP buttons unlock once it is halted or finished.")
        elif processed == 0:
            st.info("Run the evaluation first — SHAP works on the rows processed so far.")
        locked = busy or processed == 0

        class_names = [str(c) for c in self.label_encoder.classes_]
        out_dir     = xai.OUTPUT_DIR

        c1, c2 = st.columns(2)
        with c1:
            xgb_rows = st.slider("Rows per XGBoost SHAP batch", 5000, 10000, step=10, key="xai_xgb_rows",
                                 help="Independent of the sidebar Batch Window Size. The processed rows are "
                                      "cut into batches of this size; SHAP explains the batch picked below.")
            # Batches are built from the rows processed so far (not the full upload)
            _n_batches = max(1, -(-processed // xgb_rows))
            # No fixed key: when the batch size or the processed rows change, the list of batches
            # changes and the picker simply starts again at Batch 1 (no stale value can be left behind).
            batch_no = st.selectbox(
                f"Batch ({_n_batches} in total, from the {processed:,} rows processed)",
                list(range(1, _n_batches + 1)),
                format_func=lambda b: f"Batch {b}: from row {(b-1)*xgb_rows} till row {max(min(b*xgb_rows, processed)-1, 0)}")
        with c2:
            ae_rows = st.slider("Rows explained for Autoencoder SHAP (slow!)", 10, 100, 30, step=10, key="xai_ae_rows",
                                help="Number of flows (not features) the autoencoder explanation is averaged over. "
                                     "The feature list is always the same; only the averaging changes.")

        b_start = (batch_no - 1) * xgb_rows
        b_end   = min(batch_no * xgb_rows, processed)

        # Frames are (re)built only when the evaluation is idle, for exactly the chosen batch
        frame_key = (b_start, b_end, len(dataset_workspace))
        if (not locked) and st.session_state.get("xai_frame_key") != frame_key:
            try:
                st.session_state.xai_frames    = self._prepare_xai_frames(dataset_workspace, b_start, b_end)
                st.session_state.xai_frame_key = frame_key
                for k in ("xai_xgb_explainer", "xai_ae_explainer", "xai_ae_bar_done", "xai_bee_done"):
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Could not prepare SHAP inputs: {e}")
                return
        X_xgb, X_ae, X_bg, y_arr = st.session_state.get("xai_frames", (None, None, None, None))
        batch_tag = f" — Batch {batch_no} (rows {b_start}–{max(b_end-1, 0)})"

        if y_arr is not None and st.session_state.get("xai_frame_key") == frame_key:
            _cnt = pd.Series(y_arr).value_counts()
            st.caption(f"Batch {batch_no} contains {len(y_arr):,} flows: "
                       + ", ".join(f"{k} {v:,}" for k, v in _cnt.items()))

        def xgb_explainer():
            if "xai_xgb_explainer" not in st.session_state:
                st.session_state.xai_xgb_explainer = xai.make_xgb_explainer(self.xgb_model)
            return st.session_state.xai_xgb_explainer

        def ae_explainer():
            # Built on demand, so no other plot has to be generated first
            if "xai_ae_explainer" not in st.session_state:
                st.session_state.xai_ae_explainer = xai.make_ae_explainer(self.pytorch_engine, X_bg)
            return st.session_state.xai_ae_explainer

        st.markdown("---")

        st.markdown("### Tier 1 — XGBoost Global Beeswarm")
        st.caption("How to read: pick a class. Every dot is one flow from the selected batch. A dot to the RIGHT means "
                   "that feature pushed the model toward this class, to the LEFT means away from it. Red = high feature "
                   "value, blue = low. Features at the top matter most overall.")
        if st.button("Generate Beeswarm Plots", disabled=locked, key="xai_btn_bee"):
            with st.spinner("Computing TreeSHAP for all classes..."):
                try:
                    _, _ = xai.explain_xgboost_global(self.xgb_model, X_xgb, class_names=class_names,
                                                      **_opt_kwargs(xai.explain_xgboost_global, title_suffix=batch_tag))
                    st.session_state.xai_bee_done = True
                except Exception as e:
                    st.error(f"Beeswarm failed: {e}")
        if st.session_state.get("xai_bee_done"):
            avail = [c for c in class_names if os.path.exists(f"{out_dir}/xgb_summary_beeswarm_{c}.png")]
            if avail:
                pick = st.selectbox("Class", avail, key="xai_bee_pick")
                st.image(f"{out_dir}/xgb_summary_beeswarm_{pick}.png")

        st.markdown("---")

        st.markdown("### Tier 1 — XGBoost Waterfall (Case Study)")
        n_total = max(processed, 1)
        st.caption(f"Pick any processed flow — rows 0 to {n_total-1:,}. How to read: the plot starts at the model's "
                   "average score E[f(x)] and each bar is one feature pushing this flow's score up (red) or down (blue), "
                   "ending at the final score f(x) for the predicted class. Bigger bar = bigger influence.")

        if st.session_state.get("xai_labels_n") != n_total:
            st.session_state.xai_labels   = dataset_workspace['target_label'].astype(str).str.strip().values[:n_total]
            st.session_state.xai_labels_n = n_total
        all_labels = st.session_state.xai_labels

        wf_idx = int(st.number_input(f"Row number (0 – {n_total-1})", min_value=0, max_value=n_total-1,
                                     value=0, step=1, key="xai_wf_idx",
                                     help="Indexing starts from 0 here and syncs with the ledger, which starts from 1"))
        st.caption(f"Selected dataset row **{wf_idx}** — true label: **{all_labels[wf_idx]}**")

        if st.button("Generate Waterfall", disabled=locked, key="xai_btn_wf"):
            with st.spinner("Explaining one flow..."):
                try:
                    X_row, _, y_row = self._build_xai_frames(dataset_workspace.iloc[[wf_idx]])
                    xai.explain_xgboost_case_study(self.xgb_model, xgb_explainer(), X_row, y_row,
                                                   sample_index=0, display_index=wf_idx,
                                                   class_names=class_names)
                    st.session_state.xai_wf_path = f"{out_dir}/xgb_waterfall_sample_{wf_idx}.png"
                except Exception as e:
                    st.error(f"Waterfall failed: {e}")
        if st.session_state.get("xai_wf_path") and os.path.exists(st.session_state.xai_wf_path):
            st.image(st.session_state.xai_wf_path)

        st.markdown("---")

        st.markdown("### Tier 2 — Autoencoder Global Feature Importance")
        st.caption("KernelExplainer explains the reconstruction error itself. This is slow — keep the row count low. "
                   "How to read: each bar shows how much, on average, that feature raises the autoencoder's error "
                   "(longer bar = the feature makes the flow look more unusual). Rows are picked at random from the "
                   "selected batch; press the button again after changing the batch or row count.")
        if st.button("Generate Autoencoder Bar Plot", disabled=locked, key="xai_btn_ae"):
            with st.spinner("Running KernelSHAP on reconstruction error..."):
                try:
                    ae_expl, _, _ = xai.explain_autoencoder_global(
                        self.pytorch_engine, X_bg, X_ae, sample_size=ae_rows,
                        **_opt_kwargs(xai.explain_autoencoder_global, title_suffix=batch_tag))
                    st.session_state.xai_ae_explainer = ae_expl
                    st.session_state.xai_ae_bar_done = True
                except Exception as e:
                    st.error(f"Autoencoder SHAP failed: {e}")
        if st.session_state.get("xai_ae_bar_done") and os.path.exists(f"{out_dir}/ae_global_bar.png"):
            st.image(f"{out_dir}/ae_global_bar.png")

        st.markdown("---")

        st.markdown("### Zero-Day Proof — XGBoost vs Autoencoder Force Plots")
        st.caption("How to read: red features push the value higher, blue push it lower. Left plot: XGBoost found "
                   "nothing suspicious (it called the flow BENIGN). Right plot: the autoencoder's reconstruction error "
                   "went above the threshold, so the flow looks unlike normal traffic — the red features are why.")
        thr    = float(st.session_state.active_thresh)
        ledger = st.session_state.hybrid_logs

        if ledger.empty:
            st.info("The Hybrid ledger is empty — run the evaluation first so it can flag anomalies.")
        else:
            flagged = ledger[ledger['DAE Alert'].astype(str).str.contains("Anomaly")].copy()
            if flagged.empty:
                st.info("The Hybrid ledger has not flagged any anomalies yet.")
            else:
                flagged['_err']  = flagged['Reconstruction Error'].astype(float)
                zero_only = flagged[flagged['Hybrid Decision'].astype(str).str.contains("ZERO-DAY")].copy()
                zero_only = zero_only.sort_values('Log Index').reset_index(drop=True)
                st.write(f"**{len(zero_only):,}** zero-day threats (XGBoost said BENIGN, autoencoder flagged anomaly)")

            if flagged.empty or zero_only.empty:
                if not flagged.empty:
                    st.info("No zero-day threats flagged yet.")
            else:
                # Every zero-day log is listed (no cap)
                info = {int(r['Log Index']): r for _, r in zero_only.iterrows()}
                log_indices_list = list(info.keys())

                zd_pick = st.selectbox(
                    "Choose a zero-day log to explain", log_indices_list,
                    index=0, key="xai_zd_pick",
                    format_func=lambda i: (f"Log #{i} — {info[i]['Hybrid Decision']} — error {info[i]['_err']:.4f}"
                                           f" — XGBoost: {info[i]['XGBoost Prediction']} — true: {info[i]['True Label']}"),help="Log # is the ledger index (starts at 1). The dataset row it explains is Log # minus 1 (rows start at 0).")

                if st.button("Generate Zero-Day Force Plots", disabled=locked, key="xai_btn_zd"):
                    ds_row = int(zd_pick) - 1
                    with st.spinner("Explaining the selected flow..."):
                        try:
                            X_row, X_ae_row, _ = self._build_xai_frames(dataset_workspace.iloc[[ds_row]])
                            xai.explain_zero_day_proof(
                                self.xgb_model, xgb_explainer(), self.pytorch_engine,
                                ae_explainer(), X_bg, X_row, thr,
                                zero_day_index=0, display_index=ds_row,
                                class_names=class_names, X_test_ae=X_ae_row)
                            st.session_state.xai_zd_idx = ds_row
                        except Exception as e:
                            st.warning(f"Zero-day proof: {e}")

        zd_shown = st.session_state.get("xai_zd_idx")
        if zd_shown is not None:
            z1, z2 = st.columns(2)
            p_x, p_a = f"{out_dir}/zero_day_xgb_force_{zd_shown}.png", f"{out_dir}/zero_day_ae_force_{zd_shown}.png"
            if os.path.exists(p_x): z1.image(p_x)
            if os.path.exists(p_a): z2.image(p_a)

    def build_console_ui(self):
        st.sidebar.markdown("## Control Dashboard Configuration")
        st.sidebar.markdown("---")

        uploaded_file = st.sidebar.file_uploader("Upload Network Traffic Dataset (CSV)", type=["csv"],max_upload_size=2048)
        if uploaded_file is not None:
            dataset_workspace  = clean_and_parse_input_stream(uploaded_file)
            total_dataset_rows = len(dataset_workspace)
            st.sidebar.success(f"File Staged: {total_dataset_rows:,} rows.")
        else:
            st.sidebar.info("Awaiting CSV data asset...")
            dataset_workspace  = None
            total_dataset_rows = 0

        st.sidebar.markdown("---")
        batch_size    = st.sidebar.slider("Batch Window Size", 5000, 10000, step=10,
                                          key="batch_window_size")
        tick_interval = st.sidebar.slider("Console Frame Ingestion Delay", 0.00, 1.00, 0.00, step=0.01)

        def start_matrix_run():
            if st.session_state.current_index == 0:        # a new run (not a resume after Halt) -> new table
                st.session_state.ledger_table = make_table_name(uploaded_file.name)
            st.session_state.stream_active = True
        def stop_matrix_run():  st.session_state.stream_active = False
        def clear_matrix_run():
            for key in list(st.session_state.keys()): del st.session_state[key]

        st.sidebar.markdown("---")
        if not st.session_state.stream_active:
            st.sidebar.button("Run Evaluation", disabled=(dataset_workspace is None), on_click=start_matrix_run)
        else:
            st.sidebar.button("Halt Evaluation", on_click=stop_matrix_run)
        st.sidebar.button("Reset Dashboard", on_click=clear_matrix_run)

        st.markdown("# Real-Time Network Operations Center Console")

        # Status banner: idle / running (with progress %) / finished
        status_box = st.empty()
        if st.session_state.simulation_finished:
            status_box.markdown(
                '<div class="status-container">'
                '<div class="status-row"><span class="status-dot done"></span>'
                '<div>Analysis Completed: iteration terminated.</div></div>'
                '</div>',
                unsafe_allow_html=True)
        elif st.session_state.stream_active:
            status_box.markdown(
                f'<div class="status-container" style="border-left-color:#f5a623;">'
                f'<div class="status-row"><span class="status-dot active"></span>'
                f'<div>Active Scan Underway: Processing row array block: '
                f'{(st.session_state.current_index / max(1, total_dataset_rows))*100:.2f}%</div></div>'
                f'<div class="status-loading-track"><div class="status-loading-fill"></div></div>'
                f'</div>', unsafe_allow_html=True)
        else:
            status_box.markdown(
                '<div class="status-container" style="border-left-color:#768390;">'
                '<div class="status-row"><span class="status-dot idle"></span>'
                '<div> System Online: Standing by for data initialization...</div></div>'
                '</div>',
                unsafe_allow_html=True)

        tab_results, tab_insights, tab_eda, tab_xai, tab_validator = st.tabs([
            "Hybrid NIDS",
            "Result Insights",
            "Exploratory Data Analysis",
            "Explainable AI (SHAP)",
            "Hash Validator"
        ])

        # ---------------- TAB 1: HYBRID NIDS (live XGBoost + autoencoder dashboards) ----------------
        with tab_results:
            st.markdown("## Supervised XGBoost Module")
            xgb_metrics = st.container()
            xgb_charts  = st.columns(2)

            def draw_xgb_view():
                tot = st.session_state.current_index
                acc = (st.session_state.correctly_predicted / tot * 100) if tot > 0 else 0.0

                with xgb_metrics:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Packets Analyzed",            f"{tot:,} / {total_dataset_rows:,}")
                    m2.metric("Injected Attacks Identified", f"{st.session_state.total_attacks_injected:,}")
                    m3.metric("Classification Accuracy",     f"{acc:.2f}%")
                    m4.metric("Latency", f"{st.session_state.latest_inference_latency:.3f} ms")

                    st.write("")
                    bm1, bm2, bm3 = st.columns(3)
                    bm1.metric("Aligned True Predictions",       f"{st.session_state.correctly_predicted:,}")
                    bm2.metric("Cross-Class Misclassifications", f"{st.session_state.model_misclassifications:,}")
                    miss_rate = (st.session_state.completely_missed / max(1, st.session_state.total_attacks_injected) * 100)
                    bm3.metric("Undetected Incidents (FN)", f"{st.session_state.completely_missed:,}",
                               delta=f"{miss_rate:.1f}% Miss Rate" if tot > 0 else None, delta_color="inverse")

                with xgb_charts[0]:
                    st.markdown("#### Line Performance Metrics (Flow Duration)")
                    if st.session_state.flow_duration_history:
                        st.line_chart(pd.DataFrame(st.session_state.flow_duration_history[-1000:],
                                                   columns=["Flow Duration"]), use_container_width=True)
                with xgb_charts[1]:
                    st.markdown("#### Pretrained XGBoost Target Classification Densities")
                    if st.session_state.threat_distribution:
                        st.bar_chart(pd.DataFrame.from_dict(st.session_state.threat_distribution,
                                                            orient='index', columns=['Occurrences']),
                                     use_container_width=True)

            if dataset_workspace is None:
                st.info("Upload a dataset from the sidebar and click **Run Evaluation** to see real-time XGBoost visuals.")
            else:
                draw_xgb_view()

            st.markdown("---")
            st.markdown("## Unsupervised Denoising Autoencoder Module")

            if dataset_workspace is None:
                st.info("Upload a dataset from the sidebar and click **Run Evaluation** to see real-time Autoencoder visuals.")
            else:
                ae_param_c1, _, __ = st.columns([2, 2, 2])
                with ae_param_c1:
                    st.metric("K-Means Clusters", f"{KMEANS_CLUSTERS} Behaviour Groups")

                st.markdown("")
                ae_metrics     = st.container()
                st.markdown("")
                ae_charts_row1 = st.columns(2)
                ae_charts_row2 = st.columns(2)

                def draw_ae_view():
                    tot = st.session_state.current_index

                    with ae_metrics:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Rows Processed",             f"{tot:,} / {total_dataset_rows:,}")
                        m2.metric("Alerts Issued",               f"{st.session_state.ae_total_anomalies_detected:,}")
                        m3.metric("Injected Attacks Identified", f"{st.session_state.total_attacks_injected:,}")
                        m4.metric("Threshold Limit",             f"{st.session_state.active_thresh:.6f}")

                        st.write("")
                        bm1, bm2, bm3 = st.columns(3)
                        bm1.metric("Confirmed Anomalies (TP)", f"{st.session_state.ae_correct_alerts:,}")
                        bm2.metric("False Positive Flags (FP)", f"{st.session_state.ae_false_alarms:,}")
                        blind_spot_rate = (st.session_state.ae_missed_attacks / max(1, st.session_state.total_attacks_injected) * 100)
                        bm3.metric("Structural Leakage (FN)", f"{st.session_state.ae_missed_attacks:,}",
                                   delta=f"{blind_spot_rate:.1f}% Blind Spot" if tot > 0 else None, delta_color="inverse")

                    with ae_charts_row1[0]:
                        st.markdown("#### Real-Time Reconstruction Error (Mean Squared Loss)")
                        if st.session_state.anomaly_reconstruction_scores:
                            st.area_chart(pd.DataFrame(st.session_state.anomaly_reconstruction_scores[:tot:10],
                                                       columns=["Reconstruction Loss"]), use_container_width=True)
                    with ae_charts_row1[1]:
                        st.markdown("#### K-Means Behaviour Groups (flows per group)")
                        if st.session_state.cluster_id_history:
                            _group_sizes = pd.Series(st.session_state.cluster_id_history).value_counts().sort_index()
                            _c_ids = st.session_state.cluster_id_history
                            _t_labs = st.session_state.y_true_list
                            _group_sizes.index = [get_cluster_display_name(g, _c_ids, _t_labs) for g in _group_sizes.index]
                            st.bar_chart(_group_sizes)

                    with ae_charts_row2[0]:
                        st.markdown("#### Historical Reconstruction Anomaly Scoring Radar")
                        if st.session_state.anomaly_reconstruction_scores:
                            fig, ax = plt.subplots(figsize=(6, 3.2))
                            fig.patch.set_facecolor('#0b0f14')
                            ax.set_facecolor('#1c2128')
                            ax.plot(st.session_state.anomaly_reconstruction_scores[-300:],
                                    color='#ff6b6b', alpha=0.8)
                            ax.axhline(y=st.session_state.active_thresh, color='#f5a623',
                                       linestyle='--', label='Threshold')
                            ax.tick_params(colors='#adbac7')
                            plt.legend(facecolor='#1c2128', edgecolor='#444c56', labelcolor='#adbac7')
                            st.pyplot(fig)
                            plt.close()

                    with ae_charts_row2[1]:
                        st.markdown("#### Confusion Matrix Profile Matrix")
                        if st.session_state.ae_true_labels_history and st.session_state.ae_pred_labels_history:
                            fig, ax = plt.subplots(figsize=(5, 3.2))
                            fig.patch.set_facecolor('#0b0f14')
                            cm = confusion_matrix(st.session_state.ae_true_labels_history,
                                                 st.session_state.ae_pred_labels_history, labels=[0, 1])
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=['BENIGN', 'ATTACK'], yticklabels=['BENIGN', 'ATTACK'],
                                        cbar=False, ax=ax)
                            ax.tick_params(colors='#adbac7')
                            st.pyplot(fig)
                            plt.close()

                draw_ae_view()

        # ---------------- TAB 2: RESULT INSIGHTS (detection ledger + classification reports) ----------------
        with tab_insights:
            st.markdown("## Hybrid Model — Detection Ledger")

            _ui_rows = len(st.session_state.hybrid_logs)
            if ledger_db is None:
                st.warning(f"Database mirror unavailable: {LEDGER_DB_IMPORT_ERROR}")
            elif st.session_state.ledger_db_error:
                st.error(f"Database: {st.session_state.ledger_db_saved:,} / {_ui_rows:,} ledger rows saved — "
                         f"{st.session_state.ledger_db_error} (unsaved rows are retried with the next batch)")
            elif _ui_rows > 0:
                _db_rows = ledger_db.count_rows(st.session_state.ledger_table) if not st.session_state.stream_active else None
                _db_txt  = f"{_db_rows:,}" if _db_rows is not None else f"{st.session_state.ledger_db_saved:,}"
                st.caption(f"Database table `{st.session_state.ledger_table}`: {_db_txt} / {_ui_rows:,} ledger rows saved")
            
            if not st.session_state.hybrid_logs.empty and 'Hash' in st.session_state.hybrid_logs.columns:
                if hash_utils is not None:
                    try:
                        hash_stats = hash_utils.get_hash_statistics(st.session_state.hybrid_logs)
                        if hash_stats:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Hashed Threats", hash_stats.get('hashed_attacks', 0))
                            col2.metric("Benign Samples", hash_stats.get('benign_samples', 0))
                            col3.metric("Unique Signatures", hash_stats.get('unique_hashes', 0))
                            col4.metric("Threat %", f"{hash_stats.get('hash_percentage', 0)}%")
                            st.markdown("---")
                    except Exception as e:
                        pass

            if not st.session_state.hybrid_logs.empty:
                # Primary Ledger View (Upper Section)
                upper_cols = [
                    'Log Index', 'Source IP', 'Destination IP', 'Dst Port',
                    'True Label', 'XGBoost Prediction', 'DAE Alert',
                    'Reconstruction Error', 'Hybrid Decision'
                ]
                existing_upper = [c for c in upper_cols if c in st.session_state.hybrid_logs.columns]
                st.dataframe(st.session_state.hybrid_logs[existing_upper], use_container_width=True, hide_index=True, height=250)

                st.markdown("")

                # Secondary Ledger View (Lower Section with matching Log Index)
                lower_cols = ['Log Index', 'NLP Summary', 'Hash', 'Download']
                
                # Ensure Download column exists for lower table
                if 'Download' not in st.session_state.hybrid_logs.columns:
                    st.session_state.hybrid_logs['Download'] = "Ready"
                    
                existing_lower = [c for c in lower_cols if c in st.session_state.hybrid_logs.columns]
                render_lower_ledger(st.session_state.hybrid_logs, existing_lower, 200)
            else:
                st.info("Run evaluation to see the hybrid detection ledger.")

            st.markdown("---")
            st.markdown("## Hybrid Model — Classification Reports")

            xgb_ready = bool(st.session_state.y_true_list and st.session_state.y_pred_list)
            ae_ready  = bool(st.session_state.ae_true_labels_history and st.session_state.ae_pred_labels_history)

            if xgb_ready or ae_ready:
                st.markdown("#### Report 1 — Per-Class Breakdown (XGBoost)")
                if xgb_ready:
                    xgb_full = pd.DataFrame(classification_report(
                        st.session_state.y_true_list,
                        st.session_state.y_pred_list,
                        output_dict=True, zero_division=0
                    )).transpose().round(3)
                    xgb_full.index.name = 'Class'
                    st.dataframe(xgb_full, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Report 2 — Hybrid Model Summary")
                hybrid_ready = bool(st.session_state.hybrid_true_list and st.session_state.hybrid_pred_list)
                if hybrid_ready:
                    hybrid_unique = sorted(set(st.session_state.hybrid_true_list) | set(st.session_state.hybrid_pred_list))
                    hybrid_tnames = ['BENIGN', 'ATTACK'] if len(hybrid_unique) == 2 else (['BENIGN'] if hybrid_unique == [0] else ['ATTACK'])
                    hybrid_summary = pd.DataFrame(classification_report(
                        st.session_state.hybrid_true_list,
                        st.session_state.hybrid_pred_list,
                        labels=hybrid_unique,
                        target_names=hybrid_tnames,
                        output_dict=True, zero_division=0
                    )).transpose().round(3)
                    st.dataframe(hybrid_summary, use_container_width=True)

        # ---------------- TAB 3: EXPLORATORY DATA ANALYSIS (EDA) ----------------
        with tab_eda:
            eda_dataset_type = st.radio(
                "Dataset Type", ["Training", "Testing"], horizontal=True, key="eda_dataset_type"
            )
            st.markdown("---")

            if eda_dataset_type == "Training":
                TRAIN_LABEL_COUNTS = {
                    'BENIGN':                       592822,
                    'DoS Hulk':                     159048,
                    'PortScan':                      159023,
                    'DDoS':                          95123,
                    'DoS GoldenEye':                  7647,
                    'DoS slowloris':                  5707,
                    'DoS Slowhttptest':               5109,
                    'FTP-Patator':                     3984,
                    'SSH-Patator':                     2988,
                    'Bot':                            2208,
                    'Web Attack - Brute Force':       1365,
                    'Web Attack - XSS':                 561,
                    'Infiltration':                     48,
                    'Web Attack - Sql Injection':       12,
                    'Heartbleed':                       11,
                }
                TRAIN_MEDIAN_FLOW_DURATION = {
                    'PortScan':                         46,
                    'Bot':                             542,
                    'BENIGN':                        78498,
                    'DoS Hulk':                     131074,
                    'Web Attack - Sql Injection':  6063012,
                    'Web Attack - XSS':            6300000,
                    'Web Attack - Brute Force':    6677000,
                    'DDoS':                        7397628,
                    'FTP-Patator':                 9000000,
                    'DoS GoldenEye':              10485759,
                    'SSH-Patator':                12000000,
                    'DoS Slowhttptest':           87391753,
                    'Infiltration':               88080741,
                    'DoS slowloris':              99999937,
                    'Heartbleed':                119999937,
                }

                label_series = pd.Series(TRAIN_LABEL_COUNTS)
                total_train  = label_series.sum()

                st.markdown("#### Label Distribution")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                fig1.patch.set_facecolor('#0b0f14')
                ax1.set_facecolor('#1c2128')
                sns.barplot(x=label_series.index, y=label_series.values, palette='flare', ax=ax1)
                ax1.set_xlabel('Attack Type', color='#adbac7')
                ax1.set_ylabel('Count', color='#adbac7')
                ax1.set_yscale('log')
                ax1.tick_params(colors='#adbac7')
                plt.xticks(rotation=90, color='#adbac7')
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                st.markdown("---")

                st.markdown("#### BENIGN vs ATTACK")
                benign_cnt = TRAIN_LABEL_COUNTS['BENIGN']
                attack_cnt = total_train - benign_cnt
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                fig2.patch.set_facecolor('#0b0f14')
                ax2.pie(
                    [benign_cnt, attack_cnt],
                    labels=['BENIGN', 'ATTACK'],
                    autopct='%1.3f%%',
                    colors=['#4CAF50', '#F44336'],
                    textprops={'color': '#f0f6fc'}
                )
                bva_left, bva_col, bva_right = st.columns([1, 1, 1])
                with bva_col:
                    st.pyplot(fig2, use_container_width=False)
                plt.close(fig2)

                st.markdown("---")

                st.markdown("#### Traffic Classes Breakdown")
                counts = label_series
                percentages = (label_series / total_train) * 100
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                fig3.patch.set_facecolor('#0b0f14')
                colors = sns.color_palette('tab20', len(counts))
                explode = [0.05] * len(counts)
                wedges, _ = ax3.pie(
                    counts,
                    labels=None,
                    startangle=140,
                    colors=colors,
                    explode=explode
                )
                legend_labels = [
                    f'{label} ({pct:.4f}%) [{count:,}]'
                    for label, pct, count in zip(counts.index, percentages, counts.values)
                ]
                ax3.legend(
                    wedges,
                    legend_labels,
                    title="Traffic Classes",
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    facecolor='#1c2128', edgecolor='#444c56',
                    labelcolor='#adbac7', title_fontsize=9, fontsize=8
                )
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

                st.markdown("---")

                st.markdown("#### Median Flow Duration By Label")
                dur_series = pd.Series(TRAIN_MEDIAN_FLOW_DURATION).sort_values()
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                fig4.patch.set_facecolor('#0b0f14')
                ax4.set_facecolor('#1c2128')
                dur_series.plot(kind='barh', color=sns.color_palette('crest'), ax=ax4)
                ax4.set_xlabel('Median Flow Duration', color='#adbac7')
                ax4.set_ylabel('Label', color='#adbac7')
                ax4.set_xscale('log')
                ax4.tick_params(colors='#adbac7')
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)

            else:
                if dataset_workspace is None:
                    st.info("Upload a dataset CSV via the sidebar to view testing plots.")
                else:
                    st.markdown(f"#### Label Distribution")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#0b0f14')
                    ax.set_facecolor('#1c2128')
                    rt_counts = dataset_workspace['target_label'].value_counts()
                    sns.barplot(x=rt_counts.index, y=rt_counts.values, palette='flare', ax=ax)
                    ax.set_yscale('log')
                    plt.xticks(rotation=90, ha='right', color='#adbac7')
                    ax.set_xlabel('Attack Type', color='#adbac7')
                    ax.set_ylabel('Count', color='#adbac7')
                    ax.tick_params(colors='#adbac7')
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.markdown("---")

                    st.markdown(f"#### Benign vs Attack")
                    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
                    fig.patch.set_facecolor('#0b0f14')
                    rt_benign = (dataset_workspace['target_label'].str.upper() == 'BENIGN').sum()
                    rt_attack = (dataset_workspace['target_label'].str.upper() != 'BENIGN').sum()
                    ax.pie(
                        [rt_benign, rt_attack],
                        labels=['BENIGN', 'ATTACK'],
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#F44336'],
                        startangle=90,
                        textprops={'color': '#f0f6fc', 'fontsize': 28},
                    )

                    import io as _io
                    _buf_rt1 = _io.BytesIO()
                    fig.savefig(_buf_rt1, format='png', dpi=100, facecolor='#0b0f14')
                    _buf_rt1.seek(0)
                    st.image(_buf_rt1)
                    plt.close()

                    st.markdown("---")

                    st.markdown("#### Traffic Classes Breakdown")
                    rt_lc   = dataset_workspace['target_label'].value_counts()
                    rt_pcts = dataset_workspace['target_label'].value_counts(normalize=True) * 100
                    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                    fig.patch.set_facecolor('#0b0f14')
                    ax.set_facecolor('#0b0f14')
                    rt_colors  = sns.color_palette('tab20', len(rt_lc))
                    rt_explode = [0.05] * len(rt_lc)
                    wedges, _ = ax.pie(rt_lc, labels=None, startangle=140,
                                       colors=rt_colors, explode=rt_explode)
                    rt_leg = [
                        f'{lbl} ({pct:.4f}%) [{cnt:,}]'
                        for lbl, pct, cnt in zip(rt_lc.index, rt_pcts, rt_lc.values)
                    ]
                    ax.legend(wedges, rt_leg, title="Traffic Classes", loc="center left",
                              bbox_to_anchor=(1.0, 0.5), facecolor='#1c2128', edgecolor='#444c56',
                              labelcolor='#adbac7', title_fontsize=9, fontsize=8)
                    plt.tight_layout()
                    _buf_rt2 = _io.BytesIO()
                    fig.savefig(_buf_rt2, format='png', dpi=100, bbox_inches='tight', facecolor='#0b0f14')
                    _buf_rt2.seek(0)
                    st.image(_buf_rt2, use_container_width=True)
                    plt.close()


        # ---------------- TAB 4: EXPLAINABLE AI (SHAP) ----------------
        with tab_xai:
            self.render_xai_tab(dataset_workspace)

        # ---------------- TAB 5: HASH VALIDATOR ----------------
        with tab_validator:
            st.markdown("## Threat Signature & Hash Validator")
            st.caption("Verify threat hashes against standard cryptography standards and threat intelligence databases.")

            hash_input = st.text_input("Enter Hash Signature (SHA-256)", "")
            
            if st.button("Validate Hash"):
                if hash_validator is not None:
                    result = hash_validator.validate_hash_format(hash_input.strip())
                    if result["valid"]:
                        st.success(f"Valid {result['type']} Hash Format!")
                        _h = hash_input.strip().lower()
                        _led = st.session_state.hybrid_logs
                        if len(_led) == 0 or 'Hash' not in _led.columns:
                            st.info("Whole-hash check skipped: no hashes in the ledger yet. Run an evaluation first.")
                        elif len(_h) != 64:
                            st.info(f"Whole-hash check skipped: the ledger stores SHA-256 (64 characters) only, "
                                    f"and this is {result['type']}.")
                        else:
                            _stored = _led['Hash'].astype(str).str.strip().str.lower()
                            _m = _led[_stored == _h]
                            if len(_m) > 0:
                                st.success(f"Whole hash verified: all 64 characters match {len(_m):,} ledger record(s).")
                                _cols = [c for c in ['Log Index', 'Source IP', 'Destination IP', 'Dst Port',
                                                     'True Label', 'XGBoost Prediction', 'Hybrid Decision',
                                                     'Reconstruction Error', 'Flow Duration'] if c in _m.columns]
                                st.dataframe(_m[_cols], use_container_width=True, hide_index=True)
                            else:
                                st.error("Whole-hash check failed: no ledger record has this exact hash.")
                        is_threat = hash_validator.is_blacklisted_hash(hash_input.strip())
                        if is_threat:
                            st.error("Hash identified in Blacklisted Threat Database!")
                        else:
                            st.info("Hash format valid. No match in local threat blacklist.")
                    else:
                        st.error("Invalid Hash Format. Expected MD5, SHA-1, or SHA-256.")
                        _h = hash_input.strip().lower()
                        _led = st.session_state.hybrid_logs
                        if len(_h) >= 16 and 'Hash' in _led.columns and len(_led) > 0:
                            _stored = _led['Hash'].astype(str).str.strip().str.lower()
                            _pm = _led[_stored.str.startswith(_h)]
                            if len(_pm) > 0:
                                st.warning(f"Incomplete hash: these {len(_h)} characters match only the start of "
                                           f"{len(_pm):,} ledger hash(es). Paste the full 64-character SHA-256.")
                else:
                    st.error(f"Validator module error: {VALIDATOR_IMPORT_ERROR}")

        # ==============================================================================
        # PROCESSING LOOP: while a run is active, process ONE batch, then rerun the page
        # (the rerun redraws the dashboard, which is how the numbers update live)
        # ==============================================================================
        if st.session_state.stream_active and dataset_workspace is not None:
            if st.session_state.current_index < total_dataset_rows:
                start         = st.session_state.current_index
                end           = min(start + batch_size, total_dataset_rows)
                working_chunk = dataset_workspace.iloc[start:end].copy()

                # Align the batch to the autoencoder's feature list (missing columns become 0)
                ae_target_features = self.ae_config.get("feature_cols", []) if self.ae_config else []

                if ae_target_features:
                    ae_aligned_df = pd.DataFrame(index=working_chunk.index)
                    for col in ae_target_features:
                        ae_aligned_df[col] = working_chunk[col] if col in working_chunk.columns else 0.0
                    ae_aligned_df = ae_aligned_df[ae_target_features]
                else:
                    meta_cols = {'target_label', 'display_src_ip', 'display_dst_ip', 'display_dst_port'}
                    ae_aligned_df = working_chunk.drop(
                        columns=[c for c in meta_cols if c in working_chunk.columns], errors='ignore'
                    ).select_dtypes(include=[np.number])

                xgb_source = working_chunk.copy()
                xgb_source.columns = xgb_source.columns.str.strip().str.lower()
                meta_lower = {'target_label', 'display_src_ip', 'display_dst_ip', 'display_dst_port'}
                xgb_source = xgb_source.drop(
                    columns=[c for c in meta_lower if c in xgb_source.columns], errors='ignore'
                ).select_dtypes(include=[np.number])

                # STEP A: XGBoost predictions
                if self.xgb_model is not None:
                    X_xgb = xgb_source.copy()
                    if hasattr(self.xgb_model, "feature_names_in_"):
                        for mc in [c for c in self.xgb_model.feature_names_in_ if c not in X_xgb.columns]:
                            X_xgb[mc] = 0.0
                        X_xgb = X_xgb[self.xgb_model.feature_names_in_]
                    t_0 = time.perf_counter()
                    xgb_preds_encoded = self.xgb_model.predict(X_xgb)
                    xgb_preds_text    = self.label_encoder.inverse_transform(xgb_preds_encoded)
                    st.session_state.latest_inference_latency = ((time.perf_counter() - t_0) * 1000.0) / len(working_chunk)
                else:
                    xgb_preds_text = ['BENIGN'] * len(working_chunk)

                # STEP B: Autoencoder predictions
                if self.ae_scaler is not None and self.pytorch_engine is not None:
                    ae_matrix_raw    = ae_aligned_df.values.astype(np.float32)
                    ae_matrix_scaled = self.ae_scaler.transform(ae_matrix_raw)
                    ae_matrix_scaled = np.clip(ae_matrix_scaled, -10, 10)

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    try:
                        with torch.no_grad():
                            tensor_input    = torch.tensor(ae_matrix_scaled, dtype=torch.float32).to(device)
                            latent_vectors  = self.pytorch_engine.encoder(tensor_input).cpu().numpy()
                            decoded_vectors = self.pytorch_engine(tensor_input).cpu().numpy()
                            chunk_errors    = np.mean((decoded_vectors - ae_matrix_scaled) ** 2, axis=1)
                    except Exception as err:
                        st.sidebar.error(f"PyTorch Runtime Fault: {err}")
                        st.session_state.stream_active = False
                        st.stop()
                else:
                    latent_vectors = np.zeros((len(working_chunk), 32))
                    chunk_errors   = np.zeros(len(working_chunk))

                # K-Means groups the autoencoder's latent codes into behaviour clusters (re-fitted every 5th batch)
                for vector in latent_vectors:
                    st.session_state.latent_space_history.append(vector)

                for raw_label in working_chunk['target_label'].unique():
                    st.session_state.unique_labels_seen.add(raw_label)

                history_depth = len(st.session_state.latent_space_history)
                st.session_state.kmeans_batch_counter += 1

                if history_depth >= KMEANS_CLUSTERS and (
                    st.session_state.kmeans_batch_counter % RECLUSTER_INTERVAL == 0
                    or st.session_state.kmeans_model is None
                ):
                    # (Re)fit on every latent code seen so far and re-assign ALL rows
                    cluster_model        = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=4)
                    full_cluster_lineage = cluster_model.fit_predict(np.array(st.session_state.latent_space_history))
                    st.session_state.cluster_id_history = full_cluster_lineage.tolist()
                    st.session_state.kmeans_model       = cluster_model
                elif st.session_state.kmeans_model is not None:
                    # Between re-fits: give the new rows their REAL group using the last fitted model
                    st.session_state.cluster_id_history.extend(
                        st.session_state.kmeans_model.predict(latent_vectors).tolist())
                else:
                    # Not enough rows yet to build the groups
                    st.session_state.cluster_id_history.extend([0] * len(working_chunk))

                # STEP C: VECTORIZED METRICS & HYBRID DECISION
                chunk_len        = len(working_chunk)
                actual_threats_s = working_chunk['target_label'].fillna('BENIGN').astype(str).str.strip()
                xgb_preds_s      = pd.Series(xgb_preds_text, dtype=str).str.strip()

                actual_threats  = actual_threats_s.values
                xgb_preds_arr   = xgb_preds_s.values
                sips            = working_chunk['display_src_ip'].fillna('0.0.0.0').astype(str).values
                dips            = working_chunk['display_dst_ip'].fillna('0.0.0.0').astype(str).values
                ports           = working_chunk['display_dst_port'].fillna('0').astype(str).values
                recon_scores    = chunk_errors

                xgb_is_attack_mask = (xgb_preds_s.str.upper() != 'BENIGN').values
                ae_alert_mask      = recon_scores > st.session_state.active_thresh
                is_attack_mask     = (actual_threats_s.str.upper() != 'BENIGN').values
                true_binary_arr    = is_attack_mask.astype(int)
                ae_alert_arr       = ae_alert_mask.astype(int)

                both_attack =  xgb_is_attack_mask &  ae_alert_mask
                zero_day    = ~xgb_is_attack_mask &  ae_alert_mask
                xgb_only    =  xgb_is_attack_mask & ~ae_alert_mask
                both_benign = ~xgb_is_attack_mask & ~ae_alert_mask

                # Hybrid decision for every flow (every outcome now carries its emoji)
                hybrid_labels = np.where(both_attack, '🚨 ATTACK',
                               np.where(zero_day,   '🔴 ZERO-DAY THREAT',
                               np.where(xgb_only,   '🟠 THREAT (XGB)',
                                                     '🟢 BENIGN')))

                hybrid_pred_binary = (~both_benign).astype(int)

                # Update the running counters: XGBoost correct / wrong / missed, autoencoder TP / FP / FN
                correct_mask    = (xgb_preds_arr == actual_threats)
                missed_mask     = (~correct_mask) & is_attack_mask & (~xgb_is_attack_mask)
                st.session_state.total_attacks_injected   += int(is_attack_mask.sum())
                st.session_state.correctly_predicted      += int(correct_mask.sum())
                st.session_state.model_misclassifications += int((~correct_mask).sum())
                st.session_state.completely_missed        += int(missed_mask.sum())

                tp_mask = ae_alert_mask & is_attack_mask
                fp_mask = ae_alert_mask & ~is_attack_mask
                fn_mask = ~ae_alert_mask & is_attack_mask
                st.session_state.ae_total_anomalies_detected += int(ae_alert_mask.sum())
                st.session_state.ae_correct_alerts           += int(tp_mask.sum())
                st.session_state.ae_false_alarms             += int(fp_mask.sum())
                st.session_state.ae_missed_attacks           += int(fn_mask.sum())

                flow_dur_cols = [c for c in working_chunk.columns if 'flow duration' in c.lower()]
                fwd_pkt_cols  = [c for c in working_chunk.columns if 'total fwd packet' in c.lower()]
                flow_dur_vals = working_chunk[flow_dur_cols[0]].fillna(0.0).values.tolist() if flow_dur_cols else [0.0] * chunk_len
                fwd_pkt_vals  = working_chunk[fwd_pkt_cols[0]].fillna(0.0).values.tolist() if fwd_pkt_cols else [0.0] * chunk_len
                st.session_state.flow_duration_history.extend(flow_dur_vals)
                st.session_state.fwd_packets_history.extend(fwd_pkt_vals)
                st.session_state.anomaly_reconstruction_scores.extend(recon_scores.tolist())

                st.session_state.attack_sse_scores.extend(recon_scores[is_attack_mask].tolist())
                st.session_state.benign_sse_scores.extend(recon_scores[~is_attack_mask].tolist())

                st.session_state.y_true_list.extend(actual_threats.tolist())
                st.session_state.y_pred_list.extend(xgb_preds_arr.tolist())
                st.session_state.ae_true_labels_history.extend(true_binary_arr.tolist())
                st.session_state.ae_pred_labels_history.extend(ae_alert_arr.tolist())

                st.session_state.hybrid_true_list.extend(true_binary_arr.tolist())
                st.session_state.hybrid_pred_list.extend(hybrid_pred_binary.tolist())

                pred_series = pd.Series(xgb_preds_arr)
                for label, count in pred_series.value_counts().items():
                    st.session_state.threat_distribution[label] = (
                        st.session_state.threat_distribution.get(label, 0) + int(count))

                master_indices = np.arange(start + 1, start + chunk_len + 1, dtype=int)

                # Generate NLP Summaries using nlp_utils
                nlp_summaries = []
                for i in range(chunk_len):
                    row_data = pd.Series({
                        'Source IP': sips[i], 'Destination IP': dips[i],
                        'Dst Port': ports[i], 'True Label': actual_threats[i],
                        'Hybrid Decision': hybrid_labels[i]
                    })
                    nlp_summaries.append(
                        nlp_utils.generate_threat_summary(row_data) if nlp_utils else "N/A"
                    )

                hybrid_log_df = pd.DataFrame({
                    'Log Index'           : master_indices,
                    'Source IP'           : sips,
                    'Destination IP'      : dips,
                    'Dst Port'            : ports,
                    'True Label'          : actual_threats,
                    'XGBoost Prediction'  : xgb_preds_arr,
                    'DAE Alert'           : np.where(ae_alert_mask, "⚠️ Anomaly", "🟢 NORMAL"),
                    'Reconstruction Error': [f"{v:.6f}" for v in recon_scores],
                    'Hybrid Decision'     : hybrid_labels,
                    'NLP Summary'         : nlp_summaries,
                    'Download'            : ["Ready"] * chunk_len
                })

                # Compute Hashes using hash_utils
                if hash_utils is not None:
                    try:
                        hybrid_log_df = hash_utils.add_hash_column(hybrid_log_df)
                    except Exception as e:
                        hybrid_log_df['Hash'] = ''

                hybrid_log_df['Flow Duration'] = flow_dur_vals

                st.session_state.hybrid_logs = pd.concat(
                    [st.session_state.hybrid_logs, hybrid_log_df],
                    ignore_index=True)

                # Mirror the ledger into MySQL at the same moment (also retries rows left behind by an earlier failure)
                if ledger_db is not None and st.session_state.ledger_table:
                    _pending = st.session_state.hybrid_logs.iloc[st.session_state.ledger_db_saved:]
                    _ok, _msg, _n = ledger_db.save_ledger_rows(_pending, st.session_state.ledger_table)
                    if _ok:
                        st.session_state.ledger_db_saved = len(st.session_state.hybrid_logs)
                        st.session_state.ledger_db_error = ""
                    else:
                        st.session_state.ledger_db_error = _msg

                st.session_state.current_index = end
                if st.session_state.current_index >= total_dataset_rows:
                    st.session_state.stream_active    = False
                    st.session_state.simulation_finished = True

                if tick_interval > 0:
                    time.sleep(tick_interval)
                st.rerun()
            else:
                st.session_state.stream_active       = False
                st.session_state.simulation_finished = True
                st.rerun()

# Entry point (Streamlit runs this file from top to bottom on every interaction)
if __name__ == "__main__":
    console_app = ProductionNIDSWorkspace()
    console_app.build_console_ui()