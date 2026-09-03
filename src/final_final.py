
import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import warnings
import os
import subprocess
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# 1. GLOBAL PAGE LAYOUT CONFIGURATION
st.set_page_config(
    layout="wide",
    page_title="Cyber Intrusion Detection Console"
)

# 2. OPERATIONAL DASHBOARD THEME AND INTERFACE STYLING
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
            display: flex;
            align-items: center;
            gap: 12px;
        }
    </style>
""", unsafe_allow_html=True)


# 3. DEEP AUTOENCODER MODEL ARCHITECTURE DEFINITION
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


# 4. PIPELINE CACHED ARTIFACT INITIALIZATION LOADER & UPGRADED EDA PREPROCESSING
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

    # 1. Stripping whitespaces from column headers
    df.columns = df.columns.str.strip()

    # 2. Dynamic Label Parser, Standardization, and EXACT EDA CLEANUP
    label_cols = [c for c in df.columns if 'label' in c.lower()]
    if label_cols:
        df['target_label'] = df[label_cols[0]].astype(str).str.strip()
        df['target_label'] = df['target_label'].str.replace('- Attempted', '', regex=False).str.strip()
    else:
        df['target_label'] = 'BENIGN'

    # 3. Flexible Network Metadata Routing for Dashboard Display
    src_ip_col   = [c for c in df.columns if 'src ip' in c.lower() or 'source ip' in c.lower()]
    dst_ip_col   = [c for c in df.columns if 'dst ip' in c.lower() or 'destination ip' in c.lower()]
    dst_port_col = [c for c in df.columns if 'dst port' in c.lower() or 'destination port' in c.lower()]

    df['display_src_ip']   = df[src_ip_col[0]]   if src_ip_col   else '192.168.1.100'
    df['display_dst_ip']   = df[dst_ip_col[0]]   if dst_ip_col   else '10.0.0.5'
    df['display_dst_port'] = df[dst_port_col[0]] if dst_port_col else '443'

    # 4. Drop Socket/Tracking Columns
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

    # 5. Type Standardization & Infinity Value Resolution
    meta_cols = {'target_label', 'display_src_ip', 'display_dst_ip', 'display_dst_port'}
    for col in df.columns:
        if col not in meta_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())

    df.fillna(0.0, inplace=True)

    return df

# 5. CORE WORKSPACE AND ENGINE EXECUTION PIPELINE
class ProductionNIDSWorkspace:

    def __init__(self):
        self.xgb_model, self.label_encoder = load_xgb_pipeline_components()
        self.ae_config, self.ae_scaler, self.pytorch_engine = load_ae_pipeline_components()

        if 'active_thresh' not in st.session_state:
            if self.ae_config and "threshold" in self.ae_config:
                st.session_state.active_thresh = float(self.ae_config["threshold"])
            else:
                st.session_state.active_thresh = 0.616377

        # Ingestion trackers
        if 'stream_active'           not in st.session_state: st.session_state.stream_active           = False
        if 'current_index'           not in st.session_state: st.session_state.current_index           = 0
        if 'simulation_finished'     not in st.session_state: st.session_state.simulation_finished     = False

        # XGBoost metrics registers
        if 'total_attacks_injected'   not in st.session_state: st.session_state.total_attacks_injected   = 0
        if 'correctly_predicted'      not in st.session_state: st.session_state.correctly_predicted      = 0
        if 'model_misclassifications' not in st.session_state: st.session_state.model_misclassifications = 0
        if 'completely_missed'        not in st.session_state: st.session_state.completely_missed        = 0
        if 'latest_inference_latency' not in st.session_state: st.session_state.latest_inference_latency = 0.0

        # Autoencoder metrics registers
        if 'ae_total_anomalies_detected' not in st.session_state: st.session_state.ae_total_anomalies_detected = 0
        if 'ae_correct_alerts'           not in st.session_state: st.session_state.ae_correct_alerts           = 0
        if 'ae_false_alarms'             not in st.session_state: st.session_state.ae_false_alarms             = 0
        if 'ae_missed_attacks'           not in st.session_state: st.session_state.ae_missed_attacks           = 0

        # Latent space / clustering
        if 'latent_space_history' not in st.session_state: st.session_state.latent_space_history = []
        if 'cluster_id_history'   not in st.session_state: st.session_state.cluster_id_history   = []

        # ── K-Means batch counter (throttle re-clustering to every 5 batches) ──
        if 'kmeans_batch_counter' not in st.session_state: st.session_state.kmeans_batch_counter = 0

        # Label arrays for classification reports
        if 'ae_true_labels_history' not in st.session_state: st.session_state.ae_true_labels_history = []
        if 'ae_pred_labels_history' not in st.session_state: st.session_state.ae_pred_labels_history = []
        if 'y_true_list'            not in st.session_state: st.session_state.y_true_list            = []
        if 'y_pred_list'            not in st.session_state: st.session_state.y_pred_list            = []

        # Seen labels and score buckets
        if 'unique_labels_seen'              not in st.session_state: st.session_state.unique_labels_seen              = set()
        if 'benign_sse_scores'               not in st.session_state: st.session_state.benign_sse_scores               = []
        if 'attack_sse_scores'               not in st.session_state: st.session_state.attack_sse_scores               = []

        # Time-series histories
        if 'flow_duration_history'         not in st.session_state: st.session_state.flow_duration_history         = []
        if 'fwd_packets_history'           not in st.session_state: st.session_state.fwd_packets_history           = []
        if 'anomaly_reconstruction_scores' not in st.session_state: st.session_state.anomaly_reconstruction_scores = []
        if 'threat_distribution'           not in st.session_state: st.session_state.threat_distribution          = {}

        # Hybrid decision log
        if 'hybrid_logs' not in st.session_state:
            st.session_state.hybrid_logs = pd.DataFrame(
                columns=['Log Index', 'Source IP', 'Destination IP', 'Dst Port',
                         'True Label', 'XGBoost Prediction', 'DAE Alert',
                         'Reconstruction Error', 'Hybrid Decision']
            )
        # Hybrid classification lists
        if 'hybrid_true_list'  not in st.session_state: st.session_state.hybrid_true_list  = []
        if 'hybrid_pred_list'  not in st.session_state: st.session_state.hybrid_pred_list  = []

        # Log ledgers
        if 'visible_logs' not in st.session_state:
            st.session_state.visible_logs = pd.DataFrame(
                columns=['Log Index', 'Source IP', 'Destination IP', 'Dst Port', 'True Label', 'XGBoost Prediction']
            )
        if 'ae_visible_logs' not in st.session_state:
            st.session_state.ae_visible_logs = pd.DataFrame(
                columns=['Log Index', 'Source IP', 'Destination IP', 'Dst Port', 'True Label', 'Reconstruction Error', 'DAE Guard Alert']
            )

    def build_console_ui(self):
        st.sidebar.markdown("## Control Dashboard Configuration")
        st.sidebar.markdown("---")

        uploaded_file = st.sidebar.file_uploader("Upload Network Traffic Dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            dataset_workspace  = clean_and_parse_input_stream(uploaded_file)
            total_dataset_rows = len(dataset_workspace)
            st.sidebar.success(f"File Staged: {total_dataset_rows:,} rows.")
        else:
            st.sidebar.info("Awaiting CSV data asset...")
            dataset_workspace  = None
            total_dataset_rows = 0

        st.sidebar.markdown("---")
        batch_size    = st.sidebar.slider("Batch Window Size", 5000, 10000, 5000, step=10)
        tick_interval = st.sidebar.slider("Console Frame Ingestion Delay", 0.00, 1.00, 0.00, step=0.01)

        def start_matrix_run(): st.session_state.stream_active = True
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
        st.markdown("---")

        status_box = st.empty()
        if st.session_state.simulation_finished:
            status_box.markdown(
                '<div class="status-container"><div>🟢Analysis Completed: iteration terminated.</div></div>',
                unsafe_allow_html=True)
        elif st.session_state.stream_active:
            status_box.markdown(
                f'<div class="status-container" style="border-left-color:#f5a623;"><div>'
                f'🟡Active Scan Underway: Processing row array block: {st.session_state.current_index:,} / {total_dataset_rows:,}'
                f'</div></div>', unsafe_allow_html=True)
        else:
            status_box.markdown(
                '<div class="status-container" style="border-left-color:#768390;"><div>'
                '🟢System Online: Standing by for data initialization...</div></div>',
                unsafe_allow_html=True)

        tab_results, tab_eda = st.tabs([
            "🛡️ Hybrid NIDS",
            "📊 Exploratory Data Analysis"
        ])

        # ── HYBRID NIDS: XGBoost ops → AE ops → Combined Ledger → Combined Report ──
        with tab_results:

            # ══════════════════════════════════════════════
            # SECTION 1: SUPERVISED XGBOOST MODULE
            # ══════════════════════════════════════════════
            st.markdown("## Supervised XGBoost Module")
            st.markdown("---")

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

            draw_xgb_view()

            # ══════════════════════════════════════════════
            # SECTION 2: UNSUPERVISED AUTOENCODER MODULE
            # ══════════════════════════════════════════════
            st.markdown("---")
            st.markdown("## Denoising Autoencoder Module")
            st.markdown("---")

            ae_param_c1, _, __ = st.columns([2, 2, 2])
            with ae_param_c1:
                current_clusters_count = max(2, min(len(st.session_state.unique_labels_seen), 12))
                st.metric("K-Means Clusters", f"{current_clusters_count} Active Clusters")

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
                    st.markdown("#### K-Means Partitioning Matrix Cluster Assignments")
                    if st.session_state.cluster_id_history:
                        st.line_chart(st.session_state.cluster_id_history[:tot:20])

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

            # ══════════════════════════════════════════════
            # SECTION 3: HYBRID DETECTION LEDGER
            # One table: XGBoost + DAE + Hybrid Decision per row
            # ══════════════════════════════════════════════
            st.markdown("---")
            st.markdown("## 📋 Hybrid Model — Detection Ledger")
            st.markdown("---")

            if not st.session_state.hybrid_logs.empty:
                st.dataframe(st.session_state.hybrid_logs,
                             use_container_width=True, hide_index=True, height=300)
            else:
                st.info("Run evaluation to see the hybrid detection ledger.")

            # ══════════════════════════════════════════════
            # SECTION 4: HYBRID CLASSIFICATION REPORT
            # Averaged results of XGBoost + DAE as one report
            # ══════════════════════════════════════════════
            st.markdown("---")
            st.markdown("## 📊 Hybrid Model — Classification Report")
            st.markdown("---")

            hybrid_ready = bool(st.session_state.hybrid_true_list and st.session_state.hybrid_pred_list)

            if hybrid_ready:
                # ── Compute individual reports ──
                xgb_ready = bool(st.session_state.y_true_list and st.session_state.y_pred_list)
                ae_ready  = bool(st.session_state.ae_true_labels_history and st.session_state.ae_pred_labels_history)

                # XGBoost report — multi-class, binary True/False per row
                if xgb_ready:
                    xgb_is_attack_pred = pd.Series(st.session_state.y_pred_list).str.upper() != 'BENIGN'
                    xgb_is_attack_true = pd.Series(st.session_state.y_true_list).str.upper() != 'BENIGN'
                    xgb_bin_rep = pd.DataFrame(classification_report(
                        xgb_is_attack_true.astype(int).tolist(),
                        xgb_is_attack_pred.astype(int).tolist(),
                        labels=[0, 1], target_names=['BENIGN', 'ATTACK'],
                        output_dict=True, zero_division=0
                    )).transpose()
                else:
                    xgb_bin_rep = None

                # DAE report
                if ae_ready:
                    unique_classes = sorted(set(st.session_state.ae_true_labels_history) |
                                            set(st.session_state.ae_pred_labels_history))
                    tnames = ['BENIGN', 'ATTACK'] if len(unique_classes) == 2 else (
                             ['BENIGN'] if unique_classes == [0] else ['ATTACK'])
                    ae_bin_rep = pd.DataFrame(classification_report(
                        st.session_state.ae_true_labels_history,
                        st.session_state.ae_pred_labels_history,
                        labels=unique_classes, target_names=tnames,
                        output_dict=True, zero_division=0
                    )).transpose()
                else:
                    ae_bin_rep = None

                # Hybrid report — from combined decisions
                hybrid_rep = pd.DataFrame(classification_report(
                    st.session_state.hybrid_true_list,
                    st.session_state.hybrid_pred_list,
                    labels=[0, 1], target_names=['BENIGN', 'ATTACK'],
                    output_dict=True, zero_division=0
                )).transpose()

                # ── Average precision/recall/f1 across both models row-by-row ──
                metric_cols = ['precision', 'recall', 'f1-score']
                frames_to_avg = [f for f in [xgb_bin_rep, ae_bin_rep] if f is not None]

                if len(frames_to_avg) == 2:
                    avg_rep = frames_to_avg[0].copy()
                    for col in metric_cols:
                        if col in avg_rep.columns:
                            avg_rep[col] = (
                                frames_to_avg[0][col].fillna(0) +
                                frames_to_avg[1][col].fillna(0)
                            ) / 2
                    avg_rep['support'] = frames_to_avg[0]['support'].fillna(0)
                    avg_rep.index.name = 'Class'
                else:
                    avg_rep = frames_to_avg[0].copy() if frames_to_avg else hybrid_rep.copy()

                # ── Display: averaged report + hybrid decision report ──
                st.markdown("#### Averaged Report (XGBoost + DAE)")
                st.caption("Precision, Recall, F1 averaged across both models per class")
                st.dataframe(avg_rep.round(3), use_container_width=True)

                st.markdown("#### Hybrid Decision Report")
                st.caption("Based on the 4-rule hybrid decision logic applied to each row")
                st.dataframe(hybrid_rep.round(3), use_container_width=True)
            else:
                st.info("Run evaluation to see the hybrid classification report.")

        # TAB 2: GLOBAL EXPLORATORY DATA ANALYSIS (EDA) MODULE
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
                sns.barplot(x=label_series.index, y=label_series.values, palette='crest', ax=ax1)
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


        # 6. INGESTION WORKLOAD AND PARALLEL BRANCH EVALUATION PIPELINE
        if st.session_state.stream_active and dataset_workspace is not None:
            if st.session_state.current_index < total_dataset_rows:
                start         = st.session_state.current_index
                end           = min(start + batch_size, total_dataset_rows)
                working_chunk = dataset_workspace.iloc[start:end].copy()

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

                # ── STEP A: XGBoost supervised predictions ──
                if self.xgb_model is not None:
                    X_xgb = xgb_source.copy()
                    if hasattr(self.xgb_model, "feature_names_in_"):
                        for mc in [c for c in self.xgb_model.feature_names_in_ if c not in X_xgb.columns]:
                            X_xgb[mc] = 0.0
                        X_xgb = X_xgb[self.xgb_model.feature_names_in_]
                    t_0 = time.perf_counter()
                    xgb_preds_encoded = self.xgb_model.predict(X_xgb)
                    xgb_preds_text    = self.label_encoder.inverse_transform(xgb_preds_encoded)
                    st.session_state.latest_inference_latency = (
                        (time.perf_counter() - t_0) * 1000.0) / len(working_chunk)
                else:
                    xgb_preds_text = ['BENIGN'] * len(working_chunk)

                # ── STEP B: Autoencoder unsupervised predictions (MSE) ──
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

                for vector in latent_vectors:
                    st.session_state.latent_space_history.append(vector)

                for raw_label in working_chunk['target_label'].unique():
                    st.session_state.unique_labels_seen.add(raw_label)

                history_depth = len(st.session_state.latent_space_history)

                # min=2 (KMeans hard floor), max=12 (CICIDS2017 ceiling)
                # A 4-class dataset → 4 clusters; full CICIDS2017 → 12 clusters
                KMEANS_CLUSTERS    = max(2, min(len(st.session_state.unique_labels_seen), 12))
                RECLUSTER_INTERVAL = 5

                st.session_state.kmeans_batch_counter += 1

                if history_depth >= KMEANS_CLUSTERS and (
                    st.session_state.kmeans_batch_counter % RECLUSTER_INTERVAL == 0
                    or len(st.session_state.cluster_id_history) == 0
                ):
                    # Full refit on accumulated latent history
                    cluster_model        = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=4)
                    full_cluster_lineage = cluster_model.fit_predict(
                        np.array(st.session_state.latent_space_history))
                    st.session_state.cluster_id_history = full_cluster_lineage.tolist()
                else:
                    # Extend with last known assignment — no refit cost this batch
                    last_id = st.session_state.cluster_id_history[-1] if st.session_state.cluster_id_history else 0
                    st.session_state.cluster_id_history.extend([last_id] * len(working_chunk))

                # ── STEP D: VECTORIZED METRICS & HYBRID DECISION LOGIC ──

                chunk_len = len(working_chunk)

                # --- Clean string arrays ---
                actual_threats_s = working_chunk['target_label'].fillna('BENIGN').astype(str).str.strip()
                xgb_preds_s      = pd.Series(xgb_preds_text, dtype=str).str.strip()

                actual_threats  = actual_threats_s.values
                xgb_preds_arr   = xgb_preds_s.values
                sips            = working_chunk['display_src_ip'].fillna('0.0.0.0').astype(str).values
                dips            = working_chunk['display_dst_ip'].fillna('0.0.0.0').astype(str).values
                ports           = working_chunk['display_dst_port'].fillna('0').astype(str).values
                recon_scores    = chunk_errors

                # --- Core masks ---
                xgb_is_attack_mask = (xgb_preds_s.str.upper() != 'BENIGN').values
                ae_alert_mask      = recon_scores > st.session_state.active_thresh
                is_attack_mask     = (actual_threats_s.str.upper() != 'BENIGN').values
                true_binary_arr    = is_attack_mask.astype(int)
                ae_alert_arr       = ae_alert_mask.astype(int)

                # ── 4-RULE HYBRID DECISION LOGIC ──
                # Rule 1: XGB=ATTACK + AE=ATTACK  → ATTACK (both agree)
                # Rule 2: XGB=BENIGN + AE=BENIGN  → BENIGN (both agree)
                # Rule 3: XGB=BENIGN + AE=ATTACK  → ZERO-DAY THREAT (AE-only signal)
                # Rule 4: XGB=ATTACK + AE=BENIGN  → ATTACK (XGBoost's call is trusted
                #         as sufficient confirmation on its own; no secondary AE check)
                rule1_both_attack =  xgb_is_attack_mask &  ae_alert_mask
                rule2_both_benign = ~xgb_is_attack_mask & ~ae_alert_mask
                rule3_zero_day    = ~xgb_is_attack_mask &  ae_alert_mask
                rule4_xgb_only    =  xgb_is_attack_mask & ~ae_alert_mask

                hybrid_labels = np.where(rule1_both_attack, '⚠️ ATTACK',
                               np.where(rule3_zero_day,     '🔴 ZERO-DAY THREAT',
                               np.where(rule4_xgb_only,     '⚠️ ATTACK (XGB Confirmed)',
                                                             '🟢 BENIGN')))

                # Hybrid binary: attack if Rule 1, Rule 3, or Rule 4 (anything not both-benign)
                hybrid_pred_binary = (rule1_both_attack | rule3_zero_day | rule4_xgb_only).astype(int)

                # --- XGBoost counters ---
                correct_mask    = (xgb_preds_arr == actual_threats)
                missed_mask     = (~correct_mask) & is_attack_mask & (~xgb_is_attack_mask)
                st.session_state.total_attacks_injected   += int(is_attack_mask.sum())
                st.session_state.correctly_predicted      += int(correct_mask.sum())
                st.session_state.model_misclassifications += int((~correct_mask).sum())
                st.session_state.completely_missed        += int(missed_mask.sum())

                # --- AE counters ---
                tp_mask = ae_alert_mask & is_attack_mask
                fp_mask = ae_alert_mask & ~is_attack_mask
                fn_mask = ~ae_alert_mask & is_attack_mask
                st.session_state.ae_total_anomalies_detected += int(ae_alert_mask.sum())
                st.session_state.ae_correct_alerts           += int(tp_mask.sum())
                st.session_state.ae_false_alarms             += int(fp_mask.sum())
                st.session_state.ae_missed_attacks           += int(fn_mask.sum())

                # --- Flow / fwd packet histories ---
                flow_dur_cols = [c for c in working_chunk.columns if 'flow duration' in c.lower()]
                fwd_pkt_cols  = [c for c in working_chunk.columns if 'total fwd packet' in c.lower()]
                flow_dur_vals = working_chunk[flow_dur_cols[0]].fillna(0.0).values.tolist()                                 if flow_dur_cols else [0.0] * chunk_len
                fwd_pkt_vals  = working_chunk[fwd_pkt_cols[0]].fillna(0.0).values.tolist()                                 if fwd_pkt_cols else [0.0] * chunk_len
                st.session_state.flow_duration_history.extend(flow_dur_vals)
                st.session_state.fwd_packets_history.extend(fwd_pkt_vals)
                st.session_state.anomaly_reconstruction_scores.extend(recon_scores.tolist())

                # --- SSE buckets ---
                st.session_state.attack_sse_scores.extend(recon_scores[is_attack_mask].tolist())
                st.session_state.benign_sse_scores.extend(recon_scores[~is_attack_mask].tolist())

                # --- Individual model histories ---
                st.session_state.y_true_list.extend(actual_threats.tolist())
                st.session_state.y_pred_list.extend(xgb_preds_arr.tolist())
                st.session_state.ae_true_labels_history.extend(true_binary_arr.tolist())
                st.session_state.ae_pred_labels_history.extend(ae_alert_arr.tolist())

                # --- Hybrid histories ---
                st.session_state.hybrid_true_list.extend(true_binary_arr.tolist())
                st.session_state.hybrid_pred_list.extend(hybrid_pred_binary.tolist())

                # --- Threat distribution ---
                pred_series = pd.Series(xgb_preds_arr)
                for label, count in pred_series.value_counts().items():
                    st.session_state.threat_distribution[label] = (
                        st.session_state.threat_distribution.get(label, 0) + int(count))

                # --- Build individual logs ---
                master_indices = np.arange(start + 1, start + chunk_len + 1, dtype=int)

                xgb_log_df = pd.DataFrame({
                    'Log Index'         : master_indices,
                    'Source IP'         : sips,
                    'Destination IP'    : dips,
                    'Dst Port'          : ports,
                    'True Label'        : actual_threats,
                    'XGBoost Prediction': xgb_preds_arr,
                })

                ae_alert_labels = np.where(ae_alert_mask, "⚠️ Anomaly", "🟢 NORMAL")
                ae_log_df = pd.DataFrame({
                    'Log Index'           : master_indices,
                    'Source IP'           : sips,
                    'Destination IP'      : dips,
                    'Dst Port'            : ports,
                    'True Label'          : actual_threats,
                    'Reconstruction Error': [f"{v:.6f}" for v in recon_scores],
                    'DAE Guard Alert'     : ae_alert_labels,
                })

                # --- Hybrid ledger ---
                hybrid_log_df = pd.DataFrame({
                    'Log Index'           : master_indices,
                    'Source IP'           : sips,
                    'Destination IP'      : dips,
                    'Dst Port'            : ports,
                    'True Label'          : actual_threats,
                    'XGBoost Prediction'  : xgb_preds_arr,
                    'DAE Alert'           : ae_alert_labels,
                    'Reconstruction Error': [f"{v:.6f}" for v in recon_scores],
                    'Hybrid Decision'     : hybrid_labels,
                })

                st.session_state.visible_logs = pd.concat(
                    [st.session_state.visible_logs, xgb_log_df],
                    ignore_index=True)
                st.session_state.ae_visible_logs = pd.concat(
                    [st.session_state.ae_visible_logs, ae_log_df],
                    ignore_index=True)
                st.session_state.hybrid_logs = pd.concat(
                    [st.session_state.hybrid_logs, hybrid_log_df],
                    ignore_index=True)

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


# 7. RUNTIME ENVIRONMENT APPLICATION ENTRYPOINT
if __name__ == "__main__":
    console_app = ProductionNIDSWorkspace()
    console_app.build_console_ui()