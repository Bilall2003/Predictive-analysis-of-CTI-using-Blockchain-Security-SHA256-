# 🛡️ Predictive Analysis of CTI using Blockchain Security (SHA-256)
https://img.shields.io/badge/Status-Milestone%20%232%20(Second%20Last)-orange?style=for-the-badge&logo=git-orange?style=for-the-badge&logo=git)" alt="Milestone"> https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"> https://img.shields.io/badge/XGBoost-222222?style=for-the-badge&logo=xgboost" alt="XGBoost"> https://img.shields.io/badge/Security-SHA--256-green?style=for-the-badge&logo=shield" alt="SHA-256">

An enterprise-grade, hybrid Cyber Threat Intelligence (CTI) framework combining Supervised Machine Learning, Deep Unsupervised Anomaly Detection, Targeted Blockchain SHA-256 Cryptographic Hashing, and Automated NLP Threat Reporting.

---
📌 Project Overview
This project delivers a proactive, explainable, and cryptographically secure Cyber Threat Intelligence (CTI) pipeline designed to identify known network intrusions and zero-day threats in real time.

By evaluating streaming network traffic, the system classifies 14 distinct attack types, flags behavioral anomalies via reconstruction loss, generates automated NLP threat summaries, and immutably logs malicious events using conditional SHA-256 hashing.

--- ## 🚀 Key Architectural Highlights
🤖 Dual-Engine Hybrid Architecture
Supervised XGBoost Engine: Trained across multi-day telemetry (Tuesday–Friday) to classify 14 distinct attack vectors with high precision.
Unsupervised PyTorch Autoencoder: Trained exclusively on baseline benign traffic (Monday) to detect zero-day attacks via reconstruction error scoring (MSE).
🔐 Targeted Blockchain Integrity & NLP
Selective SHA-256 Hashing: Computational resources are optimized by hashing attack traffic instances only for tamper-proof auditing, while non-malicious benign traffic bypasses hashing.
Automated NLP Threat Reports: Textual, human-readable executive summaries detailing threat severity, IP traces, and recommended mitigation protocols.
--- ## 🧠 Machine Learning Strategy & Data Distribution ``` ┌─────────────────────────────────────────┐ │ Incoming Network Packet Stream │ └────────────────────┬────────────────────┘ │ ┌─────────────────┴─────────────────┐ ▼ ▼ ┌───────────────────────────┐ ┌───────────────────────────┐ │ Supervised Path │ │ Unsupervised Path │ │ (XGBoost Engine) │ │ (PyTorch Autoencoder) │ │ Trained: Tue - Fri │ │ Trained: Mon (Benign) │ └─────────────┬─────────────┘ └─────────────┬─────────────┘ │ │ ▼ ▼ Classifies Known Attack Measures Reconstruction Error (14 Attack Classes) (Flags Zero-Day Anomalies) │ │ └─────────────────┬─────────────────┘ │ ▼ ┌─────────────────────────────────┐ │ Attack Detected? (Condition) │ └────────────────┬────────────────┘ │ ┌────────────────────┴────────────────────┐ │ YES │ NO ▼ ▼ ┌───────────────────────────────────┐ ┌───────────────────────────────────┐ │ 1. Compute SHA-256 Integrity Hash │ │ Pass-through (No Hashing Required)│ │ 2. Generate Automated NLP Report │ │ Logged as Normal Operation │ │ 3. Log to Immutable Ledger Table │ └───────────────────────────────────┘ └───────────────────────────────────┘ ``` ### 📂 Dataset Segmentation
Dataset Split	Traffic Profile	Target Engine	Primary Purpose
Monday	Benign Only	Deep Autoencoder	Baseline normal traffic learning & reconstruction threshold calibration.
Tuesday – Friday	Labeled 14 Attack Classes + Benign	XGBoost Classifier	Multi-class attack pattern training & signature mapping.
Unseen Test Batches	Mixed / Zero-Day Scenarios	Hybrid Pipeline	Evaluation of real-time detection, SHA-256 hashing, and NLP text generation.
--- ## ⚔️ Supported Threat Spectrum (14 Attack Classes)
The XGBoost classification engine is trained to detect 14 distinct intrusion categories:

* DoS Attacks: `DoS Hulk`, `DoS GoldenEye`, `DoS Slowloris`, `DoS Slowhttptest` * DDoS Attacks: `DDoS` * Web Attacks: `Web Attack – Brute Force`, `Web Attack – XSS`, `Web Attack – Sql Injection` * Botnet & Malware: `Bot` * Reconnaissance: `PortScan` * System Infiltration: `Infiltration` * Exploits: `Heartbleed` * Brute Force: `FTP-Patator`, `SSH-Patator`
--- ## ⚙️ Data Preprocessing & Feature Engineering
1. Identification Column Removal: Strips non-predictive socket identifiers including Flow ID, Source/Destination IP, Ports, and Timestamp to prevent model overfitting on specific addresses.
2. Robust Cleaning: Trims whitespace from column headers, converts data types to numeric, handles infinite (inf) calculations, and fills missing values using column medians.
3. Feature Normalization: Applies RobustScaler to handle severe numerical outliers in network throughput metrics.
4. Dynamic Target Alignment: Maps clean class labels into encoded vectors for 14-class XGBoost prediction and binary masks (0=Benign,1=Attack) for anomaly evaluations.
--- ## 🛠️ Technology Stack
| Domain | Tools & Frameworks | | :--- | :--- | | **Core Language** | `Python 3.10+` | | **Machine Learning** | `XGBoost`, `Scikit-Learn` | | **Deep Learning** | `PyTorch` (Deep Autoencoders) | | **Data Manipulation** | `Pandas`, `NumPy` | | **Security & Integrity** | `SHA-256 (hashlib)` | | **NLP & Reporting** | Automated Text-Generation Modules | | **Visualization** | `Matplotlib`, `Seaborne`, `Streamlit` / `FastAPI` |
--- ## 📈 Milestone Progress & Future Roadmap - [x] **Milestone 1:** Baseline EDA, Preprocessing Pipeline, and Initial Autoencoder Setup. - [x] **Milestone 2 (Current):** XGBoost multi-day training (Tue–Fri) on **14 attack classes**, Selective **SHA-256 attack-only hashing**, and **NLP Threat Report Generation**. - [ ] **Final Milestone:** Complete dashboard deployment, end-to-end integration testing, real-time packet stream ingestion, and distributed ledger sharing protocols.
