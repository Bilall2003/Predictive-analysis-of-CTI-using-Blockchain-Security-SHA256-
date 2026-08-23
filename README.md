<div align="center">

# 🛡️ Predictive Analysis of CTI using Blockchain Security (SHA-256)

### An Explainable, Hybrid Cyber Threat Intelligence System for Known & Zero-Day Attack Detection

<p>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Classifier-orange?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Autoencoder-red?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/SHA--256-Blockchain%20Hashing-2ea44f?style=for-the-badge&logo=blockchaindotcom&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Staging-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-Deployment-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Second%20Last%20Milestone-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Hybrid%20Cascade%20Pipeline-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Attack%20Classes-14-critical?style=flat-square" />
</p>

</div>

---

## 📌 Project Overview

This project designs and develops an **intelligent, proactive, and explainable Cyber Threat Intelligence (CTI) system** capable of detecting, predicting, and analyzing cyber threats in real time. It identifies both **known attacks** and **zero-day (unknown) attacks**, while ensuring **data integrity, security, and trustworthiness** through blockchain-inspired hashing (SHA-256).

The system now implements a full **hybrid cascade pipeline**, combining supervised classification, deep anomaly detection, unsupervised clustering, explainability, and natural language reporting into a single end-to-end CTI workflow.

<div align="center">

**CICIDS2017 → XGBoost → Denoising Autoencoder → KMeans / DBSCAN → SHAP → NLP Report → SHA-256 Hashing → MySQL → Streamlit Dashboard**

</div>

The project combines:

- 🤖 **Machine Learning** — XGBoost for supervised, multi-class attack classification
- 🧠 **Deep Learning** — Denoising Autoencoder (DAE) for zero-day anomaly detection
- 🔍 **Unsupervised Clustering** — KMeans / DBSCAN for grouping anomalous behavior
- 📖 **Explainability** — SHAP for feature-level attack interpretation
- 📝 **NLP Reporting** — Automated natural-language threat summaries
- 🔐 **Blockchain-inspired SHA-256 Hashing** — Integrity validation for attack records
- 💾 **MySQL** — Persistent storage of alerts and threat metadata
- 📊 **Streamlit Dashboard** — Real-time visualization and analyst interface

---

## 🚀 Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🔐 Blockchain-Inspired Data Integrity
- SHA-256 hashing applied **only to flagged attack records** — benign traffic is not hashed, keeping the trust layer focused on threat evidence
- Ensures attack data has not been tampered with post-detection
- Provides a lightweight, auditable trust layer for the CTI pipeline

</td>
<td width="50%" valign="top">

### 🤖 Hybrid Cascade Detection System
- **XGBoost** for known attack classification, now trained across **Tuesday–Friday**, covering **14 attack classes**
- **Autoencoder** for zero-day anomaly detection via reconstruction error
- **KMeans / DBSCAN** for unsupervised clustering of anomalous traffic

</td>
</tr>
<tr>
<td width="50%" valign="top">

### ⚡ Real-Time Simulation Capability
- Evaluates unseen attack scenarios (e.g., held-out Thursday-style traffic)
- Detects deviations from learned normal network behavior
- Cascade design lets known attacks be classified first, with residual anomalies escalated for zero-day analysis

</td>
<td width="50%" valign="top">

### 📊 Explainable Security Analytics
- SHAP-based feature attribution for every flagged event
- Reconstruction error-based anomaly scoring
- **NLP-generated reports** translating model output into human-readable threat summaries
- Attack vs. benign separation visualizations
- Evaluation via Accuracy, F1-score, and ROC-AUC

</td>
</tr>
</table>

---

## 🧠 Machine Learning Approach

<details open>
<summary><b>1️⃣ Supervised Learning — XGBoost</b></summary>
<br>

- Trained on **Tuesday, Wednesday, Thursday, and Friday** datasets
- Now learns **14 known attack classes** (expanded from the original 11)
- Outputs specific attack types for known/labeled traffic

</details>

<details open>
<summary><b>2️⃣ Unsupervised Learning — Autoencoder</b></summary>
<br>

- Trained only on **Monday** benign traffic
- Learns normal network behavior patterns
- Flags high-reconstruction-error samples as anomalies (zero-day detection)

</details>

<details open>
<summary><b>3️⃣ Clustering — KMeans / DBSCAN</b></summary>
<br>

- Groups anomalous samples flagged by the autoencoder
- Helps distinguish structured attack campaigns from noise
- Feeds cluster-level context into the SHAP explainability stage

</details>

<details open>
<summary><b>4️⃣ Explainability & Reporting — SHAP + NLP</b></summary>
<br>

- SHAP values identify which network features drove each detection
- An NLP layer converts SHAP output and detection metadata into a readable **threat intelligence report**

</details>

---

## 📂 Dataset Description

| Day | Content | Used For |
|---|---|---|
| **Monday** | Benign traffic only | Autoencoder training (normal behavior baseline) |
| **Tuesday** | Labeled attack traffic | XGBoost training |
| **Wednesday** | Labeled attack traffic | XGBoost training |
| **Thursday** | Labeled attack traffic | XGBoost training *(now included — previously held out for zero-day testing)* |
| **Friday** | Labeled attack traffic | XGBoost training |

**Attack types covered (14 classes):**

`DoS Hulk` · `DoS GoldenEye` · `DoS Slowloris` · `DoS Slowhttptest` · `DDoS` · `Bot` · `PortScan` · `Heartbleed` · `Web Attack – XSS` · `Web Attack – Brute Force` · `Web Attack – SQL Injection` · `Infiltration` · `FTP-Patator` · `SSH-Patator`

> ⚠️ **Note:** Update the exact 14-class list above if your final label mapping differs — this reflects the expanded Tuesday–Friday training scope.

---

## ⚙️ Data Preprocessing Pipeline

```
Raw CICIDS2017 CSVs
      │
      ├── Remove irrelevant columns (Flow ID, Source/Destination IP, Ports, Timestamp)
      ├── Handle missing & infinite values
      ├── Align features across all daily datasets
      ├── Apply RobustScaler
      └── Binary relabeling for anomaly stage → 0 = Benign, 1 = Attack
```

---

## 🔗 Blockchain-Inspired Integrity Layer (Updated)

<div align="center">

| Traffic Type | SHA-256 Hashed? | Reason |
|:---:|:---:|---|
| ✅ Attack Record | **Yes** | Preserves tamper-proof evidence for confirmed threats |
| ⬜ Benign Record | **No** | Reduces overhead; hashing reserved for actionable threat data |

</div>

Each confirmed attack record — whether flagged by XGBoost (known) or the Autoencoder/clustering stage (zero-day) — is hashed with SHA-256 before being written to MySQL, creating an auditable, tamper-evident trail of threat evidence.

---

## 🧩 Full Hybrid Pipeline Flow

```
CICIDS2017 CSV
      │
      ▼
 ┌─────────────┐
 │  XGBoost    │  → Known attack classification (14 classes)
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │     DAE     │  → Zero-day / anomaly detection
 └─────┬───────┘
       ▼
 ┌─────────────────┐
 │ KMeans / DBSCAN │  → Cluster anomalous behavior
 └─────┬───────────┘
       ▼
 ┌─────────────┐
 │    SHAP     │  → Explainability
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │     NLP     │  → Human-readable threat report
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  SHA-256    │  → Hash attack records only
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │   MySQL     │  → Persistent storage
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  Streamlit  │  → Real-time analyst dashboard
 └─────────────┘
```

---

## 🧠 Key Insight

- XGBoost performs strongly on known attacks but, by design, struggles with truly unseen (zero-day) threats
- The Autoencoder successfully catches unknown attacks based on reconstruction error, filling the gap XGBoost leaves
- Clustering adds structure to anomalies before they reach the explainability layer
- The NLP reporting layer makes SHAP output accessible to non-technical stakeholders
- Selective SHA-256 hashing keeps the integrity layer efficient without sacrificing evidentiary trust
- The hybrid, cascaded system meaningfully improves overall CTI robustness over any single model

---

## 📌 Technologies Used

<p>
  <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/-XGBoost-EB6C1B?style=flat-square" />
  <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/-SHAP-8A2BE2?style=flat-square" />
  <img src="https://img.shields.io/badge/-NLP-006400?style=flat-square" />
  <img src="https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square" />
  <img src="https://img.shields.io/badge/-Seaborn-4C72B0?style=flat-square" />
  <img src="https://img.shields.io/badge/-hashlib%20(SHA--256)-2ea44f?style=flat-square" />
  <img src="https://img.shields.io/badge/-MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white" />
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
</p>

---

## 🚢 Deployment

<table>
<tr>
<td width="50%" valign="top">

### 🐳 Staging — Docker
- Pipeline and Streamlit dashboard containerized for staging
- Ensures a consistent, reproducible environment across local dev and testing
- Used to validate the full hybrid pipeline (XGBoost → DAE → KMeans/DBSCAN → SHAP → NLP → SHA-256 → MySQL → Streamlit) end-to-end before cloud rollout

</td>
<td width="50%" valign="top">

### ☁️ Production — AWS
- Staged Docker image promoted to AWS for deployment
- Cloud hosting for the Streamlit dashboard and MySQL-backed threat store
- Path forward for scaling real-time detection beyond local/staging limits

</td>
</tr>
</table>

<div align="center">

**Flow:** `Local Development` → `Docker (Staging)` → `AWS (Deployment)`

</div>

---

## 📈 Future Improvements

- 🔴 Integration with real-time network packet capture
- 🚀 Full deployment via FastAPI backend + Streamlit dashboard
- ⛓️ Blockchain-based distributed CTI sharing across nodes
- 🧠 Transformer-based anomaly detection models
- 🌐 Federated learning for multi-source threat intelligence

---

<div align="center">

## 👨‍💻 Author

**Cyber Threat Intelligence Research Project**
Final Year Project — Group CS_01

Focused on AI-based intrusion detection and secure, explainable cyber analytics.

</div>
