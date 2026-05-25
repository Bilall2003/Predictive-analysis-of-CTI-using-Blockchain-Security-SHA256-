# Predictive Analysis of CTI using Blockchain Security (SHA-256)
**📌 Project Overview**

The primary objective of this project is to design and develop an intelligent, proactive, and explainable Cyber Threat Intelligence (CTI) system capable of detecting, predicting, and analyzing cyber threats in real time. The system focuses on identifying both known attacks and zero-day (unknown) attacks, while ensuring data integrity, security, and trustworthiness using blockchain-inspired hashing (SHA-256).

**The project combines:**

Machine Learning (XGBoost for supervised attack classification)

Deep Learning (Autoencoder for anomaly detection)

Data preprocessing & feature 

Blockchain-inspired SHA-256 hashing for integrity validation

**🚀 Key Features**
🔐 Blockchain-inspired Data Integrity
Uses SHA-256 hashing to ensure data has not been tampered with
Provides trust layer for CTI pipeline
🤖 Hybrid Threat Detection System
XGBoost for known attack classification (DoS, DDoS, Bot, etc.)
Autoencoder for zero-day anomaly detection
⚡ Real-Time Simulation Capability
Evaluates unseen attack scenarios (e.g., Thursday dataset)
Detects deviations from normal network behavior
📊 Explainable Security Analytics
Reconstruction error-based anomaly scoring
Attack vs benign separation visualization
Performance evaluation using Accuracy, F1-score, ROC-AUC
🧠 Machine Learning Approach
1. Supervised Learning (XGBoost)
Trained on multiple days (Tuesday, Wednesday, Friday)
Learns 11 known attack classes
Outputs specific attack types
2. Unsupervised Learning (Autoencoder)
Trained only on benign traffic (Monday dataset)
Learns normal network behavior
Flags deviations as anomalies (zero-day detection)
📂 Dataset Description
Monday: Benign traffic only (used for Autoencoder training)
Tuesday / Wednesday / Friday: Labeled multi-class attack dataset (used for XGBoost training)
Thursday: Unseen zero-day attack dataset (used for testing generalization)

Attack types include:

DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)
DDoS
Bot
PortScan
Heartbleed
Web Attacks (XSS, Brute Force)
Infiltration
⚙️ Data Preprocessing Pipeline
Removal of irrelevant columns:
Flow ID, Source IP, Destination IP, Ports, Timestamp
Handling missing and infinite values
Feature alignment across datasets
Robust scaling (RobustScaler)
Binary conversion for anomaly detection:
0 → Benign
1 → Attack
🧪 Evaluation Strategy
XGBoost:
Accuracy
Precision / Recall / F1-score
Confusion Matrix
Multi-class classification performance
Autoencoder:
Reconstruction error analysis
ROC-AUC score
Threshold-based anomaly detection
Benign vs Attack separation
🧱 Blockchain Component (SHA-256)
Each record is hashed using SHA-256
Ensures:
Data integrity
Tamper detection
Trustworthy CTI pipeline
Supports secure threat intelligence logging
📊 System Architecture
Raw Network Traffic Data
Data Cleaning & Preprocessing
Feature Engineering & Scaling
Dual Model Pipeline:
XGBoost → Attack classification
Autoencoder → Anomaly detection
Prediction Layer
Security Analytics Dashboard / Output
🧠 Key Insight
XGBoost performs well on known attacks but struggles with zero-day threats.
Autoencoder successfully detects unknown attacks based on reconstruction error.
The hybrid system improves overall CTI robustness.
📌 Technologies Used
Python
Pandas, NumPy
Scikit-learn
XGBoost
PyTorch
Matplotlib, Seaborn
SHA-256 (hashlib)
Jupyter Notebook / Google Colab
📈 Future Improvements
Integration with real-time network packet capture
Deployment using FastAPI or Streamlit dashboard
Blockchain-based distributed CTI sharing
Transformer-based anomaly detection models
Federated learning for multi-source threat intelligence
👨‍💻 Author

Cyber Threat Intelligence Research Project
Focused on AI-based intrusion detection and secure cyber analytics.
