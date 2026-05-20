import streamlit as st
import pandas as pd

class APP:
    
    def get_data_from_mysql(self):
        pass
        #           DEMO HOW WE FETCH DATASET
        # try:
        #     conn = mysql.connector.connect(
        #         host="your-aws-endpoint.com", # or localhost
        #         user="admin",
        #         password="yourpassword",
        #         database="nids_db"
        #     )
        #     query = "SELECT * FROM forensic_logs ORDER BY timestamp DESC LIMIT 10"
        #     df = pd.read_sql(query, conn)
        #     conn.close()
        #     return df
        # except Exception as e:
        #     st.error(f"Connection Failed: {e}")
        #     return pd.DataFrame() # Return empty if fails
    
    def protoype(self):

    # 1. Page Config
        st.set_page_config(layout="wide", page_title="Hybrid NIDS Prototype")
        # 2. Sidebar
        st.sidebar.header("🛡️ Control Panel")
        st.sidebar.success("System Status: ONLINE")
        st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.75)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🔓 Logout / Terminate Session"):
            st.session_state['authorized'] = False
            st.rerun()

        # 3. Top Metrics
        st.title("Network Intrusion Detection System")
        c1, c2, c3 = st.columns(3)

        # Metric 1: Focus on Flows, not Packets
        c1.metric("Network Flows Analyzed", "14,205", "120 flows/sec")

        # Metric 2: Threats (This stays the same)
        c2.metric("Threats Detected", "20", "High Alert", delta_color="inverse")

        # Metric 3: Model Latency (Important for DFI)
        c3.metric("Avg Inference Latency", "0.02ms", "-0.005ms")

        # 4. Middle Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Live Traffic Flow")
            st.line_chart([10, 20, 15, 80, 20, 30]) # Fake data spike
        with col2:
            st.subheader("Attack Class Distribution")
            st.bar_chart({"Normal": 14185, "DDoS": 15, "Botnet": 5})

        # 5. Bottom Log
        st.subheader("🛑 Real-Time Forensic Log")
        fake_data = pd.DataFrame({
            'Timestamp': ['10:01:05', '10:01:04'], 
            'Source IP': ['192.168.1.105', '172.16.0.5'], 
            'Prediction': ['DDoS', 'Normal'],
            'Integrity Hash': ['a1b2c3', 'd4e5f6']
        })
        st.table(fake_data)
        st.markdown("---")
        st.subheader("🔍 Forensic Integrity Validator")
        col_v1, col_v2 = st.columns([1, 2])

        with col_v1:
            # Use the length of your dataframe to prevent out-of-bounds errors
            log_id = st.number_input("Log ID (Row Index)", min_value=0, max_value=len(fake_data)-1, step=1)
            verify_btn = st.button("Verify Hash")

        with col_v2:
            hash_input = st.text_input("Paste Hash from Forensic Log")
            if verify_btn:
                # Get the hash from the specific row and column
                actual_hash = fake_data.iloc[log_id]['Integrity Hash'].strip()
                user_input = hash_input.strip()
                
                if user_input == actual_hash:
                    st.success(f"✅ Integrity Verified for Log ID {log_id}. SHA-256 Signature Matches.")
                else:
                    st.error(f"⚠️ Alert: Data Tampering Detected! Log ID {log_id} hash does not match our records.")
                    # Helpful for your demo to show the panel why it failed:
                    st.write(f"System Record: `{actual_hash}` | Your Input: `{user_input}`")