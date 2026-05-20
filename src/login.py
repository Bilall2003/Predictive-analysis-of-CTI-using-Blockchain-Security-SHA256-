import streamlit as st

class checkin:
    def __init__(self):
        if 'authorized' not in st.session_state:
            st.session_state['authorized'] = False
    def check_password(self):
        """Returns True if the user had the correct password."""
        st.set_page_config(layout="centered")
       
        def password_entered():
            if st.session_state["password"] == "admin123": # Mock password
                st.session_state["authorized"] = True
                del st.session_state["password"]  # Security: don't store password in state
            else:
                st.session_state["authorized"] = False

        if not st.session_state["authorized"]:
            # --- LOGIN INTERFACE ---
            st.title("🛡️ SOC Access Control")
            st.warning("RESTRICTED AREA: Authorized Security Personnel Only")
            st.text_input("Username", key="username")
            st.text_input("Access Key", type="password", key="password", on_change=password_entered)
            
            return False
        else:
            return True

    # # 2. Control Flow
    # if check_password():
    #     # --- ACTUAL DASHBOARD STARTS HERE ---
    #     st.sidebar.success(f"Welcome, {st.session_state['username']}")
    #     st.title("Real-Time Hybrid NIDS Dashboard")
    #     # (Rest of your AI and Flow metrics code goes here)

if __name__=="__main__":
    
    obj=checkin()
    obj.check_password()