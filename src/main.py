from login import checkin
from app import APP
import streamlit as st

# Create the login object
obj1 = checkin()

# This will show the login screen if not authorized
if obj1.check_password():
    # Only if authorized is True, show the dashboard
    obj2 = APP()
    obj2.protoype()
else:
    # We stay on the login page
    st.info("Please log in to access the NIDS Intelligence Dashboard.")