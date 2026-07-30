import streamlit as st

from ui_sections import login

st.set_page_config(
    page_title="Bifacial PV Data Logging System",
    page_icon="📊",
    layout="wide",
)

# -------------------------
# Authentication / session
# -------------------------
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.user_role = None

if not st.session_state.auth:
    login()
    st.stop()

st.sidebar.subheader("Session")
st.sidebar.write(f"Role: {st.session_state.user_role.capitalize()}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.auth = False
    st.session_state.user_role = None
    st.rerun()

# -------------------------
# Landing page
# -------------------------
st.title("📊 Bifacial PV Data Logging System")
st.write("Pick a page from the sidebar, or jump straight there:")

st.page_link("pages/1_📁_Data_and_Reports.py", label="Data & Reports", icon="📁")
st.caption("Load a CSV from Google Drive, preview it, and generate Word/PDF reports.")

st.page_link("pages/2_📡_Live_Monitoring.py", label="Live Monitoring", icon="📡")
st.caption("Live irradiance readings, sub-zero temperature alerts, and panel meter data.")

if st.session_state.user_role == "admin":
    st.page_link("pages/3_🛠️_Admin_Controls.py", label="Admin Controls", icon="🛠️")
    st.caption("Reboot/shutdown the Pi, force-log sensors below 0°C, and diagnostics.")
