import streamlit as st

from ui_sections import login
from Data_and_Reports import render_data_reports
from Live_Monitoring import render_live_monitoring
from Admin_Controls import render_admin_controls
from Irradiance_Tracker import render_irradiance_tracker

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
# Pages — passed as functions, not file paths, so there's no path
# resolution to break no matter what the files are named or where
# they live.
# -------------------------
pages = [
    st.Page(render_data_reports, title="Data & Reports", icon="📁"),
    st.Page(render_live_monitoring, title="Live Monitoring", icon="📡"),
    st.Page(render_irradiance_tracker, title="Irradiance Tracker", icon="📈"),
]

if st.session_state.user_role == "admin":
    pages.append(
        st.Page(render_admin_controls, title="Admin Controls", icon="🛠️")
    )

nav = st.navigation(pages)
nav.run()
