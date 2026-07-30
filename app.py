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
# Pages (built in code, not from filenames — avoids the emoji/
# filename matching bug in st.page_link). Admin Controls only shows
# up in the sidebar at all when logged in as admin.
# -------------------------
pages = [
    st.Page("app_pages/data_reports.py", title="Data & Reports", icon="📁"),
    st.Page("app_pages/live_monitoring.py", title="Live Monitoring", icon="📡"),
]

if st.session_state.user_role == "admin":
    pages.append(
        st.Page("app_pages/admin_controls.py", title="Admin Controls", icon="🛠️")
    )

nav = st.navigation(pages)
nav.run()
