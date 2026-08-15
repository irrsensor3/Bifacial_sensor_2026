import streamlit as st
import pandas as pd

from ui_sections import (
    login,
    inject_theme,
    plot_gauge,
    fetch_latest_readings,
    fetch_latest_panel_readings,
)
from Data_and_Reports import render_data_reports
from Live_Monitoring import render_live_monitoring
from Admin_Controls import render_admin_controls
from Irradiance_Tracker import render_irradiance_tracker

MAX_IRRADIANCE = 1200  # W/m², matches Irradiance_Tracker.py
PANEL_CAPACITY_KW = 10  # adjust to your actual inverter/array capacity

st.set_page_config(
    page_title="Bifacial PV Data Logging System",
    page_icon="📊",
    layout="wide",
)

inject_theme()

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
# Overview strip: live gauges + panel photo — shown above every page
# as a persistent status header
# -------------------------
gauge_col, photo_col = st.columns([2, 1])

with gauge_col:
    # pull a few recent rows (not just the latest one) and forward-fill,
    # since a single row can have gaps where a sensor didn't report
    df_live = fetch_latest_readings(limit=5)
    df_panel = fetch_latest_panel_readings(limit=5)

    irr_value = 0.0
    if not df_live.empty:
        irr_cols = [c for c in df_live.columns if c.startswith("Irr_")]
        if irr_cols:
            recent = df_live[irr_cols].apply(pd.to_numeric, errors="coerce").ffill()
            mean_val = recent.iloc[-1].mean()
            irr_value = float(mean_val) if pd.notna(mean_val) else 0.0

    power_value = 0.0
    if not df_panel.empty and "active_power_kw" in df_panel.columns:
        recent_power = pd.to_numeric(df_panel["active_power_kw"], errors="coerce").ffill()
        last_val = recent_power.iloc[-1]
        power_value = float(last_val) if pd.notna(last_val) else 0.0

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            plot_gauge(irr_value, MAX_IRRADIANCE, "Irradiance", "W/m²", color="#F59E0B"),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            plot_gauge(power_value, PANEL_CAPACITY_KW, "Panel Power", "kW", color="#22C55E"),
            use_container_width=True,
        )

with photo_col:
    st.image(
        "https://commons.wikimedia.org/wiki/Special:FilePath/Solar_roof.jpg",
        caption="Rooftop solar array (stock photo — swap in your own site photo anytime)",
        use_container_width=True,
    )

st.divider()

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
