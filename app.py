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
from Panel_Array import render_panel_array
from Alerts import render_alerts

MAX_IRRADIANCE = 1200   # W/m², matches Irradiance_Tracker.py
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
# (user_role or 'unknown') — .capitalize() on None raised AttributeError
# whenever the session was half-initialised, e.g. a stale tab left open
# across a redeploy.
st.sidebar.write(f"Signed in as {(st.session_state.user_role or 'unknown').capitalize()}")

if st.sidebar.button("Sign out", use_container_width=True):
    st.session_state.auth = False
    st.session_state.user_role = None
    st.rerun()


# -------------------------
# Overview strip: live gauges + array photo.
#
# Collapsible, and collapsed on Live Monitoring which already shows this data
# in more detail. It costs two Supabase reads on every page load and used to
# push each page's real content below the fold.
# -------------------------
def _average_irradiance(df_live: pd.DataFrame) -> float:
    """Mean irradiance across recent rows and all sensors.

    A single row can have gaps where a sensor did not report, so forward-fill
    before averaging over both axes.
    """
    if df_live.empty:
        return 0.0
    irr_cols = [c for c in df_live.columns if c.startswith("Irr_")]
    if not irr_cols:
        return 0.0
    recent = df_live[irr_cols].apply(pd.to_numeric, errors="coerce").ffill()
    if recent.empty:
        return 0.0
    # The original wrapped this in a bare `except Exception` whose fallback
    # could never run — stack() only fails on an empty frame, handled above.
    # Broad excepts here hid real errors behind a plausible-looking zero.
    mean_val = recent.stack().mean()
    return float(mean_val) if pd.notna(mean_val) else 0.0


def _average_power(df_panel: pd.DataFrame) -> float:
    if df_panel.empty or "active_power_kw" not in df_panel.columns:
        return 0.0
    recent = pd.to_numeric(df_panel["active_power_kw"], errors="coerce").ffill()
    if recent.empty:
        return 0.0
    mean_power = recent.mean()
    if pd.isna(mean_power):
        return 0.0
    mean_power = float(mean_power)
    # Fallback if the column is populated in watts despite its name.
    if mean_power > 1000:
        mean_power /= 1000.0
    return mean_power


def render_overview(expanded: bool):
    with st.expander("Site overview", expanded=expanded):
        gauge_col, photo_col = st.columns([2, 1])

        with gauge_col:
            df_live = fetch_latest_readings(limit=5)
            df_panel = fetch_latest_panel_readings(limit=5)

            irr_value = _average_irradiance(df_live)
            power_value = _average_power(df_panel)

            if df_live.empty and df_panel.empty:
                st.info(
                    "No recent readings. These fill in once the logger pushes "
                    "a sample."
                )

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    plot_gauge(irr_value, MAX_IRRADIANCE,
                               "Average irradiance", "W/m²", color="#B45309"),
                    use_container_width=True,
                )
                # The gauge alone is unreadable to a screen reader, and the
                # number is hard to read off an arc. State it in text too.
                st.caption(f"Mean across all sensors, last 5 samples · {irr_value:,.0f} W/m²")
            with g2:
                st.plotly_chart(
                    plot_gauge(power_value, PANEL_CAPACITY_KW,
                               "Average panel power", "kW", color="#15803D"),
                    use_container_width=True,
                )
                st.caption(f"Mean across meters, last 5 samples · {power_value:,.2f} kW")

        with photo_col:
            # A missing image file raised and took the whole app down on every
            # page, since this block runs before navigation.
            try:
                st.image(
                    "BifacialGrid.jpeg",
                    caption="The bifacial array: four rows of four panels",
                    use_container_width=True,
                )
            except Exception:
                st.caption("Array photo unavailable (BifacialGrid.jpeg not found).")


# -------------------------
# Pages — passed as functions, not file paths, so there's no path resolution
# to break no matter what the files are named or where they live.
#
# Ordered by how often they're opened: the live view is what someone checks on
# arrival, reports are occasional. The first entry is also the landing page.
# -------------------------
pages = [
    st.Page(render_live_monitoring, title="Live Monitoring", icon="📡"),
    st.Page(render_alerts, title="Alerts", icon="🚨"),
    st.Page(render_panel_array, title="Panel Array", icon="🔲"),
    st.Page(render_irradiance_tracker, title="Irradiance Tracker", icon="📈"),
    st.Page(render_data_reports, title="Data & Reports", icon="📁"),
]

if st.session_state.user_role == "admin":
    pages.append(st.Page(render_admin_controls, title="Admin Controls", icon="🛠️"))

nav = st.navigation(pages)

render_overview(expanded=(nav.title != "Live Monitoring"))
st.divider()

nav.run()
