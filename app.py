import streamlit as st
import pandas as pd

from ui_sections import (
    login,
    inject_theme,
    solar_day_bar,
    split_irradiance,
    rear_sensor_for_meter,
    latest_per_column,
    hero,
    feature_cards,
    array_diagram,
    plot_gauge,
    fetch_latest_readings,
    fetch_latest_panel_readings,
    system_log_panel,
)
from Data_and_Reports import render_data_reports
from Live_Monitoring import render_live_monitoring
from Admin_Controls import render_admin_controls
from Irradiance_Tracker import render_irradiance_tracker
from Panel_Array import render_panel_array
from Anomalies import render_anomalies

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

if st.sidebar.button("Sign out", width="stretch"):
    st.session_state.auth = False
    st.session_state.user_role = None
    st.rerun()

system_log_panel() 
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
    """The banner every page opens with: photo, headline, live figures.

    This replaced two Plotly gauges and a floating photo. The gauges took a
    third of the screen to show one number each, and the number was the only
    part anyone read -- so the numbers are now the design, at a size you can
    take in from across a desk.
    """
    # 200 rows, not 5: the logger leaves most cells blank in any given row, so a
    # handful of rows can miss a sensor's latest reading entirely and show a
    # live channel as silent.
    df_live = fetch_latest_readings(limit=200)
    # 200 rows, not 5: the diagram needs at least one recent reading per meter,
    # and the logger polls all 20 in sequence. With only 5 rows most panels
    # would render unlit and the array would look half-dead.
    df_panel = fetch_latest_panel_readings(limit=200)

    front_irr, rear_irr, n_front, n_rear = split_irradiance(df_live)
    power = _average_power(df_panel)

    sensors_live = 0
    if not df_live.empty:
        irr_cols = [c for c in df_live.columns if c.startswith("Irr_")]
        if irr_cols:
            sensors_live = int(
                df_live[irr_cols].apply(pd.to_numeric, errors="coerce")
                .iloc[-1].notna().sum()
            )

    last_seen = "—"
    for frame in (df_live, df_panel):
        if not frame.empty and "created_at" in frame.columns:
            stamp = pd.to_datetime(frame["created_at"], errors="coerce").max()
            if pd.notna(stamp):
                last_seen = stamp.strftime("%H:%M")
                break

    hero(
        title="Rooftop array, live",
        subtitle=(
            "Sixteen bifacial panels in four rows, each row bracketed by a "
            "front reference sensor at either end. Irradiance on both faces, "
            "module temperature, and panel meter output — sampled continuously."
        ),
        stats=[
            # Front and rear kept apart: the difference between them is the
            # measurement this project exists to make, and one combined average
            # erases it. The live counts matter too, since not every front
            # sensor has been reporting.
            ("Front irradiance",
             "—" if front_irr != front_irr else f"{front_irr:,.0f}",
             f"W/m² · {n_front} of 8", "warm"),
            ("Rear irradiance",
             "—" if rear_irr != rear_irr else f"{rear_irr:,.0f}",
             f"W/m² · {n_rear} of 16", "cool"),
            ("Panel power", f"{power:,.2f}", "kW", "cool"),
            ("Last sample", last_seen, "", None),
        ],
    )

    # The array itself, lit by what each panel is producing. This is the thing
    # worth looking at: a dark panel in a bright row is obvious at a glance,
    # which no table of numbers manages.
    power = {}
    if not df_panel.empty and {"device_id", "active_power_kw"} <= set(df_panel.columns):
        latest = (df_panel.sort_values("created_at")
                  .groupby("device_id").tail(1))
        for _, r in latest.iterrows():
            dev = pd.to_numeric(r.get("device_id"), errors="coerce")
            val = pd.to_numeric(r.get("active_power_kw"), errors="coerce")
            if pd.notna(dev):
                power[int(dev)] = float(val) * 1000 if pd.notna(val) else float("nan")
    # Front reference sensors come from the Pi (Irr_1 .. Irr_24), not from the
    # meters. While that logger is silent the rings stay hollow rather than
    # showing zero -- "no sensor" and "no sunlight" must not look alike.
    # Each sensor's own most recent reading, not the last row -- the logger
    # leaves most cells in a row blank, so reading one row showed eight working
    # sensors as one.
    front = {}
    if not df_live.empty:
        got = latest_per_column(df_live, [f"Irr_{s}" for s in
                                          (1, 6, 7, 12, 13, 18, 19, 24)])
        front = {int(k.split("_")[1]): v for k, v in got.items()}

    # Rear irradiance per panel, so each cell shows what the panel is making
    # AND the light reaching its back face -- the two numbers that together
    # explain bifacial performance.
    rear = {}
    if not df_live.empty:
        pairs = rear_sensor_for_meter()
        got = latest_per_column(df_live, [f"Irr_{s}" for s in pairs.values()])
        for meter, sensor in pairs.items():
            v = got.get(f"Irr_{sensor}")
            if v is not None:
                rear[meter] = v

    array_diagram(power, unit="W", title="Live output, panel by panel",
                  front=front, rear=rear)

    if df_live.empty:
        feature_cards([
            ("☀", "Front sensors", "Eight references, two per row, reading the "
             "direct beam on the panel face.", "warm"),
            ("◐", "Rear sensors", "Sixteen under-panel sensors reading light "
             "reflected off the roof — the bifacial gain.", "cool"),
            ("⌁", "Panel meters", "Voltage, current, power and cumulative "
             "energy per string.", "cool"),
            ("◷", "Continuous log", "Samples pushed to the database and synced "
             "to Drive for reporting.", "warm"),
        ])


# -------------------------
# Pages — passed as functions, not file paths, so there's no path resolution
# to break no matter what the files are named or where they live.
#
# Ordered by how often they're opened: the live view is what someone checks on
# arrival, reports are occasional. The first entry is also the landing page.
# -------------------------
pages = [
    st.Page(render_live_monitoring, title="Live Monitoring", icon="📡"),
    st.Page(render_panel_array, title="Panel Array", icon="🔲"),
    st.Page(render_anomalies, title="Anomalies", icon="⚠️"),
    st.Page(render_irradiance_tracker, title="Irradiance Tracker", icon="📈"),
    st.Page(render_data_reports, title="Data & Reports", icon="📁"),
]

if st.session_state.user_role == "admin":
    pages.append(st.Page(render_admin_controls, title="Admin Controls", icon="🛠️"))
     with st.sidebar.expander("Debug: fetch errors"):
        for key in ("_fetch_error_system_logs", "_fetch_error_panel_readings"):
            st.code(st.session_state.get(key, "no error recorded"))

nav = st.navigation(pages)

# Shown above every page: an empty chart means one of two very different
# things, and this says which before you read a single number.
render_overview(expanded=(nav.title != "Live Monitoring"))
solar_day_bar()

nav.run()
