import streamlit as st
import pandas as pd

from ui_sections import (
    require_login,
    page_stamp,
    fetch_latest_readings,
    fetch_recent_alerts,
    get_forced_sensors,
    set_sensor_force,
)

# =========================
# SENSOR LAYOUT — EDIT THIS to match your actual wiring
# =========================
# 4 rows of 4 panels each, arranged 2x2 to match the physical array.
# Each row has 4 rear (under-panel) sensors + 2 front (short-end)
# sensors = 6 per row x 4 rows = 24, matching NUM_SENSORS in
# Admin_Controls.py / ui_sections.py. The IDs below are a placeholder
# sequential guess (Row 1 = 1-6, Row 2 = 7-12, ...) and the row->
# quadrant positions are also a guess — correct both to match your
# real array before relying on this page.
SENSOR_LAYOUT = [
    {"row": 1, "position": "Top-Left",     "front_a": 1,  "panel_sensors": [2, 3, 4, 5],   "front_b": 6},
    {"row": 2, "position": "Top-Right",    "front_a": 7,  "panel_sensors": [8, 9, 10, 11], "front_b": 12},
    {"row": 3, "position": "Bottom-Left",  "front_a": 13, "panel_sensors": [14, 15, 16, 17],"front_b": 18},
    {"row": 4, "position": "Bottom-Right", "front_a": 19, "panel_sensors": [20, 21, 22, 23],"front_b": 24},
]


def _irr_column_for(sensor_id: int) -> str:
    """Which Irr_ column in the live readings corresponds to this
    physical sensor position. Defaults to the same number as the
    sensor_id — edit this if your irradiance channels are numbered
    differently from the 24 temperature/force-log sensor IDs."""
    return f"Irr_{sensor_id}"


def render_panel_array():
    require_login()

    page_stamp("Panel Array")
    st.title("🔲 Panel Array Layout")
    st.caption(
        "Click any sensor below to see its latest readings. Rear sensors "
        "sit under each panel; front sensors sit at each short end of a "
        "row. Temperature shows the most recent sub-zero alert for that "
        "sensor — there's no continuous live temperature feed yet, only "
        "the threshold alert log."
    )

    df_live = fetch_latest_readings(limit=1)
    df_alerts = fetch_recent_alerts()
    forced_sensors = get_forced_sensors()

    if "selected_sensor" not in st.session_state:
        st.session_state.selected_sensor = None

    def sensor_button(sensor_id: int, label: str):
        is_forced = sensor_id in forced_sensors
        dot = "🟢" if is_forced else "⚪"
        if st.button(f"{dot} {label}", key=f"panel_sensor_btn_{sensor_id}", use_container_width=True):
            st.session_state.selected_sensor = sensor_id

    top_cols = st.columns(2)
    bottom_cols = st.columns(2)
    quadrant_cols = {1: top_cols[0], 2: top_cols[1], 3: bottom_cols[0], 4: bottom_cols[1]}

    for row_cfg in SENSOR_LAYOUT:
        with quadrant_cols[row_cfg["row"]]:
            with st.container(border=True):
                st.markdown(f"**Row {row_cfg['row']} — {row_cfg['position']}**")
                sensor_button(row_cfg["front_a"], f"Front A · Sensor {row_cfg['front_a']}")
                for i, sid in enumerate(row_cfg["panel_sensors"], start=1):
                    sensor_button(sid, f"Panel {i} Rear · Sensor {sid}")
                sensor_button(row_cfg["front_b"], f"Front B · Sensor {row_cfg['front_b']}")

    st.caption("🟢 = currently force-logging below 0°C · ⚪ = normal cutoff applies")

    st.divider()

    selected = st.session_state.selected_sensor
    if selected is None:
        st.info("Click a sensor above to see its details.")
        return

    st.subheader(f"Sensor {selected}")

    irr_col = _irr_column_for(selected)
    irr_val = None
    if not df_live.empty and irr_col in df_live.columns:
        val = pd.to_numeric(df_live.iloc[-1][irr_col], errors="coerce")
        irr_val = float(val) if pd.notna(val) else None

    temp_display = "No sub-zero alerts logged"
    if not df_alerts.empty and "sensor_id" in df_alerts.columns:
        sensor_alerts = df_alerts[
            pd.to_numeric(df_alerts["sensor_id"], errors="coerce") == selected
        ].sort_values("created_at")
        if not sensor_alerts.empty:
            last_alert = sensor_alerts.iloc[-1]
            temp_display = (
                f"{last_alert['temp_c']}°C @ "
                f"{last_alert['created_at'].strftime('%Y-%m-%d %H:%M')}"
            )

    is_forced = selected in forced_sensors

    c1, c2, c3 = st.columns(3)
    c1.metric("Irradiance", f"{irr_val:.1f} W/m²" if irr_val is not None else "No data")
    c2.metric("Last sub-zero reading", temp_display)
    c3.metric("Force-log status", "ON" if is_forced else "OFF")

    if st.session_state.user_role == "admin":
        toggle_label = "🔴 Turn force-log OFF" if is_forced else "🟢 Turn force-log ON"
        if st.button(toggle_label, key=f"panel_array_toggle_force_{selected}"):
            if set_sensor_force(selected, not is_forced):
                st.rerun()
            else:
                st.error(f"Couldn't reach Supabase — sensor {selected} not updated.")
    else:
        st.caption("Force-log toggle is admin-only.")
