import streamlit as st
import pandas as pd

from ui_sections import (
    require_login,
    page_stamp,
    fetch_latest_readings,
    fetch_recent_alerts,
    get_sensor_logging_config,
    set_sensor_logging_mode,
)

# =========================
# SENSOR LAYOUT — EDIT THIS to match your actual wiring
# =========================
# 4 rows of 4 panels each, arranged 2x2 to match the physical array.
# Each row has 4 rear (under-panel) sensors + 2 front (short-end) sensors
# = 6 per row x 4 rows = 24, matching NUM_SENSORS in Admin_Controls.py.
#
# NOTE: these IDs are still the placeholder sequential guess from the original
# file. Verify against the physical wiring before trusting this page — the
# ratio check at the bottom of the page will flag an obvious mismatch.
SENSOR_LAYOUT = [
    {"row": 1, "position": "Top-Left",     "front_a": 1,  "panel_sensors": [2, 3, 4, 5],     "front_b": 6},
    {"row": 2, "position": "Top-Right",    "front_a": 7,  "panel_sensors": [8, 9, 10, 11],   "front_b": 12},
    {"row": 3, "position": "Bottom-Left",  "front_a": 13, "panel_sensors": [14, 15, 16, 17], "front_b": 18},
    {"row": 4, "position": "Bottom-Right", "front_a": 19, "panel_sensors": [20, 21, 22, 23], "front_b": 24},
]

# Rear positions 1 and 4 sit at the row ends and see more ground-reflected
# light than the interior pair. Surfacing that here means the reading you see
# is interpretable without knowing the rig.
EDGE_POSITIONS = (1, 4)


def _irr_column_for(sensor_id: int) -> str:
    """Which Irr_ column corresponds to this physical sensor position."""
    return f"Irr_{sensor_id}"


# --------------------------------------------------------------------------
#  Styling. Kept to what the page actually needs: a visible keyboard focus
#  ring, a reading strip under each sensor, and respect for reduced motion.
# --------------------------------------------------------------------------
_PANEL_CSS = """
<style>
  .stButton > button:focus-visible {
      outline: 3px solid #0B6E4F;
      outline-offset: 2px;
  }
  .sensor-meta {
      font-variant-numeric: tabular-nums;
      font-size: 0.78rem;
      line-height: 1.15;
      margin: 0.1rem 0 0.55rem 0.15rem;
      opacity: 0.85;
  }
  .sensor-meta .unit { opacity: 0.6; }
  .irr-bar {
      height: 4px;
      border-radius: 2px;
      background: currentColor;
      opacity: 0.35;
      margin: 0.15rem 0 0.5rem 0.15rem;
  }
  .row-caption {
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      opacity: 0.6;
      margin-bottom: 0.35rem;
  }
  @media (prefers-reduced-motion: reduce) {
      * { animation: none !important; transition: none !important; }
  }
</style>
"""


def _read_irradiance(df_live: pd.DataFrame, sensor_id: int):
    """Latest irradiance for one sensor, or None when the column is absent."""
    col = _irr_column_for(sensor_id)
    if df_live.empty or col not in df_live.columns:
        return None
    val = pd.to_numeric(df_live.iloc[-1][col], errors="coerce")
    return float(val) if pd.notna(val) else None


def _last_subzero_alert(df_alerts: pd.DataFrame, sensor_id: int):
    """Most recent sub-zero alert for one sensor as (temp_c, when) or None.

    Supabase returns created_at as a string unless it has been parsed, and the
    original code called .strftime() on it directly, which raises. Coerce first
    and fall back to the raw value rather than crashing the page.
    """
    if df_alerts.empty or "sensor_id" not in df_alerts.columns:
        return None
    ids = pd.to_numeric(df_alerts["sensor_id"], errors="coerce")
    rows = df_alerts[ids == sensor_id]
    if rows.empty:
        return None
    when = pd.to_datetime(rows["created_at"], errors="coerce")
    rows = rows.assign(_when=when).sort_values("_when")
    last = rows.iloc[-1]
    stamp = last["_when"]
    stamp_txt = stamp.strftime("%Y-%m-%d %H:%M") if pd.notna(stamp) else str(last["created_at"])
    return last.get("temp_c"), stamp_txt


def render_panel_array():
    require_login()  # no role= — guests reach this page same as admins
    page_stamp("Panel Array")
    st.markdown(_PANEL_CSS, unsafe_allow_html=True)

    is_admin = st.session_state.get("user_role") == "admin"

    st.title("Panel array")
    st.caption(
        "Each strip below is one physical row: a front sensor at each short "
        "end, four rear sensors under the panels between them. Values are the "
        "latest irradiance reading. Select a sensor for its full detail."
    )

    df_live = fetch_latest_readings(limit=1)
    df_alerts = fetch_recent_alerts()
    # Force-log state is admin-only, end to end — guests never fetch it, so
    # there's nothing forced_sensors-shaped for a guest session to leak.
    forced_sensors = set(get_forced_sensors() or []) if is_admin else set()
    excluded_sensors = set(get_excluded_sensors() or []) if is_admin else set()

    if "selected_sensor" not in st.session_state:
        st.session_state.selected_sensor = None

    # Scale the reading bars against the brightest sensor currently reporting,
    # so the strip shows relative irradiance across the array at a glance.
    all_ids = [sid for cfg in SENSOR_LAYOUT
               for sid in [cfg["front_a"], *cfg["panel_sensors"], cfg["front_b"]]]
    readings = {sid: _read_irradiance(df_live, sid) for sid in all_ids}
    live_vals = [v for v in readings.values() if v is not None]
    peak = max(live_vals) if live_vals else None

    def sensor_tile(sensor_id: int, label: str, face: str, edge: bool = False):
        """One sensor: a button plus its reading. Never colour alone --
        force-logging is spelled out in the label so it survives greyscale,
        colour blindness and a screen reader. Force state only ever appears
        for admins; guests get the reading with no mention of the concept."""
        forced = is_admin and sensor_id in forced_sensors
        excluded = is_admin and sensor_id in excluded_sensors
        selected = st.session_state.selected_sensor == sensor_id
        state_txt = "forced" if forced else ("excluded" if excluded else "")
        marks = " ".join(x for x in (("[*]" if selected else ""), state_txt) if x)
        caption = f"{label}{(' · ' + marks) if marks else ''}"
        

        # primary = front reference, secondary = rear. Shape/weight, not hue.
        if st.button(
            caption,
            key=f"panel_sensor_btn_{sensor_id}",
            use_container_width=True,
            type="primary" if face == "front" else "secondary",
            help=("Front reference sensor at the row end"
                  if face == "front" else
                  f"Rear sensor under panel {label.split()[1]}"
                  + (" (row end, sees more reflected light)" if edge else "")),
        ):
            st.session_state.selected_sensor = sensor_id

        val = readings.get(sensor_id)
        if val is None:
            st.markdown(
                "<div class='sensor-meta'>no reading</div>", unsafe_allow_html=True
            )
        else:
            width = 0 if not peak else max(2, min(100, round(100 * val / peak)))
            st.markdown(
                f"<div class='sensor-meta'>{val:,.1f} <span class='unit'>W/m²</span></div>"
                f"<div class='irr-bar' style='width:{width}%'></div>",
                unsafe_allow_html=True,
            )

    top = st.columns(2)
    bottom = st.columns(2)
    quadrants = {1: top[0], 2: top[1], 3: bottom[0], 4: bottom[1]}

    for cfg in SENSOR_LAYOUT:
        with quadrants[cfg["row"]]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='row-caption'>Row {cfg['row']} — {cfg['position']}</div>",
                    unsafe_allow_html=True,
                )
                # Laid out left-to-right so the screen mirrors the rig: front
                # sensor, four panels, front sensor. The original stacked these
                # vertically, which read as a list rather than a row of panels.
                slots = st.columns([1.2, 1, 1, 1, 1, 1.2])
                with slots[0]:
                    sensor_tile(cfg["front_a"], "Front A", "front")
                for i, sid in enumerate(cfg["panel_sensors"], start=1):
                    with slots[i]:
                        sensor_tile(sid, f"Panel {i}", "rear", edge=i in EDGE_POSITIONS)
                with slots[5]:
                    sensor_tile(cfg["front_b"], "Front B", "front")

    caption = (
        "Bar length is irradiance relative to the highest reading in the array. "
        "Panels 1 and 4 sit at the row ends and see more ground-reflected light "
        "than panels 2 and 3."
    )
    if is_admin:
        caption = (
            "Bar length is irradiance relative to the highest reading in the "
            "array. “forced” means the sensor keeps logging below 0 °C; the "
            "rest use the normal cutoff. Panels 1 and 4 sit at the row ends "
            "and see more ground-reflected light than panels 2 and 3."
        )
    st.caption(caption)

    st.divider()
    _render_detail(df_alerts, readings, forced_sensors, excluded_sensors, is_admin)


def _render_detail(df_alerts, readings, forced_sensors, excluded_sensors, is_admin):
    selected = st.session_state.selected_sensor
    if selected is None:
        st.info("Select a sensor above to see its readings and logging status.")
        return

    face = "rear"
    row_no = pos = None
    for cfg in SENSOR_LAYOUT:
        if selected in (cfg["front_a"], cfg["front_b"]):
            face, row_no = "front", cfg["row"]
        elif selected in cfg["panel_sensors"]:
            face = "rear"
            row_no = cfg["row"]
            pos = cfg["panel_sensors"].index(selected) + 1

    where = f"row {row_no}" if row_no else "unknown row"
    if face == "front":
        where += " · front reference"
    elif pos:
        where += f" · panel {pos}" + (" (row end)" if pos in EDGE_POSITIONS else " (interior)")

    st.subheader(f"Sensor {selected}")
    st.caption(where)

    irr_val = readings.get(selected)
    alert = _last_subzero_alert(df_alerts, selected)

    cols = st.columns(3) if is_admin else st.columns(2)
    cols[0].metric("Irradiance", f"{irr_val:,.1f} W/m²" if irr_val is not None else "No reading")
    cols[1].metric(
        "Last sub-zero reading",
        f"{alert[0]} °C" if alert else "None logged",
        help=f"Recorded {alert[1]}" if alert else "This sensor has never tripped the 0 °C threshold.",
    )

    if is_admin:
        forced = selected in forced_sensors
        excluded = selected in excluded_sensors
        state_label = "Forced on" if forced else ("Force-excluded" if excluded else "Normal cutoff")
        cols[2].metric("Below-zero logging", state_label)

        c1, c2 = st.columns(2)
        with c1:
            action = "Stop forcing" if forced else "Force logging below 0 °C"
            if st.button(action, key=f"panel_array_toggle_force_{selected}", use_container_width=True):
                if set_sensor_force(selected, not forced):
                    st.rerun()
                else:
                    st.error(f"Sensor {selected} was not updated — the write did not go through.")
        with c2:
            action2 = "Stop excluding" if excluded else "Force unlogging"
            if st.button(action2, key=f"panel_array_toggle_exclude_{selected}", use_container_width=True):
                if set_sensor_exclude(selected, not excluded):
                    st.rerun()
                else:
                    st.error(f"Sensor {selected} was not updated — the write did not go through.")
