import streamlit as st

from ui_sections import (
    require_login,
    supabase,
    get_forced_sensors,
    set_sensor_force,
    NUM_SENSORS,
)


def render():
    require_login(role="admin")

    st.title("🛠️ Admin Controls")

    # -------------------------
    # Power controls
    # -------------------------
    st.subheader("🔴 Power Controls")

    if st.button("🔄 Reboot Raspberry Pi"):
        supabase.table("pi_commands").update({"command": "reboot"}).eq("id", 1).execute()
        st.success("Reboot command sent.")

    if st.button("⚫ Shutdown Raspberry Pi"):
        supabase.table("pi_commands").update({"command": "shutdown"}).eq("id", 1).execute()
        st.success("Shutdown command sent.")

    # -------------------------------------------------
    # FORCE LOGGING BELOW 0°C — per sensor (1-24)
    # -------------------------------------------------
    st.divider()
    st.subheader("🌡️ Force Logging Below 0°C")
    st.caption(
        "Toggle individual sensors to keep logging their temperature even if "
        "it reads below 0°C, instead of discarding it as invalid. Green = "
        "forced ON for that sensor. The Pi only checks this once a minute, "
        "so a toggle can take up to ~60s to take effect."
    )

    forced_sensors = get_forced_sensors()

    # 6 columns x 4 rows = 24 sensor toggle buttons
    SENSORS_PER_ROW = 6
    for row_start in range(1, NUM_SENSORS + 1, SENSORS_PER_ROW):
        row_ids = range(row_start, min(row_start + SENSORS_PER_ROW, NUM_SENSORS + 1))
        cols = st.columns(SENSORS_PER_ROW)
        for col, sid in zip(cols, row_ids):
            is_forced = sid in forced_sensors
            label = f"🟢 {sid}" if is_forced else f"🔴 {sid}"
            with col:
                if st.button(label, key=f"force_btn_{sid}", use_container_width=True):
                    if set_sensor_force(sid, not is_forced):
                        st.rerun()
                    else:
                        st.error(f"Couldn't reach Supabase — sensor {sid} not updated.")

    if forced_sensors:
        st.caption(
            "Currently forced: " + ", ".join(str(s) for s in sorted(forced_sensors))
        )
    else:
        st.caption("No sensors currently forced — sub-zero cutoff applies to all.")

    # -------------------------
    # Diagnostics
    # -------------------------
    st.divider()
    st.subheader("🧪 Diagnostics")

    if st.button("Test Supabase"):
        supabase.table("pi_commands").update({"command": "hello"}).eq("id", 1).execute()
        st.success("Database updated!")
