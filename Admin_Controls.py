import streamlit as st

from ui_sections import (
    require_login,
    page_stamp,
    supabase,
    get_forced_sensors,
    set_sensor_force,
    NUM_SENSORS,
)


def render_admin_controls():
    require_login(role="admin")

    page_stamp("Admin Controls")
    st.title("Admin controls")

    # -------------------------
    # Power controls
    # -------------------------
    st.subheader("Power controls")
    st.caption(
        "These act on the logger. A shutdown needs someone on site to power the "
        "Pi back on, and logging stops until they do."
    )

    def _send_command(command: str, label: str):
        """Write a command for the Pi to pick up, reporting failure rather than
        raising. An exception here used to dump a stack trace over the page and
        leave you unsure whether the command had been sent."""
        try:
            supabase.table("pi_commands").update(
                {"command": command}
            ).eq("id", 1).execute()
            st.success(f"{label} sent. The Pi checks about once a minute.")
        except Exception as exc:
            st.error(f"{label} was NOT sent — the database write failed: {exc}")

    # Two-step confirmation. These fired on a single click before, so one
    # mis-click took the logger offline with no way to cancel.
    for cmd, label, warning in (
        ("reboot", "Reboot",
         "The logger stops for about a minute, then comes back on its own."),
        ("shutdown", "Shut down",
         "The logger stops and stays off until someone powers it on physically."),
    ):
        pending = f"_confirm_{cmd}"
        with st.container(border=True):
            st.markdown(f"**{label} the Raspberry Pi**")
            st.caption(warning)
            if not st.session_state.get(pending):
                if st.button(f"{label} the logger", key=f"{cmd}_start"):
                    st.session_state[pending] = True
                    st.rerun()
            else:
                st.warning(f"Confirm: {label.lower()} the logger now?")
                yes, no = st.columns(2)
                with yes:
                    if st.button(f"Yes, {label.lower()} now", key=f"{cmd}_yes",
                                 type="primary", use_container_width=True):
                        _send_command(cmd, f"{label} command")
                        st.session_state[pending] = False
                with no:
                    if st.button("Cancel", key=f"{cmd}_no", use_container_width=True):
                        st.session_state[pending] = False
                        st.rerun()

    # -------------------------------------------------
    # FORCE LOGGING BELOW 0°C — per sensor (1-24)
    # -------------------------------------------------
    st.divider()
    st.subheader("Logging below 0 °C")
    st.caption(
        "By default a reading below 0 °C is discarded as invalid. Turn a sensor "
        "on here to keep those readings anyway. Sensors set to keep logging are "
        "marked ON and filled; the rest are outlined. The Pi checks this about "
        "once a minute, so a change can take up to a minute to apply."
    )

    forced_sensors = get_forced_sensors()

    # 6 columns x 4 rows = 24 sensor toggle buttons
    SENSORS_PER_ROW = 6
    for row_start in range(1, NUM_SENSORS + 1, SENSORS_PER_ROW):
        row_ids = range(row_start, min(row_start + SENSORS_PER_ROW, NUM_SENSORS + 1))
        cols = st.columns(SENSORS_PER_ROW)
        for col, sid in zip(cols, row_ids):
            is_forced = sid in forced_sensors
            # State is carried by the text and the button variant, not colour
            # alone, so it survives greyscale, colour blindness and readers.
            with col:
                if st.button(
                    f"{sid} · ON" if is_forced else f"{sid}",
                    key=f"force_btn_{sid}",
                    use_container_width=True,
                    type="primary" if is_forced else "secondary",
                    help=(f"Sensor {sid} keeps logging below 0 °C. Select to stop."
                          if is_forced else
                          f"Sensor {sid} discards readings below 0 °C. Select to keep them."),
                ):
                    if set_sensor_force(sid, not is_forced):
                        st.rerun()
                    else:
                        st.error(
                            f"Sensor {sid} was not updated — the database write "
                            f"failed. Check the connection and try again."
                        )

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
    st.subheader("Diagnostics")
    st.caption("Checks the app can reach the database. Does not touch the logger.")

    # The old check wrote command="hello" into pi_commands -- the same row the
    # Pi polls for reboot and shutdown. Writing junk into a command channel to
    # test connectivity is asking for trouble; read instead.
    if st.button("Check database connection"):
        try:
            res = supabase.table("pi_settings").select("id").eq("id", 1).execute()
            if res.data:
                st.success("Connected. Settings row found.")
            else:
                st.warning(
                    "Connected, but pi_settings has no row with id = 1. Sensor "
                    "toggles will not persist until that row exists."
                )
        except Exception as exc:
            st.error(f"Could not reach the database: {exc}")
