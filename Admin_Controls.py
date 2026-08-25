import streamlit as st

from ui_sections import (
    require_login,
    page_stamp,
    supabase,
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
            # Table is pi_command (singular) — the name the database and the
            # Pi both use. The site previously wrote to "pi_commands", which
            # does not exist, so every power command silently went nowhere
            # while still reporting success.
            supabase.table("pi_command").update(
                {"Command": command}
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
            cmd = supabase.table("pi_command").select("Command").eq("id", 1).execute()
            if res.data and cmd.data:
                pending = (cmd.data[0].get("Command") or "").strip()
                if pending:
                    st.warning(
                        f"Connected, but a '{pending}' command is still queued. "
                        f"Either the Pi has not picked it up yet, or it is not "
                        f"running the version that reads commands."
                    )
                else:
                    st.success("Connected. Settings and command rows found.")
            elif res.data:
                st.warning("Connected, but pi_command has no row with id = 1. "
                           "Power controls will not reach the Pi.")
            else:
                st.warning(
                    "Connected, but pi_settings has no row with id = 1. Sensor "
                    "toggles will not persist until that row exists."
                )
        except Exception as exc:
            st.error(f"Could not reach the database: {exc}")
