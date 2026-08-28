import streamlit as st

from ui_sections import (
    require_login,
    page_stamp,
    supabase,
)

TILT_PANEL_COUNT = 24
DEFAULT_TILT_DEG = 10.0
TILT_TABLE = "panel_tilt_config"


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

    # -------------------------
    # Panel tilt configuration
    # -------------------------
    st.divider()
    st.subheader("Panel tilt configuration")
    st.caption(
        "Tilt angle for each panel. The anomaly detector uses this to decide "
        "which panels are fair peers to compare against each other — panels "
        "within a couple of degrees of one another are treated as the same "
        "tilt, panels tilted noticeably differently are compared separately, "
        "so a deliberate difference in tilt is never reported as a fault."
    )

    render_tilt_config()


def _load_tilt_rows():
    """Current tilt_angle for every configured panel, keyed by panel_id.
    Reported as empty (not raised) on a database problem, same pattern as
    the rest of this page — a read failure here should not crash the form,
    it should just show the defaults."""
    try:
        res = supabase.table(TILT_TABLE).select("panel_id, tilt_angle").execute()
        return {int(r["panel_id"]): float(r["tilt_angle"]) for r in (res.data or [])}
    except Exception as exc:
        st.warning(
            f"Could not load saved tilt values, showing defaults instead: {exc}"
        )
        return {}


def render_tilt_config():
    existing = _load_tilt_rows()

    with st.form("tilt_config_form"):
        default_tilt = st.number_input(
            "Default tilt (°) — prefills any panel not already saved below",
            min_value=0.0, max_value=90.0, value=DEFAULT_TILT_DEG, step=1.0,
            help="Only affects panels that have never been saved. Changing "
                 "this and re-submitting does not overwrite panels you've "
                 "already set individually below.",
        )

        st.caption(f"Panel 1–{TILT_PANEL_COUNT}")

        tilt_inputs = {}
        cols_per_row = 4
        panel_ids = list(range(1, TILT_PANEL_COUNT + 1))
        for i in range(0, len(panel_ids), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for col, pid in zip(row_cols, panel_ids[i:i + cols_per_row]):
                with col:
                    tilt_inputs[pid] = st.number_input(
                        f"Panel {pid}",
                        min_value=0.0, max_value=90.0,
                        value=float(existing.get(pid, default_tilt)),
                        step=1.0,
                        key=f"tilt_panel_{pid}",
                    )

        submitted = st.form_submit_button(
            "Save tilt configuration", type="primary"
        )

    if submitted:
        rows = [
            {"panel_id": pid, "tilt_angle": angle}
            for pid, angle in tilt_inputs.items()
        ]
        try:
            supabase.table(TILT_TABLE).upsert(
                rows, on_conflict="panel_id"
            ).execute()
            st.success(
                f"Saved tilt for {len(rows)} panel(s). The anomaly detector "
                f"picks this up on its next run."
            )
        except Exception as exc:
            st.error(
                f"Tilt configuration was NOT saved — the database write "
                f"failed: {exc}"
            )
            st.caption(
                f"The {TILT_TABLE} table may not exist yet — see "
                f"panel_tilt_config.sql."
            )
