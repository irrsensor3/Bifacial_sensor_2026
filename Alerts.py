import streamlit as st
import pandas as pd

from ui_sections import (
    require_login,
    page_stamp,
    email_alerts_configured,
    send_alert_email,
    get_seen_log_alert_hashes,
    mark_log_alerts_seen,
    parse_log_alerts,
)
from drive_fetch import (
    list_available_log_files,
    download_log_text,
    device_label_for,
)

LEVEL_ORDER = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}


def render_alerts():
    require_login()

    page_stamp("Alerts")
    st.title("Alerts")
    st.caption(
        "Scans the .log files synced from the Pi and mini PC into the "
        "'panel-meter-logs' Drive folder for ERROR/WARNING/CRITICAL lines. "
        "This only checks when this page is loaded or refreshed — it "
        "isn't a 24/7 background watcher."
    )

    if not email_alerts_configured():
        st.caption(
            "Email/SMS digest isn't configured yet — dashboard-only for now. "
            "See send_alert_email()'s docstring in ui_sections.py for the "
            "secrets needed."
        )

    auto_email = st.toggle(
        "Email/SMS me new alerts when I check this page",
        value=email_alerts_configured(),
        disabled=not email_alerts_configured(),
        key="alerts_auto_email",
    )

    if st.button("🔍 Check for alerts", type="primary"):
        st.session_state["_alerts_checked"] = True
        st.rerun()

    log_files = list_available_log_files()

    if not log_files:
        st.warning(
            "Couldn't find any .log files in Drive — check that the "
            "'panel-meter-logs' folder exists and is shared with the service "
            "account. If your logs live under a different folder name, "
            "update LOG_DRIVE_FOLDER_NAME at the top of drive_fetch.py."
        )
        if st.session_state.get("_log_drive_list_error"):
            with st.expander("Error details"):
                st.code(st.session_state["_log_drive_list_error"])
        return

    devices = sorted({device_label_for(f) for f in log_files})
    st.caption(f"{len(log_files)} log file(s) across {len(devices)} device(s): {', '.join(devices)}")

    # scan every log file and collect matching lines
    all_alerts = []
    for f in log_files:
        text = download_log_text(f["id"])
        if not text:
            continue
        device = device_label_for(f)
        all_alerts.extend(parse_log_alerts(text, f["name"], device))

    if not all_alerts:
        st.success("No ERROR/WARNING/CRITICAL lines found in any log file.")
        return

    seen_hashes = get_seen_log_alert_hashes()
    new_alerts = [a for a in all_alerts if a["hash"] not in seen_hashes]

    # newest-ish first: alerts with a parsed timestamp sort by it, the
    # rest (no timestamp found) trail at the end in file order
    def _sort_key(a):
        return (a["timestamp"] is None, a["timestamp"] or "", LEVEL_ORDER.get(a["level"], 9))
    all_alerts.sort(key=_sort_key, reverse=True)

    # -------------------------
    # Filters
    # -------------------------
    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        level_filter = st.multiselect(
            "Level", ["CRITICAL", "ERROR", "WARNING"],
            default=["CRITICAL", "ERROR", "WARNING"],
        )
    with filt_col2:
        device_filter = st.multiselect("Device", devices, default=devices)

    filtered = [
        a for a in all_alerts
        if a["level"] in level_filter and a["device"] in device_filter
    ]

    # -------------------------
    # Summary
    # -------------------------
    s1, s2, s3 = st.columns(3)
    s1.metric("Total alerts (filtered)", f"{len(filtered):,}")
    s2.metric("New since last check", f"{len([a for a in filtered if a['hash'] not in seen_hashes]):,}")
    s3.metric("Critical/Error", f"{len([a for a in filtered if a['level'] in ('CRITICAL', 'ERROR')]):,}")

    # -------------------------
    # Table
    # -------------------------
    table_rows = [
        {
            "New": "●" if a["hash"] not in seen_hashes else "",
            "Level": a["level"],
            "Device": a["device"],
            "Time": a["timestamp"] or "unknown",
            "File": a["file"],
            "Message": a["message"][:300],
        }
        for a in filtered
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # -------------------------
    # Email digest for genuinely-new alerts + mark them seen
    # -------------------------
    if new_alerts:
        st.info(f"{len(new_alerts)} alert(s) haven't been surfaced before.")
        if auto_email and email_alerts_configured():
            sent, msg = send_alert_email(new_alerts)
            if sent:
                st.success(f"Emailed digest of {len(new_alerts)} new alert(s). {msg}")
            else:
                st.error(f"Couldn't send the alert email: {msg}")
        if mark_log_alerts_seen({a["hash"] for a in new_alerts}):
            pass  # silently recorded; nothing the user needs to act on
        else:
            st.warning(
                "Couldn't record these as seen (Supabase write failed) — "
                "they may be flagged as new again next check. See the "
                "log_alert_state table setup in get_seen_log_alert_hashes()'s "
                "docstring in ui_sections.py."
            )

    if st.session_state.get("user_role") == "admin" and email_alerts_configured():
        st.divider()
        if st.button("✉️ Send test alert email"):
            test_alert = [{
                "level": "ERROR", "device": "test", "file": "test.log",
                "timestamp": None, "message": "This is a test alert from the Alerts page.",
                "hash": "test",
            }]
            sent, msg = send_alert_email(test_alert)
            if sent:
                st.success(msg)
            else:
                st.error(msg)
