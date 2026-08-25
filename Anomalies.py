import streamlit as st
import pandas as pd
import numpy as np

from ui_sections import require_login, page_stamp, supabase
import pv_gapfill as G
import detector as D
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# How many rows Supabase returns per request. It caps responses, so a week of
# 20 meters at ~1,400 samples/day has to be pulled in pages rather than one go.
# Supabase returns at most 1,000 rows per request, so long periods mean many
# sequential round trips.
#
# Measured, not assumed: the database receives one sample per minute per
# device -- 20 x 1,433 = ~28,700 rows a day, about 29 requests. The CSV files
# on Drive carry the full 11-second feed instead, roughly four times denser,
# which is why a day of CSV is 124,000 rows but a day here is 28,700.
#
# A minute of resolution is ample for detection: it gives 1,433 samples per
# panel per day against the 200 the checks require.
PAGE_SIZE = 1000
MAX_ROWS = 400_000          # hard ceiling, whatever the period asks for

SEVERITY_STYLE = {
    "high": ("🔴", "High"),
    "medium": ("🟠", "Medium"),
    "low": ("🔵", "Low"),
}

FAULT_EXPLAIN = {
    "branch_diode": "Current far below peers while voltage stayed normal — the "
                    "signature of a failed Y-connector diode.",
    "diode": "Voltage at about a third or two thirds of peers while current kept "
             "flowing — a bypass diode inside the module.",
    "low_current": "Producing less current than panels in the same position at "
                   "the same moment.",
    "comparison": "Consistently outside the spread of its own peer group.",
    "disconnection": "Both current and voltage gone — the panel is not connected.",
    "datetime": "A problem with the timestamps themselves rather than a reading.",
    "nocturnal_offset":
    "The irradiance sensor is reporting meaningful light during a period "
    "when solar irradiance should be near zero.",
}

def send_alert_email(subject: str, body: str) -> bool:
    """Sends an anomaly notification email using Streamlit secrets."""
    try:
        missing = []

        if "SMTP_USER" not in st.secrets:
            missing.append("SMTP_USER")

        if "SMTP_PASSWORD" not in st.secrets:
            missing.append("SMTP_PASSWORD")

        if missing:
            st.error(
                f"Missing Streamlit secrets: {', '.join(missing)}"
            )
            return False

        smtp_host = st.secrets.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT", 587))
        smtp_user = st.secrets["SMTP_USER"]
        smtp_password = st.secrets["SMTP_PASSWORD"]

        sender = st.secrets.get("ALERT_EMAIL_FROM", smtp_user)
        recipient = st.secrets.get("ALERT_EMAIL_TO", smtp_user)

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender, [recipient], msg.as_string())

        return True

    except Exception as e:
        st.error(f"Failed to send email alert: {e}")
        return False

def fetch_panel_history(days: int, _progress=None) -> pd.DataFrame:
    """Pull recent panel_readings and reshape to the wide layout the detector
    expects.

    Works backwards from now rather than forwards from a start date, so the
    most useful rows arrive first and a truncated fetch still covers the recent
    period rather than an arbitrary slice of the past.
    """
    since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    rows, cursor, guard = [], None, 0

    # Page by timestamp, not by offset.
    #
    # range(page*1000, ...) counts from the start of the result set, but rows
    # are being inserted while the fetch runs, so every new row shifts the
    # offsets and whole pages get skipped. That returned about 62% of the data,
    # and the rows it missed then looked like gaps in the record -- reporting
    # 1,050 minutes missing where the database itself says 151.
    #
    # Walking forward from the last timestamp seen cannot skip anything,
    # because the cursor is anchored to the data rather than to a position.
    while len(rows) < MAX_ROWS and guard < 2000:
        guard += 1
        q = (supabase.table("panel_readings")
             .select("created_at,device_id,voltage_v,current_a,"
                     "active_power_kw,forward_energy_kwh")
             .order("created_at", desc=False)
             .limit(PAGE_SIZE))
        q = q.gt("created_at", cursor.isoformat()) if cursor is not None \
            else q.gte("created_at", since.isoformat())
        try:
            batch = q.execute().data or []
        except Exception as exc:
            st.error(f"Couldn't read panel_readings: {exc}")
            break
        if not batch:
            break
        rows.extend(batch)
        if _progress is not None:
            _progress(len(rows))

        newest = max(r.get("created_at") for r in batch)
        nxt = pd.to_datetime(newest, errors="coerce", utc=True)
        if pd.isna(nxt):
            break
        nxt = nxt.tz_localize(None)
        # A whole page sharing one timestamp would stall the cursor forever.
        if cursor is not None and nxt <= cursor:
            nxt = cursor + pd.Timedelta(milliseconds=1)
        cursor = nxt
        if len(batch) < PAGE_SIZE:
            break

    if not rows:
        return pd.DataFrame()

    # Report what was actually retrieved. A short fetch looks exactly like a
    # logger outage once the rows are gridded -- missing rows become missing
    # minutes -- so the two have to be told apart before any finding about
    # "missing samples" can be trusted.
    st.session_state["_fetch_rows"] = len(rows)

    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d[d.ts.notna()]
    d["dev"] = pd.to_numeric(d["device_id"], errors="coerce")
    d = d[d.dev.notna()]
    d["dev"] = d["dev"].astype(int)

    # Devices are polled in sequence, so one logical sample is spread over a
    # second or so of wall clock. Snap each burst to its start, or the detected
    # interval collapses and the grid becomes unusable.
    d = d.sort_values("ts")

    # Group each poll burst into one logical sample.
    #
    # No threshold, because every threshold I tried failed on this data. Gap
    # medians sit inside the burst; high percentiles get dragged by mid-burst
    # stalls; a span-based average breaks when the logging rate changes partway
    # through, which yours did (60 s to about 11 s). Those three attempts gave
    # 10,486, 7,407 and 2,889 cycles where the truth was near 9,900.
    #
    # This uses the structure instead: the meter polls each device once per
    # cycle, so a device's k-th reading belongs to cycle k. Exact regardless of
    # cadence, stalls, or rate changes, and still correct when a device drops
    # out entirely -- verified against a simulation of all four cases.
    d = d.sort_values("ts").reset_index(drop=True)
    d["cycle"] = d.groupby("dev").cumcount()
    d["ts"] = d.groupby("cycle")["ts"].transform("first")

    n_cycles = int(d["cycle"].nunique())
    span = (d.ts.max() - d.ts.min()).total_seconds()
    st.session_state["_fetch_cadence"] = round(span / max(n_cycles, 1), 1)

    frames = []
    for src, short in (("voltage_v", "V"), ("current_a", "I"),
                       ("active_power_kw", "P"), ("forward_energy_kwh", "E")):
        if src not in d.columns:
            continue
        vals = pd.to_numeric(d[src], errors="coerce")
        block = pd.DataFrame({"ts": d.ts.values, "dev": d.dev.values, "val": vals.values})
        wide = block.pivot_table(index="ts", columns="dev", values="val", aggfunc="last")
        wide.columns = [f"{short}_{int(c)}" for c in wide.columns]
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "Timestamp"
    return out


def _render_findings(records, empty_message):
    if not records:
        st.success(empty_message)
        return
    for f in records:
        icon, sev = SEVERITY_STYLE.get(f.get("severity", "low"), ("⚪", "Low"))
        panel = f.get("panel")
        title = f"Panel {panel}" if panel else f.get("subtype", "Timestamps")
        with st.container(border=True):
            head, meta = st.columns([3, 1])
            with head:
                st.markdown(f"**{icon} {title} — {f.get('type','').replace('_',' ')}**")
                st.write(f.get("detail", ""))
                why = FAULT_EXPLAIN.get(f.get("type"))
                if why:
                    st.caption(why)
            with meta:
                st.metric("Severity", sev)
                if f.get("days_seen") and f.get("type") != "datetime":
                    st.caption(f"seen on {f['days_seen']} day(s)")
                if f.get("first_day"):
                    st.caption(f"{f['first_day']} → {f['last_day']}")
            with st.expander("Numbers behind this"):
                st.json({k: v for k, v in f.items() if k != "detail"})


def render_anomalies():
    require_login()
    page_stamp("Anomalies")
    st.title("Anomaly detection")
    st.caption(
        "Compares every panel against others in the same position and of the "
        "same type, at the same moment. Weather affects them all equally, so a "
        "panel that stands out is standing out for its own reasons."
    )

    left, right = st.columns([1, 2])
    with left:
        days = st.number_input(
            "Days to analyse", 1, 14, 3,
            help="About 28,700 readings per day, fetched 1,000 at a time — "
                 "roughly 15 seconds per day of data. Three days or more lets "
                 "a finding be confirmed rather than left provisional.")
    with right:
        st.caption(" ")
        go = st.button("Run detection", type="primary", width="stretch")

    if go:
        status = st.status(f"Reading the last {days} day(s)…", expanded=True)
        with status:
            note = st.empty()
            note.write("Fetching from the database in pages of 1,000…")

            def tick(n):
                note.write(f"Fetched {n:,} readings…")

            wide = fetch_panel_history(int(days), _progress=tick)
            note.write(f"Fetched {len(wide):,} samples. Checking every panel…")
            if wide.empty:
                status.update(label="Nothing to analyse", state="error")
                st.warning(
                    "No meter readings came back for that period. Either the "
                    "range is empty or panel_readings is unreachable."
                )
                return
            confirmed, provisional = D.run_on_frame(wide)
            status.update(label=f"Checked {len(wide):,} samples",
                          state="complete", expanded=False)
        st.session_state["_anom"] = (confirmed, provisional, len(wide), int(days))
        # --- NEW EMAIL NOTIFICATION LOGIC ---
        if "sent_alerts" not in st.session_state:
            st.session_state.sent_alerts = set()

        for f in confirmed:
            # Create a unique ID for this specific fault so we don't spam emails on reruns
            identifier = f.get('panel', f.get('subtype', 'unknown'))
            fault_type = f.get('type', 'unknown')
            last_seen = f.get('last_day', 'unknown')
            alert_id = f"{identifier}_{fault_type}_{last_seen}"

            if alert_id not in st.session_state.sent_alerts:
                subject = f"⚠️ Solar Array Alert: {fault_type.replace('_', ' ').title()}"
                body = (
                    f"A confirmed anomaly was detected on the Bifacial PV array.\n\n"
                    f"• Type: {fault_type}\n"
                    f"• Severity: {f.get('severity', 'unknown')}\n"
                    f"• Panel / Subtype: {identifier}\n"
                    f"• Details: {f.get('detail', 'No details provided.')}\n"
                    f"• First seen: {f.get('first_day', 'N/A')}\n"
                    f"• Last seen: {last_seen}\n\n"
                    f"Log into the monitoring dashboard for more information."
                )
                
                if send_alert_email(subject, body):
                    st.session_state.sent_alerts.add(alert_id)

    if "_anom" not in st.session_state:
        st.info("Choose a period and select **Run detection** to check the array.")
        return

    confirmed, provisional, n_rows, used_days = st.session_state["_anom"]

    got = st.session_state.get("_fetch_rows", 0)
    devices = st.session_state.get("_fetch_devices", 20)
    expected = int(used_days * 24 * 60 * devices)   # one sample per device per minute
    if got and got < expected * 0.9:
        st.warning(
            f"Only {got:,} readings came back for {used_days} day(s), where "
            f"about {expected:,} were expected — roughly "
            f"{100*got/expected:.0f}% of the period. Anything reported below as "
            f"“missing samples” may be this short fetch rather than a real "
            f"logger outage. Try fewer days."
        )

    a, b, c = st.columns(3)
    a.metric("Samples checked", f"{n_rows:,}")
    cad = st.session_state.get("_fetch_cadence")
    if cad:
        a.caption(f"one reading per panel every {cad:g}s")
    b.metric("Panel faults", len([f for f in confirmed
                                   if f.get("type") != "datetime"]))
    c.metric("Provisional", len(provisional))

    # Timestamp problems are counted over the whole period rather than per day,
    # so showing them under a "seen on at least N days" heading contradicts
    # itself. They get their own section.
    data_faults = [f for f in confirmed if f.get("type") == "datetime"]
    panel_faults = [f for f in confirmed if f.get("type") != "datetime"]

    if data_faults:
        st.divider()
        st.subheader("Data quality")
        st.caption(
            "Problems with the record itself rather than with a panel — missing, "
            "duplicated or impossible timestamps, counted across the whole period."
        )
        _render_findings(data_faults, "")

    st.divider()
    st.subheader("Confirmed")
    st.caption(
        f"Seen on at least {D.PERSIST_DAYS} separate days. A one-off is usually "
        f"weather, a bird, or a passing cloud; a fault repeats."
    )
    _render_findings(
        panel_faults,
        f"No panel faults confirmed across {used_days} day(s). Either the array "
        f"is healthy, or there aren't enough days yet to establish a pattern."
    )

    if provisional:
        st.divider()
        st.subheader("Provisional")
        st.caption(
            f"Seen on fewer than {D.PERSIST_DAYS} days. Worth watching, not yet "
            f"worth acting on."
        )
        _render_findings(provisional, "")

    st.divider()
    with st.expander("What this will never flag, and why"):
        st.markdown(
            """
**Panels 27–30 running about 7.7% below the rest.** Those are the monofacial
modules in block B5. That gap *is* the bifacial gain — it is the measurement
this project exists to make, not a fault. Panels are only ever compared against
others of the same type.

**Every panel dipping to two-thirds voltage each morning.** At low sun angles
each block shades the one behind it. Panel 14 does this about 3% of the time and
panel 20 twice as often, so a rule like "flag anything below 32 V" would report
the whole array as broken every sunrise. The diode test therefore requires the
behaviour to hold for a large share of the day, which shadows never reach.
            """
        )

    if st.session_state.get("user_role") == "admin" and (confirmed or provisional):
        if st.button("Save these findings to the database"):
            ok = D.push_to_supabase(confirmed, provisional, client=supabase)
            if ok:
                st.success("Saved to sensor_anomalies.")
            else:
                st.error(
                    "Couldn't save. The sensor_anomalies table may not exist "
                    "yet — the CREATE TABLE statement is in ANOMALY_SETUP.md."
                )
