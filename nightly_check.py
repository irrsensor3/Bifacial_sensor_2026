"""Nightly anomaly check, run by GitHub Actions at midnight Malaysia time.

Does what the "Run detection" button on the Anomalies page does, without
Streamlit: fetches the last few days from Supabase, runs detector.py, emails
any confirmed faults in one summary, and saves the findings to
sensor_anomalies so the dashboard can show them.

Settings come from environment variables (GitHub repository secrets):
    SUPABASE_URL, SUPABASE_KEY          -- same values as in secrets.toml
    SMTP_USER, SMTP_PASSWORD            -- Gmail account + app password
    ALERT_EMAIL_FROM                    -- optional, defaults to SMTP_USER
    ALERT_EMAIL_TO                      -- comma-separated addresses
    DAYS                                -- optional, defaults to 3

Run it by hand with:  python nightly_check.py
"""
import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from supabase import create_client

import detector as D

DAYS = int(os.environ.get("DAYS", "3"))
PAGE_SIZE = 1000
MAX_ROWS = 150_000

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# ---------------------------------------------------------------- fetching --
# Same timestamp-cursor paging as anomalies.py: walking forward from the last
# timestamp seen can't skip rows the way offset paging does while new rows
# are being inserted.
def _fetch_rows(table: str, columns: str, days: int):
    since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    rows, cursor, guard = [], None, 0
    while len(rows) < MAX_ROWS and guard < 2000:
        guard += 1
        q = (client.table(table).select(columns)
             .order("created_at", desc=False).limit(PAGE_SIZE))
        q = q.gt("created_at", cursor.isoformat()) if cursor is not None \
            else q.gte("created_at", since.isoformat())
        batch = q.execute().data or []      # let errors fail the run loudly
        if not batch:
            break
        rows.extend(batch)
        nxt = pd.to_datetime(max(r.get("created_at") for r in batch),
                             errors="coerce", utc=True)
        if pd.isna(nxt):
            break
        nxt = nxt.tz_localize(None)
        if cursor is not None and nxt <= cursor:
            nxt = cursor + pd.Timedelta(milliseconds=1)
        cursor = nxt
        if len(batch) < PAGE_SIZE:
            break
    return rows


def fetch_panel_history(days: int) -> pd.DataFrame:
    rows = _fetch_rows("panel_readings",
                       "created_at,device_id,voltage_v,current_a,"
                       "active_power_kw,forward_energy_kwh", days)
    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["created_at"], errors="coerce",
                             utc=True).dt.tz_localize(None)
    d = d[d.ts.notna()]
    d["dev"] = pd.to_numeric(d["device_id"], errors="coerce")
    d = d[d.dev.notna()]
    d["dev"] = d["dev"].astype(int)

    # A device's k-th reading belongs to poll cycle k; snap each cycle to its
    # first timestamp (see anomalies.py for why no time threshold is used).
    d = d.sort_values("ts").reset_index(drop=True)
    d["cycle"] = d.groupby("dev").cumcount()
    d["ts"] = d.groupby("cycle")["ts"].transform("first")

    frames = []
    for src, short in (("voltage_v", "V"), ("current_a", "I"),
                       ("active_power_kw", "P"), ("forward_energy_kwh", "E")):
        if src not in d.columns:
            continue
        vals = pd.to_numeric(d[src], errors="coerce")
        block = pd.DataFrame({"ts": d.ts.values, "dev": d.dev.values,
                              "val": vals.values})
        wide = block.pivot_table(index="ts", columns="dev", values="val",
                                 aggfunc="last")
        wide.columns = [f"{short}_{int(c)}" for c in wide.columns]
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "Timestamp"
    return out


def fetch_sensor_history(days: int) -> pd.DataFrame:
    rows = _fetch_rows("sensor_readings", "created_at,readings", days)
    if not rows:
        return pd.DataFrame()
    flat = []
    for r in rows:
        rec = r.get("readings") or {}
        if isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except Exception:
                continue
        rec = dict(rec)
        rec["created_at"] = r.get("created_at")
        flat.append(rec)
    d = pd.DataFrame(flat)
    d["ts"] = pd.to_datetime(d["created_at"], errors="coerce",
                             utc=True).dt.tz_localize(None)
    d = d[d.ts.notna()].drop(columns=["created_at"]).set_index("ts").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d.index.name = "Timestamp"
    return d.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------------- email --
def send_summary(confirmed) -> None:
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    if not (user and password):
        print("SMTP_USER / SMTP_PASSWORD not set; skipping email.")
        return

    sender = os.environ.get("ALERT_EMAIL_FROM") or user
    recipients = [a.strip() for a in
                  (os.environ.get("ALERT_EMAIL_TO") or user).split(",")
                  if a.strip()]

    lines = [f"The nightly check of the last {DAYS} day(s) confirmed "
             f"{len(confirmed)} anomaly(ies) on the Bifacial PV array.\n"]
    for i, f in enumerate(confirmed, 1):
        who = f.get("panel") or f.get("subtype") or "unknown"
        lines.append(
            f"{i}. {str(f.get('type', 'unknown')).replace('_', ' ').title()}"
            f" - panel/subtype {who}, severity {f.get('severity', 'unknown')}\n"
            f"   {f.get('detail', 'No details provided.')}\n"
            f"   First seen {f.get('first_day', 'N/A')}, "
            f"last seen {f.get('last_day', 'N/A')}\n")
    lines.append("Log into the monitoring dashboard for more information.")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = (f"⚠️ Solar Array: {len(confirmed)} confirmed "
                      f"anomal{'y' if len(confirmed) == 1 else 'ies'}")
    msg.attach(MIMEText("\n".join(lines), "plain"))

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(sender, recipients, msg.as_string())
    print(f"Emailed {len(recipients)} recipient(s).")


# -------------------------------------------------------------------- main --
def main() -> int:
    print(f"Fetching the last {DAYS} day(s)…")
    wide = fetch_panel_history(DAYS)
    sensors = fetch_sensor_history(DAYS)
    print(f"{len(wide):,} meter samples, {len(sensors):,} sensor samples.")

    if wide.empty:
        print("No meter readings for this period - nothing to analyse.")
        return 1                      # non-zero so GitHub flags the run

    confirmed, provisional = D.run_on_frame(wide)
    if not sensors.empty:
        s_conf, s_prov = D.run_on_frame(sensors)
        confirmed += s_conf
        provisional += s_prov

    print(f"{len(confirmed)} confirmed, {len(provisional)} provisional.")

    if confirmed or provisional:
        try:
            ok = D.push_to_supabase(confirmed, provisional, client=client)
            print("Saved to sensor_anomalies." if ok
                  else "Couldn't save to sensor_anomalies.")
        except Exception as exc:
            print(f"Couldn't save to sensor_anomalies: {exc}")

    if confirmed:
        send_summary(confirmed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
