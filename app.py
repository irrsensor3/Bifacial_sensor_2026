import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Import grouped UI & backend helpers from the new module
from ui_sections import (
    supabase,
    login,
    plot_weather_signals,
    preview_report_content,
    generate_word_report,
    generate_pdf_report,
    get_force_log_status,
    set_force_log_status,
    fetch_latest_readings,
    fetch_recent_alerts,
    fetch_latest_panel_readings,
)

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
st.sidebar.write(f"Role: {st.session_state.user_role.capitalize()}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.auth = False
    st.session_state.user_role = None
    st.rerun()

st.title("📊 Bifacial PV Data Logging System")

# -------------------------
# File upload / report flow
# -------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

report_title = st.text_input("Report Title", "Bifacial PV Performance Report")
observation = st.text_area("Observation Notes")

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(100))

    st.subheader("📌 Dataset Info")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    time = df["Time"] if "Time" in df.columns else df.index

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    st.subheader("📈 Graph Configuration")

    selected_temps = st.multiselect(
        "Select Temperature Columns",
        numeric_cols,
        default=[c for c in numeric_cols if "temp" in c.lower()]
    )

    selected_irradiance = st.multiselect(
        "Select Irradiance Columns",
        numeric_cols,
        default=[c for c in numeric_cols if "irr" in c.lower()]
    )

    temperatures = {col: df[col].tolist() for col in selected_temps}
    irradiances = {col: df[col].tolist() for col in selected_irradiance}

    if selected_temps or selected_irradiance:
        fig = plot_weather_signals(time, temperatures, irradiances)
        st.pyplot(fig)

        st.subheader("📄 Generate Reports")

        if st.button("👁️ Preview Report"):
            st.session_state.show_preview = True

        if st.session_state.get("show_preview"):
            with st.expander("📋 Report Preview", expanded=True):
                preview_report_content(df, report_title, observation, fig)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                report = generate_word_report(df, report_title, observation, fig)
                st.download_button(
                    label="⬇️ Download Word Report",
                    data=report,
                    file_name="PV_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            with col2:
                report = generate_pdf_report(df, report_title, observation, fig)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=report,
                    file_name="PV_Report.pdf",
                    mime="application/pdf"
                )

    if st.button("Test Supabase"):
        supabase.table("pi_commands").update({"command": "hello"}).eq("id", 1).execute()
        st.success("Database updated!")

# -------------------------
# Admin controls
# -------------------------
if st.session_state.user_role == "admin":
    st.divider()
    st.subheader("🔴 Admin Controls")

    if st.button("🔄 Reboot Raspberry Pi"):
        supabase.table("pi_commands").update({"command": "reboot"}).eq("id", 1).execute()
        st.success("Reboot command sent.")

    if st.button("⚫ Shutdown Raspberry Pi"):
        supabase.table("pi_commands").update({"command": "shutdown"}).eq("id", 1).execute()
        st.success("Shutdown command sent.")

    # Force logging
    st.divider()
    st.subheader("🌡️ Force Logging Below 0°C")

    force_log_on = get_force_log_status()

    status_col, button_col = st.columns([1, 2])

    with status_col:
        if force_log_on:
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<div style='width:16px;height:16px;border-radius:50%;"
                "background:#2ecc71;box-shadow:0 0 8px #2ecc71;'></div>"
                "<span><b>FORCE LOG: ON</b></span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<div style='width:16px;height:16px;border-radius:50%;"
                "background:#e74c3c;box-shadow:0 0 8px #e74c3c;'></div>"
                "<span><b>FORCE LOG: OFF</b></span></div>",
                unsafe_allow_html=True,
            )

    with button_col:
        if force_log_on:
            if st.button("⏹️ Stop Force Logging (resume sub-zero cutoff)"):
                if set_force_log_status(False):
                    st.success("Force logging disabled — sub-zero readings will be discarded again.")
                    st.rerun()
                else:
                    st.error("Couldn't reach Supabase — try again.")
        else:
            if st.button("▶️ Force Logging (ignore sub-zero cutoff)"):
                if set_force_log_status(True):
                    st.success("Force logging enabled — the Pi will log sub-zero readings as-is.")
                    st.rerun()
                else:
                    st.error("Couldn't reach Supabase — try again.")

    st.caption(
        "When ON, the Pi keeps logging every sensor's temperature even if it "
        "reads below 0°C, instead of discarding it as invalid. The Pi only "
        "checks this once a minute, so it can take up to ~60s to take effect."
    )
else:
    st.divider()
    st.info("ℹ️ Admin controls are not available in guest mode.")

# -------------------------
# Live sensor data
# -------------------------
st.divider()
st.subheader("Live Sensor Data")

refresh_seconds = st.slider("Auto-refresh interval (seconds)", 5, 60, 15)
st_autorefresh(interval=refresh_seconds * 1000, key="live_refresh")

df_live = fetch_latest_readings()

if df_live.empty:
    st.info("No live data yet — waiting for the Pi to push a sample.")
else:
    latest = df_live.iloc[-1]
    st.caption(f"Last update: {latest.get('date', '')} {latest.get('time', '')}")

    irr_cols = [c for c in df_live.columns if c.startswith("Irr_")]
    temp_cols = [c for c in df_live.columns if c.startswith("Temp_")]

    # quick "latest values" snapshot
    cols = st.columns(4)
    for i, col in enumerate(irr_cols[:4]):
        val = latest[col]
        cols[i].metric(col, f"{val:.1f} W/m²" if val is not None else "—")

    selected_live_irr = st.multiselect(
        "Irradiance sensors to plot (live)", irr_cols, default=irr_cols[:1]
    )
    if selected_live_irr:
        st.line_chart(df_live.set_index("created_at")[selected_live_irr])

    with st.expander("Raw live data table"):
        st.dataframe(df_live, use_container_width=True, hide_index=True)

    # Sensors below 0°C
    st.markdown("### 🌡️ Sensors Below 0°C")

    df_alerts = fetch_recent_alerts()

    if df_alerts.empty:
        st.success("No sub-zero alerts recorded — all sensors logging normally.")
    else:
        latest_per_sensor = df_alerts.sort_values("created_at").groupby("sensor_id").tail(1)
        now_utc = pd.Timestamp.now(tz="UTC")
        currently_invalid = latest_per_sensor[
            now_utc - latest_per_sensor["created_at"] < pd.Timedelta(seconds=150)
        ]

        if currently_invalid.empty:
            st.success("No sensors currently below 0°C.")
        else:
            st.error(f"{len(currently_invalid)} sensor(s) currently below 0°C and not logging:")
            for _, row in currently_invalid.sort_values("sensor_id").iterrows():
                st.markdown(
                    f"🔴 **Sensor {int(row['sensor_id'])}** — {row['temp_c']}°C "
                    f"@ {row['created_at'].strftime('%H:%M:%S')} "
                    f"(bus {row['bus']}, addr {row['address']})"
                )

        with st.expander("Recent alert history"):
            st.dataframe(df_alerts, use_container_width=True, hide_index=True)

# -------------------------
# Live panel meter data
# -------------------------
st.divider()
st.subheader("Live Panel Meter Data")

df_panel = fetch_latest_panel_readings()

if df_panel.empty:
    st.info("No live panel meter data yet — waiting for the mini PC to push a sample.")
else:
    latest_per_device = (
        df_panel.sort_values("created_at").groupby("device_id").tail(1).sort_values("device_id")
    )
    latest_overall = df_panel.iloc[-1]
    st.caption(f"Last update: {latest_overall['created_at']}")

    for _, row in latest_per_device.iterrows():
        device_label = f"Meter {int(row['device_id'])}"
        has_error = row.get("error") not in (None, "No error")

        status_dot = "🔴" if has_error else "🟢"
        cols = st.columns(5)
        cols[0].markdown(f"**{status_dot} {device_label}**")
        cols[1].metric("Voltage", f"{row['voltage_v']:.1f} V" if row["voltage_v"] is not None else "—")
        cols[2].metric("Current", f"{row['current_a']:.2f} A" if row["current_a"] is not None else "—")
        cols[3].metric("Power", f"{row['active_power_kw']:.3f} kW" if row["active_power_kw"] is not None else "—")
        cols[4].metric("Energy", f"{row['forward_energy_kwh']:.1f} kWh" if row["forward_energy_kwh"] is not None else "—")

        if has_error:
            st.caption(f"⚠️ {device_label}: {row.get('error')}")

    st.markdown("### Trend")
    device_ids = sorted(df_panel["device_id"].dropna().unique().tolist())
    selected_devices = st.multiselect(
        "Meters to plot", device_ids, default=device_ids[:1], format_func=lambda d: f"Meter {int(d)}"
    )
    metric_choice = st.radio(
        "Metric", ["voltage_v", "current_a", "active_power_kw"],
        format_func=lambda m: {"voltage_v": "Voltage (V)", "current_a": "Current (A)", "active_power_kw": "Power (kW)"}[m],
        horizontal=True,
    )

    if selected_devices:
        pivot = df_panel[df_panel["device_id"].isin(selected_devices)].pivot_table(
            index="created_at", columns="device_id", values=metric_choice
        )
        pivot.columns = [f"Meter {int(c)}" for c in pivot.columns]
        st.line_chart(pivot)

    with st.expander("Raw panel meter data table"):
        st.dataframe(df_panel, use_container_width=True, hide_index=True)
