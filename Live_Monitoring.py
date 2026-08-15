import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

from ui_sections import (
    require_login,
    page_stamp,
    fetch_latest_readings,
    fetch_recent_alerts,
    fetch_latest_panel_readings,
)
from drive_fetch import (
    list_available_csvs,
    download_and_combine_csvs,
    extract_year,
    extract_month,
    month_label,
    list_available_dcm_csvs,
    download_and_combine_dcm_csvs,
)


def render_live_monitoring():
    require_login()

    page_stamp("Live Monitoring")
    st.title("📡 Live Monitoring")

    # -------------------------
    # Live sensor data
    # -------------------------
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
        # NOTE: no Temp_ columns land in sensor_readings yet — there's no
        # Supabase table for continuous live temperature logging, only the
        # sub-zero alerts below. Add a temp_cols block here once that
        # table exists.

        # quick "latest values" snapshot
        cols = st.columns(4)
        for i, col in enumerate(irr_cols[:4]):
            val = latest[col]
            cols[i].metric(col, f"{val:.1f} W/m²" if val is not None else "—")

        if "selected_live_irr" not in st.session_state:
            st.session_state.selected_live_irr = irr_cols[:1]
        else:
            st.session_state.selected_live_irr = [c for c in st.session_state.selected_live_irr if c in irr_cols]

        irr_label_col, irr_all_col, irr_none_col = st.columns([4, 1, 1])
        with irr_label_col:
            st.caption("Irradiance sensors to plot (live)")
        with irr_all_col:
            if st.button("Select all", key="live_irr_select_all", use_container_width=True):
                st.session_state.selected_live_irr = irr_cols
        with irr_none_col:
            if st.button("Remove all", key="live_irr_remove_all", use_container_width=True):
                st.session_state.selected_live_irr = []

        # -------------------------
        # Append historical Drive data onto this same chart
        # -------------------------
        with st.expander("🗄️ Append historical data from Drive"):
            available_files = list_available_csvs()
            if not available_files:
                st.caption("No Drive CSVs found to append.")
            else:
                years = sorted({extract_year(f) for f in available_files}, reverse=True)
                yr_col, mo_col = st.columns(2)
                with yr_col:
                    append_year = st.selectbox("Year to append", years, index=0, key="live_append_year_select")
                year_files = [f for f in available_files if extract_year(f) == append_year]
                months = sorted({extract_month(f) for f in year_files})
                with mo_col:
                    append_month = st.selectbox(
                        "Month to append", months, index=len(months) - 1,
                        format_func=month_label, key="live_append_month_select",
                    )
                period_files = [f for f in year_files if extract_month(f) == append_month]
                st.caption(f"{len(period_files)} file(s) found for {month_label(append_month)} {append_year}")

                btn_col, remove_col = st.columns(2)
                with btn_col:
                    if st.button("📥 Append to graph", key="append_year_btn"):
                        file_ids = tuple(f["id"] for f in period_files)
                        with st.spinner(f"Downloading {len(file_ids)} file(s) for {month_label(append_month)} {append_year}..."):
                            df_hist = download_and_combine_csvs(file_ids)
                        st.session_state["_live_append_df"] = df_hist
                        st.session_state["_live_append_year"] = f"{month_label(append_month)} {append_year}"
                        st.success(f"Appended {df_hist.shape[0]} rows from {month_label(append_month)} {append_year}")
                with remove_col:
                    if st.session_state.get("_live_append_df") is not None:
                        if st.button("✖️ Remove appended data", key="remove_append_btn"):
                            st.session_state["_live_append_df"] = None
                            st.session_state["_live_append_year"] = None
                            st.rerun()

                appended_year = st.session_state.get("_live_append_year")
                if appended_year:
                    st.caption(f"Currently appended: {appended_year}")

        selected_live_irr = st.multiselect(
            "Irradiance sensors to plot (live)",
            irr_cols,
            key="selected_live_irr",
            label_visibility="collapsed",
        )
        if selected_live_irr:
            # start with the live data on a proper datetime x-axis
            combined = df_live[["created_at"] + selected_live_irr].copy()
            combined["created_at"] = pd.to_datetime(combined["created_at"])

            df_hist = st.session_state.get("_live_append_df")
            if df_hist is not None and not df_hist.empty:
                hist = df_hist.copy()
                # build a matching timestamp column from whatever
                # Date/Time columns the historical CSV has
                if "Date" in hist.columns and "Time" in hist.columns:
                    hist["created_at"] = pd.to_datetime(
                        hist["Date"].astype(str) + " " + hist["Time"].astype(str),
                        errors="coerce",
                    )
                elif "Time" in hist.columns:
                    hist["created_at"] = pd.to_datetime(hist["Time"], errors="coerce")
                else:
                    hist["created_at"] = pd.NaT

                # a sensor selected live might not exist in the
                # historical file (e.g. it was added later) — fill
                # those with NaN so the columns still line up
                for c in selected_live_irr:
                    if c not in hist.columns:
                        hist[c] = pd.NA

                hist_slice = hist[["created_at"] + selected_live_irr]
                combined = pd.concat([hist_slice, combined], ignore_index=True)

            combined = combined.dropna(subset=["created_at"]).sort_values("created_at")
            # a longer combined time range naturally compresses onto the
            # same chart width — nothing extra needed for that
            st.line_chart(combined.set_index("created_at")[selected_live_irr])

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
        # normalize to a real datetime up front so it combines cleanly
        # with historical Drive data later, and sorts correctly
        df_panel["created_at"] = pd.to_datetime(df_panel["created_at"], errors="coerce")

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

        # -------------------------
        # Append historical DC meter data onto this same chart
        # -------------------------
        with st.expander("🗄️ Append historical DC meter data from Drive"):
            available_dcm_files = list_available_dcm_csvs()
            if not available_dcm_files:
                st.caption("No DC meter CSVs found to append.")
                if st.session_state.get("_dcm_drive_list_error"):
                    with st.expander("Error details"):
                        st.code(st.session_state["_dcm_drive_list_error"])
            else:
                dcm_years = sorted({extract_year(f) for f in available_dcm_files}, reverse=True)
                dcm_yr_col, dcm_mo_col = st.columns(2)
                with dcm_yr_col:
                    append_dcm_year = st.selectbox(
                        "Year to append", dcm_years, index=0, key="dcm_append_year_select"
                    )
                dcm_year_files = [f for f in available_dcm_files if extract_year(f) == append_dcm_year]
                dcm_months = sorted({extract_month(f) for f in dcm_year_files})
                with dcm_mo_col:
                    append_dcm_month = st.selectbox(
                        "Month to append", dcm_months, index=len(dcm_months) - 1,
                        format_func=month_label, key="dcm_append_month_select",
                    )
                dcm_period_files = [f for f in dcm_year_files if extract_month(f) == append_dcm_month]
                st.caption(f"{len(dcm_period_files)} file(s) found for {month_label(append_dcm_month)} {append_dcm_year}")

                dcm_btn_col, dcm_remove_col = st.columns(2)
                with dcm_btn_col:
                    if st.button("📥 Append to graph", key="append_dcm_year_btn"):
                        file_ids = tuple(f["id"] for f in dcm_period_files)
                        with st.spinner(f"Downloading {len(file_ids)} file(s) for {month_label(append_dcm_month)} {append_dcm_year}..."):
                            df_dcm_hist = download_and_combine_dcm_csvs(file_ids)
                        st.session_state["_live_append_dcm_df"] = df_dcm_hist
                        st.session_state["_live_append_dcm_year"] = f"{month_label(append_dcm_month)} {append_dcm_year}"
                        st.success(f"Appended {df_dcm_hist.shape[0]} rows from {month_label(append_dcm_month)} {append_dcm_year}")
                with dcm_remove_col:
                    if st.session_state.get("_live_append_dcm_df") is not None:
                        if st.button("✖️ Remove appended data", key="remove_dcm_append_btn"):
                            st.session_state["_live_append_dcm_df"] = None
                            st.session_state["_live_append_dcm_year"] = None
                            st.rerun()

                appended_dcm_year = st.session_state.get("_live_append_dcm_year")
                if appended_dcm_year:
                    st.caption(f"Currently appended: {appended_dcm_year}")

        # combine live + (optional) historical DC meter data for the
        # trend chart below; everything above (latest-values snapshot)
        # stays live-only on purpose
        df_panel_combined = df_panel
        df_dcm_hist = st.session_state.get("_live_append_dcm_df")
        if df_dcm_hist is not None and not df_dcm_hist.empty:
            needed_cols = [
                "created_at", "device_id", "voltage_v", "current_a",
                "active_power_kw", "forward_energy_kwh", "error",
            ]
            hist_slice = df_dcm_hist.reindex(columns=needed_cols)
            live_slice = df_panel.reindex(columns=needed_cols)
            df_panel_combined = pd.concat([hist_slice, live_slice], ignore_index=True)
            df_panel_combined = df_panel_combined.dropna(subset=["created_at"]).sort_values("created_at")

        st.markdown("### Trend")
        device_ids = sorted(df_panel_combined["device_id"].dropna().unique().tolist())
        if "selected_devices" not in st.session_state:
            st.session_state.selected_devices = device_ids[:1]
        else:
            st.session_state.selected_devices = [d for d in st.session_state.selected_devices if d in device_ids]

        dev_label_col, dev_all_col, dev_none_col = st.columns([4, 1, 1])
        with dev_label_col:
            st.caption("Meters to plot")
        with dev_all_col:
            if st.button("Select all", key="devices_select_all", use_container_width=True):
                st.session_state.selected_devices = device_ids
        with dev_none_col:
            if st.button("Remove all", key="devices_remove_all", use_container_width=True):
                st.session_state.selected_devices = []

        selected_devices = st.multiselect(
            "Meters to plot",
            device_ids,
            key="selected_devices",
            format_func=lambda d: f"Meter {int(d)}",
            label_visibility="collapsed",
        )
        metric_choice = st.radio(
            "Metric", ["voltage_v", "current_a", "active_power_kw"],
            format_func=lambda m: {"voltage_v": "Voltage (V)", "current_a": "Current (A)", "active_power_kw": "Power (kW)"}[m],
            horizontal=True,
        )

        if selected_devices:
            pivot = df_panel_combined[df_panel_combined["device_id"].isin(selected_devices)].pivot_table(
                index="created_at", columns="device_id", values=metric_choice
            )
            pivot.columns = [f"Meter {int(c)}" for c in pivot.columns]
            st.line_chart(pivot)

        with st.expander("Raw panel meter data table"):
            st.dataframe(df_panel, use_container_width=True, hide_index=True)
