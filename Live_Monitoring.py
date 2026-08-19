import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

from ui_sections import (
    require_login,
    page_stamp,
    plot_line_chart,
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
    st.title("Live monitoring")

    # -------------------------
    # Live sensor data
    # -------------------------
    st.subheader("Sensors")

    # Auto-refresh is now opt-out. It reruns the whole script, refetching from
    # Supabase and resetting scroll position, which fights anyone reading a
    # chart or filling in the append controls below.
    ref_on, ref_int = st.columns([1, 3])
    with ref_on:
        auto_refresh = st.toggle("Auto-refresh", value=True, key="live_auto_refresh")
    with ref_int:
        refresh_seconds = st.slider(
            "Refresh every (seconds)", 5, 60, 15,
            disabled=not auto_refresh, key="live_refresh_secs",
        )
    if auto_refresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="live_refresh")
    else:
        if st.button("Refresh now"):
            st.rerun()

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
        # f"{val:.1f}" raised on a string and printed "nan" for a missing
        # reading; coerce and show an explicit dash instead.
        def _fmt(val, unit, places=1):
            num = pd.to_numeric(val, errors="coerce")
            return f"{num:,.{places}f} {unit}" if pd.notna(num) else "no reading"

        shown = irr_cols[:4]
        cols = st.columns(max(len(shown), 1))
        for i, col in enumerate(shown):
            cols[i].metric(col.replace("_", " "), _fmt(latest[col], "W/m²"))
        if len(irr_cols) > 4:
            st.caption(
                f"Showing 4 of {len(irr_cols)} sensors. The full set is in the "
                f"chart and the raw table below."
            )

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

            with st.expander("📐 Chart scale (optional)"):
                data_min_t = combined["created_at"].min().to_pydatetime()
                data_max_t = combined["created_at"].max().to_pydatetime()
                if data_min_t < data_max_t:
                    x_start_t, x_end_t = st.slider(
                        "X range (time)", min_value=data_min_t, max_value=data_max_t,
                        value=(data_min_t, data_max_t), key="live_irr_x_range",
                    )
                else:
                    st.caption("Only one timestamp in range — X range slider needs more data.")
                    x_start_t, x_end_t = data_min_t, data_max_t

                irr_chart_auto = st.checkbox("Auto Y-axis", value=True, key="live_irr_y_auto")
                y_min_col, y_max_col = st.columns(2)
                with y_min_col:
                    irr_chart_ymin = st.number_input("Y min (W/m²)", value=0.0, key="live_irr_ymin", disabled=irr_chart_auto)
                with y_max_col:
                    irr_chart_ymax = st.number_input("Y max (W/m²)", value=1200.0, key="live_irr_ymax", disabled=irr_chart_auto)

            fig = plot_line_chart(
                combined, "created_at", selected_live_irr,
                x_range=(x_start_t, x_end_t),
                y_range=None if irr_chart_auto else (irr_chart_ymin, irr_chart_ymax),
                y_title="Irradiance (W/m²)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw readings table"):
            st.caption(f"{len(df_live):,} rows, newest last.")
            st.dataframe(df_live, use_container_width=True, hide_index=True)

        # Sensors below 0°C
        st.markdown("### Sensors below 0 °C")

        df_alerts = fetch_recent_alerts()

        if df_alerts.empty:
            st.success("No sub-zero alerts recorded — all sensors logging normally.")
        else:
            latest_per_sensor = df_alerts.sort_values("created_at").groupby("sensor_id").tail(1)
            # Subtracting a tz-aware "now" from a tz-naive column raises
            # TypeError and blanks the page. Normalise to UTC first.
            stamps = pd.to_datetime(latest_per_sensor["created_at"], errors="coerce", utc=True)
            now_utc = pd.Timestamp.now(tz="UTC")
            recent = (now_utc - stamps) < pd.Timedelta(seconds=150)
            currently_invalid = latest_per_sensor[recent.fillna(False)]

            if currently_invalid.empty:
                st.success("No sensors currently below 0°C.")
            else:
                st.error(f"{len(currently_invalid)} sensor(s) currently below 0°C and not logging:")
                for _, row in currently_invalid.sort_values("sensor_id").iterrows():
                    when = pd.to_datetime(row["created_at"], errors="coerce")
                    when_txt = when.strftime("%H:%M:%S") if pd.notna(when) else "unknown time"
                    st.markdown(
                        f"**Sensor {int(row['sensor_id'])}** — {row['temp_c']} °C "
                        f"at {when_txt} (bus {row.get('bus', '?')}, "
                        f"address {row.get('address', '?')}) — not logging"
                    )

            with st.expander("Recent alert history"):
                st.dataframe(df_alerts, use_container_width=True, hide_index=True)

    # -------------------------
    # Live panel meter data
    # -------------------------
    st.divider()
    st.subheader("Panel meters")

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

            def _m(val, unit, places=1):
                num = pd.to_numeric(val, errors="coerce")
                return f"{num:,.{places}f} {unit}" if pd.notna(num) else "no reading"

            # Status spelled out, not carried by a coloured dot alone.
            cols = st.columns(5)
            cols[0].markdown(
                f"**{device_label}**  \n"
                + ("⚠️ Fault" if has_error else "OK")
            )
            cols[1].metric("Voltage", _m(row.get("voltage_v"), "V"))
            cols[2].metric("Current", _m(row.get("current_a"), "A", 2))
            cols[3].metric("Power", _m(row.get("active_power_kw"), "kW", 3))
            cols[4].metric("Energy", _m(row.get("forward_energy_kwh"), "kWh"))

            if has_error:
                st.caption(f"{device_label} reported: {row.get('error')}")

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
            pivot_reset = pivot.reset_index()
            meter_cols = [c for c in pivot_reset.columns if c != "created_at"]

            with st.expander("📐 Chart scale (optional)"):
                data_min_t = pivot_reset["created_at"].min().to_pydatetime()
                data_max_t = pivot_reset["created_at"].max().to_pydatetime()
                if data_min_t < data_max_t:
                    dcm_x_start, dcm_x_end = st.slider(
                        "X range (time)", min_value=data_min_t, max_value=data_max_t,
                        value=(data_min_t, data_max_t), key="dcm_trend_x_range",
                    )
                else:
                    st.caption("Only one timestamp in range — X range slider needs more data.")
                    dcm_x_start, dcm_x_end = data_min_t, data_max_t

                dcm_chart_auto = st.checkbox("Auto Y-axis", value=True, key="dcm_trend_y_auto")
                default_min = float(pivot[meter_cols].min(numeric_only=True).min()) if not pivot.empty else 0.0
                default_max = float(pivot[meter_cols].max(numeric_only=True).max()) if not pivot.empty else 1.0
                dcm_y_min_col, dcm_y_max_col = st.columns(2)
                with dcm_y_min_col:
                    dcm_chart_ymin = st.number_input("Y min", value=default_min, key="dcm_trend_ymin", disabled=dcm_chart_auto)
                with dcm_y_max_col:
                    dcm_chart_ymax = st.number_input("Y max", value=default_max, key="dcm_trend_ymax", disabled=dcm_chart_auto)

            metric_axis_label = {
                "voltage_v": "Voltage (V)", "current_a": "Current (A)", "active_power_kw": "Power (kW)",
            }[metric_choice]

            dcm_fig = plot_line_chart(
                pivot_reset, "created_at", meter_cols,
                x_range=(dcm_x_start, dcm_x_end),
                y_range=None if dcm_chart_auto else (dcm_chart_ymin, dcm_chart_ymax),
                y_title=metric_axis_label,
            )
            st.plotly_chart(dcm_fig, use_container_width=True)

        with st.expander("Raw panel meter data table"):
            st.dataframe(df_panel, use_container_width=True, hide_index=True)
