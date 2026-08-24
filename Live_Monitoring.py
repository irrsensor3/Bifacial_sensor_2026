import calendar
import time
from datetime import date, datetime

import streamlit as st
import pandas as pd

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
    resolve_period_files,
    list_available_dcm_csvs,
    download_and_combine_dcm_csvs,
)

def _prepare_plot_data(
    df: pd.DataFrame,
    timestamp_col: str = "created_at",
) -> pd.DataFrame:
    """Prepare data for plotting.

    <= 1 month       -> raw data
    > 1 to 6 months  -> 1-hour averages
    > 6 months       -> 4-hour averages

    The original dataframe is never modified.
    """
    if df.empty or timestamp_col not in df.columns:
        return df

    work = df.copy()

    work[timestamp_col] = pd.to_datetime(
        work[timestamp_col],
        errors="coerce",
    )

    work = (
        work
        .dropna(subset=[timestamp_col])
        .sort_values(timestamp_col)
    )

    if work.empty:
        return work

    span = work[timestamp_col].max() - work[timestamp_col].min()

    # 1 month or less → keep raw resolution
    if span <= pd.Timedelta(days=31):
        return work

    # More than 1 month up to 6 months → 1-hour average
    if span <= pd.Timedelta(days=183):
        rule = "1h"

    # More than 6 months → 4-hour average
    else:
        rule = "4h"

    value_cols = [
        c for c in work.columns
        if c != timestamp_col
    ]

    numeric_cols = work[value_cols].select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_cols:
        return work

    plot_df = (
        work[[timestamp_col] + numeric_cols]
        .set_index(timestamp_col)
        .resample(rule)
        .mean()
        .dropna(how="all")
        .reset_index()
    )

    return plot_df
    
def st_autorefresh_builtin(seconds: int):
    """Re-run the page every `seconds`, using whatever the installed Streamlit
    provides. st.autorefresh exists on newer builds; older ones fall back to a
    timed fragment. Either way there is no third-party component to break."""
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=seconds * 1000, key="live_refresh")
        return

    @st.fragment(run_every=seconds)
    def _tick():
        st.caption(" ")

    try:
        _tick()
    except Exception:
        # Very old Streamlit: leave the manual Refresh button as the only path.
        pass


def _irr_build_created_at(df: pd.DataFrame) -> pd.DataFrame:
    """The irradiance CSVs ship separate Date/Time columns rather than a
    single timestamp. Build `created_at` from them so historical rows can be
    trimmed to an exact date range and merged onto the live chart's x-axis,
    same as the DC meter CSVs (which already carry `created_at` once
    download_and_combine_dcm_csvs standardizes them)."""
    df = df.copy()
    if "Date" in df.columns and "Time" in df.columns:
        df["created_at"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
        )
    elif "Time" in df.columns:
        df["created_at"] = pd.to_datetime(df["Time"], errors="coerce")
    else:
        df["created_at"] = pd.NaT
    return df


def _load_range(period_files, download_fn, build_created_at, start_date, end_date):
    """Downloads+combines the given Drive files and trims to exactly
    [start_date, end_date]. Shared by both the auto-load-on-open path and
    the manual range picker so they can't drift apart."""
    file_entries = tuple((f["id"], f.get("modifiedTime")) for f in period_files)
    df_hist = download_fn(file_entries)
    if build_created_at is not None:
        df_hist = build_created_at(df_hist)
    if "created_at" in df_hist.columns:
        ca = (
            pd.to_datetime(
                df_hist["created_at"],
                errors="coerce",
                utc=True,
            )
            .dt.tz_localize(None)
        )
    
        day_after_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    
        df_hist = df_hist[
            (ca >= pd.Timestamp(start_date)) &
            (ca < day_after_end)
        ].copy()
    
        df_hist["created_at"] = ca.loc[df_hist.index]
    
    return df_hist


# Google Drive historical data is intentionally much slower than the live
# Supabase refresh.  Live data can refresh every 15 seconds, while Drive is
# checked only once every 30 minutes.  This prevents repeated Drive downloads
# from blocking/crashing the Streamlit app.
DRIVE_SYNC_INTERVAL_SECONDS = 1800


def _sync_drive_history_if_due(key_prefix, available_files, download_fn, build_created_at=None):
    """
    Refresh today's Drive data periodically.

    Historical data already loaded into the session is NOT touched.

    For today's data:
        - check Drive every DRIVE_SYNC_INTERVAL_SECONDS
        - if the CSV has not changed, do nothing
        - if it changed, download the latest version
        - replace only today's cached data

    This prevents historical CSVs from being repeatedly processed.
    """

    if not available_files:
        return st.session_state.get(f"_{key_prefix}_df")

    start_date = st.session_state.get(f"_{key_prefix}_start_date")
    end_date = st.session_state.get(f"_{key_prefix}_end_date")

    # Only automatically synchronize a single day.
    today = date.today()

    if start_date != today or end_date != today:
        return st.session_state.get(f"_{key_prefix}_df")

    now = time.monotonic()
    last_sync = st.session_state.get(f"_{key_prefix}_last_sync_monotonic", 0.0)

    # Not time to sync yet.
    if now - last_sync < DRIVE_SYNC_INTERVAL_SECONDS:
        return st.session_state.get(f"_{key_prefix}_df")

    # ---------------------------------------------------------
    # Find today's CSV
    # ---------------------------------------------------------
    today_files = resolve_period_files(available_files, today, today)

    if not today_files:
        return st.session_state.get(f"_{key_prefix}_df")

    # Safety protection.
    if len(today_files) > 10:
        st.session_state[f"_{key_prefix}_sync_error"] = (
            f"Today's Drive folder contains "
            f"{len(today_files)} CSV files. "
            f"Automatic sync skipped."
        )
        return st.session_state.get(f"_{key_prefix}_df")

    # ---------------------------------------------------------
    # Detect whether today's files actually changed
    # ---------------------------------------------------------
    current_signature = tuple((f["id"], f.get("modifiedTime", "")) for f in today_files)
    previous_signature = st.session_state.get(f"_{key_prefix}_today_signature")

    # Mark sync time regardless of whether data changed.
    st.session_state[f"_{key_prefix}_last_sync_monotonic"] = now

    # Nothing changed on Drive.
    if current_signature == previous_signature:
        return st.session_state.get(f"_{key_prefix}_df")

    # ---------------------------------------------------------
    # Download today's updated CSV
    # ---------------------------------------------------------
    try:
        df_today = _load_range(today_files, download_fn, build_created_at, today, today)

        if df_today is None or df_today.empty:
            return st.session_state.get(f"_{key_prefix}_df")

        # -----------------------------------------------------
        # Replace ONLY today's cached dataset.
        #
        # Historical data is never downloaded here.
        # -----------------------------------------------------
        st.session_state[f"_{key_prefix}_df"] = df_today
        st.session_state[f"_{key_prefix}_today_signature"] = current_signature
        st.session_state[f"_{key_prefix}_label"] = f"{today:%d %b %Y}"
        st.session_state[f"_{key_prefix}_sync_error"] = None

    except Exception as exc:
        st.session_state[f"_{key_prefix}_sync_error"] = str(exc)

    return st.session_state.get(f"_{key_prefix}_df")


def _time_range_controls(key_prefix, data_min_t, data_max_t):
    """From/to date+time pickers for zooming a chart's X axis, in place of a
    two-handle range slider. When the two ends of the loaded data are close
    together (e.g. only a few hours of "today" logged so far), a slider's
    handles overlap and become nearly impossible to grab separately,
    especially by touch — typing or tapping a date and time directly has no
    such problem."""
    if data_min_t >= data_max_t:
        st.caption("Only one timestamp in range — nothing to adjust yet.")
        return data_min_t, data_max_t

    start_col, end_col = st.columns(2)
    with start_col:
        st.caption("From")
        start_date_v = st.date_input(
            "From date", value=data_min_t.date(),
            min_value=data_min_t.date(), max_value=data_max_t.date(),
            key=f"{key_prefix}_start_date", label_visibility="collapsed",
        )
        start_time_v = st.time_input(
            "From time", value=data_min_t.time(),
            key=f"{key_prefix}_start_time", label_visibility="collapsed",
        )
    with end_col:
        st.caption("To")
        end_date_v = st.date_input(
            "To date", value=data_max_t.date(),
            min_value=data_min_t.date(), max_value=data_max_t.date(),
            key=f"{key_prefix}_end_date", label_visibility="collapsed",
        )
        end_time_v = st.time_input(
            "To time", value=data_max_t.time(),
            key=f"{key_prefix}_end_time", label_visibility="collapsed",
        )

    x_start_t = datetime.combine(start_date_v, start_time_v)
    x_end_t = datetime.combine(end_date_v, end_time_v)
    if x_start_t >= x_end_t:
        st.caption("'From' is after 'to' — showing the full range instead.")
        return data_min_t, data_max_t
    return x_start_t, x_end_t


def _historical_append_controls(key_prefix, available_files, download_fn, build_created_at=None):
    """Range-selection controls (month / year / date range / from date /
    until date) for showing historical Drive data on a chart.

    On first load each session, this auto-loads *today's* data with no
    click needed — the picker below is only for switching to a different
    range afterwards. Because it's an auto-load rather than something tied
    to a previous click, it reappears on its own every time the page/site is
    reopened, not just within one browser tab.

    Returns (df_or_None, label_or_None). The result is kept in
    st.session_state under f"_{key_prefix}_df" / f"_{key_prefix}_label" so it
    survives other widget interactions on the page (selecting sensors,
    changing the metric, etc. no longer clear it).

    `download_fn` is download_and_combine_csvs or
    download_and_combine_dcm_csvs — both are disk-cached (see drive_fetch.py),
    so re-requesting a period anyone has already viewed is instant rather
    than a fresh Drive download.
    """
    if not available_files:
        st.caption("No Drive CSVs found yet.")
        return st.session_state.get(f"_{key_prefix}_df"), st.session_state.get(f"_{key_prefix}_label")

    years = sorted({int(y) for f in available_files if (y := extract_year(f)).isdigit()})
    if not years:
        st.caption("Couldn't determine dates for the files found.")
        return st.session_state.get(f"_{key_prefix}_df"), st.session_state.get(f"_{key_prefix}_label")
    earliest, latest = date(min(years), 1, 1), date(max(years), 12, 31)

    # Auto-load today's data once per session, no button press. Guarded by
    # a flag (not just "is there data yet") so explicitly removing it via
    # "Remove appended data" below doesn't get silently undone on the next
    # refresh tick.
    auto_flag = f"_{key_prefix}_auto_loaded"
    if not st.session_state.get(auto_flag):
        st.session_state[auto_flag] = True
        today = date.today()
        if earliest <= today <= latest:
            today_files = resolve_period_files(available_files, today, today)
            if today_files:
                df_today = _load_range(today_files, download_fn, build_created_at, today, today)

                st.session_state[f"_{key_prefix}_df"] = df_today
                st.session_state[f"_{key_prefix}_label"] = f"{today:%d %b %Y}"
                st.session_state[f"_{key_prefix}_start_date"] = today
                st.session_state[f"_{key_prefix}_end_date"] = today

                # Remember exactly which Drive version we loaded.
                st.session_state[f"_{key_prefix}_today_signature"] = tuple(
                    (f["id"], f.get("modifiedTime", "")) for f in today_files
                )
                st.session_state[f"_{key_prefix}_last_sync_monotonic"] = time.monotonic()

    mode = st.radio(
        "Range",
        ["Month", "Year", "Date range", "From date", "Until date"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )

    start_date = end_date = None
    if mode == "Month":
        yr_col, mo_col = st.columns(2)
        with yr_col:
            yr = st.selectbox("Year", sorted(years, reverse=True), key=f"{key_prefix}_year")
        year_files = [f for f in available_files if extract_year(f) == str(yr)]
        months = sorted({extract_month(f) for f in year_files if extract_month(f).isdigit()})
        if months:
            with mo_col:
                mo = st.selectbox(
                    "Month", months, index=len(months) - 1,
                    format_func=month_label, key=f"{key_prefix}_month",
                )
            start_date = date(yr, int(mo), 1)
            end_date = date(yr, int(mo), calendar.monthrange(yr, int(mo))[1])
    elif mode == "Year":
        yr = st.selectbox("Year", sorted(years, reverse=True), key=f"{key_prefix}_yronly")
        start_date, end_date = date(yr, 1, 1), date(yr, 12, 31)
    elif mode == "Date range":
        picked = st.date_input(
            "From / to", value=(earliest, latest),
            min_value=earliest, max_value=latest, key=f"{key_prefix}_range",
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            start_date, end_date = picked
        else:
            st.caption("Pick both a start and an end date.")
    elif mode == "≥ From date":
        start_date = st.date_input(
            "≥", value=earliest, min_value=earliest, max_value=latest,
            key=f"{key_prefix}_from",
        )
        end_date = latest
    elif mode == "≤ Until date":
        end_date = st.date_input(
            "≤", value=latest, min_value=earliest, max_value=latest,
            key=f"{key_prefix}_until",
        )
        start_date = earliest

    if not start_date or not end_date or start_date > end_date:
        st.caption("Pick a valid range.")
        return st.session_state.get(f"_{key_prefix}_df"), st.session_state.get(f"_{key_prefix}_label")

    period_files = resolve_period_files(available_files, start_date, end_date)
    st.caption(
        f"{len(period_files)} file(s) found covering "
        f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"
    )

    btn_col, remove_col = st.columns(2)

    with btn_col:
        go = st.button(
            "📥 Load this range",
            key=f"{key_prefix}_append_btn",
            disabled=not period_files,
            use_container_width=True,
        )

    if go:
        with st.spinner(f"Loading {len(period_files)} file(s)..."):
            df_hist = _load_range(period_files, download_fn, build_created_at, start_date, end_date)
        st.session_state[f"_{key_prefix}_df"] = df_hist
        st.session_state[f"_{key_prefix}_label"] = (
            f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"
            if start_date != end_date
            else f"{start_date:%d %b %Y}"
        )
        st.session_state[f"_{key_prefix}_start_date"] = start_date
        st.session_state[f"_{key_prefix}_end_date"] = end_date
        st.session_state[f"_{key_prefix}_last_sync_monotonic"] = time.monotonic()
        st.session_state[f"_{key_prefix}_sync_error"] = None
        st.success(f"Loaded {df_hist.shape[0]} rows from {start_date:%d %b %Y} – {end_date:%d %b %Y}")

    with remove_col:
        if st.session_state.get(f"_{key_prefix}_df") is not None:
            if st.button(
                "✖️ Clear", key=f"{key_prefix}_remove_btn",
                use_container_width=True,
            ):
                st.session_state[f"_{key_prefix}_df"] = None
                st.session_state[f"_{key_prefix}_label"] = None
                st.session_state[f"_{key_prefix}_start_date"] = None
                st.session_state[f"_{key_prefix}_end_date"] = None
                st.session_state[f"_{key_prefix}_last_sync_monotonic"] = 0.0
                st.session_state[f"_{key_prefix}_sync_error"] = None
                st.rerun(scope="fragment")

    # Refresh the currently loaded Drive range only every 5 minutes.
    # This applies to TODAY as well as any older/custom range the user loaded.
    _sync_drive_history_if_due(key_prefix, available_files, download_fn, build_created_at=build_created_at)

    sync_error = st.session_state.get(f"_{key_prefix}_sync_error")
    if sync_error:
        st.caption("Drive sync temporarily unavailable; keeping the last good data.")

    label = st.session_state.get(f"_{key_prefix}_label")
    if label:
        st.caption(f"Currently appended: {label} • Drive sync every 30 min")

    return st.session_state.get(f"_{key_prefix}_df"), label


def render_live_monitoring():
    require_login()

    page_stamp("Live Monitoring")
    st.title("Live monitoring")

    # -------------------------
    # Refresh controls (outside the fragments below — changing these is rare,
    # so it's fine for it to rerun the whole page).
    # -------------------------
    st.subheader("Sensors")
    ref_on, ref_int = st.columns([1, 3])
    with ref_on:
        auto_refresh = st.toggle("Auto-refresh", value=True, key="live_auto_refresh")
    with ref_int:
        refresh_seconds = st.slider(
            "Refresh every (seconds)", 5, 60, 15,
            disabled=not auto_refresh, key="live_refresh_secs",
        )
    if st.button("Refresh now"):
        # Clear only the live-reading caches, not the hour(s)-long Drive
        # download cache — a manual refresh shouldn't force everyone's
        # already-appended historical data to re-download from Drive.
        fetch_latest_readings.clear()
        fetch_recent_alerts.clear()
        fetch_latest_panel_readings.clear()
        st.rerun()

    run_every = refresh_seconds if auto_refresh else None

    # -------------------------
    # Live sensor data — its own fragment, so the refresh timer only reruns
    # this block (and its own widgets: sensor picker, historical append,
    # chart-scale controls) instead of the whole app on every tick. Before,
    # each tick also re-ran the login check, theme CSS injection, the
    # overview banner (2 more Supabase reads + the array diagram SVG), the
    # sidebar, and page navigation — which is where the lag came from, not
    # from the data fetches themselves.
    # -------------------------
    @st.fragment(run_every=run_every)
    def _live_sensors_fragment():
        df_live = fetch_latest_readings()

        if df_live.empty:
            st.info("No live data yet — waiting for the Pi to push a sample.")
            return

        latest = df_live.iloc[-1]
        st.caption(f"Last update: {latest.get('date', '')} {latest.get('time', '')}")

        irr_cols = [c for c in df_live.columns if c.startswith("Irr_")]

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

        with st.expander("🗄️ Append historical data from Drive"):
            available_files = list_available_csvs()
            df_hist, _ = _historical_append_controls(
                "live_append", available_files, download_and_combine_csvs,
                build_created_at=_irr_build_created_at,
            )

        selected_live_irr = st.multiselect(
            "Irradiance sensors to plot (live)",
            irr_cols,
            key="selected_live_irr",
            label_visibility="collapsed",
        )
        if selected_live_irr:
            combined = df_live[["created_at"] + selected_live_irr].copy()

            # Normalize LIVE timestamps
            combined["created_at"] = (
                pd.to_datetime(
                    combined["created_at"],
                    errors="coerce",
                    utc=True,
                )
                .dt.tz_localize(None)
            )
            
            if df_hist is not None and not df_hist.empty:
                hist = df_hist.copy()
            
                for c in selected_live_irr:
                    if c not in hist.columns:
                        hist[c] = pd.NA
            
                hist_slice = hist[["created_at"] + selected_live_irr].copy()
            
                # Normalize HISTORICAL timestamps
                hist_slice["created_at"] = (
                    pd.to_datetime(
                        hist_slice["created_at"],
                        errors="coerce",
                        utc=True,
                    )
                    .dt.tz_localize(None)
                )
            
                combined = pd.concat(
                    [hist_slice, combined],
                    ignore_index=True,
                )
            
            # Both historical and live timestamps are now
            # timezone-naive UTC timestamps.
            combined = (
                combined
                .dropna(subset=["created_at"])
                .sort_values("created_at")
                .reset_index(drop=True)
            )
            with st.expander("📐 Chart scale (optional)"):
                data_min_t = combined["created_at"].min().to_pydatetime()
                data_max_t = combined["created_at"].max().to_pydatetime()
                st.caption("X range (time)")
                x_start_t, x_end_t = _time_range_controls("live_irr_x", data_min_t, data_max_t)

                irr_chart_auto = st.checkbox("Auto Y-axis", value=True, key="live_irr_y_auto")
                y_min_col, y_max_col = st.columns(2)
                with y_min_col:
                    irr_chart_ymin = st.number_input("Y min (W/m²)", value=0.0, key="live_irr_ymin", disabled=irr_chart_auto)
                with y_max_col:
                    irr_chart_ymax = st.number_input("Y max (W/m²)", value=1200.0, key="live_irr_ymax", disabled=irr_chart_auto)

            combined_plot = _prepare_plot_data(combined)

            fig = plot_line_chart(
                combined_plot,
                "created_at",
                selected_live_irr,
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

    _live_sensors_fragment()

    # -------------------------
    # Live panel meter data — same fragment treatment.
    # -------------------------
    st.divider()
    st.subheader("Panel meters")

    @st.fragment(run_every=run_every)
    def _panel_meters_fragment():
        df_panel = fetch_latest_panel_readings()

        if df_panel.empty:
            st.info("No live panel meter data yet — waiting for the mini PC to push a sample.")
            return

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

            state = ('<span class="state bad">Fault</span>' if has_error
                     else '<span class="state ok">OK</span>')
            st.markdown(f'<div class="meter-head">{device_label}{state}</div>',
                        unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].metric("Voltage", _m(row.get("voltage_v"), "V"))
            cols[1].metric("Current", _m(row.get("current_a"), "A", 2))
            cols[2].metric("Power", _m(row.get("active_power_kw"), "kW", 3))
            cols[3].metric("Energy", _m(row.get("forward_energy_kwh"), "kWh"))

            if has_error:
                st.caption(f"{device_label} reported: {row.get('error')}")

        
        with st.expander("🗄️ Append historical DC meter data from Drive"):

            dcm_data_type = st.radio(
                "Historical data",
                ["Normal", "Average"],
                horizontal=True,
                key="dcm_historical_data_type",
            )
        
            use_avg = dcm_data_type == "Average"
        
            available_dcm_files = list_available_dcm_csvs(
                include_avg=use_avg
            )
        
            if not available_dcm_files and st.session_state.get("_dcm_drive_list_error"):
                with st.expander("Error details"):
                    st.code(st.session_state["_dcm_drive_list_error"])
        
            df_dcm_hist, _ = _historical_append_controls(
                "live_append_dcm_avg" if use_avg else "live_append_dcm",
                available_dcm_files,
                download_and_combine_dcm_csvs,
            )

        if df_dcm_hist is not None and not df_dcm_hist.empty:
            df_panel_combined = df_dcm_hist.copy()
        else:
            df_panel_combined = df_panel.copy()
        if df_dcm_hist is not None and not df_dcm_hist.empty:
            needed_cols = [
                "created_at", "device_id", "voltage_v", "current_a",
                "active_power_kw", "forward_energy_kwh", "error",
            ]
            hist_slice = df_dcm_hist.reindex(columns=needed_cols).copy()
            live_slice = df_panel.reindex(columns=needed_cols).copy()

            # ---------------------------------------------------------
            # NORMALIZE TIMESTAMPS
            # ---------------------------------------------------------
            # Historical Drive data and live Supabase data may use
            # different timezone representations.
            #
            # Convert BOTH sides to UTC, then remove timezone information
            # so pandas sees both as datetime64[ns].
            # ---------------------------------------------------------
            hist_slice["created_at"] = pd.to_datetime(
                hist_slice["created_at"], errors="coerce", utc=True,
            ).dt.tz_localize(None)

            live_slice["created_at"] = pd.to_datetime(
                live_slice["created_at"], errors="coerce", utc=True,
            ).dt.tz_localize(None)

            # ---------------------------------------------------------
            # COMBINE HISTORICAL + LIVE
            # ---------------------------------------------------------
            df_panel_combined = pd.concat([hist_slice, live_slice], ignore_index=True)

            # Remove invalid timestamps BEFORE sorting
            df_panel_combined = (
                df_panel_combined
                .dropna(subset=["created_at"])
                .sort_values("created_at")
                .reset_index(drop=True)
            )

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
            pivot_plot = _prepare_plot_data(pivot_reset)

            with st.expander("📐 Chart scale (optional)"):
                data_min_t = pivot_reset["created_at"].min().to_pydatetime()
                data_max_t = pivot_reset["created_at"].max().to_pydatetime()
                st.caption("X range (time)")
                dcm_x_start, dcm_x_end = _time_range_controls("dcm_trend_x", data_min_t, data_max_t)

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
                pivot_plot,
                "created_at",
                meter_cols,
                x_range=None,
                y_range=None,
                y_title=metric_axis_label,
            )
            st.plotly_chart(dcm_fig, use_container_width=True)

        with st.expander("Raw panel meter data table"):
            st.dataframe(df_panel, use_container_width=True, hide_index=True)

    _panel_meters_fragment()
