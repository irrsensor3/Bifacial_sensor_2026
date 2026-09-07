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
    df = df.copy()
    if "Date" in df.columns and "Time" in df.columns:
        naive = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
        )
        df["created_at"] = (
            naive.dt.tz_localize("Asia/Kuala_Lumpur", ambiguous="NaT", nonexistent="NaT")
                 .dt.tz_convert("UTC")
                 .dt.tz_localize(None)
        )
    elif "Time" in df.columns:
        naive = pd.to_datetime(df["Time"], errors="coerce")
        df["created_at"] = (
            naive.dt.tz_localize("Asia/Kuala_Lumpur", ambiguous="NaT", nonexistent="NaT")
                 .dt.tz_convert("UTC")
                 .dt.tz_localize(None)
        )
    else:
        df["created_at"] = pd.NaT
    return df

# Streamlit Community Cloud gives the app roughly 1 GB. One day of sensor data
# is about 8.7 MB on disk and several times that once loaded into pandas as 74
# float columns, so an unbounded month-long range exhausted memory and the app
# was killed. These caps keep a request bounded; the user is told when one bites.
LOCAL_TZ = "Asia/Kuala_Lumpur"


def to_local(series):
    """Parse a created_at column and move it to array-local time.

    The database stores UTC. Historical files loaded from Drive are already
    converted, but live rows were not, so the two sat eight hours apart on the
    same chart -- a reading taken at 11:24 in the morning appeared at 03:24 and
    looked like the array was generating in the middle of the night.
    """
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)


MAX_LOAD_FILES = 7          # days per download
MAX_PLOT_POINTS = 4000      # per series, after downsampling


def _downsample_for_plot(df, time_col="created_at", max_points=MAX_PLOT_POINTS):
    """Thin a frame to a size a browser chart can actually draw.

    A month at five-second resolution is half a million points per series. No
    screen has that many pixels, so the detail is invisible, but the browser
    still has to receive and render every one of them -- which is what made the
    page hang before it crashed.
    """
    if df is None or df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def _load_range(period_files, download_fn, build_created_at, start_date, end_date):
    """Downloads+combines the given Drive files and trims to exactly
    [start_date, end_date]. Shared by both the auto-load-on-open path and
    the manual range picker so they can't drift apart."""
    # Take the most recent files when a range exceeds the cap: a truncated
    # window ending at the requested date is more useful than one that starts
    # there and stops early.
    truncated = len(period_files) > MAX_LOAD_FILES
    if truncated:
        period_files = sorted(
            period_files, key=lambda f: f.get("name", ""))[-MAX_LOAD_FILES:]
        st.warning(
            f"That range covers more days than can be loaded at once. Showing "
            f"the most recent {MAX_LOAD_FILES} day(s). For longer periods, "
            f"analyse the CSV files directly rather than through this page."
        )

    file_entries = tuple((f["id"], f.get("modifiedTime")) for f in period_files)
    df_hist = download_fn(file_entries)
    if build_created_at is not None:
        df_hist = build_created_at(df_hist)
    # inside _load_range, replace the existing "if 'created_at' in df_hist.columns:" block
    if "created_at" in df_hist.columns:
        # Parse created_at as UTC-aware timestamps (works whether values are naive UTC strings
        # or tz-aware UTC). We'll do comparisons in UTC to avoid ambiguity.
        ca_utc = pd.to_datetime(df_hist["created_at"], errors="coerce", utc=True)
    
        # Build timezone-aware boundaries for the requested start/end in the local tz,
        # then convert to UTC for comparison against ca_utc.
        LOCAL_TZ = "Asia/Kuala_Lumpur"
        start_local = pd.Timestamp(start_date, tz=LOCAL_TZ)
        day_after_end_local = pd.Timestamp(end_date, tz=LOCAL_TZ) + pd.Timedelta(days=1)
    
        start_utc = start_local.tz_convert("UTC")
        day_after_end_utc = day_after_end_local.tz_convert("UTC")
    
        # Filter using UTC-aware comparisons
        mask = (ca_utc >= start_utc) & (ca_utc < day_after_end_utc)
        df_hist = df_hist.loc[mask].copy()
    
        # For downstream code / plotting the app expects timezone-naive local times.
        # Convert the kept UTC timestamps into local timezone then drop tz info.
        df_hist["created_at"] = ca_utc.loc[df_hist.index].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        
    return df_hist


# Google Drive historical data is intentionally slower than the live Supabase
# refresh, but not by much: live data can refresh every 15 seconds, while
# Drive is re-checked every 15 minutes. That interval is a MERGE, not a
# replace -- Supabase's live table only keeps a rolling window, and the Pi's
# rclone sync of today's CSV can itself lag behind real time, so a slice of
# "today" can briefly exist in neither source. Re-downloading today's Drive
# file periodically and merging (deduping by timestamp) into the cache means
# that slice gets filled in as soon as rclone catches up, without ever
# discarding rows the live feed already has that Drive doesn't have yet.
DRIVE_SYNC_INTERVAL_SECONDS = 15 * 60


def _sync_drive_history_if_due(key_prefix, available_files, download_fn, build_created_at=None):
    """
    Every DRIVE_SYNC_INTERVAL_SECONDS, re-download today's Drive file(s) and
    MERGE the result into today's cached dataset (dedup by timestamp),
    rather than trusting a fresh download as a full replacement.

    Historical (non-today) data already loaded into the session is never
    touched by this -- it only ever acts on the currently-loaded range when
    that range is exactly today.
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

    st.session_state[f"_{key_prefix}_last_sync_monotonic"] = now

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
    # Download today's current CSV and MERGE it into the cache. Merging
    # (instead of a signature-gated replace) means this still catches rows
    # that arrived on Drive late even if nothing else about the file's
    # identity changed between checks, which is exactly the Supabase-missed
    # data this exists to backfill.
    # ---------------------------------------------------------
    try:
        df_fresh = _load_range(today_files, download_fn, build_created_at, today, today)

        if df_fresh is None or df_fresh.empty:
            return st.session_state.get(f"_{key_prefix}_df")

        df_cached = st.session_state.get(f"_{key_prefix}_df")
        if df_cached is None or df_cached.empty:
            merged = df_fresh
        else:
            merged = pd.concat([df_cached, df_fresh], ignore_index=True)
            # IRR frames are wide (one row per timestamp, no device column);
            # DCM frames are long (one row per device per timestamp) -- dedup
            # on whichever key actually identifies a unique reading.
            id_col = "device_id" if "device_id" in merged.columns else None
            subset = ["created_at"] + ([id_col] if id_col else [])
            merged = (
                merged.dropna(subset=["created_at"])
                      .drop_duplicates(subset=subset, keep="last")
                      .sort_values("created_at")
                      .reset_index(drop=True)
            )

        st.session_state[f"_{key_prefix}_df"] = merged
        st.session_state[f"_{key_prefix}_today_signature"] = tuple(
            (f["id"], f.get("modifiedTime", "")) for f in today_files
        )
        st.session_state[f"_{key_prefix}_label"] = f"{today:%d %b %Y}"
        st.session_state[f"_{key_prefix}_sync_error"] = None

    except Exception as exc:
        st.session_state[f"_{key_prefix}_sync_error"] = str(exc)

    return st.session_state.get(f"_{key_prefix}_df")


def _reseed_widget_value(session_key, signal_key, signal, seed_value):
    """Streamlit widgets only honor their `value=` argument the first time a
    given `key` is created in a session -- every later rerun just returns
    whatever is already in session_state, ignoring `value=` entirely. Call
    this BEFORE creating the widget: when `signal` differs from what was
    last seen, this overwrites session_state[session_key] directly so the
    widget actually picks up the new default on this rerun (e.g. so a chart
    axis defaulting to "today" keeps following today as today changes,
    instead of freezing at whichever day/range it first saw).
    """
    if st.session_state.get(signal_key) != signal:
        st.session_state[signal_key] = signal
        st.session_state[session_key] = seed_value


def _time_range_controls(key_prefix, data_min_t, data_max_t,
                          day_start=None, day_end=None, reset_signal=None):
    """From/to date+time pickers for zooming a chart's X axis, in place of a
    two-handle range slider. When the two ends of the loaded data are close
    together (e.g. only a few hours of "today" logged so far), a slider's
    handles overlap and become nearly impossible to grab separately,
    especially by touch — typing or tapping a date and time directly has no
    such problem.

    day_start/day_end: calendar bounds of the loaded period (e.g. 00:00 of
    the first loaded day through 23:59:59 of the last). When given, these --
    not data_min_t/data_max_t -- seed the pickers, so auto-loaded "today,
    only partially logged so far" still defaults to a 00:00-23:59 frame
    (with Auto Y-axis on, the trace itself just occupies whatever slice has
    actually arrived) instead of the axis shrink-wrapping to only the
    logged portion.

    reset_signal: anything that changes when the loaded period changes
    (e.g. the (start_date, end_date) tuple _historical_append_controls
    already tracks). See _reseed_widget_value -- without this, the pickers
    would freeze on whatever range existed the moment they were first
    created and never widen to a full day on their own.
    """
    if data_min_t >= data_max_t:
        st.caption("Only one timestamp in range — nothing to adjust yet.")
        return data_min_t, data_max_t

    if reset_signal is not None:
        seed_start = day_start or data_min_t
        seed_end = day_end or data_max_t
        sig_key = f"_{key_prefix}_reset_signal"
        if st.session_state.get(sig_key) != reset_signal:
            st.session_state[sig_key] = reset_signal
            st.session_state[f"{key_prefix}_start_date"] = seed_start.date()
            st.session_state[f"{key_prefix}_start_time"] = seed_start.time()
            st.session_state[f"{key_prefix}_end_date"] = seed_end.date()
            st.session_state[f"{key_prefix}_end_time"] = seed_end.time()

    # min/max_value bounds must cover both the actual data and the (possibly
    # wider) calendar-day seed, or Streamlit clamps/errors on out-of-range values.
    min_d = min(data_min_t.date(), (day_start or data_min_t).date())
    max_d = max(data_max_t.date(), (day_end or data_max_t).date())

    start_col, end_col = st.columns(2)
    with start_col:
        st.caption("From")
        start_date_v = st.date_input(
            "From date", value=data_min_t.date(),
            min_value=min_d, max_value=max_d,
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
            min_value=min_d, max_value=max_d,
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
        ["Single day", "Month", "Year", "Date range", "From date", "Until date"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )

    start_date = end_date = None
    if mode == "Single day":
        # The common case: one date, one day of data. Offered first because
        # picking a month to look at one day loads thirty times more than is
        # wanted and is the main way this page was made to run out of memory.
        pick = st.date_input(
            "Date", value=latest, min_value=earliest, max_value=latest,
            key=f"{key_prefix}_single_day")
        start_date = end_date = pick

    elif mode == "Month":
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

    # Refresh the currently loaded Drive range (merge, not replace) every
    # DRIVE_SYNC_INTERVAL_SECONDS. This applies to TODAY as well as any
    # older/custom range the user loaded.
    _sync_drive_history_if_due(key_prefix, available_files, download_fn, build_created_at=build_created_at)

    sync_error = st.session_state.get(f"_{key_prefix}_sync_error")
    if sync_error:
        st.caption("Drive sync temporarily unavailable; keeping the last good data.")

    label = st.session_state.get(f"_{key_prefix}_label")
    if label:
        st.caption(f"Currently appended: {label} • Drive sync every 15 min")

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
    # Auto-refresh reruns the whole fragment on a timer, and each rerun
    # re-slices, re-concatenates and re-plots whatever history is loaded. With
    # a month in session state that is hundreds of megabytes rebuilt every few
    # seconds, and memory climbs until the app is killed -- which is why the
    # chart appeared first and the crash came later. Refusing to auto-refresh
    # while a large history is loaded removes the repetition, not the data.
    _loaded = [st.session_state.get(f"_{k}_df")
               for k in ("live_append_irr", "live_append_dcm",
                         "live_append_dcm_avg")]
    _loaded_rows = sum(len(d) for d in _loaded if d is not None)
    if _loaded_rows > 50_000:
        st.session_state["live_auto_refresh"] = False
        st.info(
            f"Auto-refresh paused: {_loaded_rows:,} rows of historical data are "
            f"loaded. Redrawing that every few seconds is what exhausts memory. "
            f"Use Refresh now, or clear the history to resume automatic updates."
        )

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
        # Clear whichever of these are cached. Calling .clear() directly raises
        # AttributeError on an uncached function, which took the whole page down
        # when one of the three lost its decorator -- a refresh button should
        # never be able to do that.
        for _fn in (fetch_latest_readings, fetch_recent_alerts,
                    fetch_latest_panel_readings):
            clear = getattr(_fn, "clear", None)
            if callable(clear):
                clear()
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
                .dt.tz_convert("Asia/Kuala_Lumpur")
                .dt.tz_localize(None)
            )
            
            if df_hist is not None and not df_hist.empty:
                hist = df_hist.copy()
            
                for c in selected_live_irr:
                    if c not in hist.columns:
                        hist[c] = pd.NA
            
                hist_slice = hist[["created_at"] + selected_live_irr].copy()
            
                # _load_range already returns timezone-naive LOCAL times --
                # see the conversion at the end of that function. Converting
                # again here added a second eight hours, so a 13:14 reading
                # plotted at 21:14. Parse only; do not shift.
                hist_slice["created_at"] = pd.to_datetime(
                    hist_slice["created_at"], errors="coerce")
            
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

                # Default the axis to the whole calendar day currently loaded
                # (not just whichever timestamps happen to exist yet), so
                # auto-loaded "today, partially logged" still shows 00:00-23:59.
                loaded_start = st.session_state.get("_live_append_start_date")
                loaded_end = st.session_state.get("_live_append_end_date")
                day_start_dt = (datetime.combine(loaded_start, datetime.min.time())
                                if loaded_start else None)
                day_end_dt = (datetime.combine(loaded_end, datetime.max.time())
                              if loaded_end else None)

                x_start_t, x_end_t = _time_range_controls(
                    "live_irr_x", data_min_t, data_max_t,
                    day_start=day_start_dt, day_end=day_end_dt,
                    reset_signal=(loaded_start, loaded_end),
                )

                irr_chart_auto = st.checkbox("Auto Y-axis", value=True, key="live_irr_y_auto")
                y_min_col, y_max_col = st.columns(2)
                with y_min_col:
                    irr_chart_ymin = st.number_input("Y min (W/m²)", value=0.0, key="live_irr_ymin", disabled=irr_chart_auto)
                with y_max_col:
                    irr_chart_ymax = st.number_input("Y max (W/m²)", value=1200.0, key="live_irr_ymax", disabled=irr_chart_auto)

            # Thin before plotting. Sending half a million points to the
            # browser hangs the tab long before the extra detail becomes
            # visible on screen.
            plot_src = _downsample_for_plot(combined)
            if len(plot_src) < len(combined):
                st.caption(
                    f"Showing {len(plot_src):,} of {len(combined):,} points. "
                    f"The shape of the trace is preserved; narrow the date "
                    f"range to see every reading."
                )

            fig = plot_line_chart(
                plot_src, "created_at", selected_live_irr,
                x_range=(x_start_t, x_end_t),
                y_range=None if irr_chart_auto else (irr_chart_ymin, irr_chart_ymax),
                x_title="Time (Malaysia, UTC+8)",
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
                    when = pd.to_datetime(row["created_at"], errors="coerce", utc=True)
                    when_txt = (when.tz_convert(LOCAL_TZ).strftime("%H:%M:%S")
                                if pd.notna(when) else "unknown time")
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

        df_panel["created_at"] = to_local(df_panel["created_at"])

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
        
            dcm_prefix = "live_append_dcm_avg" if use_avg else "live_append_dcm"
            df_dcm_hist, _ = _historical_append_controls(
                dcm_prefix,
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
            # Both sides must end up as timezone-naive local time.
            # ---------------------------------------------------------
            # Already local, for the same reason as the sensor chart above.
            hist_slice["created_at"] = pd.to_datetime(
                hist_slice["created_at"], errors="coerce")

            # df_panel["created_at"] was already converted from UTC to
            # array-local naive time by to_local() at the top of this
            # fragment. Re-labeling it "utc=True" here and converting to
            # Asia/Kuala_Lumpur AGAIN added a second +8-hour shift on top of
            # that first one -- a reading actually taken at 16:30 local got
            # stamped 00:30 the *next* calendar day, which is exactly what
            # would break a whole-day 00:00-23:59 view. Parse only; do not
            # shift again.
            live_slice["created_at"] = pd.to_datetime(
                live_slice["created_at"], errors="coerce")

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
        # A segmented control rather than a radio: it reads as a row of tabs,
        # but selects one value, so only the chosen chart is built. st.tabs
        # renders every tab body, which would draw four charts on every refresh
        # tick and throw three away.
        METRICS = {"Current (A)": "current_a",
                   "Voltage (V)": "voltage_v",
                   "Power (kW)": "active_power_kw",
                   "Energy (kWh)": "forward_energy_kwh"}
        labels = list(METRICS)
        if hasattr(st, "segmented_control"):
            picked = st.segmented_control(
                "Metric", labels, default=labels[0], key="meter_metric_pick")
        else:
            # Older Streamlit: same choice, plainer control.
            picked = st.radio("Metric", labels, horizontal=True,
                              key="meter_metric_pick")
        metric_choice = METRICS.get(picked or labels[0], "current_a")

        # Trend answers "what is it doing now", distribution answers "what does
        # it usually do" -- the same numbers, two different questions, so the
        # view is a switch rather than a separate page.
        view = st.segmented_control(
            "View", ["Trend over time", "Distribution"],
            default="Trend over time", key="meter_view") \
            if hasattr(st, "segmented_control") else \
            st.radio("View", ["Trend over time", "Distribution"],
                     horizontal=True, key="meter_view")

        if selected_devices:
            pivot = df_panel_combined[df_panel_combined["device_id"].isin(selected_devices)].pivot_table(
                index="created_at", columns="device_id", values=metric_choice
            )
            pivot.columns = [f"Meter {int(c)}" for c in pivot.columns]
            pivot_reset = pivot.reset_index()
            n_full = len(pivot_reset)
            pivot_reset = _downsample_for_plot(pivot_reset)
            if len(pivot_reset) < n_full:
                st.caption(
                    f"Showing {len(pivot_reset):,} of {n_full:,} points."
                )
            meter_cols = [c for c in pivot_reset.columns if c != "created_at"]

            with st.expander("📐 Chart scale (optional)"):
                data_min_t = pivot_reset["created_at"].min().to_pydatetime()
                data_max_t = pivot_reset["created_at"].max().to_pydatetime()
                st.caption("X range (time)")

                loaded_start = st.session_state.get(f"_{dcm_prefix}_start_date")
                loaded_end = st.session_state.get(f"_{dcm_prefix}_end_date")
                day_start_dt = (datetime.combine(loaded_start, datetime.min.time())
                                if loaded_start else None)
                day_end_dt = (datetime.combine(loaded_end, datetime.max.time())
                              if loaded_end else None)

                dcm_x_start, dcm_x_end = _time_range_controls(
                    "dcm_trend_x", data_min_t, data_max_t,
                    day_start=day_start_dt, day_end=day_end_dt,
                    reset_signal=(loaded_start, loaded_end),
                )

                dcm_chart_auto = st.checkbox("Auto Y-axis", value=True, key="dcm_trend_y_auto")
                default_min = float(pivot[meter_cols].min(numeric_only=True).min()) if not pivot.empty else 0.0
                default_max = float(pivot[meter_cols].max(numeric_only=True).max()) if not pivot.empty else 1.0
                dcm_y_min_col, dcm_y_max_col = st.columns(2)
                with dcm_y_min_col:
                    dcm_chart_ymin = st.number_input("Y min", value=default_min, key="dcm_trend_ymin", disabled=dcm_chart_auto)
                with dcm_y_max_col:
                    dcm_chart_ymax = st.number_input("Y max", value=default_max, key="dcm_trend_ymax", disabled=dcm_chart_auto)

            # A whole day, midnight to midnight. Comparing one day against
            # another is far easier when both start and end at the same clock
            # time than when the window floats with whatever data is loaded.
            day_col, span_col = st.columns([1, 2])
            with day_col:
                whole_day = st.checkbox(
                    "Whole day view", value=False, key="dcm_whole_day",
                    help="Fix the axis to 00:00-23:59 on one date, so days can "
                         "be compared like for like.")
            day_start = day_end = None
            if whole_day:
                avail = pd.to_datetime(pivot_reset["created_at"], errors="coerce")
                dmin, dmax = avail.min(), avail.max()
                if pd.notna(dmin) and pd.notna(dmax):
                    # Reseed the picker's default whenever the data's most
                    # recent date moves (a new day, or a different range
                    # loaded) -- otherwise `value=dmax.date()` below is only
                    # honored the first time this widget key is created and
                    # freezes on whatever date it happened to see first.
                    _reseed_widget_value(
                        "dcm_day_pick", "_dcm_day_pick_signal",
                        dmax.date(), dmax.date(),
                    )
                    with span_col:
                        chosen = st.date_input(
                            "Date", value=dmax.date(),
                            min_value=dmin.date(), max_value=dmax.date(),
                            key="dcm_day_pick")
                    day_start = pd.Timestamp(chosen)
                    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    in_day = (avail >= day_start) & (avail <= day_end)
                    if in_day.sum() == 0:
                        st.warning(
                            f"No readings loaded for {chosen}. Use the "
                            f"historical append control above to fetch that day."
                        )
                    else:
                        pivot_reset = pivot_reset[in_day.values].copy()

            metric_axis_label = {
                "voltage_v": "Voltage (V)", "current_a": "Current (A)", "active_power_kw": "Power (kW)",
            }.get(metric_choice, "Energy (kWh)")

            if view == "Distribution":
                # Same numbers, different question. The trend says what a meter
                # is doing now; the distribution says what it usually does, so a
                # meter sitting left of the others is consistently low rather
                # than momentarily low.
                import plotly.graph_objects as _go
                hist = _go.Figure()
                for c in meter_cols:
                    vals = pd.to_numeric(pivot_reset[c], errors="coerce").dropna()
                    if len(vals):
                        hist.add_trace(_go.Histogram(x=vals, name=str(c),
                                                     opacity=0.55, nbinsx=40))
                hist.update_layout(
                    barmode="overlay", height=430,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Manrope, sans-serif", "color": "#0F1B2A"},
                    xaxis={"title": metric_axis_label, "gridcolor": "#E3E9EF"},
                    yaxis={"title": "Number of readings", "gridcolor": "#E3E9EF"},
                    legend={"font": {"size": 11}},
                )
                st.plotly_chart(hist, use_container_width=True)
                st.caption(
                    f"How often each meter sat at each {metric_axis_label} "
                    f"over the period shown."
                )
                return

            dcm_fig = plot_line_chart(
                pivot_reset, "created_at", meter_cols,
                # Fixing the axis to the whole day means a partial day reads as
                # partial, instead of being stretched to fill the plot and
                # looking like a complete one.
                x_range=(day_start.to_pydatetime(), day_end.to_pydatetime())
                        if day_start is not None else None,
                y_range=None,
                x_title="Time (Malaysia, UTC+8)",
                y_title=metric_axis_label,
            )
            st.plotly_chart(dcm_fig, use_container_width=True)

        with st.expander("Raw panel meter data table"):
            st.dataframe(df_panel, use_container_width=True, hide_index=True)

    _panel_meters_fragment()
