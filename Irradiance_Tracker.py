import calendar
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from ui_sections import (
    require_login,
    page_stamp,
)
from drive_fetch import (
    list_available_csvs,
    download_csv_as_df,
    extract_year,
    extract_month,
    month_label,
    resolve_period_files,
)

MAX_IRRADIANCE = 1200  # W/m²

# ---------------------------------------------------------------------------
#  Counts, not rows
# ---------------------------------------------------------------------------
# This page used to download every CSV in the period, concatenate them into
# one frame and keep it in session state -- a month is half a million rows by
# 74 columns, several hundred megabytes, which on its own is most of what
# Streamlit Community Cloud allows. Then all 24 histograms were redrawn on
# every rerun, so even nudging the bin-width slider took seconds.
#
# A histogram only needs counts. Each file is now reduced, once, to counts in
# fine 10 W/m² bins per sensor (24 x 120 integers), and those counts are
# cached on disk against the file's id and modifiedTime. A period is the sum
# of its files' counts, any bin width the slider offers (a multiple of 10) is
# an exact regrouping of the fine bins, and the raw data is never held. The
# charts are also cached as images, so changing the sensor selection redraws
# nothing.
FINE_BIN = 10
_FINE_EDGES = np.arange(0, MAX_IRRADIANCE + FINE_BIN, FINE_BIN)


@st.cache_data(ttl=None, persist="disk", max_entries=5000, show_spinner=False)
def _file_counts(file_id: str, modified_time: str):
    """(rows, {Irr_n: counts in 10 W/m² bins}) for one day's CSV.

    Same filtering as before: non-numeric readings are dropped, and so is
    anything outside 0..MAX_IRRADIANCE.
    """
    df = download_csv_as_df(file_id)
    counts = {}
    for col in df.columns:
        if not str(col).startswith("Irr_"):
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        v = v[~np.isnan(v)]
        v = v[(v >= 0) & (v <= MAX_IRRADIANCE)]
        counts[col] = np.histogram(v, bins=_FINE_EDGES)[0].astype(np.int64)
    return len(df), counts


def _regroup(fine_counts, bin_width):
    """Fine 10 W/m² counts regrouped into bin_width bins, using exactly the
    edges the page always used: 0, w, 2w, ... up to and past MAX_IRRADIANCE."""
    edges = np.arange(0, MAX_IRRADIANCE + bin_width, bin_width)
    target = (_FINE_EDGES[:-1] // bin_width).astype(int)
    coarse = np.bincount(target, weights=fine_counts, minlength=len(edges) - 1)
    return edges, coarse[:len(edges) - 1]


@st.cache_data(max_entries=400, show_spinner=False)
def _histogram_png(col: str, fine_bytes: bytes, bin_width: int):
    """One sensor's histogram as PNG bytes, drawn the way the page always drew
    it (same size, colours and axes). Keyed on the counts themselves, so a
    chart is only ever drawn once per period and bin width."""
    fine = np.frombuffer(fine_bytes, dtype=np.int64)
    edges, coarse = _regroup(fine, bin_width)
    # Figure() rather than plt.subplots(): not registered with pyplot, so it
    # can't leak between reruns and needs no close. Fixed margins instead of
    # tight_layout()/bbox_inches="tight", which each lay the chart out again
    # and together nearly doubled the drawing time -- with 24 charts that was
    # the difference between a slider that responds and one that doesn't.
    fig = Figure(figsize=(10, 4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.14)
    ax = fig.subplots()
    ax.hist(edges[:-1], bins=edges, weights=coarse, color="#f5a623",
            edgecolor="black")
    ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel("Frequency")
    ax.set_title(str(col))
    ax.set_xlim(0, MAX_IRRADIANCE)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    return buf.getvalue()


def _sensor_sort_key(col):
    try:
        return int(str(col).split("_", 1)[1])
    except (IndexError, ValueError):
        return 10_000


def _period_picker(available_files):
    """Month / Year / Date range / From date / Until date selector, same
    pattern as the historical-append controls on Live Monitoring. Returns
    (start_date, end_date, period_files) or (None, None, []) if nothing
    valid is selected yet."""

    years = sorted({int(y) for f in available_files if (y := extract_year(f)).isdigit()})
    if not years:
        st.warning("Found CSVs in Drive, but none had a recognisable date in the filename.")
        return None, None, []

    earliest, latest = date(min(years), 1, 1), date(max(years), 12, 31)

    mode = st.radio(
        "Range",
        ["Month", "Year", "Date range", "From date", "Until date"],
        horizontal=True,
        key="irr_tracker_mode",
    )

    start_date = end_date = None

    if mode == "Month":
        yr_col, mo_col = st.columns(2)
        with yr_col:
            yr = st.selectbox("Year", sorted(years, reverse=True), key="irr_tracker_year")
        year_files = [f for f in available_files if extract_year(f) == str(yr)]
        months = sorted({extract_month(f) for f in year_files if extract_month(f).isdigit()})
        if not months:
            st.warning(f"No files with a recognisable month for {yr}.")
            return None, None, []
        with mo_col:
            mo = st.selectbox(
                "Month", months, index=len(months) - 1,
                format_func=month_label, key="irr_tracker_month",
            )
        start_date = date(yr, int(mo), 1)
        end_date = date(yr, int(mo), calendar.monthrange(yr, int(mo))[1])

    elif mode == "Year":
        yr = st.selectbox("Year", sorted(years, reverse=True), key="irr_tracker_yronly")
        start_date, end_date = date(yr, 1, 1), date(yr, 12, 31)

    elif mode == "Date range":
        picked = st.date_input(
            "From / to", value=(earliest, latest),
            min_value=earliest, max_value=latest, key="irr_tracker_range",
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            start_date, end_date = picked
        else:
            st.caption("Pick both a start and an end date.")

    elif mode == "From date":
        start_date = st.date_input(
            "From", value=earliest, min_value=earliest, max_value=latest,
            key="irr_tracker_from",
        )
        end_date = latest

    elif mode == "Until date":
        end_date = st.date_input(
            "Until", value=latest, min_value=earliest, max_value=latest,
            key="irr_tracker_until",
        )
        start_date = earliest

    if not start_date or not end_date or start_date > end_date:
        st.caption("Pick a valid range.")
        return None, None, []

    period_files = resolve_period_files(available_files, start_date, end_date)
    return start_date, end_date, period_files


def _period_label(start_date, end_date):
    if start_date == end_date:
        return f"{start_date:%d %b %Y}"
    return f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"


def render_irradiance_tracker():
    require_login()

    page_stamp("Irradiance Tracker")
    st.title("Irradiance tracker")
    st.caption(
        f"Frequency distribution of irradiance readings (0-{MAX_IRRADIANCE} W/m²), "
        "built from every CSV synced from Google Drive for the selected period — "
        "one chart per sensor."
    )

    available_files = list_available_csvs()

    if not available_files:
        st.warning(
            "Couldn't find any CSVs in Drive — check that the "
            "'bifacial-data' folder is shared with the service account, "
            "and that rclone has synced at least one file."
        )
        if st.session_state.get("_drive_list_error"):
            with st.expander("Error details"):
                st.code(st.session_state["_drive_list_error"])
        return

    start_date, end_date, period_files = _period_picker(available_files)

    if start_date is None:
        return

    st.caption(f"{len(period_files)} file(s) found for {_period_label(start_date, end_date)}")

    if st.button("Load period data", type="primary", disabled=not period_files):
        label = _period_label(start_date, end_date)
        progress = st.progress(0.0, text=f"Reading {len(period_files)} file(s) for {label}…")
        totals, rows, failed = {}, 0, []
        ordered = sorted(period_files, key=lambda f: f.get("name", ""))
        for n, f in enumerate(ordered, start=1):
            progress.progress(n / len(ordered),
                              text=f"{f.get('name', '')}  ({n} of {len(ordered)})")
            try:
                n_rows, counts = _file_counts(f["id"], f.get("modifiedTime") or "")
            except Exception as exc:
                failed.append(f"{f.get('name', f['id'])}: {exc}")
                continue
            rows += n_rows
            for col, c in counts.items():
                totals[col] = totals[col] + c if col in totals else c.copy()
        progress.empty()
        st.session_state["_irr_tracker_counts"] = totals
        st.session_state["_irr_tracker_rows"] = rows
        st.session_state["_irr_tracker_period"] = (start_date, end_date)
        # The old page kept the whole period's rows here; drop any left over
        # from before this version so they don't sit in memory.
        st.session_state.pop("_irr_tracker_df", None)
        st.success(f"Loaded {rows:,} rows across {len(period_files) - len(failed)} file(s)")
        if failed:
            with st.expander(f"{len(failed)} file(s) could not be read"):
                for msg in failed[:20]:
                    st.write(msg)

    totals = st.session_state.get("_irr_tracker_counts")

    if not totals:
        st.info("Load a period above to see the frequency charts.")
        return

    if st.session_state.get("_irr_tracker_period") != (start_date, end_date):
        st.info("You've changed the period — click 'Load period data' again to refresh the charts.")

    irr_cols = sorted(totals, key=_sensor_sort_key)

    if not irr_cols:
        st.warning("No Irr_ columns found in the loaded data.")
        return

    if "irr_tracker_selected" not in st.session_state:
        st.session_state.irr_tracker_selected = irr_cols[:]
    else:
        st.session_state.irr_tracker_selected = [
            c for c in st.session_state.irr_tracker_selected if c in irr_cols
        ]

    label_col, all_col, none_col = st.columns([4, 1, 1])
    with label_col:
        st.caption("Irradiance sensors to include")
    with all_col:
        if st.button("Select all", key="irr_tracker_select_all", use_container_width=True):
            st.session_state.irr_tracker_selected = irr_cols
    with none_col:
        if st.button("Remove all", key="irr_tracker_remove_all", use_container_width=True):
            st.session_state.irr_tracker_selected = []

    selected_cols = st.multiselect(
        "Irradiance sensors to include",
        irr_cols,
        key="irr_tracker_selected",
        label_visibility="collapsed",
    )

    bin_width = st.slider("Bin width (W/m²)", min_value=10, max_value=200, value=50, step=10)

    if not selected_cols:
        st.info("Select at least one irradiance sensor to see its frequency chart.")
        return

    # Two per row: 24 sensors stacked single-file meant a very long scroll and
    # no way to compare distributions side by side.
    for i in range(0, len(selected_cols), 2):
        pair = selected_cols[i:i + 2]
        cols = st.columns(len(pair))
        for slot, col in zip(cols, pair):
            with slot:
                st.image(_histogram_png(col, totals[col].tobytes(), int(bin_width)))
