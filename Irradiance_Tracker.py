import calendar
from datetime import date

import streamlit as st

from ui_sections import (
    require_login,
    page_stamp,
    plot_irradiance_frequency,
    close_figures,
)
from drive_fetch import (
    list_available_csvs,
    download_and_combine_csvs,
    extract_year,
    extract_month,
    month_label,
    resolve_period_files,
)

MAX_IRRADIANCE = 1200  # W/m²


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
        file_ids = tuple((f["id"], f.get("modifiedTime")) for f in period_files)
        with st.spinner(f"Downloading {len(file_ids)} file(s) for {_period_label(start_date, end_date)}..."):
            df_period = download_and_combine_csvs(file_ids)
        st.session_state["_irr_tracker_df"] = df_period
        st.session_state["_irr_tracker_period"] = (start_date, end_date)
        st.success(f"Loaded {df_period.shape[0]} rows across {len(period_files)} file(s)")

    df_month = st.session_state.get("_irr_tracker_df")

    if df_month is None or df_month.empty:
        st.info("Load a period above to see the frequency charts.")
        return

    if st.session_state.get("_irr_tracker_period") != (start_date, end_date):
        st.info("You've changed the period — click 'Load period data' again to refresh the charts.")

    irr_cols = [c for c in df_month.columns if c.startswith("Irr_")]

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

    figs = plot_irradiance_frequency(
        df_month, selected_cols, bin_width=bin_width, max_irr=MAX_IRRADIANCE
    )

    # Two per row: 24 sensors stacked single-file meant a very long scroll and
    # no way to compare distributions side by side.
    for i in range(0, len(selected_cols), 2):
        pair = selected_cols[i:i + 2]
        cols = st.columns(len(pair))
        for slot, col in zip(cols, pair):
            with slot:
                st.pyplot(figs[col])

    # Every rerun built new figures and left them open, leaking memory for as
    # long as the app stayed up.
    close_figures(figs)
