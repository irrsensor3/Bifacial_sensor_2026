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
)

MAX_IRRADIANCE = 1200  # W/m²


def render_irradiance_tracker():
    require_login()

    page_stamp("Irradiance Tracker")
    st.title("Irradiance tracker")
    st.caption(
        f"Frequency distribution of irradiance readings (0-{MAX_IRRADIANCE} W/m²), "
        "built from every CSV synced from Google Drive for the selected month — "
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

    years = sorted({extract_year(f) for f in available_files}, reverse=True)
    if not years:
        st.warning("Found CSVs in Drive, but none had a recognisable date in the filename.")
        return
    year_col, month_col = st.columns(2)
    with year_col:
        selected_year = st.selectbox("Year", years, index=0)

    year_files = [f for f in available_files if extract_year(f) == selected_year]
    months = sorted({extract_month(f) for f in year_files})
    if not months:
        st.warning(f"No files with a recognisable month for {selected_year}.")
        return
    with month_col:
        # index=len(months)-1 raised IndexError on an empty list
        selected_month = st.selectbox("Month", months, index=len(months) - 1,
                                      format_func=month_label)

    period_files = [f for f in year_files if extract_month(f) == selected_month]
    st.caption(f"{len(period_files)} file(s) found for {month_label(selected_month)} {selected_year}")

    if st.button("Load month data", type="primary"):
        file_ids = tuple(f["id"] for f in period_files)
        with st.spinner(f"Downloading {len(file_ids)} file(s) for {month_label(selected_month)} {selected_year}..."):
            df_month = download_and_combine_csvs(file_ids)
        st.session_state["_irr_tracker_df"] = df_month
        st.session_state["_irr_tracker_period"] = (selected_year, selected_month)
        st.success(f"Loaded {df_month.shape[0]} rows across {len(period_files)} file(s)")

    df_month = st.session_state.get("_irr_tracker_df")

    if df_month is None or df_month.empty:
        st.info("Load a month's data above to see the frequency charts.")
        return

    if st.session_state.get("_irr_tracker_period") != (selected_year, selected_month):
        st.info("You've changed the period — click 'Load month data' again to refresh the charts.")

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
