import streamlit as st

from ui_sections import require_login, page_stamp, plot_irradiance_frequency
from drive_fetch import list_available_csvs, download_and_combine_csvs, extract_year

MAX_IRRADIANCE = 1200  # W/m²


def render_irradiance_tracker():
    require_login()

    page_stamp("Irradiance Tracker")
    st.title("📈 Annual Irradiance Tracker")
    st.caption(
        f"Frequency distribution of irradiance readings (0-{MAX_IRRADIANCE} W/m²), "
        "built from every CSV synced from Google Drive for the selected year — "
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
    selected_year = st.selectbox("Year", years, index=0)

    year_files = [f for f in available_files if extract_year(f) == selected_year]
    st.caption(f"{len(year_files)} file(s) found for {selected_year}")

    if st.button("📥 Load year data"):
        file_ids = tuple(f["id"] for f in year_files)
        with st.spinner(f"Downloading {len(file_ids)} file(s) for {selected_year}..."):
            df_year = download_and_combine_csvs(file_ids)
        st.session_state["_irr_tracker_df"] = df_year
        st.session_state["_irr_tracker_year"] = selected_year
        st.success(f"Loaded {df_year.shape[0]} rows across {len(year_files)} file(s)")

    df_year = st.session_state.get("_irr_tracker_df")

    if df_year is None or df_year.empty:
        st.info("Load a year's data above to see the frequency charts.")
        return

    if st.session_state.get("_irr_tracker_year") != selected_year:
        st.info("You've changed the year — click 'Load year data' again to refresh the charts.")

    irr_cols = [c for c in df_year.columns if c.startswith("Irr_")]

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
        df_year, selected_cols, bin_width=bin_width, max_irr=MAX_IRRADIANCE
    )

    for col in selected_cols:
        st.pyplot(figs[col])
