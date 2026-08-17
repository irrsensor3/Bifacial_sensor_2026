import streamlit as st
import pandas as pd

from ui_sections import (
    require_login,
    page_stamp,
    plot_weather_signals,
    preview_report_content,
    generate_word_report,
    generate_pdf_report,
)
from drive_fetch import (
    list_available_csvs,
    download_csv_as_df,
    format_file_label,
    list_available_dcm_csvs,
    download_dcm_csv_as_df,
)


def render_data_reports():
    require_login()

    page_stamp("Data & Reports")
    st.title("📁 Data & Reports")

    # -------------------------
    # Report source: pick a CSV from Google Drive
    # -------------------------
    st.subheader("📁 Select Data Source (Google Drive)")

    available_files = list_available_csvs()

    df = None

    if not available_files:
        st.warning(
            "Couldn't find any CSVs in Drive — check that the "
            "'bifacial-data' folder is shared with the service account, "
            "and that rclone has synced at least one file."
        )
        if st.session_state.get("_drive_list_error"):
            with st.expander("Error details"):
                st.code(st.session_state["_drive_list_error"])
    else:
        file_labels = [format_file_label(f) for f in available_files]
        selected_idx = st.selectbox(
            "CSV file (most recent first)",
            options=range(len(available_files)),
            format_func=lambda i: file_labels[i],
            index=0,  # defaults to the newest file
        )
        selected_file = available_files[selected_idx]

        if st.button("📥 Load selected file"):
            try:
                df = download_csv_as_df(selected_file["id"])
                st.session_state["_loaded_df"] = df
                st.session_state["_loaded_filename"] = selected_file["name"]
                st.success(f"Loaded {selected_file['name']} ({df.shape[0]} rows)")
            except Exception as e:
                st.error(f"Couldn't download that file: {e}")

        # keep the loaded dataframe around across reruns (e.g. when
        # toggling widgets below) until a different file is explicitly loaded
        if df is None and "_loaded_df" in st.session_state:
            df = st.session_state["_loaded_df"]
            st.caption(f"Currently loaded: {st.session_state.get('_loaded_filename', '')}")

    # -------------------------
    # Optional: DC meter data from the separate panel-meter-data folder
    # -------------------------
    st.subheader("🔌 Select DC Meter Data (optional)")

    available_dcm_files = list_available_dcm_csvs()

    df_dcm = None

    if not available_dcm_files:
        st.caption(
            "No DC meter CSVs found in Drive — reports still work "
            "without this, it just won't include a DC Meter Summary section."
        )
        if st.session_state.get("_dcm_drive_list_error"):
            with st.expander("Error details"):
                st.code(st.session_state["_dcm_drive_list_error"])
    else:
        dcm_labels = [format_file_label(f) for f in available_dcm_files]
        dcm_selected_idx = st.selectbox(
            "DC meter CSV file (most recent first)",
            options=range(len(available_dcm_files)),
            format_func=lambda i: dcm_labels[i],
            index=0,
            key="dcm_file_select",
        )
        selected_dcm_file = available_dcm_files[dcm_selected_idx]

        if st.button("📥 Load DC meter file"):
            try:
                df_dcm = download_dcm_csv_as_df(selected_dcm_file["id"])
                st.session_state["_loaded_dcm_df"] = df_dcm
                st.session_state["_loaded_dcm_filename"] = selected_dcm_file["name"]
                st.success(f"Loaded {selected_dcm_file['name']} ({df_dcm.shape[0]} rows)")
            except Exception as e:
                st.error(f"Couldn't download that file: {e}")

        if df_dcm is None and "_loaded_dcm_df" in st.session_state:
            df_dcm = st.session_state["_loaded_dcm_df"]
            st.caption(f"Currently loaded: {st.session_state.get('_loaded_dcm_filename', '')}")

    report_title = st.text_input("Report Title", "Bifacial PV Performance Report")
    observation = st.text_area("Observation Notes")

    if df is not None:
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(100))

        st.subheader("📌 Dataset Info")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        time = df["Time"] if "Time" in df.columns else df.index

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        st.subheader("📈 Graph Configuration")

        # if a different file was loaded earlier, drop any selected columns
        # that no longer exist so the multiselect widget doesn't error out
        if "selected_temps" not in st.session_state:
            st.session_state.selected_temps = [c for c in numeric_cols if "temp" in c.lower()]
        else:
            st.session_state.selected_temps = [c for c in st.session_state.selected_temps if c in numeric_cols]

        temp_label_col, temp_all_col, temp_none_col = st.columns([4, 1, 1])
        with temp_label_col:
            st.caption("Temperature Columns")
        with temp_all_col:
            if st.button("Select all", key="temp_select_all", use_container_width=True):
                st.session_state.selected_temps = numeric_cols
        with temp_none_col:
            if st.button("Remove all", key="temp_remove_all", use_container_width=True):
                st.session_state.selected_temps = []

        selected_temps = st.multiselect(
            "Select Temperature Columns",
            numeric_cols,
            key="selected_temps",
            label_visibility="collapsed",
        )

        if "selected_irradiance" not in st.session_state:
            st.session_state.selected_irradiance = [c for c in numeric_cols if "irr" in c.lower()]
        else:
            st.session_state.selected_irradiance = [c for c in st.session_state.selected_irradiance if c in numeric_cols]

        irr_label_col, irr_all_col, irr_none_col = st.columns([4, 1, 1])
        with irr_label_col:
            st.caption("Irradiance Columns")
        with irr_all_col:
            if st.button("Select all", key="irr_select_all", use_container_width=True):
                st.session_state.selected_irradiance = numeric_cols
        with irr_none_col:
            if st.button("Remove all", key="irr_remove_all", use_container_width=True):
                st.session_state.selected_irradiance = []

        selected_irradiance = st.multiselect(
            "Select Irradiance Columns",
            numeric_cols,
            key="selected_irradiance",
            label_visibility="collapsed",
        )

        temperatures = {col: df[col].tolist() for col in selected_temps}
        irradiances = {col: df[col].tolist() for col in selected_irradiance}

        if selected_temps or selected_irradiance:
            with st.expander("📐 Chart scale (optional)"):
                st.caption(
                    "Time is a categorical axis here, so the X range trims "
                    "which rows are plotted rather than setting numeric bounds."
                )
                n_rows = len(time)
                x_start, x_end = st.slider(
                    "X range (row index)", 0, n_rows - 1, (0, n_rows - 1)
                )

                temp_y_col, irr_y_col = st.columns(2)
                with temp_y_col:
                    st.caption("Temperature Y-axis (°C)")
                    temp_auto = st.checkbox("Auto", value=True, key="temp_y_auto")
                    temp_y_min = st.number_input("Min", value=0.0, key="temp_y_min", disabled=temp_auto)
                    temp_y_max = st.number_input("Max", value=50.0, key="temp_y_max", disabled=temp_auto)
                with irr_y_col:
                    st.caption("Irradiance Y-axis (W/m²)")
                    irr_auto = st.checkbox("Auto", value=True, key="irr_y_auto")
                    irr_y_min = st.number_input("Min", value=0.0, key="irr_y_min", disabled=irr_auto)
                    irr_y_max = st.number_input("Max", value=1200.0, key="irr_y_max", disabled=irr_auto)

            sliced_time = time[x_start:x_end + 1]
            sliced_temps = {col: vals[x_start:x_end + 1] for col, vals in temperatures.items()}
            sliced_irr = {col: vals[x_start:x_end + 1] for col, vals in irradiances.items()}

            fig = plot_weather_signals(
                sliced_time,
                sliced_temps,
                sliced_irr,
                temp_ylim=None if temp_auto else (temp_y_min, temp_y_max),
                irr_ylim=None if irr_auto else (irr_y_min, irr_y_max),
            )
            st.pyplot(fig)

            st.subheader("📄 Generate Reports")

            if st.button("👁️ Preview Report"):
                st.session_state.show_preview = True

            if st.session_state.get("show_preview"):
                with st.expander("📋 Report Preview", expanded=True):
                    preview_report_content(df, report_title, observation, fig, df_dcm=df_dcm)

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    report = generate_word_report(df, report_title, observation, fig, df_dcm=df_dcm)
                    st.download_button(
                        label="⬇️ Download Word Report",
                        data=report,
                        file_name="PV_Report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

                with col2:
                    report = generate_pdf_report(df, report_title, observation, fig, df_dcm=df_dcm)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=report,
                        file_name="PV_Report.pdf",
                        mime="application/pdf"
                    )
