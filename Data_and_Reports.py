import streamlit as st
import pandas as pd

from ui_sections import (
    require_login,
    plot_weather_signals,
    preview_report_content,
    generate_word_report,
    generate_pdf_report,
)
from drive_fetch import list_available_csvs, download_csv_as_df, format_file_label


def render_data_reports():
    require_login()

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
