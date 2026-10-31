"""Data & Reports.

Two ways out of this page:

  single   the loaded file, framed exactly as it is on screen, as one Word or
           PDF document.
  batch    the same file cut into days, months or years -- one document per
           period, delivered as a single ZIP.

Either can carry the weather channels (irradiance and temperature), the DC
meter data, or both.

Report formats (title, notes, Word/PDF, columns, split, Y-axis limits) can be
saved by name and re-applied in later sessions -- see report_presets.py.

Changes from the previous version:

  * A DC-meter-only report no longer crashes. It has no weather chart, and the
    report generators now treat "no figure" as a normal case.

  * Timestamps. The raw irradiance CSVs carry separate Date and Time columns.
    The old code picked "Time" on its own, which parses as time-of-day on
    today's date, so every row landed in ONE period and "Split by Day" built a
    single report. Date and Time are now combined. DC-meter timestamps arrive
    as naive UTC, so they are shifted to site time first; otherwise the
    07:00-08:00 readings fell into the previous day's report.

  * The file dropdowns sort by the day in the filename. They used to sort by
    Drive's modified time, so an old day that rclone re-synced jumped to the
    top and became the default "newest" file.

  * The batch ZIP, like the single report, is withdrawn when the data or
    settings change after it was built.
"""

import hashlib
import io
import re
import zipfile
from datetime import date

import pandas as pd
import streamlit as st

from drive_fetch import (
    list_available_csvs,
    download_csv_as_df,
    format_file_label,
    list_available_dcm_csvs,
    download_dcm_csv_as_df,
)
from report_presets import render_preset_bar
from ui_sections import (
    SITE_ZONE,
    require_login,
    page_stamp,
    close_figures,
    plot_weather_signals,
    preview_report_content,
    generate_word_report,
    generate_pdf_report,
)


# ---------------------------------------------------------------------------
#  Batch limits
# ---------------------------------------------------------------------------
# Each report embeds rendered matplotlib images, so a year of daily documents
# is hundreds of megabytes held in memory at once -- against roughly 1 GB on
# Streamlit Community Cloud, which is enough to have the app killed halfway
# through. The period picker defaults to the first MAX_REPORTS periods rather
# than refusing outright, so a long file still produces something useful and
# you can pick the rest in a second pass.
MAX_REPORTS = 120

# A period with almost nothing in it produces a document of empty tables.
MIN_ROWS_PER_REPORT = 10

PERIOD_CODE = {"Day": "D", "Month": "M", "Year": "Y"}

# Column names worth trying before falling back to parsing. Checked
# case-insensitively.
TIME_COLUMN_NAMES = ("time", "timestamp", "datetime", "date", "created_at", "ts")

CONTENT_WEATHER = "Irradiance & temperature"
CONTENT_DCM = "DC meter"
CONTENT_BOTH = "Both"


# ---------------------------------------------------------------------------
#  Chart size limits
# ---------------------------------------------------------------------------
# Matplotlib cannot show more points than the figure has pixels, but it will
# happily try. With 72 columns selected on a 7,590-row file that is 546,000
# points, each one first copied into a Python list -- minutes of work for a
# picture that is a few hundred pixels wide. _thin caps the rows, and
# _axis_time hands matplotlib real datetimes so the time axis is numeric
# rather than thousands of text categories.
CHART_MAX_POINTS = 3_000

# Past this many lines the chart is unreadable anyway, and it is nearly always
# an accident -- "Select all" on a 48-column file, or Irr_* and IrrAvg_* both
# matching the same guess.
CHART_WARN_SERIES = 16


def _thin(frame):
    """Every Nth row, so a chart carries at most CHART_MAX_POINTS."""
    if frame is None or len(frame) <= CHART_MAX_POINTS:
        return frame, 1
    step = len(frame) // CHART_MAX_POINTS + 1
    return frame.iloc[::step], step


# ---------------------------------------------------------------------------
#  Timestamps
# ---------------------------------------------------------------------------

def _find_time_column(frame):
    """The column holding the timestamps, or None.

    Tries the usual names first. Only if none of them are present does it fall
    back to parsing, and then only on text columns and only on the first 200
    rows -- parsing 24 float columns of a 124,000-row file to discover that
    none of them are dates is a slow way to learn nothing.
    """
    if frame is None or frame.empty:
        return None

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in TIME_COLUMN_NAMES:
        if name in lowered:
            return lowered[name]

    head = frame.head(200)
    for col in frame.columns:
        dtype = str(head[col].dtype)
        if not (dtype == "object" or dtype.startswith("datetime")
                or dtype.startswith("str")):
            continue
        parsed = pd.to_datetime(head[col], errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return col
    return None


def _timestamps(frame):
    """A datetime Series aligned to the frame's own index, or None.

    Order of preference: a datetime index; a Date column plus a Time column
    (the raw irradiance CSVs -- "Time" alone is only a time of day, and parses
    as today's date); then whichever single column looks like timestamps.
    """
    if frame is None or frame.empty:
        return None

    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, index=frame.index)

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    date_col, time_col = lowered.get("date"), lowered.get("time")
    if date_col is not None and time_col is not None:
        stamped = (frame[date_col].astype(str).str.strip() + " "
                   + frame[time_col].astype(str).str.strip())
        # Naming the format matters at this size: left to infer, pandas hands
        # every one of a day's ~17,000 rows to dateutil individually.
        ts = pd.to_datetime(stamped, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if ts.notna().sum() == 0:
            ts = pd.to_datetime(stamped, errors="coerce")
        if ts.notna().sum() > 0:
            return ts

    col = _find_time_column(frame)
    if col is None:
        return None
    ts = pd.to_datetime(frame[col], errors="coerce")
    return ts if ts.notna().sum() > 0 else None


def _axis_time(frame):
    """The x values for a chart: real datetimes where possible.

    Falls back to the raw column, which plots as categories -- correct, just
    slow -- and finally to the index.
    """
    ts = _timestamps(frame)
    if ts is not None and ts.notna().mean() >= 0.5:
        return ts
    col = _find_time_column(frame)
    return frame.index if col is None else frame[col]


def _periods(frame, granularity):
    """Period labels for every row, aligned to the frame's index."""
    ts = _timestamps(frame)
    if ts is None:
        return None
    return ts.dt.to_period(PERIOD_CODE[granularity])


def _slice(frame, periods, target):
    """The rows of `frame` falling inside one period."""
    if frame is None or periods is None:
        return None
    mask = (periods == target).fillna(False)
    part = frame[mask.values]
    return part if len(part) else None


def _dcm_to_local(frame):
    """Shift the DC meter's created_at from naive UTC to site time.

    drive_fetch localises the file's local timestamps and converts them to
    UTC, which is what the live pages want. For a per-day report it puts the
    07:00-08:00 readings (23:00-24:00 UTC) on the previous day.
    """
    if frame is None or "created_at" not in frame.columns:
        return frame
    out = frame.copy()
    ts = pd.to_datetime(out["created_at"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    out["created_at"] = ts.dt.tz_convert(SITE_ZONE).dt.tz_localize(None)
    return out


def _slug(text, fallback="report"):
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return out or fallback


# ---------------------------------------------------------------------------
#  File listings
# ---------------------------------------------------------------------------

def _newest_first(files):
    """Newest DATA first. drive_fetch sorts by Drive's modified time, which an
    rclone re-sync of an old day resets -- that day then headed the list and
    was preselected as the 'most recent' file."""
    return sorted(
        files or [],
        key=lambda f: (f.get("data_date") or date.min, f.get("modifiedTime", "")),
        reverse=True,
    )


def _file_label(entry):
    day = entry.get("data_date")
    base = format_file_label(entry)
    return f"{day} · {base}" if day else base


# ---------------------------------------------------------------------------
#  Report building
# ---------------------------------------------------------------------------

def _figure_for(part, temps, irrs, temp_ylim, irr_ylim):
    """A weather chart for one slice, or None if there is nothing to draw.

    The caller owns the figure and must close it -- matplotlib keeps every
    figure in a global registry, and a batch of ninety that are never released
    is how this app runs out of memory.
    """
    if part is None or part.empty:
        return None

    cols_t = [c for c in temps if c in part.columns]
    cols_i = [c for c in irrs if c in part.columns]
    if not cols_t and not cols_i:
        return None

    thin, _step = _thin(part)
    return plot_weather_signals(
        _axis_time(thin),
        {c: thin[c].tolist() for c in cols_t},
        {c: thin[c].tolist() for c in cols_i},
        temp_ylim=temp_ylim,
        irr_ylim=irr_ylim,
    )


def _build_report(fmt, df_part, dcm_part, title, observation, fig):
    """One document, as bytes, plus the extension and MIME type to serve it as."""
    if fmt.startswith("Word"):
        data = generate_word_report(df_part, title, observation, fig,
                                    df_dcm=dcm_part)
        return data, "docx", (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document")

    data = generate_pdf_report(df_part, title, observation, fig,
                               df_dcm=dcm_part)
    return data, "pdf", "application/pdf"


def _report_parts(content, df_part, dcm_part):
    """What to hand the generator for the chosen content.

    An empty DataFrame rather than None for the weather half of a DC-only
    report, because the generators take it as a positional argument.
    """
    if content == CONTENT_DCM:
        return pd.DataFrame(), dcm_part
    if content == CONTENT_WEATHER:
        return df_part, None
    return df_part, dcm_part


# ---------------------------------------------------------------------------
#  Single report
# ---------------------------------------------------------------------------

def _render_single(df, df_dcm, report_title, observation, fig, signature):
    st.subheader("Report")
    st.caption("The file as it is framed above, as one document.")

    if st.button("Preview report"):
        st.session_state.show_preview = True

    if st.session_state.get("show_preview"):
        with st.expander("Report preview", expanded=True):
            preview_report_content(df, report_title, observation, fig,
                                   df_dcm=df_dcm)

    # Build only on request: both documents used to be generated on EVERY
    # rerun, so moving a slider silently rendered a Word file and a PDF.
    fmt_col, build_col = st.columns([2, 1])
    with fmt_col:
        fmt = st.radio("Report format", ["Word (.docx)", "PDF"],
                       horizontal=True, key="report_format")
    with build_col:
        st.caption(" ")
        build = st.button("Build report", type="primary",
                          use_container_width=True)

    if build:
        with st.spinner(f"Building the {fmt} report…"):
            try:
                data, ext, mime = _build_report(
                    fmt, df, df_dcm, report_title, observation, fig)
                st.session_state["_report_bytes"] = data
                st.session_state["_report_name"] = \
                    f"{_slug(report_title, 'PV_Report')}.{ext}"
                st.session_state["_report_mime"] = mime
                st.session_state["_report_sig"] = signature + (fmt,)
            except Exception as exc:
                st.session_state["_report_bytes"] = None
                st.session_state["_report_sig"] = None
                st.error(f"The report could not be built: {exc}")

    # Withdraw a stale build rather than serving last file's numbers under
    # this file's name.
    if st.session_state.get("_report_bytes") is not None:
        if st.session_state.get("_report_sig") != signature + (fmt,):
            st.session_state["_report_bytes"] = None
            st.caption("The data or framing changed — build the report again.")

    if st.session_state.get("_report_bytes") is not None:
        st.download_button(
            label=f"Download {st.session_state['_report_name']}",
            data=st.session_state["_report_bytes"],
            file_name=st.session_state["_report_name"],
            mime=st.session_state["_report_mime"],
            type="primary",
        )


# ---------------------------------------------------------------------------
#  Batch reports
# ---------------------------------------------------------------------------

def _render_batch(df, df_dcm, report_title, observation,
                  selected_temps, selected_irradiance,
                  temp_ylim, irr_ylim):
    st.divider()
    st.subheader("Batch reports")
    st.caption(
        "Cut the loaded file into periods and build one document per period. "
        "Everything arrives as a single ZIP."
    )

    content = st.radio(
        "Include", [CONTENT_WEATHER, CONTENT_DCM, CONTENT_BOTH],
        horizontal=True, key="batch_content")

    use_weather = content in (CONTENT_WEATHER, CONTENT_BOTH)
    use_dcm = content in (CONTENT_DCM, CONTENT_BOTH)

    if use_weather and df is None:
        st.warning("No sensor CSV is loaded, so there is nothing to split.")
        return
    if use_dcm and df_dcm is None:
        st.warning("No DC meter CSV is loaded. Load one above, or choose "
                   f"“{CONTENT_WEATHER}”.")
        return

    gran = st.radio("Split by", ["Day", "Month", "Year"],
                    horizontal=True, key="batch_gran")

    per_weather = _periods(df, gran) if use_weather else None
    per_dcm = _periods(df_dcm, gran) if use_dcm else None

    if use_weather and per_weather is None and (not use_dcm or per_dcm is None):
        st.error(
            "Couldn't find a timestamp column to split on. The file needs a "
            "Date and Time column, or a single column named Timestamp, "
            "Datetime or created_at, or a datetime index."
        )
        return
    if use_dcm and per_dcm is None and (not use_weather or per_weather is None):
        st.error("The DC meter file has no usable timestamps to split on.")
        return

    if use_weather and per_weather is None:
        st.warning("The sensor file has no usable timestamps; only the DC "
                   "meter data will be split.")
    if use_dcm and per_dcm is None:
        st.warning("The DC meter file has no usable timestamps; only the "
                   "sensor data will be split.")

    keys = set()
    for per in (per_weather, per_dcm):
        if per is not None:
            keys |= set(per.dropna().unique())
    keys = sorted(keys)

    if not keys:
        st.error("No complete periods found in the loaded data.")
        return

    by_label = {str(k): k for k in keys}
    labels = list(by_label)

    if len(labels) > MAX_REPORTS:
        st.info(
            f"{len(labels)} periods in this file. The first {MAX_REPORTS} are "
            f"selected — building all of them at once risks running the app "
            f"out of memory. Adjust the selection below and run a second pass "
            f"for the rest."
        )

    # The key changes with the options, so switching Split-by (or loading a
    # different file) starts from a fresh default instead of carrying over
    # labels that no longer exist.
    options_id = hashlib.md5("|".join(labels).encode()).hexdigest()[:10]
    chosen = st.multiselect(
        "Periods", options=labels, default=labels[:MAX_REPORTS],
        key=f"batch_periods_{options_id}",
        help="Every period found in the loaded file. Deselect any you don't "
             "need.")

    fmt = st.radio("Report format", ["Word (.docx)", "PDF"],
                   horizontal=True, key="batch_format")

    if not chosen:
        st.caption("Select at least one period.")
        return

    # What this ZIP would be built from. If any of it differs from the ZIP
    # sitting in session state, that ZIP describes something else.
    signature = (
        st.session_state.get("_loaded_filename") if use_weather else None,
        st.session_state.get("_loaded_dcm_filename") if use_dcm else None,
        report_title, observation, content, gran, fmt, tuple(chosen),
        tuple(selected_temps), tuple(selected_irradiance), temp_ylim, irr_ylim,
    )

    st.caption(f"{len(chosen)} document(s) will be built.")

    if not st.button(f"Build {len(chosen)} report(s)", type="primary",
                     key="batch_build"):
        _offer_batch_download(signature)
        return

    progress = st.progress(0.0, text="Starting…")
    written, skipped, failures = 0, [], []
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, label in enumerate(chosen, start=1):
            progress.progress(n / len(chosen),
                              text=f"{label}  ({n} of {len(chosen)})")
            target = by_label[label]

            part = _slice(df, per_weather, target) if use_weather else None
            dcm_part = _slice(df_dcm, per_dcm, target) if use_dcm else None

            rows = (0 if part is None else len(part)) + \
                   (0 if dcm_part is None else len(dcm_part))
            if rows < MIN_ROWS_PER_REPORT:
                skipped.append(label)
                continue

            fig = None
            try:
                if use_weather:
                    fig = _figure_for(part, selected_temps, selected_irradiance,
                                      temp_ylim, irr_ylim)

                df_arg, dcm_arg = _report_parts(content, part, dcm_part)
                data, ext, _mime = _build_report(
                    fmt, df_arg, dcm_arg,
                    f"{report_title} — {label}", observation, fig)

                zf.writestr(f"{_slug(report_title, 'PV_Report')}_{label}.{ext}",
                            data)
                written += 1
            except Exception as exc:
                failures.append((label, str(exc)))
            finally:
                # Released every iteration, not at the end: ninety open
                # figures is the whole memory budget.
                if fig is not None:
                    close_figures([fig])

    progress.empty()

    if failures:
        st.error(f"{len(failures)} period(s) failed.")
        with st.expander("What went wrong"):
            for label, msg in failures[:20]:
                st.write(f"**{label}** — {msg}")

    if skipped:
        st.caption(
            f"Skipped {len(skipped)} period(s) with fewer than "
            f"{MIN_ROWS_PER_REPORT} rows: {', '.join(skipped[:10])}"
            + ("…" if len(skipped) > 10 else ""))

    if not written:
        st.session_state["_batch_zip"] = None
        st.warning("Nothing was built, so there is no ZIP to download.")
        return

    st.session_state["_batch_zip"] = buf.getvalue()
    st.session_state["_batch_sig"] = signature
    st.session_state["_batch_name"] = (
        f"{_slug(report_title, 'PV_Reports')}_{gran.lower()}ly.zip")
    st.success(f"Built {written} report(s).")
    _offer_batch_download(signature)


def _offer_batch_download(signature):
    data = st.session_state.get("_batch_zip")
    if not data:
        return
    if st.session_state.get("_batch_sig") != signature:
        st.caption("The data or settings changed since the last ZIP was "
                   "built — build again to refresh it.")
        return
    st.download_button(
        label=f"Download {st.session_state['_batch_name']} "
              f"({len(data) / 1e6:.1f} MB)",
        data=data,
        file_name=st.session_state["_batch_name"],
        mime="application/zip",
        type="primary",
        key="batch_download",
    )


# ---------------------------------------------------------------------------
#  Page
# ---------------------------------------------------------------------------

def render_data_reports():
    require_login()

    page_stamp("Data & Reports")
    st.title("Data and reports")

    # -------------------------
    # Report source: pick a CSV from Google Drive
    # -------------------------
    st.subheader("Data source")

    available_files = _newest_first(list_available_csvs())
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
        file_labels = [_file_label(f) for f in available_files]
        selected_idx = st.selectbox(
            "CSV file (newest data first)",
            options=range(len(available_files)),
            format_func=lambda i: file_labels[i],
            index=0,  # defaults to the newest day
        )
        selected_file = available_files[selected_idx]

        if st.button("Load selected file"):
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
            st.caption(
                f"Currently loaded: {st.session_state.get('_loaded_filename', '')}")

    # -------------------------
    # Optional: DC meter data from the separate panel-meter-data folder
    # -------------------------
    st.subheader("DC meter data (optional)")

    available_dcm_files = _newest_first(list_available_dcm_csvs())
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
        dcm_labels = [_file_label(f) for f in available_dcm_files]
        dcm_selected_idx = st.selectbox(
            "DC meter CSV file (newest data first)",
            options=range(len(available_dcm_files)),
            format_func=lambda i: dcm_labels[i],
            index=0,
            key="dcm_file_select",
        )
        selected_dcm_file = available_dcm_files[dcm_selected_idx]

        if st.button("Load DC meter file"):
            try:
                df_dcm = _dcm_to_local(
                    download_dcm_csv_as_df(selected_dcm_file["id"]))
                st.session_state["_loaded_dcm_df"] = df_dcm
                st.session_state["_loaded_dcm_filename"] = selected_dcm_file["name"]
                st.success(
                    f"Loaded {selected_dcm_file['name']} ({df_dcm.shape[0]} rows)")
            except Exception as e:
                st.error(f"Couldn't download that file: {e}")

        if df_dcm is None and "_loaded_dcm_df" in st.session_state:
            df_dcm = st.session_state["_loaded_dcm_df"]
            st.caption(
                f"Currently loaded: {st.session_state.get('_loaded_dcm_filename', '')}")

    # -------------------------
    # Saved report formats, then the title and notes they fill in
    # -------------------------
    render_preset_bar()

    # Defaults go in session state rather than `value=`: a widget that has both
    # a default and a value set from session state (which is what applying a
    # saved format does) makes Streamlit print a warning.
    st.session_state.setdefault("report_title", "Bifacial PV Performance Report")
    report_title = st.text_input("Report Title", key="report_title")
    observation = st.text_area("Observation Notes", key="observation")

    if df is None:
        # The DC meter data can still be batched on its own.
        if df_dcm is not None:
            _render_batch(None, df_dcm, report_title, observation, [], [],
                          None, None)
        else:
            st.info("Load a CSV above to chart it and build reports.")
        return

    st.subheader("Data preview")
    st.dataframe(df.head(100))

    st.subheader("Dataset")
    info_a, info_b = st.columns(2)
    info_a.metric("Rows", f"{df.shape[0]:,}")
    info_b.metric("Columns", f"{df.shape[1]:,}")

    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    st.subheader("Chart columns")

    # if a different file was loaded earlier, drop any selected columns
    # that no longer exist so the multiselect widget doesn't error out
    if "selected_temps" not in st.session_state:
        st.session_state.selected_temps = [c for c in numeric_cols
                                           if "temp" in c.lower()]
    else:
        st.session_state.selected_temps = [c for c in st.session_state.selected_temps
                                           if c in numeric_cols]

    temp_label_col, temp_all_col, temp_none_col = st.columns([4, 1, 1])
    with temp_label_col:
        st.caption("Temperature Columns")
    with temp_all_col:
        if st.button("Select all", key="temp_select_all",
                     use_container_width=True):
            st.session_state.selected_temps = numeric_cols
    with temp_none_col:
        if st.button("Remove all", key="temp_remove_all",
                     use_container_width=True):
            st.session_state.selected_temps = []

    selected_temps = st.multiselect(
        "Select Temperature Columns", numeric_cols,
        key="selected_temps", label_visibility="collapsed")

    # "irr" matched Irr_1..24 AND IrrAvg_1..24, so the page opened with 48
    # irradiance lines instead of 24. The averages are still in the list to
    # pick; they are just not preselected alongside the raw channels.
    if "selected_irradiance" not in st.session_state:
        st.session_state.selected_irradiance = [c for c in numeric_cols
                                                if "irr" in c.lower()
                                                and "avg" not in c.lower()]
    else:
        st.session_state.selected_irradiance = [
            c for c in st.session_state.selected_irradiance if c in numeric_cols]

    irr_label_col, irr_all_col, irr_none_col = st.columns([4, 1, 1])
    with irr_label_col:
        st.caption("Irradiance Columns")
    with irr_all_col:
        if st.button("Select all", key="irr_select_all",
                     use_container_width=True):
            st.session_state.selected_irradiance = numeric_cols
    with irr_none_col:
        if st.button("Remove all", key="irr_remove_all",
                     use_container_width=True):
            st.session_state.selected_irradiance = []

    selected_irradiance = st.multiselect(
        "Select Irradiance Columns", numeric_cols,
        key="selected_irradiance", label_visibility="collapsed")

    temp_ylim = irr_ylim = None
    fig = None

    if selected_temps or selected_irradiance:
        for key, default in (("temp_y_auto", True), ("temp_y_min", 0.0),
                             ("temp_y_max", 50.0), ("irr_y_auto", True),
                             ("irr_y_min", 0.0), ("irr_y_max", 1200.0)):
            st.session_state.setdefault(key, default)

        with st.expander("📐 Chart scale (optional)"):
            st.caption(
                "The X range trims which rows are plotted; the Y ranges fix "
                "the axis limits."
            )
            x_start, x_end = st.slider(
                "X range (row index)", 0, max(n_rows - 1, 1),
                (0, max(n_rows - 1, 1))
            )

            temp_y_col, irr_y_col = st.columns(2)
            with temp_y_col:
                st.caption("Temperature Y-axis (°C)")
                temp_auto = st.checkbox("Auto", key="temp_y_auto")
                temp_y_min = st.number_input("Min", key="temp_y_min",
                                             disabled=temp_auto)
                temp_y_max = st.number_input("Max", key="temp_y_max",
                                             disabled=temp_auto)
            with irr_y_col:
                st.caption("Irradiance Y-axis (W/m²)")
                irr_auto = st.checkbox("Auto", key="irr_y_auto")
                irr_y_min = st.number_input("Min", key="irr_y_min",
                                            disabled=irr_auto)
                irr_y_max = st.number_input("Max", key="irr_y_max",
                                            disabled=irr_auto)

        temp_ylim = None if temp_auto else (temp_y_min, temp_y_max)
        irr_ylim = None if irr_auto else (irr_y_min, irr_y_max)

        n_series = len(selected_temps) + len(selected_irradiance)
        if n_series > CHART_WARN_SERIES:
            st.warning(
                f"{n_series} columns selected. The chart will be slow and hard "
                f"to read — matplotlib draws every line individually. Use "
                f"“Remove all” and pick the handful you actually want to show."
            )

        # Slice, thin, then convert: a chart a few hundred pixels wide cannot
        # show 7,590 rows, let alone 7,590 x 72.
        sliced = df.iloc[x_start:x_end + 1]
        thin, step = _thin(sliced)
        if step > 1:
            st.caption(f"Charting every {step}th row ({len(thin):,} of "
                       f"{len(sliced):,}) for speed. Reports do the same; the "
                       f"tables in them still use every row.")

        fig = plot_weather_signals(
            _axis_time(thin),
            {c: thin[c].tolist() for c in selected_temps},
            {c: thin[c].tolist() for c in selected_irradiance},
            temp_ylim=temp_ylim,
            irr_ylim=irr_ylim,
        )
        st.pyplot(fig)
    else:
        # No chart, but the report is still buildable -- a DC-meter-only
        # document has no weather trace to draw, and requiring one is what
        # made that combination unreachable before.
        x_start, x_end = 0, max(n_rows - 1, 0)
        st.caption("No chart columns selected. Reports will be built without "
                   "a weather chart.")

    # Which halves of the data the single report carries. Batch keeps its own
    # copy of this choice further down, because the two are built separately.
    single_options = [CONTENT_WEATHER]
    if df_dcm is not None:
        single_options += [CONTENT_DCM, CONTENT_BOTH]
    # Default to the fullest choice. When the options change (a DC file gets
    # loaded) move to the new fullest one, as the page always has; otherwise
    # leave the user's pick -- or a saved format's -- alone unless it names an
    # option this session doesn't offer (a DC meter choice with no DC file).
    options_now = tuple(single_options)
    options_before = st.session_state.get("_single_options")
    st.session_state["_single_options"] = options_now
    if options_before is not None and options_before != options_now:
        st.session_state["single_content"] = single_options[-1]
    elif st.session_state.get("single_content") not in single_options:
        st.session_state["single_content"] = single_options[-1]
    single_content = st.radio(
        "Include in the single report", single_options,
        horizontal=True, key="single_content")

    # What the current build corresponds to. If any of it changes, the
    # bytes sitting in session state are describing something else.
    signature = (
        st.session_state.get("_loaded_filename"),
        st.session_state.get("_loaded_dcm_filename") if df_dcm is not None else None,
        report_title, observation, single_content,
        tuple(selected_temps), tuple(selected_irradiance),
        x_start, x_end, temp_ylim, irr_ylim,
    )

    # The report gets the same rows the chart does, so "as framed on screen"
    # is literally true rather than approximately. A DC-only report carries no
    # weather chart even when one is on screen.
    single_df, single_dcm = _report_parts(
        single_content, df.iloc[x_start:x_end + 1], df_dcm)
    report_fig = None if single_content == CONTENT_DCM else fig
    _render_single(single_df, single_dcm, report_title, observation,
                   report_fig, signature)

    _render_batch(df, df_dcm, report_title, observation,
                  selected_temps, selected_irradiance, temp_ylim, irr_ylim)

    # matplotlib figures were never released; every rerun made more
    if fig is not None:
        close_figures([fig])
