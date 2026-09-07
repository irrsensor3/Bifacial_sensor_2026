import io
import re
from datetime import date, datetime

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import time

# The Drive folder rclone syncs your CSVs into (see: rclone sync
# "/home/skyimager5/Desktop/bifacial data" gdrive:bifacial-data)
DRIVE_FOLDER_NAME = "bifacial-data"

# Separate Drive folder for DC meter (voltage/current/power) CSVs —
# organized as <root>/<device_id>/<year>/<month>/*.csv, one subfolder
# per meter device (e.g. dcm_3366)
DCM_DRIVE_FOLDER_NAME = "panel-meter-data"

# Gap-filled irradiance/temperature output. Unlike the two folders above
# this one is nested and flat inside: every CSV sits directly in OUTPUT and
# carries its date in the filename rather than in a <year>/<month> folder.
#
#   My Drive/Pi Data Downloaded by Hng/OUTPUT/
#       filled_2023-03-16.csv    <- the values (gaps already filled in)
#       flags_2023-03-16.csv     <- per-cell "measured" / "filled_gap"
#
# The service account can only see what has been shared with it. Sharing
# just OUTPUT is enough — the parent lookup below falls back to finding
# OUTPUT by name when its parent isn't visible.
FILLED_DRIVE_FOLDER_PATH = ("Pi Data Downloaded by Hng", "OUTPUT")
FILLED_PREFIX = "filled_"
FLAGS_PREFIX = "flags_"

# Suffix for the boolean companion column that says whether a given reading
# was reconstructed. Irr_1 -> Irr_1__filled. Kept as a module constant so the
# UI can build the same name without hard-coding it in two places.
FILLED_FLAG_SUFFIX = "__filled"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@st.cache_resource
def _get_drive_service():
    """Builds a Drive API client from the service account credentials
    stored in Streamlit secrets (see [gcp_service_account] in
    secrets.toml). Cached as a resource so it's only built once per
    session, not on every rerun."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES
    )
    return build("drive", "v3", credentials=credentials)


def _escape_drive_name(name: str) -> str:
    """Drive query strings are single-quoted, so a name containing a quote
    or backslash has to be escaped or the query is rejected."""
    return str(name).replace("\\", "\\\\").replace("'", "\\'")


def _get_folder_id(service, folder_name=DRIVE_FOLDER_NAME):
    """Looks up the Drive folder ID by name. Assumes the folder name
    is unique enough (top-level, shared directly with the service
    account) — takes the first match."""
    query = (
        f"name = '{_escape_drive_name(folder_name)}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    res = service.files().list(q=query, fields="files(id, name)").execute()
    files = res.get("files", [])
    if not files:
        return None
    return files[0]["id"]


def _resolve_folder_path(service, path_parts):
    """Walks a folder path one level at a time ('Pi Data Downloaded by
    Hng' -> 'OUTPUT') and returns the ID of the last folder, or None if
    any level is missing.

    Resolving level by level rather than searching for the last name
    alone matters when a generic name like "OUTPUT" could exist in more
    than one place: constraining each step to the previous folder's ID
    picks the right one.
    """
    folder_id = None
    for name in path_parts:
        query = (
            f"name = '{_escape_drive_name(name)}' and "
            "mimeType = 'application/vnd.google-apps.folder' and "
            "trashed = false"
        )
        if folder_id:
            query += f" and '{folder_id}' in parents"
        res = service.files().list(
            q=query, fields="files(id, name)", pageSize=10
        ).execute()
        files = res.get("files", [])
        if not files:
            return None
        folder_id = files[0]["id"]
    return folder_id


# A date anywhere in a filename: filled_2023-03-16.csv,
# Bifacial_2026-08-23.csv, 2026-08-23_dcm_3366_bifaical.csv.
_DATE_IN_NAME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _date_from_name(name: str):
    """The calendar day a CSV holds, taken from its filename, or None.

    Knowing the exact day (not just the year/month folder it lives in)
    is what lets resolve_period_files hand back one file for a one-day
    request instead of a whole month's worth.
    """
    match = _DATE_IN_NAME_RE.search(str(name or ""))
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _list_children(service, folder_id):
    """Every non-trashed child of a folder, following nextPageToken.

    A single page tops out at 1000 entries. A flat folder holding a
    couple of years of daily filled_/flags_ pairs passes that, and
    without paging the oldest files would simply never appear.
    """
    entries = []
    page_token = None
    while True:
        res = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        entries.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            return entries


def _find_all_csvs_recursive(service, root_folder_id):
    """
    Walk through the Drive folder tree and collect ONLY CSV files that
    belong to a valid year/month folder structure.

    Accepted structures:

        bifacial-data/
            <year>/
                <month>/
                    file.csv

        panel-meter-data/
            <device_id>/
                <year>/
                    <month>/
                        file.csv

    CSV files outside a year/month structure are ignored.

    Example:

        bifacial-data/
            2026/
                08/
                    Bifacial_2026-08-23.csv     <-- INCLUDED
            alerts.csv                         <-- IGNORED

        panel-meter-data/
            dcm_3366/
                2026/
                    08/
                        2026-08-23.csv         <-- INCLUDED
            alerts.csv                         <-- IGNORED
    """

    csv_files = []

    # Each item is:
    # (folder_id, path_parts)
    folders_to_search = [(root_folder_id, [])]

    while folders_to_search:
        current_id, path_parts = folders_to_search.pop()

        for entry in _list_children(service, current_id):

            # ---------------------------------------------------------
            # Folder
            # ---------------------------------------------------------
            if entry["mimeType"] == "application/vnd.google-apps.folder":

                folders_to_search.append(
                    (
                        entry["id"],
                        path_parts + [entry["name"]],
                    )
                )

                continue

            # ---------------------------------------------------------
            # Ignore anything that isn't CSV
            # ---------------------------------------------------------
            if not entry["name"].lower().endswith(".csv"):
                continue

            # ---------------------------------------------------------
            # Check whether this CSV is inside:
            #
            #     ... / YEAR / MONTH / file.csv
            #
            # We deliberately DO NOT accept:
            #
            #     ... / alerts.csv
            #     ... / YEAR / alerts.csv
            #
            # because those are not inside a year/month folder.
            # ---------------------------------------------------------
            valid_year_month = False

            for i in range(len(path_parts) - 1):

                year_part = path_parts[i]
                month_part = path_parts[i + 1]

                # Year must be exactly 4 digits
                if not (
                    year_part.isdigit()
                    and len(year_part) == 4
                ):
                    continue

                # Allow month folder names that are 1 or 2 digit numbers
                # (e.g. "8" or "08"). Be explicit about the pattern so
                # single-digit folders are accepted reliably.
                m = str(month_part).strip()
                if not re.match(r"^\d{1,2}$", m):
                    continue

                month_number = int(m)

                if 1 <= month_number <= 12:
                    valid_year_month = True
                    break

            # ---------------------------------------------------------
            # CSV is outside a valid year/month structure.
            # Ignore it completely.
            # ---------------------------------------------------------
            if not valid_year_month:
                continue

            # ---------------------------------------------------------
            # Store the file, its folder path, and (when the filename
            # carries one) the exact day it covers.
            # ---------------------------------------------------------
            entry = dict(entry)
            entry["folder_path"] = path_parts
            entry["data_date"] = _date_from_name(entry["name"])

            csv_files.append(entry)

    return csv_files


def _find_csvs_anywhere(service, root_folder_id):
    """Every CSV under a folder, whatever the nesting.

    The OUTPUT folder is flat and its files carry their date in the
    filename, so the year/month folder rule enforced above would reject
    all of them.
    """
    csv_files = []
    folders_to_search = [(root_folder_id, [])]

    while folders_to_search:
        current_id, path_parts = folders_to_search.pop()

        for entry in _list_children(service, current_id):
            if entry["mimeType"] == "application/vnd.google-apps.folder":
                folders_to_search.append((entry["id"], path_parts + [entry["name"]]))
                continue
            if not entry["name"].lower().endswith(".csv"):
                continue
            entry = dict(entry)
            entry["folder_path"] = path_parts
            entry["data_date"] = _date_from_name(entry["name"])
            csv_files.append(entry)

    return csv_files


@st.cache_data(ttl=1800)
def list_available_csvs():
    """
    Returns only irradiance CSV files located inside a valid
    year/month folder structure under bifacial-data.

    Example accepted:

        bifacial-data/2026/08/Bifacial_2026-08-23.csv

    Example ignored:

        bifacial-data/alerts.csv
        bifacial-data/something.csv
        bifacial-data/2026/alerts.csv
    """

    try:
        service = _get_drive_service()

        folder_id = _get_folder_id(service, DRIVE_FOLDER_NAME)

        if folder_id is None:
            return []

        files = _find_all_csvs_recursive(
            service,
            folder_id,
        )

        files.sort(
            key=lambda f: f.get("modifiedTime", ""),
            reverse=True,
        )

        return files

    except Exception as e:
        st.session_state["_drive_list_error"] = str(e)
        return []


def download_csv_as_df(file_id: str) -> pd.DataFrame:
    """Download one CSV from Google Drive."""

    service = _get_drive_service()

    request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False

    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)

    return pd.read_csv(buffer)


def format_file_label(file_entry: dict) -> str:
    """Human-friendly label for a dropdown option, e.g.
    'Bifacial_ 2026-07-29.csv — modified 2026-07-30 03:12'."""
    name = file_entry.get("name", "unknown.csv")
    modified = file_entry.get("modifiedTime", "")
    try:
        dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
        modified_label = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        modified_label = modified
    return f"{name} — modified {modified_label}" if modified_label else name


def extract_year(file_entry: dict) -> str:
    """Best-effort year for a CSV. Prefers the <year> folder it was
    found under — searches every level of its folder path (not just
    the first) since different Drive folders nest at different depths
    (e.g. bifacial-data is <root>/<year>/<month>, panel-meter-data is
    <root>/<device_id>/<year>/<month>). Falls back to the file's Drive
    modifiedTime for files that aren't organized that way."""
    path = file_entry.get("folder_path") or []
    for part in path:
        if part.isdigit() and len(part) == 4:
            return part
    data_date = file_entry.get("data_date")
    if data_date is not None:
        return f"{data_date.year:04d}"
    modified = file_entry.get("modifiedTime", "")
    return modified[:4] if modified else "unknown"


MONTH_LABELS = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}


def extract_month(file_entry: dict) -> str:
    """Best-effort zero-padded month ('01'-'12') for a CSV — the
    folder immediately after whichever <year> folder was found in its
    path. Falls back to the file's Drive modifiedTime for files that
    aren't organized that way."""
    path = file_entry.get("folder_path") or []
    for i, part in enumerate(path):
        if part.isdigit() and len(part) == 4:
            if i + 1 < len(path) and path[i + 1].isdigit():
                return path[i + 1].zfill(2)
            break
    data_date = file_entry.get("data_date")
    if data_date is not None:
        return f"{data_date.month:02d}"
    modified = file_entry.get("modifiedTime", "")
    return modified[5:7] if len(modified) >= 7 else "unknown"


def month_label(month: str) -> str:
    """'07' -> '07 - July'; falls back to the raw value if unrecognized."""
    name = MONTH_LABELS.get(month)
    return f"{month} - {name}" if name else month


def resolve_period_files(available_files, start_date, end_date):
    """Returns the subset of available_files covering [start_date,
    end_date] (both inclusive, as date objects).

    A file whose name contains its date is matched on that exact day.
    That matters because the caller caps how many files it will load at
    once: resolving a one-day request to a whole month of files and then
    truncating to the newest few could drop the very day that was asked
    for. Files with no date in the name fall back to overlapping on the
    (year, month) folder they live in, as before. Files with neither are
    skipped rather than raising — a handful of stray files shouldn't
    block loading everything else."""
    if not available_files:
        return []
    start_ym = (start_date.year, start_date.month)
    end_ym = (end_date.year, end_date.month)
    out = []
    for f in available_files:
        data_date = f.get("data_date")
        if data_date is not None:
            if start_date <= data_date <= end_date:
                out.append(f)
            continue
        y, m = extract_year(f), extract_month(f)
        if not (y.isdigit() and m.isdigit()):
            continue
        ym = (int(y), int(m))
        if start_ym <= ym <= end_ym:
            out.append(f)
    return out


# ============================================================
# SAFE DRIVE CSV CACHE
# ============================================================

@st.cache_data(
    ttl=None,
    persist="disk",
    max_entries=600,
)
def _download_single_csv_cached(
    file_id: str,
    modified_time: str,
) -> pd.DataFrame:
    """
    Cache individual CSV files.

    Historical files normally never change, so they remain cached.

    The modified_time is part of the cache key. Therefore, if a file
    changes, Streamlit automatically downloads the newer version.
    """

    return download_csv_as_df(file_id)


def download_and_combine_csvs(file_entries: tuple) -> pd.DataFrame:
    """
    Download and combine requested irradiance CSVs.

    Each CSV is cached independently.

    This is important because if today's CSV changes, only that CSV
    needs to be downloaded again. Previously downloaded historical
    CSVs remain cached.
    """

    if not file_entries:
        return pd.DataFrame()


    dfs = []

    for file_id, modified_time in file_entries:

        try:

            df = _download_single_csv_cached(
                file_id,
                modified_time or "",
            )

            if df is not None and not df.empty:
                dfs.append(df)

        except Exception:
            # One failed file should not kill the application.
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(
        dfs,
        ignore_index=True,
        copy=False,
    )

    return combined

# =========================
# DC METER (panel-meter-data) — separate Drive folder, separate CSV
# schema (long format: one row per device per timestamp), so it gets
# its own listing + download helpers rather than reusing the sensor
# ones above.
# =========================

@st.cache_data(ttl=1800)
def list_available_dcm_csvs(include_avg=False):
    """
    Returns DCM 3366 CSV files under panel-meter-data.

    include_avg=False:
        Returns normal daily CSVs only.

    include_avg=True:
        Returns daily average CSVs only.

    Examples:

        Normal:
            2026-08-23_dcm_3366_bifaical.csv

        Average:
            2026-08-23_dcm_3366_bifaical_avg.csv
    """

    try:
        service = _get_drive_service()

        folder_id = _get_folder_id(
            service,
            DCM_DRIVE_FOLDER_NAME
        )

        if folder_id is None:
            return []

        all_files = _find_all_csvs_recursive(
            service,
            folder_id
        )

        files = []

        for f in all_files:

            name = f.get("name", "").lower()

            # Must be CSV
            if not name.endswith(".csv"):
                continue

            # Must be DCM 3366
            if "_dcm_3366_" not in name:
                continue

            # Must begin with YYYY-MM-DD
            if not re.match(r"^\d{4}-\d{2}-\d{2}_", name):
                continue

            is_avg = "_avg.csv" in name

            # Select either normal or average files
            if include_avg:
                if not is_avg:
                    continue
            else:
                if is_avg:
                    continue

            files.append(f)

        files.sort(
            key=lambda f: f.get("modifiedTime", ""),
            reverse=True,
        )

        return files

    except Exception as e:
        st.session_state["_dcm_drive_list_error"] = str(e)
        return []


def _standardize_dcm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize historical DC-meter CSV columns so they match the
    live Supabase panel_readings schema.

    The important part is created_at:
    - Historical CSV timestamps may be timezone-naive.
    - Live Supabase timestamps may be timezone-aware (UTC).

    Convert everything to UTC and then remove the timezone information.
    This gives both historical and live data the same timestamp type:

        datetime64[ns]

    This prevents:
        TypeError: Cannot compare tz-naive and tz-aware timestamps
    """

    rename_map = {
        "Datetime": "created_at",
        "Device_ID": "device_id",
        "Forward_energy_kWh": "forward_energy_kwh",
        "Active_power_kW": "active_power_kw",
        "Current_A": "current_a",
        "Voltage_V": "voltage_v",
        "Error": "error",
    }

    df = df.rename(columns=rename_map)

    if "created_at" in df.columns:
        # The Datetime column in these files is ARRAY-LOCAL time, not UTC --
        # confirmed against a recorded day, where generation runs 07:00 to
        # 18:00 and peaks at 13:00.
        #
        # Passing utc=True to a naive value LABELS it as UTC without shifting
        # it, so local 13:00 became "13:00 UTC" and the chart then displayed it
        # as 21:00. Localising to the array's zone first, then converting,
        # gives the genuine UTC instant.
        naive = pd.to_datetime(df["created_at"], errors="coerce")
        if getattr(naive.dt, "tz", None) is not None:
            # Already carries an offset: convert rather than assume.
            df["created_at"] = naive.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df["created_at"] = (
                naive.dt.tz_localize("Asia/Kuala_Lumpur",
                                     ambiguous="NaT", nonexistent="NaT")
                     .dt.tz_convert("UTC")
                     .dt.tz_localize(None)
            )

    return df


def download_dcm_csv_as_df(file_id: str) -> pd.DataFrame:
    """Downloads a single DC-meter CSV and standardizes its columns.
    Raises on failure, same as download_csv_as_df."""
    return _standardize_dcm_columns(download_csv_as_df(file_id))


@st.cache_data(
    ttl=None,
    persist="disk",
    max_entries=60,
)
def _download_single_dcm_csv_cached(
    file_id: str,
    modified_time: str,
) -> pd.DataFrame:
    """
    Cache individual DC meter CSV files.
    """
    return download_dcm_csv_as_df(file_id)


def download_and_combine_dcm_csvs(file_entries: tuple) -> pd.DataFrame:
    """
    Download and combine requested DC meter CSVs.

    Each CSV is cached independently.

    There is intentionally NO fixed maximum number of files here.
    This allows loading an entire month, year, or larger historical
    period when required.

    Only the files passed in file_entries are downloaded.
    """

    if not file_entries:
        return pd.DataFrame()

    dfs = []

    for file_id, modified_time in file_entries:
        try:
            df = _download_single_dcm_csv_cached(
                file_id,
                modified_time or "",
            )

            if df is not None and not df.empty:
                dfs.append(df)

        except Exception:
            # One failed file should not kill the application.
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(
        dfs,
        ignore_index=True,
        copy=False,
    )

    return combined


# =========================
# GAP-FILLED IRRADIANCE / TEMPERATURE (OUTPUT folder)
#
# Two files per day, side by side in one flat folder:
#
#   filled_2023-03-16.csv   Date, Time, Irr_1..Irr_24, Temp_1..Temp_24
#   flags_2023-03-16.csv    same shape, each cell "measured" or "filled_gap"
#
# The flags file is optional: a filled file with no matching flags file
# loads normally, everything just counts as measured.
#
# The frame handed back is wide, like the raw irradiance frames, with one
# extra boolean column per series (Irr_1__filled) saying whether that
# reading was reconstructed. Keeping the flag beside the value means the
# two survive concat, sorting and downsampling together, so the chart can
# never colour the wrong point.
# =========================

# Irr_1 / temp_04 / Irr 1 / Irr_1_flag all match; the number is captured
# without its leading zeros so Irr_01 in the flags file pairs with Irr_1
# in the filled file.
_SERIES_COL_RE = re.compile(
    r"^(irr|temp)[\s_-]*0*(\d{1,2})(?:[\s_-]*(?:flag|flags|status))?$",
    re.IGNORECASE,
)

MAX_SERIES_INDEX = 24


def _series_columns(df: pd.DataFrame, max_index: int = MAX_SERIES_INDEX):
    """Sensor columns in a filled/flags frame, as (original name, key).

    The key ('Irr_1') is only used to pair a flags column with its filled
    column. The original name is what gets written out, so whatever the
    CSV calls a series is exactly what the live feed and the chart see —
    no silent renaming that could stop historical and live traces from
    lining up.
    """
    found = []
    for col in df.columns:
        match = _SERIES_COL_RE.match(str(col).strip())
        if not match:
            continue
        index = int(match.group(2))
        if not (1 <= index <= max_index):
            continue
        kind = "Irr" if match.group(1).lower() == "irr" else "Temp"
        found.append((col, f"{kind}_{index}"))
    return found


def _filled_created_at(df: pd.DataFrame) -> pd.Series:
    """created_at (naive UTC) from the Date and Time columns.

    These files start with Date (2023-09-14) and Time (0:00:00) in
    array-local time. Everything downstream expects naive UTC from a
    download function — _load_range converts back to local for display —
    so localise to Asia/Kuala_Lumpur first and convert, rather than
    labelling local times as UTC and shifting the whole day by eight
    hours.
    """
    by_name = {str(c).strip().lower(): c for c in df.columns}
    date_col = by_name.get("date")
    time_col = by_name.get("time")

    if date_col is None or time_col is None:
        # Fall back on position: the first two columns are Date and Time.
        if len(df.columns) >= 2:
            date_col, time_col = df.columns[0], df.columns[1]
        else:
            return pd.Series(pd.NaT, index=df.index)

    stamped = (
        df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
    )
    # These files are written as YYYY-MM-DD and H:MM:SS. Naming the format
    # rather than letting pandas infer it matters at this size: without it
    # every one of a day's ~17k rows is handed to dateutil individually,
    # which takes seconds per file instead of milliseconds.
    naive = pd.to_datetime(stamped, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if naive.isna().all() and stamped.str.strip().ne("").any():
        naive = pd.to_datetime(stamped, errors="coerce")
    if getattr(naive.dt, "tz", None) is not None:
        return naive.dt.tz_convert("UTC").dt.tz_localize(None)
    return (
        naive.dt.tz_localize("Asia/Kuala_Lumpur", ambiguous="NaT", nonexistent="NaT")
             .dt.tz_convert("UTC")
             .dt.tz_localize(None)
    )


def _build_filled_frame(df_filled: pd.DataFrame, df_flags=None) -> pd.DataFrame:
    """Combine one filled_<date>.csv with its flags_<date>.csv.

    Returns created_at + every Irr_/Temp_ column found + one
    <column>__filled boolean per series. With no flags file (or an
    unreadable one) every flag is False, which simply means the chart
    draws no gap-filled overlay for that day.
    """
    if df_filled is None or df_filled.empty:
        return pd.DataFrame()

    series = _series_columns(df_filled)
    if not series:
        return pd.DataFrame()

    out = pd.DataFrame(index=df_filled.index)
    out["created_at"] = _filled_created_at(df_filled)
    for original, _key in series:
        out[str(original).strip()] = pd.to_numeric(df_filled[original], errors="coerce")

    flag_by_key = {}
    if df_flags is not None and not df_flags.empty:
        raw = {}
        for original, key in _series_columns(df_flags):
            # "filled_gap" vs "measured". Matching on "fill" rather than
            # the exact string keeps working if the writer ever emits
            # "gap_filled" or "Filled_Gap".
            raw[key] = (
                df_flags[original].astype(str)
                .str.contains("fill", case=False, na=False)
            )

        if raw:
            flags_ts = _filled_created_at(df_flags)
            if flags_ts.notna().any():
                # Align on the timestamp, not on row position: a flags
                # file that skips or repeats a row would otherwise shift
                # every marking after it onto the wrong reading.
                lookup = pd.DataFrame(raw)
                lookup["created_at"] = flags_ts
                lookup = (
                    lookup.dropna(subset=["created_at"])
                          .drop_duplicates(subset=["created_at"], keep="last")
                          .set_index("created_at")
                )
                aligned = lookup.reindex(out["created_at"])
                flag_by_key = {
                    key: aligned[key].fillna(False).to_numpy(dtype=bool)
                    for key in raw
                }
            elif len(df_flags) == len(df_filled):
                # No usable Date/Time in the flags file, but the same
                # number of rows: fall back to row order.
                flag_by_key = {key: values.to_numpy(dtype=bool) for key, values in raw.items()}

    for original, key in series:
        column = str(original).strip()
        values = flag_by_key.get(key)
        out[column + FILLED_FLAG_SUFFIX] = False if values is None else values

    return out.dropna(subset=["created_at"]).reset_index(drop=True)


@st.cache_data(ttl=1800)
def list_available_filled_csvs():
    """
    Returns the gap-filled irradiance CSVs in the OUTPUT folder, newest
    first, each carrying:

        data_date       the day it covers, from the filename
        folder_path     synthesised ["YYYY", "MM"] so the year/month
                        pickers in the UI work unchanged
        flags_id        the matching flags_<date>.csv, or None
        flags_modified  that file's modifiedTime, for cache keying

    A filled file with no matching flags file is still returned.
    """
    try:
        service = _get_drive_service()

        folder_id = _resolve_folder_path(service, FILLED_DRIVE_FOLDER_PATH)
        if folder_id is None:
            # The service account may have been given OUTPUT directly,
            # in which case its parent is invisible and the path walk
            # above can't succeed. Fall back to the leaf name.
            folder_id = _get_folder_id(service, FILLED_DRIVE_FOLDER_PATH[-1])
        if folder_id is None:
            st.session_state["_filled_drive_list_error"] = (
                "Couldn't find the folder "
                f"{'/'.join(FILLED_DRIVE_FOLDER_PATH)} in Drive. Share it "
                "with the service account in secrets.toml (client_email)."
            )
            return []

        all_csvs = _find_csvs_anywhere(service, folder_id)

        flags_by_date = {}
        filled_files = []
        for entry in all_csvs:
            name = str(entry.get("name", ""))
            data_date = entry.get("data_date")
            if data_date is None:
                continue
            lowered = name.lower()
            if lowered.startswith(FLAGS_PREFIX):
                flags_by_date[data_date] = entry
            elif lowered.startswith(FILLED_PREFIX):
                filled_files.append(entry)

        out = []
        for entry in filled_files:
            data_date = entry["data_date"]
            enriched = dict(entry)
            enriched["folder_path"] = [f"{data_date.year:04d}", f"{data_date.month:02d}"]
            flags = flags_by_date.get(data_date)
            enriched["flags_id"] = flags["id"] if flags else None
            enriched["flags_modified"] = flags.get("modifiedTime", "") if flags else ""
            out.append(enriched)

        out.sort(key=lambda f: f["data_date"], reverse=True)
        st.session_state["_filled_drive_list_error"] = None
        return out

    except Exception as e:
        st.session_state["_filled_drive_list_error"] = str(e)
        return []


@st.cache_data(
    ttl=None,
    persist="disk",
    max_entries=300,
)
def _download_single_filled_csv_cached(
    file_id: str,
    modified_time: str,
    flags_id: str,
    flags_modified: str,
) -> pd.DataFrame:
    """
    Cache one filled+flags pair, already merged.

    Both modifiedTimes are part of the key, so re-running the gap filler
    over a day — regenerating either file — invalidates just that day.
    """
    df_filled = download_csv_as_df(file_id)

    df_flags = None
    if flags_id:
        try:
            df_flags = download_csv_as_df(flags_id)
        except Exception:
            # Flags are optional. Losing them costs the colouring, not
            # the data, so the day still loads.
            df_flags = None

    return _build_filled_frame(df_filled, df_flags)


def download_and_combine_filled_csvs(file_entries: tuple) -> pd.DataFrame:
    """
    Download and combine gap-filled CSVs.

    Takes the same (file_id, modified_time) entries as the other
    download_and_combine_* functions so it can be dropped into the same
    range-loading machinery; the matching flags file is looked up here
    from the cached listing.
    """
    if not file_entries:
        return pd.DataFrame()

    flags_by_file_id = {
        f["id"]: (f.get("flags_id") or "", f.get("flags_modified") or "")
        for f in list_available_filled_csvs()
    }

    dfs = []
    for file_id, modified_time in file_entries:
        flags_id, flags_modified = flags_by_file_id.get(file_id, ("", ""))
        try:
            df = _download_single_filled_csv_cached(
                file_id,
                modified_time or "",
                flags_id,
                flags_modified,
            )
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception:
            # One failed file should not kill the application.
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True, copy=False)

    # Days don't have to carry the same sensor set. Where concat filled a
    # missing column with NaN the reading is absent, not reconstructed.
    for column in combined.columns:
        if column.endswith(FILLED_FLAG_SUFFIX):
            combined[column] = combined[column].fillna(False).astype(bool)

    return combined
