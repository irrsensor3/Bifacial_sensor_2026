
import io
import re
from datetime import datetime
 
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
 
 
def _get_folder_id(service, folder_name=DRIVE_FOLDER_NAME):
    """Looks up the Drive folder ID by name. Assumes the folder name
    is unique enough (top-level, shared directly with the service
    account) — takes the first match."""
    query = (
        f"name = '{folder_name}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    res = service.files().list(q=query, fields="files(id, name)").execute()
    files = res.get("files", [])
    if not files:
        return None
    return files[0]["id"]
 
 
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

        query = f"'{current_id}' in parents and trashed = false"

        res = (
            service.files()
            .list(
                q=query,
                fields="files(id, name, mimeType, modifiedTime)",
                pageSize=1000,
            )
            .execute()
        )

        for entry in res.get("files", []):

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

                # Month must be 01-12
                if not month_part.isdigit():
                    continue

                month_number = int(month_part)

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
            # Store the file and its folder path.
            # ---------------------------------------------------------
            entry = dict(entry)
            entry["folder_path"] = path_parts

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
    modified = file_entry.get("modifiedTime", "")
    return modified[5:7] if len(modified) >= 7 else "unknown"
 
 
def month_label(month: str) -> str:
    """'07' -> '07 - July'; falls back to the raw value if unrecognized."""
    name = MONTH_LABELS.get(month)
    return f"{month} - {name}" if name else month
 
 
def resolve_period_files(available_files, start_date, end_date):
    """Returns the subset of available_files whose (year, month) folder
    overlaps [start_date, end_date] (both inclusive, as date objects).
    Files with an unparseable year/month are skipped rather than raising —
    a handful of stray files shouldn't block loading everything else."""
    if not available_files:
        return []
    start_ym = (start_date.year, start_date.month)
    end_ym = (end_date.year, end_date.month)
    out = []
    for f in available_files:
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
    max_entries=60,
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

    # Safety protection.
    MAX_FILES = 31

    if len(file_entries) > MAX_FILES:
        raise RuntimeError(
            f"Too many CSV files requested ({len(file_entries)}). "
            f"Maximum allowed is {MAX_FILES}."
        )

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
def list_available_dcm_csvs():
    """
    Returns the actual daily DCM CSV files under panel-meter-data.

    Accepts:
        YYYY-MM-DD_dcm_3366_bifaical.csv

    Rejects:
        YYYY-MM-DD_dcm_3366_bifaical_avg.csv

    Also ignores unrelated CSV files.
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

            # Must be a CSV
            if not name.endswith(".csv"):
                continue

            # Must be a DCM 3366 file
            if "_dcm_3366_" not in name:
                continue

            # IMPORTANT:
            # Ignore average files
            if "_avg.csv" in name:
                continue

            # Must begin with YYYY-MM-DD
            if not re.match(r"^\d{4}-\d{2}-\d{2}_", name):
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
    """Renames a raw DC-meter CSV's columns (Datetime, Device_ID,
    Forward_energy_kWh, Active_power_kW, Current_A, Voltage_V, Error)
    to match the live Supabase panel_readings schema (created_at,
    device_id, forward_energy_kwh, active_power_kw, current_a,
    voltage_v, error), and parses created_at to a real datetime — so
    historical and live DC meter data can be combined and handled
    identically downstream."""
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
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
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
    """
    if not file_entries:
        return pd.DataFrame()

    MAX_FILES = 31

    if len(file_entries) > MAX_FILES:
        raise RuntimeError(
            f"Too many CSV files requested ({len(file_entries)}). "
            f"Maximum allowed is {MAX_FILES}."
        )

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
