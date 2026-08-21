import io
from datetime import datetime

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# The Drive folder rclone syncs your CSVs into (see: rclone sync
# "/home/skyimager5/Desktop/bifacial data" gdrive:bifacial-data)
DRIVE_FOLDER_NAME = "bifacial-data"

# Separate Drive folder for DC meter (voltage/current/power) CSVs —
# organized as <root>/<device_id>/<year>/<month>/*.csv, one subfolder
# per meter device (e.g. dcm_3366)
DCM_DRIVE_FOLDER_NAME = "panel-meter-data"

# Separate Drive folder for device .log files (Pi + mini PC) — see
# LOG_DRIVE_FOLDER_NAME below for the alert-scanning feature
LOG_DRIVE_FOLDER_NAME = "device-logs"

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


def _find_all_files_recursive(service, root_folder_id, extension=".csv"):
    """Walks the folder tree starting at root_folder_id (breadth-first)
    and collects every file ending in `extension` found at any depth —
    needed because the Pi logger organizes files into
    <root>/<year>/<month>/*.ext rather than dropping them flat in the
    top-level folder. Each returned entry gets a "folder_path" list
    (e.g. ["2026", "07"]) recording which subfolders it was found
    under, so callers can group files by year/device without guessing
    from filenames or Drive's modifiedTime."""
    matches = []
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
            if entry["mimeType"] == "application/vnd.google-apps.folder":
                folders_to_search.append((entry["id"], path_parts + [entry["name"]]))
            elif entry["name"].lower().endswith(extension):
                entry["folder_path"] = path_parts
                matches.append(entry)

    return matches


def _find_all_csvs_recursive(service, root_folder_id):
    """Back-compat wrapper — CSV-specific callers unchanged."""
    return _find_all_files_recursive(service, root_folder_id, ".csv")


@st.cache_data(ttl=60)
def list_available_csvs():
    """Returns a list of dicts (id, name, modifiedTime) for every CSV
    anywhere under the Drive folder (including year/month
    subfolders), newest first. Returns an empty list (rather than
    raising) on any failure, so the UI can show a friendly message
    instead of crashing the whole app."""
    try:
        service = _get_drive_service()
        folder_id = _get_folder_id(service)
        if folder_id is None:
            return []

        files = _find_all_csvs_recursive(service, folder_id)
        files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
        return files
    except Exception as e:
        st.session_state["_drive_list_error"] = str(e)
        return []


def download_csv_as_df(file_id: str) -> pd.DataFrame:
    """Downloads a single CSV from Drive by file ID and returns it as
    a DataFrame. Raises on failure — the caller should wrap this in
    a try/except and show an error, since a report can't be built
    without the actual data."""
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


@st.cache_data(ttl=3600)
def download_and_combine_csvs(file_ids: tuple) -> pd.DataFrame:
    """Downloads several CSVs by file ID and concatenates them into one
    DataFrame — used to build a full year's worth of data for the
    annual irradiance tracker. Cached for an hour since a year's data
    doesn't change minute to minute. Skips any individual file that
    fails to download rather than failing the whole batch, since one
    corrupt/partial sync shouldn't block the rest of the year."""
    dfs = []
    for file_id in file_ids:
        try:
            dfs.append(download_csv_as_df(file_id))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# =========================
# DC METER (panel-meter-data) — separate Drive folder, separate CSV
# schema (long format: one row per device per timestamp), so it gets
# its own listing + download helpers rather than reusing the sensor
# ones above.
# =========================

@st.cache_data(ttl=60)
def list_available_dcm_csvs():
    """Same idea as list_available_csvs(), but for the DC meter CSVs
    under the separate 'panel-meter-data' Drive folder. Returns an
    empty list (rather than raising) on any failure."""
    try:
        service = _get_drive_service()
        folder_id = _get_folder_id(service, DCM_DRIVE_FOLDER_NAME)
        if folder_id is None:
            return []

        files = _find_all_csvs_recursive(service, folder_id)
        files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
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


@st.cache_data(ttl=3600)
def download_and_combine_dcm_csvs(file_ids: tuple) -> pd.DataFrame:
    """Downloads several DC-meter CSVs by file ID, concatenates them,
    and standardizes columns — used to append a full year of DC meter
    history onto the live chart. Skips any file that fails to
    download rather than failing the whole batch."""
    dfs = []
    for file_id in file_ids:
        try:
            dfs.append(download_csv_as_df(file_id))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return _standardize_dcm_columns(pd.concat(dfs, ignore_index=True))


# =========================
# DEVICE LOGS (device-logs) — .log files from the Pi and mini PC,
# scanned for the alert system. Kept as plain text rather than parsed
# into a DataFrame, since log lines aren't naturally tabular.
# =========================

@st.cache_data(ttl=60)
def list_available_log_files():
    """Lists every .log file anywhere under the 'device-logs' Drive
    folder (whatever subfolder structure it uses — device/year/month
    or otherwise, same recursive walk as the CSV listers). Returns an
    empty list (rather than raising) on any failure."""
    try:
        service = _get_drive_service()
        folder_id = _get_folder_id(service, LOG_DRIVE_FOLDER_NAME)
        if folder_id is None:
            return []

        files = _find_all_files_recursive(service, folder_id, ".log")
        files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
        return files
    except Exception as e:
        st.session_state["_log_drive_list_error"] = str(e)
        return []


@st.cache_data(ttl=60)
def download_log_text(file_id: str) -> str:
    """Downloads a single .log file's raw text content. Cached briefly
    since the same file may be re-scanned across reruns while the
    Alerts page is open. Returns "" on failure rather than raising, so
    one unreadable file doesn't stop the rest of the scan."""
    try:
        service = _get_drive_service()
        request = service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buffer.seek(0)
        return buffer.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def device_label_for(file_entry: dict) -> str:
    """Best-effort device name for a log file — the first folder
    segment that isn't a 4-digit year or a 2-digit month (e.g. 'pi' or
    'mini-pc' in <root>/<device>/<year>/<month>/*.log). Falls back to
    the filename if the folder structure doesn't have one."""
    path = file_entry.get("folder_path") or []
    for part in path:
        if part.isdigit() and (len(part) == 4 or len(part) == 2):
            continue
        return part
    return file_entry.get("name", "unknown device")
