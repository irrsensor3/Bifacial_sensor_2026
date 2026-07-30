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


@st.cache_data(ttl=60)
def list_available_csvs():
    """Returns a list of dicts (id, name, modifiedTime) for every CSV
    in the Drive folder, newest first. Returns an empty list (rather
    than raising) on any failure, so the UI can show a friendly
    message instead of crashing the whole app."""
    try:
        service = _get_drive_service()
        folder_id = _get_folder_id(service)
        if folder_id is None:
            return []

        query = (
            f"'{folder_id}' in parents and "
            "mimeType = 'text/csv' and "
            "trashed = false"
        )
        res = (
            service.files()
            .list(
                q=query,
                fields="files(id, name, modifiedTime)",
                orderBy="modifiedTime desc",
                pageSize=200,
            )
            .execute()
        )
        return res.get("files", [])
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
