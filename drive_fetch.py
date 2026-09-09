import io
import re
import time
from datetime import date, datetime
from http.client import HTTPException

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

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

# HTTP statuses worth a retry. 403 is ambiguous on Drive: it covers both
# rate limiting and permission denied, so a genuine permission problem
# burns the full backoff before surfacing. That costs ~14s once, which is
# cheaper than the alternative of never retrying a rate limit.
RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}


class DriveListingError(RuntimeError):
    """A listing could not be completed.

    The point of this class is that it is RAISED out of the cached
    listing functions rather than turned into an empty list. Streamlit
    caches return values, not exceptions, so a failed run leaves the
    cache untouched and the next rerun retries — instead of serving
    "no files found" for the full 300s TTL.
    """


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


def service_account_email() -> str:
    """The client_email from secrets, for "did you share the folder with
    this address?" checks in the diagnostics."""
    try:
        return str(st.secrets["gcp_service_account"].get("client_email", "unknown"))
    except Exception:
        return "unknown"


def _drive_execute(request, attempts: int = 4):
    """Execute one Drive API request, retrying transient failures with
    exponential backoff (1s, 2s, 4s).

    Without this, a single hiccup anywhere in a folder walk aborts the
    whole walk. Non-retryable errors (404, 401, malformed query) are
    re-raised immediately so real problems still surface fast.
    """
    for attempt in range(attempts):
        try:
            return request.execute()
        except HttpError as e:
            status = getattr(getattr(e, "resp", None), "status", None)
            if status not in RETRYABLE_STATUS or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
        except (OSError, HTTPException):
            # Socket timeouts, dropped connections, SSL errors, incomplete
            # reads. All transient by nature.
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def _escape_drive_name(name: str) -> str:
    """Drive query strings are single-quoted, so a name containing a quote
    or backslash has to be escaped or the query is rejected."""
    return str(name).replace("\\", "\\\\").replace("'", "\\'")


def _get_folder_ids(service, folder_name=DRIVE_FOLDER_NAME):
    """EVERY visible folder with this name, not just the first match.

    The old version took files[0]. Drive does not guarantee ordering on
    an unsorted list call, so if the service account can see two folders
    with the same name (a duplicate, an old copy, a re-share) the walk
    picks a different one at random each time the cache expires — which
    looks exactly like intermittent failure. Walking all matches and
    unioning by file ID makes that case harmless.
    """
    query = (
        f"name = '{_escape_drive_name(folder_name)}' and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    res = _drive_execute(
        service.files().list(
            q=query,
            fields="files(id, name, parents, owners(emailAddress))",
            pageSize=100,
        )
    )
    return res.get("files", [])


def _get_folder_id(service, folder_name=DRIVE_FOLDER_NAME):
    """First matching folder ID, or None. Kept for the callers that only
    need one (the OUTPUT fallback below)."""
    folders = _get_folder_ids(service, folder_name)
    return folders[0]["id"] if folders else None


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
        res = _drive_execute(
            service.files().list(q=query, fields="files(id, name)", pageSize=10)
        )
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
        res = _drive_execute(
            service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
            )
        )
        entries.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            return entries


def _find_all_csvs_recursive(service, root_folder_id, stats=None):
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

    stats, when passed, is a dict this fills in for the diagnostics:
    how many folders were visited, how many CSVs were seen, and a sample
    of the paths that were rejected. Rejection is the quietest failure
    mode here — a correct walk over a correctly-shared folder still
    returns nothing if the layout isn't <year>/<month> — so it needs to
    be visible somewhere.
    """

    csv_files = []

    # Each item is:
    # (folder_id, path_parts)
    folders_to_search = [(root_folder_id, [])]

    while folders_to_search:
        current_id, path_parts = folders_to_search.pop()

        if stats is not None:
            stats["folders_visited"] = stats.get("folders_visited", 0) + 1

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
                if stats is not None:
                    if entry["mimeType"] == "application/vnd.google-apps.shortcut":
                        # Shortcuts are neither folder nor file to this
                        # walk, so a tree reached through one looks empty.
                        stats.setdefault("shortcuts", []).append(
                            "/".join(path_parts + [entry["name"]])
                        )
                    else:
                        stats["non_csv_seen"] = stats.get("non_csv_seen", 0) + 1
                continue

            if stats is not None:
                stats["csvs_seen"] = stats.get("csvs_seen", 0) + 1

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
                if stats is not None:
                    stats["csvs_rejected"] = stats.get("csvs_rejected", 0) + 1
                    sample = stats.setdefault("rejected_sample", [])
                    if len(sample) < 10:
                        sample.append("/".join(path_parts + [entry["name"]]))
                continue

            # ---------------------------------------------------------
            # Store the file, its folder path, and (when the filename
            # carries one) the exact day it covers.
            # ---------------------------------------------------------
            entry = dict(entry)
            entry["folder_path"] = path_parts
            entry["data_date"] = _date_from_name(entry["name"])

            csv_files.append(entry)

    if stats is not None:
        stats["csvs_accepted"] = stats.get("csvs_accepted", 0) + len(csv_files)

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


# ============================================================
# IRRADIANCE LISTING (bifacial-data)
#
# Split in two on purpose:
#
#   _list_available_csvs_cached   cached, RAISES on any failure
#   list_available_csvs           uncached, catches and reports
#
# Reporting has to live outside the cache. Writing to st.session_state
# inside a cached function only runs on a cache MISS, so on a hit the
# error text is never refreshed — and st.cache_data is global while
# session_state is per-session, so a second browser session hitting a
# cached result sees no error at all.
# ============================================================

@st.cache_data(ttl=300)
def _list_available_csvs_cached():
    """
    Returns only irradiance CSV files located inside a valid
    year/month folder structure under bifacial-data.

    Example accepted:

        bifacial-data/2026/08/Bifacial_2026-08-23.csv

    Example ignored:

        bifacial-data/alerts.csv
        bifacial-data/2026/alerts.csv

    Raises DriveListingError rather than returning [] so a failure is
    never cached as a legitimate "no files" answer.
    """
    service = _get_drive_service()

    folders = _get_folder_ids(service, DRIVE_FOLDER_NAME)
    if not folders:
        raise DriveListingError(
            f"No folder named {DRIVE_FOLDER_NAME!r} is visible to the service "
            f"account ({service_account_email()}). Check that the folder "
            "exists, isn't trashed, and is shared with that address."
        )

    stats = {}
    files, seen = [], set()
    for folder in folders:
        for entry in _find_all_csvs_recursive(service, folder["id"], stats=stats):
            if entry["id"] not in seen:
                seen.add(entry["id"])
                files.append(entry)

    if not files:
        raise DriveListingError(
            f"Walked {len(folders)} folder(s) named {DRIVE_FOLDER_NAME!r} "
            f"({stats.get('folders_visited', 0)} subfolders, "
            f"{stats.get('csvs_seen', 0)} CSVs seen) but none sat inside a "
            f"<year>/<month> structure. Rejected: "
            f"{stats.get('csvs_rejected', 0)}. Run the Drive diagnostics for "
            "the rejected paths."
        )

    files.sort(
        key=lambda f: f.get("modifiedTime", ""),
        reverse=True,
    )

    return files


def list_available_csvs():
    """Irradiance CSVs, newest first. Returns [] on failure and puts the
    reason in st.session_state['_drive_list_error']."""
    try:
        files = _list_available_csvs_cached()
        st.session_state["_drive_list_error"] = None
        return files
    except Exception as e:
        st.session_state["_drive_list_error"] = f"{type(e).__name__}: {e}"
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
    'Bifacial_2026-07-29.csv — modified 2026-07-30 03:12'."""
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

    Failures are recorded in st.session_state['_drive_download_errors']
    so a partially-loaded range doesn't look like a complete one.
    """

    if not file_entries:
        return pd.DataFrame()

    dfs = []
    errors = []

    for file_id, modified_time in file_entries:

        try:

            df = _download_single_csv_cached(
                file_id,
                modified_time or "",
            )

            if df is not None and not df.empty:
                dfs.append(df)

        except Exception as e:
            # One failed file should not kill the application, but it
            # should not vanish either.
            errors.append(f"{file_id}: {type(e).__name__}: {e}")
            continue

    st.session_state["_drive_download_errors"] = errors

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

@st.cache_data(ttl=300)
def _list_available_dcm_csvs_cached(include_avg=False):
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

    Raises DriveListingError when the folder or its year/month contents
    can't be found. An empty result AFTER the dcm_3366/avg filter is a
    legitimate answer (that kind of file may genuinely not exist), so
    that case returns [] instead.
    """
    service = _get_drive_service()

    folders = _get_folder_ids(service, DCM_DRIVE_FOLDER_NAME)
    if not folders:
        raise DriveListingError(
            f"No folder named {DCM_DRIVE_FOLDER_NAME!r} is visible to the "
            f"service account ({service_account_email()})."
        )

    stats = {}
    all_files, seen = [], set()
    for folder in folders:
        for entry in _find_all_csvs_recursive(service, folder["id"], stats=stats):
            if entry["id"] not in seen:
                seen.add(entry["id"])
                all_files.append(entry)

    if not all_files:
        raise DriveListingError(
            f"Walked {len(folders)} folder(s) named {DCM_DRIVE_FOLDER_NAME!r} "
            f"({stats.get('folders_visited', 0)} subfolders, "
            f"{stats.get('csvs_seen', 0)} CSVs seen) but none sat inside a "
            "<year>/<month> structure."
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


def list_available_dcm_csvs(include_avg=False):
    """DC-meter CSVs, newest first. Returns [] on failure and puts the
    reason in st.session_state['_dcm_drive_list_error']."""
    try:
        files = _list_available_dcm_csvs_cached(include_avg)
        st.session_state["_dcm_drive_list_error"] = None
        return files
    except Exception as e:
        st.session_state["_dcm_drive_list_error"] = f"{type(e).__name__}: {e}"
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
    errors = []

    for file_id, modified_time in file_entries:
        try:
            df = _download_single_dcm_csv_cached(
                file_id,
                modified_time or "",
            )

            if df is not None and not df.empty:
                dfs.append(df)

        except Exception as e:
            # One failed file should not kill the application.
            errors.append(f"{file_id}: {type(e).__name__}: {e}")
            continue

    st.session_state["_dcm_download_errors"] = errors

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


@st.cache_data(ttl=300)
def _list_available_filled_csvs_cached():
    """
    Returns the gap-filled irradiance CSVs in the OUTPUT folder, newest
    first, each carrying:

        data_date       the day it covers, from the filename
        folder_path     synthesised ["YYYY", "MM"] so the year/month
                        pickers in the UI work unchanged
        flags_id        the matching flags_<date>.csv, or None
        flags_modified  that file's modifiedTime, for cache keying

    A filled file with no matching flags file is still returned.
    Raises DriveListingError rather than returning [].
    """
    service = _get_drive_service()

    folder_id = _resolve_folder_path(service, FILLED_DRIVE_FOLDER_PATH)
    if folder_id is None:
        # The service account may have been given OUTPUT directly,
        # in which case its parent is invisible and the path walk
        # above can't succeed. Fall back to the leaf name.
        folder_id = _get_folder_id(service, FILLED_DRIVE_FOLDER_PATH[-1])
    if folder_id is None:
        raise DriveListingError(
            "Couldn't find the folder "
            f"{'/'.join(FILLED_DRIVE_FOLDER_PATH)} in Drive. Share it "
            f"with the service account ({service_account_email()})."
        )

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

    if not filled_files:
        raise DriveListingError(
            f"Found {len(all_csvs)} CSV(s) under "
            f"{'/'.join(FILLED_DRIVE_FOLDER_PATH)} but none named "
            f"{FILLED_PREFIX}<date>.csv."
        )

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
    return out


def list_available_filled_csvs():
    """Gap-filled CSVs, newest first. Returns [] on failure and puts the
    reason in st.session_state['_filled_drive_list_error']."""
    try:
        files = _list_available_filled_csvs_cached()
        st.session_state["_filled_drive_list_error"] = None
        return files
    except Exception as e:
        st.session_state["_filled_drive_list_error"] = f"{type(e).__name__}: {e}"
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
    errors = []
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
        except Exception as e:
            # One failed file should not kill the application.
            errors.append(f"{file_id}: {type(e).__name__}: {e}")
            continue

    st.session_state["_filled_download_errors"] = errors

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True, copy=False)

    # Days don't have to carry the same sensor set. Where concat filled a
    # missing column with NaN the reading is absent, not reconstructed.
    for column in combined.columns:
        if column.endswith(FILLED_FLAG_SUFFIX):
            combined[column] = combined[column].fillna(False).astype(bool)

    return combined


# =========================
# DIAGNOSTICS
#
# Everything here bypasses the caches on purpose: it answers "what does
# Drive say RIGHT NOW", which is the one question a cached listing can't.
# =========================

def diagnose_drive(folder_name=DRIVE_FOLDER_NAME) -> str:
    """A plain-text report on what the service account can actually see
    under folder_name. Nothing here is cached.

    Distinguishes the three failure modes that all produce an empty
    dropdown:

        1. folder not found / not shared      -> no matches listed
        2. duplicate folders                  -> more than one match
        3. layout doesn't match <year>/<month> -> CSVs seen, 0 accepted
    """
    lines = []
    add = lines.append

    add(f"=== Drive diagnostics: {folder_name!r} ===")
    add(f"time: {datetime.now().isoformat(timespec='seconds')}")
    add(f"service account: {service_account_email()}")
    add("")

    try:
        service = _get_drive_service()
    except Exception as e:
        add(f"FAILED to build Drive client: {type(e).__name__}: {e}")
        add("Check the [gcp_service_account] block in secrets.toml.")
        return "\n".join(lines)

    try:
        folders = _get_folder_ids(service, folder_name)
    except Exception as e:
        add(f"FAILED name lookup: {type(e).__name__}: {e}")
        return "\n".join(lines)

    add(f"folders matching that name: {len(folders)}")
    for f in folders:
        owners = ", ".join(
            o.get("emailAddress", "?") for o in (f.get("owners") or [])
        )
        add(f"  id={f['id']}  parents={f.get('parents')}  owners={owners or '?'}")

    if not folders:
        add("")
        add("=> Nothing with that name is visible. Either the name is wrong,")
        add("   the folder is trashed, or it was never shared with the")
        add("   service account address above.")
        return "\n".join(lines)

    if len(folders) > 1:
        add("")
        add("=> MORE THAN ONE match. Drive does not guarantee ordering on")
        add("   an unsorted list call, so the old files[0] lookup picked a")
        add("   different one at random each time the cache expired. That")
        add("   alone explains intermittent empty dropdowns.")

    add("")
    for f in folders:
        add(f"--- walking {f['id']} ---")
        stats = {}
        try:
            found = _find_all_csvs_recursive(service, f["id"], stats=stats)
        except Exception as e:
            add(f"  WALK FAILED: {type(e).__name__}: {e}")
            status = getattr(getattr(e, "resp", None), "status", None)
            if status is not None:
                add(f"  http status: {status}")
            continue

        add(f"  subfolders visited: {stats.get('folders_visited', 0)}")
        add(f"  CSVs seen:          {stats.get('csvs_seen', 0)}")
        add(f"  CSVs accepted:      {len(found)}")
        add(f"  CSVs rejected:      {stats.get('csvs_rejected', 0)}")
        add(f"  non-CSV files:      {stats.get('non_csv_seen', 0)}")

        shortcuts = stats.get("shortcuts") or []
        if shortcuts:
            add(f"  SHORTCUTS ({len(shortcuts)}) — the walk cannot follow these:")
            for s in shortcuts[:5]:
                add(f"    {s}")

        rejected = stats.get("rejected_sample") or []
        if rejected:
            add("  rejected sample (not inside <year>/<month>):")
            for r in rejected:
                add(f"    {r}")

        if found:
            add("  accepted sample:")
            for entry in found[:5]:
                path = "/".join(entry.get("folder_path") or [])
                add(f"    {path}/{entry['name']}  modified={entry.get('modifiedTime')}")

    add("")
    add("If 'CSVs seen' is high and 'CSVs accepted' is 0, the folder layout")
    add("is the problem, not access or rate limits.")

    return "\n".join(lines)


def render_drive_diagnostics():
    """Sidebar/expander widget wrapping diagnose_drive for all three
    folders, plus a cache-clear button.

    Drop `render_drive_diagnostics()` anywhere in the app (the sidebar is
    a good spot) and re-run it the moment the dropdown comes up empty.
    """
    with st.expander("Drive diagnostics", expanded=False):
        for key, label in [
            ("_drive_list_error", "irradiance listing"),
            ("_dcm_drive_list_error", "DC meter listing"),
            ("_filled_drive_list_error", "gap-filled listing"),
        ]:
            err = st.session_state.get(key)
            if err:
                st.error(f"{label}: {err}")

        for key, label in [
            ("_drive_download_errors", "irradiance downloads"),
            ("_dcm_download_errors", "DC meter downloads"),
            ("_filled_download_errors", "gap-filled downloads"),
        ]:
            errs = st.session_state.get(key) or []
            if errs:
                st.warning(f"{label}: {len(errs)} file(s) failed")
                st.code("\n".join(errs[:10]))

        target = st.selectbox(
            "Folder to probe",
            [DRIVE_FOLDER_NAME, DCM_DRIVE_FOLDER_NAME, FILLED_DRIVE_FOLDER_PATH[-1]],
        )

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Run diagnostics", use_container_width=True):
                with st.spinner("Asking Drive..."):
                    st.session_state["_drive_diag_report"] = diagnose_drive(target)

        with col_b:
            if st.button("Clear caches & rerun", use_container_width=True):
                # Clears the 300s listing caches so the next call really
                # hits Drive. Downloads are keyed on modifiedTime and are
                # left alone.
                _list_available_csvs_cached.clear()
                _list_available_dcm_csvs_cached.clear()
                _list_available_filled_csvs_cached.clear()
                st.rerun()

        report = st.session_state.get("_drive_diag_report")
        if report:
            st.code(report, language="text")
            st.download_button(
                "Download report",
                report,
                file_name="drive_diagnostics.txt",
                use_container_width=True,
            )
