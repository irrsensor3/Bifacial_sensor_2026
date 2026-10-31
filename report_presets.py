"""Saved report formats for the Data & Reports page.

A "format" is a snapshot of the page's layout choices -- title, notes, Word or
PDF, what to include, how to split, which columns to chart, the Y-axis limits.
Formats are named, kept in one JSON file, and re-applied with a click in any
later session.

WHERE THE FILE LIVES
    Streamlit Community Cloud wipes the app's disk on every reboot, redeploy
    and container recycle, so a local file there would lose every format
    sooner or later. The formats are therefore kept in a file in Google Drive,
    which the app already has a service account for.

    One-time setup:
      1. On your computer create a file named report_presets.json containing
         exactly:  {}
      2. Upload it to Google Drive (anywhere outside the bifacial-data and
         panel-meter-data folders).
      3. Share it with the service account's address as EDITOR. The address is
         the client_email in the [gcp_service_account] secrets block, and is
         also printed in this page's "Saved report formats" panel.
      4. Open the file in Drive and copy the ID out of the URL
         (drive.google.com/file/d/<THIS PART>/view).
      5. Add one line to the app secrets:
             REPORT_PRESET_FILE_ID = "<THIS PART>"

    Without that secret the formats are written to report_presets.json next to
    the app instead. That works while testing, but on Cloud it will not
    survive a restart, and the panel says so.

WHO CAN CHANGE THEM
    Everyone can apply a format. Only admins can save or delete one (the file
    is shared by every user). Set PRESETS_ADMIN_ONLY = False to let guests too.
"""

import json
from io import BytesIO
from pathlib import Path

import streamlit as st

from drive_fetch import service_account_email

# The session_state keys that make up a "report format". Selected files and
# the X-range slider are left out on purpose: they describe one dataset, not a
# layout.
PRESET_KEYS = [
    "report_title", "observation",
    "report_format", "batch_format",
    "single_content", "batch_content", "batch_gran",
    "selected_temps", "selected_irradiance",
    "temp_y_auto", "temp_y_min", "temp_y_max",
    "irr_y_auto", "irr_y_min", "irr_y_max",
]

PRESETS_ADMIN_ONLY = True
LOCAL_PRESET_FILE = Path("report_presets.json")
SECRET_NAME = "REPORT_PRESET_FILE_ID"

# Drive's full scope, not the read-only one the rest of the app uses: updating
# a file that somebody else owns needs write access.
_SCOPES = ["https://www.googleapis.com/auth/drive"]

_CACHE_KEY = "_presets_cache"
_ERROR_KEY = "_presets_error"


class PresetStoreError(RuntimeError):
    """The formats file could not be read or written. The message is written
    for the person looking at the page."""


# ---------------------------------------------------------------------------
#  Storage
# ---------------------------------------------------------------------------

def _file_id():
    try:
        value = st.secrets.get(SECRET_NAME)
    except Exception:
        return None
    return str(value).strip() if value else None


def storage_label():
    return "Google Drive" if _file_id() else "this server only"


@st.cache_resource
def _drive():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _http_message(exc, file_id):
    status = getattr(getattr(exc, "resp", None), "status", None)
    sa = service_account_email()
    if status == 404:
        return (f"Drive can't find file {file_id}. Check {SECRET_NAME} in the "
                f"app secrets, and that the file is shared with {sa}.")
    if status == 403:
        return (f"The service account can see the formats file but may not "
                f"change it. Share it with {sa} as Editor.")
    return f"Drive error ({status}): {exc}"


def _parse(raw):
    """JSON bytes/text to a dict. An empty file is an empty store; a file that
    is not JSON is an ERROR, not an empty store -- treating it as empty would
    let the next save wipe whatever was in it."""
    text = raw.decode("utf-8-sig") if isinstance(raw, (bytes, bytearray)) else str(raw)
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PresetStoreError(
            f"The formats file is not valid JSON ({exc}). Replace its "
            f"contents with {{}} to start again.") from exc
    if not isinstance(data, dict):
        raise PresetStoreError("The formats file must contain a JSON object, "
                               "e.g. {}.")
    return data


def _read_store():
    fid = _file_id()
    if fid is None:
        try:
            return _parse(LOCAL_PRESET_FILE.read_bytes())
        except FileNotFoundError:
            return {}

    from googleapiclient.errors import HttpError
    try:
        raw = _drive().files().get_media(
            fileId=fid, supportsAllDrives=True).execute()
    except HttpError as exc:
        raise PresetStoreError(_http_message(exc, fid)) from exc
    except Exception as exc:
        raise PresetStoreError(f"Couldn't reach Drive: {exc}") from exc
    return _parse(raw)


def _write_store(presets):
    payload = json.dumps(presets, indent=2, ensure_ascii=False,
                         default=str).encode("utf-8")
    fid = _file_id()
    if fid is None:
        try:
            LOCAL_PRESET_FILE.write_bytes(payload)
        except OSError as exc:
            raise PresetStoreError(f"Couldn't write {LOCAL_PRESET_FILE}: {exc}") from exc
        return

    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseUpload
    try:
        media = MediaIoBaseUpload(BytesIO(payload), mimetype="application/json",
                                  resumable=False)
        _drive().files().update(
            fileId=fid, media_body=media, supportsAllDrives=True).execute()
    except HttpError as exc:
        raise PresetStoreError(_http_message(exc, fid)) from exc
    except Exception as exc:
        raise PresetStoreError(f"Couldn't reach Drive: {exc}") from exc


def load_presets(force=False):
    """(presets, error). Cached per session so the page does not call Drive on
    every rerun; `force` re-reads it. A failed read is cached too, so a broken
    setup costs one Drive call, not one per click."""
    if not force and _CACHE_KEY in st.session_state:
        return st.session_state[_CACHE_KEY], st.session_state.get(_ERROR_KEY)
    try:
        presets, error = _read_store(), None
    except PresetStoreError as exc:
        presets, error = {}, str(exc)
    st.session_state[_CACHE_KEY] = presets
    st.session_state[_ERROR_KEY] = error
    return presets, error


# ---------------------------------------------------------------------------
#  Callbacks (run before the page reruns, so they may set widget keys)
# ---------------------------------------------------------------------------

def _say(kind, text):
    st.session_state["_preset_msg"] = (kind, text)


def _can_edit():
    return (not PRESETS_ADMIN_ONLY) or st.session_state.get("user_role") == "admin"


def _coerce(key, value):
    # Y-axis limits are float widgets; an int that slipped through JSON
    # ("50" instead of "50.0") makes number_input raise a type error.
    if key.endswith(("_min", "_max")):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _save_cb():
    if not _can_edit():
        return _say("error", "Only an admin can save formats.")
    name = str(st.session_state.get("preset_name", "")).strip()
    if not name:
        return _say("error", "Give the format a name first.")
    if len(name) > 60:
        return _say("error", "Keep the name under 60 characters.")
    try:
        # Re-read first, so a format somebody else saved a minute ago is
        # merged rather than overwritten. If this read fails the save is
        # abandoned: writing back a partial list would delete the rest.
        presets = _read_store()
        presets[name] = {k: st.session_state[k] for k in PRESET_KEYS
                         if k in st.session_state}
        _write_store(presets)
    except PresetStoreError as exc:
        return _say("error", str(exc))
    st.session_state[_CACHE_KEY] = presets
    st.session_state[_ERROR_KEY] = None
    _say("success", f"Saved “{name}”.")


def _apply_cb():
    name = st.session_state.get("preset_pick")
    preset = load_presets()[0].get(name)
    if not preset:
        return _say("error", "That format is no longer in the list.")
    for key, value in preset.items():
        if key in PRESET_KEYS:
            st.session_state[key] = _coerce(key, value)
    _say("success", f"Applied “{name}”.")


def _delete_cb():
    if not _can_edit():
        return _say("error", "Only an admin can delete formats.")
    name = st.session_state.get("preset_pick")
    try:
        presets = _read_store()
        if presets.pop(name, None) is None:
            return _say("error", "That format was already gone.")
        _write_store(presets)
    except PresetStoreError as exc:
        return _say("error", str(exc))
    st.session_state[_CACHE_KEY] = presets
    st.session_state[_ERROR_KEY] = None
    st.session_state.pop("preset_pick", None)
    _say("success", f"Deleted “{name}”.")


def _reload_cb():
    load_presets(force=True)
    _say("success", "Reloaded the list.")


# ---------------------------------------------------------------------------
#  UI
# ---------------------------------------------------------------------------

def _describe(preset):
    """One line saying what a format holds, so the list is not just names."""
    bits = []
    if preset.get("report_title"):
        bits.append(str(preset["report_title"]))
    if preset.get("batch_content"):
        bits.append(str(preset["batch_content"]))
    if preset.get("batch_gran"):
        bits.append(f"split by {str(preset['batch_gran']).lower()}")
    if preset.get("batch_format"):
        bits.append(str(preset["batch_format"]))
    nt = len(preset.get("selected_temps") or [])
    ni = len(preset.get("selected_irradiance") or [])
    if nt or ni:
        bits.append(f"{nt} temperature + {ni} irradiance columns")
    return " · ".join(bits)


def render_preset_bar():
    presets, error = load_presets()
    can_edit = _can_edit()

    with st.expander("💾 Saved report formats", expanded=False):
        if error:
            st.error(error)
        if storage_label() != "Google Drive":
            st.caption(
                f"Formats are stored on this server only, so a restart or "
                f"redeploy will erase them. To keep them, set {SECRET_NAME} in "
                f"the app secrets (see the top of report_presets.py). Service "
                f"account: {service_account_email()}")

        if presets:
            if st.session_state.get("preset_pick") not in presets:
                st.session_state.pop("preset_pick", None)
            st.selectbox("Saved formats", list(presets), key="preset_pick")
            picked = st.session_state.get("preset_pick")
            if picked in presets:
                st.caption(_describe(presets[picked]))

            apply_col, delete_col, reload_col = st.columns(3)
            apply_col.button("Apply", on_click=_apply_cb, key="preset_apply",
                             type="primary")
            if can_edit:
                delete_col.button("Delete", on_click=_delete_cb,
                                  key="preset_delete")
            reload_col.button("Reload list", on_click=_reload_cb,
                              key="preset_reload")
        else:
            st.caption("No saved formats yet.")
            st.button("Reload list", on_click=_reload_cb, key="preset_reload")

        if can_edit:
            st.text_input("Save the current settings as", key="preset_name",
                          placeholder="e.g. Monthly summary")
            st.button("Save format", on_click=_save_cb, key="preset_save")
        else:
            st.caption("Only an admin can save or delete formats.")

        message = st.session_state.pop("_preset_msg", None)
        if message:
            getattr(st, message[0])(message[1])
