"""Gap filling — run the machine learning model from the browser.

This is the model that also runs in Colab, invoked through
pv_gapfill.run_in_memory() so it returns results rather than writing files to a
notebook's filesystem.

Two constraints shape everything here.

Memory. Streamlit Community Cloud gives the app roughly 1 GB. One day of sensor
data is about 8.7 MB on disk and several times that in pandas, so the range is
capped and the caller is told when the cap bites.

Time. Training takes one to three minutes. Results are cached against the exact
file set used, so pressing the button twice does not retrain, and a rehearsed
demonstration is instant the second time.
"""
import io
import re
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# pv_gapfill computes solar position from the site's own clock, so anything
# handed to it must be in array-local time, not UTC.
LOCAL_TZ = "Asia/Kuala_Lumpur"

import pv_gapfill as G
from drive_fetch import (
    list_available_csvs,
    list_available_dcm_csvs,
    download_and_combine_csvs,
    download_and_combine_dcm_csvs,
)

# Beyond about a week the load itself exhausts memory before training starts.
MAX_DAYS = 7
# Fewer boosting rounds than the offline default: the accuracy difference is
# small and the wait is what makes a live demonstration awkward.
DEMO_MAX_ITER = 400
MAX_PLOT_POINTS = 3000

DATE_IN_NAME = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")

# list_available_csvs() walks the whole bifacial-data tree and returns every
# CSV it finds, meter files included. Keyed by date alone, a meter file for the
# same day overwrote the sensor file, and the sensor branch was then handed
# columns it does not understand. Filter by name so each source only ever sees
# its own files.
DCM_IN_NAME = re.compile(r"dcm|meter", re.I)
SENSOR_IN_NAME = re.compile(r"bifacial[_\s-]*\d{4}", re.I)


def _is_meter_file(name):
    return bool(DCM_IN_NAME.search(name or ""))


def _file_date(name):
    m = DATE_IN_NAME.search(name or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def available_days(source):
    """Dates present in Drive, read live rather than assumed.

    Returns a sorted list of dates. Nothing is hard-coded, so a day uploaded
    this morning appears without a code change.
    """
    files = list_available_csvs() if source == "sensors" else list_available_dcm_csvs()
    out, undated, wrong_kind = {}, [], 0
    for f in files or []:
        name = f.get("name") or ""
        # Keep only files belonging to the chosen source.
        if source == "sensors" and _is_meter_file(name):
            wrong_kind += 1
            continue
        if source == "meters" and not _is_meter_file(name):
            wrong_kind += 1
            continue
        d = _file_date(name)
        if d:
            out[d] = f
        else:
            undated.append(name)
    st.session_state["_gf_wrong_kind"] = wrong_kind
    # Kept so the page can distinguish "no files at all" from "files whose
    # names this code could not read a date out of".
    st.session_state["_gf_undated"] = undated[:10]
    st.session_state["_gf_total_seen"] = len(files or [])
    return sorted(out), out


def _write_temp_csvs(df, folder, source):
    """Write the selected rows to disk in the layout the loader expects.

    The model reads a directory of CSVs, so frames fetched from Drive are
    written back out per day. Wasteful in principle, but the web path and the
    Colab path then run identical code, and a result obtained here is
    reproducible there.

    Returns (paths, note). `note` explains an empty result rather than leaving
    the caller to report "nothing loaded", which says nothing about the cause.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    written = []

    if source == "sensors":
        # Work out the day for each row. Three shapes turn up depending on
        # which loader produced the frame, so all three are handled rather
        # than assuming one.
        day = None
        if "created_at" in df.columns:
            day = pd.to_datetime(df["created_at"], errors="coerce").dt.date
        if (day is None or day.isna().all()) and "Date" in df.columns:
            if "Time" in df.columns:
                day = pd.to_datetime(
                    df["Date"].astype(str).str.strip() + " " +
                    df["Time"].astype(str).str.strip(),
                    errors="coerce").dt.date
            else:
                day = pd.to_datetime(df["Date"], errors="coerce").dt.date
        if day is None:
            return [], (f"The downloaded data has no Date or created_at column. "
                        f"Columns present: {', '.join(map(str, df.columns[:8]))}")
        if day.isna().all():
            sample = df["Date"].iloc[0] if "Date" in df.columns else "?"
            return [], (f"No row had a readable date. First value seen: "
                        f"{sample!r}.")

        keep = df.drop(columns=[c for c in ("created_at",) if c in df.columns])
        for d, chunk in keep.groupby(day):
            if pd.isna(d):
                continue
            p = os.path.join(folder, f"Bifacial_{d}.csv")
            chunk.to_csv(p, index=False)
            written.append(p)
    else:
        # Back to local time before writing. drive_fetch converts meter
        # timestamps to UTC for the dashboard, but pv_gapfill reads its index
        # as local clock time when computing sun position -- a UTC stamp made
        # midday look like midnight, and reconstructed current was clamped to
        # near zero as a result.
        src = "created_at" if "created_at" in df.columns else "Datetime"
        if src not in df.columns:
            return [], (f"No Datetime or created_at column. Columns present: "
                        f"{', '.join(map(str, df.columns[:8]))}")
        ts = pd.to_datetime(df[src], errors="coerce", utc=True)
        if ts.isna().all():
            return [], "No row had a readable timestamp."
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        out = df.copy()
        out["Datetime"] = ts
        for d, chunk in out.groupby(ts.dt.date):
            if pd.isna(d):
                continue
            p = os.path.join(folder, f"{d}_dcm.csv")
            keep = chunk.rename(columns={"device_id": "Device_ID",
                                         "voltage_v": "Voltage_V",
                                         "current_a": "Current_A",
                                         "active_power_kw": "Active_power_kW",
                                         "forward_energy_kwh": "Forward_energy_kWh"})
            cols = [c for c in ("Datetime", "Device_ID", "Forward_energy_kWh",
                                "Active_power_kW", "Current_A", "Voltage_V")
                    if c in keep.columns]
            keep[cols].to_csv(p, index=False)
            written.append(p)

    if not written:
        return [], "The rows had dates, but grouping them by day produced no files."
    return written, None


@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
def _run_model(entries, source, min_hours, max_interp_minutes, exclude_sensors):
    """Fetch, write and run. Cached on the exact file set, so the same request
    does not retrain — which is what makes a second demonstration instant.

    The parameter is called `entries`, NOT `_entries`. Streamlit excludes any
    argument whose name starts with an underscore from the cache key, so the
    earlier name meant the key was `source` alone — and the first meter result
    was replayed for every later request no matter which dates were chosen. A
    tuple of strings hashes fine, so there was never a reason to hide it.

    Deliberately free of Streamlit calls: a cached function that writes to the
    screen raises CacheReplayClosureError on a cache hit, because the elements
    it wrote to no longer exist.
    """
    import tempfile
    fetch = download_and_combine_csvs if source == "sensors" else download_and_combine_dcm_csvs
    df = fetch(tuple(entries))
    if df is None or df.empty:
        return None, "Drive returned no rows for that range."

    folder = tempfile.mkdtemp(prefix="gapfill_")
    paths, note = _write_temp_csvs(df, folder, source)
    if note:
        raise RuntimeError(
            f"{note}  ({len(df):,} rows were downloaded from Drive, so the "
            f"fetch itself worked.)")
    try:
        res = G.run_in_memory(folder, max_iter=DEMO_MAX_ITER,
                              min_hours=min_hours,
                              max_interp_minutes=max_interp_minutes,
                              exclude_sensors=exclude_sensors)
    except Exception as exc:
        # Keep the location, not just the message. "window shape cannot be
        # larger than input array shape" says nothing about which step raised
        # it, and the pipeline has a dozen candidates.
        import traceback, os as _os
        # Walk the whole chain, including any exception this one was raised
        # from. joblib re-raises worker errors, so the final frames are library
        # internals; the useful ones are the deepest that live in this project.
        frames, e = [], exc
        seen = 0
        while e is not None and seen < 4:
            frames.extend(traceback.extract_tb(e.__traceback__))
            e = e.__cause__ or e.__context__
            seen += 1
        here = _os.path.dirname(_os.path.abspath(__file__))
        mine = [f for f in frames if _os.path.abspath(f.filename).startswith(here)]
        pick = mine[-3:] if mine else frames[-3:]
        where = " -> ".join(
            f"{_os.path.basename(f.filename)}:{f.name}:{f.lineno}" for f in pick)
        shape = ""
        try:
            probe = pd.read_csv(paths[0], nrows=5) if paths else None
            if probe is not None:
                shape = (f"  First file has {len(probe.columns)} columns; "
                         f"{len(pd.read_csv(paths[0])):,} rows.")
        except Exception:
            pass
        raise RuntimeError(f"{exc}\n\nRaised in: {where}.{shape}") from exc
        # Raise rather than return, so Streamlit does not cache the failure.
        # A cached error was being replayed for the full TTL, which made every
        # subsequent attempt — on any date — repeat the first message and look
        # as though nothing could be read at all.
        raise RuntimeError(str(exc)) from exc
    # The grid holds every column; keep it, but drop the pre-fill copy, which
    # doubles memory for the sake of one chart.
    res["before_cols"] = {c: res["before"][c].copy() for c in res["found"].get(
        "Irr", res["found"].get("V", {})).values()}
    del res["before"]
    return res, None


def _accuracy_table(acc):
    """Tidy the accuracy report for display.

    evaluate() already returns a DataFrame with one row per channel family and
    gap length, so this only renames the columns into something a reader
    unfamiliar with the code can follow.
    """
    if not isinstance(acc, pd.DataFrame) or acc.empty:
        return pd.DataFrame()
    rename = {
        "family": "Channel", "gap": "Gap length", "n_points": "Points tested",
        "mean_actual": "Mean reading", "MAE_model": "Model MAE",
        "MAE%_model": "Model error %", "MAE_interp": "Interpolation MAE",
        "MAE%_interp": "Interpolation error %", "chosen": "Method used",
        "best_MAE%": "Error %",
    }
    out = acc.rename(columns=rename)
    order = [c for c in ["Channel", "Gap length", "Method used", "Error %",
                         "Model error %", "Interpolation error %",
                         "Model MAE", "Interpolation MAE", "Mean reading",
                         "Points tested"] if c in out.columns]
    out = out[order].copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(2)
    return out


def _zip_outputs(res):
    """Filled data and the flag file, as one download.

    Both are included deliberately. In the filled file a reconstructed value is
    indistinguishable from a measured one; the flag file is the only record of
    which is which, and anyone reporting measurements needs it.
    """
    grid, found, filled = res["grid"], res["found"], res["filled"]
    cols = [c for m in found.values() for c in m.values()]
    # Excluded positions are written out too. They were held out of the model,
    # not out of the record, so they can only ever read measured or empty.
    cols += [c for c in res.get("excluded_cols", []) if c in grid.columns]
    flags = pd.DataFrame("measured", index=grid.index, columns=cols)
    for c, mask in filled.items():
        if c in flags.columns:
            flags.loc[np.asarray(mask), c] = "filled_gap"
    for c in cols:
        flags.loc[grid[c].isna().values, c] = "empty_gap"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("filled.csv", grid[cols].to_csv())
        z.writestr("flags.csv", flags.to_csv())
        z.writestr("README.txt",
                   "filled.csv  — measurements with short gaps reconstructed\n"
                   "flags.csv   — provenance of every cell:\n"
                   "                measured    the logger recorded it\n"
                   "                filled_gap  the model reconstructed it\n"
                   "                empty_gap   no honest basis existed\n\n"
                   + (f"Held out of the model (bench/test positions): "
                      f"{', '.join(sorted(res.get('excluded_cols', [])))}\n"
                      f"Their values are the logger's own and were never used\n"
                      f"as peers or training data.\n\n"
                      if res.get("excluded_cols") else "")
                   +
                   "A filled value looks identical to a measured one in\n"
                   "filled.csv. Consult flags.csv before reporting anything\n"
                   "as a measurement.\n")
    buf.seek(0)
    return buf


def render_gap_filling():
    st.subheader("Gap filling")
    st.caption(
        "Runs the machine learning model over data from Drive, reconstructing "
        "short gaps in the record. The same model runs offline in Colab; this "
        "is the same code, invoked from the browser."
    )

    source = st.radio(
        "Data source", ["sensors", "meters"], horizontal=True,
        format_func=lambda s: "Irradiance and temperature" if s == "sensors"
                              else "DC meters",
        key="gf_source")

    try:
        days, lookup = available_days(source)
    except Exception as exc:
        st.error(f"Could not list files in Drive: {exc}")
        return

    if not days:
        st.warning("No dated CSVs found in Drive for that source.")
        # list_available_csvs() catches every exception and returns an empty
        # list, stashing the reason in session state. Without surfacing it, a
        # credentials or permissions failure is indistinguishable from an empty
        # folder — which is why this looked like "cannot read anything".
        seen = st.session_state.get("_gf_total_seen", 0)
        undated = st.session_state.get("_gf_undated") or []
        if seen:
            st.info(
                f"Drive returned {seen} file(s), but no date could be read from "
                f"their names. Expected a name containing YYYY-MM-DD."
            )
            if undated:
                st.caption("Examples: " + ", ".join(undated[:5]))
        err = st.session_state.get("_drive_list_error")
        if err:
            st.error(f"Drive reported: {err}")
        else:
            st.caption(
                "Drive answered without an error, so the folder was reachable "
                "but held no files whose names contain a date."
            )
        with st.expander("What to check"):
            st.markdown(
                "- The `bifacial-data` folder is shared with the service "
                "account address in `gcp_service_account.client_email`\n"
                "- Files sit under a `year/month` folder, e.g. "
                "`bifacial-data/2026/09/Bifacial_2026-09-09.csv`\n"
                "- Filenames contain a date as `YYYY-MM-DD`\n"
                "- The DC meter option works, so credentials are valid — if "
                "only this source fails, it is the folder or the naming, not "
                "authentication"
            )
        return

    # What is actually there, read live — so nobody picks a date with no data.
    months = sorted({(d.year, d.month) for d in days}, reverse=True)
    wrong = st.session_state.get("_gf_wrong_kind", 0)
    st.caption(
        f"{len(days)} day(s) available in Drive, "
        f"{days[0]:%d %b %Y} to {days[-1]:%d %b %Y} · "
        + ", ".join(f"{date(y, m, 1):%b %Y}" for y, m in months[:6])
        + (" …" if len(months) > 6 else "")
        + (f" · {wrong} file(s) of the other type ignored" if wrong else "")
    )

    mode = st.radio("Select by", ["Single day", "Date range", "Whole month"],
                    horizontal=True, key="gf_mode")

    if mode == "Single day":
        pick = st.selectbox("Day", days, index=len(days) - 1,
                            format_func=lambda d: f"{d:%d %b %Y}", key="gf_day")
        chosen = [pick]
    elif mode == "Whole month":
        ym = st.selectbox("Month", months,
                          format_func=lambda t: f"{date(t[0], t[1], 1):%B %Y}",
                          key="gf_month")
        chosen = [d for d in days if (d.year, d.month) == ym]
    else:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("From", value=days[-1], min_value=days[0],
                                  max_value=days[-1], key="gf_from")
        with c2:
            end = st.date_input("To", value=days[-1], min_value=days[0],
                                max_value=days[-1], key="gf_to")
        chosen = [d for d in days if start <= d <= end]

    if not chosen:
        st.warning("No files in Drive for that selection.")
        return

    if len(chosen) > MAX_DAYS:
        st.info(
            f"{len(chosen)} days selected; using the most recent {MAX_DAYS}. "
            f"Beyond that the data no longer fits in memory, and filling a gap "
            f"does not need data from weeks either side of it."
        )
        chosen = chosen[-MAX_DAYS:]

    # A grid is built across the whole span, so a two-day selection nine days
    # apart produces eight days of empty cells that all count as gaps. In one
    # test that turned 7,517 real rows into 5.4 million reported gaps.
    span_days = (chosen[-1] - chosen[0]).days + 1
    if span_days > len(chosen):
        st.warning(
            f"The {len(chosen)} day(s) selected span {span_days} days. Missing "
            f"days in between are counted as gaps, which will inflate the gap "
            f"total and depress coverage. Choose consecutive days for a "
            f"meaningful figure."
        )

    entries = tuple((lookup[d]["id"], lookup[d].get("modifiedTime")) for d in chosen)
    st.write(f"**{len(chosen)} day(s) selected** — "
             f"{chosen[0]:%d %b} to {chosen[-1]:%d %b %Y}")

    # Partial days are common here: the logger is restarted, or today is only
    # half over. An hour is a sensible default for training, but a shorter day
    # is still worth reconstructing.
    min_hours = st.slider(
        "Minimum usable hours per day", 0.1, 4.0, 1.0, 0.1, key="gf_minhours",
        help="A day with less than this is excluded from training. Lower it to "
             "include a partial day; raise it to train only on full ones.")

    # An outage stops every channel at once, so no peer survives to predict
    # from and the model declines. Interpolation had no such limit and drew a
    # straight line across the hole instead -- 18% of the fills in the 03-09
    # Sep run were single runs longer than half an hour, including a 38-minute
    # diagonal from 756 W/m2 down to zero that no sensor recorded, and an
    # 82-minute one across three channels at once.
    max_interp = st.slider(
        "Longest gap to interpolate (minutes)", 1.0, 60.0, 30.0, 1.0,
        key="gf_maxinterp",
        help="Holes longer than this are left empty and flagged, rather than "
             "closed with a straight line. Raise it only if you are willing to "
             "publish a line drawn between two readings an hour apart.")

    # Positions being bench-tested rather than measuring the array. Their
    # numbers are real numbers, so nothing downstream can tell they are not
    # sky, and one of them measuring is enough to license a fill on a channel
    # that has stopped. Kept manual: a channel reading much higher than its
    # neighbours might be on a lamp, or might be the only one not in shade.
    exclude = st.multiselect(
        "Sensor positions to exclude from the model", list(range(1, 25)),
        default=[], key="gf_exclude",
        help="Positions under test rather than on the array. Excluded channels "
             "are not used as peers, not trained on, and not reconstructed. "
             "Their recorded values still appear in the download, flagged as "
             "measured.")

    if st.button("Run gap filling", type="primary", key="gf_run"):
        with st.status("Running the model…", expanded=True) as status:
            st.write("Fetching from Drive, then training. One to three minutes.")
            try:
                res, err = _run_model(entries, source, min_hours, max_interp,
                                      tuple(sorted(exclude)))
            except Exception as exc:
                status.update(label="Could not complete", state="error")
                st.error(str(exc))
                st.caption(
                    "This result is not cached, so correcting the selection "
                    "and pressing again will retry rather than repeat."
                )
                return
            if err:
                status.update(label="Could not complete", state="error")
                st.error(err)
                return
            status.update(label="Finished", state="complete", expanded=False)
        st.session_state["_gf_result"] = res

    res = st.session_state.get("_gf_result")
    if not res:
        st.info("Choose a period and select **Run gap filling** to start.")
        return

    # ---- headline numbers -------------------------------------------------
    #
    # Coverage is quoted against gaps that COULD be filled, not against every
    # empty cell. A channel that never reported is not a gap the model can
    # reconstruct -- there is nothing to infer from -- and counting those cells
    # made a run that filled nearly every real gap report 0.6%.
    live_cols = res.get("live_cols", 0)
    silent_cols = res.get("silent_cols", 0)
    live_gaps = res.get("live_gaps", res["total_gaps"])
    live_cov = res.get("live_coverage", res["coverage"])

    a, b, c, d = st.columns(4)
    a.metric("Channels reporting", f"{live_cols} of {live_cols + silent_cols}")
    b.metric("Gaps in those channels", f"{live_gaps:,}")
    c.metric("Reconstructed", f"{res['filled_cells']:,}")
    d.metric("Coverage", f"{100 * live_cov:.0f}%")
    st.caption(
        f"Gaps longer than {res.get('max_interp_minutes', 0):g} min were left "
        f"empty rather than closed with a straight line. "
        f"Coverage is measured against gaps in channels that reported at some "
        f"point during the period. Trained on {len(res['train_days'])} day(s); "
        f"accuracy measured on {len(res['test_days'])} held-out day(s)."
    )

    if silent_cols:
        st.warning(
            f"**{silent_cols} of {live_cols + silent_cols} channels reported "
            f"nothing at all** during this period — {res.get('silent_gaps', 0):,} "
            f"empty cells — this counts every stretch with no instrument "
            f"attached, including days a channel that reported elsewhere in "
            f"the period was not recording. These are not gaps that can be "
            f"reconstructed: with no reading, there is nothing to infer from. "
            f"They are excluded from the coverage figure above and left empty in "
            f"the output. This is a hardware matter, not a modelling one."
        )

    status = res.get("live_status") or {}
    if status:
        with st.expander("Which channels counted as reporting, and why"):
            tab = pd.DataFrame.from_dict(status, orient="index")
            tab.index.name = "channel"
            tab = tab.rename(columns={"family": "Family", "measured": "Readings",
                                      "hours_recorded": "Hours recorded",
                                      "days_recording": "Days recording",
                                      "days_in_range": "Days in range",
                                      "live": "Reporting"})
            tab = tab.sort_values(["Reporting", "Days recording", "Hours recorded"],
                                  ascending=[False, False, False])
            st.dataframe(tab, use_container_width=True)
            st.caption(
                "A channel counts as reporting when it recorded at least the "
                "minimum usable hours set above, at its own write rate. Below "
                "that it is left exactly as the logger wrote it: nothing is "
                "reconstructed into it and it is not counted in coverage. The "
                "irradiance, temperature and average columns of one sensor are "
                "three channels, so three working sensors is nine. **Days "
                "recording** is the one to read when sensors are moved between "
                "positions: a channel is only reconstructed on its own "
                "recording days, so a channel showing 2 of 7 is untouched on "
                "the other five."
            )

    if len(res["usable_days"]) < len(chosen):
        st.warning(
            f"{len(chosen)} day(s) were requested but only "
            f"{len(res['usable_days'])} had enough usable data to train on."
        )
    if len(res["test_days"]) == 0:
        st.info(
            "No day could be held back for testing, so no accuracy figure can "
            "be produced. Select at least two days with data."
        )

    if res.get("dropped_days"):
        with st.expander(f"{len(res['dropped_days'])} day(s) excluded from training"):
            st.caption(
                "A day needs at least the minimum usable hours set above, across "
                "enough channels. Days below that are listed here rather than "
                "being silently dropped."
            )
            st.write(", ".join(str(x) for x in res["dropped_days"]))

    if live_cov < 0.5 and live_gaps:
        st.info(
            f"{100 * (1 - live_cov):.0f}% of the fillable gaps were still left "
            f"empty. That is the right outcome when an outage takes every "
            f"channel at once: with no peer recording at that moment, there is "
            f"no honest basis for a value."
        )

    # ---- accuracy ---------------------------------------------------------
    no_model = [f for f in res.get("found", {})
                if f in G.MODEL_FAMILIES and not res.get("models_ok", {}).get(f, True)]
    if no_model:
        st.info(
            f"No model could be trained for: {', '.join(no_model)}. Those gaps "
            f"were filled by interpolation instead, which is often the better "
            f"method for short gaps in any case."
        )

    st.markdown("#### Accuracy")
    table = _accuracy_table(res.get("accuracy"))
    if table.empty:
        st.warning("Not enough held-out gaps in this period to measure accuracy. "
                   "Try a longer range.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Measured on days withheld from training. **Error %** is mean "
            "absolute error as a share of the channel's typical reading. The "
            "model and interpolation columns are both shown, and whichever is "
            "better is used for that channel and gap length — so the choice is "
            "visible rather than assumed."
        )
        best = table.dropna(subset=["Error %"])
        if not best.empty:
            r = best.loc[best["Error %"].idxmin()]
            st.success(
                f"Best result: **{r['Channel']}** reconstructed to "
                f"**{r['Error %']}% error** on {r['Gap length']} gaps, "
                f"using {r['Method used']}."
            )
        won = (table["Method used"] == "model").sum() if "Method used" in table else 0
        st.caption(
            f"The model was the better method for {won} of {len(table)} "
            f"channel-and-gap-length combinations; interpolation won the rest. "
            f"Showing both is what makes that claim checkable."
        )

    # ---- before and after -------------------------------------------------
    st.markdown("#### Before and after")
    grid, found, filled = res["grid"], res["found"], res["filled"]
    fam = "Irr" if "Irr" in found else next(iter(found), None)
    if fam:
        cols = list(found[fam].values())
        pick = st.selectbox("Channel", cols, key="gf_chan")
        s_after = pd.to_numeric(grid[pick], errors="coerce")
        mask = np.asarray(filled.get(pick, np.zeros(len(grid), bool)))

        step = max(1, len(s_after) // MAX_PLOT_POINTS)
        idx = s_after.index[::step]
        plot = pd.DataFrame({
            "measured": s_after[::step].where(~mask[::step]),
            "reconstructed": s_after[::step].where(mask[::step]),
        }, index=idx)
        st.line_chart(plot, use_container_width=True)
        st.caption(
            f"{int(mask.sum()):,} of {len(s_after):,} points on this channel "
            f"were reconstructed. Where the orange trace is absent, no value "
            f"could be justified and the gap was left open."
        )

    # ---- download ---------------------------------------------------------
    st.markdown("#### Download")
    st.download_button(
        "Download filled data and flags (ZIP)",
        data=_zip_outputs(res),
        file_name=f"gapfill_{chosen[0]}_{chosen[-1]}.zip",
        mime="application/zip",
    )
    st.caption(
        "Two files. In the filled data a reconstructed value is "
        "indistinguishable from a measured one — the flag file is the only "
        "record of which is which."
    )
