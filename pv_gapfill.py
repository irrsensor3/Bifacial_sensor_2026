# =============================================================================
#  Bifacial PV Logger — Gap Filling with Panel Configuration
#  Part 1 of 2: irradiance / IrrAvg / temperature
# =============================================================================
#
#  SITE LAYOUT (edit the ROWS / SENSORS_PER_ROW block below if panels move)
#      16 bifacial panels in 4 rows of 4, plus 8 front reference sensors --
#      one at EACH short end of every row (Front A and Front B).
#
#          row 1:  ch1=FrontA  ch2..ch5 = rear panels 1-4  ch6=FrontB
#          row 2:  ch7=FrontA  ch8..ch11                   ch12=FrontB
#          row 3:  ch13=FrontA ch14..ch17                  ch18=FrontB
#          row 4:  ch19=FrontA ch20..ch23                  ch24=FrontB
#
#      Rear positions 1 and 4 sit at the row ends and catch more ground-
#      reflected light; positions 2 and 3 are interior and catch less.
#
#  WHY CONFIGURATION LABELS MATTER
#      A rear sensor at position 1 and one at position 2 do not read alike, so
#      pooling every panel as interchangeable teaches the model an average that
#      fits neither. Pooling is still what lets one model serve any panel count
#      -- it just needs to know which panel is which. Face, row and position
#      therefore become model features, and because every row is bracketed by
#      its own front pair the model also gets row_front_ref (that row's own sky)
#      and row_front_span (Front B - Front A, non-zero when a cloud edge is
#      partway across the row -- exactly when the two ends diverge).
#
#  WHAT THIS DOES NOT DO
#      It never alters a value you logged. No spike rejection, no clipping, no
#      offset correction. A reading is either passed through untouched or it is
#      a gap. Fault values (-30, negative temperatures) stay in the output so a
#      broken sensor stays visible, and are excluded from training so the model
#      never learns that -9.5 C is a normal module temperature.
#
#  ACCURACY REPORTING
#      Days are split 70/30. The model trains only on the 70%. Synthetic gaps
#      are then cut into the held-out 30% and refilled, and the error against
#      your logged values is reported. Splitting by DAY rather than by row is
#      deliberate: a random row split puts 14:30:00 in training and 14:30:30 in
#      test, so the model reads the answer off the adjacent row. That scored
#      MAE 2.2 on an earlier build while realistic gaps scored 118.
# =============================================================================

import os
import re
import glob
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
#  CONFIGURATION
# =============================================================================

DATA_ROOT = "/content/drive/MyDrive/Pi data downloaded by Hng/UNZIPPED"
OUT_DIR = "/content/drive/MyDrive/Pi data downloaded by Hng/OUTPUT"

LAT, LON, TZ_HOURS = 3.02, 101.62, 8.0     # Puchong, Selangor

# --- PANEL MAP: channel number -> (face, array, position) --------------------
#   face     "front" or "rear"
#   row      1..4, the physical row the sensor belongs to (was called "array"
#            when the rig was 5 arrays of 4; it is a row now)
#   position 0 = Front A, 1-4 = rear panel under that position, 5 = Front B
#
#   Change this block if the rig is re-wired. Nothing else needs editing.
# Layout: 4 rows, 6 sensors each = 24 channels.
#   Front A  at one short end of the row      (position 0)
#   4 rear sensors, one under each panel      (positions 1-4)
#   Front B  at the other short end           (position 5)
# Rear positions 1 and 4 are row-end panels and catch more ground-reflected
# light; 2 and 3 are interior and catch less. Position is therefore a real
# physical covariate, not a label.
ROWS = 4
SENSORS_PER_ROW = 6
REAR_PER_ROW = 4
FRONT_A_POS, FRONT_B_POS = 0, 5

PANEL_MAP = {}
for _r in range(ROWS):
    _base = _r * SENSORS_PER_ROW + 1
    PANEL_MAP[_base] = ("front", _r + 1, FRONT_A_POS)
    for _k in range(REAR_PER_ROW):
        PANEL_MAP[_base + 1 + _k] = ("rear", _r + 1, _k + 1)
    PANEL_MAP[_base + 1 + REAR_PER_ROW] = ("front", _r + 1, FRONT_B_POS)

FRONT_CHANNELS = [p for p, v in PANEL_MAP.items() if v[0] == "front"]
N_ARRAYS, N_POSITIONS = ROWS, REAR_PER_ROW

VERIFY_LAYOUT = True     # cross-check the map against measured irradiance

# PANEL_MAP above describes the 24-channel rig. When loading OLDER data from a
# different rig, that map is wrong and the model trains on mislabelled faces --
# which measurably distorts accuracy. With this True, the face (front/rear) is
# taken from what the sensors actually read and only array/position come from
# the map. Leave False for the 24-channel logger, where the map is authoritative.
AUTO_FACE_FROM_DATA = False

# Filenames containing any of these are skipped entirely. Use this when a
# different rig reuses the same column names: the 2_ADS files are a 4-sensor
# test rig whose Irr_1..Irr_4 are physically different sensors from the main
# rig's, so stacking them teaches the model that one column means two things.
EXCLUDE_FILE_PATTERNS = ["2_ADS"]

# --- Behaviour ---------------------------------------------------------------
KEEP_OUT_OF_RANGE = True    # keep faulty readings visible in the output
KEEP_FAULTCODES = True
MAX_FILL_GAP_HOURS = 2.0    # gaps only interpolation can reach: fill whole, or
                            # leave whole. Never a partial fill cut mid-gap.
TEST_FRACTION = 0.30        # held-out days
RANDOM_STATE = 42

# --- Data validation ---------------------------------------------------------
MIN_USABLE_HOURS = 1.0      # a day with less than this is dropped from training
MAX_COLUMN_FAULT_FRAC = 0.50  # a column worse than this is dropped for that day

# --- Model -------------------------------------------------------------------
MAX_TRAIN_ROWS = 300_000
# Measured on this data: 300 iterations hit the ceiling without early stopping
# ever triggering, i.e. the model was capacity-limited rather than converged.
# Raising it cut error from 10.8% to 8.6%, and at 1500 early stopping finally
# halted at ~1250, so this is the point of diminishing returns, not a guess.
MAX_ITER = 1500
LEARNING_RATE = 0.05
MAX_LEAF_NODES = 63
MIN_PEERS_TO_PREDICT = 1
LAG_STEPS = (-11, -1, 1, 11)
BUCKET_SPLIT_MINUTES = 15
SHORT_BUCKET_MINUTES = (1, 3, 8, 15)
LONG_BUCKET_MINUTES = (30, 60, 120)
GAPS_PER_FAMILY_PER_BUCKET = 120

NIGHT_ELEV_DEG = -5.0
NIGHT_OFFSET_LIMIT = 15.0
SENTINELS = (-30.0,)
FLOAT_DECIMALS = 3
TIME_FORMAT = "%H:%M:%S"    # Excel renders "%H:%M:%S.%f" as elapsed time
PER_DAY_OUTPUT = True

CHANNELS = {
    "Irr": {"patterns": [r"^Irr[_\s]?(\d+)$", r"^Irradiance[_\s]?(\d+)$"],
            "lo": -15.0, "hi": 1500.0, "night": "zero", "derived": None},
    "Temp": {"patterns": [r"^Temp[_\s]?(\d+)$", r"^Temperature[_\s]?(\d+)$"],
             "lo": 5.0, "hi": 90.0, "night": "free", "derived": None},
    "IrrAvg": {"patterns": [r"^IrrAvg[_\s]?(\d+)$", r"^Irr[_\s]?avg[_\s]?(\d+)$"],
               "lo": -15.0, "hi": 1500.0, "night": "zero",
               "derived": {"op": "rolling_mean", "source": "Irr"}},
    # ---- Part 2 hooks: enable when the electrical CSVs are wired in ---------
    # "V": {"patterns": [r"^V[_\s]?(\d+)$"], "lo": -5.0, "hi": 100.0,
    #       "night": "free", "derived": None},
    # "I": {"patterns": [r"^I[_\s]?(\d+)$"], "lo": -1.0, "hi": 30.0,
    #       "night": "zero", "derived": None},
    # "P": {"patterns": [r"^P[_\s]?(\d+)$"], "lo": -5.0, "hi": 1000.0,
    #       "night": "zero", "derived": {"op": "product", "sources": ["V", "I"]}},
}
# --- Electrical channels, from the DC meter CSVs --------------------------
# These arrive in LONG format (one row per device per sample) rather than the
# wide layout the irradiance logger writes; load_long_csv() pivots them so the
# rest of the pipeline sees the same shape either way.
CHANNELS.update({
    "V": {"patterns": [r"^V[_\s]?(\d+)$"], "lo": -5.0, "hi": 100.0,
          "night": "free", "derived": None},
    "I": {"patterns": [r"^I[_\s]?(\d+)$"], "lo": -2.0, "hi": 40.0,
          "night": "zero", "derived": None},
    # Checked against 63,735 daylight rows: median |P - V*I| = 0.27 W. Power is
    # arithmetic, so it is recomputed rather than predicted -- a model would
    # emit a power that disagrees with its own voltage and current.
    "P": {"patterns": [r"^P[_\s]?(\d+)$"], "lo": -5.0, "hi": 5.0,
          "night": "zero", "derived": {"op": "product", "sources": ["V", "I"]}},
    # Cumulative energy only ever rises, so a gap is bounded by the readings on
    # either side and any decrease is a fault rather than something to fill.
    "E": {"patterns": [r"^E[_\s]?(\d+)$"], "lo": 0.0, "hi": 1e6,
          "night": "free", "derived": {"op": "monotonic"}},
})

# Meters 11-26 are bifacial, 27-30 monofacial. Measured over 7 days the mono
# group runs 7.7% lower on current -- that is the bifacial gain, not a fault,
# and pooling the two would flag all four mono panels forever. Peers are
# therefore compared only within their own type.
BIFACIAL_DEVICES = list(range(11, 27))
MONOFACIAL_DEVICES = [27, 28, 29, 30]

# Wiring map, from the installer's notes. Five blocks of four panels; within a
# block the Modbus id ascends with position. B1-B4 are the 16 bifacial panels
# this project is about; B5 is monofacial and will not be part of the final rig.
#
#   B1: 11 12 13 14      B2: 15 16 17 18      B3: 19 20 21 22
#   B4: 23 24 25 26      B5: 27 28 29 30  (monofacial)
#
# Position matters: P1 and P4 sit at the ends of a block and catch more
# ground-reflected light than the interior P2 and P3. Comparing a P1 against a
# P3 builds that difference into the baseline as though it were noise, which
# hides small faults. Peers are therefore matched on position as well as face.
METER_BLOCKS = {1: [11, 12, 13, 14], 2: [15, 16, 17, 18], 3: [19, 20, 21, 22],
                4: [23, 24, 25, 26], 5: [27, 28, 29, 30]}
EDGE_POSITIONS_METER = (1, 4)

METER_MAP = {}
for _blk, _ids in METER_BLOCKS.items():
    for _pos, _d in enumerate(_ids, start=1):
        _face = "monofacial" if _d in MONOFACIAL_DEVICES else "bifacial"
        METER_MAP[_d] = (_face, _blk, _pos)

ELECTRICAL_FAMILIES = ("V", "I", "P", "E")

MODEL_FAMILIES = [k for k, v in CHANNELS.items() if v["derived"] is None]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def config_of(panel, fam=None):
    """(face, row, position) for a channel.

    Sensors and meters use different numbering -- sensors 1-24, meters 11-30 --
    so the family decides which map to read.
    """
    if fam in ELECTRICAL_FAMILIES:
        return METER_MAP.get(panel, ("bifacial", 0, panel))
    return PANEL_MAP.get(panel, ("rear", 0, 0))


# =============================================================================
#  SOLAR GEOMETRY (NOAA, self-contained)
# =============================================================================

def solar_position(idx, lat_deg=LAT, lon_deg=LON, tz_hours=TZ_HOURS):
    idx = pd.DatetimeIndex(idx)
    jd = (idx - pd.Timedelta(hours=tz_hours)).to_julian_date().values
    jc = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * jc) + np.sin(3 * Mr) * 0.000289)
    app = (L0 + C) - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * jc))
    mo = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obl = mo + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc))
    decl = np.degrees(np.arcsin(np.sin(np.radians(obl)) * np.sin(np.radians(app))))
    y = np.tan(np.radians(obl / 2.0)) ** 2
    L0r = np.radians(L0)
    eqt = 4.0 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr)
                           + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                           - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    mod = idx.hour * 60.0 + idx.minute + idx.second / 60.0
    tst = (mod + eqt + 4.0 * lon_deg - 60.0 * tz_hours) % 1440.0
    ha = np.where(tst / 4.0 < 0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)
    latr, dr, har = np.radians(lat_deg), np.radians(decl), np.radians(ha)
    cosz = np.clip(np.sin(latr) * np.sin(dr) + np.cos(latr) * np.cos(dr) * np.cos(har), -1, 1)
    return pd.DataFrame({"elevation": 90.0 - np.degrees(np.arccos(cosz)),
                         "cos_zenith": cosz, "hour_angle": ha, "declination": decl},
                        index=idx)


def clearsky_ghi(cz):
    cz = np.clip(np.asarray(cz, float), 0.0, 1.0)
    out = np.zeros_like(cz)
    m = cz > 0.01
    out[m] = 1098.0 * cz[m] * np.exp(-0.059 / cz[m])
    return out


# =============================================================================
#  LOADING
# =============================================================================

def discover(columns):
    found = {fam: {} for fam in CHANNELS}
    unmatched = []
    for col in columns:
        c = str(col).strip()
        if c.lower() in ("date", "time", "timestamp", "datetime"):
            continue
        hit = False
        for fam in sorted(CHANNELS, key=lambda f: -len(f)):   # IrrAvg before Irr
            for pat in CHANNELS[fam]["patterns"]:
                m = re.match(pat, c, flags=re.IGNORECASE)
                if m:
                    found[fam][int(m.group(1))] = col
                    hit = True
                    break
            if hit:
                break
        if not hit:
            unmatched.append(col)
    return {f: v for f, v in found.items() if v}, unmatched


LONG_COLUMN_MAP = {
    "Voltage_V": "V", "Current_A": "I",
    "Active_power_kW": "P", "Forward_energy_kWh": "E",
}


def load_long_csv(path, rep):
    """Read a DC-meter CSV (one row per device per sample) and pivot it wide.

    The meters write long format while the irradiance logger writes wide. Rather
    than carry two code paths through the whole pipeline, this reshapes to
    V_11, I_11, P_11, E_11, ... so everything downstream -- peer statistics,
    validation, flags -- is shared.
    """
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    rep["rows_read"] += len(raw)

    ts = pd.to_datetime(raw["Datetime"].str.strip(), errors="coerce", format="mixed")
    ok = ts.notna()
    rep["rows_bad_time"] += int((~ok).sum())
    raw, ts = raw[ok], ts[ok]

    dev = pd.to_numeric(raw["Device_ID"], errors="coerce")
    good = dev.notna()
    raw, ts, dev = raw[good], ts[good], dev[good].astype(int)

    # The meter polls its 20 devices in sequence, so one logical sample is
    # spread over ~1.3 s of wall clock. Left alone, consecutive timestamps look
    # 0.07 s apart, the detected interval collapses, and the regular grid built
    # from it is large enough to exhaust memory. Group each burst into a cycle
    # and stamp every reading in it with the cycle's start time.
    order = np.argsort(ts.values)
    ts_sorted = ts.values[order]
    gaps = np.diff(ts_sorted).astype("timedelta64[ms]").astype(float) / 1000.0
    typical = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 0.1
    cycle_break = max(typical * 5, 1.0)
    new_cycle = np.concatenate([[True], gaps > cycle_break])
    cycle_id = np.cumsum(new_cycle) - 1
    cycle_start = pd.Series(ts_sorted).groupby(cycle_id).transform("first")
    snapped = pd.Series(index=raw.index, dtype="datetime64[ns]")
    snapped.iloc[order] = cycle_start.values
    ts = snapped
    rep["cycles_detected"] = int(cycle_id.max() + 1)

    frames = []
    for src, short in LONG_COLUMN_MAP.items():
        if src not in raw.columns:
            continue
        vals = pd.to_numeric(raw[src], errors="coerce")
        block = pd.DataFrame({"ts": ts.values, "dev": dev.values, "val": vals.values})
        wide = block.pivot_table(index="ts", columns="dev", values="val",
                                 aggfunc="last")
        wide.columns = [f"{short}_{int(c)}" for c in wide.columns]
        frames.append(wide)

    if not frames:
        raise ValueError("no recognised meter columns")
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "Timestamp"
    rep["column_order"] += [c for c in out.columns if c not in rep["column_order"]]
    return out


def _is_long_format(path):
    try:
        cols = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    except Exception:
        return False
    return "Device_ID" in cols and "Datetime" in cols


def load_one(path, rep):
    if _is_long_format(path):
        return load_long_csv(path, rep)
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    rep["rows_read"] += len(raw)
    date_col = next((c for c in raw.columns if c.lower() == "date"), None)
    time_col = next((c for c in raw.columns
                     if c.lower() in ("time", "timestamp", "datetime")), None)
    if time_col is None:
        raise ValueError("no Time column")
    if date_col is not None:
        ok = raw[date_col].str.strip().str.match(DATE_RE, na=False)
        rep["rows_torn"] += int((~ok).sum())     # "4,,,,,," fragments
        raw = raw[ok]
        stamp = raw[date_col].str.strip() + " " + raw[time_col].str.strip()
    else:
        stamp = raw[time_col].str.strip()
    ts = pd.to_datetime(stamp, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    if ts.isna().any():
        ts = ts.fillna(pd.to_datetime(stamp, errors="coerce", format="mixed"))
    good = ts.notna()
    rep["rows_bad_time"] += int((~good).sum())
    raw, ts = raw[good], ts[good]
    found, unmatched = discover(raw.columns)
    if not found:
        raise ValueError("no recognised sensor columns")
    keep = [c for c in raw.columns if c in {v for f in found for v in found[f].values()}]
    rep["column_order"] += [c for c in keep if c not in rep["column_order"]]
    out = raw[keep].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(ts, name="Timestamp")
    return out


def load_all(root):
    rep = {"rows_read": 0, "rows_torn": 0, "rows_bad_time": 0, "rows_duplicate": 0,
           "files_ok": 0, "files_failed": [], "column_order": []}
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)) \
        if os.path.isdir(root) else [root]
    files = [f for f in files
             if not os.path.basename(f).lower().startswith(("filled_", "flags_"))]
    skipped = [f for f in files
               if any(pat.lower() in os.path.basename(f).lower()
                      for pat in EXCLUDE_FILE_PATTERNS)]
    files = [f for f in files if f not in skipped]
    rep["excluded_by_pattern"] = [os.path.basename(f) for f in skipped]
    if skipped:
        print(f"[LOAD] excluding {len(skipped)} file(s) by EXCLUDE_FILE_PATTERNS: "
              f"{', '.join(os.path.basename(f) for f in skipped[:4])}"
              + (" ..." if len(skipped) > 4 else ""))
    rep["files_found"] = len(files)
    print(f"[LOAD] scanning {len(files)} file(s)")
    frames = []
    for fp in files:
        try:
            d = load_one(fp, rep)
        except Exception as exc:
            rep["files_failed"].append([os.path.basename(fp), repr(exc)])
            print(f"  [SKIP] {os.path.basename(fp)}: {exc}")
            continue
        if not d.empty:
            rep["files_ok"] += 1
            frames.append(d)
    if not frames:
        raise ValueError("nothing loaded")
    wide = pd.concat(frames).sort_index()
    dup = wide.index.duplicated()
    rep["rows_duplicate"] = int(dup.sum())
    wide = wide[~dup]
    found, _ = discover(wide.columns)
    ids = sorted({p for f in found for p in found[f]})
    print(f"[LOAD] {rep['files_ok']}/{rep['files_found']} files, {len(wide):,} rows, "
          f"{len(ids)} channels, families {sorted(found)}")
    print(f"       dropped {rep['rows_torn']:,} torn rows, "
          f"{rep['rows_bad_time']:,} bad timestamps, {rep['rows_duplicate']:,} duplicates")
    return wide, found, rep


# =============================================================================
#  CADENCE  (Temp / IrrAvg are written every 11th sample, not every sample)
# =============================================================================

def detect_interval(index):
    d = pd.Series(index).diff().dt.total_seconds().dropna()
    d = d[(d > 0) & (d < 3600)]
    return float(d.round(2).mode().iloc[0]) if len(d) else 5.0


def detect_cadence(series):
    pos = np.where(series.notna().values)[0]
    if len(pos) < 5:
        return 1
    d = np.diff(pos)
    d = d[d > 0]
    if not len(d):
        return 1
    vals, counts = np.unique(d, return_counts=True)
    return max(1, int(vals[np.argmax(counts)]))


def scheduled_mask(series, period):
    """True where a write was expected. Phase is re-derived after every gap,
    because the Pi counts samples rather than clock time: after an outage the
    writes resume on a new offset."""
    n = len(series)
    if period <= 1:
        return np.ones(n, bool)
    pos = np.where(series.notna().values)[0]
    mask = np.zeros(n, bool)
    if not len(pos):
        return mask
    mask[pos] = True
    for a, b in zip(pos[:-1], pos[1:]):
        if b - a > period:
            k = a + period
            while k < b:
                mask[k] = True
                k += period
    k = pos[-1] + period
    while k < n:
        mask[k] = True
        k += period
    return mask


def to_grid(wide, interval_s):
    freq = pd.Timedelta(seconds=interval_s)
    span = (wide.index.max() - wide.index.min()).total_seconds()
    if span / max(interval_s, 1e-6) > 5_000_000:
        raise ValueError(
            f"A {interval_s:g}s grid over this span would run to millions of "
            f"rows. The detected sampling interval is almost certainly wrong; "
            f"check the timestamps before rerunning.")
    # Use the index resample() produces rather than building a separate
    # date_range and reindexing onto it. On a fractional interval -- the meters
    # poll every 11.37 s -- floor() and resample() land on different boundaries,
    # the reindex matches nothing, and every column comes back 100% missing.
    g = wide.resample(freq).mean()
    g.index.name = "Timestamp"
    return g


# =============================================================================
#  FAULT MARKING  (detection only — values are never modified)
# =============================================================================

def mark_faults(grid, found):
    g = grid.copy()
    rep, prov = {}, {}
    for fam, cols in found.items():
        for c in cols.values():
            prov[c] = np.where(grid[c].notna().values, "measured", "empty").astype(object)

    n = 0
    for fam, cols in found.items():
        for c in cols.values():
            m = g[c].isin(SENTINELS).values
            n += int(m.sum())
            prov[c][m] = "faultcode"
            if not KEEP_FAULTCODES:
                g.loc[m, c] = np.nan
    rep["faultcode"] = n

    n, per_ch = 0, {}
    for fam, cols in found.items():
        lo, hi = CHANNELS[fam]["lo"], CHANNELS[fam]["hi"]
        for c in cols.values():
            m = ((g[c] < lo) | (g[c] > hi)).values
            k = int(m.sum())
            if k:
                per_ch[c] = k
            n += k
            prov[c][m] = "out_of_range"
            if not KEEP_OUT_OF_RANGE:
                g.loc[m, c] = np.nan
    rep["out_of_range"] = n
    rep["out_of_range_by_channel"] = per_ch
    kept = "KEPT in output, excluded from training"
    print(f"[FAULT] fault codes {kept}: {rep['faultcode']:,}   "
          f"out-of-range {kept}: {rep['out_of_range']:,}")
    return g, rep, prov


# =============================================================================
#  DATA VALIDATION  (which files / columns may be used for training)
# =============================================================================

def validate_data(grid, found, prov, interval_s):
    """Decide what is fit to train on, per day and per column.

    A day with almost no usable data teaches nothing and is dropped whole. A day
    with one bad sensor is still valuable -- only that column is dropped, and
    the other nineteen still contribute.
    """
    day = grid.index.normalize()
    days = sorted(pd.unique(day))
    per_day, usable_days, dropped_days = {}, [], []
    blocked = {}          # column -> boolean mask of rows barred from training

    for fam, cols in found.items():
        for c in cols.values():
            blocked[c] = np.zeros(len(grid), bool)

    rows = []
    for d in days:
        m = (day == d)
        n_rows = int(m.sum())
        day_cols, bad_cols = 0, []
        for fam, cols in found.items():
            for p, c in cols.items():
                pv = prov[c][m]
                good = int((pv == "measured").sum())
                fault = int(((pv == "out_of_range") | (pv == "faultcode")).sum())
                frac_fault = fault / max(good + fault, 1)
                # A low-variability column is NOT excluded: a sensor can legitimately
                # sit near-constant (overcast day, shaded panel), and treating that
                # as a defect would throw away real measurements.
                if frac_fault > MAX_COLUMN_FAULT_FRAC:
                    blocked[c][m] = True
                    bad_cols.append(f"{c}({frac_fault:.0%} faults)")
                else:
                    day_cols += 1
                    # fault rows are never trainable even in an accepted column
                    blocked[c][np.where(m)[0][(pv == "out_of_range") | (pv == "faultcode")]] = True
        usable_h = 0.0
        allc = [c for fam in MODEL_FAMILIES for c in found.get(fam, {}).values()]
        anygood = np.zeros(n_rows, bool)
        for c in allc:
            anygood |= (prov[c][m] == "measured")
        usable_h = anygood.sum() * interval_s / 3600.0
        ok = usable_h >= MIN_USABLE_HOURS and day_cols > 0
        (usable_days if ok else dropped_days).append(d)
        rows.append({"date": str(pd.Timestamp(d).date()), "usable_hours": round(usable_h, 1),
                     "good_columns": day_cols, "excluded_columns": len(bad_cols),
                     "train": "yes" if ok else "NO",
                     "notes": ", ".join(bad_cols[:3])})
        per_day[str(pd.Timestamp(d).date())] = rows[-1]

    tab = pd.DataFrame(rows)
    print(f"\n[VALIDATE] {len(usable_days)} day(s) usable for training, "
          f"{len(dropped_days)} dropped (< {MIN_USABLE_HOURS} h of data)")
    bad = tab[(tab.train == "NO") | (tab.excluded_columns > 0)]
    if len(bad):
        print(bad.head(15).to_string(index=False))
    else:
        print("           every day passed with all columns usable")
    return usable_days, dropped_days, blocked, tab


def split_days(usable_days, test_fraction=TEST_FRACTION):
    """70/30 by DAY. A random row split would leak: 14:30:00 in training and
    14:30:30 in test means the model reads the answer off the adjacent row."""
    rng = np.random.default_rng(RANDOM_STATE)
    d = np.array(sorted(usable_days))
    idx = rng.permutation(len(d))
    n_test = int(round(len(d) * test_fraction))
    n_test = min(max(n_test, 1 if len(d) > 1 else 0), max(len(d) - 1, 0))
    test = set(pd.Timestamp(x) for x in d[idx[:n_test]])
    train = set(pd.Timestamp(x) for x in d[idx[n_test:]])
    if not train:
        # Never leave the training set empty. With a single day there is nothing
        # to hold out, so train on it and say so rather than reporting the
        # misleading "not enough data to train".
        train = set(pd.Timestamp(x) for x in d)
        test = set()
        print(f"[SPLIT] only {len(d)} day(s) available: training on all of them, "
              f"no held-out accuracy can be measured")
    else:
        print(f"[SPLIT] train {len(train)} day(s) / test {len(test)} day(s)")
    return train, test


# =============================================================================
#  LAYOUT CROSS-CHECK
# =============================================================================

def resolve_faces(grid, found, feats):
    """Override the declared face with the measured one, when asked.

    Front meters read several times what rear ones do, so the split is taken at
    the largest gap in log-ratio space rather than a fixed threshold. Array and
    position still come from PANEL_MAP -- only the face is inferred.
    """
    if not AUTO_FACE_FROM_DATA:
        return
    cols = found.get("Irr", {})
    if len(cols) < 2:
        return
    bright = feats["ghi_cs"].values > 200
    med = grid.loc[bright, list(cols.values())].median()
    base = float(np.nanmedian(med.values))
    if not np.isfinite(base) or base <= 0:
        return
    ratio = {p: float(med[cols[p]] / base) for p in cols if np.isfinite(med[cols[p]])}
    order = sorted(ratio, key=lambda p: ratio[p])
    lv = np.log([max(ratio[p], 1e-3) for p in order])
    gaps = np.diff(lv)
    changed = []
    if len(gaps) and np.exp(gaps.max()) >= 1.8:
        cut = int(np.argmax(gaps))
        front = set(order[cut + 1:])
    else:
        front = set()
    for p in cols:
        face = "front" if p in front else "rear"
        old, arr, pos = config_of(p)
        if face != old:
            changed.append(f"ch{p}: {old} -> {face}")
        PANEL_MAP[p] = (face, 0 if face == "front" else arr,
                        0 if face == "front" else pos)
    print(f"[FACE ] inferred from data: {len(front)} front / "
          f"{len(cols) - len(front)} rear")
    if changed:
        print("        overriding PANEL_MAP: " + ", ".join(changed[:8]))


def verify_layout(grid, found, feats):
    """Compare the declared PANEL_MAP against what the sensors actually read.

    A front meter reads several times what a rear one does, so a mislabelled or
    miswired channel shows up immediately. Reported, never auto-corrected --
    silently overriding the map would hide a wiring fault.
    """
    cols = found.get("Irr", {})
    if len(cols) < 2:
        return {}
    bright = feats["ghi_cs"].values > 200
    med = grid.loc[bright, list(cols.values())].median()
    rear = [p for p in cols if config_of(p)[0] == "rear"]
    base = float(np.nanmedian([med[cols[p]] for p in rear])) if rear else np.nan
    if not np.isfinite(base) or base <= 0:
        return {}
    report, warn = {}, []
    for p, c in cols.items():
        r = float(med[c] / base) if np.isfinite(med[c]) else np.nan
        face, arr, pos = config_of(p)
        report[str(p)] = {"declared": face, "row": arr, "position": pos,
                          "ratio_to_rear_median": round(r, 2)}
        if face == "rear" and r > 2.0:
            warn.append(f"ch{p} declared rear but reads {r:.1f}x rear median")
        if face == "front" and r < 1.5:
            warn.append(f"ch{p} declared front but reads only {r:.1f}x rear median")
    print(f"[LAYOUT] {sum(1 for p in cols if config_of(p)[0]=='front')} front / "
          f"{sum(1 for p in cols if config_of(p)[0]=='rear')} rear declared")
    for w in warn:
        print(f"         [WARN] {w}")
    if not warn:
        print("         measured ratios agree with PANEL_MAP")
    return report


# =============================================================================
#  FEATURES
# =============================================================================

def solar_features(index):
    sp = solar_position(index)
    f = pd.DataFrame(index=index)
    f["elevation"] = sp["elevation"].values
    f["cos_zenith"] = np.clip(sp["cos_zenith"].values, 0, None)
    f["hour_angle"] = sp["hour_angle"].values
    f["declination"] = sp["declination"].values
    f["ghi_cs"] = clearsky_ghi(sp["cos_zenith"].values)
    f["airmass"] = 1.0 / np.clip(sp["cos_zenith"].values, 0.05, None)
    h = index.hour + index.minute / 60 + index.second / 3600
    f["sin_hour"], f["cos_hour"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    doy = index.dayofyear
    f["sin_doy"], f["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["is_night"] = (sp["elevation"].values < NIGHT_ELEV_DEG).astype(np.float32)
    return f


def _days_mask(index, days):
    """Boolean mask of rows whose date is in `days`, compared as int64 ns."""
    if not len(days):
        return np.zeros(len(index), bool)
    # Force BOTH sides through datetime64[ns] before comparing. Timestamp.value
    # is nanoseconds while DatetimeIndex.asi8 follows the index's own unit, so a
    # microsecond-resolution index (which is what mixed timestamp formats across
    # months produce) compared against nanosecond Timestamps matches nothing and
    # silently empties the training set.
    day_i = pd.DatetimeIndex(index).normalize().to_numpy("datetime64[ns]").astype("int64")
    want = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in days)) \
        .normalize().to_numpy("datetime64[ns]").astype("int64")
    return np.isin(day_i, want)


def loo_stats(mat):
    """Leave-one-out summaries. Including a panel's own value in the summary
    would leak the training target into its own feature."""
    n, k = mat.shape
    out = [np.full((n, k), np.nan, np.float32) for _ in range(3)]
    cnt = np.zeros((n, k), np.float32)
    for j in range(k):
        rest = np.delete(mat, j, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[0][:, j] = np.nanmean(rest, axis=1)
            out[1][:, j] = np.nanmedian(rest, axis=1)
            out[2][:, j] = np.nanstd(rest, axis=1)
        cnt[:, j] = np.sum(~np.isnan(rest), axis=1)
    return out[0], out[1], out[2], cnt


def build_long(grid, found, feats, fam, blocked=None, train_days=None):
    """Wide -> long: one row per (timestamp, panel), with configuration labels.

    Pooling is what lets one model serve 8 panels or 24. The configuration
    features are what stop it averaging a front meter, an edge rear panel and
    an interior rear panel into a single meaningless relationship.
    """
    cols = found.get(fam, {})
    if not cols:
        return None
    panels = sorted(cols)
    n, k = len(grid), len(panels)
    mat = grid[[cols[p] for p in panels]].to_numpy(np.float32)

    cfg = [config_of(p, fam) for p in panels]
    face = np.array([1.0 if c[0] in ("front", "monofacial") else 0.0 for c in cfg], np.float32)
    arr = np.array([c[1] for c in cfg], np.float32)
    pos = np.array([c[2] for c in cfg], np.float32)
    edge = np.array([1.0 if c[2] in (1, 4) else 0.0 for c in cfg], np.float32)

    # peers within the same face, and within the same face+position across
    # arrays (position 1 on array A behaves like position 1 on array B)
    grp = np.full((n, k), np.nan, np.float32)
    gmed = np.full((n, k), np.nan, np.float32)
    gstd = np.full((n, k), np.nan, np.float32)
    gcnt = np.zeros((n, k), np.float32)
    for fv in np.unique(face):
        idx = np.where(face == fv)[0]
        if len(idx) > 1:
            a, b, c_, d = loo_stats(mat[:, idx])
            grp[:, idx], gmed[:, idx], gstd[:, idx], gcnt[:, idx] = a, b, c_, d

    pmed = np.full((n, k), np.nan, np.float32)
    pcnt = np.zeros((n, k), np.float32)
    for pv in np.unique(pos):
        idx = np.where((pos == pv) & (face == 0.0))[0]
        if len(idx) > 1:
            _, b, _, d = loo_stats(mat[:, idx])
            pmed[:, idx], pcnt[:, idx] = b, d

    amean, amed, astd, acnt = loo_stats(mat)

    # Two front references, because they answer different questions.
    #   site_front_ref : mean of ALL front sensors -- what the sky is doing
    #   row_front_ref  : mean of THIS row's own Front A and Front B -- what the
    #                    sky is doing directly over these panels. A row's own
    #                    pair sees the same cloud as its rear sensors, so it is
    #                    the stronger predictor; the site mean is the fallback
    #                    when a row's own front pair is down.
    fidx = np.where(face == 1.0)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site = (np.nanmean(mat[:, fidx], axis=1) if len(fidx)
                else np.full(n, np.nan, np.float32))
    site = np.repeat(site[:, None], k, axis=1).astype(np.float32)

    rowref = np.full((n, k), np.nan, np.float32)
    rowspan = np.full((n, k), np.nan, np.float32)
    rowmed = np.full((n, k), np.nan, np.float32)
    rowcnt = np.zeros((n, k), np.float32)
    for rv in np.unique(arr):
        same_row = np.where(arr == rv)[0]
        fr = sorted(j for j in same_row if face[j] == 1.0)
        if fr:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rr = np.nanmean(mat[:, fr], axis=1)
            rowref[:, same_row] = rr[:, None]
            if len(fr) >= 2:
                # Front B minus Front A: a non-zero span means the cloud edge is
                # partway across this row, which is exactly when the rear panels
                # at each end diverge. Zero span means uniform sky.
                rowspan[:, same_row] = (mat[:, fr[-1]] - mat[:, fr[0]])[:, None]
        rr_rear = [j for j in same_row if face[j] == 0.0]
        if len(rr_rear) > 1:
            _, b_, _, d_ = loo_stats(mat[:, rr_rear])
            rowmed[:, rr_rear], rowcnt[:, rr_rear] = b_, d_

    lags = {}
    for L in LAG_STEPS:
        sh = np.full_like(gmed, np.nan)
        if L > 0:
            sh[L:, :] = gmed[:-L, :]
        elif L < 0:
            sh[:L, :] = gmed[-L:, :]
        lags[L] = sh

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = np.nanmedian(np.where(gmed > 20, mat / np.where(gmed > 20, gmed, np.nan),
                                    np.nan), axis=0)
    cal = np.where(np.isfinite(cal), cal, 1.0).astype(np.float32)

    F = feats.to_numpy(np.float32)
    parts, names = [], []

    def add(block, name):
        parts.append(block.reshape(-1, 1))
        names.append(name)

    add(mat, "__target__")
    add(np.repeat(face[None, :], n, 0), "face_front")
    add(np.repeat(arr[None, :], n, 0), "row_id")
    add(np.repeat(pos[None, :], n, 0), "position")
    add(np.repeat(edge[None, :], n, 0), "is_edge")
    add(np.repeat(cal[None, :], n, 0), "panel_cal")
    add(grp, "face_mean"); add(gmed, "face_median"); add(gstd, "face_std"); add(gcnt, "face_count")
    add(pmed, "pos_median"); add(pcnt, "pos_count")
    add(rowmed, "row_median"); add(rowcnt, "row_count")
    add(amean, "all_mean"); add(amed, "all_median"); add(astd, "all_std"); add(acnt, "all_count")
    add(site, "site_front_ref")
    add(rowref, "row_front_ref")
    add(rowspan, "row_front_span")
    for L in LAG_STEPS:
        add(lags[L], f"face_median_lag{L}")
    for i, nm in enumerate(feats.columns):
        add(np.repeat(F[:, i][:, None], k, 1), nm)

    X = np.hstack(parts)
    t = names.index("__target__")
    y = X[:, t].copy()
    X = np.delete(X, t, axis=1)
    names.pop(t)

    trainable = np.ones(X.shape[0], bool)
    if blocked is not None:
        bad = np.zeros((n, k), bool)
        for j, p in enumerate(panels):
            bad[:, j] = blocked.get(cols[p], np.zeros(n, bool))
        trainable &= ~bad.reshape(-1)
    if train_days is not None:
        # Compare as int64 nanoseconds. np.isin() on a DatetimeIndex against a
        # list of Timestamps is version-sensitive and silently returns all-False
        # on some numpy/pandas combinations, which zeroes the training set and
        # reports "not enough data to train" on a dataset with 1.4M usable rows.
        inday = _days_mask(grid.index, train_days)
        trainable &= np.repeat(inday, k)

    return {"X": X, "y": y, "names": names, "panels": panels, "cols": cols,
            "n_time": n, "k": k, "trainable": trainable}


# =============================================================================
#  MODEL
# =============================================================================

def _hgb():
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=MAX_ITER, learning_rate=LEARNING_RATE,
        max_leaf_nodes=MAX_LEAF_NODES, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=RANDOM_STATE)


def fit_family(long):
    X, y = long["X"], long["y"]
    ok = np.isfinite(y) & long["trainable"] & np.isfinite(X).any(axis=1)
    idx = np.where(ok)[0]
    if len(idx) < 500:
        return None
    if len(idx) > MAX_TRAIN_ROWS:
        idx = np.sort(np.random.default_rng(RANDOM_STATE).choice(idx, MAX_TRAIN_ROWS, False))
    return _hgb().fit(X[idx], y[idx])


def runs_of_true(mask):
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def real_gaps(grid, found):
    gapmask, info = {}, {}
    for fam, cols in found.items():
        for p, c in cols.items():
            cad = detect_cadence(grid[c])
            sched = scheduled_mask(grid[c], cad)
            gapmask[c] = grid[c].isna().values & sched
            info[c] = {"family": fam, "cadence": cad,
                       "scheduled": int(sched.sum()), "gaps": int(gapmask[c].sum())}
    return gapmask, info


def model_fill(grid, found, fam, feats, gapmask, model, blocked=None):
    long = build_long(grid, found, feats, fam, blocked)
    if long is None or model is None:
        return {c: np.full(len(grid), np.nan) for c in found.get(fam, {}).values()}, long
    names, k, panels, cols = long["names"], long["k"], long["panels"], long["cols"]
    peers = np.zeros(long["X"].shape[0])
    for nm in ("row_count", "face_count", "pos_count", "all_count"):
        if nm in names:
            peers = np.maximum(peers, np.nan_to_num(long["X"][:, names.index(nm)]))
    want = np.zeros(long["X"].shape[0], bool)
    for j, p in enumerate(panels):
        want[j::k] = gapmask[cols[p]]
    rows = want & (peers >= MIN_PEERS_TO_PREDICT) & np.isfinite(long["X"]).any(axis=1)
    flat = np.full(long["X"].shape[0], np.nan)
    if rows.any():
        flat[rows] = model.predict(long["X"][rows])
    return {cols[p]: flat[j::k] for j, p in enumerate(panels)}, long


def interp_fill(grid, found, fam):
    return {c: grid[c].interpolate(method="time", limit_area="inside").values
            for c in found.get(fam, {}).values()}


def combine(grid, found, fam, interp, modelled, gapmask, interval_s, decisions=None):
    out = grid.copy()
    max_pts = max(1, int(MAX_FILL_GAP_HOURS * 3600 / interval_s))
    split_pts = max(1, int(BUCKET_SPLIT_MINUTES * 60 / interval_s))
    dec = (decisions or {}).get(fam, {"short": "interp", "long": "model"})
    filled, tally = {}, {"model": 0, "interp": 0, "left_empty": 0}
    for p, c in found.get(fam, {}).items():
        v = out[c].to_numpy(np.float64, copy=True)
        was = np.zeros(len(v), bool)
        for a, b in runs_of_true(gapmask[c]):
            need = np.ones(b - a, bool)
            bucket = "short" if (b - a) <= split_pts else "long"
            # the held-out days decided this, per family and per gap length
            prefer_model = dec.get(bucket, "model") == "model"
            seg = (np.asarray(modelled.get(c))[a:b] if prefer_model
                   else np.full(b - a, np.nan))
            take = need & np.isfinite(seg)
            if take.any():
                blk = v[a:b]; blk[take] = seg[take]; v[a:b] = blk
                wb = was[a:b]; wb[take] = True; was[a:b] = wb
                tally["model"] += int(take.sum()); need &= ~take
            if need.any():
                iseg = np.asarray(interp.get(c))[a:b]
                if not prefer_model:
                    pass   # interpolation is the chosen method for this bucket
                itake = need & np.isfinite(iseg)
                # all-or-nothing: a long stretch is left whole, not part-filled
                for ha, hb in runs_of_true(need):
                    if (hb - ha) > max_pts:
                        itake[ha:hb] = False
                if itake.any():
                    blk = v[a:b]; blk[itake] = iseg[itake]; v[a:b] = blk
                    wb = was[a:b]; wb[itake] = True; was[a:b] = wb
                    tally["interp"] += int(itake.sum()); need &= ~itake
            if need.any() and not prefer_model:
                mseg = np.asarray(modelled.get(c))[a:b]      # model as fallback
                mtake = need & np.isfinite(mseg)
                if mtake.any():
                    blk = v[a:b]; blk[mtake] = mseg[mtake]; v[a:b] = blk
                    wb = was[a:b]; wb[mtake] = True; was[a:b] = wb
                    tally["model"] += int(mtake.sum()); need &= ~mtake
            tally["left_empty"] += int(need.sum())
        out[c] = v
        filled[c] = was
    print(f"  [{fam}] model={tally['model']:,}  interp={tally['interp']:,}  "
          f"left empty={tally['left_empty']:,}")
    return out, filled, tally


def constrain_filled(grid, found, feats, filled):
    """Physical limits on generated values only. A measured reading is never
    touched, however implausible it looks."""
    if not filled:
        return grid
    night = feats["is_night"].values > 0.5
    cap = feats["ghi_cs"].values * 1.35 + 25.0
    for fam, cols in found.items():
        spec = CHANNELS[fam]
        for c in cols.values():
            fm = filled.get(c)
            if fm is None or not fm.any():
                continue
            v = grid[c].to_numpy(np.float64, copy=True)
            sel = fm & np.isfinite(v)
            x = np.clip(v[sel], spec["lo"], spec["hi"])
            if spec["night"] == "zero":
                x = np.minimum(x, cap[sel])
                x = np.where(night[sel], np.clip(x, 0.0, 2.0), np.clip(x, 0.0, None))
            v[sel] = x
            grid[c] = v
    return grid


# =============================================================================
#  ACCURACY ON HELD-OUT DAYS
# =============================================================================

def evaluate(grid, found, feats, models, test_days, interval_s, blocked):
    """Cut synthetic gaps into the held-out days and measure the error.

    Gaps are contiguous blocks in daylight, placed per panel, matching the way
    the logger actually fails. Test days took no part in fitting the model.
    """
    if not test_days:
        print("[ACCURACY] no held-out days, so no accuracy can be reported. "
              "Fills use interpolation for short gaps and the model for long ones.")
        return None, {}
    in_test = _days_mask(grid.index, test_days)
    daylight = feats["is_night"].values < 0.5
    rng = np.random.default_rng(RANDOM_STATE)
    results = []

    for fam in MODEL_FAMILIES:
        cols = found.get(fam, {})
        if not cols or models.get(fam) is None:
            continue
        # One gap at a time, each against otherwise-intact data.
        #
        # Punching every test gap into ONE shared copy made gaps overlap in
        # time, so a channel being reconstructed often had its peers removed
        # too. Measured on this data: 38.5% of test rows ended up with peers
        # artificially missing, and they contributed 58.5% of all error --
        # rows with 7 live peers scored 18.8% while rows with 5 scored 53.8%.
        # That is not how the rig fails. One sensor drops, the rest keep
        # logging, so that is what the test must reproduce.
        truth = {}
        base_cad = max(1, int(np.median([detect_cadence(grid[c]) for c in cols.values()])))
        for bucket, opts in (("short", SHORT_BUCKET_MINUTES), ("long", LONG_BUCKET_MINUTES)):
            for _ in range(GAPS_PER_FAMILY_PER_BUCKET):
                p = list(cols)[rng.integers(len(cols))]
                c = cols[p]
                ok = np.where(grid[c].notna().values & in_test & daylight
                              & ~blocked.get(c, np.zeros(len(grid), bool)))[0]
                if len(ok) < 50:
                    continue
                L = max(base_cad * 3, int(opts[rng.integers(len(opts))] * 60 / interval_s))
                for _t in range(20):
                    s_ = int(ok[rng.integers(len(ok))])
                    if s_ + L >= len(grid) or not in_test[s_:s_ + L].all():
                        continue
                    if not daylight[s_:s_ + L].all():
                        continue
                    seg = grid[c].iloc[s_:s_ + L]
                    if seg.notna().sum() < 3:
                        continue
                    truth.setdefault(bucket, []).append((c, s_, s_ + L, seg.copy()))
                    break

        if not truth:
            continue
        # Model predictions come from features built on the INTACT grid. That is
        # not leakage: every feature the model reads is leave-one-out across
        # peers or derived from the group median, so a panel's own values never
        # enter its own prediction. Blanking them would change nothing except by
        # damaging OTHER panels that happen to share the window.
        # Interpolation is different -- it reads the target's own neighbours --
        # so each gap really is blanked for it, one column at a time.
        long_full = build_long(grid, found, feats, fam, blocked)
        mdl = models[fam]
        names, kk, pnl = long_full["names"], long_full["k"], long_full["panels"]
        peers_ok = np.zeros(long_full["X"].shape[0])
        for nm in ("row_count", "face_count", "pos_count", "all_count"):
            if nm in names:
                peers_ok = np.maximum(peers_ok, np.nan_to_num(long_full["X"][:, names.index(nm)]))

        for bucket, items in truth.items():
            yt, ym, yi = [], [], []
            per_gap = []
            for c, a, b, seg in items:
                p_ = next(q for q, cc in cols.items() if cc == c)
                j = pnl.index(p_)
                ridx = np.arange(a, b) * kk + j
                mm = np.full(b - a, np.nan)
                usable = peers_ok[ridx] >= MIN_PEERS_TO_PREDICT
                if usable.any():
                    mm[usable] = mdl.predict(long_full["X"][ridx[usable]])
                col = grid[c].copy()
                col.iloc[a:b] = np.nan
                ii = col.interpolate(method="time", limit_area="inside").values[a:b]
                t = seg.values
                yt.append(t); ym.append(mm); yi.append(ii)
                per_gap.append((t, mm, ii))
            yt, ym, yi = map(np.concatenate, (yt, ym, yi))
            okm = np.isfinite(yt) & np.isfinite(ym)
            oki = np.isfinite(yt) & np.isfinite(yi)
            # Percentages are normalised by the MEAN of the true values, not
            # computed per point. A plain MAPE divides by each reading, and
            # irradiance passes through near-zero at dawn and dusk, so a 2 W/m2
            # error on a 0.5 W/m2 reading becomes 400% and swamps the average.
            # Normalising by the mean is the convention in PV performance work.
            base = float(np.mean(np.abs(yt[np.isfinite(yt)]))) if np.isfinite(yt).any() else np.nan
            n_truth = int(np.isfinite(yt).sum())
            row = {"family": fam, "gap": bucket, "n_points": int(okm.sum() or oki.sum()),
                   "mean_actual": base,
                   # what share of the test gap the method could reach at all.
                   # Error is only meaningful alongside this: a method that
                   # answers 10% of cases and gets them right is not accurate,
                   # it is selective.
                   "answered%_model": 100.0 * okm.sum() / max(n_truth, 1),
                   "answered%_interp": 100.0 * oki.sum() / max(n_truth, 1)}
            if okm.sum() >= 20:
                mae = mean_absolute_error(yt[okm], ym[okm])
                rmse = float(np.sqrt(mean_squared_error(yt[okm], ym[okm])))
                # normalise over the SAME points that were scored, not over all
                # truth, or a method answering only the easy points flatters itself
                b = float(np.mean(np.abs(yt[okm])))
                row["MAE_model"] = mae
                row["RMSE_model"] = rmse
                row["MAE%_model"] = 100.0 * mae / b if b else np.nan
                row["RMSE%_model"] = 100.0 * rmse / b if b else np.nan
            if oki.sum() >= 20:
                mae = mean_absolute_error(yt[oki], yi[oki])
                rmse = float(np.sqrt(mean_squared_error(yt[oki], yi[oki])))
                b = float(np.mean(np.abs(yt[oki])))
                row["MAE_interp"] = mae
                row["RMSE_interp"] = rmse
                row["MAE%_interp"] = 100.0 * mae / b if b else np.nan
                row["RMSE%_interp"] = 100.0 * rmse / b if b else np.nan
            # The error is dominated by a few cloud-transition gaps, so a single
            # pooled number is a noisy draw: reshuffling the gaps moves it by
            # 10+ percentage points. Bootstrapping over WHOLE gaps gives an
            # honest interval instead of a falsely precise point estimate.
            brng = np.random.default_rng(RANDOM_STATE)
            for key, pick in (("model", 1), ("interp", 2)):
                vals = []
                for _ in range(300):
                    sel = brng.integers(0, len(per_gap), len(per_gap))
                    num = den = 0.0
                    for gi in sel:
                        t, pr = per_gap[gi][0], per_gap[gi][pick]
                        o = np.isfinite(t) & np.isfinite(pr)
                        if o.any():
                            num += np.abs(t[o] - pr[o]).sum()
                            den += np.abs(t[o]).sum()
                    if den > 0:
                        vals.append(100.0 * num / den)
                if vals:
                    row[f"MAE%_{key}_lo"] = float(np.percentile(vals, 5))
                    row[f"MAE%_{key}_hi"] = float(np.percentile(vals, 95))
            row["n_gaps"] = len(per_gap)
            results.append(row)
    if not results:
        return None, {}
    res = pd.DataFrame(results)
    dec = {}
    for _, r in res.iterrows():
        mm = r.get("MAE_model", np.inf)
        mi = r.get("MAE_interp", np.inf)
        mm = mm if np.isfinite(mm) else np.inf
        mi = mi if np.isfinite(mi) else np.inf
        dec.setdefault(r["family"], {})[r["gap"]] = "model" if mm < mi else "interp"
    res["chosen"] = [dec[r["family"]][r["gap"]] for _, r in res.iterrows()]
    res["best_MAE%"] = [r.get(f"MAE%_{'model' if r['chosen'] == 'model' else 'interp'}", np.nan)
                        for _, r in res.iterrows()]
    print("\n[ACCURACY] measured on held-out test days (never used for training)")
    print("           % is error / mean of the true values over the same points")
    show = ["family", "gap", "n_points", "mean_actual",
            "MAE_model", "MAE%_model", "MAE_interp", "MAE%_interp",
            "chosen", "best_MAE%"]
    show = [c for c in show if c in res.columns]
    print(res[show].round(2).to_string(index=False))
    for _, r in res.iterrows():
        unit = "W/m2" if r["family"] in ("Irr", "IrrAvg") else "C"
        ch = r["chosen"]
        lo, hi = r.get(f"MAE%_{ch}_lo", np.nan), r.get(f"MAE%_{ch}_hi", np.nan)
        val = r.get("MAE_model" if ch == "model" else "MAE_interp", np.nan)
        print(f"           {r['family']:7s} {r['gap']:5s} gaps: filled by {ch:6s} "
              f"with {r['best_MAE%']:.1f}% error "
              f"(90% CI {lo:.1f}-{hi:.1f}%, {val:.2f} {unit}, {int(r['n_gaps'])} test gaps)")
    return res, dec


# =============================================================================
#  OUTPUT
# =============================================================================

def write_outputs(grid, found, prov, filled, out_dir, summary, column_order):
    os.makedirs(out_dir, exist_ok=True)
    if column_order:
        order = [c for c in column_order if c in grid.columns]
        grid = grid[order + [c for c in grid.columns if c not in order]]

    flags = pd.DataFrame(index=grid.index)
    UNFILLED = {"empty": "empty_gap",
                "faultcode": "kept_faultcode" if KEEP_FAULTCODES else "empty_faultcode",
                "out_of_range": "kept_out_of_range" if KEEP_OUT_OF_RANGE else "empty_out_of_range"}
    FILLED = {"empty": "filled_gap", "faultcode": "filled_faultcode",
              "out_of_range": "filled_out_of_range"}
    for fam, cols in found.items():
        for p, c in cols.items():
            was = np.asarray(prov[c], dtype=object)
            f = np.empty(len(grid), dtype=object)
            f[:] = "measured"
            fm = filled.get(c, np.zeros(len(grid), bool))
            for key in UNFILLED:
                sel = (was == key)
                f[sel & fm] = FILLED[key]
                f[sel & ~fm] = UNFILLED[key]
            sched = scheduled_mask(grid[c], detect_cadence(grid[c]))
            f[~sched & (was == "empty") & ~fm] = "not_scheduled"
            flags[c] = f

    paths = []
    if PER_DAY_OUTPUT:
        for d, chunk in grid.groupby(grid.index.normalize()):
            o = chunk.round(FLOAT_DECIMALS).copy()
            o.insert(0, "Time", chunk.index.strftime(TIME_FORMAT))
            o.insert(0, "Date", chunk.index.strftime("%Y-%m-%d"))
            fp = os.path.join(out_dir, f"filled_{d.date()}.csv")
            o.to_csv(fp, index=False)
            paths.append(fp)
            fc = flags.loc[chunk.index].copy()
            fc.insert(0, "Time", chunk.index.strftime(TIME_FORMAT))
            fc.insert(0, "Date", chunk.index.strftime("%Y-%m-%d"))
            fc.to_csv(os.path.join(out_dir, f"flags_{d.date()}.csv"), index=False)
    else:
        fp = os.path.join(out_dir, "filled_all.csv")
        grid.round(FLOAT_DECIMALS).to_csv(fp)
        flags.to_csv(os.path.join(out_dir, "flags_all.csv"))
        paths.append(fp)
    with open(os.path.join(out_dir, "run_report.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return paths, flags


# =============================================================================
#  MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print("  Bifacial PV Logger — Gap Filling with Panel Configuration")
    print("=" * 78)

    wide, found, load_rep = load_all(DATA_ROOT)
    interval_s = detect_interval(wide.index)
    print(f"[GRID] sampling interval: {interval_s:g} s")
    grid = to_grid(wide, interval_s)
    feats = solar_features(grid.index)
    grid, fault_rep, prov = mark_faults(grid, found)

    resolve_faces(grid, found, feats)
    layout = verify_layout(grid, found, feats) if VERIFY_LAYOUT else {}
    usable_days, dropped_days, blocked, val_table = validate_data(grid, found, prov, interval_s)
    train_days, test_days = split_days(usable_days)

    gapmask, cad_info = real_gaps(grid, found)
    for fam in found:
        cads = sorted({cad_info[c]["cadence"] for c in found[fam].values()})
        g = sum(cad_info[c]["gaps"] for c in found[fam].values())
        sch = sum(cad_info[c]["scheduled"] for c in found[fam].values())
        print(f"       {fam:7s} cadence={cads}  scheduled={sch:,}  "
              f"real gaps={g:,} ({100 * g / max(sch, 1):.1f}%)")

    models = {}
    for fam in MODEL_FAMILIES:
        if fam not in found:
            continue
        long = build_long(grid, found, feats, fam, blocked, train_days)
        models[fam] = fit_family(long) if long else None
        print(f"  [{fam}] trained on {int((np.isfinite(long['y']) & long['trainable']).sum()):,} "
              f"panel-rows from {len(train_days)} day(s)"
              if models[fam] is not None else f"  [{fam}] not enough data to train")

    acc, decisions = evaluate(grid, found, feats, models, test_days,
                              interval_s, blocked)

    all_filled = {}
    for fam in MODEL_FAMILIES:
        if fam not in found:
            continue
        mfill, _ = model_fill(grid, found, fam, feats, gapmask, models.get(fam), blocked)
        ifill = interp_fill(grid, found, fam)
        grid, filled, tally = combine(grid, found, fam, ifill, mfill, gapmask,
                                      interval_s, decisions)
        all_filled.update(filled)
    grid = constrain_filled(grid, found, feats, all_filled)

    summary = {"load": load_rep, "interval_s": interval_s,
               "panel_map": {str(k): v for k, v in PANEL_MAP.items()},
               "layout_check": layout, "faults": fault_rep,
               "validation": val_table.to_dict("records"),
               "train_days": sorted(str(pd.Timestamp(d).date()) for d in train_days),
               "test_days": sorted(str(pd.Timestamp(d).date()) for d in test_days),
               "dropped_days": sorted(str(pd.Timestamp(d).date()) for d in dropped_days),
               "accuracy": acc.round(4).to_dict("records") if acc is not None else None,
               "method_chosen": decisions}

    paths, _ = write_outputs(grid, found, prov, all_filled, OUT_DIR, summary,
                             load_rep.get("column_order"))
    print("\n" + "=" * 78)
    print(f"[DONE] {time.time() - t0:.0f} s   wrote {len(paths)} file(s) to {OUT_DIR}")
    return grid, summary


if __name__ == "__main__":
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass
    main()
