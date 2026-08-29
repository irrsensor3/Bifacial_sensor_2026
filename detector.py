# =============================================================================
#  Bifacial PV — Anomaly Detection
# =============================================================================
#  Finds panels behaving unlike their peers, and says which kind of fault it
#  looks like. Built on top of pv_gapfill.py: the gap-filling model already
#  predicts what a channel *should* read from its neighbours without ever
#  looking at that channel's own value, so the difference between prediction
#  and reality is the fault signal. Nothing extra needs training.
#
#  FAULT TYPES
#    comparison   a panel consistently outside its peer group's spread
#    datetime     missing, duplicated, out-of-order or torn timestamps
#    low_current  current below peers at the same irradiance
#    diode        voltage at ~1/3 or ~2/3 of peers while current keeps flowing
#
#  WHAT IT MUST NOT FLAG  (both confirmed from real data, see NEGATIVE_CASES)
#    panel 14  dips to 2/3 voltage each morning -- so does every panel, and
#              panel 20 does it twice as often. Row-to-row shading, not a fault.
#    27-30     run 7.7% below the rest on current -- they are the monofacial
#              modules. That gap IS the bifacial gain, and it is the result.
#
#  The second case is why peer groups matter more than thresholds. A rule like
#  "flag anything 5% below the array" reports the correct answer as a fault.
# =============================================================================

import os
import json
import numpy as np
import pandas as pd

import pv_gapfill as G   # loader, panel maps, solar geometry, model

# ---------------------------------------------------------------- thresholds
# How far outside the peer group's own spread a panel must sit before it counts.
# Expressed in standard deviations of the healthy group, not in percent, so a
# tightly-matched array is held to a tighter standard than a noisy one.
SIGMA_FLAG = 4.0
SIGMA_WARN = 2.5

# A panel must be BOTH statistically unusual and practically different. With
# only four monofacial panels the group spread is so tight that a 0.3%
# difference came out at 10 sigma -- true, and completely uninteresting. Real
# faults move the number by a few percent, not a few tenths.
MIN_PCT_DEVIATION = 2.0

MIN_SAMPLES = 200        # per panel per day, below this say nothing
DAYLIGHT_MIN_CURRENT = 0.5   # A -- ignore night, where every panel reads zero
PERSIST_DAYS = 3         # a fault must repeat before it is called a fault

# Bypass-diode signature: one of a module's three substrings shorted out.
DIODE_BANDS = ((0.58, 0.75, "one substring bypassed"),
               (0.25, 0.42, "two substrings bypassed"))
DIODE_MIN_FRACTION = 0.15   # of that panel's daylight samples, per day

NEGATIVE_CASES = {
    "morning_shading": "every panel dips at low sun angles; compare to peers, "
                       "not to a fixed voltage",
    "monofacial": "devices 27-30 read ~7.7% low because they are monofacial",
}
# Nocturnal offset:
# An irradiance sensor should be close to zero when the sun is sufficiently
# below the horizon. A small amount of residual signal is allowed for sensor
# noise, dark current, reflections, etc.
NOCTURNAL_IRR_THRESHOLD = 20.0      # W/m²
NOCTURNAL_MIN_SAMPLES = 10
NOCTURNAL_MIN_FRACTION = 0.20       # 20% of nighttime samples

# Sensor flatlining
FLATLINE_MIN_SAMPLES = 20
FLATLINE_MIN_DURATION_MIN = 10
FLATLINE_VALUE_TOLERANCE = {
    "Irr": 1.0,      # W/m²
    "T": 0.05,       # °C
}
FLATLINE_MIN_FRACTION = 0.20
FLATLINE_DAYLIGHT_IRR = 50.0

#  INTERMITTENT CURRENT FLICKERING
FLICKER_MIN_SAMPLES = 200
FLICKER_MIN_FRACTION = 0.05

# Consecutive-sample change relative to peer current.
FLICKER_STEP_THRESHOLD = 0.15      # 15%
FLICKER_PEER_RATIO_TOLERANCE = 0.08

# Number of abnormal rapid changes required before reporting.
FLICKER_MIN_EVENTS = 20

#  ASYMMETRIC BIFACIAL RATIO

BIFACIAL_RATIO_MIN_SAMPLES = 200

# Minimum practical difference from comparable panels.
BIFACIAL_RATIO_MIN_DEVIATION = 10.0   # percent

# Statistical warning / fault levels.
BIFACIAL_RATIO_SIGMA_WARN = 2.5
BIFACIAL_RATIO_SIGMA_FLAG = 4.0

# Ignore very low front irradiance where the ratio becomes unstable.
BIFACIAL_RATIO_MIN_FRONT_IRR = 50.0


# Physical mapping:
#
# Each row:
#
#   Front A -> Panel 1 -> Panel 2 -> Panel 3 -> Panel 4 -> Front B
#
# Panels 1 and 2 use Front A.
# Panels 3 and 4 use Front B.
#
# These are sensor IDs from Panel_array.py.
BIFACIAL_ROWS = {
    1: {
        "front_a": 1,
        "rear": [2, 3, 4, 5],
        "front_b": 6,
    },
    2: {
        "front_a": 7,
        "rear": [8, 9, 10, 11],
        "front_b": 12,
    },
    3: {
        "front_a": 13,
        "rear": [14, 15, 16, 17],
        "front_b": 18,
    },
    4: {
        "front_a": 19,
        "rear": [20, 21, 22, 23],
        "front_b": 24,
    },
}

# =============================================================================
#  OUT-OF-BOUNDS CHECKS
# =============================================================================
# Physical / sensor validity limits.
#
# These are deliberately conservative. The purpose is to catch readings that
# are physically impossible or clearly invalid, not normal operating extremes.

OUT_OF_BOUNDS_LIMITS = {
    "Irr": {
        "min": 0.0,       # W/m²
        "max": 1500.0,    # W/m²
    },
    "T": {
        "min": -20.0,     # °C
        "max": 100.0,     # °C
    },
    "I": {
        "min": 0.0,       # A
        "max": 30.0,      # A
    },
    "V": {
        "min": 0.0,       # V
        "max": 100.0,     # V
    },
}

# =============================================================================
#  TILT-AWARE PEER GROUPING
# =============================================================================
#  Panel tilt angles, keyed by physical sensor/panel id (1-24). Set by the
#  admin from the "Panel tilt configuration" screen in Admin Controls and
#  stored in Supabase (table: panel_tilt_config). Panels within
#  TILT_TOLERANCE_DEG of each other are treated as fair peers for comparison
#  detectors; panels tilted meaningfully differently from the rest of their
#  group are compared separately instead, because tilt changes the
#  irradiance/output a healthy panel should read -- comparing a 10 deg panel
#  against a 20 deg panel would otherwise report that geometry difference as
#  a fault, the same mistake PEER_BY_POSITION exists to avoid for row-end
#  panels.
#
#  Every panel defaults to DEFAULT_TILT_DEG until the admin sets otherwise,
#  which reproduces today's behaviour (tilt plays no role in grouping) until
#  someone actually configures mixed tilts.

DEFAULT_TILT_DEG = 10.0
TILT_TOLERANCE_DEG = 2.0   # panels within this many degrees count as "same tilt"

_TILT_CACHE = None   # populated by load_tilt_config(), reused for the run


def load_tilt_config(client=None, force=False):
    """Load per-panel tilt angles from Supabase.

    Any panel not present in the table is treated as DEFAULT_TILT_DEG, so an
    unconfigured array behaves exactly as before this feature existed.
    Cached for the process; pass force=True (e.g. once at the start of a run)
    to pick up admin edits made since the cache was filled.

    Never raises: a database problem should not stop anomaly detection, it
    should just fall back to "every panel is at the default tilt", i.e. tilt
    plays no role in peer grouping for that run.
    """
    global _TILT_CACHE
    if _TILT_CACHE is not None and not force:
        return _TILT_CACHE

    tilt = {}

    if client is None:
        try:
            from ui_sections import supabase as client
        except Exception:
            client = None

    if client is not None:
        try:
            res = (
                client.table("panel_tilt_config")
                .select("panel_id, tilt_angle")
                .execute()
            )
            for row in (res.data or []):
                pid = row.get("panel_id")
                ang = row.get("tilt_angle")
                if pid is not None and ang is not None:
                    tilt[int(pid)] = float(ang)
        except Exception as exc:
            print(f"[TILT ] could not read panel_tilt_config: {exc}")
            print("        Falling back to a flat "
                  f"{DEFAULT_TILT_DEG:g} deg for every panel; the table may "
                  "not exist yet -- see panel_tilt_config.sql")

    _TILT_CACHE = tilt
    return tilt


def _tilt_of(panel, tilt_config):
    try:
        return float(tilt_config.get(int(panel), DEFAULT_TILT_DEG))
    except (TypeError, ValueError):
        return DEFAULT_TILT_DEG


def _tilt_bucket(panel, tilt_config, tolerance=TILT_TOLERANCE_DEG):
    """A coarse bucket such that two panels within `tolerance` degrees of
    each other usually land in the same bucket. Not exact right at a bucket
    edge -- good enough for deciding who is a fair peer, not for measurement.
    """
    if tolerance <= 0:
        return 0
    return int(round(_tilt_of(panel, tilt_config) / tolerance))


# =============================================================================
#  DATETIME FAULTS
# =============================================================================

def detect_datetime_faults(raw_index, interval_s):
    out = []

    original_idx = pd.DatetimeIndex(raw_index)

    if len(original_idx) < 3:
        return out

    # -------------------------------------------------------------
    # Duplicate timestamps
    # -------------------------------------------------------------

    dup = int(
        pd.Series(original_idx).duplicated().sum()
    )

    if dup:
        out.append({
            "type": "datetime",
            "subtype": "duplicate_timestamps",
            "count": dup,
            "severity": "low",
            "detail": (
                f"{dup:,} rows share a timestamp with another row"
            )
        })

    # -------------------------------------------------------------
    # Impossible timestamps
    # -------------------------------------------------------------

    bad = int(
        (
            (original_idx < pd.Timestamp("2000-01-01")) |
            (original_idx > pd.Timestamp.now() + pd.Timedelta(days=1))
        ).sum()
    )

    if bad:
        out.append({
            "type": "datetime",
            "subtype": "impossible_timestamp",
            "count": bad,
            "severity": "high",
            "detail": (
                f"{bad:,} rows dated outside any plausible range"
            )
        })

    # -------------------------------------------------------------
    # Check backwards movement BEFORE sorting
    # -------------------------------------------------------------

    original_step = (
        pd.Series(original_idx)
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    back = int((original_step < 0).sum())

    if back:
        out.append({
            "type": "datetime",
            "subtype": "time_went_backwards",
            "count": back,
            "severity": "high",
            "detail": (
                f"time moved backwards {back} time(s)"
            )
        })

    # -------------------------------------------------------------
    # Sort only for missing-gap analysis
    # -------------------------------------------------------------

    idx = original_idx.sort_values()

    step = (
        pd.Series(idx)
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    local = (
        step
        .rolling(
            101,
            center=True,
            min_periods=15
        )
        .median()
    )

    local = (
        local
        .fillna(step.median())
        .clip(lower=1.0)
    )

    gaps = step[step > local * 3]

    if len(gaps):
        out.append({
            "type": "datetime",
            "subtype": "missing_samples",
            "count": int(len(gaps)),
            "minutes_lost": round(
                float(gaps.sum() / 60),
                1
            ),
            "severity": (
                "high"
                if gaps.sum() > 3600
                else "medium"
            ),
            "detail": (
                f"{len(gaps)} gap(s), "
                f"{gaps.sum()/60:.0f} min of missing samples"
            )
        })

    return out
# =============================================================================
#  ASYMMETRIC BIFACIAL RATIO
# =============================================================================


def detect_asymmetric_bifacial_ratio(grid, day_label):
    """Detect abnormal rear/front irradiance ratios.

    Each rear sensor is compared with the nearest front reference sensor:

        Front A -> Panel 1
        Front A -> Panel 2
        Front B -> Panel 3
        Front B -> Panel 4

    The ratio itself is then compared between equivalent physical positions
    across rows. This avoids treating naturally different edge/interior
    positions as faults.

    Example:

        Row 1 Panel 2  ratio = 0.31
        Row 2 Panel 2  ratio = 0.30
        Row 3 Panel 2  ratio = 0.29
        Row 4 Panel 2  ratio = 0.12   <- suspicious

    A fixed absolute ratio threshold is deliberately avoided because the
    bifacial ratio changes with ground reflectance, sun position and weather.
    """

    out = []

    if grid is None or grid.empty:
        return out

    # -------------------------------------------------------------------------
    # Build irradiance columns.
    # -------------------------------------------------------------------------

    irr_cols = {
        sensor_id: f"Irr_{sensor_id}"
        for row in BIFACIAL_ROWS.values()
        for sensor_id in (
            [row["front_a"], row["front_b"]] + row["rear"]
        )
        if f"Irr_{sensor_id}" in grid.columns
    }

    if not irr_cols:
        return out

    irr = pd.DataFrame({
        sensor_id: pd.to_numeric(grid[col], errors="coerce")
        for sensor_id, col in irr_cols.items()
    })

    # -------------------------------------------------------------------------
    # Calculate nearest-front reference ratio for every rear sensor.
    # -------------------------------------------------------------------------

    ratios = {}

    for row_no, cfg in BIFACIAL_ROWS.items():

        front_a = cfg["front_a"]
        front_b = cfg["front_b"]

        if front_a not in irr.columns or front_b not in irr.columns:
            continue

        for pos, rear_id in enumerate(cfg["rear"], start=1):

            if rear_id not in irr.columns:
                continue

            # Panels 1 and 2 -> Front A
            # Panels 3 and 4 -> Front B
            front_id = front_a if pos <= 2 else front_b

            front = irr[front_id]
            rear = irr[rear_id]

            valid = (
                front.notna()
                & rear.notna()
                & (front > BIFACIAL_RATIO_MIN_FRONT_IRR)
            )

            ratio = (rear / front).where(valid)

            ratios[(row_no, pos)] = ratio

    # -------------------------------------------------------------------------
    # Compare equivalent panel positions between rows.
    # -------------------------------------------------------------------------

    for pos in range(1, 5):

        position_ratios = {
            row_no: ratios[(row_no, pos)]
            for row_no in range(1, 5)
            if (row_no, pos) in ratios
        }

        if len(position_ratios) < 3:
            continue

        # Build one dataframe so each row is one physical panel position.
        frame = pd.DataFrame(position_ratios)

        # Median ratio over the day for each row.
        medians = frame.median(axis=0)

        valid_medians = medians.dropna()

        if len(valid_medians) < 3:
            continue

        centre = float(valid_medians.median())

        mad = float(
            np.median(
                np.abs(valid_medians.values - centre)
            )
        )

        spread = mad * 1.4826

        if spread <= 0:
            spread = float(np.std(valid_medians.values))

        if spread <= 0:
            continue

        # ---------------------------------------------------------------------
        # Evaluate each row against its equivalent-position peers.
        # ---------------------------------------------------------------------

        for row_no, ratio_series in position_ratios.items():

            median_ratio = medians.get(row_no)

            if not np.isfinite(median_ratio):
                continue

            pct_deviation = (
                100 * (median_ratio / centre - 1)
                if centre
                else 0.0
            )

            z = (
                (median_ratio - centre) / spread
                if spread
                else 0.0
            )

            # Only report meaningful deviations.
            if (
                abs(z) < BIFACIAL_RATIO_SIGMA_WARN
                or abs(pct_deviation) < BIFACIAL_RATIO_MIN_DEVIATION
            ):
                continue

            rear_id = BIFACIAL_ROWS[row_no]["rear"][pos - 1]

            front_id = (
                BIFACIAL_ROWS[row_no]["front_a"]
                if pos <= 2
                else BIFACIAL_ROWS[row_no]["front_b"]
            )

            valid_samples = int(ratio_series.dropna().shape[0])

            if valid_samples < BIFACIAL_RATIO_MIN_SAMPLES:
                continue

            out.append({
                "type": "asymmetric_bifacial_ratio",
                "family": "Irr",
                "panel": rear_id,
                "row": row_no,
                "position": pos,
                "front_reference": front_id,
                "day": day_label,
                "ratio": round(float(median_ratio), 3),
                "peer_median_ratio": round(float(centre), 3),
                "pct_vs_peers": round(float(pct_deviation), 1),
                "sigma": round(float(z), 1),
                "samples": valid_samples,
                "severity": (
                    "high"
                    if abs(z) >= BIFACIAL_RATIO_SIGMA_FLAG
                    else "medium"
                ),
                "detail": (
                    f"Irr_{rear_id} rear/front ratio was "
                    f"{median_ratio:.3f} versus "
                    f"{centre:.3f} for equivalent panel positions "
                    f"({pct_deviation:+.1f}%, {z:+.1f} sigma), "
                    f"using Irr_{front_id} as the nearest front reference"
                ),
            })

    return out
  
# =============================================================================
#  NOCTURNAL OFFSET
# =============================================================================

def detect_nocturnal_offset(grid, found, day_label):
    """Detect irradiance sensors reporting meaningful irradiance at night.

    During sufficiently dark periods, irradiance should be close to zero.
    A persistent positive reading may indicate sensor offset, calibration
    error, wiring problems, or a sensor that is not returning to its
    expected baseline.

    This detector deliberately avoids a simple clock-based definition of
    night. It uses the available irradiance channels themselves to establish
    when the array is actually dark, which avoids falsely classifying dawn
    and dusk as nighttime.
    """
    out = []

    if "Irr" not in found:
        return out

    cols = found["Irr"]
    if not cols:
        return out

    # Build irradiance frame.
    irr = pd.DataFrame({
        p: pd.to_numeric(grid[col], errors="coerce")
        for p, col in cols.items()
    })

    if irr.empty:
        return out

    # Use the median of all available irradiance sensors as the array-level
    # reference. When essentially the whole array is dark, this should be
    # close to zero.
    array_ref = irr.median(axis=1)

    # Define nighttime conservatively.
    #
    # If the whole array is below the nocturnal threshold, it is considered
    # dark. This prevents dawn/dusk from being classified as nighttime.
    night = array_ref <= NOCTURNAL_IRR_THRESHOLD

    if night.sum() < NOCTURNAL_MIN_SAMPLES:
        return out

    night_data = irr.loc[night]

    for panel, series in night_data.items():
        valid = series.dropna()

        if len(valid) < NOCTURNAL_MIN_SAMPLES:
            continue

        # Count readings that are meaningfully above zero while the array
        # itself is considered dark.
        elevated = valid > NOCTURNAL_IRR_THRESHOLD

        fraction = float(elevated.mean())

        if fraction < NOCTURNAL_MIN_FRACTION:
            continue

        median_offset = float(valid[elevated].median()) \
            if elevated.any() else 0.0

        max_value = float(valid.max())

        # Stronger readings and a larger fraction of the night make the
        # finding more serious.
        severity = "high" if (
            fraction >= 0.50 and median_offset >= 50
        ) else "medium"

        out.append({
            "type": "nocturnal_offset",
            "family": "Irr",
            "panel": panel,
            "day": day_label,
            "fraction_of_night": round(fraction, 3),
            "median_offset": round(median_offset, 2),
            "max_value": round(max_value, 2),
            "severity": severity,
            "detail": (
                f"Irr_{panel} reported more than "
                f"{NOCTURNAL_IRR_THRESHOLD:.0f} W/m² during "
                f"{100*fraction:.0f}% of detected nighttime samples "
                f"(median elevated reading {median_offset:.1f} W/m², "
                f"maximum {max_value:.1f} W/m²)"
            ),
        })

    return out

# =============================================================================
#  SENSOR FLATLINING
# =============================================================================

def detect_sensor_flatlining(grid, found, day_label, interval_s):
    """Detect sensors that remain effectively constant during daylight.

    Nighttime is deliberately excluded. An irradiance sensor remaining near
    zero at night is normal and must not be interpreted as a flatlined sensor.

    For irradiance sensors, daylight is determined from the median of the
    available irradiance sensors. This prevents a single faulty sensor from
    defining whether the array is in daylight.

    Temperature sensors are evaluated only during the same daylight window,
    but are not required to exceed an irradiance threshold themselves.
    """

    out = []

    if grid is None or grid.empty:
        return out

    # -------------------------------------------------------------------------
    # Determine the array-level daylight window.
    # -------------------------------------------------------------------------

    irr_cols = found.get("Irr", {})

    if irr_cols:
        irr_frame = pd.DataFrame({
            p: pd.to_numeric(grid[col], errors="coerce")
            for p, col in irr_cols.items()
        })

        array_irr = irr_frame.median(axis=1)

        daylight = array_irr > FLATLINE_DAYLIGHT_IRR
    else:
        # Without irradiance channels we cannot reliably determine daylight.
        # Do not attempt flatline detection rather than guessing from time.
        return out

    if daylight.sum() < FLATLINE_MIN_SAMPLES:
        return out

    # -------------------------------------------------------------------------
    # Examine each physical sensor family.
    # -------------------------------------------------------------------------

    for fam, cols in found.items():

        if G.CHANNELS.get(fam, {}).get("derived"):
            continue

        tolerance = FLATLINE_VALUE_TOLERANCE.get(fam)

        if tolerance is None:
            continue

        for panel, col in cols.items():

            series = pd.to_numeric(
                grid[col], errors="coerce"
            )

            # Only evaluate daylight.
            series = series[daylight].dropna()

            if len(series) < MIN_SAMPLES:
                continue

            # -----------------------------------------------------------------
            # Find consecutive flat runs.
            #
            # A sample is considered flat when it changes by no more than the
            # family-specific tolerance from the previous sample.
            # -----------------------------------------------------------------

            delta = series.diff().abs()

            flat = delta <= tolerance

            # The first sample has no previous sample to compare against.
            flat.iloc[0] = False

            groups = (flat != flat.shift()).cumsum()

            max_run = 0
            total_flat = 0
            longest_value = None

            for _, run in flat.groupby(groups):

                if not bool(run.iloc[0]):
                    continue

                run_len = len(run)

                if run_len < FLATLINE_MIN_SAMPLES:
                    continue

                total_flat += run_len

                if run_len > max_run:
                    max_run = run_len

                    idx = run.index

                    if len(idx):
                        longest_value = float(
                            series.loc[idx].median()
                        )

            if max_run < FLATLINE_MIN_SAMPLES:
                continue

            # -----------------------------------------------------------------
            # Convert the longest run to real time.
            # -----------------------------------------------------------------

            duration_min = (
                max(0, max_run - 1) * float(interval_s) / 60.0
            )

            if duration_min < FLATLINE_MIN_DURATION_MIN:
                continue

            flat_fraction = total_flat / len(series)

            if flat_fraction < FLATLINE_MIN_FRACTION:
                continue

            # -----------------------------------------------------------------
            # Severity.
            # -----------------------------------------------------------------

            severity = (
                "high"
                if duration_min >= 30 and flat_fraction >= 0.50
                else "medium"
            )

            out.append({
                "type": "sensor_flatlining",
                "family": fam,
                "panel": panel,
                "day": day_label,
                "duration_minutes": round(duration_min, 1),
                "flat_fraction": round(flat_fraction, 3),
                "value": (
                    round(longest_value, 3)
                    if longest_value is not None
                    else None
                ),
                "tolerance": tolerance,
                "severity": severity,
                "detail": (
                    f"{fam}_{panel} remained effectively constant at "
                    f"{longest_value:.3g} for about "
                    f"{duration_min:.0f} min during daylight "
                    f"({100 * flat_fraction:.0f}% of valid daylight readings)"
                ),
            })

    return out
# =============================================================================
#  INTERMITTENT CURRENT FLICKERING
# =============================================================================

def detect_current_flickering(grid, found, day_label):
    """Detect rapid, repeated current fluctuations unique to one panel.

    The detector compares each panel's current against the peer median at the
    same instant. This removes common irradiance changes such as clouds.

    A flickering fault is characterised by repeated rapid changes in the
    panel/peer current ratio while the peer group itself remains relatively
    stable.

    This is different from low_current:
        low_current          -> persistently low output
        current_flickering  -> repeated rapid up/down changes
    """

    out = []

    if "I" not in found:
        return out

    icols = found["I"]

    for face, panels in _peer_groups(icols, "I").items():

        if len(panels) < 3:
            continue

        # ---------------------------------------------------------------------
        # Current frame
        # ---------------------------------------------------------------------

        I = pd.DataFrame({
            p: pd.to_numeric(grid[icols[p]], errors="coerce")
            for p in panels
        })

        # Only consider periods where the array is producing.
        live = I.max(axis=1) > DAYLIGHT_MIN_CURRENT
        I = I[live]

        if len(I) < FLICKER_MIN_SAMPLES:
            continue

        # ---------------------------------------------------------------------
        # Peer reference at each instant
        # ---------------------------------------------------------------------

        peer_ref = I.median(axis=1)

        # Avoid division by very small values.
        valid_ref = peer_ref > DAYLIGHT_MIN_CURRENT

        I = I[valid_ref]
        peer_ref = peer_ref[valid_ref]

        if len(I) < FLICKER_MIN_SAMPLES:
            continue

        # ---------------------------------------------------------------------
        # Examine each panel
        # ---------------------------------------------------------------------

        for p in panels:

            ratio = (
                I[p] / peer_ref
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if len(ratio) < FLICKER_MIN_SAMPLES:
                continue

            # ---------------------------------------------------------------
            # Consecutive change in panel/peer ratio.
            #
            # Example:
            #
            # 1.00 -> 0.99 -> 1.01 -> 0.55 -> 1.02
            #
            # The 0.55 transition is a strong flicker event.
            # ---------------------------------------------------------------

            step = ratio.diff().abs()

            # Require the change itself to be substantial.
            large_step = step >= FLICKER_STEP_THRESHOLD

            # ---------------------------------------------------------------
            # Check whether the peers themselves moved.
            #
            # If the entire array changes suddenly, it is probably irradiance
            # variation rather than a panel-specific electrical problem.
            # ---------------------------------------------------------------

            peer_without_panel = I.drop(columns=[p], errors="ignore")

            if peer_without_panel.shape[1] < 2:
                continue

            peer_median = peer_without_panel.median(axis=1)

            peer_step = peer_median.pct_change().abs()

            # A panel-specific event should happen while the peer group is
            # comparatively stable.
            peer_stable = peer_step <= FLICKER_PEER_RATIO_TOLERANCE

            events = large_step & peer_stable

            event_count = int(events.sum())

            if event_count < FLICKER_MIN_EVENTS:
                continue

            # ---------------------------------------------------------------
            # Require the behaviour to occupy a meaningful fraction of the
            # panel's producing time.
            # ---------------------------------------------------------------

            event_fraction = event_count / len(ratio)

            if event_fraction < FLICKER_MIN_FRACTION:
                continue

            # ---------------------------------------------------------------
            # Look for actual back-and-forth behaviour.
            #
            # A single step down is not necessarily flickering. We want
            # repeated reversals in direction.
            # ---------------------------------------------------------------

            direction = ratio.diff()

            sign = np.sign(direction)

            reversal = (
                (sign != sign.shift()) &
                (sign != 0) &
                (sign.shift() != 0)
            )

            reversal_events = reversal & peer_stable

            reversal_count = int(reversal_events.sum())

            # Flickering should contain repeated reversals, not just one
            # permanent change.
            if reversal_count < FLICKER_MIN_EVENTS:
                continue

            # ---------------------------------------------------------------
            # Severity
            # ---------------------------------------------------------------

            severity = (
                "high"
                if event_count >= FLICKER_MIN_EVENTS * 3
                or event_fraction >= 0.15
                else "medium"
            )

            out.append({
                "type": "current_flickering",
                "family": "I",
                "panel": p,
                "group": face,
                "day": day_label,
                "event_count": event_count,
                "reversal_count": reversal_count,
                "event_fraction": round(event_fraction, 3),
                "median_ratio": round(float(ratio.median()), 3),
                "severity": severity,
                "detail": (
                    f"I_{p} showed {event_count} rapid current changes "
                    f"({reversal_count} reversals) relative to its peers "
                    f"during {100 * event_fraction:.1f}% of producing samples, "
                    f"consistent with intermittent current flickering"
                ),
            })

    return out
# =============================================================================
#   PEER COMPARISON
# =============================================================================

# Match peers on position and tilt as well as face, wherever there are enough
# of them. P1 and P4 are block ends and see more reflected light than P2 and
# P3, so a mixed group carries that spread as if it were noise -- which
# raises the bar a real fault has to clear. Two panels tilted differently are
# the same story: their output legitimately differs by geometry, not health.
# Falls back to the coarser grouping whenever a finer split would leave a
# group too small to have a trustworthy median.
PEER_BY_POSITION = True
PEER_BY_TILT = True
MIN_PEER_GROUP = 3


def _peer_groups(cols, fam):
    """Panels split into the groups it is fair to compare within.

    Bifacial against bifacial, P1 against P1 where possible, and (once tilt
    is configured) same-tilt against same-tilt. Mixing any of these is the
    mistake that would report the four monofacial panels as broken every
    single day -- their 7.7% shortfall is the bifacial gain, i.e. the result
    -- or would do the same to a panel that is simply tilted differently on
    purpose.
    """
    tilt_config = load_tilt_config()

    by_face = {}
    for p in cols:
        face, _blk, _pos = G.config_of(p, fam)
        by_face.setdefault(face, []).append(p)

    groups = {}

    for face, panels in by_face.items():

        # Split by position first, same as before, falling back to the whole
        # face when a position group would be too small to trust.
        base = {face: panels}
        if PEER_BY_POSITION:
            by_pos = {}
            for p in panels:
                pos = G.config_of(p, fam)[2]
                by_pos.setdefault(pos, []).append(p)
            if all(len(v) >= MIN_PEER_GROUP for v in by_pos.values()) and len(by_pos) > 1:
                base = {f"{face}_P{pos}": v for pos, v in by_pos.items()}

        # Then split each of those groups by tilt bucket, again falling back
        # whenever the split would leave too few panels to compare against.
        for key, members in base.items():
            if not PEER_BY_TILT:
                groups[key] = members
                continue

            by_tilt = {}
            for p in members:
                by_tilt.setdefault(_tilt_bucket(p, tilt_config), []).append(p)

            if len(by_tilt) > 1 and all(len(v) >= MIN_PEER_GROUP for v in by_tilt.values()):
                for bucket, v in by_tilt.items():
                    deg = round(bucket * TILT_TOLERANCE_DEG)
                    groups[f"{key}_T{deg}"] = v
            else:
                groups[key] = members

    return groups


def detect_comparison_faults(grid, found, fam, day_label):
    """Panels sitting outside their own peer group's spread.

    Uses the group's standard deviation rather than a percentage, so the bar
    adapts: your 16 bifacial panels agree to within +/-0.06 A, and against that
    a 0.46 A shortfall is eight sigma. On a noisier array the same absolute gap
    would be unremarkable, and should be.
    """
    out = []
    cols = found.get(fam, {})
    if len(cols) < 3:
        return out

    # Daylight only. At night every panel reads zero and agrees perfectly,
    # which would dilute the comparison into meaninglessness.
    cur = None
    for icol in ("I", "Irr"):
        if icol in found:
            any_c = list(found[icol].values())[0]
            cur = pd.to_numeric(grid[any_c], errors="coerce")
            break
    live = (cur > DAYLIGHT_MIN_CURRENT) if cur is not None else pd.Series(True, index=grid.index)

    for face, panels in _peer_groups(cols, fam).items():
        if len(panels) < 3:
            continue   # cannot judge a group of one or two against itself
        means = {}
        for p in panels:
            v = pd.to_numeric(grid[cols[p]], errors="coerce")[live].dropna()
            if len(v) >= MIN_SAMPLES:
                means[p] = float(v.mean())
        if len(means) < 3:
            continue

        vals = np.array(list(means.values()))
        # median and MAD, so one badly broken panel cannot drag the reference
        # it is being measured against
        centre = float(np.median(vals))
        mad = float(np.median(np.abs(vals - centre)))
        spread = mad * 1.4826 if mad > 0 else float(np.std(vals))
        if spread <= 0:
            continue

        for p, m in means.items():
            z = (m - centre) / spread
            pct = 100 * (m / centre - 1) if centre else 0.0
            if abs(z) < SIGMA_WARN or abs(pct) < MIN_PCT_DEVIATION:
                continue
            out.append({
                "type": "comparison", "family": fam, "panel": p, "group": face,
                "day": day_label,
                "value": round(m, 3), "group_median": round(centre, 3),
                "pct_vs_group": round(100 * (m / centre - 1), 1),
                "sigma": round(z, 1),
                "severity": "high" if abs(z) >= SIGMA_FLAG else "medium",
                "detail": (f"{fam}_{p} averaged {m:.3g} against a group median of "
                           f"{centre:.3g} ({100*(m/centre-1):+.1f}%, {z:+.1f} sigma)"),
            })
    return out


# =============================================================================
#  LOW CURRENT
# =============================================================================

def detect_low_current(grid, found, day_label):
    """Current below peers, judged per-irradiance where irradiance exists.

    A bare threshold cannot tell "cloudy" from "faulty". Comparing panels to
    each other at the same instant can, because the weather is shared.
    """
    out = []
    if "I" not in found:
        return out
    cols = found["I"]
    for face, panels in _peer_groups(cols, "I").items():
        if len(panels) < 3:
            continue
        frame = pd.DataFrame({p: pd.to_numeric(grid[cols[p]], errors="coerce")
                              for p in panels})
        live = frame.max(axis=1) > DAYLIGHT_MIN_CURRENT
        frame = frame[live]
        if len(frame) < MIN_SAMPLES:
            continue
        # each panel as a share of the group median at the same instant, so
        # cloud cover cancels out
        ref = frame.median(axis=1)
        ratio = frame.div(ref, axis=0).replace([np.inf, -np.inf], np.nan)
        med_ratio = ratio.median()
        centre = float(np.median(med_ratio.values))
        mad = float(np.median(np.abs(med_ratio.values - centre)))
        spread = mad * 1.4826 if mad > 0 else float(np.std(med_ratio.values))
        if spread <= 0:
            continue
        for p in panels:
            r = float(med_ratio.get(p, np.nan))
            if not np.isfinite(r):
                continue
            z = (r - centre) / spread
            pct_below = 100 * (1 - r / centre) if centre else 0.0
            if z > -SIGMA_WARN or pct_below < MIN_PCT_DEVIATION:
                continue
            out.append({
                "type": "low_current", "panel": p, "group": face, "day": day_label,
                "ratio_to_peers": round(r, 3), "sigma": round(z, 1),
                "pct_below": round(100 * (1 - r / centre), 1),
                "severity": "high" if z <= -SIGMA_FLAG else "medium",
                "detail": (f"I_{p} runs {100*(1-r/centre):.1f}% below its group "
                           f"at matched irradiance ({z:+.1f} sigma)"),
            })
    return out


# =============================================================================
#  DIODE FAULTS
# =============================================================================

def detect_diode_faults(grid, found, day_label):
    """Voltage at a third or two thirds of peers while current keeps flowing.

    A module is three substrings, each with a bypass diode. When one conducts,
    that substring is shorted and voltage drops by a third at unchanged current.

    Crucially this is measured against the peer median AT THE SAME INSTANT and
    requires the behaviour to occupy a large share of the day. Every panel in
    the real array dips to 2/3 briefly each morning as the row in front shades
    it -- panel 14 does it 3% of the time and panel 20 twice as often. Those are
    shadows. A failed diode sits there.
    """
    out = []
    if "V" not in found or "I" not in found:
        return out
    vcols, icols = found["V"], found["I"]
    for face, panels in _peer_groups(vcols, "V").items():
        if len(panels) < 3:
            continue
        V = pd.DataFrame({p: pd.to_numeric(grid[vcols[p]], errors="coerce")
                          for p in panels})
        I = pd.DataFrame({p: pd.to_numeric(grid[icols[p]], errors="coerce")
                          for p in panels if p in icols})
        live = (I.max(axis=1) > DAYLIGHT_MIN_CURRENT) if len(I.columns) else V.notna().any(axis=1)
        V, I = V[live], I[live]
        if len(V) < MIN_SAMPLES:
            continue
        ref = V.median(axis=1)
        for p in panels:
            ratio = (V[p] / ref).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) < MIN_SAMPLES:
                continue
            # only count samples where this panel is still producing current --
            # otherwise a disconnected panel looks like a diode fault
            if p in I.columns:
                ok = I[p].reindex(ratio.index) > DAYLIGHT_MIN_CURRENT
                ratio = ratio[ok.fillna(False)]
            if len(ratio) < MIN_SAMPLES:
                continue
            for lo, hi, label in DIODE_BANDS:
                frac = float(((ratio >= lo) & (ratio <= hi)).mean())
                if frac < DIODE_MIN_FRACTION:
                    continue
                out.append({
                    "type": "diode", "panel": p, "group": face, "day": day_label,
                    "fraction_of_day": round(frac, 3),
                    "median_ratio": round(float(ratio.median()), 3),
                    "severity": "high" if frac > 0.5 else "medium",
                    "detail": (f"V_{p} sat at {lo:.2f}-{hi:.2f} of the peer median "
                               f"for {100*frac:.0f}% of its producing time "
                               f"({label}) while still passing current"),
                })
    return out


# =============================================================================
#  Y-CONNECTOR BRANCH FAILURE
# =============================================================================
#  Each panel feeds its meter through a Y connector, with a cheap diode on each
#  of the two branches. If a diode fails open, the meter loses one branch.
#
#  The branches are in PARALLEL, so they share voltage and split current. Losing
#  one therefore takes current down while leaving voltage alone -- the opposite
#  of a bypass-diode fault inside the module, which takes voltage down and
#  leaves current alone. That difference is what separates the two, and it is
#  why voltage has to be checked as well as current.
#
#  Expected signatures:
#    current ~50% of peers, voltage normal   -> one branch lost (meter sums)
#    current ~0%, voltage normal or at Voc   -> both branches, or a single-
#                                               channel meter losing its branch
#  A diode fails abruptly, so this also looks for a STEP rather than a drift --
#  which is what tells it apart from soiling or gradual degradation.
# =============================================================================

BRANCH_BANDS = ((0.35, 0.65, "one of two branches lost"),
                (0.00, 0.15, "no current path"))
BRANCH_MIN_FRACTION = 0.20     # of that panel's producing time, per day
BRANCH_VOLTAGE_OK = 0.85       # voltage must stay this fraction of peers
STEP_DROP = 0.35               # a fall of this much between consecutive samples
STEP_MIN_SUSTAIN = 20          # samples it must stay down to count as a step


def detect_branch_failure(grid, found, day_label):
    out = []
    if "I" not in found:
        return out
    icols = found["I"]
    vcols = found.get("V", {})

    for face, panels in _peer_groups(icols, "I").items():
        if len(panels) < 3:
            continue
        I = pd.DataFrame({p: pd.to_numeric(grid[icols[p]], errors="coerce")
                          for p in panels})
        live = I.max(axis=1) > DAYLIGHT_MIN_CURRENT
        I = I[live]
        if len(I) < MIN_SAMPLES:
            continue
        iref = I.median(axis=1)

        V = None
        if vcols:
            V = pd.DataFrame({p: pd.to_numeric(grid[vcols[p]], errors="coerce")
                              for p in panels if p in vcols})[live]
            vref = V.median(axis=1)

        for p in panels:
            ratio = (I[p] / iref).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) < MIN_SAMPLES:
                continue

            # Voltage must be healthy, or this is a different fault entirely --
            # a panel that is simply disconnected loses both, and a bypass diode
            # inside the module loses voltage rather than current.
            v_ok = True
            v_note = ""
            if V is not None and p in V.columns:
                vr = (V[p] / vref).reindex(ratio.index)
                v_ok = bool(vr.median() >= BRANCH_VOLTAGE_OK)
                v_note = f", voltage at {vr.median():.2f} of peers"

            for lo, hi, label in BRANCH_BANDS:
                frac = float(((ratio >= lo) & (ratio <= hi)).mean())
                if frac < BRANCH_MIN_FRACTION:
                    continue
                if not v_ok:
                    out.append({
                        "type": "disconnection", "panel": p, "group": face,
                        "day": day_label, "fraction_of_day": round(frac, 3),
                        "severity": "high",
                        "detail": (f"panel {p} lost BOTH current and voltage for "
                                   f"{100*frac:.0f}% of the day{v_note} -- reads "
                                   f"as a disconnection, not a diode"),
                    })
                    break
                out.append({
                    "type": "branch_diode", "panel": p, "group": face,
                    "day": day_label, "fraction_of_day": round(frac, 3),
                    "current_ratio": round(float(ratio.median()), 3),
                    "severity": "high",
                    "detail": (f"panel {p} carried {100*float(ratio.median()):.0f}% "
                               f"of peer current for {100*frac:.0f}% of the day "
                               f"while voltage stayed normal{v_note} -- "
                               f"{label}, consistent with a failed Y-connector diode"),
                })
                break

            # A step tells a diode apart from dirt: diodes fail in an instant.
            r = ratio.rolling(5, min_periods=3).median().dropna()
            if len(r) > STEP_MIN_SUSTAIN * 2:
                drop = r.shift(STEP_MIN_SUSTAIN) - r
                idx = drop[drop > STEP_DROP].index
                if len(idx):
                    when = idx[0]
                    after = r.loc[when:]
                    if len(after) >= STEP_MIN_SUSTAIN and \
                       float(after.head(STEP_MIN_SUSTAIN).median()) < \
                       float(r.loc[:when].tail(STEP_MIN_SUSTAIN).median()) - STEP_DROP:
                        out.append({
                            "type": "branch_diode", "panel": p, "group": face,
                            "day": day_label, "subtype": "step_change",
                            "at": str(when), "severity": "high",
                            "detail": (f"panel {p} dropped abruptly at {when} and "
                                       f"stayed down -- a step, not a drift, which "
                                       f"is how a diode fails rather than how a "
                                       f"panel gets dirty"),
                        })
    return out


# =============================================================================
#  OUT-OF-BOUNDS CHECKS
# =============================================================================
# Physical / sensor validity limits, checked independently of peer grouping.
# (See detect_out_of_bounds below -- this section header intentionally
# mirrors the earlier one; the implementation lives with the other family-
# level checks so it can run alongside them in the per-day loop.)


def detect_out_of_bounds(grid, found, day_label):
    """Detect physically impossible or clearly invalid sensor readings.

    Each sensor family is checked against its own physical validity range.

    This detector is intentionally independent of peer comparison. A reading
    can be abnormal even when all of its peers are also abnormal, and a single
    impossible reading should still be reported even if the rest of the array
    is healthy.

    Examples:
        Irr < 0 W/m²
        Irr > physically plausible maximum
        Temperature outside sensor/application limits
        Negative current
        Negative voltage
    """

    out = []

    if grid is None or grid.empty:
        return out

    for fam, limits in OUT_OF_BOUNDS_LIMITS.items():

        cols = found.get(fam, {})

        if not cols:
            continue

        lower = limits.get("min")
        upper = limits.get("max")

        for panel, col in cols.items():

            series = pd.to_numeric(
                grid[col],
                errors="coerce"
            )

            valid = series.notna()

            if not valid.any():
                continue

            low = valid & (series < lower)
            high = valid & (series > upper)

            low_count = int(low.sum())
            high_count = int(high.sum())
            total_bad = low_count + high_count

            if total_bad == 0:
                continue

            bad_values = series[low | high]

            min_bad = float(bad_values.min())
            max_bad = float(bad_values.max())

            total_valid = int(valid.sum())
            fraction = total_bad / total_valid if total_valid else 0.0

            # A physically impossible value is serious even if it happens
            # only once. Repeated invalid values are also reported.
            severity = (
                "high"
                if fraction >= 0.05 or total_bad >= 20
                else "medium"
            )

            directions = []

            if low_count:
                directions.append(
                    f"{low_count:,} below {lower:g}"
                )

            if high_count:
                directions.append(
                    f"{high_count:,} above {upper:g}"
                )

            out.append({
                "type": "out_of_bounds",
                "family": fam,
                "panel": panel,
                "day": day_label,
                "lower_limit": lower,
                "upper_limit": upper,
                "bad_samples": total_bad,
                "bad_fraction": round(fraction, 4),
                "low_count": low_count,
                "high_count": high_count,
                "minimum_bad_value": round(min_bad, 3),
                "maximum_bad_value": round(max_bad, 3),
                "severity": severity,
                "detail": (
                    f"{fam}_{panel} produced {total_bad:,} out-of-bounds "
                    f"reading(s): {', '.join(directions)}; "
                    f"observed invalid range "
                    f"{min_bad:.3g} to {max_bad:.3g}"
                ),
            })

    return out


# =============================================================================
#  PERSISTENCE
# =============================================================================

# Front reference sensors per block, from the Pi's channel numbering: each row
# runs Front A, four rear sensors, Front B. Everything between the pair is
# rear-facing.
FRONT_BY_BLOCK = {"B1": (1, 6), "B2": (7, 12), "B3": (13, 18), "B4": (19, 24)}


# =============================================================================
#  MODULE SPECIFICATION  —  Trina Solar TSM-645DEG21C.20
# =============================================================================
#  From the label on the back of a panel, plus the temperature coefficients
#  from the Vertex DEG21C.20 datasheet. Change these if the modules change.
MODULE = {
    "pmax": 645.0,        # W at STC
    "vmp": 37.5,          # V
    "imp": 17.23,         # A
    "voc": 45.3,          # V
    "isc": 18.31,         # A
    "gamma_pmax": -0.0034,   # per degree C  (-0.34 %/C)
    "beta_voc": -0.0025,     # per degree C  (-0.25 %/C)
    "noct": 43.0,         # C, cell temp at 800 W/m2 / 20 C ambient / 1 m/s
    "bifaciality": 0.70,  # rear face converts 70% as well as the front
}

# Performance ratio = measured power / power the module should make in the
# light and temperature it is actually sitting in. A healthy rooftop module
# lands around 0.80-0.95 once wiring, mismatch and soiling are counted.
PR_LOW = 0.70                 # below this, something is wrong
PR_MIN_IRRADIANCE = 300.0     # W/m2 — PR is meaningless in dim light
PR_MIN_SAMPLES = 60


def expected_power(irradiance, cell_temp_c, module=MODULE):
    """What one module should produce, given light and temperature.

    Linear in irradiance and linear in temperature about STC. Good to a few
    percent across normal operating conditions, which is well inside the
    tolerance that matters for spotting a fault.
    """
    g = np.asarray(irradiance, dtype=float) / 1000.0
    t = np.asarray(cell_temp_c, dtype=float)
    return module["pmax"] * g * (1.0 + module["gamma_pmax"] * (t - 25.0))


def _effective_irradiance(grid, found, module=MODULE):
    """Front irradiance plus the rear contribution, weighted by bifaciality.

    A bifacial module is not lit by its front face alone -- ignoring the rear
    would understate expected power and make every panel look like it is
    over-performing.
    """
    cols = found.get("Irr", {})
    if not cols:
        return None
    fronts = [c for pair in FRONT_BY_BLOCK.values() for c in pair]
    f = [cols[p] for p in cols if p in fronts]
    r = [cols[p] for p in cols if p not in fronts]
    if not f:
        return None
    front = pd.DataFrame({c: pd.to_numeric(grid[c], errors="coerce")
                          for c in f}).mean(axis=1)
    if not r:
        return front
    rear = pd.DataFrame({c: pd.to_numeric(grid[c], errors="coerce")
                         for c in r}).mean(axis=1)
    return front + module["bifaciality"] * rear.fillna(0.0)


def detect_mppt_mismatch(grid, found, day_label, module=MODULE):
    """Panels operating below the power the conditions should give them.

    Two findings, and they answer different questions:

      per-panel  -- one module's performance ratio adrift from its peers.
                    Points at that module: soiling, a cell fault, a bad
                    connection, or an inverter port not tracking properly.

      array-wide -- every module low together. Points at something shared:
                    dirt across the roof, a wiring loss, or the modules
                    genuinely degrading.

    Needs irradiance, so it is silent until the Pi is reporting. Falls back to
    ambient temperature where module temperature is missing, which biases the
    expected figure high -- so the check errs towards NOT flagging.
    """
    out = []
    if "P" not in found and "I" not in found:
        return out

    eff = _effective_irradiance(grid, found, module)
    if eff is None:
        return out

    # Module temperature, else ambient, else an NOCT-based estimate.
    tcols = found.get("Temp", {})
    if tcols:
        tcell = pd.DataFrame({c: pd.to_numeric(grid[c], errors="coerce")
                              for c in tcols.values()}).mean(axis=1)
    else:
        tcell = pd.Series(25.0 + (module["noct"] - 20.0) * eff / 800.0,
                          index=grid.index)

    bright = (eff > PR_MIN_IRRADIANCE) & tcell.notna()
    if bright.sum() < PR_MIN_SAMPLES:
        return out

    exp_w = pd.Series(expected_power(eff, tcell, module), index=grid.index)

    pcols = found.get("P", {})
    icols = found.get("I", {})
    vcols = found.get("V", {})

    ratios = {}
    for dev in (pcols or icols):
        if pcols:
            meas = pd.to_numeric(grid[pcols[dev]], errors="coerce") * 1000.0
        elif dev in vcols:
            meas = (pd.to_numeric(grid[icols[dev]], errors="coerce")
                    * pd.to_numeric(grid[vcols[dev]], errors="coerce"))
        else:
            continue
        ok = bright & meas.notna() & (exp_w > 0)
        if ok.sum() < PR_MIN_SAMPLES:
            continue
        ratios[dev] = float((meas[ok] / exp_w[ok]).median())

    if not ratios:
        return out

    vals = np.array(list(ratios.values()))
    centre = float(np.median(vals))
    mad = float(np.median(np.abs(vals - centre)))
    spread = mad * 1.4826 if mad > 0 else float(np.std(vals))

    # One panel adrift from the rest
    if spread > 0:
        for dev, pr in ratios.items():
            z = (pr - centre) / spread
            pct = 100 * (pr / centre - 1) if centre else 0.0
            if z > -SIGMA_WARN or abs(pct) < MIN_PCT_DEVIATION:
                continue
            out.append({
                "type": "mppt_mismatch", "panel": dev, "day": day_label,
                "performance_ratio": round(pr, 3),
                "peer_median_ratio": round(centre, 3),
                "pct_below_peers": round(-pct, 1), "sigma": round(z, 1),
                "severity": "high" if z <= -SIGMA_FLAG else "medium",
                "detail": (f"panel {dev} made {100*pr:.0f}% of the power its "
                           f"light and temperature should give, against "
                           f"{100*centre:.0f}% for its peers ({pct:+.1f}%)"),
            })

    # Everything low together
    if centre < PR_LOW:
        out.append({
            "type": "mppt_mismatch", "subtype": "array_wide", "day": day_label,
            "performance_ratio": round(centre, 3),
            "severity": "high" if centre < 0.6 else "medium",
            "detail": (f"the whole array averaged {100*centre:.0f}% of expected "
                       f"power — below the {100*PR_LOW:.0f}% a healthy rooftop "
                       f"module should manage, so this is shared: soiling, a "
                       f"wiring loss, or degradation rather than one panel"),
        })
    return out


def require_persistence(findings, min_days=PERSIST_DAYS):
    """Collapse per-day findings, and drop anything that happened once.

    A single day's anomaly is usually weather, a bird, or a passing cloud. The
    faults worth acting on repeat. This is what separates "panel 14 dipped one
    morning" from "these four panels are consistently low".
    """
    keyed = {}
    for f in findings:
        if f["type"] == "datetime":
            keyed.setdefault(("datetime", f["subtype"]), []).append(f)
            continue
        keyed.setdefault((f["type"], f.get("family", ""), f["panel"]), []).append(f)

    confirmed, provisional = [], []
    for key, group in keyed.items():
        days = sorted({g.get("day") for g in group if g.get("day")})
        rec = dict(group[-1])
        rec["days_seen"] = len(days) or 1
        rec["first_day"], rec["last_day"] = (days[0], days[-1]) if days else (None, None)
        if key[0] == "datetime" or rec["days_seen"] >= min_days:
            confirmed.append(rec)
        else:
            rec["note"] = (f"seen on {rec['days_seen']} day(s); needs "
                           f"{min_days} before it counts as a fault")
            provisional.append(rec)
    order = {"high": 0, "medium": 1, "low": 2}
    confirmed.sort(key=lambda r: (order.get(r.get("severity"), 3), -r.get("days_seen", 0)))
    return confirmed, provisional


# =============================================================================
#  MAIN
# =============================================================================

ANOMALY_TABLE = "sensor_anomalies"


def push_to_supabase(confirmed, provisional, client=None):
    """Write findings to Supabase so the dashboard can read them.

    Deliberately separate from detection: the heavy work runs on a schedule and
    the website only ever reads a small table, so a page load never waits for a
    model. Failures here are reported, never raised -- a database problem should
    not lose the analysis that has already been done.
    """
    if client is None:
        try:
            from ui_sections import supabase as client
        except Exception:
            print("[PUSH ] no Supabase client available; skipping "
                  "(the JSON report is still written)")
            return False

    rows = []
    for rec, status in [(r, "confirmed") for r in confirmed] + \
                       [(r, "provisional") for r in provisional]:
        rows.append({
            "fault_type": rec.get("type"),
            "subtype": rec.get("subtype"),
            "device_id": rec.get("panel"),
            "family": rec.get("family"),
            "severity": rec.get("severity", "low"),
            "status": status,
            "days_seen": rec.get("days_seen", 1),
            "first_seen": rec.get("first_day"),
            "last_seen": rec.get("last_day"),
            "detail": rec.get("detail"),
            "evidence": json.dumps({k: v for k, v in rec.items()
                                    if k not in ("detail", "type")}, default=str),
        })
    if not rows:
        print("[PUSH ] nothing to write")
        return True
    try:
        client.table(ANOMALY_TABLE).insert(rows).execute()
        print(f"[PUSH ] wrote {len(rows)} row(s) to {ANOMALY_TABLE}")
        return True
    except Exception as exc:
        print(f"[PUSH ] could not write to {ANOMALY_TABLE}: {exc}")
        print("        The table may not exist yet -- see CREATE TABLE in "
              "ANOMALY_SETUP.md. The JSON report is unaffected.")
        return False


def run_on_frame(wide, interval_s=None):
    """Detect on an already-loaded wide frame.

    Split out from run() so the website can hand in rows pulled from Supabase
    without writing a CSV first. Does no model training, so it is fast enough
    to sit behind a button -- the expensive part of this project is the
    gap-filling model, and detection does not use it.
    """
    if wide is None or wide.empty:
        return [], []

    load_tilt_config(force=True)   # pick up any admin edits since last run

    interval_s = interval_s or G.detect_interval(wide.index)
    findings = detect_datetime_faults(wide.index, interval_s)

    grid = G.to_grid(wide, interval_s)
    found, _ = G.discover(grid.columns)
    grid, _, _ = G.mark_faults(grid, found)

    for day, chunk in grid.groupby(grid.index.normalize()):
        day_label = str(day.date())
        if len(chunk) < MIN_SAMPLES:
            continue
        for fam in found:
            if G.CHANNELS.get(fam, {}).get("derived"):
                continue
            findings += detect_comparison_faults(chunk, found, fam, day_label)
        findings += detect_current_flickering(chunk, found, day_label)  
        findings += detect_low_current(chunk, found, day_label)
        findings += detect_diode_faults(chunk, found, day_label)
        findings += detect_branch_failure(chunk, found, day_label)
        findings += detect_asymmetric_bifacial_ratio(chunk,day_label)
        findings += detect_nocturnal_offset(chunk, found, day_label)
        findings += detect_sensor_flatlining(chunk, found, day_label, interval_s)
        findings += detect_out_of_bounds(chunk, found, day_label)
        findings += detect_mppt_mismatch(chunk, found, day_label)

    return require_persistence(findings)


def run(data_root=None, out_dir=None, push=False):
    data_root = data_root or G.DATA_ROOT
    out_dir = out_dir or G.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  Bifacial PV — Anomaly Detection")
    print("=" * 78)

    tilt_config = load_tilt_config(force=True)
    n_custom = sum(1 for v in tilt_config.values() if v != DEFAULT_TILT_DEG)
    print(f"[TILT ] {len(tilt_config)} panel(s) configured, "
          f"{n_custom} away from the {DEFAULT_TILT_DEG:g} deg default")

    wide, found, load_rep = G.load_all(data_root)
    interval_s = G.detect_interval(wide.index)
    print(f"[SCAN ] {len(wide):,} rows, {interval_s:g}s interval, "
          f"families {sorted(found)}")

    findings = detect_datetime_faults(wide.index, interval_s)
    for k in ("rows_torn", "rows_bad_time", "rows_duplicate"):
        if load_rep.get(k):
            findings.append({"type": "datetime", "subtype": k,
                             "count": load_rep[k], "severity": "medium",
                             "detail": f"{load_rep[k]:,} rows dropped on load ({k})"})

    grid = G.to_grid(wide, interval_s)
    grid, _, _ = G.mark_faults(grid, found)

    for day, chunk in grid.groupby(grid.index.normalize()):
        label = str(day.date())
    
        if len(chunk) < MIN_SAMPLES:
            continue
        for fam in found:
            if G.CHANNELS.get(fam, {}).get("derived"):
                continue   # derived channels inherit their inputs' faults
            findings += detect_comparison_faults(chunk, found, fam, label)
        findings += detect_low_current(chunk, found, label)
        findings += detect_diode_faults(chunk, found, label)
        findings += detect_branch_failure(chunk, found, label)
        findings += detect_asymmetric_bifacial_ratio(chunk, label)
        findings += detect_nocturnal_offset(chunk, found, label)
        findings += detect_current_flickering(chunk, found, label)
        findings += detect_out_of_bounds(chunk, found, label)
        findings += detect_mppt_mismatch(chunk, found, label)

    confirmed, provisional = require_persistence(findings)

    print(f"\n[RESULT] {len(confirmed)} confirmed, {len(provisional)} provisional "
          f"(seen on fewer than {PERSIST_DAYS} days)\n")
    if confirmed:
        for f in confirmed:
            tag = f.get("severity", "?").upper()
            days = f.get("days_seen", 1)
            print(f"  [{tag:6s}] {f['type']:12s} {f['detail']}")
            if f.get("first_day"):
                print(f"           seen on {days} day(s), "
                      f"{f['first_day']} to {f['last_day']}")
    else:
        print("  nothing confirmed. Either the array is healthy, or there are "
              "too few days to establish persistence.")

    if provisional:
        print(f"\n  provisional ({len(provisional)}):")
        for f in provisional[:10]:
            print(f"    {f['type']:12s} {f['detail']}  [{f.get('note','')}]")

    report = {"scanned_rows": int(len(wide)), "interval_s": interval_s,
              "families": sorted(found), "thresholds": {
                  "sigma_flag": SIGMA_FLAG, "sigma_warn": SIGMA_WARN,
                  "persist_days": PERSIST_DAYS,
                  "tilt_tolerance_deg": TILT_TOLERANCE_DEG},
              "negative_cases": NEGATIVE_CASES,
              "confirmed": confirmed, "provisional": provisional}
    path = os.path.join(out_dir, "anomalies.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    if push:
        push_to_supabase(confirmed, provisional)

    print(f"\n[DONE] {path}")
    return confirmed, provisional


if __name__ == "__main__":
    run()
