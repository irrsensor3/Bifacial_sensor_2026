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

# =============================================================================
#  INTERMITTENT CURRENT FLICKERING
# =============================================================================

FLICKER_MIN_SAMPLES = 200
FLICKER_MIN_FRACTION = 0.05

# Consecutive-sample change relative to peer current.
FLICKER_STEP_THRESHOLD = 0.15      # 15%
FLICKER_PEER_RATIO_TOLERANCE = 0.08

# Number of abnormal rapid changes required before reporting.
FLICKER_MIN_EVENTS = 20
# =============================================================================
#  DATETIME FAULTS
# =============================================================================

def detect_datetime_faults(raw_index, interval_s):
    """Problems with time itself, before any measurement is considered.

    Mechanical, no model involved. Every one of these was found in the real
    logger output at some point: torn writes, duplicate stamps, impossible
    dates, and whole missing stretches.
    """
    out = []
    idx = pd.DatetimeIndex(raw_index).sort_values()
    if len(idx) < 3:
        return out

    dup = int(pd.Series(idx).duplicated().sum())
    if dup:
        out.append({"type": "datetime", "subtype": "duplicate_timestamps",
                    "count": dup, "severity": "low",
                    "detail": f"{dup:,} rows share a timestamp with another row"})

    bad = int(((idx < pd.Timestamp("2000-01-01")) |
               (idx > pd.Timestamp.now() + pd.Timedelta(days=1))).sum())
    if bad:
        out.append({"type": "datetime", "subtype": "impossible_timestamp",
                    "count": bad, "severity": "high",
                    "detail": f"{bad:,} rows dated outside any plausible range"})

    step = pd.Series(idx).diff().dt.total_seconds().dropna()

    # Compare each gap against the LOCAL cadence, not one global figure.
    #
    # A logger whose rate changes -- yours went from 60 s to about 11 s -- has
    # no single interval. Measured against a global one, every sample from the
    # slower stretch looks like a missing sample: 2,903 false gaps out of 9,556
    # readings, where the database itself reports 10.
    #
    # A rolling median tracks the rate as it changes, so only a genuine pause
    # stands out from its own neighbourhood.
    local = step.rolling(101, center=True, min_periods=15).median()
    local = local.fillna(step.median()).clip(lower=1.0)
    gaps = step[step > local * 3]
    if len(gaps):
        out.append({"type": "datetime", "subtype": "missing_samples",
                    "count": int(len(gaps)),
                    "minutes_lost": round(float(gaps.sum() / 60), 1),
                    "severity": "high" if gaps.sum() > 3600 else "medium",
                    "detail": f"{len(gaps)} gap(s), {gaps.sum()/60:.0f} min of "
                              f"missing samples"})

    # Clock stepping backwards mid-file: the logger restarted or NTP corrected.
    back = int((step < 0).sum())
    if back:
        out.append({"type": "datetime", "subtype": "time_went_backwards",
                    "count": back, "severity": "high",
                    "detail": f"time moved backwards {back} time(s)"})
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

FLATLINE_MIN_SAMPLES = 20
FLATLINE_MIN_DURATION_MIN = 10

FLATLINE_VALUE_TOLERANCE = {
    "Irr": 1.0,      # W/m²
    "T": 0.05,       # °C
}

FLATLINE_MIN_FRACTION = 0.20

# Minimum array irradiance considered daylight for flatline detection.
FLATLINE_DAYLIGHT_IRR = 50.0


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
                max_run * float(interval_s) / 60.0
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
#   PEER COMPARISON
# =============================================================================

# Match peers on position as well as face where there are enough of them.
# P1 and P4 are block ends and see more reflected light than P2 and P3, so a
# mixed group carries that spread as if it were noise -- which raises the bar a
# real fault has to clear. Falls back to face-only when a position group is too
# small to have a trustworthy median.
PEER_BY_POSITION = True
MIN_PEER_GROUP = 3


def _peer_groups(cols, fam):
    """Panels split into the groups it is fair to compare within.

    Bifacial against bifacial, and where possible P1 against P1. Mixing faces is
    the mistake that would report the four monofacial panels as broken every
    single day -- their 7.7% shortfall is the bifacial gain, i.e. the result.
    """
    by_face = {}
    for p in cols:
        face, _blk, _pos = G.config_of(p, fam)
        by_face.setdefault(face, []).append(p)

    if not PEER_BY_POSITION:
        return by_face

    groups = {}
    for face, panels in by_face.items():
        by_pos = {}
        for p in panels:
            pos = G.config_of(p, fam)[2]
            by_pos.setdefault(pos, []).append(p)
        if all(len(v) >= MIN_PEER_GROUP for v in by_pos.values()) and len(by_pos) > 1:
            for pos, v in by_pos.items():
                groups[f"{face}_P{pos}"] = v
        else:
            groups[face] = panels
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
#  PERSISTENCE
# =============================================================================

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


def run_on_frame(wide, interval_s=None, label="supabase"):
    """Detect on an already-loaded wide frame.

    Split out from run() so the website can hand in rows pulled from Supabase
    without writing a CSV first. Does no model training, so it is fast enough
    to sit behind a button -- the expensive part of this project is the
    gap-filling model, and detection does not use it.
    """
    if wide is None or wide.empty:
        return [], []
    interval_s = interval_s or G.detect_interval(wide.index)
    findings = detect_datetime_faults(wide.index, interval_s)

    grid = G.to_grid(wide, interval_s)
    found, _ = G.discover(grid.columns)
    grid, _, _ = G.mark_faults(grid, found)

    for day, chunk in grid.groupby(grid.index.normalize()):
        day_label = str(day.date())
        if len(chunk) < MIN_SAMPLES:
            continue
        findings += detect_nocturnal_offset(
          chunk,
          found,
          day_label
      )
      
      findings += detect_sensor_flatlining(
          chunk,
          found,
          day_label,
          interval_s
      )
        for fam in found:
            if G.CHANNELS.get(fam, {}).get("derived"):
                continue
            findings += detect_comparison_faults(chunk, found, fam, day_label)
        findings += detect_low_current(chunk, found, day_label)
        findings += detect_diode_faults(chunk, found, day_label)
        findings += detect_branch_failure(chunk, found, day_label)

    return require_persistence(findings)


def run(data_root=None, out_dir=None, push=False):
    data_root = data_root or G.DATA_ROOT
    out_dir = out_dir or G.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  Bifacial PV — Anomaly Detection")
    print("=" * 78)

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
    
        findings += detect_nocturnal_offset(
            chunk,
            found,
            label
        )
        
        findings += detect_sensor_flatlining(
            chunk,
            found,
            label,
            interval_s
        )
        for fam in found:
            if G.CHANNELS.get(fam, {}).get("derived"):
                continue   # derived channels inherit their inputs' faults
            findings += detect_comparison_faults(chunk, found, fam, label)
        findings += detect_low_current(chunk, found, label)
        findings += detect_diode_faults(chunk, found, label)
        findings += detect_branch_failure(chunk, found, label)

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
                  "persist_days": PERSIST_DAYS},
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
