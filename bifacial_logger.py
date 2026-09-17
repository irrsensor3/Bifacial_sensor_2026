"""
Bifacial sensor data logger — reads 24 sensors (irradiance + temperature),
8 sensors per I2C bus, across 3 separate I2C buses. If a sensor or board
isn't plugged in, it just logs blank values instead of crashing.
 
HARDWARE SUMMARY
-----------------
There are 3 separate I2C buses on the Pi, each one identical: 4 ADS1115
boards per bus, 2 sensors per board, so 8 sensors per bus x 3 buses = 24
sensors total.
 
Extra buses (beyond the built-in one) are enabled in
/boot/firmware/config.txt with these two lines:
 
    dtoverlay=i2c3,pins_4_5      # GPIO4=SDA(pin7),  GPIO5=SCL(pin29)   -> /dev/i2c-3
    dtoverlay=i2c5,pins_12_13    # GPIO12=SDA(pin32), GPIO13=SCL(pin33) -> /dev/i2c-5
 
i2c1 (on GPIO2/3) is built into the Pi, so it doesn't need a config line.
 
Each extra bus needs its own 5V<->3.3V level shifter (a small board that
converts voltage so the Pi and the sensors can talk safely).
 
I2C ACCESS NOTE
-----------------
This talks to the buses via `adafruit_extended_bus.ExtendedI2C(bus_number)`,
NOT `busio.I2C(scl, sda)`. Blinka's normal busio.I2C() tries to guess which
bus a given SCL/SDA pin pair belongs to using a hardcoded per-board table —
that table has no idea /dev/i2c-3 and /dev/i2c-5 exist just because
config.txt says so, so it fails with "No Hardware I2C on (scl,sda)=...".
ExtendedI2C sidesteps that entirely by opening the numbered device
directly. Install it with:
 
    pip install adafruit-extended-bus --break-system-packages
 
SAMPLING / LOGGING RATES
-------------------------
Irradiance is read every SAMPLE_EVERY_SEC (5 seconds).
Temperature is only read once a minute (on the :00 second) since it
doesn't change quickly, so there's no need to check it as often.
 
Every row in the CSV has 3 columns per sensor:
 
    Irr_i     -> the irradiance reading for THIS row
    Temp_i    -> the temperature reading, only filled in on the
                 once-a-minute row (blank the rest of the time)
    IrrAvg_i  -> the average of every Irr_i reading collected since the
                 last minute mark — filled in on the same once-a-minute
                 row, then the running total resets for the next minute
 
VALIDITY CHECK
---------------
These sensors should never read below 0°C. If one does, that reading
gets thrown out instead of saved, since it's almost certainly wrong.
 
ALERTS (so a Streamlit dashboard can show what's happening)
--------------------------------------------------------------
Every time a sensor's reading gets thrown out for being sub-zero, that
event gets written to two places:
  1. "<DATA_DIR>/alerts.csv" — a plain CSV file a Streamlit app can
     read directly, no extra setup needed.
  2. Supabase's "sensor_alerts" table — so a website can check for
     alerts live, without needing to read a local file.
Both of these writes are wrapped in error-handling, so if either one
fails (e.g. no internet), it just skips that step and moves on — it
never stops the rest of the program from running.
 
TIMING NOTE
-----------
All 3 I2C buses are read AT THE SAME TIME (in parallel, using threads)
instead of one after another, since they're independent hardware.
  - Normal 5-second pass (irradiance only): about 2.8s per bus.
  - Once-a-minute pass (irradiance + temperature): about 5.6s per bus.
"""
 
from datetime import datetime
import os
import csv
import subprocess
import time
import threading
from collections import defaultdict
 
from adafruit_extended_bus import ExtendedI2C
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
 
# Supabase (the cloud database) is optional. If the "supabase" package
# isn't installed, or there's no internet, the logger keeps working
# fine — it just saves to CSV only and skips sending data online.
try:
    from supabase import create_client
    SUPABASE_LIB_AVAILABLE = True
except ImportError:
    SUPABASE_LIB_AVAILABLE = False
 
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_ENABLED = True   # set this to False to turn off the live push entirely
 
# ----------------------------------------------------------------------
# SETTINGS — change these numbers to change how the logger behaves
# ----------------------------------------------------------------------
 
DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "bifacial data")
NUM_SENSORS = 24
SAMPLE_EVERY_SEC = 5     # how often irradiance is read (seconds)
SETTLE_SLEEP = 0.35      # short pause after each reading, so the ADC has time to settle
IDLE_SLEEP = 0.5         # how often the main loop checks the clock
RETRY_BOARD_EVERY = 20   # if a board is missing, only re-check for it every N samples
 
IRR_SCALE = 240.0        # multiply the raw voltage by this to get irradiance (W/m^2)
TEMP_SCALE = 20.0        # multiply the raw voltage by this...
TEMP_OFFSET = -30.0      # ...then subtract this, to get temperature (°C)
 
MIN_VALID_TEMP_C = 0.0   # any reading below this is treated as broken/invalid
 
ALERTS_FILENAME = "alerts.csv"
ALERTS_HEADER = ["Date", "Time", "SensorID", "Bus", "Address", "Temp_C", "Message"]
 
# ----------------------------------------------------------------------
# I2C BUSES — which /dev/i2c-N device number each bus name maps to.
# These match the dtoverlay bus numbers set in config.txt (see the
# HARDWARE SUMMARY note above). We open buses by NUMBER via
# ExtendedI2C rather than by (SCL, SDA) pin pair — see the I2C ACCESS
# NOTE above for why.
# ----------------------------------------------------------------------
 
BUS_IDS = {
    "i2c1": 1,   # GPIO3 / GPIO2  (built into the Pi)
    "i2c3": 3,   # GPIO5 / GPIO4  (needs the i2c3 overlay)
    "i2c5": 5,   # GPIO13 / GPIO12 (needs the i2c5 overlay)
}
 
# ----------------------------------------------------------------------
# SENSOR MAP — tells the code exactly where to find each sensor:
# which bus, which ADS1115 board (by I2C address), and which of its
# 4 channels is irradiance vs. temperature.
# ----------------------------------------------------------------------
 
def _entry(bus, addr, irr_pin, temp_pin):
    return {"bus": bus, "addr": addr, "irr_pin": irr_pin, "temp_pin": temp_pin}
 
SENSOR_MAP = {
    # --- bus i2c1 (GPIO2/3) ---
    1:  _entry("i2c1", 0x48, 0, 1),
    2:  _entry("i2c1", 0x48, 2, 3),
    3:  _entry("i2c1", 0x49, 0, 1),
    4:  _entry("i2c1", 0x49, 2, 3),
    5:  _entry("i2c1", 0x4B, 0, 1),
    6:  _entry("i2c1", 0x4B, 2, 3),
    7:  _entry("i2c1", 0x4A, 0, 1),
    8:  _entry("i2c1", 0x4A, 2, 3),
    # --- bus i2c3 (GPIO4/5) ---
    9:  _entry("i2c3", 0x48, 0, 1),
    10: _entry("i2c3", 0x48, 2, 3),
    11: _entry("i2c3", 0x49, 0, 1),
    12: _entry("i2c3", 0x49, 2, 3),
    13: _entry("i2c3", 0x4B, 0, 1),
    14: _entry("i2c3", 0x4B, 2, 3),
    15: _entry("i2c3", 0x4A, 0, 1),
    16: _entry("i2c3", 0x4A, 2, 3),
    # --- bus i2c5 (GPIO12/13) ---
    17: _entry("i2c5", 0x48, 0, 1),
    18: _entry("i2c5", 0x48, 2, 3),
    19: _entry("i2c5", 0x49, 0, 1),
    20: _entry("i2c5", 0x49, 2, 3),
    21: _entry("i2c5", 0x4B, 0, 1),
    22: _entry("i2c5", 0x4B, 2, 3),
    23: _entry("i2c5", 0x4A, 0, 1),
    24: _entry("i2c5", 0x4A, 2, 3),
}
# If NUM_SENSORS is set lower than 24, this trims the map down so the
# code never tries to read sensors that don't actually exist.
SENSOR_MAP = {k: v for k, v in SENSOR_MAP.items() if k <= NUM_SENSORS}
 
# Groups sensor IDs by which bus they're on, e.g. {"i2c1": [1,2,...,8], ...}
# This makes it easy later to say "read all sensors on this one bus."
BUS_SENSORS = defaultdict(list)
for _sid, _cfg in SENSOR_MAP.items():
    BUS_SENSORS[_cfg["bus"]].append(_sid)
 
# ----------------------------------------------------------------------
# HARDWARE LAYER — handles talking to the actual I2C buses and boards.
# Built to be "self-healing": if a bus or board isn't there, it remembers
# that and doesn't waste time retrying every single second — but it will
# automatically try again later in case the hardware gets reconnected.
# ----------------------------------------------------------------------
 
class HardwareManager:
    def __init__(self):
        self._i2c_buses = {}          # bus name -> the open I2C connection
        self._bus_fail = set()        # buses that failed to open (don't retry every time)
        self._boards = {}             # (bus, address) -> the ADS1115 board object
        self._channels = {}           # (bus, address, pin) -> the specific channel we read from
        self._board_fail_count = {}   # (bus, address) -> how many samples since it last failed
        self._warned = set()          # tracks which missing boards we've already printed a warning for
        self._lock = threading.Lock()  # keeps the dictionaries above safe when multiple threads touch them at once
        self.known_grounded = set()   # sensor IDs currently believed to have no sensor plugged in
                                       # (channel reads ~0V -> sub-zero temp). Only touched from the
                                       # main loop (single-threaded), so no lock needed for this one.
 
    def _get_i2c(self, bus_name):
        """Opens (or reuses) the connection to one I2C bus, by /dev/i2c-N
        bus number (via ExtendedI2C) rather than by SCL/SDA pin guessing."""
        with self._lock:
            if bus_name in self._i2c_buses:
                return self._i2c_buses[bus_name]  # already open, just reuse it
            if bus_name in self._bus_fail:
                return None  # we already know this bus doesn't work, don't try again this time
            bus_id = BUS_IDS.get(bus_name)
            if bus_id is None:
                print("WARNING: unknown bus '{}' — check BUS_IDS.".format(bus_name))
                self._bus_fail.add(bus_name)
                return None
            try:
                i2c = ExtendedI2C(bus_id)
                self._i2c_buses[bus_name] = i2c
                print("Opened I2C bus '{}' (/dev/i2c-{})".format(bus_name, bus_id))
                return i2c
            except Exception as e:
                # Couldn't open the bus at all (wrong wiring, overlay not
                # enabled, etc.) — remember this so we stop trying to
                # reopen it on every single sample.
                print("WARNING: couldn't open bus '{}' ({}). Check config.txt "
                      "overlay + reboot. Sensors on this bus will be skipped.".format(bus_name, e))
                self._bus_fail.add(bus_name)
                return None
 
    def _get_board(self, bus_name, addr):
        """Finds (or reuses) one ADS1115 board on a given bus."""
        key = (bus_name, addr)
        with self._lock:
            if key in self._boards:
                return self._boards[key]  # already found this board, just reuse it
            fails = self._board_fail_count.get(key, 0)
            if fails > 0 and fails < RETRY_BOARD_EVERY:
                # We recently failed to find this board — don't bother
                # checking again yet, just count up towards the next retry.
                self._board_fail_count[key] = fails + 1
                return None
 
        i2c = self._get_i2c(bus_name)  # note: this grabs its own lock, so call it outside ours
        if i2c is None:
            with self._lock:
                self._board_fail_count[key] = 1
            return None
 
        try:
            ads = ADS.ADS1115(i2c, address=addr, data_rate=250, gain=2 / 3)
            with self._lock:
                self._boards[key] = ads
                self._board_fail_count[key] = 0
            print("Found ADS1115 board at {} on bus '{}'".format(hex(addr), bus_name))
            return ads
        except Exception as e:
            # Board isn't there (not plugged in, bad wiring, etc.) —
            # only print the warning once, then just retry quietly.
            with self._lock:
                if key not in self._warned:
                    print("WARNING: no ADS1115 found at {} on bus '{}': {}. "
                          "Will retry every {} samples.".format(hex(addr), bus_name, e, RETRY_BOARD_EVERY))
                    self._warned.add(key)
                self._board_fail_count[key] = 1
            return None
 
    def read_voltage(self, bus_name, addr, pin):
        """Reads the raw voltage from one specific channel on one board."""
        chan_key = (bus_name, addr, pin)
        with self._lock:
            chan = self._channels.get(chan_key)
 
        if chan is None:
            # First time reading this channel — set it up.
            ads = self._get_board(bus_name, addr)
            if ads is None:
                return None
            try:
                chan = AnalogIn(ads, [ADS.P0, ADS.P1, ADS.P2, ADS.P3][pin])
                with self._lock:
                    self._channels[chan_key] = chan
            except Exception as e:
                print("WARNING: couldn't open channel {} on {}: {}".format(pin, hex(addr), e))
                return None
 
        try:
            # This is the actual I2C read. It's deliberately done OUTSIDE
            # the lock, so multiple buses can read at the same time
            # instead of waiting on each other.
            return chan.voltage
        except Exception as e:
            # The read failed even though we thought the channel was
            # working — forget this board/channel so the next attempt
            # starts completely fresh instead of getting stuck.
            print("WARNING: read failed on {} pin {}: {}".format(hex(addr), pin, e))
            with self._lock:
                self._boards.pop((bus_name, addr), None)
                self._channels.pop(chan_key, None)
            return None
 
    def invalidate_sensor(self, bus_name, addr):
        """Wipes out everything we know about one board, so the next
        time it's read, the code reconnects to it completely fresh.
        Used when a reading looks wrong (e.g. sub-zero temperature) —
        we treat that the same way we'd treat a hardware failure."""
        with self._lock:
            self._boards.pop((bus_name, addr), None)
            for pin in range(4):
                self._channels.pop((bus_name, addr, pin), None)
 
    def tick(self):
        """Called once per sample. Counts up how long each missing
        board has been missing, and resets the counter once it's been
        long enough to justify trying that board again."""
        with self._lock:
            for key in list(self._board_fail_count.keys()):
                if key not in self._boards:
                    self._board_fail_count[key] = self._board_fail_count.get(key, 0) + 1
                    if self._board_fail_count[key] >= RETRY_BOARD_EVERY:
                        self._board_fail_count[key] = 0
 
 
hw = HardwareManager()
 
# ----------------------------------------------------------------------
# SUPABASE LIVE PUSH — sends each reading to the cloud database so a
# website can show it live. Every function here is written so that if
# something goes wrong (no internet, bad credentials, etc.), it just
# prints a warning and keeps going — it never stops the logger.
# ----------------------------------------------------------------------
 
_supabase_client = None
_supabase_warned = False
 
 
def _get_supabase_client():
    """Connects to Supabase the first time this is called, then
    reuses that same connection every time after."""
    global _supabase_client, _supabase_warned
    if not SUPABASE_ENABLED or not SUPABASE_LIB_AVAILABLE:
        return None
    if _supabase_client is not None:
        return _supabase_client  # already connected, reuse it
    if not SUPABASE_URL or not SUPABASE_KEY:
        if not _supabase_warned:
            print("WARNING: SUPABASE_URL/SUPABASE_KEY not set — live push disabled, CSV logging continues.")
            _supabase_warned = True
        return None
    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase for live push.")
        return _supabase_client
    except Exception as e:
        if not _supabase_warned:
            print("WARNING: couldn't connect to Supabase ({}). "
                  "Live push disabled, CSV logging continues.".format(e))
            _supabase_warned = True
        return None
 
 
# ----------------------------------------------------------------------
# SUPABASE OUTAGE TRACKING + BACKFILL
# ----------------------------------------------------------------------
# When a push fails, we note the time. The next time a push succeeds,
# we know wifi is back — before resuming normal live pushes, we scan
# today's (and possibly yesterday's, if the outage crossed midnight)
# CSV for every row timestamped during the outage window and bulk-
# insert them in one go, so the live dashboard catches up fast instead
# of just resuming from "now" with a permanent gap.
 
_supabase_down_since = None   # datetime, or None if last push succeeded
 
BACKFILL_BATCH_SIZE = 200     # rows per insert call, keeps payloads reasonable
 
 
def _row_to_readings_dict(row_dict):
    """Turns one CSV row (as a DictReader dict of strings) into the
    same shape push_to_supabase() sends live: {"Irr_1": 512.3, ...}.
    Blank CSV cells (Temp/IrrAvg on non-minute rows) become None."""
    readings = {}
    for key, val in row_dict.items():
        if key in ("Date", "Time"):
            continue
        if val == "" or val is None:
            readings[key] = None
        else:
            try:
                readings[key] = float(val)
            except ValueError:
                readings[key] = val
    return readings
 
 
def _find_csv_rows_since(outage_start, recovery_time):
    """Reads whichever daily CSV file(s) cover [outage_start, recovery_time]
    and returns a list of (date_str, time_str, readings_dict) for every
    row in that window. Usually this is just today's file, but if the
    outage started before midnight and recovered after, it checks
    yesterday's file too."""
    rows_out = []
    seen_dates = {outage_start.date(), recovery_time.date()}
 
    for d in sorted(seen_dates):
        year, month = str(d.year), str(d.month)
        month_dir = os.path.join(DATA_DIR, year, month)
        path = os.path.join(month_dir, "Bifacial_{}.csv".format(d))
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        row_dt = datetime.strptime(
                            "{} {}".format(row["Date"], row["Time"]),
                            "%Y-%m-%d %H:%M:%S.%f"
                        )
                    except ValueError:
                        try:
                            row_dt = datetime.strptime(
                                "{} {}".format(row["Date"], row["Time"]),
                                "%Y-%m-%d %H:%M:%S"
                            )
                        except ValueError:
                            continue  # unparsable row, skip it
 
                    if outage_start <= row_dt <= recovery_time:
                        rows_out.append((row["Date"], row["Time"], _row_to_readings_dict(row)))
        except Exception as e:
            print("WARNING: backfill couldn't read {}: {}".format(path, e))
 
    return rows_out
 
 
def _backfill_supabase(outage_start, recovery_time):
    """Bulk-pushes every CSV row from the outage window into
    sensor_readings, in batches, as fast as the connection allows.
    Uses upsert on (date,time) so re-sending an already-pushed row
    (e.g. the one that triggered recovery) is harmless."""
    client = _get_supabase_client()
    if client is None:
        return
 
    rows = _find_csv_rows_since(outage_start, recovery_time)
    if not rows:
        return
 
    print("INFO: Supabase back online — backfilling {} row(s) from {} to {}."
          .format(len(rows), outage_start, recovery_time))
 
    payload = [
        {"date": d, "time": t, "readings": r}
        for d, t, r in rows
    ]
 
    for i in range(0, len(payload), BACKFILL_BATCH_SIZE):
        batch = payload[i:i + BACKFILL_BATCH_SIZE]
        try:
            client.table("sensor_readings").upsert(
                batch, on_conflict="date,time"
            ).execute()
        except Exception as e:
            print("WARNING: backfill batch failed ({} rows starting at {}): {}"
                  .format(len(batch), i, e))
            # Don't keep retrying this same batch forever inside one call —
            # if wifi drops again mid-backfill, the outage-tracking logic
            # below will just pick it up again next time a push succeeds.
            return
 
    print("INFO: backfill complete.")
 
 
def push_to_supabase(date_str, time_str, readings_dict):
    """Sends one row of readings to Supabase's 'sensor_readings' table.
    Also tracks outage start/end so a recovered connection triggers a
    catch-up backfill instead of just leaving a permanent gap."""
    global _supabase_down_since
 
    client = _get_supabase_client()
    if client is None:
        return
    try:
        client.table("sensor_readings").upsert(
            {"date": date_str, "time": time_str, "readings": readings_dict},
            on_conflict="date,time",
        ).execute()
 
        # Success — if we were down, we just came back online.
        if _supabase_down_since is not None:
            outage_start = _supabase_down_since
            _supabase_down_since = None  # clear immediately so we don't re-trigger
            threading.Thread(
                target=_backfill_supabase,
                args=(outage_start, datetime.now()),
                daemon=True,
            ).start()
 
    except Exception as e:
        print("WARNING: Supabase push failed: {}".format(e))
        if _supabase_down_since is None:
            _supabase_down_since = datetime.now()
 
 
_logging_config_cache = {}
 
 
def fetch_logging_config():
    """Fetch logging mode for each sensor from Supabase.
 
    Returns:
        {sensor_id: "normal" | "force_log" | "force_unlog"}
 
    If Supabase cannot be reached, the last known configuration is
    retained so a temporary network failure does not unexpectedly
    change logging behaviour.
    """
    global _logging_config_cache
 
    client = _get_supabase_client()
    if client is None:
        return _logging_config_cache
 
    try:
        res = (
            client.table("sensor_logging_config")
            .select("sensor_id, logging_mode")
            .execute()
        )
 
        if res.data:
            _logging_config_cache = {
                int(row["sensor_id"]): row["logging_mode"]
                for row in res.data
            }
 
    except Exception as e:
        print(
            "WARNING: couldn't fetch sensor logging config "
            "({}); using last known configuration: {}".format(
                e, _logging_config_cache
            )
        )
 
    return _logging_config_cache
 
# ---------------------------------------------------------------------------
#  Remote power commands
# ---------------------------------------------------------------------------
#  The website writes a word into pi_command.command; this checks for it once a
#  minute and acts on it. Nothing here existed before, so the dashboard's
#  Reboot and Shutdown buttons wrote a value that nothing ever read.
#
#  The command is cleared BEFORE the machine is told to go down. If it were
#  cleared afterwards the write would never happen, and the Pi would reboot
#  again the moment it came back -- forever.
# ---------------------------------------------------------------------------
 
COMMAND_TABLE = "pi_command"   # must match the table the website writes to
COMMAND_ROW_ID = 1
ALLOWED_COMMANDS = ("reboot", "shutdown")
 
 
def _clear_command():
    """Blank the command field. Returns True only if Supabase confirms it."""
    client = _get_supabase_client()
    if client is None:
        return False
    try:
        client.table(COMMAND_TABLE).update(
            {"Command": None}
        ).eq("id", COMMAND_ROW_ID).execute()
        return True
    except Exception as e:
        print("WARNING: couldn't clear command field: {}".format(e))
        return False
 
 
def check_for_commands():
    """Look for a pending reboot/shutdown and carry it out.
 
    Anything not in ALLOWED_COMMANDS is ignored and cleared, so a stray value
    in that column can never be handed to the shell.
    """
    client = _get_supabase_client()
    if client is None:
        return
    try:
        res = (client.table(COMMAND_TABLE)
               .select("Command").eq("id", COMMAND_ROW_ID).execute())
    except Exception as e:
        print("WARNING: couldn't check for commands: {}".format(e))
        return
 
    if not res.data:
        return
    command = (res.data[0].get("Command") or "").strip().lower()
    if not command:
        return
 
    if command not in ALLOWED_COMMANDS:
        print("WARNING: ignoring unknown command {!r}".format(command))
        _clear_command()
        return
 
    # Refuse to act unless the command is definitely cleared first, or the Pi
    # would loop: boot, see the same command, go down again.
    if not _clear_command():
        print("ERROR: {} requested but the command could not be cleared; "
              "refusing to act, so the Pi cannot end up in a reboot loop."
              .format(command))
        return
 
    print("COMMAND: {} requested from the dashboard. Acting now.".format(command))
    try:
        log_alert(str(datetime.now().date()), str(datetime.now().time()),
                  0, "-", "-", None,
                  "{} requested from the dashboard".format(command))
    except Exception:
        pass
 
    time.sleep(2)   # give the log and the database write a moment to land
    if command == "reboot":
        subprocess.call(["sudo", "/sbin/shutdown", "-r", "now"])
    else:
        subprocess.call(["sudo", "/sbin/shutdown", "-h", "now"])
 
 
def _ensure_alerts_file():
    """Makes sure alerts.csv exists (with a header row) before we try
    to write to it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, ALERTS_FILENAME)
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(ALERTS_HEADER)
    return path
 
 
def log_alert(date_str, time_str, sensor_id, bus_name, addr, temp_c, message):
    """Records one 'this sensor's reading looked invalid' event, in
    two places at once:
      1. alerts.csv on the Pi itself.
      2. Supabase's 'sensor_alerts' table, so a website can show it live.
    Both writes are wrapped in error-handling — if one fails, it just
    prints a warning and moves on to the next step."""
    try:
        path = _ensure_alerts_file()
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(
                [date_str, time_str, sensor_id, bus_name, hex(addr) if isinstance(addr, int) else str(addr), temp_c, message]
            )
    except Exception as e:
        print("WARNING: couldn't write to alerts.csv: {}".format(e))
 
    client = _get_supabase_client()
    if client is not None:
        try:
            client.table("sensor_alerts").insert({
                "date": date_str,
                "time": time_str,
                "sensor_id": sensor_id,
                "bus": bus_name,
                "address": hex(addr) if isinstance(addr, int) else str(addr),
                "temp_c": temp_c,
                "message": message,
            }).execute()
        except Exception as e:
            print("WARNING: Supabase alert push failed: {}".format(e))
 
# ----------------------------------------------------------------------
# CSV FILE HELPERS — every sample just adds one new row to the end of
# the file, instead of rewriting the whole thing each time. This keeps
# things fast even once the file has thousands of rows in it.
# ----------------------------------------------------------------------
 
def build_header():
    """Builds the column names for the CSV: Date, Time, then 3 columns
    per sensor (Irr_1, Temp_1, IrrAvg_1, Irr_2, Temp_2, IrrAvg_2, ...)."""
    header = ["Date", "Time"]
    for i in range(1, NUM_SENSORS + 1):
        header += ["Irr_{}".format(i), "Temp_{}".format(i), "IrrAvg_{}".format(i)]
    return header
 
 
def ensure_dirs(year, month):
    """Makes sure the folder for this year/month exists (e.g.
    'bifacial data/2026/7/'), creating it if needed."""
    month_dir = os.path.join(DATA_DIR, year, month)
    os.makedirs(month_dir, exist_ok=True)
    return month_dir
 
 
def ensure_file_ready(path, header):
    """Makes sure today's CSV file exists and has the right columns.
    - If the file doesn't exist yet, creates it with a header row.
    - If it exists but has different/older columns (e.g. you changed
      NUM_SENSORS, or this file is from before IrrAvg existed),
      rewrites it once so all the old rows line up with the new
      columns. This only happens the first time the file is touched
      each day, not on every single sample."""
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return
 
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames
        if existing_header == header:
            return  # columns already match, nothing to do
        rows = list(reader)
 
    # Columns don't match — rewrite the file with the new column set,
    # filling in blanks for any column that didn't exist before.
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})
 
 
def append_row(path, row):
    """Adds one new row to the end of the CSV file."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
 
# ----------------------------------------------------------------------
# SENSOR READS — reading all 3 buses AT THE SAME TIME using threads,
# instead of one bus after another, so a full sweep of all 24 sensors
# takes a few seconds instead of much longer.
# ----------------------------------------------------------------------
 
def read_sensor(sensor_id, read_temp):
    """Reads one sensor's irradiance (always) and temperature (only if
    read_temp is True — i.e. only on the once-a-minute sample)."""
    cfg = SENSOR_MAP.get(sensor_id)
    if cfg is None:
        return None, None
 
    irr_v = hw.read_voltage(cfg["bus"], cfg["addr"], cfg["irr_pin"])
    time.sleep(SETTLE_SLEEP)  # give the ADC a moment to settle before trusting the reading
    irr = round(irr_v * IRR_SCALE, 2) if irr_v is not None else None
    if irr is not None and irr < 0:
        irr = 0.0
    
    temp = None
    if read_temp:
        temp_v = hw.read_voltage(cfg["bus"], cfg["addr"], cfg["temp_pin"])
        time.sleep(SETTLE_SLEEP)
        temp = round(temp_v * TEMP_SCALE + TEMP_OFFSET, 1) if temp_v is not None else None
 
    return irr, temp
 
 
def _read_bus_worker(bus_name, sensor_ids, results, read_temp, logging_config):
    for sid in sensor_ids:
 
        # Completely skip force-unlogged sensors.
        # They are not read from the ADC at all.
        if logging_config.get(sid, "normal") == "force_unlog":
            results[sid] = (None, None)
            continue
 
        results[sid] = read_sensor(sid, read_temp)
 
 
def sample_all_sensors(read_temp, logging_config):
    results = {}
    threads = []
 
    for bus_name, sensor_ids in BUS_SENSORS.items():
        t = threading.Thread(
            target=_read_bus_worker,
            args=(
                bus_name,
                sensor_ids,
                results,
                read_temp,
                logging_config,
            )
        )
        t.start()
        threads.append(t)
 
    for t in threads:
        t.join()
 
    hw.tick()
    return results
 
# ----------------------------------------------------------------------
# MAIN LOOP — the part that actually runs forever, sampling sensors
# and saving/pushing the data.
# ----------------------------------------------------------------------
 
def main():
    header = build_header()
    last_sample_second = None
    current_path = None
 
    # Keeps a running list of every irradiance reading collected since
    # the last minute mark, per sensor — used to calculate IrrAvg_i.
    irr_accum = defaultdict(list)
 
    while True:
        sec = int(time.strftime("%S"))
 
        # Sample every SAMPLE_EVERY_SEC seconds.
        if sec % SAMPLE_EVERY_SEC == 0 and sec != last_sample_second:
            last_sample_second = sec
            start = time.monotonic()
 
            # Temperature is read and IrrAvg is calculated once per minute.
            is_minute_mark = (sec == 0)
 
            now = datetime.now()
            year, month = str(now.year), str(now.month)
            month_dir = ensure_dirs(year, month)
            filename = "Bifacial_{}.csv".format(now.date())
            path = os.path.join(month_dir, filename)
 
            # Prepare today's CSV file only when the date changes.
            if path != current_path:
                ensure_file_ready(path, header)
                current_path = path
 
            
            # Get the latest sensor logging configuration first.
            # This is refreshed once per minute.
            logging_config = (
                fetch_logging_config()
                if is_minute_mark
                else _logging_config_cache
            )
 
            # Now read the physical sensors.
            # force_unlog sensors will be skipped completely.
            results = sample_all_sensors(
                read_temp=is_minute_mark,
                logging_config=logging_config
            )
 
            # Check for remote reboot/shutdown once per minute.
            if is_minute_mark:
                check_for_commands()
 
            # ----------------------------------------------------------
            # IRRADIANCE ACCUMULATION / PREVIOUS-MINUTE AVERAGE
            # ----------------------------------------------------------
            #
            # IrrAvg at :00 represents the samples collected during the
            # minute that just ended. The current :00 reading starts the
            # NEW minute and is therefore not included in this average.
            #
            previous_minute_avg = {}
 
            if is_minute_mark:
                for sid, samples in irr_accum.items():
                    if samples:
                        previous_minute_avg[sid] = round(
                            sum(samples) / len(samples), 2
                        )
            else:
                for sid, (irr, _temp) in results.items():
                    mode = logging_config.get(sid, "normal")
                    if (
                        mode != "force_unlog"
                        and sid not in hw.known_grounded
                        and irr is not None
                    ):
                        irr_accum[sid].append(irr)
 
            # ----------------------------------------------------------
            # BUILD CSV / SUPABASE ROW
            # ----------------------------------------------------------
 
            row = [str(now.date()), str(now.time())]
            readings_dict = {}
            invalid_sensors = []
            force_unlogged = []
 
            for sid in range(1, NUM_SENSORS + 1):
 
                irr, temp = results.get(sid, (None, None))
                irr_avg = None
 
                mode = logging_config.get(sid, "normal")
 
                # ------------------------------------------------------
                # FORCE UNLOG
                # ------------------------------------------------------
                #
                # Completely suppress this sensor.
                #
                # This applies to:
                #   - 5-second irradiance
                #   - once-per-minute temperature
                #   - once-per-minute irradiance average
                #
                if mode == "force_unlog":
                    irr = None
                    temp = None
                    irr_avg = None
                    force_unlogged.append(sid)
 
                elif not is_minute_mark:
 
                    # --------------------------------------------------
                    # 5-SECOND TICK, ALREADY-KNOWN-GROUNDED CHANNEL
                    # --------------------------------------------------
                    #
                    # A channel with no sensor wired in reads ~0V, which
                    # looks like a small-but-real irradiance number. Once
                    # the once-a-minute check below has confirmed a
                    # channel is grounded, null its irradiance here too
                    # instead of showing that near-zero value every 5s.
                    #
                    if sid in hw.known_grounded:
                        irr = None
 
                else:
 
                    # --------------------------------------------------
                    # MINUTE AVERAGE
                    # --------------------------------------------------
                    samples = irr_accum.get(sid, [])
 
                    irr_avg = (
                        round(sum(samples) / len(samples), 2)
                        if samples
                        else None
                    )
 
                    # --------------------------------------------------
                    # NORMAL MODE: SUB-ZERO / GROUNDED-CHANNEL CHECK
                    # --------------------------------------------------
                    #
                    # A channel with nothing plugged in reads ~0V, which
                    # comes out as a sub-zero temperature (~TEMP_OFFSET).
                    # That's not a hardware fault -- it just means no
                    # sensor is wired there -- so instead of alerting and
                    # resetting the board connection every single minute,
                    # we log it ONCE, remember it, and silently null it
                    # from then on until a real sensor shows up.
                    #
                    # force_log deliberately bypasses this whole check.
                    #
                    if mode == "normal":
                        if temp is not None and temp < MIN_VALID_TEMP_C:
                            if sid not in hw.known_grounded:
                                hw.known_grounded.add(sid)
                                cfg = SENSOR_MAP.get(sid)
                                if cfg is not None:
                                    log_alert(
                                        str(now.date()),
                                        str(now.time()),
                                        sid,
                                        cfg["bus"],
                                        cfg["addr"],
                                        temp,
                                        "Sub-zero reading -- treating channel "
                                        "as grounded/unpopulated. Values will "
                                        "be nulled until a sensor is detected "
                                        "here.",
                                    )
 
                            invalid_sensors.append(sid)
 
                            # Null this sensor's complete minute reading.
                            irr = None
                            temp = None
                            irr_avg = None
 
                        elif sid in hw.known_grounded:
                            # A real sensor just showed up on a channel we
                            # previously thought was grounded/unpopulated --
                            # start logging it normally again.
                            hw.known_grounded.discard(sid)
                            cfg = SENSOR_MAP.get(sid)
                            if cfg is not None:
                                log_alert(
                                    str(now.date()),
                                    str(now.time()),
                                    sid,
                                    cfg["bus"],
                                    cfg["addr"],
                                    temp,
                                    "Valid reading detected on a previously "
                                    "grounded/unpopulated channel -- "
                                    "resuming normal logging.",
                                )
 
                    # --------------------------------------------------
                    # FORCE LOG
                    # --------------------------------------------------
                    #
                    # Nothing special needs to happen here.
                    # The sensor is simply allowed through even if
                    # temperature is below 0°C.
                    #
                    # mode == "force_log"
                    # therefore keeps its original values.
 
                # ------------------------------------------------------
                # CSV CELLS
                # ------------------------------------------------------
 
                temp_cell = temp if is_minute_mark else ""
                irr_avg_cell = irr_avg if is_minute_mark else ""
 
                row += [irr, temp_cell, irr_avg_cell]
 
                # ------------------------------------------------------
                # SUPABASE LIVE READINGS
                # ------------------------------------------------------
                #
                # force_unlog sensors are NOT included at all.
                #
                if mode != "force_unlog":
                    readings_dict["Irr_{}".format(sid)] = irr
 
                    if is_minute_mark:
                        readings_dict["Temp_{}".format(sid)] = temp
                        readings_dict["IrrAvg_{}".format(sid)] = irr_avg
 
            # ----------------------------------------------------------
            # SAVE LOCALLY + PUSH TO SUPABASE
            # ----------------------------------------------------------
 
            append_row(path, row)
            push_to_supabase(
                str(now.date()),
                str(now.time()),
                readings_dict
            )
 
            # ----------------------------------------------------------
            # FINISH MINUTE
            # ----------------------------------------------------------
 
            if is_minute_mark:
 
                if invalid_sensors:
                    print(
                        "WARNING: sub-zero temp on sensors {} — "
                        "reading(s) discarded and channel(s) treated as grounded."
                        .format(invalid_sensors)
                    )
 
                if force_unlogged:
                    print(
                        "INFO: force-unlogged sensors: {}"
                        .format(force_unlogged)
                    )
 
                irr_accum.clear()
 
            # ----------------------------------------------------------
            # STATUS OUTPUT
            # ----------------------------------------------------------
 
            present = []
 
            for sid, (i, t) in results.items():
 
                mode = logging_config.get(sid, "normal")
 
                # Don't count force-unlogged sensors as present.
                if mode == "force_unlog":
                    continue
 
                # Don't count sensors whose reading was discarded.
                if sid in invalid_sensors:
                    continue
 
                if i is not None or t is not None:
                    present.append(sid)
 
            elapsed = time.monotonic() - start
 
            print(
                "[{}] sampled {}/{} sensors in {:.1f}s "
                "(temp {}): {}".format(
                    row[1],
                    len(present),
                    NUM_SENSORS,
                    elapsed,
                    "read" if is_minute_mark else "skipped",
                    present
                )
            )
 
        # Prevent unnecessary CPU usage.
        time.sleep(IDLE_SLEEP)
 
 
if __name__ == "__main__":
    main()
 
