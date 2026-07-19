"""
Bifacial sensor data logger — 24 irradiance/temperature sensors (8 per
bus) across 3 independent I2C buses. Works fine even if not every
sensor/board is plugged in — missing hardware just logs blank values
instead of crashing.

HARDWARE SUMMARY
-----------------
3 independent I2C buses on the Pi 4, each an exact copy of your original
4-board ADS1115 cluster (8 sensors: 4 boards x 2 sensors each).
I2C5 was chosen over I2C4 because I2C4 sits on GPIO8/9, which conflict
with SPI0. Enable the extra buses in /boot/firmware/config.txt and
reboot:

    dtoverlay=i2c3,pins_4_5      # GPIO4=SDA(pin7),  GPIO5=SCL(pin29)   -> /dev/i2c-3
    dtoverlay=i2c5,pins_12_13    # GPIO12=SDA(pin32), GPIO13=SCL(pin33) -> /dev/i2c-5

(i2c1 on GPIO2/3 is built-in, no overlay needed.)

Each new bus needs its own 5V<->3.3V bi-directional level shifter, same
as the one in your original diagram.

TIMING NOTE
-----------
Since the 3 I2C buses are independent hardware, this script reads each
bus in its own thread IN PARALLEL. Each bus has 8 sensors (0.35s settle
x 2 reads each = 0.7s x 8 = 5.6s), so the whole sample takes ~5.6s
instead of 24 x 0.7s = 16.8s if done serially — comfortable margin
inside a 30s sample window.

The original script also re-read and rewrote the entire day's CSV file
on every loop pass, which gets slower as the file grows. This script
appends a single row with the standard csv module instead — O(1) per
write regardless of file size.
"""

from datetime import datetime
import os
import csv
import time
import threading
from collections import defaultdict

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Supabase live push — optional. If the package isn't installed or the
# network/credentials aren't available, the logger keeps working with
# CSV logging only; it just skips the live push and warns once.
try:
    from supabase import create_client
    SUPABASE_LIB_AVAILABLE = True
except ImportError:
    SUPABASE_LIB_AVAILABLE = False

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_ENABLED = True   # set False to disable live push entirely

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "bifacial data")
NUM_SENSORS = 24
SAMPLE_EVERY_SEC = 30
SETTLE_SLEEP = 0.35
IDLE_SLEEP = 0.5
RETRY_BOARD_EVERY = 20   # re-probe a missing board every N samples

IRR_SCALE = 240.0
TEMP_SCALE = 20.0
TEMP_OFFSET = -30.0

# ----------------------------------------------------------------------
# I2C BUSES
# ----------------------------------------------------------------------

BUS_PINS = {
    "i2c1": (board.SCL, board.SDA),   # GPIO3 / GPIO2  (built-in)
    "i2c3": (board.D5, board.D4),     # GPIO5 / GPIO4  (dtoverlay=i2c3,pins_4_5)
    "i2c5": (board.D13, board.D12),   # GPIO13 / GPIO12 (dtoverlay=i2c5,pins_12_13)
}

# ----------------------------------------------------------------------
# SENSOR MAP — 8 sensors (4 boards) per bus, 24 total
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
SENSOR_MAP = {k: v for k, v in SENSOR_MAP.items() if k <= NUM_SENSORS}

BUS_SENSORS = defaultdict(list)
for _sid, _cfg in SENSOR_MAP.items():
    BUS_SENSORS[_cfg["bus"]].append(_sid)

# ----------------------------------------------------------------------
# HARDWARE LAYER (I2C / ADS1115) — thread-safe, lazy, self-healing
# ----------------------------------------------------------------------

class HardwareManager:
    def __init__(self):
        self._i2c_buses = {}
        self._bus_fail = set()
        self._boards = {}
        self._channels = {}
        self._board_fail_count = {}
        self._warned = set()
        self._lock = threading.Lock()   # protects the dicts above only

    def _get_i2c(self, bus_name):
        with self._lock:
            if bus_name in self._i2c_buses:
                return self._i2c_buses[bus_name]
            if bus_name in self._bus_fail:
                return None
            pins = BUS_PINS.get(bus_name)
            if pins is None:
                print("WARNING: unknown bus '{}' — check BUS_PINS.".format(bus_name))
                self._bus_fail.add(bus_name)
                return None
            scl, sda = pins
            try:
                i2c = busio.I2C(scl, sda)
                self._i2c_buses[bus_name] = i2c
                print("Opened I2C bus '{}'".format(bus_name))
                return i2c
            except Exception as e:
                print("WARNING: couldn't open bus '{}' ({}). Check config.txt "
                      "overlay + reboot. Sensors on this bus will be skipped.".format(bus_name, e))
                self._bus_fail.add(bus_name)
                return None

    def _get_board(self, bus_name, addr):
        key = (bus_name, addr)
        with self._lock:
            if key in self._boards:
                return self._boards[key]
            fails = self._board_fail_count.get(key, 0)
            if fails > 0 and fails < RETRY_BOARD_EVERY:
                self._board_fail_count[key] = fails + 1
                return None

        i2c = self._get_i2c(bus_name)  # acquires lock internally, so call outside
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
            with self._lock:
                if key not in self._warned:
                    print("WARNING: no ADS1115 found at {} on bus '{}': {}. "
                          "Will retry every {} samples.".format(hex(addr), bus_name, e, RETRY_BOARD_EVERY))
                    self._warned.add(key)
                self._board_fail_count[key] = 1
            return None

    def read_voltage(self, bus_name, addr, pin):
        chan_key = (bus_name, addr, pin)
        with self._lock:
            chan = self._channels.get(chan_key)

        if chan is None:
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
            return chan.voltage   # actual I2C transaction — NOT under lock, runs in parallel across buses
        except Exception as e:
            print("WARNING: read failed on {} pin {}: {}".format(hex(addr), pin, e))
            with self._lock:
                self._boards.pop((bus_name, addr), None)
                self._channels.pop(chan_key, None)
            return None

    def tick(self):
        with self._lock:
            for key in list(self._board_fail_count.keys()):
                if key not in self._boards:
                    self._board_fail_count[key] = self._board_fail_count.get(key, 0) + 1
                    if self._board_fail_count[key] >= RETRY_BOARD_EVERY:
                        self._board_fail_count[key] = 0


hw = HardwareManager()

# ----------------------------------------------------------------------
# SUPABASE LIVE PUSH — fault-tolerant, never blocks/crashes the logger
# ----------------------------------------------------------------------

_supabase_client = None
_supabase_warned = False


def _get_supabase_client():
    global _supabase_client, _supabase_warned
    if not SUPABASE_ENABLED or not SUPABASE_LIB_AVAILABLE:
        return None
    if _supabase_client is not None:
        return _supabase_client
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


def push_to_supabase(date_str, time_str, readings_dict):
    """Best-effort push of one sample to Supabase. Any failure (no
    network, bad credentials, table missing, etc.) just logs a warning —
    CSV logging is never affected."""
    client = _get_supabase_client()
    if client is None:
        return
    try:
        client.table("sensor_readings").insert({
            "date": date_str,
            "time": time_str,
            "readings": readings_dict,
        }).execute()
    except Exception as e:
        print("WARNING: Supabase push failed: {}".format(e))

# ----------------------------------------------------------------------
# CSV FILE HELPERS — append-only, O(1) per write
# ----------------------------------------------------------------------

def build_header():
    header = ["Date", "Time"]
    for i in range(1, NUM_SENSORS + 1):
        header += ["Irr_{}".format(i), "Temp_{}".format(i)]
    return header


def ensure_dirs(year, month):
    month_dir = os.path.join(DATA_DIR, year, month)
    os.makedirs(month_dir, exist_ok=True)
    return month_dir


def ensure_file_ready(path, header):
    """Creates the file with a header row if it doesn't exist. If it
    exists but was created with a different set of sensor columns (e.g.
    you changed NUM_SENSORS), migrates it once using only the standard
    csv module — this only runs at most once per file, not per sample."""
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames
        if existing_header == header:
            return
        rows = list(reader)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})


def append_row(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ----------------------------------------------------------------------
# SENSOR READS — parallel across buses
# ----------------------------------------------------------------------

def read_sensor(sensor_id):
    cfg = SENSOR_MAP.get(sensor_id)
    if cfg is None:
        return None, None
    irr_v = hw.read_voltage(cfg["bus"], cfg["addr"], cfg["irr_pin"])
    time.sleep(SETTLE_SLEEP)
    temp_v = hw.read_voltage(cfg["bus"], cfg["addr"], cfg["temp_pin"])
    time.sleep(SETTLE_SLEEP)
    irr = round(irr_v * IRR_SCALE, 2) if irr_v is not None else None
    temp = round(temp_v * TEMP_SCALE + TEMP_OFFSET, 1) if temp_v is not None else None
    return irr, temp


def _read_bus_worker(bus_name, sensor_ids, results):
    for sid in sensor_ids:
        results[sid] = read_sensor(sid)


def sample_all_sensors():
    """Reads every sensor, one thread per I2C bus running in parallel."""
    results = {}
    threads = []
    for bus_name, sensor_ids in BUS_SENSORS.items():
        t = threading.Thread(target=_read_bus_worker, args=(bus_name, sensor_ids, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    hw.tick()
    return results

# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------

def main():
    header = build_header()
    last_sample_second = None
    current_path = None

    while True:
        sec = int(time.strftime("%S"))
        if sec % SAMPLE_EVERY_SEC == 0 and sec != last_sample_second:
            last_sample_second = sec
            start = time.monotonic()

            now = datetime.now()
            year, month = str(now.year), str(now.month)
            month_dir = ensure_dirs(year, month)
            filename = "Bifacial_ {}.csv".format(now.date())
            path = os.path.join(month_dir, filename)

            if path != current_path:
                ensure_file_ready(path, header)
                current_path = path

            results = sample_all_sensors()

            row = [str(now.date()), str(now.time())]
            for sid in range(1, NUM_SENSORS + 1):
                irr, temp = results.get(sid, (None, None))
                row += [irr, temp]

            append_row(path, row)

            readings_dict = {}
            for sid in range(1, NUM_SENSORS + 1):
                irr, temp = results.get(sid, (None, None))
                readings_dict["Irr_{}".format(sid)] = irr
                readings_dict["Temp_{}".format(sid)] = temp
            push_to_supabase(str(now.date()), str(now.time()), readings_dict)

            present = [sid for sid, (i, t) in results.items() if i is not None or t is not None]
            elapsed = time.monotonic() - start
            print("[{}] sampled {}/{} sensors in {:.1f}s: {}".format(
                row[1], len(present), NUM_SENSORS, elapsed, present))

        time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    main()
