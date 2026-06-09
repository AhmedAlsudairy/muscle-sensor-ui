# serial_reader.py — Reads EMG values from Arduino over USB serial
# Drop-in replacement for serial.js (Node.js → Python using pyserial)
import os
import threading
import time

import serial
import serial.tools.list_ports

from db import insert_reading
from gpio_control import set_gpio_state
from simple_emitter import EventEmitter

emitter = EventEmitter()

_port: serial.Serial | None = None
_thread: threading.Thread | None = None
_active = threading.Event()


def list_ports() -> list:
    """List all available serial ports."""
    result = []
    for p in serial.tools.list_ports.comports():
        result.append({
            "path": p.device,
            "manufacturer": p.manufacturer,
            "serialNumber": p.serial_number,
            "pnpId": p.pnp_id,
            "location": p.location,
            "product": p.product,
            "vendorId": p.vid,
            "productId": p.pid,
        })
    return result


def _read_loop(baud_rate: int) -> None:
    """Background thread: continuously read lines from Arduino."""
    global _port, _active
    while _active.is_set() and _port and _port.is_open:
        try:
            line = _port.readline()
            if not line:
                time.sleep(0.001)
                continue

            raw_str = line.decode("utf-8", errors="ignore").strip()
            raw = int(raw_str)

            if raw < 0 or raw > 1023:
                continue

            row = insert_reading(raw)
            set_gpio_state(row["status"])
            emitter.emit("reading", row)

        except (ValueError, UnicodeDecodeError):
            pass  # skip non-numeric noise
        except serial.SerialException as e:
            print(f"[Serial] Read error: {e}")
            break
        except Exception as e:
            print(f"[Serial] DB insert failed: {e}")


def start_serial(port_path: str, baud_rate: int = 9600) -> None:
    """Open Arduino serial port and start reading thread."""
    global _port, _thread, _active

    if _port and _port.is_open:
        print("[Serial] Port already open.")
        return

    try:
        _port = serial.Serial(port_path, baud_rate, timeout=1)
    except serial.SerialException as e:
        print(f"[Serial] Failed to open {port_path}: {e}")
        return

    print(f"[Serial] Opened {port_path} at {baud_rate} baud")

    _active.set()
    _thread = threading.Thread(target=_read_loop, args=(baud_rate,), daemon=True)
    _thread.start()


def stop_serial() -> None:
    """Close serial port and stop reading thread."""
    global _port, _thread, _active
    _active.clear()

    if _thread and _thread.is_alive():
        _thread.join(timeout=2)

    if _port and _port.is_open:
        _port.close()
        print("[Serial] Port closed")

    _port = None
    _thread = None


def is_connected() -> bool:
    """Check if serial port is open."""
    return _port is not None and _port.is_open
