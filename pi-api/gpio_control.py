# gpio_control.py — LED + Buzzer control via GPIO
# Drop-in replacement for gpio.js
#
# GPIO 14 = Red  LED (Fatigue indicator)
# GPIO 15 = Green LED (Normal indicator)
# GPIO 18 = Buzzer (on when Fatigue detected)
#
# Two-class system:
#   Normal  (<30%)  → Green ON,  Red OFF, Buzzer OFF
#   Fatigue (>=30%) → Green OFF, Red ON,  Buzzer ON

import os
import subprocess
import sys

_pinctrl_available = False
_state = None


def _pinctrl(pin: int, high: bool) -> None:
    """Set pin high (dh) or low (dl) via pinctrl."""
    subprocess.run(
        ["pinctrl", "set", str(pin), "dh" if high else "dl"],
        capture_output=True,
        timeout=2,
    )


# Try RPi.GPIO first, fall back to pinctrl
try:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(14, GPIO.OUT)
    GPIO.setup(15, GPIO.OUT)
    GPIO.setup(18, GPIO.OUT)
    _use_rpi = True
    print("[GPIO] Ready (RPi.GPIO). Green=GPIO15 (Normal) Red=GPIO14 (Fatigue) Buzzer=GPIO18")
except ImportError:
    try:
        _pinctrl(14, False)
        _pinctrl(15, False)
        _pinctrl(18, False)
        _pinctrl_available = True
        _use_rpi = False
        print("[GPIO] Ready (pinctrl). Green=GPIO15 Red=GPIO14 Buzzer=GPIO18")
    except Exception as e:
        _pinctrl_available = False
        _use_rpi = False
        print(f"[GPIO] GPIO hardware not available — skipping. ({e})")


def set_gpio_state(status: str) -> None:
    """Drive LEDs and buzzer based on EMG status."""
    global _state

    if not _pinctrl_available and not _use_rpi:
        return

    if status == _state:
        return
    _state = status

    try:
        if status == "normal":
            _write_pin(15, True)   # Green ON  — sensor on muscle, normal
            _write_pin(14, False)  # Red OFF
            _write_pin(18, False)  # Buzzer OFF
        elif status == "fatigue":
            _write_pin(15, False)  # Green OFF
            _write_pin(14, True)   # Red ON   — fatigue detected
            _write_pin(18, True)   # Buzzer ON
        else:                      # 'idle' — leads off / sensor not on muscle
            _write_pin(15, False)  # Green OFF
            _write_pin(14, False)  # Red OFF
            _write_pin(18, False)  # Buzzer OFF
        print(f"[GPIO] -> {status}")
    except Exception as e:
        print(f"[GPIO] Write error: {e}")


def _write_pin(pin: int, high: bool) -> None:
    if _use_rpi:
        import RPi.GPIO as GPIO

        GPIO.output(pin, GPIO.HIGH if high else GPIO.LOW)
    else:
        _pinctrl(pin, high)


def cleanup_gpio() -> None:
    try:
        if _use_rpi:
            import RPi.GPIO as GPIO

            GPIO.output(14, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            GPIO.output(18, GPIO.LOW)
            GPIO.cleanup()
        elif _pinctrl_available:
            _pinctrl(14, False)
            _pinctrl(15, False)
            _pinctrl(18, False)
    except Exception:
        pass


import atexit
atexit.register(cleanup_gpio)
