# server.py — Flask API + SSE live stream + serial management (Python)
import os
import time
import json
import queue
import threading

from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from db import init_db, get_readings, get_stats, insert_reading, \
    set_calibration_baseline, get_calibration_baseline, get_current_pct
from serial_reader import list_ports, start_serial, stop_serial, is_connected, emitter, get_last_raw

load_dotenv()

app = Flask(__name__)
CORS(app)

PORT = int(os.getenv("PORT", 3001))


# ── EventSource helpers ───────────────────────────────────────────────────

_sse_clients: list = []
_sse_lock = threading.Lock()


def _broadcast(row: dict) -> None:
    """Send a reading to all connected SSE clients."""
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put(row)
            except Exception:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


emitter.on("reading", _broadcast)


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "serial": is_connected()})


@app.route("/ports")
def ports():
    try:
        return jsonify(list_ports())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/connect", methods=["POST"])
def connect():
    data = request.get_json(silent=True) or {}
    port_path = data.get("port") or os.getenv("SERIAL_PORT")
    baud_rate = int(data.get("baudRate") or os.getenv("BAUD_RATE", "9600"))

    if not port_path:
        return jsonify({"error": "Provide 'port' in request body or set SERIAL_PORT in .env"}), 400

    if is_connected():
        return jsonify({"error": "Already connected. POST /disconnect first."}), 409

    start_serial(port_path, baud_rate)
    return jsonify({"message": f"Connecting to {port_path} at {baud_rate} baud..."})


@app.route("/disconnect", methods=["POST"])
def disconnect():
    stop_serial()
    return jsonify({"message": "Serial port closed."})


@app.route("/readings", methods=["GET"])
def readings():
    try:
        limit = min(int(request.args.get("limit", 100)), 1000)
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "Invalid limit or offset."}), 400

    if limit < 1 or offset < 0:
        return jsonify({"error": "Invalid limit or offset."}), 400

    try:
        rows = get_readings(limit, offset)
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        minutes = int(request.args.get("minutes", 5))
    except ValueError:
        return jsonify({"error": "minutes must be an integer 1-1440."}), 400

    if minutes < 1 or minutes > 1440:
        return jsonify({"error": "minutes must be 1-1440."}), 400

    try:
        return jsonify(get_stats(minutes))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/readings", methods=["POST"])
def post_reading():
    data = request.get_json(silent=True) or {}
    try:
        raw_value = int(data.get("rawValue", -1))
    except (ValueError, TypeError):
        raw_value = -1

    if raw_value < 0 or raw_value > 1023:
        return jsonify({"error": "rawValue must be an integer 0-1023."}), 400

    try:
        row = insert_reading(raw_value)
        return jsonify(row), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calibrate", methods=["GET"])
def calibrate_get():
    """Return current calibration state and live EMA envelope percentage."""
    baseline = get_calibration_baseline()
    pct      = get_current_pct()
    thresh   = 15
    status   = "idle" if not is_connected() else ("fatigue" if pct >= thresh else "normal")
    return jsonify({
        "baseline_raw":  int(baseline),
        "baseline_pct":  round(baseline / 1023 * 100, 2),
        "last_raw":      get_last_raw(),
        "current_pct":   pct,
        "threshold_pct": thresh,
        "status":        status,
        "serial":        is_connected(),
    })


@app.route("/calibrate", methods=["POST"])
def calibrate_post():
    """Set current ADC reading as the baseline (call when sensor is OFF the muscle)."""
    if not is_connected():
        return jsonify({"error": "Serial not connected. POST /connect first."}), 400
    raw = get_last_raw()
    set_calibration_baseline(raw)
    return jsonify({
        "message":      "Calibration saved. Percentages now relative to this baseline.",
        "baseline_raw": raw,
        "baseline_pct": round(raw / 1023 * 100, 2),
    })


@app.route("/stream")
def stream():
    """Server-Sent Events live feed."""
    q = queue.Queue(maxsize=200)

    with _sse_lock:
        _sse_clients.append(q)

    def generate():
        heartbeat = ':' + ' ' * 8 + 'heartbeat\n\n'
        try:
            while True:
                try:
                    row = q.get(timeout=15)
                    yield f"data: {json.dumps(row)}\n\n"
                except queue.Empty:
                    yield heartbeat
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Startup ───────────────────────────────────────────────────────────────

def _init_db_retry() -> None:
    """Retry DB init in background with exponential back-off until it succeeds."""
    delay = 10
    attempt = 0
    while True:
        attempt += 1
        try:
            init_db()
            print(f"[DB] Ready (attempt {attempt}).")
            return
        except Exception as e:
            print(f"[DB] Attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def main():
    # Try DB once synchronously; if it fails keep retrying in a daemon thread
    # so the HTTP server always starts — never crash-loops.
    try:
        init_db()
        print("[DB] Ready.")
    except Exception as e:
        print(f"[DB] Startup connect failed: {e}")
        print("[DB] Starting in degraded mode — retrying DB in background...")
        threading.Thread(target=_init_db_retry, daemon=True).start()

    if os.getenv("SERIAL_PORT"):
        baud_rate = int(os.getenv("BAUD_RATE", "9600"))
        start_serial(os.getenv("SERIAL_PORT"), baud_rate)

    print(f"[API] EMG server running on http://0.0.0.0:{PORT}")
    print("[API] Endpoints:")
    print("       GET  /health")
    print("       GET  /ports")
    print("       POST /connect      { port, baudRate }")
    print("       POST /disconnect")
    print("       GET  /readings     ?limit=100&offset=0")
    print("       GET  /stats        ?minutes=5")
    print("       POST /readings     { rawValue }  (test)")
    print("       GET  /stream       (SSE live feed)")

    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
