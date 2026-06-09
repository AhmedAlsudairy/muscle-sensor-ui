# db.py — Neon PostgreSQL connection + batch inserts
# Drop-in replacement for db.js (pg → psycopg2)
import os
import time
import threading

import dns.resolver
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

# Force IPv4 — Neon on Raspberry Pi needs it
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]

_pool: psycopg2.extensions.connection | None = None
_lock = threading.Lock()

# ── Connection ────────────────────────────────────────────────────────────

def _get_conn():
    global _pool
    if _pool is None or _pool.closed:
        _pool = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=30)
        _pool.autocommit = True
    return _pool


# ── Schema ───────────────────────────────────────────────────────────────

def init_db() -> None:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS muscle_readings (
                id               SERIAL PRIMARY KEY,
                signal_value     DECIMAL(10, 2) NOT NULL,
                signal_percentage DECIMAL(5, 2) NOT NULL,
                status           VARCHAR(20)   NOT NULL,
                peak_value       DECIMAL(10, 2),
                average_value    DECIMAL(10, 2),
                created_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_muscle_readings_created_at
            ON muscle_readings(created_at DESC)
        """)
    print("[DB] Table muscle_readings ready.")


# ── Buffered Writer ──────────────────────────────────────────────────────

_peak = 0.0
_sum = 0.0
_count = 0
_buffer: list = []
_buffer_lock = threading.Lock()
_flush_timer: threading.Timer | None = None
_circuit_open = False
_circuit_timer: threading.Timer | None = None

FLUSH_INTERVAL = 1.0
MAX_BUFFER = 1000
CIRCUIT_RESET = 30.0
FATIGUE_THRESHOLD = 30


def _derive_status(pct: float) -> str:
    return "normal" if pct < FATIGUE_THRESHOLD else "fatigue"


def _flush_buffer() -> None:
    global _flush_timer, _circuit_open, _circuit_timer
    _flush_timer = None

    with _buffer_lock:
        if not _buffer:
            return
        if _circuit_open:
            _schedule_flush()
            return
        batch = list(_buffer)
        _buffer.clear()

    try:
        from psycopg2 import extras

        conn = _get_conn()
        with conn.cursor() as cur:
            extras.execute_values(
                cur,
                """INSERT INTO muscle_readings
                   (signal_value, signal_percentage, status, peak_value, average_value)
                   VALUES %s""",
                [(r["rawValue"], r["pct"], r["status"], r["peak"], r["avg"]) for r in batch],
            )
    except Exception as e:
        print(f"[DB] Flush failed ({len(batch)} readings dropped): {e}")
        _circuit_open = True
        if _circuit_timer:
            _circuit_timer.cancel()
        _circuit_timer = threading.Timer(CIRCUIT_RESET, _reset_circuit)
        _circuit_timer.start()


def _reset_circuit() -> None:
    global _circuit_open
    _circuit_open = False
    print("[DB] Circuit reset — resuming inserts.")


def _schedule_flush() -> None:
    global _flush_timer
    if _flush_timer:
        return
    _flush_timer = threading.Timer(FLUSH_INTERVAL, _flush_buffer)
    _flush_timer.daemon = True
    _flush_timer.start()


def insert_reading(raw_value: int) -> dict:
    global _peak, _sum, _count

    pct = round((raw_value / 1023) * 100, 2)
    if pct > _peak:
        _peak = pct
    _sum += pct
    _count += 1
    avg = round(_sum / _count, 2)
    status = _derive_status(pct)

    with _buffer_lock:
        if len(_buffer) < MAX_BUFFER:
            _buffer.append({"rawValue": raw_value, "pct": pct, "status": status, "peak": _peak, "avg": avg})
            _schedule_flush()

    return {
        "signal_value": raw_value,
        "signal_percentage": pct,
        "status": status,
        "peak_value": _peak,
        "average_value": avg,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }


# ── Queries ──────────────────────────────────────────────────────────────

def get_readings(limit: int = 100, offset: int = 0) -> list:
    conn = _get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT id, signal_value, signal_percentage, status,
                      peak_value, average_value, created_at
               FROM muscle_readings ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            (limit, offset),
        )
        return [dict(r) for r in cur.fetchall()]


def get_stats(minutes: int = 5) -> dict:
    conn = _get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT COUNT(*)::int AS total,
                      ROUND(MIN(signal_percentage), 2) AS min_pct,
                      ROUND(MAX(signal_percentage), 2) AS max_pct,
                      ROUND(AVG(signal_percentage), 2) AS avg_pct
               FROM muscle_readings
               WHERE created_at >= NOW() - INTERVAL '1 minute' * %s""",
            (minutes,),
        )
        return dict(cur.fetchone() or {})
