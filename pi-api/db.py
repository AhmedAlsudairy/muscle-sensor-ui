# db.py — Neon PostgreSQL connection + batch inserts (Python)
# Uses socket.getaddrinfo + psycopg2 hostaddr to force IPv4.
# Raspberry Pi has no IPv6 route; connecting to all resolved IPs hangs.
import os
import socket
import threading
import time
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

# Parse the connection URL once so we can use hostaddr= for IPv4 forcing
_parsed  = urlparse(DATABASE_URL)
_DB_HOST = _parsed.hostname
_DB_PORT = _parsed.port or 5432
_DB_USER = _parsed.username
_DB_PASS = _parsed.password
_DB_NAME = _parsed.path.lstrip("/")

_conn: psycopg2.extensions.connection | None = None
_conn_lock = threading.Lock()

# ── Connection ────────────────────────────────────────────────────────────

def _resolve_ipv4(hostname: str, port: int) -> str:
    """Return the first IPv4 address for hostname (skips IPv6 entirely)."""
    results = socket.getaddrinfo(hostname, port, socket.AF_INET, socket.SOCK_STREAM)
    return results[0][4][0]


def _get_conn() -> psycopg2.extensions.connection:
    global _conn
    if _conn is None or _conn.closed:
        ipv4 = _resolve_ipv4(_DB_HOST, _DB_PORT)
        _conn = psycopg2.connect(
            host=_DB_HOST,   # kept for SSL SNI — Neon routes connections by hostname
            hostaddr=ipv4,   # actual IP to dial — forces IPv4, bypasses IPv6
            port=_DB_PORT,
            user=_DB_USER,
            password=_DB_PASS,
            dbname=_DB_NAME,
            sslmode="require",
            connect_timeout=30,
        )
        _conn.autocommit = True
        print(f"[DB] Connected to {_DB_HOST} via {ipv4}.")
    return _conn


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
