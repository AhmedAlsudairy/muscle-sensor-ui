// db.js — Neon DB connection + schema setup (using pg for Raspberry Pi)
// Force IPv4 DNS — Pi has no IPv6 route; must be set before pg connects
const dns = require("dns");
dns.setDefaultResultOrder("ipv4first");

const { Pool } = require("pg");
require("dotenv").config();

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL environment variable is required");
}

const dbUrl = process.env.DATABASE_URL.replace(/[&?]channel_binding=[^&]*/g, "");

const pool = new Pool({
  connectionString: dbUrl,
  ssl: { rejectUnauthorized: false },
  max: 3,
  idleTimeoutMillis: 60000,
  connectionTimeoutMillis: 30000,
});

pool.on("error", (err) => console.error("[DB] Pool error:", err.message));

async function initDB() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS muscle_readings (
      id               SERIAL PRIMARY KEY,
      signal_value     DECIMAL(10, 2) NOT NULL,
      signal_percentage DECIMAL(5, 2) NOT NULL,
      status           VARCHAR(20)   NOT NULL,
      peak_value       DECIMAL(10, 2),
      average_value    DECIMAL(10, 2),
      created_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_muscle_readings_created_at
    ON muscle_readings(created_at DESC)
  `);
  console.log("[DB] Table muscle_readings ready.");
}

let _peak = 0;
let _sum = 0;
let _count = 0;

const FLUSH_INTERVAL = 1000;
const MAX_BUFFER = 1000;
let _buffer = [];
let _flushTimer = null;

let _circuitOpen = false;
let _circuitResetTimer = null;
const CIRCUIT_RESET_MS = 30000;

// Two-class system matching dashboard: Normal < 30%, Fatigue >= 30%
const FATIGUE_THRESHOLD = 30;

function deriveStatus(pct) {
  return pct < FATIGUE_THRESHOLD ? "normal" : "fatigue";
}

async function _flushBuffer() {
  _flushTimer = null;
  if (_buffer.length === 0) return;
  if (_circuitOpen) {
    _scheduleFlush();
    return;
  }

  const batch = _buffer.splice(0);
  try {
    const placeholders = batch
      .map((_, i) => `($${i * 5 + 1}, $${i * 5 + 2}, $${i * 5 + 3}, $${i * 5 + 4}, $${i * 5 + 5})`)
      .join(", ");
    const params = batch.flatMap((r) => [r.rawValue, r.pct, r.status, r.peak, r.avg]);
    await pool.query(
      `INSERT INTO muscle_readings (signal_value, signal_percentage, status, peak_value, average_value) VALUES ${placeholders}`,
      params
    );
  } catch (err) {
    console.error(`[DB] Flush failed (${batch.length} readings dropped):`, err.message);
    _circuitOpen = true;
    if (_circuitResetTimer) clearTimeout(_circuitResetTimer);
    _circuitResetTimer = setTimeout(() => {
      _circuitOpen = false;
      console.log("[DB] Circuit reset — resuming inserts.");
    }, CIRCUIT_RESET_MS);
  }
}

function _scheduleFlush() {
  if (_flushTimer) return;
  _flushTimer = setTimeout(_flushBuffer, FLUSH_INTERVAL);
}

async function insertReading(rawValue) {
  const pct = parseFloat(((rawValue / 1023) * 100).toFixed(2));
  if (pct > _peak) _peak = pct;
  _sum += pct;
  _count++;
  const avg = parseFloat((_sum / _count).toFixed(2));
  const status = deriveStatus(pct);

  if (_buffer.length < MAX_BUFFER) {
    _buffer.push({ rawValue, pct, status, peak: _peak, avg });
    _scheduleFlush();
  }

  return { signal_value: rawValue, signal_percentage: pct, status, peak_value: _peak, average_value: avg, created_at: new Date().toISOString() };
}

async function getReadings(limit = 100, offset = 0) {
  const { rows } = await pool.query(
    `SELECT id, signal_value, signal_percentage, status, peak_value, average_value, created_at
     FROM muscle_readings ORDER BY created_at DESC LIMIT $1 OFFSET $2`,
    [limit, offset]
  );
  return rows;
}

async function getStats(minutes = 5) {
  const { rows } = await pool.query(
    `SELECT
       COUNT(*)::int              AS total,
       ROUND(MIN(signal_percentage), 2) AS min_pct,
       ROUND(MAX(signal_percentage), 2) AS max_pct,
       ROUND(AVG(signal_percentage), 2) AS avg_pct
     FROM muscle_readings
     WHERE created_at >= NOW() - INTERVAL '1 minute' * $1`,
    [minutes]
  );
  return rows[0];
}

module.exports = { pool, initDB, insertReading, getReadings, getStats };
