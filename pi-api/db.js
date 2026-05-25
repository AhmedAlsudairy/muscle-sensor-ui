// db.js — Neon DB connection + schema setup (using pg for Raspberry Pi)
const { Pool } = require("pg");
require("dotenv").config();

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL environment variable is required");
}

// Strip channel_binding param — not supported by the pg driver
const dbUrl = process.env.DATABASE_URL.replace(/[&?]channel_binding=[^&]*/g, "");

const pool = new Pool({
  connectionString: dbUrl,
  ssl: { rejectUnauthorized: false },
  max: 3,
  idleTimeoutMillis: 60000,
  connectionTimeoutMillis: 30000, // Neon serverless needs time to wake up
});

// Warm up connection on startup
pool.on("error", (err) => console.error("[DB] Pool error:", err.message));

/**
 * Creates the muscle_readings table (same schema as the Next.js UI).
 * Call once at startup.
 */
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

// Running stats for peak / average tracking
let _peak = 0;
let _sum = 0;
let _count = 0;

function deriveStatus(pct) {
  if (pct < 30) return "relaxed";
  if (pct < 70) return "moderate";
  return "contracted";
}

/**
 * Insert a single EMG reading into muscle_readings.
 * @param {number} rawValue  0–1023
 */
async function insertReading(rawValue) {
  const pct = parseFloat(((rawValue / 1023) * 100).toFixed(2));
  if (pct > _peak) _peak = pct;
  _sum += pct;
  _count++;
  const avg = parseFloat((_sum / _count).toFixed(2));
  const status = deriveStatus(pct);

  const { rows } = await pool.query(
    `INSERT INTO muscle_readings
       (signal_value, signal_percentage, status, peak_value, average_value)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING *`,
    [rawValue, pct, status, _peak, avg]
  );
  return rows[0];
}

/**
 * Fetch the most recent readings.
 */
async function getReadings(limit = 100, offset = 0) {
  const { rows } = await pool.query(
    `SELECT id, signal_value, signal_percentage, status, peak_value, average_value, created_at
     FROM muscle_readings ORDER BY created_at DESC LIMIT $1 OFFSET $2`,
    [limit, offset]
  );
  return rows;
}

/**
 * Return basic stats for the last N minutes.
 */
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
