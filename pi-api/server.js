// server.js — Express API + SSE live stream + serial management
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { initDB, getReadings, getStats, insertReading } = require("./db");
const { listPorts, startSerial, stopSerial, isConnected, emitter } = require("./serial");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// GET /health
app.get("/health", (_req, res) => {
  res.json({ status: "ok", serial: isConnected() });
});

// GET /ports  →  list available serial ports
app.get("/ports", async (_req, res) => {
  try {
    const ports = await listPorts();
    res.json(ports);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /connect  →  open Arduino serial port
// Body: { "port": "/dev/ttyUSB0", "baudRate": 9600 }
app.post("/connect", (req, res) => {
  const portPath = req.body.port || process.env.SERIAL_PORT;
  const baudRate = parseInt(req.body.baudRate || process.env.BAUD_RATE || "9600", 10);

  if (!portPath) {
    return res.status(400).json({
      error: "Provide 'port' in request body or set SERIAL_PORT in .env",
    });
  }

  if (isConnected()) {
    return res.status(409).json({ error: "Already connected. POST /disconnect first." });
  }

  startSerial(portPath, baudRate);
  res.json({ message: `Connecting to ${portPath} at ${baudRate} baud...` });
});

// POST /disconnect  →  close Arduino serial port
app.post("/disconnect", (_req, res) => {
  stopSerial();
  res.json({ message: "Serial port closed." });
});

// GET /readings?limit=100&offset=0
app.get("/readings", async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit || "100", 10), 1000);
  const offset = parseInt(req.query.offset || "0", 10);

  if (isNaN(limit) || isNaN(offset) || limit < 1 || offset < 0) {
    return res.status(400).json({ error: "Invalid limit or offset." });
  }

  try {
    const rows = await getReadings(limit, offset);
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /stats?minutes=5
app.get("/stats", async (req, res) => {
  const minutes = parseInt(req.query.minutes || "5", 10);

  if (isNaN(minutes) || minutes < 1 || minutes > 1440) {
    return res.status(400).json({ error: "minutes must be 1-1440." });
  }

  try {
    const stats = await getStats(minutes);
    res.json(stats);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /readings  →  manual insert (for testing without Arduino)
// Body: { "rawValue": 512 }
app.post("/readings", async (req, res) => {
  const rawValue = parseInt(req.body.rawValue, 10);

  if (isNaN(rawValue) || rawValue < 0 || rawValue > 1023) {
    return res.status(400).json({ error: "rawValue must be an integer 0-1023." });
  }

  try {
    const row = await insertReading(rawValue);
    res.status(201).json(row);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /stream  →  Server-Sent Events live feed
app.get("/stream", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const heartbeat = setInterval(() => {
    res.write(": heartbeat\n\n");
  }, 15_000);

  const onReading = (row) => {
    res.write(`data: ${JSON.stringify(row)}\n\n`);
  };

  emitter.on("reading", onReading);

  req.on("close", () => {
    clearInterval(heartbeat);
    emitter.off("reading", onReading);
  });
});

// Startup
async function main() {
  await initDB();

  if (process.env.SERIAL_PORT) {
    const baudRate = parseInt(process.env.BAUD_RATE || "9600", 10);
    startSerial(process.env.SERIAL_PORT, baudRate);
  }

  app.listen(PORT, () => {
    console.log(`[API] EMG server running on http://localhost:${PORT}`);
    console.log(`[API] Endpoints:`);
    console.log(`       GET  /health`);
    console.log(`       GET  /ports`);
    console.log(`       POST /connect      { port, baudRate }`);
    console.log(`       POST /disconnect`);
    console.log(`       GET  /readings     ?limit=100&offset=0`);
    console.log(`       GET  /stats        ?minutes=5`);
    console.log(`       POST /readings     { rawValue }  (test)`);
    console.log(`       GET  /stream       (SSE live feed)`);
  });
}

main().catch((err) => {
  console.error("[Startup] Fatal error:", err);
  process.exit(1);
});
