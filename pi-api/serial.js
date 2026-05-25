// serial.js — Reads EMG values from Arduino over USB serial
const { SerialPort } = require("serialport");
const { ReadlineParser } = require("@serialport/parser-readline");
const { insertReading } = require("./db");

// EventEmitter so the API can push live readings via SSE
const { EventEmitter } = require("events");
const emitter = new EventEmitter();

let port = null;

/**
 * List all available serial ports. Useful for finding Arduino device path.
 */
async function listPorts() {
  return SerialPort.list();
}

/**
 * Open the serial port and start streaming data to Neon DB.
 * @param {string} portPath  e.g. "/dev/ttyUSB0" or "/dev/ttyACM0"
 * @param {number} baudRate  default 9600 (matches Arduino sketch)
 */
function startSerial(portPath, baudRate = 9600) {
  if (port && port.isOpen) {
    console.warn("[Serial] Port already open. Close it first.");
    return;
  }

  port = new SerialPort({ path: portPath, baudRate }, (err) => {
    if (err) {
      console.error("[Serial] Failed to open port:", err.message);
      return;
    }
    console.log(`[Serial] Opened ${portPath} at ${baudRate} baud.`);
  });

  const parser = port.pipe(new ReadlineParser({ delimiter: "\r\n" }));

  parser.on("data", async (line) => {
    const raw = parseInt(line.trim(), 10);

    // Ignore non-numeric noise
    if (isNaN(raw) || raw < 0 || raw > 1023) return;

    try {
      const row = await insertReading(raw);
      // Notify SSE subscribers
      emitter.emit("reading", row);
    } catch (err) {
      console.error("[Serial] DB insert failed:", err.message);
    }
  });

  port.on("error", (err) => console.error("[Serial] Port error:", err.message));
  port.on("close", () => console.log("[Serial] Port closed."));
}

/**
 * Close the serial port gracefully.
 */
function stopSerial() {
  if (port && port.isOpen) {
    port.close((err) => {
      if (err) console.error("[Serial] Error closing port:", err.message);
      else console.log("[Serial] Port closed.");
    });
  }
}

/**
 * Returns true if the serial port is currently open.
 */
function isConnected() {
  return port !== null && port.isOpen;
}

module.exports = { listPorts, startSerial, stopSerial, isConnected, emitter };
