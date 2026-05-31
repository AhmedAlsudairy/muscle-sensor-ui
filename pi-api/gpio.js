// gpio.js — LED + Buzzer control
// GPIO 14 = Red  LED
// GPIO 15 = Green LED
// GPIO 18 = Buzzer (active buzzer, HIGH = on)

let Gpio;
let gpioAvailable = false;

try {
  Gpio = require("onoff").Gpio;
  gpioAvailable = true;
} catch (e) {
  console.warn("[GPIO] onoff module not found — GPIO disabled.");
}

let redLed = null;
let greenLed = null;
let buzzer = null;

if (gpioAvailable) {
  try {
    redLed   = new Gpio(14, "out");
    greenLed = new Gpio(15, "out");
    buzzer   = new Gpio(18, "out");
    // Start with all off
    redLed.writeSync(0);
    greenLed.writeSync(0);
    buzzer.writeSync(0);
    console.log("[GPIO] Ready. Red=GPIO14  Green=GPIO15  Buzzer=GPIO18");
  } catch (e) {
    console.warn("[GPIO] Pin init failed:", e.message);
    gpioAvailable = false;
  }
}

// Track current state to avoid redundant writes
let _state = null;

/**
 * Drive LEDs and buzzer based on EMG status.
 *   relaxed   (<30%)  → green ON,  red OFF, buzzer OFF
 *   moderate  (30-70%)→ red ON,    green OFF, buzzer OFF
 *   contracted(>70%)  → red ON,    green OFF, buzzer ON
 *
 * @param {"relaxed"|"moderate"|"contracted"} status
 */
function setGpioState(status) {
  if (!gpioAvailable) return;
  if (status === _state) return; // no change — skip write
  _state = status;

  try {
    if (status === "relaxed") {
      greenLed.writeSync(1);
      redLed.writeSync(0);
      buzzer.writeSync(0);
    } else if (status === "moderate") {
      redLed.writeSync(1);
      greenLed.writeSync(0);
      buzzer.writeSync(0);
    } else {
      // contracted
      redLed.writeSync(1);
      greenLed.writeSync(0);
      buzzer.writeSync(1);
    }
    console.log(`[GPIO] → ${status}`);
  } catch (e) {
    console.error("[GPIO] Write error:", e.message);
  }
}

function cleanupGpio() {
  if (!gpioAvailable) return;
  try {
    if (redLed)   { redLed.writeSync(0);   redLed.unexport(); }
    if (greenLed) { greenLed.writeSync(0); greenLed.unexport(); }
    if (buzzer)   { buzzer.writeSync(0);   buzzer.unexport(); }
  } catch (_) { /* ignore */ }
}

process.on("SIGTERM", cleanupGpio);
process.on("SIGINT",  cleanupGpio);
process.on("exit",    cleanupGpio);

module.exports = { setGpioState, cleanupGpio };
