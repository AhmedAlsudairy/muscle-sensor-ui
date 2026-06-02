// gpio.js — LED + Buzzer control via pinctrl (Pi 5 / kernel 6.x compatible)
// GPIO 14 = Red  LED
// GPIO 15 = Green LED
// GPIO 18 = Buzzer (active buzzer, HIGH = on)

const { execSync } = require("child_process");

let gpioAvailable = false;

// Verify pinctrl exists and initialise all pins LOW
try {
  execSync("which pinctrl", { stdio: "ignore" });
  execSync("pinctrl set 14 op dl");
  execSync("pinctrl set 15 op dl");
  execSync("pinctrl set 18 op dl");
  gpioAvailable = true;
  console.log("[GPIO] Ready. Red=GPIO14  Green=GPIO15  Buzzer=GPIO18");
} catch (e) {
  console.warn("[GPIO] pinctrl not available — GPIO disabled.", e.message);
}

// Track current state to avoid redundant writes
let _state = null;

function _pin(num, high) {
  execSync(`pinctrl set ${num} ${high ? "dh" : "dl"}`);
}

/**
 * Drive LEDs and buzzer based on EMG status.
 *   normal    (<30%)  → green ON,  red OFF, buzzer OFF
 *   moderate  (30-70%)→ red ON,    green OFF, buzzer OFF
 *   fatigue   (>70%)  → red ON,    green OFF, buzzer ON
 *
 * @param {"normal"|"moderate"|"fatigue"} status
 */
function setGpioState(status) {
  if (!gpioAvailable) return;
  if (status === _state) return; // no change — skip write
  _state = status;

  try {
    if (status === "normal") {
      _pin(15, true);   // green ON
      _pin(14, false);  // red OFF
      _pin(18, false);  // buzzer OFF
    } else if (status === "moderate") {
      _pin(14, true);   // red ON
      _pin(15, false);  // green OFF
      _pin(18, false);  // buzzer OFF
    } else {
      // fatigue
      _pin(14, true);   // red ON
      _pin(15, false);  // green OFF
      _pin(18, true);   // buzzer ON
    }
    console.log(`[GPIO] → ${status}`);
  } catch (e) {
    console.error("[GPIO] Write error:", e.message);
  }
}

function cleanupGpio() {
  if (!gpioAvailable) return;
  try {
    _pin(14, false);
    _pin(15, false);
    _pin(18, false);
  } catch (_) { /* ignore */ }
}

process.on("SIGTERM", cleanupGpio);
process.on("SIGINT",  cleanupGpio);
process.on("exit",    cleanupGpio);

module.exports = { setGpioState, cleanupGpio };
