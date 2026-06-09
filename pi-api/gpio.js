// gpio.js — LED + Buzzer control via pinctrl (Pi 5 / kernel 6.x compatible)
// GPIO 14 = Red  LED (Fatigue indicator)
// GPIO 15 = Green LED (Normal indicator)
// GPIO 18 = Buzzer (on when Fatigue detected)
//
// Two-class system matching dashboard:
//   Normal  (<30%)  → Green ON,  Red OFF, Buzzer OFF
//   Fatigue (>=30%) → Green OFF, Red ON,  Buzzer ON

const { execSync } = require("child_process");

let gpioAvailable = false;

try {
  execSync("which pinctrl", { stdio: "ignore" });
  execSync("pinctrl set 14 op dl");
  execSync("pinctrl set 15 op dl");
  execSync("pinctrl set 18 op dl");
  gpioAvailable = true;
  console.log("[GPIO] Ready. Green=GPIO15 (Normal)  Red=GPIO14 (Fatigue)  Buzzer=GPIO18");
} catch (e) {
  console.warn("[GPIO] pinctrl not available — GPIO disabled.", e.message);
}

let _state = null;

function _pin(num, high) {
  execSync(`pinctrl set ${num} ${high ? "dh" : "dl"}`);
}

/**
 * Drive LEDs and buzzer based on two-class EMG status.
 * @param {"normal"|"fatigue"} status
 */
function setGpioState(status) {
  if (!gpioAvailable) return;
  if (status === _state) return;
  _state = status;

  try {
    if (status === "normal") {
      _pin(15, true);   // Green ON
      _pin(14, false);  // Red OFF
      _pin(18, false);  // Buzzer OFF
    } else {
      // fatigue
      _pin(15, false);  // Green OFF
      _pin(14, true);   // Red ON
      _pin(18, true);   // Buzzer ON
    }
    console.log(`[GPIO] → ${status}`);
  } catch (e) {
    console.error("[GPIO] Write error:", e.message);
  }
}

function cleanupGpio() {
  if (!gpioAvailable) return;
  try { _pin(14, false); _pin(15, false); _pin(18, false); } catch (_) {}
}

process.on("SIGTERM", cleanupGpio);
process.on("SIGINT",  cleanupGpio);
process.on("exit",    cleanupGpio);

module.exports = { setGpioState, cleanupGpio };
