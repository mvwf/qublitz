# Photon Duel replaces the board Arena

The existing QuBlitz Arena route now opens **Photon Duel** and **Duel Training** (seven missions). The former turn-based board game, army combat, local two-player board mode and Sage advisor are retired from the supported release. Historical versions remain available in Git history. The simulator pages are unchanged.

Move with WASD, aim with arrows or pointer, hold/release Shift to fire and F to self-pulse. Space measures; E entangles; 1/2/3 select beams; Escape pauses. The in-game hub documents pointer and standard-gamepad bindings. Click inside the embedded game for keyboard focus. Touch controls are not supported.

The static Pages build and Streamlit both use `pages/_assets/quantum_chess.html`. `VENDOR_SOURCE.md` records its canonical source and `VENDOR_SHA` pins its content. Run `bash scripts/verify.sh` before release. The vendor guard rejects missing or changed assets.

Fitted simulator T₁ drives the Physics Lab illustration only, not duel combat. Progress and records are stored locally in the browser. Physical controllers have not been certified; automated checks use virtual browser-standard pads. The Phase Rift composite-rotation test remains explicitly pending.

## Release verification — 2026-11-04

Canonical HTML source: `f9ac49919e128759b88b08fc85a1a30b15466e68`; SHA-256 `f23548b9bdb58f51d48a78ee0a9d1dc51737ddb7778d71bad64f2b6846630cbd`. Screenshot below was captured from that unchanged HTML.

![Photon Duel laser arena](photon-duel-release.png)

The fresh-upstream platform gate passes, including Python tests, lint, vendored physics and duel checks. All upstream JavaScript engine suites pass. The static site assembles, every local hub link resolves, and its Arena is byte-identical to the canonical source. Missing-manifest and modified-asset mutations both fail the vendor guard.

The original release build was also driven through the real Streamlit iframe: both mode launchers, focused keyboard movement, keyup with zero drift, and Escape pause passed without uncaught browser errors. The frame permits gamepad access; hardware remains untested.

This build's game repository CI passes on that commit: physics and duel rules, the browser suites on Chromium, Firefox and WebKit, and entry into Photon Duel through the real Arena page on this repository's pinned Streamlit. WebKit skips the offline check because its driver cannot load `file://` while offline. The browser suites ignore one Firefox message logged around a reload with no script location; its source is not identified.
