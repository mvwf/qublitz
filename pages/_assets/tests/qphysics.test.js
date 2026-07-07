// Regression test for the QPhysics engine. Extracts the REAL C / GATES /
// QPhysics blocks from quantum_chess.html (no copy) and asserts the critique's
// physics success criteria. Run: node tests/qphysics.test.js
const fs = require('fs');
const path = require('path');
const htmlPath = process.argv[2] || path.join(__dirname, '..', 'quantum_chess.html');
const src = fs.readFileSync(htmlPath, 'utf8').split('\n');
// Find a block by a start-anchor regex, reading until the first line that
// matches the close regex (robust to line-number drift as the file changes).
function findBlock(startRe, closeRe) {
  const start = src.findIndex(l => startRe.test(l));
  if (start < 0) throw new Error('start not found: ' + startRe);
  for (let i = start; i < src.length; i++) {
    if (closeRe.test(src[i])) return src.slice(start, i + 1).join('\n');
  }
  throw new Error('close not found for: ' + startRe);
}
const piSrc    = src.find(l => /^const π\s*=/.test(l));        // const π = Math.PI;
const cSrc     = findBlock(/^const C = \{/,        /^};$/);
const gatesSrc = findBlock(/^const GATES = \{/,    /^};$/);
const qSrc     = findBlock(/^const QPhysics = \(\(\) => \{/, /^\}\)\(\);$/);

const code = [piSrc, cSrc, gatesSrc, qSrc,
  'module.exports = QPhysics;'].join('\n');
const QPhysics = eval('(function(){' + code + '\nreturn module.exports;})()');

let fail = 0;
const st = QPhysics.selfTest();
if (!st.pass) { console.error('SELF-TEST FAILED'); fail++; }

console.log('\n--- Falsifiability diagnostic (decoRate=2, 20 turns) ---');
const d = QPhysics.diagnose({ decoRate: 2, turns: 20 });
console.log('R²(exponential) =', d.R2_exponential, ' R²(linear) =', d.R2_linear);
console.log('verdict         =', d.verdict);
console.log('T1 input/fit    =', d.T1_input, '/', d.T1_recovered);
console.log('charge by turn  =', d.charge_by_turn.slice(0, 8).join(', '), '...');

// Hard assertions for the critique's success criteria.
function assert(name, ok) { console.log((ok ? '  ✓ ' : '  ✗ ') + name); if (!ok) fail++; }
assert('R²(exp) > 0.99 (critique success criterion)', d.R2_exponential > 0.99);
assert('R²(exp) > R²(lin) — model is exponential, not linear', d.R2_exponential > d.R2_linear);
assert('fitted T1 recovers input T1 within 1%', Math.abs(d.T1_recovered - d.T1_input) / d.T1_input < 0.01);

// Charge must be monotonically non-increasing for a unit left in |1⟩.
let mono = true;
for (let i = 1; i < d.charge_by_turn.length; i++)
  if (d.charge_by_turn[i] > d.charge_by_turn[i - 1] + 1e-9) mono = false;
assert('charge decays monotonically toward 0', mono && d.charge_by_turn[20] < 0.05);

// Round-trip: pure→bloch→pure preserves charge for a few states.
const C0 = QPhysics; const approx = (a, b) => Math.abs(a - b) < 1e-9;
[[0, 0, 1], [0, 0, -1], [1, 0, 0], [0.3, -0.4, 0.5]].forEach(([x, y, z]) => {
  const p = C0.pureFromBloch({ x, y, z });
  const r = C0.blochFromPure(p);
  assert(`pure↔bloch preserves charge for z=${z}`, approx(C0.charge(r), C0.charge({ x, y, z })));
});

// ── QB-5: defensive interference makes relative phase mechanically real ──
// A GUARDing unit is measured in the X-basis, so its crit risk is (1−x)/2.
console.log('\n--- QB-5 defensive interference (guardCritProb) ---');
const gcp = QPhysics.guardCritProb;
const approxP = (a, b) => Math.abs(a - b) < 1e-9;
assert('|+⟩ (x=+1) guard is crit-immune  → 0%', approxP(gcp({ x: 1, y: 0, z: 0 }), 0));
assert('|−⟩ (x=−1) guard is fully exposed → 100%', approxP(gcp({ x: -1, y: 0, z: 0 }), 1));
assert('|0⟩ guard is a coin flip           → 50%', approxP(gcp({ x: 0, y: 0, z: 1 }), 0.5));
assert('|1⟩ guard is a coin flip           → 50%', approxP(gcp({ x: 0, y: 0, z: -1 }), 0.5));
// Using phase (reaching |+⟩) must be strictly safer than ignoring it (|0⟩):
assert('phase-aware |+⟩ guard beats naive |0⟩ guard',
  gcp({ x: 1, y: 0, z: 0 }) < gcp({ x: 0, y: 0, z: 1 }));

// ── QB-6: real 2-qubit Bell pair (correlated collapse + no-communication) ──
console.log('\n--- QB-6 Bell pair (entanglement) ---');
const Bell = QPhysics.Bell;

// |Φ+⟩ marginals are 50/50 on each qubit.
assert('Bell |Φ+⟩ marginal P1(q0) = 0.5', approx(Bell.marginalP1(Bell.phiPlus(), 0), 0.5));
assert('Bell |Φ+⟩ marginal P1(q1) = 0.5', approx(Bell.marginalP1(Bell.phiPlus(), 1), 0.5));

// Measuring one half forces the partner to the SAME outcome (perfect correlation).
{
  let correlated = true;
  for (const r of [0.1, 0.9, 0.49, 0.51, 0.0, 0.999]) {
    const res = Bell.measure(Bell.phiPlus(), 0, () => r);
    if (res.outcome !== res.partnerOutcome) correlated = false;
  }
  assert('measuring q0 collapses q1 to the correlated outcome', correlated);
}

// No-communication: a LOCAL gate on q0 must not change q1's marginal.
{
  const X = QPhysics.gateMatrix('X');
  const H = QPhysics.gateMatrix('H');
  const Z = QPhysics.gateMatrix('Z');
  let invariant = true;
  for (const U of [X, H, Z]) {
    const after = Bell.applyLocal(Bell.phiPlus(), 0, U);
    if (!approx(Bell.marginalP1(after, 1), 0.5)) invariant = false;
  }
  assert('local gate on q0 leaves q1 marginal unchanged (no-communication)', invariant);
}

console.log(fail === 0 ? '\nALL CHECKS PASSED ✓' : `\n${fail} CHECK(S) FAILED ✗`);
process.exit(fail === 0 ? 0 : 1);
