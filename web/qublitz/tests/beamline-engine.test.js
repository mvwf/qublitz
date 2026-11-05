// Checks the physics and the levels in web/qublitz/beamline-engine.js (the Beamline puzzle game).
// Usage: node web/qublitz/tests/beamline-engine.test.js
'use strict';
const E = require('../beamline-engine.js');
const R = require('../rabi-engine.js');
const C = require('../copter-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const near = (a, b, tol = 1e-12) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

// A one-row board: laser at the left, the given pieces, then a qubit.
const strip = (pieces, start = '|0⟩') => ({ w: 3 + pieces.length, h: 1, lasers: [{ x: 0, y: 0, dir: 0 }],
  qubits: [{ x: 1 + pieces.length, y: 0, start, target: '|0⟩' }] });
const onStrip = (types, start) => {
  const L = strip(types, start);
  return E.run(L, types.map((type, i) => ({ x: 1 + i, y: 0, type }))).states[0];
};

// ---- rotation convention: the same right-handed turn as the other qublitz engines ----
{
  const rnd = R.rng('beamline'); let worst = 0;
  for (let i = 0; i < 200; i++) {
    const r = R.fromAngles(Math.acos(2 * rnd() - 1), 2 * Math.PI * rnd()), ph = 2 * Math.PI * rnd(), th = 4 * rnd();
    const n = E.axisOf(ph), a = E.rotate(r, n, th), b = R.rotate(r, n.map(c => c * th), 1);
    worst = Math.max(worst, ...a.map((v, k) => Math.abs(v - b[k])));
  }
  check('rotate() matches RabiEngine.rotate (right-handed, |0⟩ at +z)', worst < 1e-12, `worst ${worst.toExponential(1)}`);
}

// ---- the bare laser is an X gate; the pieces make √X, √Y, and so on, as in Qubit Copter ----
{
  check('a bare laser pulse is the X gate: |0⟩ → |1⟩', near(onStrip([]), C.applyGate([0, 0, 1], 'X'), 1e-12));
  check('attenuator: half the area, a √X: |0⟩ → |−i⟩', near(onStrip(['atten']), C.applyGate([0, 0, 1], 'SX'), 1e-12) && near(onStrip(['atten']), E.KETS['|−i⟩'], 1e-12));
  check('phase plate + attenuator: an x rotation becomes a y rotation, √Y: |0⟩ → |+⟩', near(onStrip(['phase', 'atten']), C.applyGate([0, 0, 1], 'SY'), 1e-12));
  const L = strip(['atten', 'atten']), path = E.trace(L, [{ x: 1, y: 0, type: 'atten' }, { x: 2, y: 0, type: 'atten' }])[0];
  check('each attenuator halves the area (π → π/2 → π/4)', path.hits[0].area === 0.25);
  const ph4 = onStrip(['phase', 'phase', 'phase', 'phase', 'atten']);
  check('four phase plates are a full period: back to an x rotation', near(ph4, onStrip(['atten']), 1e-12));
  const rev = onStrip(['phase', 'phase', 'atten']);
  check('two phase plates reverse the sense: |0⟩ → |+i⟩ (opposite of |−i⟩)', near(rev, E.KETS['|+i⟩'], 1e-12));
}

// ---- 2π returns the Bloch vector; X and Y turns do not commute ----
{
  const r = [0.3, -0.5, Math.sqrt(1 - 0.34)];
  const twice = E.rotate(E.rotate(r, E.axisOf(0), Math.PI), E.axisOf(0), Math.PI);
  check('two π pulses (2π) return the Bloch vector', near(twice, r, 1e-12));
  const q = Math.PI / 4, xy = E.rotate(E.rotate([0, 0, 1], E.axisOf(0), q), E.axisOf(Math.PI / 2), q),
    yx = E.rotate(E.rotate([0, 0, 1], E.axisOf(Math.PI / 2), q), E.axisOf(0), q);
  check('X(π/4) then Y(π/4) differs from Y(π/4) then X(π/4)', E.dot(xy, yx) < 0.99, `xy=${xy.map(v => v.toFixed(3))} yx=${yx.map(v => v.toFixed(3))}`);
  check('…and X then Y lands at (½, −1/√2, ½), the "Order matters" goal', near(xy, [0.5, -Math.SQRT1_2, 0.5], 1e-12));
}

// ---- mirrors ----
{
  // reversed outgoing beam (out + 2) must reflect into the reversed incoming beam (d + 2)
  const ok = ['/', '\\'].every(o => [0, 1, 2, 3].every(d => E.REFLECT[o][(E.REFLECT[o][d] + 2) % 4] === (d + 2) % 4));
  check('mirrors are reversible: a beam sent back retraces its path', ok);
  check('"/" sends a rightward beam up, "\\" sends it down', E.REFLECT['/'][0] === 3 && E.REFLECT['\\'][0] === 1);
  const box = { w: 4, h: 4, lasers: [{ x: 0, y: 1, dir: 0 }], qubits: [] };
  const loop = E.trace(box, [{ x: 2, y: 1, type: 'mirror', o: '/' }, { x: 2, y: 0, type: 'mirror', o: '\\' }, { x: 1, y: 0, type: 'mirror', o: '/' }, { x: 1, y: 1, type: 'mirror', o: '\\' }])[0];
  check('a beam that would circle forever is stopped', ['loop', 'laser', 'edge'].includes(loop.end) && loop.cells.length < 400, `end=${loop.end}`);
}

// ---- the lab pulses: area = amplitude × length, so an attenuated pulse keeps its length ----
{
  const L = strip(['atten']), r = E.run(L, [{ x: 1, y: 0, type: 'atten' }]);
  const full = E.labPulses(E.run(strip([]), []).history, 0)[0], half = E.labPulses(r.history, 0)[0];
  check('lab pulses: an attenuated pulse has half the amplitude and the same length', Math.abs(half.sx - 0.5) < 1e-12 && half.dur === full.dur);
  check('lab pulses: a full pulse at Ω/2π = 50 MHz lasts 10 ns (a π pulse)', Math.abs(full.dur - 10) < 1e-9);
}

// ---- every level: in bounds, uses its own inventory, is solved by its stored solution, and par is the true minimum ----
function minPieces(L, maxK) {
  const occupied = new Set([...(L.walls || []).map(([x, y]) => x + ',' + y), ...L.qubits.map(q => q.x + ',' + q.y), ...L.lasers.map(l => l.x + ',' + l.y)]);
  let best = Infinity;
  // Every piece of a solution lies on its final beam path, so placing pieces in path order, each next piece is on
  // the path the previous ones make: searching the current path only is exhaustive.
  function rec(placed, inv, k) {
    const r = E.run(L, placed);
    if (r.solved) { best = Math.min(best, placed.length); return; }
    if (placed.length >= k) return;
    const used = new Set(placed.map(p => p.x + ',' + p.y)), cells = new Set();
    for (const p of r.paths) for (const c of p.cells) { const key = c.x + ',' + c.y; if (!occupied.has(key) && !used.has(key)) cells.add(key); }
    for (const key of cells) {
      const [x, y] = key.split(',').map(Number);
      for (const type of Object.keys(inv)) {
        if (inv[type] <= 0) continue;
        for (const o of type === 'mirror' ? ['/', '\\'] : [undefined]) {
          placed.push({ x, y, type, o }); inv[type]--; rec(placed, inv, k); inv[type]++; placed.pop();
        }
      }
    }
  }
  for (let k = 0; k <= maxK && best === Infinity; k++) rec([], { ...L.inv }, k);
  return best;
}
for (const L of E.LEVELS) {
  const occupied = new Set([...(L.walls || []).map(([x, y]) => x + ',' + y), ...L.qubits.map(q => q.x + ',' + q.y), ...L.lasers.map(l => l.x + ',' + l.y)]);
  const inBounds = L.solution.every(p => p.x >= 0 && p.y >= 0 && p.x < L.w && p.y < L.h && !occupied.has(p.x + ',' + p.y));
  const used = {}; for (const p of L.solution) used[p.type] = (used[p.type] || 0) + 1;
  const invOk = Object.entries(used).every(([k, n]) => (L.inv[k] || 0) >= n);
  const r = E.run(L, L.solution), empty = E.run(L, []);
  check(`level "${L.title}": stored solution reaches every target at F ≥ 0.99`, inBounds && invOk && r.solved && r.fid.every(f => f >= 0.99),
    `F = ${r.fid.map(f => f.toFixed(4)).join(', ')}`);
  check(`level "${L.title}": not already solved with no pieces`, !empty.solved);
  const m = minPieces(L, L.solution.length);
  check(`level "${L.title}": par ${L.par.pieces} piece${L.par.pieces > 1 ? 's' : ''} is the fewest that solve it`, m === L.par.pieces, `min ${m}`);
}
check('there are 10 to 12 levels with unique ids', E.LEVELS.length >= 10 && E.LEVELS.length <= 12 && new Set(E.LEVELS.map(l => l.id)).size === E.LEVELS.length);

// ---- stars ----
{
  const L = E.LEVELS[0];
  check('stars: ★★★ at par, ★★ with a hint or extra fires, ★ over par pieces',
    E.stars(L, 1, 1, false) === 3 && E.stars(L, 1, 2, false) === 2 && E.stars(L, 1, 1, true) === 2 && E.stars(L, 2, 1, false) === 1);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
