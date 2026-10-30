// Checks the physics and the course in web/qublitz/copter-engine.js (the Qubit Copter game).
// Usage: node web/qublitz/tests/copter-engine.test.js
'use strict';
const C = require('../copter-engine.js');
const R = require('../rabi-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ---- no precession: with no drive the state stays put ----
{
  const w = new C.World({ seed: 'still' }); w.r = R.fromAngles(1.1, 0.4); const r0 = w.r.slice();
  for (let i = 0; i < 600; i++) w.step({ x: 0, y: 0 }, 1 / 120);
  check('without a drive the Bloch vector does not move (on resonance)', R.angle(w.r, r0) < 1e-12);
}

// ---- a full drive about x for π/Ω is a π pulse ----
{
  const w = new C.World({ seed: 'pi' }); w.amp.x = 1;
  const T = Math.PI / C.P.OMEGA, n = Math.round(T * 1200);
  for (let i = 0; i < n; i++) { w.amp.x = 1; w.r = R.unit(R.rotate(w.r, w.omega(), T / n)); }
  check('a π pulse about x takes |0⟩ to |1⟩', close(w.r[2], -1, 1e-9));
}

// ---- steering: the chosen equatorial axis joins r and p in one turn ----
{
  const rnd = R.rng('steer'); let worst = 0, slowest = 0;
  for (let i = 0; i < 300; i++) {
    const r = R.fromAngles(Math.acos(2 * rnd() - 1), 2 * Math.PI * rnd()), p = R.fromAngles(Math.acos(2 * rnd() - 1), 2 * Math.PI * rnd());
    const s = C.steerToward(r, p), n = Math.hypot(s.x, s.y);
    if (n < 1e-9) continue;
    const axis = [s.x / n, s.y / n, 0];
    worst = Math.max(worst, Math.abs(R.dot(r, axis) - R.dot(p, axis)));   // r and p on one circle about the axis
    // and following it gets there
    let q = r; for (let k = 0; k < 2000; k++) { const d = C.steerToward(q, p); q = R.unit(R.rotate(q, [C.P.OMEGA * d.x, C.P.OMEGA * d.y, 0], 0.005)); }
    slowest = Math.max(slowest, R.angle(q, p));
  }
  check('the steering axis keeps r·n = p·n (one rotation joins them)', worst < 1e-9, worst.toExponential(1));
  check('following the steering reaches the point', slowest < 1 * Math.PI / 180, `${(slowest * 180 / Math.PI).toFixed(2)}° left`);
}

// ---- circuit targets are the states of the gate sequence ----
{
  const w = new C.World({ seed: 'circ', mode: 'circuit' }); let r = [0, 0, 1], bad = 0;
  for (let k = 0; k < 12; k++) { w.t += 5; w.generate(); }   // schedule a minute of the course
  for (const o of w.targets) { r = C.applyGate(r, o.gate); if (R.angle(r, o.v) > 1e-6) bad++; }
  check('each circuit target is the previous one with its gate applied, from |0⟩', bad === 0 && w.targets.length > 10, `${w.targets.length} targets`);
  check('H|0⟩ = |+⟩ and √X|0⟩ = |−i⟩', C.stateName(C.applyGate([0, 0, 1], 'H')) === '|+⟩' && C.stateName(C.applyGate([0, 0, 1], 'SX')) === '|−i⟩');
}

// ---- the course is fair: walls never cover a target, and a steering bot reaches (almost) every target in time ----
{
  let covered = 0, hits = 0, total = 0;
  for (const mode of ['targets', 'circuit']) for (const seed of ['a', 'b', 'c']) {
    const w = new C.World({ seed, mode }); w.hearts = 1e9;
    for (let i = 0; i < 180 * 120; i++) {
      const tg = w.targets[0];
      for (const wl of w.walls) for (const o of w.targets) if (o.t >= wl.t0 && o.t <= wl.t1 && o.v[wl.axis] > wl.lo && o.v[wl.axis] < wl.hi) covered++;
      for (const e of w.step(tg ? C.steerToward(w.r, tg.v) : {}, 1 / 120)) { if (e.type === 'hit') { hits++; total++; } if (e.type === 'miss') total++; }
    }
  }
  check('no wall covers a target', covered === 0, `${covered}`);
  check('a bot that points at each target reaches at least 98 % of them', hits / total >= 0.98, `${hits}/${total}`);
  const a = new C.World({ seed: 'same' }), b = new C.World({ seed: 'same' });
  check('the same seed gives the same course', JSON.stringify(a.targets.map(o => o.v)) === JSON.stringify(b.targets.map(o => o.v)));
}

// ---- Z: precession only while you hold it, and it never changes P(1) ----
{
  const w = new C.World({ mode: 'free' }); w.r = R.fromAngles(1.1, 0.4); const z0 = w.r[2], r0 = w.r.slice();
  for (let i = 0; i < 240; i++) w.step({ z: 1 }, 1 / 120);
  check('holding Z turns φ by Ω·t and leaves ⟨σz⟩ alone', close(w.r[2], z0, 1e-12) && close(R.angle(w.r, R.rotate(r0, [0, 0, C.P.OMEGA], 2)), 0, 1e-9));
  const r1 = w.r.slice(); for (let i = 0; i < 240; i++) w.step({}, 1 / 120);
  check('letting go of Z stops the precession', R.angle(w.r, r1) < 1e-12);
}

// ---- pulses are exact, whatever the frame rate ----
{
  const w = new C.World({ mode: 'free' });
  w.pulse('y', Math.PI / 2); for (let i = 0; i < 100; i++) w.step({}, 1 / 37);
  check('a π/2 pulse about y takes |0⟩ exactly to |+⟩', R.angle(w.r, [1, 0, 0]) < 1e-9 && C.stateName(w.r) === '|+⟩');
  w.pulse('z', Math.PI / 2); w.pulse('x', -Math.PI); for (let i = 0; i < 400; i++) w.step({}, 1 / 61);
  check('queued pulses run in order (|+⟩ → |+i⟩ → |−i⟩)', R.angle(w.r, [0, -1, 0]) < 1e-9);
}

// ---- every lesson can be finished with the axes it allows ----
{
  // a few pulses (π/2 or π, either sign, on the allowed axes) searched breadth-first, then played into the World
  const plan = l => {
    const goal = C.lessonVec(l.goal), moves = [];
    for (const ax of l.axes) for (const a of [Math.PI / 2, -Math.PI / 2, Math.PI, Math.PI / 4, -Math.PI / 4, Math.PI / 3, -Math.PI / 3]) moves.push([ax, a]);
    let frontier = [{ r: C.lessonVec(l.start), seq: [] }];
    for (let depth = 0; depth < 3; depth++) {
      const next = [];
      for (const f of frontier) for (const [ax, a] of moves) {
        const r = R.rotate(f.r, { x: [a, 0, 0], y: [0, a, 0], z: [0, 0, a] }[ax], 1), seq = f.seq.concat([[ax, a]]);
        if (R.angle(r, goal) < 1e-6) return seq; next.push({ r, seq });
      }
      frontier = next;
    }
    return null;
  };
  // θ = 60° is not a multiple of these angles from |0⟩ except by π/3 about y; φ = 135° is π/2 + π/4 about z
  let ok = 0;
  for (const l of C.LESSONS) {
    const seq = plan(l); if (!seq) { console.log('  no plan for', l.id); continue; }
    const w = new C.World({ mode: 'learn', lesson: l.id }); let done = false;
    for (const [ax, a] of seq) w.pulse(ax, a);
    for (let i = 0; i < 1200 && !done; i++) for (const e of w.step({}, 1 / 120)) if (e.type === 'goal') done = true;
    if (done) ok++; else console.log('  lesson not finished:', l.id);
  }
  check('each lesson is finished by pulses on its own axes', ok === C.LESSONS.length, `${ok}/${C.LESSONS.length}`);
  const w = new C.World({ mode: 'learn', lesson: 'flip' }); let early = false;
  for (let i = 0; i < 120; i++) { for (const e of w.step({ x: 1 }, 1 / 120)) if (e.type === 'goal') early = true; }
  check('a lesson does not finish while you are still driving through the goal', !early);
}

// ---- the Virtual Qubit Lab replays a run: same P(1), same state up to the leftover virtual Z ----
// (at a 50 MHz Rabi rate on a 5 GHz qubit; the lab keeps the counter-rotating terms, so much faster drives drift by degrees)
{
  const Q = require('../qubit-engine.js');
  const w = new C.World({ mode: 'free' }), dt = 1 / 120;
  const hold = (inp, secs) => { for (let i = 0; i < Math.round(secs / dt); i++) w.step(inp, dt); };
  hold({ y: 1 }, 0.6); hold({ z: 1 }, 0.5); hold({ x: -1 }, 0.4); hold({}, 0.5); hold({ x: 0.6, y: 0.8 }, 0.5); hold({ z: -1 }, 0.3);
  hold({ x: 1, z: 1 }, 0.4); w.pulse('y', Math.PI / 2); hold({}, 1.2);
  const seq = C.labSequence(w.log, { fGHz: 0.05 }), tF = seq.end + 2, n = Math.ceil(tF * 100);
  const I = new Float64Array(n), Qe = new Float64Array(n), t = Array.from({ length: n }, (_, i) => i * tF / (n - 1));
  for (const p of seq.pulses) for (let i = 0; i < n; i++) if (t[i] >= p.start && t[i] < p.start + p.dur) { I[i] = p.sx; Qe[i] = p.sy; }
  const r = Q.runQuantumSimulation({ omegaQ: 5, omegaRabi: 0.05, tFinal: tF, nSteps: n, omegaD: 5, envI: I, envQ: Qe, T1: 1e7, T2: 2e7 }, 0);
  const rot = Q.rotatingFrame(r.t, r.x, r.y, 5), k = n - 1;
  // the lab draws its rotating frame with x and y swapped and |0⟩ at z = −1 (a proper rotation of the game's axes)
  const lab = [rot.y[k], rot.x[k], -r.z[k]], want = R.rotate(w.r, [0, 0, -seq.phi], 1);
  check('the lab\'s P(1) matches the game', close(r.p1[k], (1 - w.r[2]) / 2, 0.01), `${r.p1[k].toFixed(3)} vs ${((1 - w.r[2]) / 2).toFixed(3)}`);
  check('the lab\'s state is the game\'s turned back by the virtual Z', R.angle(lab, want) < 1.5 * Math.PI / 180, `${(R.angle(lab, want) * 180 / Math.PI).toFixed(2)}°`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
