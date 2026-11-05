/* beamline-engine.js: physics, beam tracing and levels for Beamline (beamline.html). No dependencies.
 * Loads as a classic <script> (window.BeamlineEngine) or from Node (require), so
 * tests/beamline-engine.test.js checks the same code the game runs.
 *
 * The board is a grid. A laser (a resonant drive pulse) leaves its source in a straight line. Everything is in
 * the frame rotating with the drive, exactly on resonance, so there is no free precession: a qubit's Bloch vector
 * moves only while the beam is passing through it. A pulse with drive phase φ and area θ turns the qubit by θ
 * about the equatorial axis n = (cos φ, sin φ, 0), right-handed, with |0⟩ at +z:
 *
 *   H = (ħΩ/2)(cos φ σx + sin φ σy),   θ = ∫Ω dt,   r → R_n(θ) r
 *
 * the same convention as copter-engine.js and rabi-engine.js (asserted in the test). The pieces change the beam,
 * not the qubits:
 *
 *   mirror (/ or \)  turns the beam by 90°. The pulse itself is unchanged (an ideal metal mirror).
 *   phase plate      delays the drive by a quarter period: φ → φ + 90°, so an x rotation becomes a y rotation.
 *   attenuator       −6 dB: halves the field amplitude Ω, so the same pulse length carries half the area θ.
 *
 * Simplifications, stated on the page too: the beam does not lose strength or spread as it travels or passes a
 * qubit (each qubit takes a negligible share of the photons), the pulse is short enough that no qubit decays
 * while the beam is out (no T₁/T₂ here), and lasers fire one after another in the order shown on the board.
 * Only one beam can be in a cell direction at a time; there is no beam splitter (a passive 50/50 splitter would
 * send θ/√2 down each branch, which no puzzle here needs).
 */
(function (root) {
  'use strict';
  const D2R = Math.PI / 180;
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  // Right-handed rotation of r by angle θ about the unit axis n (Rodrigues).
  function rotate(r, n, th) {
    const c = Math.cos(th), s = Math.sin(th), nxr = cross(n, r), nr = dot(n, r);
    return [0, 1, 2].map(i => r[i] * c + nxr[i] * s + n[i] * nr * (1 - c));
  }
  const axisOf = phase => [Math.cos(phase), Math.sin(phase), 0];
  // State fidelity |⟨t|ψ⟩|² between pure states, from their Bloch vectors.
  const fidelity = (r, t) => (1 + dot(r, t)) / 2;

  const KETS = { '|0⟩': [0, 0, 1], '|1⟩': [0, 0, -1], '|+⟩': [1, 0, 0], '|−⟩': [-1, 0, 0], '|+i⟩': [0, 1, 0], '|−i⟩': [0, -1, 0] };
  const fromAngles = (th, ph) => [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
  // A state is a ket name, [θ°, φ°], or { v: [x, y, z] } (a Bloch vector, normalised here).
  const vec = s => typeof s === 'string' ? KETS[s].slice() : Array.isArray(s) ? fromAngles(s[0] * D2R, s[1] * D2R)
    : (m => s.v.map(c => c / m))(Math.hypot(...s.v));
  function stateName(r) {
    for (const [n, v] of Object.entries(KETS)) if (dot(r, v) > 0.9995) return n;
    const th = Math.acos(Math.max(-1, Math.min(1, r[2])));
    return `θ ${(th / D2R).toFixed(0)}°, φ ${((Math.atan2(r[1], r[0]) / D2R + 360) % 360).toFixed(0)}°`;
  }

  // Directions: 0 right, 1 down, 2 left, 3 up (screen coordinates, y grows down).
  const DX = [1, 0, -1, 0], DY = [0, 1, 0, -1];
  // '/' sends right→up, up→right, left→down, down→left; '\' sends right→down, down→right, left→up, up→left.
  const REFLECT = { '/': [3, 2, 1, 0], '\\': [1, 0, 3, 2] };
  const PHASE_NAMES = ['x', 'y', '−x', '−y'];   // drive phase in quarter turns
  const phaseName = q => PHASE_NAMES[((q % 4) + 4) % 4];

  const PIECES = {
    mirror: { label: 'Mirror', glyph: '/', what: 'Turns the beam 90°. Click a placed mirror to flip it between / and \\.' },
    phase: { label: 'Phase plate', glyph: '+90°', what: 'Shifts the drive phase by 90°: an x rotation becomes a y rotation (y → −x, and so on).' },
    atten: { label: 'Attenuator', glyph: '½', what: '−6 dB: halves the field Ω, so the pulse area halves (π → π/2 → π/4).' },
  };

  /* Trace every laser through the board. level: { w, h, lasers, qubits, walls, fixed }, placed: [{ x, y, type, o }]
   * (o is '/' or '\\' for mirrors). Returns one path per laser, in firing order, each a list of cells the beam
   * crosses with the pulse it carries there (phase in quarter turns, area in units of π), and the hits on qubits. */
  function trace(level, placed) {
    const cell = new Map(), key = (x, y) => x + ',' + y;
    for (const [x, y] of level.walls || []) cell.set(key(x, y), { kind: 'wall' });
    for (const p of level.fixed || []) cell.set(key(p.x, p.y), { kind: 'piece', type: p.type, o: p.o, fixed: true });
    for (const p of placed || []) cell.set(key(p.x, p.y), { kind: 'piece', type: p.type, o: p.o });
    (level.qubits || []).forEach((q, i) => cell.set(key(q.x, q.y), { kind: 'qubit', i }));
    (level.lasers || []).forEach((l, i) => cell.set(key(l.x, l.y), { kind: 'laser', i }));
    const order = (level.lasers || []).map((l, i) => i).sort((a, b) => (level.lasers[a].order || 0) - (level.lasers[b].order || 0));
    const paths = [];
    for (const li of order) {
      const L = level.lasers[li];
      let x = L.x, y = L.y, d = L.dir, phase = L.phase || 0, area = L.area == null ? 1 : L.area;
      const cells = [], hits = [], seen = new Set();
      let end = 'edge';
      for (let step = 0; step < 400; step++) {
        x += DX[d]; y += DY[d];
        if (x < 0 || y < 0 || x >= level.w || y >= level.h) { end = 'edge'; break; }
        const sig = `${x},${y},${d},${phase},${area}`;
        if (seen.has(sig)) { end = 'loop'; break; }
        seen.add(sig);
        const c = cell.get(key(x, y));
        const inDir = d;
        if (c && c.kind === 'wall') { cells.push({ x, y, dIn: inDir, dOut: null, phase, area }); end = 'wall'; break; }
        if (c && c.kind === 'laser') { cells.push({ x, y, dIn: inDir, dOut: null, phase, area }); end = 'laser'; break; }
        if (c && c.kind === 'qubit') hits.push({ qubit: c.i, phase, area, at: cells.length });
        if (c && c.kind === 'piece') {
          if (c.type === 'mirror') d = REFLECT[c.o][d];
          else if (c.type === 'phase') phase = (phase + 1) % 4;
          else if (c.type === 'atten') area = area / 2;
        }
        cells.push({ x, y, dIn: inDir, dOut: d, phase, area });
      }
      paths.push({ laser: li, cells, hits, end });
    }
    return paths;
  }

  // Run the pulses: every qubit starts in its start state and takes each hit in order (lasers fire one after
  // another). Returns the final states, the step-by-step history for animation, and the fidelity with each target.
  function run(level, placed) {
    const paths = trace(level, placed);
    const states = (level.qubits || []).map(q => vec(q.start || '|0⟩'));
    const history = [];   // { laser, qubit, phase, area, from, to }
    for (const p of paths) for (const h of p.hits) {
      const from = states[h.qubit], to = rotate(from, axisOf(h.phase * Math.PI / 2), h.area * Math.PI);
      states[h.qubit] = to;
      history.push({ laser: p.laser, qubit: h.qubit, phase: h.phase, area: h.area, from, to, at: h.at });
    }
    const fid = (level.qubits || []).map((q, i) => fidelity(states[i], vec(q.target || q.start || '|0⟩')));
    const solved = fid.length > 0 && fid.every(f => f >= WIN);
    return { paths, states, history, fid, solved };
  }
  const WIN = 0.99;

  // Stars, as in Rabi Rider: ★ solved, ★★ with at most par pieces, ★★★ also at most par fires (and no hint).
  function stars(level, pieces, fires, hinted) {
    const par = level.par || { pieces: 99, fires: 99 };
    if (pieces > par.pieces) return 1;
    return fires <= par.fires && !hinted ? 3 : 2;
  }

  // The Virtual Qubit Lab replays one qubit's hits as square pulses on its σx/σy channels: phase sets the channel
  // mix, and an attenuated pulse keeps its length but has a smaller amplitude (area = Ω·t), exactly as on the board.
  function labPulses(history, qubit, opts = {}) {
    const f = opts.fGHz ?? 0.05, rate = 2 * Math.PI * f, dur = Math.PI / rate;   // full-area pulse = a π pulse
    let t = opts.start ?? 5; const out = [];
    for (const h of history) {
      if (h.qubit !== qubit) continue;
      const a = h.area, ph = h.phase * Math.PI / 2;
      out.push({ sx: a * Math.cos(ph), sy: a * Math.sin(ph), start: t, dur });
      t += dur + 2;
    }
    return out;
  }

  /* Levels. Coordinates are (column, row) from the top left. A laser sits on a board cell and fires in dir; its
   * pulse starts as an x drive (phase 0) of area π unless the level says otherwise. Each level teaches one idea;
   * solution is a placement that solves it (checked by the test), par is the fewest pieces and fires for ★★★. */
  const LEVELS = [
    { id: 'flip', title: 'Flip the bit', w: 6, h: 4,
      lasers: [{ x: 0, y: 1, dir: 0 }], qubits: [{ x: 3, y: 3, start: '|0⟩', target: '|1⟩' }],
      inv: { mirror: 1 }, par: { pieces: 1, fires: 1 },
      solution: [{ x: 3, y: 1, type: 'mirror', o: '\\' }],
      brief: 'The laser fires a <b>π pulse</b> about x. The qubit sits off the beam\'s line: place the <b>mirror</b> so the beam passes through it, then press <b>Fire</b>.',
      learn: 'A resonant π pulse turns the Bloch vector half a turn about x, from |0⟩ to |1⟩: the quantum NOT gate. The mirror only redirected the drive; the pulse was the same.' },
    { id: 'corner', title: 'Around the wall', w: 7, h: 5,
      lasers: [{ x: 0, y: 0, dir: 0 }], qubits: [{ x: 5, y: 4, start: '|0⟩', target: '|1⟩' }],
      walls: [[3, 0], [3, 1], [4, 3], [5, 3], [6, 3]], inv: { mirror: 2 }, par: { pieces: 2, fires: 1 },
      solution: [{ x: 2, y: 0, type: 'mirror', o: '\\' }, { x: 2, y: 4, type: 'mirror', o: '\\' }],
      brief: 'Walls absorb the beam. Use two mirrors to bring the pulse to the qubit in the corner. The dashed line shows where the beam will go.',
      learn: 'However the beam is routed, the qubit feels the same π pulse. In a real chip the drive comes down a wire, and only its phase and area matter to the qubit, not the route.' },
    { id: 'half', title: 'Half a pulse', w: 6, h: 4,
      lasers: [{ x: 0, y: 1, dir: 0 }], qubits: [{ x: 4, y: 1, start: '|0⟩', target: '|−i⟩' }],
      inv: { atten: 1 }, par: { pieces: 1, fires: 1 },
      solution: [{ x: 2, y: 1, type: 'atten' }],
      brief: 'This time the goal is on the equator: |−i⟩. An <b>attenuator</b> halves the field Ω. What does that do to a pulse of the same length?',
      learn: 'The rotation angle is the pulse area θ = Ω·t. Half the field for the same time is half the area: a π/2 pulse, which takes |0⟩ to |−i⟩ = (|0⟩ − i|1⟩)/√2. P(1) is now ½.' },
    { id: 'phase', title: 'The drive phase', w: 6, h: 4,
      lasers: [{ x: 0, y: 2, dir: 0 }], qubits: [{ x: 5, y: 2, start: '|0⟩', target: '|+⟩' }],
      inv: { atten: 1, phase: 1 }, par: { pieces: 2, fires: 1 },
      solution: [{ x: 2, y: 2, type: 'atten' }, { x: 3, y: 2, type: 'phase' }],
      brief: 'Same P(1) = ½, different state: |+⟩. A <b>phase plate</b> delays the drive by a quarter period. Which axis does the qubit turn about then?',
      learn: 'The drive phase sets the rotation axis: phase 0 turns about x, a quarter period later about y. A π/2 pulse about y takes |0⟩ to |+⟩, while about x it gave |−i⟩. Same P(1), different relative phase.' },
    { id: 'spectator', title: 'Leave it alone', w: 7, h: 4,
      lasers: [{ x: 0, y: 1, dir: 0 }],
      qubits: [{ x: 3, y: 1, start: '|0⟩', target: '|0⟩', spectator: true }, { x: 5, y: 1, start: '|0⟩', target: '|1⟩' }],
      inv: { mirror: 4 }, par: { pieces: 3, fires: 1 },
      solution: [{ x: 1, y: 1, type: 'mirror', o: '/' }, { x: 1, y: 0, type: 'mirror', o: '/' }, { x: 5, y: 0, type: 'mirror', o: '\\' }],
      brief: 'The grey qubit is a <b>spectator</b>: it must stay in |0⟩. Route the pulse around it to flip the other one.',
      learn: 'On a real chip a drive meant for one qubit can leak onto its neighbours (crosstalk). Addressing each qubit separately, here by routing the beam, keeps the spectator untouched.' },
    { id: 'chain', title: 'One beam, two qubits', w: 7, h: 3,
      lasers: [{ x: 0, y: 1, dir: 0 }],
      qubits: [{ x: 2, y: 1, start: '|0⟩', target: '|1⟩' }, { x: 5, y: 1, start: '|0⟩', target: '|−i⟩' }],
      inv: { atten: 1 }, par: { pieces: 1, fires: 1 },
      solution: [{ x: 3, y: 1, type: 'atten' }],
      brief: 'One beam passes both qubits. The first needs a full π pulse, the second only π/2.',
      learn: 'A piece changes the pulse for everything after it on the beam, not before. Order along the beamline matters, just as gate order in a circuit does.' },
    { id: 'reverse', title: 'Turn the other way', w: 6, h: 4,
      lasers: [{ x: 0, y: 1, dir: 0 }], qubits: [{ x: 5, y: 1, start: '|0⟩', target: '|+i⟩' }],
      inv: { atten: 1, phase: 2 }, par: { pieces: 3, fires: 1 },
      solution: [{ x: 1, y: 1, type: 'atten' }, { x: 2, y: 1, type: 'phase' }, { x: 3, y: 1, type: 'phase' }],
      brief: 'A π/2 pulse about x takes |0⟩ to |−i⟩. The goal is the opposite point, |+i⟩. Two phase plates make half a period of delay.',
      learn: 'Shifting the drive phase by 180° flips the axis to −x, so the same pulse turns the other way. Rotations about +x and −x by π/2 undo each other.' },
    { id: 'twopi', title: 'Full circle', w: 6, h: 5,
      lasers: [{ x: 0, y: 1, dir: 0 }],
      qubits: [{ x: 2, y: 1, start: '|0⟩', target: '|0⟩', spectator: true }, { x: 2, y: 4, start: '|0⟩', target: '|1⟩' }],
      walls: [[1, 0], [1, 2], [1, 3], [1, 4], [0, 4], [3, 4], [4, 4], [5, 4], [3, 2], [3, 3]], inv: { mirror: 3 }, par: { pieces: 3, fires: 1 },
      solution: [{ x: 4, y: 1, type: 'mirror', o: '/' }, { x: 4, y: 0, type: 'mirror', o: '\\' }, { x: 2, y: 0, type: 'mirror', o: '/' }],
      brief: 'The spectator blocks the only way down. Every path to the target crosses it. Can a second pass help?',
      learn: 'Two π pulses about x make a 2π rotation, which returns the Bloch vector to where it started. (The state picks up an overall sign, −|0⟩, which no single-qubit measurement can see.)' },
    { id: 'order', title: 'Order matters', w: 7, h: 5,
      lasers: [{ x: 0, y: 2, dir: 0 }], qubits: [{ x: 3, y: 2, start: '|0⟩', target: { v: [0.5, -Math.SQRT1_2, 0.5] } }],
      inv: { mirror: 3, phase: 1, atten: 2 }, par: { pieces: 6, fires: 2 },
      solution: [{ x: 1, y: 2, type: 'atten' }, { x: 2, y: 2, type: 'atten' }, { x: 4, y: 2, type: 'phase' },
        { x: 5, y: 2, type: 'mirror', o: '/' }, { x: 5, y: 0, type: 'mirror', o: '\\' }, { x: 3, y: 0, type: 'mirror', o: '/' }],
      brief: 'The gold mark is off every axis: no turn about x alone or y alone gets there. Send the beam through the qubit twice: a π/4 turn about x, then a π/4 turn about y.',
      learn: 'x then y ended at the gold mark. y then x would end somewhere else, at (0.71, −0.5, 0.5) instead of (0.5, −0.71, 0.5). Rotations about different axes do not commute, which is why gate order in a circuit matters.' },
    { id: 'pair', title: 'Two lasers', w: 7, h: 5,
      lasers: [{ x: 0, y: 1, dir: 0, order: 0 }, { x: 3, y: 4, dir: 3, order: 1, phase: 1, area: 0.5 }],
      qubits: [{ x: 3, y: 1, start: '|0⟩', target: '|1⟩' }, { x: 5, y: 2, start: '|0⟩', target: '|+⟩' }],
      inv: { mirror: 2 }, par: { pieces: 1, fires: 1 },
      solution: [{ x: 3, y: 2, type: 'mirror', o: '/' }],
      brief: 'Laser 1 fires a π pulse about x, then laser 2 fires a π/2 pulse about y. Make each qubit get the pulse it needs, and nothing else.',
      learn: 'Real processors drive many qubits with separate lines, each with its own phase and area, timed one after another. Here laser 1 flipped the left qubit and laser 2 made |+⟩ on the right.' },
    { id: 'eighth', title: 'A quarter of a quarter', w: 7, h: 3,
      lasers: [{ x: 0, y: 1, dir: 0 }], qubits: [{ x: 6, y: 1, start: '|0⟩', target: [45, 270] }],
      inv: { atten: 2 }, par: { pieces: 2, fires: 1 },
      solution: [{ x: 2, y: 1, type: 'atten' }, { x: 4, y: 1, type: 'atten' }],
      brief: 'The goal is a quarter of the way down: θ = 45°. Each attenuator halves the area.',
      learn: 'Two attenuators make π/4. P(1) = sin²(θ/2) = sin²(22.5°) ≈ 15 %: the probability is not linear in the pulse area. That is the Rabi formula.' },
    { id: 'final', title: 'Three targets', w: 8, h: 5,
      lasers: [{ x: 0, y: 0, dir: 0 }],
      qubits: [{ x: 2, y: 0, start: '|0⟩', target: '|1⟩' }, { x: 6, y: 2, start: '|0⟩', target: '|+⟩' }, { x: 6, y: 0, start: '|0⟩', target: '|0⟩', spectator: true }],
      walls: [[7, 2]], inv: { mirror: 2, atten: 1, phase: 1 }, par: { pieces: 4, fires: 2 },
      solution: [{ x: 3, y: 0, type: 'atten' }, { x: 4, y: 0, type: 'mirror', o: '\\' }, { x: 4, y: 2, type: 'mirror', o: '\\' }, { x: 5, y: 2, type: 'phase' }],
      brief: 'Flip the first qubit, make |+⟩ on the second, and keep the spectator at |0⟩. Plan the beamline before you fire.',
      learn: 'You built a small pulse schedule: an X gate on one qubit, a √Y on another, and nothing on the third. Compilers for real quantum computers solve this same routing-and-ordering problem.' },
  ];

  const api = { D2R, dot, cross, rotate, axisOf, fidelity, KETS, vec, fromAngles, stateName, DX, DY, REFLECT, PIECES, phaseName,
    trace, run, WIN, stars, labPulses, LEVELS };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.BeamlineEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
