/* qcirc-engine.js: a small state-vector simulator and the error-correction circuits shown in ErrorSweeper's
 * Learn → Circuits chapter. No dependencies. Loads as a classic <script> (window.QCirc) or from Node (require), so
 * tests/qcirc-engine.test.js checks the circuits the page draws.
 *
 * The circuits follow S. M. Girvin, "Introduction to quantum error correction and fault tolerance", SciPost Phys. Lect.
 * Notes 70 (2023), arXiv:2111.08894: the 3-qubit encoder (Fig. 16), joint-parity measurement with two CNOTs onto an
 * ancilla (Box 5) and the syndrome table (Table 3), and error correction without measurement, with Toffoli gates and an
 * ancilla reset to a cold bath (Fig. 17). The phase-flip code uses his observation that a Hadamard swaps X and Z; the
 * stabilizer-check circuits are the standard ones (Nielsen & Chuang §10.5.8; Fowler et al. 2012); Shor's encoder is
 * from Shor (1995).
 *
 * State: amplitudes re[i] + i·im[i] over 2^n basis states; qubit q is bit q of the index. Rotations use the
 * convention RX(θ) = exp(−iθX/2), so a rotation by θ flips the qubit with probability sin²(θ/2). (Girvin writes
 * exp(−iθX), with probability sin²θ: the same physics with θ halved.)
 */
(function (root) {
  'use strict';
  function State(n) { this.n = n; this.N = 1 << n; this.re = new Float64Array(this.N); this.im = new Float64Array(this.N); this.re[0] = 1; this.bits = {}; }
  State.prototype.clone = function () { const s = new State(this.n); s.re.set(this.re); s.im.set(this.im); s.bits = Object.assign({}, this.bits); return s; };
  const bit = (i, q) => (i >> q) & 1;
  // single-qubit gate [[a, b], [c, d]] with complex entries given as [re, im]
  function apply1(s, q, a, b, c, d) {
    const m = 1 << q;
    for (let i = 0; i < s.N; i++) if (!(i & m)) {
      const j = i | m, xr = s.re[i], xi = s.im[i], yr = s.re[j], yi = s.im[j];
      s.re[i] = a[0] * xr - a[1] * xi + b[0] * yr - b[1] * yi; s.im[i] = a[0] * xi + a[1] * xr + b[0] * yi + b[1] * yr;
      s.re[j] = c[0] * xr - c[1] * xi + d[0] * yr - d[1] * yi; s.im[j] = c[0] * xi + c[1] * xr + d[0] * yi + d[1] * yr;
    }
  }
  const R2 = Math.SQRT1_2;
  const G1 = {
    X: [[0, 0], [1, 0], [1, 0], [0, 0]], Z: [[1, 0], [0, 0], [0, 0], [-1, 0]], Y: [[0, 0], [0, -1], [0, 1], [0, 0]],
    H: [[R2, 0], [R2, 0], [R2, 0], [-R2, 0]],
  };
  const rx = th => [[Math.cos(th / 2), 0], [0, -Math.sin(th / 2)], [0, -Math.sin(th / 2)], [Math.cos(th / 2), 0]];
  const rz = th => [[Math.cos(th / 2), -Math.sin(th / 2)], [0, 0], [0, 0], [Math.cos(th / 2), Math.sin(th / 2)]];
  function gate(s, name, q) { const g = G1[name]; apply1(s, q, g[0], g[1], g[2], g[3]); }
  // X on t when every control is 1
  function cx(s, cs, t) {
    const cm = cs.reduce((m, c) => m | (1 << c), 0), tm = 1 << t;
    for (let i = 0; i < s.N; i++) if ((i & cm) === cm && !(i & tm)) { const j = i | tm; let r = s.re[i]; s.re[i] = s.re[j]; s.re[j] = r; r = s.im[i]; s.im[i] = s.im[j]; s.im[j] = r; }
  }
  const prob1 = (s, q) => { let p = 0; for (let i = 0; i < s.N; i++) if (bit(i, q)) p += s.re[i] * s.re[i] + s.im[i] * s.im[i]; return p; };
  // projective Z measurement: Born-rule outcome, collapse and renormalize
  function measure(s, q, rng) {
    const p1 = prob1(s, q), out = (rng || Math.random)() < p1 ? 1 : 0, keep = out ? p1 : 1 - p1, k = 1 / Math.sqrt(Math.max(keep, 1e-300));
    for (let i = 0; i < s.N; i++) { if (bit(i, q) !== out) { s.re[i] = 0; s.im[i] = 0; } else { s.re[i] *= k; s.im[i] *= k; } }
    return { out, p1 };
  }

  // ---- circuits ----
  // op kinds: { g: 'H'|'X'|'Z', q } · { g: 'CX', c: [..], t } (one control = CNOT, two = Toffoli) · { g: 'ERR' } (the
  // error the reader picks) · { g: 'M', q, b } (measure into classical bit b) · { g: 'FIX', b: [b1, b2], p: 'X'|'Z',
  // table: { '10': q, ... } } (a correction chosen from the bits) · { g: 'RESET', q } (cold-bath reset to |0⟩).
  // Each op has say: the sentence the page shows when it runs. prep: the input state, as ops before the circuit.
  const enc = [{ g: 'CX', c: [0], t: 1 }, { g: 'CX', c: [0], t: 2 }];
  const bitParity = [{ g: 'CX', c: [0], t: 3, say: 'CNOT from qubit 1 onto ancilla a₁: a₁ picks up q₁’s value…' },
    { g: 'CX', c: [1], t: 3, say: '…and q₂’s, so now a₁ = q₁ ⊕ q₂: the parity Z₁Z₂. Both |000⟩ and |111⟩ give 0, so a₁ carries no information about α or β.' },
    { g: 'CX', c: [1], t: 4, say: 'The same for a₂: first q₂…' },
    { g: 'CX', c: [2], t: 4, say: '…then q₃, so a₂ = q₂ ⊕ q₃: the parity Z₂Z₃.' }];
  const SYN = { '10': 0, '11': 1, '01': 2 };   // Girvin's Table 3: (a₁, a₂) = (1,0) → X₁, (1,1) → X₂, (0,1) → X₃
  const CIRCUITS = [
    { id: 'encode', title: 'Encode one qubit into three', fig: 'Girvin Fig. 16', n: 3, data: [0, 1, 2], labels: ['q₁ = |ψ⟩', 'q₂ = |0⟩', 'q₃ = |0⟩'],
      blurb: 'Two CNOTs spread α|0⟩ + β|1⟩ over three qubits. This is not copying: the result is one entangled state, α|000⟩ + β|111⟩, not three copies of |ψ⟩.',
      ops: [{ g: 'CX', c: [0], t: 1, say: 'CNOT: q₂ flips if q₁ is 1. Since q₁ is in a superposition, so is the pair: α|00⟩ + β|11⟩. They are now entangled.' },
        { g: 'CX', c: [0], t: 2, say: 'A second CNOT brings in q₃: α|000⟩ + β|111⟩. Three physical qubits, one logical qubit.' }],
      errors: ['none'] },
    { id: 'measure', title: 'Find a bit flip by measuring parities', fig: 'Girvin Box 5 and Table 3', n: 5, data: [0, 1, 2], labels: ['q₁', 'q₂', 'q₃', 'a₁ = |0⟩', 'a₂ = |0⟩'],
      blurb: 'Two ancillas each collect the parity of two data qubits, then are measured. The two answers (the syndrome) name the flipped qubit, and a classical decision undoes it.',
      prep: enc, ops: [{ g: 'ERR', say: 'Noise strikes the encoded state.' }].concat(bitParity, [
        { g: 'M', q: 3, b: 's1', say: 'Measure a₁. This only asks “do q₁ and q₂ agree?”, so it does not collapse α|000⟩ + β|111⟩.' },
        { g: 'M', q: 4, b: 's2', say: 'Measure a₂: “do q₂ and q₃ agree?”' },
        { g: 'FIX', b: ['s1', 's2'], p: 'X', table: SYN, say: 'Look up the syndrome (a₁, a₂): 00 → nothing, 10 → X₁, 11 → X₂, 01 → X₃, and apply it.' }]),
      errors: ['none', 'X0', 'X1', 'X2', 'RX1', 'X0X1'] },
    { id: 'auto', title: 'Correct without measuring', fig: 'Girvin Fig. 17', n: 5, data: [0, 1, 2], labels: ['q₁', 'q₂', 'q₃', 'q₄ = |0⟩', 'q₅ = |0⟩'],
      blurb: 'The same parities, but Toffoli gates do the correcting, controlled by the ancillas themselves. Resetting the ancillas dumps the error’s entropy into a cold bath.',
      prep: enc, ops: [{ g: 'ERR', say: 'Noise strikes the encoded state.' }].concat(bitParity.map((o, k) => Object.assign({}, o, { say: ['CNOT q₁ → q₄…', '…and q₂ → q₄: q₄ = q₁ ⊕ q₂, the parity Z₁Z₂.', 'CNOT q₂ → q₅…', '…and q₃ → q₅: q₅ = q₂ ⊕ q₃, the parity Z₂Z₃.'][k] })), [
        { g: 'CX', c: [3, 4], t: 1, say: 'Toffoli: flip q₂ if q₄ = q₅ = 1. That syndrome means q₂ was hit.' },
        { g: 'X', q: 4, say: 'NOT q₅, so that the next Toffoli fires on the syndrome (1, 0)…' },
        { g: 'CX', c: [3, 4], t: 0, say: '…which means q₁ was hit: flip q₁.' },
        { g: 'X', q: 3, say: 'NOT q₄…' },
        { g: 'X', q: 4, say: '…and NOT q₅ back, so the last Toffoli fires on the syndrome (0, 1)…' },
        { g: 'CX', c: [3, 4], t: 2, say: '…which means q₃ was hit: flip q₃. The data qubits are fixed, and no measurement was needed.' },
        { g: 'RESET', q: 3, say: 'Reset q₄ to |0⟩ by letting it relax into a cold bath. The record of the error leaves with it: entropy is pumped out of the computer.' },
        { g: 'RESET', q: 4, say: 'Reset q₅ too. The ancillas are ready for the next round.' }]),
      errors: ['none', 'X0', 'X1', 'X2', 'RX1', 'X0X1'] },
    { id: 'phase', title: 'The phase-flip code', fig: 'Girvin §3: a Hadamard swaps X and Z', n: 5, data: [0, 1, 2], labels: ['q₁', 'q₂', 'q₃', 'a₁ = |0⟩', 'a₂ = |0⟩'],
      blurb: 'H X H = Z, so turning the bit-flip code by a Hadamard on every qubit gives a code for phase flips: α|+++⟩ + β|−−−⟩. To read its syndrome, turn the data back with Hadamards, measure the same parities, and turn them forward again.',
      prep: enc.concat([{ g: 'H', q: 0 }, { g: 'H', q: 1 }, { g: 'H', q: 2 }]),
      ops: [{ g: 'ERR', say: 'A phase flip strikes. It turns |+⟩ into |−⟩, which a bit-flip check cannot see.' },
        { g: 'H', q: 0, say: 'Hadamards on the data turn |±⟩ back into |0⟩/|1⟩, and turn the phase flip into a bit flip…' }, { g: 'H', q: 1, say: '' }, { g: 'H', q: 2, say: '' }]
        .concat(bitParity.map(o => Object.assign({}, o, { say: o.say.replace('Z₁Z₂', 'X₁X₂ of the phase code').replace('Z₂Z₃', 'X₂X₃ of the phase code') })), [
          { g: 'H', q: 0, say: '…and Hadamards turn the data back into the phase code.' }, { g: 'H', q: 1, say: '' }, { g: 'H', q: 2, say: '' },
          { g: 'M', q: 3, b: 's1', say: 'Measure a₁: X₁X₂.' }, { g: 'M', q: 4, b: 's2', say: 'Measure a₂: X₂X₃.' },
          { g: 'FIX', b: ['s1', 's2'], p: 'Z', table: SYN, say: 'Same table, but the correction is a Z.' }]),
      errors: ['none', 'Z0', 'Z1', 'Z2', 'RZ1'] },
    { id: 'check', title: 'One lamp of the surface code', fig: 'stabilizer measurement (Fowler et al. 2012)', n: 6, zeroIn: true, data: [0, 1, 2, 3], labels: ['d₁', 'd₂', 'd₃', 'd₄', 'Z-check |0⟩', 'X-check |0⟩'],
      blurb: 'Every lamp in the game is one of these little circuits, run every round. A blue (Z-type) lamp: four CNOTs from the data onto an ancilla, then measure. A gold (X-type) lamp: put the ancilla in |+⟩, four CNOTs from it onto the data, then measure in the X basis. Here both share the same four qubits (the [[4,2,2]] code), starting in the code state (|0000⟩ + |1111⟩)/√2.',
      prep: [{ g: 'H', q: 0 }, { g: 'CX', c: [0], t: 1 }, { g: 'CX', c: [0], t: 2 }, { g: 'CX', c: [0], t: 3 }],
      ops: [{ g: 'ERR', say: 'An error strikes one data qubit.' },
        { g: 'CX', c: [0], t: 4, say: 'Z-check: CNOTs from each data qubit onto the ancilla add up the parity of all four…' }, { g: 'CX', c: [1], t: 4, say: '' }, { g: 'CX', c: [2], t: 4, say: '' }, { g: 'CX', c: [3], t: 4, say: '…so the ancilla holds Z₁Z₂Z₃Z₄: 1 if an odd number of bit flips touched the face.' },
        { g: 'H', q: 5, say: 'X-check: put its ancilla in |+⟩…' },
        { g: 'CX', c: [5], t: 0, say: '…and let it control X on each data qubit. A Z error on the data kicks a phase back onto the ancilla.' }, { g: 'CX', c: [5], t: 1, say: '' }, { g: 'CX', c: [5], t: 2, say: '' }, { g: 'CX', c: [5], t: 3, say: '' },
        { g: 'H', q: 5, say: 'A Hadamard turns that phase into a bit: the ancilla now holds X₁X₂X₃X₄.' },
        { g: 'M', q: 4, b: 's1', say: 'Measure the Z-check: the blue lamp.' }, { g: 'M', q: 5, b: 's2', say: 'Measure the X-check: the gold lamp. In a real surface code the four CNOTs run in a fixed order, so that one faulty gate cannot spread into a long error.' }],
      errors: ['none', 'X0', 'Z2', 'Y1'] },
    { id: 'shor', title: 'Shor’s nine-qubit code', fig: 'Shor 1995', n: 9, data: [0, 1, 2, 3, 4, 5, 6, 7, 8], labels: ['q₁ = |ψ⟩', 'q₂', 'q₃', 'q₄', 'q₅', 'q₆', 'q₇', 'q₈', 'q₉'],
      blurb: 'The first quantum code (1995): a phase-flip code whose three qubits are each a bit-flip code. It corrects any single X, Y or Z error.',
      ops: [{ g: 'CX', c: [0], t: 3, say: 'Outer layer, the phase-flip code: spread q₁ to q₄…' }, { g: 'CX', c: [0], t: 6, say: '…and q₇.' },
        { g: 'H', q: 0, say: 'Hadamards make it a phase-flip code: α|+++⟩ + β|−−−⟩ on q₁, q₄, q₇.' }, { g: 'H', q: 3, say: '' }, { g: 'H', q: 6, say: '' },
        { g: 'CX', c: [0], t: 1, say: 'Inner layer: each of those becomes a bit-flip code of three…' }, { g: 'CX', c: [0], t: 2, say: '' }, { g: 'CX', c: [3], t: 4, say: '' }, { g: 'CX', c: [3], t: 5, say: '' }, { g: 'CX', c: [6], t: 7, say: '' },
        { g: 'CX', c: [6], t: 8, say: '…giving α(|000⟩ + |111⟩)³/2√2 + β(|000⟩ − |111⟩)³/2√2.' }],
      errors: ['none'] },
  ];
  const ERR_LABEL = { none: 'no error', X0: 'X on q₁', X1: 'X on q₂', X2: 'X on q₃', Z0: 'Z on q₁', Z1: 'Z on q₂', Z2: 'Z on q₃', Y1: 'Y on q₂', RX1: 'a small rotation of q₂ (θ)', RZ1: 'a small phase rotation of q₂ (θ)', X0X1: 'X on q₁ and q₂ (two errors)' };
  function applyError(s, e, theta) {
    if (e === 'none') return;
    if (e === 'RX1') { const m = rx(theta); apply1(s, 1, m[0], m[1], m[2], m[3]); return; }
    if (e === 'RZ1') { const m = rz(theta); apply1(s, 1, m[0], m[1], m[2], m[3]); return; }
    for (let k = 0; k < e.length; k += 2) gate(s, e[k], +e[k + 1]);
  }
  // run one op on a state (mutates); returns a note for measurements
  function step(s, op, opts) {
    const o = opts || {};
    if (op.g === 'H' || op.g === 'X' || op.g === 'Z') gate(s, op.g, op.q);
    else if (op.g === 'CX') cx(s, op.c, op.t);
    else if (op.g === 'ERR') applyError(s, o.error || 'none', o.theta == null ? 0.6 : o.theta);
    else if (op.g === 'M') { const r = measure(s, op.q, o.rng); s.bits[op.b] = r.out; return r; }
    else if (op.g === 'RESET') { const r = measure(s, op.q, o.rng); if (r.out) gate(s, 'X', op.q); return r; }
    else if (op.g === 'FIX') { const key = op.b.map(b => s.bits[b]).join(''), q = op.table[key]; if (q != null) gate(s, op.p, q); return { key, q }; }
    return null;
  }
  // the initial state: q₁ = α|0⟩ + β|1⟩ (real α, β), everything else |0⟩, then the circuit's prep ops
  function initial(C, alpha, beta) {
    const s = new State(C.n); if (!C.zeroIn) { s.re[0] = alpha; s.re[1] = beta; }   // zeroIn: the circuit prepares its own state from |0…0⟩
    for (const op of C.prep || []) step(s, op);
    return s;
  }
  // the states before each op and after the last: snaps[k] is the state after k ops
  function run(C, opts) {
    const o = opts || {}, s = initial(C, o.alpha == null ? 0.8 : o.alpha, o.beta == null ? 0.6 : o.beta), snaps = [s.clone()], notes = [];
    for (const op of C.ops) { notes.push(step(s, op, o)); snaps.push(s.clone()); }
    return { snaps, notes };
  }
  // ⟨φ|ρ_data|φ⟩ for a pure target φ on the data qubits (the others traced out)
  function dataFidelity(s, data, target) {
    const other = []; for (let q = 0; q < s.n; q++) if (!data.includes(q)) other.push(q);
    let tot = 0;
    for (let a = 0; a < (1 << other.length); a++) {
      let r = 0, im = 0;
      for (let d = 0; d < (1 << data.length); d++) {
        let i = 0; data.forEach((q, k) => { if ((d >> k) & 1) i |= 1 << q; }); other.forEach((q, k) => { if ((a >> k) & 1) i |= 1 << q; });
        r += target.re[d] * s.re[i] + target.im[d] * s.im[i]; im += target.re[d] * s.im[i] - target.im[d] * s.re[i];
      }
      tot += r * r + im * im;
    }
    return tot;
  }
  // the ideal encoded state of a circuit on its data qubits, and its images under the logical operators
  function idealData(C, alpha, beta) {
    const s = initial(C, alpha, beta);
    if (C.id === 'encode' || C.id === 'shor') { for (const op of C.ops) step(s, op); }
    const d = C.data.length, t = { re: new Float64Array(1 << d), im: new Float64Array(1 << d) };
    for (let i = 0; i < s.N; i++) { let k = 0, other = false; for (let q = 0; q < s.n; q++) if ((i >> q) & 1) { const j = C.data.indexOf(q); if (j < 0) other = true; else k |= 1 << j; } if (!other) { t.re[k] += s.re[i]; t.im[k] += s.im[i]; } }
    return t;
  }
  // what the data qubits hold at the end of a run: 'ok', 'logicalX', 'logicalZ', 'logicalY' or null (none fits)
  function verdict(C, s, alpha, beta) {
    const ideal = idealData(C, alpha, beta), d = C.data.length;
    const map = (tf) => { const o = { re: new Float64Array(1 << d), im: new Float64Array(1 << d) }; for (let k = 0; k < (1 << d); k++) { const [k2, sr, si] = tf(k); o.re[k2] += sr * ideal.re[k] - si * ideal.im[k]; o.im[k2] += sr * ideal.im[k] + si * ideal.re[k]; } return o; };
    const all = (1 << d) - 1, par = k => { let p = 0; for (let x = k; x; x &= x - 1) p ^= 1; return p; };
    // logical X = X on every data qubit (bit-flip code) or Z on every one (phase code), and vice versa for logical Z
    const flipAll = k => [all ^ k, 1, 0], signAll = k => [k, par(k) ? -1 : 1, 0];
    const LX = C.id === 'phase' ? signAll : flipAll, LZ = C.id === 'phase' ? flipAll : signAll;
    const fits = { ok: dataFidelity(s, C.data, ideal), logicalX: dataFidelity(s, C.data, map(LX)), logicalZ: dataFidelity(s, C.data, map(LZ)) };
    for (const k of ['ok', 'logicalX', 'logicalZ']) if (fits[k] > 0.999) return { kind: k, fits };
    return { kind: null, fits };
  }
  // the nonzero terms of a state, largest first: [{ ket: '0110', re, im, p }] with qubits written q₁ first
  function terms(s, tol) {
    const out = [];
    for (let i = 0; i < s.N; i++) { const p = s.re[i] * s.re[i] + s.im[i] * s.im[i]; if (p > (tol || 1e-9)) { let k = ''; for (let q = 0; q < s.n; q++) k += (i >> q) & 1; out.push({ i, ket: k, re: s.re[i], im: s.im[i], p }); } }
    return out;
  }
  function rng(seed) { let x = (seed >>> 0) || 0x9e3779b9; return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }

  // ---- classical codes (Girvin §2) ----
  const binom = (n, k) => { let r = 1; for (let j = 1; j <= k; j++) r = r * (n - k + j) / j; return r; };
  // majority vote over N = 2m + 1 copies fails when more than m flip: ε_L = Σ_{k>m} C(N,k) ε^k (1−ε)^{N−k}
  function repFail(N, eps) { let s = 0; for (let k = (N + 1) / 2; k <= N; k++) s += binom(N, k) * Math.pow(eps, k) * Math.pow(1 - eps, N - k); return s; }
  // a single parity bit on m data bits misses an even number (≥ 2) of flips among the m + 1 bits
  function parityMiss(m, eps) { let s = 0; for (let k = 2; k <= m + 1; k += 2) s += binom(m + 1, k) * Math.pow(eps, k) * Math.pow(1 - eps, m + 1 - k); return s; }
  // Hamming [7,4,3]: column k (1..7) of H is k in binary, so the syndrome of a flip on bit k reads k
  const HAMMING = [0, 1, 2].map(r => [1, 2, 3, 4, 5, 6, 7].map(k => (k >> r) & 1));
  const hammingSyndrome = word => HAMMING.map(row => row.reduce((s, h, j) => s ^ (h & word[j]), 0));

  const api = { State, apply1, gate, cx, measure, prob1, rx, rz, CIRCUITS, ERR_LABEL, applyError, step, initial, run, dataFidelity, idealData, verdict, terms, rng,
    binom, repFail, parityMiss, HAMMING, hammingSyndrome, SYN };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QCirc = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
