// Checks the stabilizer-code engine behind ErrorSweeper (web/qublitz/qec-engine.js).
// Usage: node web/qublitz/tests/qec-engine.test.js
'use strict';
const Q = require('../qec-engine.js');
let failures = 0;
function check(label, ok, detail) { console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`); if (!ok) failures++; }

// ---- code parameters [[n, k, d]] (Kitaev 2003; Tomita & Svore 2014; Steane 1996; Shor 1995) ----
const want = [
  [Q.repetition(5), 5, 1, 5, 1], [Q.rotatedSurface(3), 9, 1, 3, 3], [Q.rotatedSurface(5), 25, 1, 5, 5],
  [Q.toric(3), 18, 2, 3, 3], [Q.toric(4), 32, 2, 4, 4], [Q.steane(), 7, 1, 3, 3], [Q.shor(), 9, 1, 3, 3],
];
for (const [c, n, k, dx, dz] of want) {
  const DX = Q.distanceOf(c, 'X', 6).d, DZ = Q.distanceOf(c, 'Z', 6).d;
  check(`${c.name}: [[${n},${k}]], d_X = ${dx}, d_Z = ${dz}, checks commute`, c.n === n && c.k === k && DX === dx && DZ === dz && c.commuteOK,
    `n=${c.n} k=${c.k} dX=${DX} dZ=${DZ}`);
}
check('rotated d = 5 has (d²−1)/2 = 12 checks of each type', Q.rotatedSurface(5).X.length === 12 && Q.rotatedSurface(5).Z.length === 12);

// ---- syndromes ----
{
  const c = Q.rotatedSurface(3), e = Q.pauli(9); e.x[4] = 1;   // centre qubit
  const s = Q.syndrome(c, e), lit = c.checks.filter((ch, i) => s[i]);
  check('an X on the centre of d = 3 lights exactly its two Z faces', lit.length === 2 && lit.every(ch => ch.type === 'Z'));
  const y = Q.pauli(9); y.x[4] = 1; y.z[4] = 1;
  check('a Y on the centre lights two Z and two X faces', Q.syndrome(c, y).filter(Boolean).length === 4);
  for (const ch of c.checks) { const st = Q.pauli(9); for (const q of ch.q) (ch.type === 'X' ? st.x : st.z)[q] = 1;
    if (Q.syndrome(c, st).some(Boolean) || !Q.classify(c, st).trivial) { check('stabilizers have zero syndrome and act trivially', false, ch.type + ch.q); break; } }
  check('every stabilizer has zero syndrome and is trivial', true);
}

// ---- decoders correct every error of weight ≤ (d−1)/2 ----
for (const c of [Q.rotatedSurface(3), Q.rotatedSurface(5), Q.toric(3), Q.steane(), Q.shor(), Q.repetition(5)]) {
  const t = Math.floor((Math.min(c.d || 3, 5) - 1) / 2), types = c.errors === 'X' ? ['x'] : ['x', 'z', 'y'];
  let bad = 0, tried = 0;
  const qs = [...Array(c.n).keys()];
  const sets = t >= 2 ? qs.flatMap(a => qs.filter(b => b > a).map(b => [a, b])) : qs.map(a => [a]);
  for (const set of sets) for (const ty of types) {
    const e = Q.pauli(c.n); for (const q of set) { if (ty !== 'z') e.x[q] = 1; if (ty !== 'x') e.z[q] = 1; }
    const r = Q.combine(e, Q.decode(c, Q.syndrome(c, e))); tried++;
    if (Q.syndrome(c, r).some(Boolean) || !Q.classify(c, r).trivial) bad++;
  }
  check(`${c.name}: decoder corrects all ${tried} errors of weight ≤ ${t}`, bad === 0, `${bad} failed`);
}

// ---- a logical operator is detected as one ----
{
  const c = Q.rotatedSurface(3), L = Q.distanceOf(c, 'X', 4).support, r = Q.pauli(9); for (const q of L) r.x[q] = 1;
  const cl = Q.classify(c, r);
  check('a weight-3 X string from boundary to boundary is a logical X (zero syndrome)', !Q.syndrome(c, r).some(Boolean) && cl.logicalX && !cl.logicalZ);
  const t = Q.toric(3), lx = Q.distanceOf(t, 'X', 4).support, rt = Q.pauli(18); for (const q of lx) rt.x[q] = 1;
  check('toric: a non-contractible loop is a logical', !Q.syndrome(t, rt).some(Boolean) && Q.classify(t, rt).logicalX);
}

// ---- threshold behaviour (code capacity, X noise, matching): below ~10% bigger codes fail less ----
{
  const R = Q.rng(11);
  const pl = d => Q.logicalRate(Q.rotatedSurface(d), 0.02, 'X', 6000, R);
  const a = pl(3), b = pl(5);
  check('at p = 2%, d = 5 fails less often than d = 3', b < a, `${a.toExponential(2)} vs ${b.toExponential(2)}`);
  const hi3 = Q.logicalRate(Q.rotatedSurface(3), 0.16, 'X', 3000, R), hi5 = Q.logicalRate(Q.rotatedSurface(5), 0.16, 'X', 3000, R);
  check('at p = 16% (above threshold), d = 5 fails more often than d = 3', hi5 > hi3, `${hi3.toFixed(3)} vs ${hi5.toFixed(3)}`);
}

// ---- custom codes from the code lab ----
{
  const bad = Q.custom({ qubits: [{ x: 0, y: 0 }, { x: 1, y: 0 }], checks: [{ type: 'X', q: [0] }, { type: 'Z', q: [0, 1] }] });
  check('a non-commuting pair of checks is flagged', !bad.commuteOK);
  const st = Q.steane(), copy = Q.custom({ qubits: st.qubits, checks: st.checks.map(c => ({ type: c.type, q: c.q })) });
  check('a hand-drawn Steane code has k = 1 and d = 3', copy.k === 1 && Q.distanceOf(copy, 'X', 5).d === 3 && Q.distanceOf(copy, 'Z', 5).d === 3);
}

// ---- the new search tools: distance by branch-and-bound, exact table decoding, matching traces ----
{
  for (const [c, d] of [[Q.rotatedSurface(7), 7], [Q.toric(5), 5], [Q.repetition(7), 7]]) {
    const r = Q.distanceOf(c, 'X'); check(`${c.name}: branch-and-bound distance d_X = ${d} (exact)`, r.d === d && r.exact, `got ${r.d}`);
  }
  // brute force agrees with the table decoder on Steane: every syndrome's correction has the minimum weight
  const st = Q.steane(); let ok = true;
  for (let mask = 0; mask < 8; mask++) {
    const s = [0, 1, 2].map(i => (mask >> i) & 1), H = st.Z.map(c => c.vec);
    const brute = Q.lightest(H, 7, s, 7), synd = st.checks.map(c => (c.type === 'Z' ? s[st.Z.indexOf(c)] : 0));
    const tr = Q.traceType(st, synd, 'X');
    if (tr.method !== (mask ? 'table' : 'none') && mask) ok = false;
    if (Q.weight(tr.corr) !== Q.weight(brute) || H.some((h, i) => Q.dot(h, tr.corr) !== s[i])) ok = false;
  }
  check('Steane: the lookup-table decoder returns a lightest correction for all 8 syndromes', ok);
  check('Steane is not a matching graph: some qubit sits in three Z checks', !!Q.nonGraphQubit(st, 'Z') && !st.graphX);
  check('the surface code is a matching graph', !Q.nonGraphQubit(Q.rotatedSurface(5), 'Z') && !!Q.rotatedSurface(5).graphX);
  // a matching trace: one X error in the bulk of d = 5 lights two Z checks, matched to each other through that qubit
  const c = Q.rotatedSurface(5), e = Q.pauli(25); e.x[12] = 1;
  const tr = Q.traceType(c, Q.syndrome(c, e), 'X');
  check('matching pairs the two lamps around one error, through that qubit', tr.method === 'matching' && tr.pairs.length === 1 && tr.pairs[0].b !== null && tr.pairs[0].path.join() === '12');
  const e2 = Q.pauli(25); e2.x[0] = 1; const tr2 = Q.traceType(c, Q.syndrome(c, e2), 'X');
  check('an error at the code edge lights one lamp, matched to the boundary', tr2.pairs.length === 1 && tr2.pairs[0].b === null && tr2.pairs[0].path.join() === '0');
  // a hand-drawn non-graph code with many checks still decodes every single error (Steane drawn twice side by side)
  const two = Q.custom({ qubits: st.qubits.concat(st.qubits.map(q => ({ x: q.x + 5, y: q.y }))),
    checks: st.checks.map(ch => ({ type: ch.type, q: ch.q })).concat(st.checks.map(ch => ({ type: ch.type, q: ch.q.map(i => i + 7) }))) });
  let bad = 0; for (let q = 0; q < 14; q++) for (const ty of ['x', 'z']) { const er = Q.pauli(14); er[ty][q] = 1; if (!Q.classify(two, Q.combine(er, Q.decode(two, Q.syndrome(two, er)))).trivial) bad++; }
  check('two Steane blocks: k = 2, d = 3, every single error corrected', two.k === 2 && Q.distanceOf(two, 'Z').d === 3 && bad === 0, `${bad} failed`);
  // figure of merit and pseudo-threshold helpers
  check('k·d²/n = 1 for every rotated surface code', [3, 5, 7].every(d => Q.kd2n(d * d, 1, d) === 1));
  const pts = [0.01, 0.03, 0.1, 0.3].map(p => ({ p, v: 3 * p * p * 30 }));   // v = 90 p², crosses v = p at p = 1/90
  const pth = Q.crossing(pts, 'X', 1);
  check('pseudo-threshold of P_L = 90 p² is p = 1/90', pth && Math.abs(pth - 1 / 90) / (1 / 90) < 0.25, pth && pth.toFixed(4));
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
