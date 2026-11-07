/* qec-engine.js: stabilizer codes, syndromes and decoders for ErrorSweeper (errorsweeper.html). No dependencies.
 * Loads as a classic <script> (window.QECEngine) or from Node (require), so tests/qec-engine.test.js checks the code
 * the game runs.
 *
 * The model is a CSS stabilizer code (Gottesman 1997; Calderbank & Shor 1996; Steane 1996) under code-capacity noise:
 * each round, Pauli errors hit the data qubits, every stabilizer is measured once and perfectly, and the player (or a
 * decoder) chooses a correction. A Z-type check flips when it contains an odd number of X-errored qubits; an X-type
 * check, likewise for Z errors; Y = XZ counts for both. After the correction c, the residual r = e + c (mod 2, per Pauli
 * type) has zero syndrome. The round succeeds if r is a product of stabilizers (it acts trivially on the encoded
 * state) and fails if r is a nontrivial logical operator: for the X part, r ∉ rowspace(H_X); for the Z part,
 * r ∉ rowspace(H_Z). k = n − rank H_X − rank H_Z logical qubits; the distance d is the weight of the lightest logical.
 *
 * Codes built in: the repetition code; the rotated surface code (Bombin & Martin-Delgado 2007; Tomita & Svore 2014),
 * n = d², with X and Z boundaries; the toric code (Kitaev 2003), n = 2L² on a torus, k = 2; Steane's [[7,1,3]]
 * (the smallest 2D color code); Shor's [[9,1,3]]; and any CSS code drawn in the code lab.
 *
 * Decoders. All of them aim for the same thing, the lightest error that explains the lamps (minimum-weight decoding).
 * When every qubit sits in at most two checks of a type (repetition, surface, toric codes) the checks and qubits form a
 * graph (checks are nodes, qubits edges, one boundary node for qubits in a single check), and the lightest explanation
 * is a minimum-weight perfect matching of the lit checks (Dennis, Kitaev, Landahl & Preskill 2002). We solve the small
 * matchings exactly by dynamic programming over subsets of lit checks (greedy above 16 of one type) instead of with
 * Edmonds' blossom algorithm. When some qubit sits in three or more checks of a type (Steane and other color codes,
 * many hand-drawn codes) one error lights three or more lamps and pairing does not apply; then a lookup table built by
 * breadth-first search over syndromes gives the exact lightest correction (up to 18 checks of a type), and beyond that
 * an exact branch-and-bound search with a greedy fallback. X and Z are decoded independently (the standard CSS
 * treatment), so for Y errors the decoder ignores their correlation. Distances come from an exact branch-and-bound
 * search for the lightest logical operator.
 */
(function (root) {
  'use strict';

  // ---- GF(2) rows as arrays of 0/1 ----
  const zeros = n => new Uint8Array(n);
  const fromSet = (n, set) => { const v = zeros(n); for (const i of set) v[i] ^= 1; return v; };
  const weight = v => { let w = 0; for (const b of v) w += b; return w; };
  const add = (a, b) => { const v = zeros(a.length); for (let i = 0; i < a.length; i++) v[i] = a[i] ^ b[i]; return v; };
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s ^= a[i] & b[i]; return s; };
  // row-reduce a copy; returns { rows (independent, reduced), pivots }
  function reduce(rows, n) {
    const R = rows.map(r => Uint8Array.from(r)), piv = [];
    let r = 0;
    for (let c = 0; c < n && r < R.length; c++) {
      let p = -1; for (let i = r; i < R.length; i++) if (R[i][c]) { p = i; break; }
      if (p < 0) continue;
      [R[r], R[p]] = [R[p], R[r]];
      for (let i = 0; i < R.length; i++) if (i !== r && R[i][c]) for (let j = c; j < n; j++) R[i][j] ^= R[r][j];
      piv.push(c); r++;
    }
    return { rows: R.slice(0, r), pivots: piv };
  }
  const rank = (rows, n) => reduce(rows, n).rows.length;
  // is v in the row space of a reduced basis?
  function inSpan(basis, v) {
    const w = Uint8Array.from(v);
    for (let i = 0; i < basis.rows.length; i++) if (w[basis.pivots[i]]) for (let j = 0; j < w.length; j++) w[j] ^= basis.rows[i][j];
    return weight(w) === 0;
  }
  // null space basis of rows (vectors v with rows·v = 0)
  function nullspace(rows, n) {
    const { rows: R, pivots } = reduce(rows, n), free = [];
    for (let c = 0; c < n; c++) if (!pivots.includes(c)) free.push(c);
    return free.map(f => { const v = zeros(n); v[f] = 1; R.forEach((row, i) => { if (row[f]) v[pivots[i]] = 1; }); return v; });
  }

  // ---- codes ----
  // A code: { name, n, qubits: [{x, y}], checks: [{type: 'X'|'Z', q: [indices], x, y, poly?}], layout } plus derived
  // matrices. finalize() fills Hx, Hz (rows over n), the reduced bases, k, and the decoding graphs.
  function finalize(code) {
    const n = code.n;
    code.checks.forEach((c, i) => { c.id = i; c.vec = fromSet(n, c.q); });
    code.X = code.checks.filter(c => c.type === 'X');
    code.Z = code.checks.filter(c => c.type === 'Z');
    code.Hx = code.X.map(c => c.vec); code.Hz = code.Z.map(c => c.vec);
    code.bx = reduce(code.Hx, n); code.bz = reduce(code.Hz, n);
    code.k = n - code.bx.rows.length - code.bz.rows.length;
    code.commuteOK = code.X.every(a => code.Z.every(b => dot(a.vec, b.vec) === 0));
    code.graphX = graphFor(code, 'Z');   // X errors are seen by Z checks
    code.graphZ = graphFor(code, 'X');
    return code;
  }

  function repetition(d) {
    const qubits = [], checks = [];
    for (let i = 0; i < d; i++) qubits.push({ x: i, y: 0 });
    for (let i = 0; i < d - 1; i++) checks.push({ type: 'Z', q: [i, i + 1], x: i + 0.5, y: 0 });
    return finalize({ name: `Repetition code, d = ${d}`, key: `rep${d}`, family: 'rep', d, n: d, qubits, checks, errors: 'X', layout: 'line' });
  }

  // Rotated surface code (Tomita & Svore 2014, Fig. 1 layout): data qubits on a d×d grid, a checkerboard of
  // weight-4 faces, weight-2 X faces on the top and bottom edges and weight-2 Z faces on the left and right edges.
  function rotatedSurface(d) {
    const qubits = [], checks = [], at = (i, j) => j * d + i;
    for (let j = 0; j < d; j++) for (let i = 0; i < d; i++) qubits.push({ x: i, y: j });
    for (let j = -1; j < d; j++) for (let i = -1; i < d; i++) {
      const type = ((i + j) % 2 + 2) % 2 === 0 ? 'X' : 'Z';
      const q = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].filter(([a, b]) => a >= 0 && b >= 0 && a < d && b < d).map(([a, b]) => at(a, b));
      const interior = i >= 0 && j >= 0 && i < d - 1 && j < d - 1;
      const topBottom = (j === -1 || j === d - 1) && i >= 0 && i < d - 1 && q.length === 2;
      const leftRight = (i === -1 || i === d - 1) && j >= 0 && j < d - 1 && q.length === 2;
      if (interior || (topBottom && type === 'X') || (leftRight && type === 'Z')) checks.push({ type, q, x: i + 0.5, y: j + 0.5 });
    }
    return finalize({ name: `Rotated surface code, d = ${d}`, key: `surf${d}`, family: 'surface', d, n: d * d, qubits, checks, errors: 'XZ', layout: 'faces' });
  }

  // Toric code (Kitaev 2003): an L×L torus, qubits on edges, X checks on vertices (stars), Z checks on faces
  // (plaquettes). Two logical qubits; logicals are loops around the two cycles of the torus.
  function toric(L) {
    const qubits = [], checks = [];
    const h = (i, j) => ((j + L) % L) * L + ((i + L) % L);            // horizontal edge from vertex (i,j) to (i+1,j)
    const v = (i, j) => L * L + ((j + L) % L) * L + ((i + L) % L);    // vertical edge from (i,j) to (i,j+1)
    for (let j = 0; j < L; j++) for (let i = 0; i < L; i++) qubits[h(i, j)] = { x: i + 0.5, y: j };
    for (let j = 0; j < L; j++) for (let i = 0; i < L; i++) qubits[v(i, j)] = { x: i, y: j + 0.5 };
    for (let j = 0; j < L; j++) for (let i = 0; i < L; i++) {
      checks.push({ type: 'X', q: [h(i, j), h(i - 1, j), v(i, j), v(i, j - 1)], x: i, y: j });              // star at vertex
      checks.push({ type: 'Z', q: [h(i, j), h(i, j + 1), v(i, j), v(i + 1, j)], x: i + 0.5, y: j + 0.5 });  // plaquette
    }
    return finalize({ name: `Toric code, L = ${L} (a torus)`, key: `toric${L}`, family: 'toric', d: L, L, n: 2 * L * L, qubits, checks, errors: 'XZ', layout: 'torus' });
  }

  // Steane [[7,1,3]]: the distance-3 triangular color code. Three faces, each both an X and a Z check.
  function steane() {
    const qubits = [{ x: 0, y: 2 }, { x: 2, y: 2 }, { x: 4, y: 2 }, { x: 1, y: 1 }, { x: 3, y: 1 }, { x: 2, y: 1.45 }, { x: 2, y: 0 }];
    const faces = [[0, 1, 5, 3], [1, 2, 4, 5], [3, 5, 4, 6]];
    const checks = [];
    for (const f of faces) for (const type of ['X', 'Z']) checks.push({ type, q: f });
    return finalize({ name: 'Steane code [[7,1,3]] (color code)', key: 'steane', family: 'small', d: 3, n: 7, qubits, checks, errors: 'XZ', layout: 'faces' });
  }

  // Shor [[9,1,3]]: three blocks of three. Z checks Z_iZ_{i+1} inside each block, X checks X^{⊗6} on neighbouring blocks.
  function shor() {
    const qubits = [], checks = [];
    for (let b = 0; b < 3; b++) for (let i = 0; i < 3; i++) qubits.push({ x: i, y: b });
    for (let b = 0; b < 3; b++) for (let i = 0; i < 2; i++) checks.push({ type: 'Z', q: [3 * b + i, 3 * b + i + 1] });
    for (let b = 0; b < 2; b++) checks.push({ type: 'X', q: [0, 1, 2, 3, 4, 5].map(i => i + 3 * b) });
    return finalize({ name: 'Shor code [[9,1,3]]', key: 'shor', family: 'small', d: 3, n: 9, qubits, checks, errors: 'XZ', layout: 'faces', pairs: 'bar' });
  }

  // A code drawn in the code lab: { qubits: [{x,y}], checks: [{type, q}] }.
  function custom(spec, name) {
    return finalize({ name: name || 'Your code', key: 'custom', family: 'custom', n: spec.qubits.length,
      qubits: spec.qubits.map(p => ({ x: p.x, y: p.y })), checks: spec.checks.map(c => ({ type: c.type, q: c.q.slice() })), errors: 'XZ', layout: 'faces' });
  }

  // ---- syndromes, residuals, logical failure ----
  // e = { x: Uint8Array(n), z: Uint8Array(n) }; Y on qubit i means x[i] = z[i] = 1.
  const pauli = n => ({ x: zeros(n), z: zeros(n) });
  function syndrome(code, e) { return code.checks.map(c => dot(c.vec, c.type === 'Z' ? e.x : e.z)); }
  function combine(a, b) { return { x: add(a.x, b.x), z: add(a.z, b.z) }; }
  // classify a zero-syndrome residual: { trivial, logicalX, logicalZ }
  function classify(code, r) {
    const logicalX = !inSpan(code.bx, r.x), logicalZ = !inSpan(code.bz, r.z);
    return { trivial: !logicalX && !logicalZ, logicalX, logicalZ };
  }

  // ---- error sampling ----
  // mode 'X' (bit flips), 'Z' (phase flips), 'XZ' (independent X and Z, each with prob p), 'dep' (depolarizing:
  // X, Y, Z each with p/3). rng() in [0,1).
  function sample(code, p, mode, rng) {
    const e = pauli(code.n), R = rng || Math.random;
    for (let i = 0; i < code.n; i++) {
      if (mode === 'dep') { const u = R(); if (u < p / 3) e.x[i] = 1; else if (u < 2 * p / 3) e.z[i] = 1; else if (u < p) { e.x[i] = 1; e.z[i] = 1; } }
      else { if (mode !== 'Z' && R() < p) e.x[i] = 1; if (mode !== 'X' && R() < p) e.z[i] = 1; }
    }
    return e;
  }

  // ---- decoding graphs and exact minimum-weight matching ----
  // For errors of type T seen by checks of the other type: nodes = those checks plus a boundary node B; each qubit in
  // one or two of them is an edge. Returns null if some qubit is in three or more (not a graph code).
  function graphFor(code, checkType) {
    const cs = code.checks.filter(c => c.type === checkType);
    if (!cs.length) return null;
    const idx = new Map(cs.map((c, i) => [c.id, i])), B = cs.length, edges = [];
    for (let q = 0; q < code.n; q++) {
      const touching = cs.filter(c => c.vec[q]).map(c => idx.get(c.id));
      if (touching.length > 2) return null;
      if (touching.length === 0) continue;
      edges.push({ q, a: touching[0], b: touching.length === 2 ? touching[1] : B });
    }
    const adj = Array.from({ length: B + 1 }, () => []);
    for (const e of edges) { adj[e.a].push(e); adj[e.b].push(e); }
    const hasBoundary = edges.some(e => e.b === B);
    // all-pairs shortest paths by BFS (unit weights), with the qubits along one shortest path
    const dist = [], via = [];
    for (let s = 0; s <= B; s++) {
      const d = new Array(B + 1).fill(Infinity), prev = new Array(B + 1).fill(null); d[s] = 0; const queue = [s];
      while (queue.length) { const u = queue.shift(); for (const e of adj[u]) { const w = e.a === u ? e.b : e.a; if (d[w] === Infinity) { d[w] = d[u] + 1; prev[w] = { u, q: e.q }; queue.push(w); } } }
      dist.push(d); via.push(prev);
    }
    const path = (s, t) => { const out = []; let x = t; while (x !== s && via[s][x]) { out.push(via[s][x].q); x = via[s][x].u; } return out; };
    return { checks: cs, idx, B, edges, dist, path, hasBoundary };
  }
  // exact minimum-weight matching of defects (indices into graph nodes), each paired with another defect or the
  // boundary; dynamic programming over subsets (fine up to ~18 defects)
  function matchDefects(G, defects) {
    const m = defects.length;
    if (!m) return [];
    if (m > 16) return greedyMatch(G, defects);   // rare on the boards used here; exact DP costs 2^m·m
    const full = (1 << m) - 1, best = new Float64Array(1 << m).fill(Infinity), choice = new Int32Array(1 << m).fill(-2);
    best[0] = 0;
    for (let mask = 1; mask <= full; mask++) {
      let i = 0; while (!(mask >> i & 1)) i++;
      const rest = mask & ~(1 << i), a = defects[i];
      if (G.hasBoundary) { const c = best[rest] + G.dist[a][G.B]; if (c < best[mask]) { best[mask] = c; choice[mask] = -1; } }
      for (let j = i + 1; j < m; j++) if (rest >> j & 1) {
        const c = best[rest & ~(1 << j)] + G.dist[a][defects[j]];
        if (c < best[mask]) { best[mask] = c; choice[mask] = j; }
      }
    }
    const pairs = []; let mask = full;
    while (mask) {
      let i = 0; while (!(mask >> i & 1)) i++;
      const j = choice[mask];
      if (j === -1) { pairs.push([defects[i], G.B]); mask &= ~(1 << i); }
      else { pairs.push([defects[i], defects[j]]); mask &= ~(1 << i) & ~(1 << j); }
    }
    return pairs;
  }
  function greedyMatch(G, defects) {
    const left = defects.slice(), pairs = [];
    while (left.length) {
      const a = left.shift(); let bj = -1, bd = G.hasBoundary ? G.dist[a][G.B] : Infinity;
      left.forEach((b, j) => { if (G.dist[a][b] < bd) { bd = G.dist[a][b]; bj = j; } });
      if (bj < 0) pairs.push([a, G.B]); else pairs.push([a, left.splice(bj, 1)[0]]);
    }
    return pairs;
  }
  // brute force: the lightest v with H v = s (rows of H are the given checks), searching by weight up to wmax
  function lightest(H, n, s, wmax) {
    const target = s.join('');
    const synd = v => H.map(h => dot(h, v)).join('');
    if (synd(zeros(n)) === target) return zeros(n);
    const pick = [];
    for (let w = 1; w <= Math.min(wmax, n); w++) {
      const found = (function rec(start, depth) {
        if (depth === w) { const v = fromSet(n, pick); return synd(v) === target ? v : null; }
        for (let i = start; i < n; i++) { pick.push(i); const r = rec(i + 1, depth + 1); pick.pop(); if (r) return r; }
        return null;
      })(0, 0);
      if (found) return found;
    }
    return null;
  }
  // ---- exact minimum-weight search by branching (any CSS code) ----
  // Finds the lightest v with H v = target. If u ⊂ v is a partial guess whose syndrome still differs from the target
  // on some check, then v \ u must contain a qubit of that check (its parity there is odd), so branching over the
  // qubits of one mismatched check at a time reaches every minimal v; iterative deepening over the weight makes the
  // first hit a lightest one. With target = 0 and accept(v) = "v is not a stabilizer", the same search finds the
  // lightest logical operator, i.e. the distance: a minimal logical has no proper zero-syndrome subset, since that
  // subset would be a stabilizer (and removing it leaves a lighter logical) or a lighter logical itself.
  // budget caps the number of search nodes; past it the result is marked inexact.
  function rowsOf(H, n) { return H.map(h => { const a = []; for (let q = 0; q < n; q++) if (h[q]) a.push(q); return a; }); }
  function incidence(H, n) { const inc = Array.from({ length: n }, () => []); H.forEach((h, r) => { for (let q = 0; q < n; q++) if (h[q]) inc[q].push(r); }); return inc; }
  function branchSearch(H, n, target, opts) {
    const o = opts || {}, wmax = Math.min(o.wmax || n, n), budget = o.budget || 2e5, accept = o.accept || null, logical = !!o.logical;
    const m = H.length, rows = rowsOf(H, n), inc = incidence(H, n), maxInc = Math.max(1, ...inc.map(a => a.length));
    const cur = Uint8Array.from(target), chosen = new Uint8Array(n), pick = [];
    let mis = 0; for (let r = 0; r < m; r++) mis += cur[r];
    let nodes = 0, exhausted = false;
    const flip = q => { chosen[q] ^= 1; for (const r of inc[q]) { cur[r] ^= 1; mis += cur[r] ? 1 : -1; } };
    function rec(w, minQ) {
      if (++nodes > budget) { exhausted = true; return null; }
      if (mis === 0 && pick.length) return !accept || accept(pick) ? pick.slice() : null;
      if (mis === 0) return null;
      if (w === 0 || mis > w * maxInc) return null;
      let best = -1, bc = Infinity;   // the mismatched check with the fewest free qubits
      for (let r = 0; r < m; r++) if (cur[r]) { let c = 0; for (const q of rows[r]) if (!chosen[q] && q >= minQ) c++; if (c < bc) { bc = c; best = r; } }
      if (!bc) return null;
      for (const q of rows[best]) if (!chosen[q] && q >= minQ) {
        flip(q); pick.push(q); const f = rec(w - 1, minQ); pick.pop(); flip(q);
        if (f || exhausted) return f;
      }
      return null;
    }
    for (let w = 1; w <= wmax; w++) {
      let found = null;
      if (logical) {   // target 0: the first qubit is the smallest index of v; the rest must be larger
        for (let q0 = 0; q0 < n && !found && !exhausted; q0++) { flip(q0); pick.push(q0); found = w === 1 ? (mis === 0 && (!accept || accept(pick)) ? pick.slice() : null) : rec(w - 1, q0 + 1); pick.pop(); flip(q0); }
      } else found = mis === 0 ? [] : rec(w, 0);
      if (found) return { v: fromSet(n, found), support: found.slice().sort((a, b) => a - b), w: found.length, exact: !exhausted };
      if (exhausted) return { v: null, support: null, w: null, exact: false, atLeast: w };
    }
    return { v: null, support: null, w: null, exact: true, atLeast: wmax + 1 };
  }
  // the lightest correction for a syndrome, as a lookup table built once by breadth-first search over syndromes
  // (2^m entries for m checks of one type): exact minimum-weight decoding for any small code
  function tableFor(code, checkType) {
    const key = '_tab' + checkType; if (code[key] !== undefined) return code[key];
    const cs = code.checks.filter(c => c.type === checkType), m = cs.length;
    if (m > 18) return (code[key] = null);
    const col = new Int32Array(code.n); cs.forEach((c, r) => { for (const q of c.q) col[q] ^= (1 << r); });
    const N = 1 << m, par = new Int16Array(N).fill(-1), seen = new Uint8Array(N), queue = new Int32Array(N);
    seen[0] = 1; let head = 0, tail = 0; queue[tail++] = 0;
    while (head < tail) { const s = queue[head++]; for (let q = 0; q < code.n; q++) { const t = s ^ col[q]; if (!seen[t]) { seen[t] = 1; par[t] = q; queue[tail++] = t; } } }
    return (code[key] = { m, col, par, seen });
  }
  function fromTable(T, n, s) {
    let mask = 0; s.forEach((b, r) => { if (b) mask |= (1 << r); });
    const v = zeros(n); if (!T.seen[mask]) return null;
    while (mask) { const q = T.par[mask]; v[q] ^= 1; mask ^= T.col[q]; }
    return v;
  }
  // which qubit (if any) stops a check type from forming a matching graph: it sits in three or more such checks
  function nonGraphQubit(code, checkType) {
    const cs = code.checks.filter(c => c.type === checkType);
    for (let q = 0; q < code.n; q++) { const t = cs.filter(c => c.vec[q]); if (t.length > 2) return { q, checks: t.map(c => c.id) }; }
    return null;
  }

  // decode one Pauli type, with a record of how: returns { corr, method, pairs?, defects }
  // pairs: [{ a, b, path }] with a, b check ids (b = null for the boundary) and path the qubits flipped between them
  function traceType(code, synd, type) {
    const G = type === 'X' ? code.graphX : code.graphZ, checkType = type === 'X' ? 'Z' : 'X';
    const cs = code.checks.filter(c => c.type === checkType), s = cs.map(c => synd[c.id]);
    const defects = cs.filter((c, i) => s[i]).map(c => c.id);
    if (!cs.length || !defects.length) return { corr: zeros(code.n), method: G ? 'matching' : 'none', pairs: [], defects };
    if (G) {
      const nodes = []; s.forEach((b, i) => { if (b) nodes.push(i); });
      const corr = zeros(code.n), pairs = [];
      for (const [a, b] of matchDefects(G, nodes)) {
        const path = G.path(a, b); for (const q of path) corr[q] ^= 1;
        pairs.push({ a: G.checks[a].id, b: b === G.B ? null : G.checks[b].id, path });
      }
      return { corr, method: nodes.length > 16 ? 'greedy' : 'matching', pairs, defects };
    }
    const T = tableFor(code, checkType);
    if (T) { const v = fromTable(T, code.n, s); if (v) return { corr: v, method: 'table', defects }; }
    const H = cs.map(c => c.vec), r = branchSearch(H, code.n, s, { wmax: Math.min(code.n, 10), budget: 6e4 });
    if (r.v) return { corr: r.v, method: r.exact ? 'search' : 'search-partial', defects };
    return { corr: greedyFlip(H, code.n, s), method: 'greedy-flip', defects };
  }
  // last resort for big non-graph codes: flip whichever qubit fixes the most mismatched checks, until none remain
  function greedyFlip(H, n, s) {
    const inc = incidence(H, n), cur = Uint8Array.from(s), v = zeros(n);
    for (let it = 0; it < 4 * n && cur.some(Boolean); it++) {
      let bq = -1, bg = -Infinity;
      for (let q = 0; q < n; q++) { let g = 0; for (const r of inc[q]) g += cur[r] ? 1 : -1; if (g > bg) { bg = g; bq = q; } }
      v[bq] ^= 1; for (const r of inc[bq]) cur[r] ^= 1;
    }
    return v;
  }
  function decodeType(code, synd, type) { return traceType(code, synd, type).corr; }
  function decode(code, synd) { return { x: decodeType(code, synd, 'X'), z: decodeType(code, synd, 'Z') }; }
  function trace(code, synd) { return { X: traceType(code, synd, 'X'), Z: traceType(code, synd, 'Z') }; }

  // ---- distances ----
  // lightest nontrivial X logical: v with Hz v = 0, v ∉ rowspace(Hx); exact up to the node budget
  function distanceOf(code, type, wmax, budget) {
    const n = code.n, H = type === 'X' ? code.Hz : code.Hx, other = type === 'X' ? code.bx : code.bz;
    if (code.k <= 0) return { d: null, support: null, exact: true };
    const r = branchSearch(H, n, zeros(H.length), { logical: true, wmax: wmax || n, budget: budget || 4e5, accept: pick => !inSpan(other, fromSet(n, pick)) });
    return { d: r.w, support: r.support, exact: r.exact, atLeast: r.atLeast };
  }
  // the 2D figure of merit k·d²/n: codes with local checks in a plane obey k·d² ≤ c·n (Bravyi, Poulin & Terhal 2010)
  const kd2n = (n, k, d) => (k > 0 && d ? k * d * d / n : 0);

  // probability that k bare (unencoded) qubits suffer a logical-equivalent error under the same noise
  function bareRate(p, mode, k) {
    const one = mode === 'XZ' ? 1 - (1 - p) * (1 - p) : p;
    return 1 - Math.pow(1 - one, Math.max(1, k || 1));
  }
  // the pseudo-threshold: where the code's failure rate crosses the bare rate (interpolated on log scales);
  // pts = [{ p, v }] sorted by p. Returns null if no crossing inside the range.
  function crossing(pts, mode, k) {
    const f = q => Math.log(Math.max(q.v, 1e-9)) - Math.log(bareRate(q.p, mode, k));
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i]; if (!(a.v > 0 && b.v > 0)) continue;
      const fa = f(a), fb = f(b);
      if (fa < 0 && fb >= 0) { const t = fa / (fa - fb); return Math.exp(Math.log(a.p) + t * (Math.log(b.p) - Math.log(a.p))); }
    }
    return null;
  }

  // ---- Monte Carlo: logical failure rate of the decoder ----
  function logicalRate(code, p, mode, shots, rng) {
    let fail = 0;
    for (let t = 0; t < shots; t++) {
      const e = sample(code, p, mode, rng), c = decode(code, syndrome(code, e)), r = combine(e, c);
      if (!classify(code, r).trivial) fail++;
    }
    return fail / shots;
  }

  // a seedable generator (xorshift32), so tests and shared challenges are reproducible
  function rng(seed) { let x = (seed >>> 0) || 0x9e3779b9; return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }

  const BUILD = { rep: repetition, surface: rotatedSurface, toric, steane, shor };
  const api = { zeros, fromSet, weight, add, dot, reduce, rank, inSpan, nullspace, finalize, repetition, rotatedSurface, toric,
    steane, shor, custom, pauli, syndrome, combine, classify, sample, graphFor, matchDefects, lightest, decode, decodeType,
    distanceOf, logicalRate, rng, BUILD, branchSearch, tableFor, nonGraphQubit, trace, traceType, kd2n, bareRate, crossing };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QECEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
