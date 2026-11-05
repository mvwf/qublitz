// PD-12 — Photon Duel rules harness (design doc §9), same shape as
// mission_shield_harness.test.js: regex-extract the REAL source blocks out of
// quantum_chess.html and eval them in Node, so this can never drift from what
// the browser runs.
//
// D0 scope is §9 cases 1, 3, 7, 8 plus 1b (see below). Cases 2/4/5/6 arrive
// with the mechanics they test (eigenstate shield boundary and bot liveness in
// D1, biomes in D2) — they are listed as PENDING at the end of the run rather
// than silently omitted, so the gap is visible in the output.
//
// Case 1b is not in §9 and is the most important test here. DuelCore.rotateBloch
// is the one piece of duel physics that is not a direct QPhysics call (QPhysics
// rotates pure spinors; a decohering duel qubit is a mixed state with |r|<1
// that no spinor can express). 1b asserts rotateBloch agrees with
// QPhysics.rabiRotateState to machine precision on pure states — i.e. that it
// is the engine's own rotation generalized, not a second physics. If that ever
// fails, the duel has silently forked from the campaign's engine.
//
// Run: node tests/duel_harness.test.js [path-to-alternate-html]
'use strict';
const fs = require('fs');
const path = require('path');

const HTML = process.argv[2] || path.join(__dirname, '..', 'quantum_chess.html');
const lines = fs.readFileSync(HTML, 'utf8').split('\n');

function findBlock(startRe, endRe) {
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    if (start === -1 && startRe.test(lines[i])) { start = i; continue; }
    if (start !== -1 && endRe.test(lines[i])) return lines.slice(start, i + 1).join('\n');
  }
  throw new Error(`block not found: ${startRe}`);
}

const piSrc    = lines.find(l => /^const π\s+=/.test(l));
const cSrc     = findBlock(/^const C = \{/,                    /^};$/);
const gatesSrc = findBlock(/^const GATES = \{/,                /^};$/);
const qSrc     = findBlock(/^const QPhysics = \(\(\) => \{/,   /^\}\)\(\);$/);
const duelSrc  = findBlock(/^const DuelCore = \(\(\) => \{/,   /^\}\)\(\);$/);
const botSrc   = findBlock(/^const DuelBot = \(\(\) => \{/,    /^\}\)\(\);$/);

const code = [piSrc, cSrc, gatesSrc, qSrc, duelSrc, botSrc,
  'module.exports = { C, GATES, QPhysics, DuelCore, DuelBot };'].join('\n');
const { GATES, QPhysics, DuelCore, DuelBot } = eval('(function(){' + code + '\nreturn module.exports;})()');

const K = DuelCore.K;
const results = [];
const check = (name, pass, detail) => results.push({ name, pass, detail });
const approx = (a, b, eps) => Math.abs(a - b) <= eps;

const KET0 = { x: 0, y: 0, z: 1 };
const KET1 = { x: 0, y: 0, z: -1 };
const PLUS = { x: 1, y: 0, z: 0 };
const MINUS = { x: -1, y: 0, z: 0 };

// ── Case 1 — charge-to-θ mapping (§3.2: θ = π·min(t/1.2, 1)) ──────────
{
  const want = [[0.15, Math.PI / 8], [0.6, Math.PI / 2], [1.2, Math.PI], [2.0, Math.PI]];
  const got = want.map(([t]) => DuelCore.chargeToTheta(t));
  const ok = want.every(([, exp], i) => approx(got[i], exp, 1e-12));
  check('1. hold {0.15,0.6,1.2,2.0}s → θ {π/8,π/2,π,π}', ok,
    got.map(v => (v / Math.PI).toFixed(4) + 'π').join(', '));

  // A full-hold X-beam on |0⟩ must reach exactly 100% charge — the π-pulse
  // is the whole reason the bar is labelled in π-fractions.
  const hit = DuelCore.applyBeamHit(KET0, 'X', DuelCore.chargeToTheta(1.2));
  check('1a. full-hold X-beam on |0⟩ → charge 1.000 (π-pulse)',
    approx(DuelCore.odds(hit.bloch), 1, 1e-12),
    `charge=${DuelCore.odds(hit.bloch).toFixed(12)}`);
}

// ── Case 1b — rotateBloch IS QPhysics' rotation (see header) ──────────
{
  const ket0 = { alpha: { r: 1, i: 0 }, beta: { r: 0, i: 0 } };
  const states = [ket0,
    { alpha: { r: 0.6, i: 0 }, beta: { r: 0.8, i: 0 } },
    GATES.H(ket0), GATES.T(GATES.H(ket0)), GATES.Y(GATES.H(ket0))];
  const thetas = [0.1, Math.PI / 4, Math.PI / 2, Math.PI, 1.9 * Math.PI, 2 * Math.PI];
  let worst = 0;
  for (const th of thetas) for (const st of states) {
    const viaEngine = QPhysics.blochFromPure(QPhysics.rabiRotateState(st, th));
    const viaDuel = DuelCore.rotateBloch(QPhysics.blochFromPure(st), DuelCore.AXES.X, th);
    worst = Math.max(worst, Math.abs(viaEngine.x - viaDuel.x),
      Math.abs(viaEngine.y - viaDuel.y), Math.abs(viaEngine.z - viaDuel.z));
  }
  check('1b. rotateBloch ≡ QPhysics.rabiRotateState on pure states (30 cases)',
    worst < 1e-12, `max deviation ${worst.toExponential(2)}`);

  // Unitary: a rotation must never change |r| (that is decoherence's job).
  const mixed = { x: 0.3, y: -0.2, z: 0.5 };
  const before = Math.hypot(mixed.x, mixed.y, mixed.z);
  const after = DuelCore.rotateBloch(mixed, DuelCore.AXES.Y, 1.234);
  check('1c. rotation preserves |r| on a MIXED state (no purity leak)',
    approx(Math.hypot(after.x, after.y, after.z), before, 1e-12),
    `|r| ${before.toFixed(12)} → ${Math.hypot(after.x, after.y, after.z).toFixed(12)}`);
}

// ── Case 3 — Born rule: 10k seeded MEASUREs at P=0.7 → 0.7 ± 0.02 ─────
{
  // Build the exact state whose P(|1⟩)=0.7 by construction: z = 1-2p.
  const target = { x: 0, y: 0, z: 1 - 2 * 0.7 };
  const p = DuelCore.odds(target);
  const rand = DuelCore.mulberry32(0xC0FFEE);
  let wins = 0;
  const N = 10000;
  for (let i = 0; i < N; i++) if (DuelCore.measureRoll(target, rand).outcome === 1) wins++;
  const rate = wins / N;
  check('3. 10,000 seeded MEASUREs at P=0.7 → win-rate 0.7 ± 0.02',
    approx(p, 0.7, 1e-12) && approx(rate, 0.7, 0.02),
    `stated P=${p.toFixed(4)}, observed ${rate.toFixed(4)} (${wins}/${N})`);
}

// ── Case 7 — dud MEASURE resets the target to |0⟩ ─────────────────────
{
  const rand0 = () => 0.999999;   // forces outcome 0 against any p<1
  const rand1 = () => 0;          // forces outcome 1 against any p>0
  const charged = { x: 0, y: 0, z: 1 - 2 * 0.9 };

  const dud = DuelCore.measureRoll(charged, rand0);
  const okDud = dud.outcome === 0
    && approx(dud.bloch.z, 1, 1e-12)
    && approx(DuelCore.odds(dud.bloch), 0, 1e-12);
  check('7. dud MEASURE (outcome 0) resets target to |0⟩, charge 0', okDud,
    `outcome=${dud.outcome} z=${dud.bloch.z} charge=${DuelCore.odds(dud.bloch).toFixed(6)}`);

  const win = DuelCore.measureRoll(charged, rand1);
  check('7a. winning MEASURE (outcome 1) collapses to |1⟩', win.outcome === 1
    && approx(win.bloch.z, -1, 1e-12), `outcome=${win.outcome} z=${win.bloch.z}`);

  // After a dud the target is at an X-beam's disposal again from scratch:
  // one full π-pulse must take it back to 100%, or the "re-rotate from
  // scratch" cost isn't real.
  const back = DuelCore.applyBeamHit(dud.bloch, 'X', DuelCore.chargeToTheta(1.2));
  check('7b. post-dud |0⟩ is re-chargeable to 1.0 by one π-pulse',
    approx(DuelCore.odds(back.bloch), 1, 1e-12),
    `charge=${DuelCore.odds(back.bloch).toFixed(12)}`);
}

// ── Eigenstate no-op (D1's case 2 boundary is not asserted yet, but the
//    core invariant — parking at ±axis makes that axis inert — is live in
//    D0 because applyBeamHit already implements it) ─────────────────────
{
  const shielded = DuelCore.applyBeamHit(PLUS, 'X', Math.PI);
  const hit = DuelCore.applyBeamHit(KET0, 'X', Math.PI);
  check('e1. |+⟩ is inert to an X-beam (shielded, state unchanged)',
    shielded.shielded === true && approx(shielded.bloch.x, 1, 1e-12),
    `shielded=${shielded.shielded}`);
  check('e2. |0⟩ is NOT shielded from an X-beam', hit.shielded === false,
    `shielded=${hit.shielded}`);
  check('e3. |−⟩ also inert to X (either pole of the axis)',
    DuelCore.isEigenstate(MINUS, 'X') === true,
    `angle=${DuelCore.eigenstateAngleDeg(MINUS, 'X').toFixed(2)}°`);
}

// ── Decoherence routes through the campaign's own Lindblad step ───────
{
  const ref = QPhysics.lindbladStep(PLUS, { T1: K.T1_S, T2: K.T2_S, dt: 5 });
  const via = DuelCore.decayStep(PLUS, 5);
  check('d1. decayStep is exactly QPhysics.lindbladStep (T1=90s,T2=30s)',
    approx(ref.x, via.x, 1e-15) && approx(ref.z, via.z, 1e-15),
    `x ${via.x.toFixed(9)} vs ${ref.x.toFixed(9)}`);

  // Idling must actually cost you: a charged, coherent state loses odds.
  const start = DuelCore.applyBeamHit(KET0, 'X', Math.PI / 2).bloch;
  const idled = DuelCore.decayStep(start, 10);
  check('d2. idling 10s strictly reduces MEASURE odds (T1 relaxation)',
    DuelCore.odds(idled) < DuelCore.odds(start),
    `${DuelCore.odds(start).toFixed(4)} → ${DuelCore.odds(idled).toFixed(4)}`);

  // …and it shrinks the vector, which is what the HUD sphere shows.
  const rBefore = Math.hypot(start.x, start.y, start.z);
  const rAfter = Math.hypot(idled.x, idled.y, idled.z);
  check('d3. idling shrinks |r| (visible decoherence, not just relaxation)',
    rAfter < rBefore, `|r| ${rBefore.toFixed(4)} → ${rAfter.toFixed(4)}`);
}

// ── T-beam vs Z-beam: the anti-camp claim (Clifford / non-Clifford) ───
// DuelCore's header asserts that a Clifford phase rotation cannot break an
// X-camp but the non-Clifford T can. That is a physics claim doing tactical
// work, so it gets tested rather than asserted in prose.
{
  const rand = DuelCore.mulberry32(7);
  // A camper sits at |+⟩: an X-eigenstate, immune to the X attack beam.
  const camper = PLUS;
  check('t1. camper at |+⟩ is immune to the X attack beam',
    DuelCore.resolveBeam(camper, 'X', Math.PI, { rand }).shielded === true);

  // Z (Clifford, π about z) maps |+⟩ → |−⟩ — still an X-eigenstate, still camped.
  const afterZ = DuelCore.rotateBloch(camper, DuelCore.AXES.Z, Math.PI);
  check('t2. Z (Clifford) maps |+⟩→|−⟩ — camper STILL immune to X',
    DuelCore.isEigenstate(afterZ, 'X') === true,
    `|−⟩ x=${afterZ.x.toFixed(3)}, X-angle ${DuelCore.eigenstateAngleDeg(afterZ,'X').toFixed(1)}°`);

  // T (non-Clifford, π/4 about z) lands them at 45° — off every cardinal axis.
  const afterT = DuelCore.resolveBeam(camper, 'T', 0, { rand }).bloch;
  const stillCamped = DuelCore.isEigenstate(afterT, 'X');
  check('t3. T (non-Clifford) breaks the camp — no longer an X-eigenstate',
    stillCamped === false,
    `X-angle after T = ${DuelCore.eigenstateAngleDeg(afterT,'X').toFixed(1)}°`);
  check('t4. …and off EVERY cardinal axis (not just re-parked elsewhere)',
    DuelCore.isCardinal(afterT) === false,
    `cardinal=${DuelCore.isCardinal(afterT)}`);
  check('t5. T changes MEASURE odds by exactly 0 (it is pure phase)',
    approx(DuelCore.odds(afterT), DuelCore.odds(camper), 1e-12),
    `${DuelCore.odds(camper).toFixed(6)} → ${DuelCore.odds(afterT).toFixed(6)}`);
  // …so the X beam now lands on them.
  check('t6. after T, the X beam is no longer shielded',
    DuelCore.resolveBeam(afterT, 'X', Math.PI, { rand }).shielded === false);
}

// ── Decoherence hotspot: sanctuary bought with your own offence ───────
{
  const zones = [{ x: 100, y: 100, r: 60, t1: DuelCore.ZONE.HOT_T1, t2: DuelCore.ZONE.HOT_T2 }];
  check('h1. zoneAt finds a hotspot inside its radius, not outside',
    !!DuelCore.zoneAt(120, 110, zones) && !DuelCore.zoneAt(400, 400, zones));

  // Charged up, then 3s inside vs 3s outside.
  const charged = DuelCore.applyBeamHit(KET0, 'X', Math.PI).bloch;   // odds 1.0
  const inHot = DuelCore.decayStep(charged, 3, zones[0]);
  const outside = DuelCore.decayStep(charged, 3);
  check('h2. 3s in a hotspot discharges far faster than open space',
    DuelCore.odds(inHot) < DuelCore.odds(outside) - 0.2,
    `hotspot ${DuelCore.odds(inHot).toFixed(3)} vs open ${DuelCore.odds(outside).toFixed(3)}`);
  check('h3. …enough to stop being a MEASURE target (odds < 0.5 in 3s)',
    DuelCore.odds(inHot) < 0.5, `odds ${DuelCore.odds(inHot).toFixed(3)}`);
  // The cost: it is symmetric. You cannot hold a charge in there either.
  const tryCharge = DuelCore.decayStep(
    DuelCore.applyBeamHit(KET0, 'X', Math.PI).bloch, 2, zones[0]);
  check('h4. the sanctuary is not free — you cannot hold charge inside',
    DuelCore.odds(tryCharge) < 0.65, `odds after 2s ${DuelCore.odds(tryCharge).toFixed(3)}`);
}

// ── Gate fidelity: sloppy pulses miss the angle they aimed for ────────
{
  const N = 4000;
  function spread(fid) {
    const rand = DuelCore.mulberry32(99);
    const xs = [];
    for (let i = 0; i < N; i++) xs.push(DuelCore.applyFidelity(Math.PI, fid, rand));
    // Two-pass variance: the single-pass E[x²]−E[x]² form goes slightly
    // NEGATIVE here when σ is exactly 0 (every sample is π), and sqrt of that
    // is NaN — which is how this test first "failed" on a correct product.
    const mean = xs.reduce((a, b) => a + b, 0) / N;
    const varr = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / N;
    return { mean, sd: Math.sqrt(Math.max(0, varr)) };
  }
  const sloppy = spread(0.2), tuned = spread(0.95), perfect = spread(1);
  // σ is exactly 0 at fidelity 1 (sigmaFor(1)===0), so every sample IS π;
  // the ~1e-13 residual is float error in summing 4000 copies of π, not
  // scatter in the product.
  const perfectExact = DuelCore.applyFidelity(Math.PI, 1, DuelCore.mulberry32(1)) === Math.PI;
  check('f1. perfect fidelity delivers exactly the commanded θ (σ=0)',
    perfectExact && DuelCore.sigmaFor(1) === 0 && perfect.sd < 1e-9,
    `sigmaFor(1)=${DuelCore.sigmaFor(1)}, sample===π: ${perfectExact}`);
  check('f2. lower fidelity → wider angle error, mean still on target',
    sloppy.sd > tuned.sd && approx(sloppy.mean, Math.PI, 0.05),
    `σ: fid0.2=${sloppy.sd.toFixed(3)} rad, fid0.95=${tuned.sd.toFixed(3)} rad`);
  check('f3. σ matches the documented model σ0·(1−fidelity)',
    approx(sloppy.sd, DuelCore.sigmaFor(0.2), 0.03),
    `observed ${sloppy.sd.toFixed(4)} vs model ${DuelCore.sigmaFor(0.2).toFixed(4)}`);
}

// ── Bell trap: entangling resets the round to an honest coin flip ─────
{
  const amps = DuelCore.bellPair();
  check('b1. a fresh Bell pair reads 50% for BOTH halves',
    approx(DuelCore.bellOdds(amps, 0), 0.5, 1e-12) &&
    approx(DuelCore.bellOdds(amps, 1), 0.5, 1e-12),
    `${DuelCore.bellOdds(amps,0).toFixed(6)} / ${DuelCore.bellOdds(amps,1).toFixed(6)}`);

  // Correlated collapse: |Φ+⟩ must give the SAME outcome on both halves, every time.
  const rand = DuelCore.mulberry32(2024);
  let ones = 0, correlated = 0;
  for (let i = 0; i < 5000; i++) {
    const r = DuelCore.bellMeasure(DuelCore.bellPair(), 0, rand);
    if (r.outcome === r.partnerOutcome) correlated++;
    if (r.outcome === 1) ones++;
  }
  check('b2. |Φ+⟩ collapse is perfectly correlated (5000 trials)',
    correlated === 5000, `${correlated}/5000 matched`);
  check('b3. …and fair — 50/50 over those trials',
    Math.abs(ones / 5000 - 0.5) < 0.03, `P(1)=${(ones / 5000).toFixed(4)}`);

  // The comeback property: a 5%-vs-95% fight becomes 50/50 for both.
  const losing = { x: 0, y: 0, z: 1 - 2 * 0.05 };
  check('b4. entangling a 5% qubit resets it to a 50% coin flip (comeback)',
    approx(DuelCore.odds(losing), 0.05, 1e-9) &&
    approx(DuelCore.bellOdds(DuelCore.bellPair(), 0), 0.5, 1e-12),
    `0.05 → 0.50`);
}

// ── Zeno lock: a watched qubit cannot evolve ──────────────────────────
{
  const rand = DuelCore.mulberry32(5);
  const pinned = DuelCore.resolveBeam(KET0, 'X', Math.PI,
    { rand, zenoUntil: 10, now: 8 });
  check('z1. a Zeno-pinned target ignores an incoming beam entirely',
    pinned.zenoBlocked === true && approx(pinned.bloch.z, 1, 1e-12),
    `z stayed ${pinned.bloch.z}`);
  const freed = DuelCore.resolveBeam(KET0, 'X', Math.PI,
    { rand, zenoUntil: 10, now: 11 });
  check('z2. …and takes the hit normally once the lock expires',
    freed.zenoBlocked === false && approx(DuelCore.odds(freed.bloch), 1, 0.02),
    `charge ${DuelCore.odds(freed.bloch).toFixed(4)}`);
  check('z3. lock parameters are declared, not magic',
    DuelCore.ZENO.NEEDED === 2 && DuelCore.ZENO.WINDOW > 0 && DuelCore.ZENO.HOLD > 0,
    `${DuelCore.ZENO.NEEDED} measures within ${DuelCore.ZENO.WINDOW}s → ${DuelCore.ZENO.HOLD}s hold`);
}

// ── Mission gates: the campaign's shield property, ported ─────────────
// mission_shield_harness proves an X-only player can NEVER clear campaign
// missions 4-6. Same bar here: a "concept-free" duellist — one who charges,
// shoots and measures but never entangles, never spends a T, never fires from
// superposition, never Zeno-pins, never calibrates — must be structurally
// unable to clear missions 2-7, no matter how lucky the Born roll is.
{
  const MISSIONS = DuelCore.DUEL_MISSIONS;
  check('m0. seven duel missions, each naming its concept',
    MISSIONS.length === 7 && MISSIONS.every(m => m.concept && m.brief && m.win && m.gate),
    MISSIONS.map(m => m.concept).join(' · '));

  // The concept-free player: perfect aim, instant, but uses no concept.
  const naive = { entangled:false, usedT:false, superposedShot:false,
                  zenoLocked:false, fidelity:0.55, elapsed:1, measuredIdx:0 };
  const blocked = MISSIONS.filter(m => DuelCore.missionGate(m.id, naive).blocked).map(m => m.id + 1);
  check('m1. concept-free play is REFUSED by missions 3,4,5,6,7',
    JSON.stringify(blocked) === JSON.stringify([3,4,5,6,7]),
    `blocked: ${blocked.join(',') || 'none'}`);

  // Mission 1 is the tutorial (no gate) and mission 2 gates on the clock,
  // which a fast naive player legitimately beats — that is by design.
  check('m2. mission 1 has no gate; mission 2 gates on the T₁ clock only',
    !DuelCore.missionGate(0, naive).blocked &&
    !DuelCore.missionGate(1, naive).blocked &&
    DuelCore.missionGate(1, Object.assign({}, naive, { elapsed: 25 })).blocked,
    'fast run clears M2, slow run does not');

  // Each gate must open for its OWN concept and stay shut for the others —
  // otherwise a mission is teaching the wrong thing.
  const withConcept = {
    2: { entangled:true },
    3: { superposedShot:true },
    4: { usedT:true },
    5: { zenoLocked:true },
    6: { fidelity:0.95 },
  };
  let allOpen = true, crossTalk = [];
  for (const [idx, ev] of Object.entries(withConcept)) {
    const i = +idx;
    if (DuelCore.missionGate(i, Object.assign({}, naive, ev)).blocked) allOpen = false;
    // the same evidence must NOT unlock a different gated mission
    for (const other of [2,3,4,5,6]) {
      if (other === i) continue;
      if (!DuelCore.missionGate(other, Object.assign({}, naive, ev)).blocked)
        crossTalk.push(`M${other+1} opened by M${i+1}'s concept`);
    }
  }
  check('m3. each gate opens for its own concept', allOpen);
  check('m4. …and for NO other mission\'s concept (no cross-talk)',
    crossTalk.length === 0, crossTalk.join('; ') || 'clean');

  // Every refusal has to TELL the student what to do — a silent block is a
  // bug report, not a lesson.
  const reasons = [2,3,4,5,6].map(i => DuelCore.missionGate(i, naive).reason || '');
  check('m5. every refusal explains the fix in words',
    reasons.every(r => r.length > 30),
    `shortest: "${reasons.reduce((a,b)=>a.length<b.length?a:b).slice(0,52)}…"`);
}

// ── parkPulse: you cannot reach an X-eigenstate with an X-pulse ───────
{
  // The invariant that makes parkPulse necessary, stated directly.
  const start = { x: 0, y: 0.6, z: 0.8 };
  const xPulsed = DuelCore.rotateBloch(start, DuelCore.AXES.X, 1.1);
  check('p1. an X-rotation cannot change X-alignment (the trap)',
    approx(xPulsed.x, start.x, 1e-12),
    `x stayed ${xPulsed.x.toFixed(12)} — so an X-pulse can never park you on X`);

  // The two-phase controller must converge onto the axis within 2 pulses,
  // from many starting states — not just the easy ones.
  function parkIn(r0, target, maxPulses) {
    let r = r0, n = 0;
    while (n < maxPulses && !DuelCore.isEigenstate(r, target)) {
      // NOTE: phase 1 deliberately pulses the TARGET axis — not a no-op, it
      // clears the blocking component that the perpendicular pulse can't touch.
      const pp = DuelCore.parkPulse(r, target);
      r = DuelCore.rotateBloch(r, DuelCore.AXES[pp.beam], pp.theta);
      n++;
    }
    return { n, r };
  }
  const seeds = [ {x:0,y:0.6,z:0.8}, {x:0,y:0,z:1}, {x:0.3,y:-0.5,z:0.81},
                  {x:-0.7,y:0.7,z:0.14}, {x:0,y:1,z:0}, {x:0.6,y:0.0,z:-0.8} ];
  for (const target of ['X','Y']) {
    const runs = seeds.map(s => parkIn(s, target, 2));
    check(`p2-${target}. parkPulse converges onto the ${target}-axis in ≤2 pulses (6 starts)`,
      runs.every(r => r.n <= 2),
      `pulses needed: ${runs.map(r => r.n).join(', ')}`);
  }

  // …and a parked state really is immune to that axis's beam.
  const parked = parkIn(start, 'X', 2).r;
  check('p3. the parked state shields a full π X-beam',
    DuelCore.resolveBeam(parked, 'X', Math.PI, { rand: DuelCore.mulberry32(1) }).shielded === true,
    `X-angle ${DuelCore.eigenstateAngleDeg(parked,'X').toFixed(3)}°`);
}

// ── Case 6 (§9) — DuelBot liveness: 60s headless, every level ─────────
// The D0 stand-in drone was replaced by a real opponent, so §9.6's bar
// applies for the first time: over a full simulated minute each level must
// keep acting, must never produce a NaN, and must actually use the mechanics
// its tier unlocks. This runs the PURE decide() against a scripted world —
// no DOM, no canvas, no timers.
{
  function simulate(levelIdx, seconds, seed) {
    const rand = DuelCore.mulberry32(seed);
    const L = DuelBot.levelOf(levelIdx);
    const dt = 1/60;
    let me = { x: 900, y: 400, bloch: { x:0, y:0, z:1 }, tCharges: 2 };
    let foe = { x: 300, y: 400, bloch: { x:0, y:0, z:1 } };
    let acted = 0, nan = false, sinceAct = 0, maxGap = 0, tick = 0;
    let incoming = 1.5, shielded = 0, parkingAxis = null;
    const used = { fire:0, selfPulse:0, measure:0, entangle:0, T:0 };
    for (let t = 0; t < seconds; t += dt) {
      // decay both, as the real loop does
      me.bloch = DuelCore.decayStep(me.bloch, dt);
      foe.bloch = DuelCore.decayStep(foe.bloch, dt);
      // …and let the scripted player SHOOT BACK on the X-axis every ~1.5s.
      // Without this the bot is never exposed, so it correctly never needs to
      // park — which made the tiering assertion vacuous rather than false.
      incoming -= dt;
      if (incoming <= 0) {
        incoming = 1.5;
        const hit = DuelCore.resolveBeam(me.bloch, 'X', Math.PI/2, { rand, fidelity: 1 });
        if (!hit.shielded) me.bloch = hit.bloch; else shielded++;
      }
      tick -= dt;
      if (tick <= 0) {
        tick = L.tick;
        const view = {
          me: { x: me.x, y: me.y, inHot:false, tCharges: me.tCharges,
                beamReady:true, measReady:true, entReady:true },
          foe: { x: foe.x, y: foe.y, measReady:true },
          foeOdds: DuelCore.odds(foe.bloch),
          foeOddsFalling: false,
          myOdds: DuelCore.odds(me.bloch),
          myEigen: { X:DuelCore.isEigenstate(me.bloch,'X'), Y:DuelCore.isEigenstate(me.bloch,'Y'), Z:DuelCore.isEigenstate(me.bloch,'Z') },
          foeEigen:{ X:DuelCore.isEigenstate(foe.bloch,'X'), Y:DuelCore.isEigenstate(foe.bloch,'Y'), Z:DuelCore.isEigenstate(foe.bloch,'Z') },
          foeAxisCounts: { X: 12, Y: 1 },
          myMeasStreak: 1,
          myBloch: me.bloch,
          parkingAxis: (parkingAxis && !DuelCore.isEigenstate(me.bloch, parkingAxis)) ? parkingAxis : null,
          nearestZone: { x: 640, y: 200, r: 110 },
          bell: false, t,
        };
        const it = DuelBot.decide(view, levelIdx, rand, 0);
        parkingAxis = it.parkingAxis || null;   // shell mirrors the commitment
        if (!isFinite(it.aim) || !isFinite(it.moveX) || !isFinite(it.moveY) || !isFinite(it.hold)) nan = true;
        const did = it.fire || it.selfPulse || it.measure || it.entangle;
        if (it.fire) { used.fire++; if (it.beam === 'T') { used.T++; me.tCharges = Math.max(0, me.tCharges-1); } }
        if (it.selfPulse) { used.selfPulse++;
          const th = it.theta != null ? it.theta : DuelCore.chargeToTheta(it.hold);
          me.bloch = DuelCore.rotateBloch(me.bloch, DuelCore.AXES[it.beam], th); }
        if (it.measure) used.measure++;
        if (it.entangle) used.entangle++;
        if (it.fire) foe.bloch = DuelCore.rotateBloch(foe.bloch, DuelCore.AXES[it.beam==='Y'?'Y':'X'], DuelCore.chargeToTheta(it.hold));
        if (did) { acted++; maxGap = Math.max(maxGap, sinceAct); sinceAct = 0; }
      }
      sinceAct += dt;   // one clock only — adding L.tick here too double-counted
    }
    return { acted, nan, maxGap, used, shielded };
  }

  const runs = [0,1,2].map(i => simulate(i, 60, 1000 + i));
  check('6. every bot level keeps acting over 60s (no stalls, gap < 3s)',
    runs.every(r => r.acted > 20 && r.maxGap < 3),
    runs.map((r,i) => `${DuelBot.LEVELS[i].name}: ${r.acted} acts, max gap ${r.maxGap.toFixed(2)}s`).join(' · '));
  check('6a. …and never produces a NaN in 60s of decisions',
    runs.every(r => !r.nan));

  // Tiering must be real: a Cadet is not a Duelist with worse aim, it is a
  // Duelist that has not learned to park, phase or entangle.
  const cadet = runs[0].used, duelist = runs[1].used, menace = runs[2].used;
  check('6b. Cadet never parks, phases or entangles (its tier is locked)',
    cadet.selfPulse === 0 && cadet.T === 0 && cadet.entangle === 0,
    `cadet: park=${cadet.selfPulse} T=${cadet.T} bell=${cadet.entangle}`);
  check('6c. higher tiers DO park once the player starts landing hits',
    (duelist.selfPulse + menace.selfPulse) > 0,
    `duelist park=${duelist.selfPulse} · menace park=${menace.selfPulse}`);
  check('6h. parking actually WORKS — parked levels shield incoming beams',
    (runs[1].shielded + runs[2].shielded) > 0 && runs[0].shielded === 0,
    `shielded hits — cadet ${runs[0].shielded}, duelist ${runs[1].shielded}, menace ${runs[2].shielded}`);
  check('6d. aim error tightens monotonically Cadet → Duelist → Menace',
    DuelBot.LEVELS[0].sigma > DuelBot.LEVELS[1].sigma &&
    DuelBot.LEVELS[1].sigma > DuelBot.LEVELS[2].sigma,
    DuelBot.LEVELS.map(l => `${l.name} ${(l.sigma*180/Math.PI).toFixed(0)}°`).join(' > '));
  check('6e. the 3-loss assist widens aim by 15% and is level-relative',
    Math.abs(DuelBot.assistedSigma(DuelBot.LEVELS[1], 3) / DuelBot.LEVELS[1].sigma - 1.15) < 1e-12 &&
    DuelBot.assistedSigma(DuelBot.LEVELS[1], 2) === DuelBot.LEVELS[1].sigma,
    'streak 2 → no help; streak 3 → ×1.15');

  // The anti-camp rule must actually fire: a camped player draws a T.
  const campedView = {
    me:{x:900,y:400,inHot:false,tCharges:2,beamReady:true,measReady:false,entReady:true},
    foe:{x:300,y:400,measReady:false},
    foeOdds:0.3, foeOddsFalling:false, myOdds:0.1,
    myEigen:{X:true,Y:false,Z:false}, foeEigen:{X:true,Y:true,Z:false},
    foeAxisCounts:{X:9,Y:0}, myMeasStreak:0, nearestZone:null, bell:false, t:5,
  };
  const vsCamper = DuelBot.decide(campedView, 2, DuelCore.mulberry32(5), 0);
  check('6f. a player camped on BOTH attack axes draws a T-phase',
    vsCamper.beam === 'T' && vsCamper.fire === true,
    `beam=${vsCamper.beam} reason="${vsCamper.reason}"`);

  // …and the bot must NOT waste a scarce T when a free axis is still open.
  const oneAxisCamp = Object.assign({}, campedView, { foeEigen:{X:true,Y:false,Z:false} });
  const vsHalfCamp = DuelBot.decide(oneAxisCamp, 2, DuelCore.mulberry32(5), 0);
  check('6g. …but spends nothing when a free axis is open (Y instead of T)',
    vsHalfCamp.beam === 'Y', `beam=${vsHalfCamp.beam}`);
}

// ── Debrief card: three lines, and the tip must match what happened ───
{
  const base = { won:true, method:'measure', p:0.93, theta:'1.00',
                 shieldedAgainstYou:0, duds:0, selfPulses:1, hotspotTime:1 };
  const win = DuelCore.debriefLines(base);
  check('D1. a debrief is exactly three lines: outcome, concept, tip',
    win.length === 3 && /won/i.test(win[0]) && /^Concept:/.test(win[1]) && /^Tip:/.test(win[2]),
    win[0]);
  check('D2. it quotes the actual P and pulse from the round',
    win[0].includes('93%') && win[0].includes('1.00'), win[0]);

  const bell = DuelCore.debriefLines(Object.assign({}, base, { method:'bell-measure' }));
  check('D3. a Bell win names entanglement, not the Born rule',
    /entanglement/i.test(bell[1]), bell[1]);

  // The tip is keyed to the dominant pattern, so each pattern gets its own.
  const shielded = DuelCore.debriefLines(Object.assign({}, base, { won:false, shieldedAgainstYou:4 }));
  const noSelf   = DuelCore.debriefLines(Object.assign({}, base, { won:false, selfPulses:0 }));
  const duds     = DuelCore.debriefLines(Object.assign({}, base, { won:false, duds:3 }));
  check('D4. tips are pattern-specific (eigenstate / self-pulse / duds differ)',
    new Set([shielded[2], noSelf[2], duds[2]]).size === 3 &&
    /T-phase/.test(shielded[2]) && /Right-mouse/.test(noSelf[2]) && /lockout/.test(duds[2]),
    shielded[2].slice(0, 58) + '…');
  check('D5. a loss never claims you won',
    /lost/i.test(shielded[0]) && !/You won/.test(shielded[0]), shielded[0]);
}

// ── Case P — free-duel power-ups and the measurement reveal ───────────
{
  // P1. Detuned rotation at Δ = 0 is the resonant rotation.
  const r0 = { x: 0.3, y: -0.4, z: Math.sqrt(1 - 0.25) };
  const a = DuelCore.detunedRotate(r0, DuelCore.AXES.X, 1.1, 0);
  const b = DuelCore.rotateBloch(r0, DuelCore.AXES.X, 1.1);
  check('P1. detunedRotate(Δ=0) equals rotateBloch',
    approx(a.x, b.x, 1e-12) && approx(a.y, b.y, 1e-12) && approx(a.z, b.z, 1e-12));

  // P2. From |0⟩ the detuned drive never beats the Rabi ceiling Ω²/(Ω²+Δ²),
  //     and it reaches it (at a generalised-π pulse).
  const d = DuelCore.PICKUPS.detune.dOverO, ceil = 1 / (1 + d * d);
  let pmax = 0;
  for (let i = 0; i <= 4000; i++) {
    const th = i / 4000 * 2 * Math.PI;
    for (const ax of ['X', 'Y'])
      pmax = Math.max(pmax, DuelCore.odds(DuelCore.detunedRotate(KET0, DuelCore.AXES[ax], th, d)));
  }
  check('P2. detuned X/Y drive from |0⟩ peaks at P(1) = 1/(1+(Δ/Ω)²)',
    pmax <= ceil + 1e-9 && pmax >= ceil - 1e-4, `max ${pmax.toFixed(5)}, ceiling ${ceil.toFixed(5)}`);

  // P3. Rotation stays unitary (|r| preserved).
  const m = DuelCore.detunedRotate(r0, DuelCore.AXES.Y, 2.3, d);
  check('P3. detunedRotate preserves |r|',
    approx(Math.hypot(m.x, m.y, m.z), Math.hypot(r0.x, r0.y, r0.z), 1e-12));

  // P4. resolveBeam: a detuned target cannot be shield-parked, T is unaffected.
  const fixed = () => 0.5;   // fidelity 1 → σ = 0, so the draw does not matter
  const onPlus = DuelCore.resolveBeam(PLUS, 'X', Math.PI, { fidelity: 1, rand: fixed, detune: d });
  const tOnly = DuelCore.resolveBeam(PLUS, 'T', 0, { fidelity: 1, rand: fixed, detune: d });
  const tRef = DuelCore.resolveBeam(PLUS, 'T', 0, { fidelity: 1, rand: fixed });
  check('P4. detuned |+⟩ is not shielded from X; T-phase ignores detuning',
    onPlus.detuned === true && onPlus.shielded === false &&
    approx(tOnly.bloch.x, tRef.bloch.x, 1e-12) && approx(tOnly.bloch.y, tRef.bloch.y, 1e-12));

  // P5. DRIVE ×2: same hold, twice the angle, still capped at π.
  check('P5. driveTheta doubles θ for a given hold and caps at π',
    approx(DuelCore.driveTheta(0.3, 2), 2 * DuelCore.chargeToTheta(0.3), 1e-12) &&
    approx(DuelCore.driveTheta(0.6, 2), Math.PI, 1e-12) &&
    approx(DuelCore.driveTheta(1.0, 2), Math.PI, 1e-12));

  // P6. Reduced state of |Φ+⟩ is maximally mixed; of |0⟩|1⟩ is the poles.
  const bell = QPhysics.Bell.phiPlus();
  const ra = DuelCore.reducedBloch(bell, 0), rb = DuelCore.reducedBloch(bell, 1);
  const z = { r: 0, i: 0 }, one = { r: 1, i: 0 };
  const prod = [z, one, z, z];                       // |01⟩: qubit0 = |0⟩, qubit1 = |1⟩
  const p0 = DuelCore.reducedBloch(prod, 0), p1 = DuelCore.reducedBloch(prod, 1);
  const plusPlus = [{r:.5,i:0},{r:.5,i:0},{r:.5,i:0},{r:.5,i:0}];
  const pp = DuelCore.reducedBloch(plusPlus, 0);
  check('P6. reducedBloch: Φ+ → (0,0,0); |01⟩ → |0⟩,|1⟩; |++⟩ → +x',
    [ra, rb].every(v => approx(v.x, 0, 1e-12) && approx(v.y, 0, 1e-12) && approx(v.z, 0, 1e-12)) &&
    approx(p0.z, 1, 1e-12) && approx(p1.z, -1, 1e-12) && approx(pp.x, 1, 1e-12) && approx(pp.z, 0, 1e-12));

  // P7. measureRoll exposes the draw it used, and outcome = [draw < p].
  let consistent = true;
  const rng = DuelCore.mulberry32(7);
  for (let i = 0; i < 500; i++) {
    const th = (i % 50) / 50 * Math.PI;
    const r = DuelCore.rotateBloch(KET0, DuelCore.AXES.X, th);
    const res = DuelCore.measureRoll(r, rng);
    if (!(res.draw >= 0 && res.draw < 1) || res.outcome !== (res.draw < res.p ? 1 : 0)) consistent = false;
  }
  check('P7. measureRoll: outcome is 1 exactly when draw < P(|1⟩)', consistent);

  // P8. Pickup rotation is fixed and every kind is catalogued.
  check('P8. PICKUP_CYCLE covers six catalogued kinds, calib and tgate included',
    DuelCore.PICKUP_CYCLE.length === 6 && DuelCore.PICKUP_CYCLE.every(k => DuelCore.PICKUPS[k]) &&
    DuelCore.PICKUP_CYCLE.includes('calib') && DuelCore.PICKUP_CYCLE.includes('tgate'));
}

// ── Case 8 — the campaign engine is untouched ─────────────────────────
{
  const st = QPhysics.selfTest();
  check('8. QPhysics.selfTest() still green (engine unchanged by the duel)',
    st.pass === true, `${st.tests.filter(t => t[1]).length}/${st.tests.length} engine tests`);
}

// ── Report ───────────────────────────────────────────────────────────
console.log('=== PD-12 Photon Duel harness (D0 scope) ===\n');
let failed = 0;
for (const r of results) {
  if (!r.pass) failed++;
  console.log(`  ${r.pass ? '✓' : '✗'} ${r.name}`);
  if (r.detail) console.log(`      ${r.detail}`);
}
console.log('\n  PENDING (arrive with the mechanics they test):');
console.log('    · §9.5 phase-rift composite rotation — D2 (biome bands)');
console.log(failed ? `\n${failed} CHECK(S) FAILED ✗` : '\nALL CHECKS PASSED ✓');
process.exit(failed ? 1 : 0);
