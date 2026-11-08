// Checks web/shared/qvault.js (the Q-Vault bridge) on a small fixture in Q-Vault's CSV format, so it runs offline.
// Usage: node web/shared/tests/qvault.test.js
'use strict';
const path = require('path');
const QV = require('../qvault.js');
const F = require(path.join(__dirname, '../../qublitz/fridge-engine.js'));
let fails = 0;
const check = (name, ok, detail) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  · ' + detail : ''}`); if (!ok) fails++; };

const csv = 'a,b,c\n1,"x, ""y""",3\n2,"two\nlines",\n';
const P = QV.parseCSV(csv);
check('CSV: quoted commas, doubled quotes and line breaks', P.length === 2 && P[0].b === 'x, "y"' && P[1].b === 'two\nlines' && P[1].c === '');

const HEAD = 'citekey,setup_id,devices,fridge,base_temperature_mK,shielding,line,role,line_label,line_count,stated_attenuation_dB,position,stage,part,dB,cutoff_GHz,gain_dB,count,model,optional,details,data_location,source_version\n';
const rows = QV.parseCSV(HEAD +
  'bal,W1,D1,,,,1,input,,,,1,4K,attenuator,20,,,,,,,Fig. S1,VoR\n' +
  'bal,W1,D1,,,,1,input,,,,2,still,attenuator,6,,,,,,,Fig. S1,VoR\n' +
  'bal,W1,D1,,,,1,input,,,,3,cold_plate,attenuator,6,,,,,,,Fig. S1,VoR\n' +
  'bal,W1,D1,,,,1,input,,,,4,mxc,attenuator,20,,,,,,,Fig. S1,VoR\n' +
  'bal,W1,D1,,,,1,input,,,,5,mxc,lowpass_filter,,8,,,,,,Fig. S1,VoR\n' +
  'gan,W1,D2,,20,,1,input,,,,1,4K,attenuator,20,,,,,,,Fig. 5,VoR\n' +
  'gan,W1,D2,,20,,1,input,,,,3,mxc,attenuator,30,,,2,,,,Fig. 5,VoR\n' +
  'odd,W1,D1,,,,1,input,,,,1,4K,attenuator,,,,,,,,Fig. 2,VoR\n' +
  'out,W1,D1,,,,2,output,,,,1,mxc,attenuator,10,,,,,,,Fig. 2,VoR\n');
const C = QV.chains(rows);
check('chains: input lines with every dB stated; output lines and unknown dB left out', C.length === 2 && C.map(c => c.citekey).join() === 'bal,gan');
const bal = C[0], gan = C[1];
check('chain totals count repeated attenuators', bal.total === 52 && gan.total === 80, `${bal.total}, ${gan.total}`);
check('a chain with a cold-plate attenuator cannot be loaded into the five-stage explorer; one without can', !bal.mappable && gan.mappable);
check('base temperature is read', gan.base_mK === 20);

// the noise of a chain equals the explorer's own model, stage by stage
const temps = { room_temperature: 300, '50K': 50, '4K': 4, still: 0.7, cold_plate: 0.1, mxc: 0.01 };
const n = QV.chainNoise(bal, F, temps, 5);
let m = F.thermalN(300, 5); for (const [T, dB] of [[4, 20], [0.7, 6], [0.1, 6], [0.01, 20]]) { const L = 10 ** (dB / 10); m = m / L + (1 - 1 / L) * F.thermalN(T, 5); }
check('chain noise = attenuator-by-attenuator beam-splitter model', Math.abs(n / m - 1) < 1e-12, n.toExponential(3));
// loading gan's chain into the explorer's five rows (300 K, 50 K, 4 K, still, MXC) gives the same number
const five = F.propagateChain([5], [300, 50, 4, 0.7, 0.01], [0, 0, 20, 0, 60]).nEff[4][0];
check('a mappable chain loaded into the explorer gives the same n_eff', Math.abs(QV.chainNoise(gan, F, temps, 5) / five - 1) < 1e-12);

const MH = 'citekey,device_id,quantity,value,unit,statistic,qualifier,data_location,status,type,base_metal,substrate,freq_GHz\n';
const R = QV.resonatorQs(QV.parseCSV(MH + 'cro,D1,Qi_SP,15000000,1,best,,Fig. 3,current,resonator_cpw,Ta,sapphire,\nkim,D3,Qi_SP,272000,1,single,,Table 1,current,resonator_cpw,NbN,Si,9.796\n' +
  'kim,D3,Qi_SP,1,1,single,,Table 1,superseded,resonator_cpw,NbN,Si,9.796\nx,D1,T1,5,us,best,,t,current,transmon,Al,Si,\n'));
check('resonator Qs: current resonator Qi rows only, best first', R.length === 2 && R[0].value === 15e6 && R[1].f === 9.796);
check('citations drop a stray leading "and"', QV.cite({ first_author: 'and Acharya', year: '2024' }, 'k') === 'Acharya 2024');

// ── links from Q-Vault and the real qubit loader ──
check('parseQV: citekey~device~label, labels may hold ~', JSON.stringify(QV.parseQV('bal2023systematic~D1~ST1~17')) === JSON.stringify({ citekey: 'bal2023systematic', device: 'D1', label: 'ST1~17' }));
check('parseQV rejects junk', QV.parseQV('') === null && QV.parseQV('<script>~D1') === null);
const PH = 'citekey,device_id,label,type,f01_GHz,T1_us,T2R_us,T2E_us,anharmonicity_MHz,Qi_SP_x,f_res_GHz\n';
const PR = QV.parseCSV(PH + 'bal,D1,ST1-17,transmon,5.005,102,,155,,,\nbal,D1,ST1-18,transmon,4.746,118,,,-210,,\nsom,D1,,fluxonium,0.44,1000,,,,,\n' +
  'kim,D3,,resonator_cpw,,,,,,272000,9.796\nx,D2,,transmon,6,30,20,,-300,,\n');
check('findParam: exact label, and no silent fallback to a sibling when the label is wrong', QV.findParam(PR, QV.parseQV('bal~D1~ST1-18')).T1_us === '118' && QV.findParam(PR, QV.parseQV('bal~D1~nope')) === null);
check('findParam: a device without a label', QV.findParam(PR, QV.parseQV('kim~D3')).Qi_SP_x === '272000');
const L = QV.labQubit({ f01: 5.005, t1: 102, t2: 155 });
check('labQubit: f01 exact, 1 µs real = 1 ns in the lab, T2/T1 kept', L.ok && L.omegaQ === 5.005 && L.T1 === 102 && L.T2 === 155 && QV.TIME_SCALE === 1000);
const Lc = QV.labQubit({ f01: 4, t1: 30000, t2: 20000 });
check('labQubit: T1 capped at 10 µs keeps T2/T1', Lc.ok && Lc.T1 === 1e4 && Math.abs(Lc.T2 / Lc.T1 - 2 / 3) < 1e-3 && Lc.notes.length === 1);
const Lx = QV.labQubit({ f01: 5, t1: 50, t2: 140 });
check('labQubit: T2 above 2T1 (not physical) is set to 2T1, with a note', Lx.T2 === 100 && /2T₁/.test(Lx.notes[0]));
check('labQubit: no T2 means T2 = 2T1', QV.labQubit({ f01: 5, t1: 80 }).T2 === 160);
check('labQubit: a fluxonium below 1 GHz is refused with a reason', !QV.labQubit({ f01: 0.44, t1: 1000 }).ok && /1–10 GHz/.test(QV.labQubit({ f01: 0.44, t1: 1000 }).why));

// ── real-quantity statistics ──
const vT1 = QV.realValues(PR, QV.REAL['transmon.T1']);
check('realValues: only the listed device types', vT1.length === 3 && vT1.every(v => v.citekey !== 'som'));
const st = QV.realStats(vT1);
check('realStats: median, quartiles, counts', st.n === 3 && st.median === 102 && st.lo === 66 && st.hi === 110 && st.papers === 2, JSON.stringify(st));
check('realValues: signed columns as magnitudes when asked (anharmonicity)', QV.realValues(PR, QV.REAL['transmon.alpha']).map(v => v.v).join() === '210,300');
const MR = QV.parseCSV('citekey,device_id,quantity,value,qualifier,status\na,D1,F2Q,99.5,,current\nb,D1,F2Q,99.9,,current\nc,D1,F2Q,99.0,,superseded\nd,D1,readout_error,0.08,<,current\n');
check('measurement quantities: superseded rows and bounds (<, >) left out', QV.realValues(MR, QV.REAL['qubit.F2Q']).length === 2 && QV.realValues(MR, QV.REAL['qubit.readout']).length === 0);
const ML = QV.parseCSV('citekey,device_id,quantity,value,qualifier,status,notes\na,D1,QEC_Lambda,2.14,,current,Distance-7 surface code\na,D1,QEC_Lambda,8.4,,current,Repetition code average\n');
check('Λ: surface-code values only, not the repetition code', QV.realValues(ML, QV.REAL['qec.Lambda']).map(v => v.v).join() === '2.14');
check('realStats of nothing is null, and the text says so', QV.realStats([]) === null && /No published/.test(QV.realText('qubit.readout', null)));
check('realText: median and middle half with units', /median 102 µs/.test(QV.realText('transmon.T1', st)) && /3 values from 2 papers/.test(QV.realText('transmon.T1', st)));
check('fmt: big and small numbers in powers of ten', QV.fmt(3.4e6, '') === '3.4×10⁶' && QV.fmt(2.1e-6, '') === '2.1×10⁻⁶' && QV.fmt(99.5, '%') === '99.5%');
check('every real quantity names a Q-Vault page', Object.keys(QV.REAL).every(k => QV.realURL(k).startsWith(QV.SITE)));

// ── dispersive readout rows (Readout Lab) ──
const RP = QV.parseCSV('citekey,device_id,label,type,chi_MHz,two_chi_MHz,kappa_MHz,T1_us,f_readout_GHz,P_excited_percent\n' +
  'a,D1,Q1,transmon,-2.3,,2.0,7.4,6.9,\nb,D2,Q2,transmon,,0.3,0.288,559,,\nc,D1,G,fluxonium,0.22,,2.19,1070,6.69,1.5\nd,D1,Q,transmon,0.5,,,100,,\ne,D1,Q,transmon,,,1.2,100,,\n');
const RM = QV.parseCSV('citekey,device_id,quantity,value,notes\nc,D1,chi,0.22,"Designed. The paper does not say whether it is half or the full g-e shift."\na,D1,chi,-2.3,Half shift.\n');
const RR = QV.readoutRows(RP, RM);
check('readoutRows: only rows with both κ and χ (or 2χ)', RR.map(r => r.citekey).join() === 'a,b,c');
check('readoutRows: χ as a magnitude, and 2χ halved', RR[0].chi === 2.3 && RR[1].chi === 0.15 && RR[1].chiFrom === 'two_chi' && RR[0].chiFrom === 'chi');
check('readoutRows: unclear χ convention is flagged from the measurement note', RR[2].unsure && !RR[0].unsure && !RR[1].unsure);
check('readoutRows: T₁, readout frequency and leftover population carried', RR[2].T1 === 1070 && RR[2].fr === 6.69 && RR[2].pexc === 1.5 && RR[1].pexc === null);
check('χ chip: two_chi rows count, halved', QV.realValues(RP, QV.REAL['transmon.chi']).map(v => v.v).sort().join() === '0.15,0.5,2.3');

// ── mystery qubits (Virtual Qubit Lab): real transmons from Q-Vault ──
const MP = QV.parseCSV('citekey,device_id,label,type,f01_GHz,T1_us,T2R_us,T2E_us,base_metal\n' +
  'b,D1,Q2,transmon,4.4,49,95,,Al\na,D1,Q1,transmon,5.2,120,,150,Ta\nc,D1,,transmon,7.1,100,,,Nb\nd,D1,,transmon,4.8,10,,,Al\n' +
  'e,D1,,fluxonium,0.4,500,,,Al\nf,D2,Q3,transmon,6.0,400,,,Ta\n');
const pool = QV.mysteryPool(MP);
check('mysteryPool: transmons in 3.5–6.5 GHz with 30–600 µs T₁, sorted by id', pool.map(r => r.citekey).join() === 'a,b,f');
const picks = new Set(['K7RX4P', 'ABCD12', 'ZZZZZZ', 'Q1Q1Q1', 'HELLO1', 'X9X9X9'].map(c => QV.mysteryPick(pool, c).citekey));
check('mysteryPick: same code → same qubit; codes spread over the pool', QV.mysteryPick(pool, 'k7rx4p ') === QV.mysteryPick(pool, 'K7RX4P') && picks.size >= 2);
const mq = QV.mysteryQubit(pool[1], 'K7RX4P');
check('mysteryQubit: f₀₁ exact, T₁ and T₂ in lab ns = chip µs, paper values kept', mq.omegaQ === 4.4 && mq.T1 === 49 && mq.T2 === 95 && mq.real.t1 === 49 && mq.real.kind === 'Ramsey');
check('mysteryQubit: no T₂ in the paper → 2T₁, echo used when no Ramsey', QV.mysteryQubit(pool[2], 'A').T2 === 800 && QV.mysteryQubit(pool[0], 'A').real.kind === 'echo');
const rabis = ['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(c => QV.mysteryQubit(pool[1], c).omegaRabi * 1e3);
check('mysteryQubit: drive 100–300 kHz on the chip, π pulse ≤ 1% of T₁ where possible', rabis.every(r => r >= 5e4 / 49 - 0.1 || r === 300) && rabis.every(r => r >= 100 && r <= 300));
check('mysteryQubit: a long-T₁ qubit can get a gentle drive', ['A', 'B', 'C', 'D', 'E', 'F', 'G'].some(c => QV.mysteryQubit(pool[2], c).omegaRabi * 1e3 < 200));
const tok = QV.mysteryToken(pool[2], 'K7RX4P');
check('device token: round-trips with its code, not with another, and hides the id', QV.mysteryFromToken(pool, tok, 'K7RX4P') === pool[2] && QV.mysteryFromToken(pool, tok, 'OTHER1') !== pool[2] && !/f~D2/.test(tok) && /^[A-Za-z0-9_-]+$/.test(tok));
check('device token: garbage gives null', QV.mysteryFromToken(pool, '!!!', 'K7RX4P') === null && QV.mysteryFromToken(pool, 'AAAA', 'K7RX4P') === null);

// ── funding: the award numbers agree wherever they are stated ──
const fs = require('fs'), rd = f => fs.readFileSync(path.join(__dirname, '../../..', f), 'utf8');
const awards = t => ['2540460', '2553633'].every(id => t.includes('AWD_ID=' + id)) && /Unitary Foundation/.test(t);
check('funding: site footer (sitebar.js), hub and README name the Unitary Foundation and both NSF awards with links',
  awards(rd('web/shared/sitebar.js')) && awards(rd('web/index.html')) && awards(rd('README.md')));
console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
process.exit(fails ? 1 : 0);
