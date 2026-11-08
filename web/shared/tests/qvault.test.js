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
console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
process.exit(fails ? 1 : 0);
