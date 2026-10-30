/* fridge-engine.js: browser port of the model in
 * pages/Dilution_Refrigerator_Noise_Explorer.py. No dependencies. Loads as a
 * classic <script> (window.FridgeEngine) or from Node (require), so the same file
 * is what the page runs and what tests/fridge-engine.test.js checks against Python.
 *
 * Model, same as the Python page:
 *   n̄(T, f) = 1 / (exp(h·f / kB·T) − 1)                    Bose–Einstein occupation
 *   each attenuator is a thermal beam splitter at its stage temperature:
 *     n_out = n_in / L + (1 − 1/L)·n̄(T_i, f),   L = 10^(A_i/10)
 *   the input to the first stage is thermal at that stage's temperature
 *   T_eff inverts n̄ (n floored at 1e-20); noise reduction = 10·log10(n_room / n_eff)
 *   drive power drops by each stage's attenuation; the cable insertion loss is
 *   applied to the drive at the MXC only, not to the noise.
 * Frequencies are in GHz, temperatures in K, powers in dBm.
 */
(function (root) {
  'use strict';
  const kB = 1.380649e-23;        // J/K
  const h = 6.62607015e-34;       // J·s

  /** Mean thermal photon number at temperature T [K] and frequency f [GHz]. */
  function thermalN(T, fGHz) {
    const x = h * fGHz * 1e9 / (kB * T);     // T = 0 gives x = ∞ and n = 0, as numpy does
    return 1 / (Math.exp(x) - 1);
  }

  /** Effective temperature [K] for occupation n at frequency f [GHz]. */
  function nToTeff(n, fGHz) {
    const hfOverK = h * fGHz * 1e9 / kB;
    return hfOverK / Math.log(1 + 1 / Math.max(n, 1e-20));
  }

  /**
   * Propagate thermal noise through the attenuator chain, like propagate_chain().
   * @returns {{nEff:Float64Array[], TEff:Float64Array[]}} one row per stage, one column per frequency
   */
  function propagateChain(freqsGHz, stageTemps, stageAttenDB) {
    const S = stageTemps.length, N = freqsGHz.length;
    const nEff = [], TEff = [];
    for (let i = 0; i < S; i++) { nEff.push(new Float64Array(N)); TEff.push(new Float64Array(N)); }
    for (let k = 0; k < N; k++) {
      const f = freqsGHz[k];
      let nIn = thermalN(stageTemps[0], f);
      for (let i = 0; i < S; i++) {
        const L = 10 ** (stageAttenDB[i] / 10);
        const nOut = L === 1 ? nIn : nIn / L + (1 - 1 / L) * thermalN(stageTemps[i], f);
        nEff[i][k] = nOut;
        TEff[i][k] = nToTeff(nOut, f);
        nIn = nOut;
      }
    }
    return { nEff, TEff };
  }

  /** 10·log10(n_room / n_eff), with n_eff floored at 1e-30 as in the page. */
  function noiseReductionDB(nRoom, nEff) {
    return 10 * Math.log10(nRoom / Math.max(nEff, 1e-30));
  }

  const dBmToW = P => 10 ** ((P - 30) / 10);
  const WTodBm = P => 10 * Math.log10(Math.max(P, 1e-30)) + 30;

  /** Drive power [dBm] at the output of each stage: input minus cumulative
   *  attenuation, with the cable loss taken off the last (MXC) stage only. */
  function drivePowers(PinDBm, stageAttenDB, cableLossDB) {
    let cum = 0;
    const P = stageAttenDB.map(A => PinDBm - (cum += A));
    P[P.length - 1] -= cableLossDB;
    return P;
  }

  /** Coherent drive photon flux [photons/s] for power P [dBm] at f [GHz]. */
  const photonFlux = (PdBm, fGHz) => dBmToW(PdBm) / (h * fGHz * 1e9);

  /** Matches format_temp(): mK below 0.5 K, otherwise K to 3 significant figures. */
  function formatTemp(T) {
    if (T < 0.5) return `${(T * 1e3).toFixed(1)} mK`;
    return `${+T.toPrecision(3)} K`;
  }

  function linspace(a, b, n) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = n === 1 ? a : a + (b - a) * i / (n - 1);
    return out;
  }

  const api = { kB, h, thermalN, nToTeff, propagateChain, noiseReductionDB, dBmToW, WTodBm,
    drivePowers, photonFlux, formatTemp, linspace };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FridgeEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
