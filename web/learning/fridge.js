/* fridge.js: learning material for the Fridge Noise Explorer (qublitz/fridge-noise.html).
 * Numbers below come from fridge-engine.js with the page's default stage temperatures (300 K, 50 K, 4 K, 0.7 K,
 * 10 mK) at 5 GHz: n̄(300 K) ≈ 1250, n̄(4 K) ≈ 16, n̄(100 mK) ≈ 0.10; at 100 mK n̄ is 0.31 at 3 GHz and 0.06 at 6 GHz;
 * 20/20/20 dB at 4 K/Still/MXC leaves n_eff ≈ 0.027 at the MXC, 60 dB at 4 K alone ≈ 16, 60 dB at the MXC ≈ 1.3×10⁻³. */
if (window.QL) QL.learning({
  id: 'fridge',
  title: 'Fridge Noise Explorer',
  tagline: 'A qubit sits at 10 mK, but the cable that drives it starts at room temperature. Follow the thermal ' +
    'photons down the drive line of a dilution refrigerator and decide where the attenuators should go.',
  fun: 'it is a puzzle with a budget: you have a fixed number of decibels to spend, and the photon counter at the ' +
    'qubit tells you right away whether you spent them well.',
  objectives: [
    { id: 'nbar', text: 'Predict how the thermal photon number n̄ in a mode changes with temperature and with frequency.' },
    { id: 'beam_splitter', text: 'Explain how an attenuator removes incoming noise and adds its own noise at its stage temperature.' },
    { id: 'placement', text: 'Choose where to put attenuation so the qubit at the mixing chamber (MXC) sees few thermal photons.' },
    { id: 'drive', text: 'Relate attenuation to the drive power that reaches the qubit, and say why labs attenuate anyway.' },
  ],
  howTo: [
    'Each row is one plate of the fridge, from 300 K at the top to the MXC at the bottom. The box in the middle column is the attenuator on that plate, in dB.',
    'Type a value into an attenuator (0 to 60 dB) or a plate temperature, or press one of the <b>Try</b> presets. Every number updates at once.',
    'Read the right-hand column: n<sub>eff</sub> is the mean number of thermal photons in the line after that plate, and T<sub>eff</sub> is the temperature that would give the same n<sub>eff</sub>. The MXC row is what the qubit sees.',
    'Use the plot tabs to see the same quantities across the frequency band, and the reference-frequency slider to choose where the table and rows are evaluated.',
    'Work through the challenges at the top. Each one ticks itself when the line meets its goal.',
  ],
  goals: [
    'Complete the four challenges: see the bare line, waste 60 dB, spend 60 dB well (below 0.01 photons at the MXC with at most 30 dB there), and warm the MXC.',
    'Explain in one sentence why the last attenuator has to sit on a cold plate.',
  ],
  quiz: [
    { id: 'room', objective: 'nbar',
      q: 'About how many thermal photons are in a 5 GHz mode at room temperature (300 K)?',
      choices: ['About 0.01', 'About 1', 'About 1250', 'About a million'],
      answer: 2,
      explain: 'When k<sub>B</sub>T ≫ hf, n̄ ≈ k<sub>B</sub>T/hf. At 5 GHz, hf/k<sub>B</sub> ≈ 0.24 K, so at 300 K n̄ ≈ 300/0.24 ≈ 1250. That is the noise a bare cable would carry down to the qubit.' },
    { id: 'freq', objective: 'nbar',
      q: 'At a fixed temperature of 100 mK, what happens to the thermal photon number n̄ as the frequency goes from 3 GHz to 6 GHz?',
      choices: ['It falls, from about 0.3 to about 0.06', 'It rises, because higher frequencies carry more power', 'It stays the same: n̄ depends only on temperature', 'It doubles, in proportion to the frequency'],
      answer: 0,
      explain: 'n̄ = 1/(e<sup>hf/k<sub>B</sub>T</sup> − 1). A higher-frequency photon costs more energy, so at the same temperature there are fewer of them. When hf ≫ k<sub>B</sub>T the fall is exponential.' },
    { id: 'warm4k', objective: 'beam_splitter',
      q: 'You put all 60 dB on the 4 K plate and none below it (standard temperatures, 5 GHz). Roughly how many thermal photons reach the MXC?',
      choices: ['About 0.001: room noise 1250 divided by 10⁶', 'About 16: the 4 K level, because the attenuator adds its own 4 K noise', 'Zero: 60 dB removes essentially everything', 'About 1250: attenuators do not change noise'],
      answer: 1,
      explain: 'An attenuator is a thermal beam splitter: it passes 1/L of the incoming noise and adds (1 − 1/L)·n̄(T<sub>stage</sub>). With L = 10⁶ the room noise is gone, but the 4 K attenuator emits n̄(4 K) ≈ 16 photons, and nothing below it cools the line further.' },
    { id: 'where', objective: 'placement',
      q: 'With 60 dB in total, which split (4 K / Still / MXC) leaves the fewest thermal photons at the MXC in this model?',
      choices: ['60 / 0 / 0', '20 / 20 / 20', '40 / 20 / 0', '0 / 0 / 60'],
      answer: 3,
      explain: 'The noise leaving the last attenuator can never be colder than that attenuator, so attenuation is worth most on the coldest plate: 0/0/60 gives about 1.3×10⁻³ photons, 20/20/20 about 0.03, 40/20/0 about 2.6 and 60/0/0 about 16. Real fridges still spread the attenuation over several plates, because the power an attenuator absorbs heats its plate and the MXC has only microwatts of cooling power. This page tracks noise, not heat.' },
    { id: 'drive', objective: 'drive',
      q: 'The attenuators also cut the drive signal by 60 dB (and more for the cable). Why is that acceptable?',
      choices: ['It is not: labs remove the attenuators when they want to drive the qubit', 'The drive is only scaled down, so you turn up the generator, while the noise is reset to the cold plate temperature', 'Attenuators only absorb noise and let the drive through untouched', 'The drive is at a different frequency from the noise, so attenuators do not affect it'],
      answer: 1,
      explain: 'Attenuation is frequency independent here: the drive and the incoming noise drop by the same factor. The difference is that each attenuator then adds back only its own, much colder thermal noise. Raising the generator power restores the drive, and the qubit sees a signal with far less noise on it.' },
  ],
});
