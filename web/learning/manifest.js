/* manifest.js: every page that reports to the outreach collector, with its id. The id is what the collector
 * and the dashboard group by; each page's learning objectives, how-to-play text and quiz live in
 * web/learning/<id>.js, which calls QL.learning({...}) (see web/shared/outreach.js).
 *
 * Add a page here when it gains a learning file. Paths are relative to the site root.
 */
(function (root) {
  'use strict';
  root.QL_GAMES = [
    { id: 'qubit-lab',   title: 'Qublitz Virtual Qubit Lab',      path: 'qublitz/qubit-simulator.html',             group: 'Virtual Qubit Labs' },
    { id: 'copter',      title: 'Qubit Copter',                   path: 'qublitz/qubit-copter.html',                group: 'Virtual Qubit Labs' },
    { id: 'rider',       title: 'Rabi Rider',                     path: 'qublitz/rabi-rider.html',                  group: 'Virtual Qubit Labs' },
    { id: 'beamline',    title: 'Beamline',                       path: 'qublitz/beamline.html',                    group: 'Virtual Qubit Labs' },
    { id: 'errorsweeper', title: 'ErrorSweeper',                 path: 'qublitz/errorsweeper.html',                group: 'Virtual Qubit Labs' },
    { id: 'arena',       title: 'QuBlitz Arena (Photon Duel)',    path: 'qublitz/arena.html',                       group: 'Virtual Qubit Labs' },
    { id: 'fridge',      title: 'Fridge Noise Explorer',          path: 'qublitz/fridge-noise.html',                group: 'Research' },
    { id: 'resonator',   title: 'Resonator Lab',                  path: 'qublitz/resonator.html',                   group: 'Research' },
    { id: 'ep-tpd',      title: 'Exceptional Points & TPDs',      path: 'qublitz/ep-tpd.html',                      group: 'Research' },
    { id: 'trio',        title: 'Three Equations, One Eigenbasis', path: 'courses/engs23/heat-wave-schrodinger.html', group: 'ENGS 23' },
    { id: 'string',      title: 'The Vibrating String',           path: 'courses/engs23/vibrating-string.html',     group: 'ENGS 23' },
    { id: 'drum',        title: 'The Drum Head',                  path: 'courses/engs23/drum-head.html',            group: 'ENGS 23' },
    { id: 'spectrogram', title: 'Spectrogram Explorer',           path: 'courses/engs23/spectrogram.html',          group: 'ENGS 23' },
    { id: 'hydrogen',    title: 'The Hydrogen Atom',              path: 'courses/engs23/hydrogen.html',             group: 'ENGS 23' },
    { id: 'sonify',      title: 'Sonify Images',                  path: 'qublitz/sonify.html',                      group: 'Fun' },
    { id: 'hub',         title: 'Demo hub',                       path: '',                                         group: 'Site' },
  ];
})(typeof window !== 'undefined' ? window : globalThis);
