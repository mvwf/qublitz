/* outreach-config.js: where anonymous usage counts and quiz answers go.
 *
 * Leave `endpoint` empty and nothing is ever sent (the quizzes and learning panels still work, they just stay
 * in the browser). To switch collection on, deploy tools/outreach-collector/Code.gs as a Google Apps Script
 * web app (see tools/outreach-collector/README.md) and paste its /exec URL here.
 */
window.QL_CONFIG = {
  endpoint: 'https://script.google.com/macros/s/AKfycbyrHUf-aPDXaVMV0Gy8VfwvytOPBKIqG6NaVOg4psJnwA47Tug1g0a_zmCnf3R0S7wg/exec',
  // Shown in the small notice at the bottom of each page while collection is on.
  owner: 'the Fitzpatrick Lab at Dartmouth',
};
