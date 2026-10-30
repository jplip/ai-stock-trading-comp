/* Independent accounting and time-order checks for the browser engine. */
const assert = require('node:assert/strict');
const E = require('../dist/engine.js');
const bars = E.generateDemo(42, 760);
const options = { initialCapital: 10000, feeBps: 10, threshold: .55, synthetic: true };
const result = E.runExperiment(bars, options);
const indexByDate = new Map(bars.map((bar, index) => [bar.date, index]));
const almost = (a, b, label) => assert.ok(Math.abs(a - b) < 1e-7 * Math.max(1, Math.abs(b)), `${label}: ${a} vs ${b}`);
assert.ok(result.predictions.length >= 30, 'Enough observations in the final window');
assert.ok(result.models.length >= 3, 'At least three trained models');
almost(result.models.reduce((sum, model) => sum + model.weight, 0), 1, 'Ensemble weights sum to one');
let capital = options.initialCapital;
let previousWeight = 0;
let peak = capital;
let worstDrawdown = 0;
const fee = options.feeBps / 10000;
for (const prediction of result.predictions) {
  const index = indexByDate.get(prediction.signalDate);
  assert.equal(prediction.executionDate, bars[index + 1].date, 'Fill at the following open');
  assert.equal(prediction.exitDate, bars[index + 2].date, 'Hold over the following open-to-open interval');
  const realized = bars[index + 2].open / bars[index + 1].open - 1;
  almost(prediction.actualReturn, realized, 'Realized return alignment');
  assert.ok(prediction.probability >= 0 && prediction.probability <= 1, 'Probability bounded');
  assert.ok(prediction.position >= 0 && prediction.position <= 1, 'No leverage or short selling');
  capital *= (1 - fee * Math.abs(prediction.position - previousWeight)) * (1 + prediction.position * realized);
  previousWeight = prediction.position * (1 + realized) / (1 + prediction.position * realized);
  peak = Math.max(peak, capital);
  worstDrawdown = Math.max(worstDrawdown, 1 - capital / peak);
}
capital *= 1 - fee * previousWeight;
worstDrawdown = Math.max(worstDrawdown, 1 - capital / peak);
almost(result.metrics.endingEquity, capital, 'Independent net equity calculation including liquidation');
almost(result.metrics.totalReturn, capital / options.initialCapital - 1, 'Total net return');
almost(result.metrics.maxDrawdown, worstDrawdown, 'Maximum drawdown includes initial capital');
const firstIndex = indexByDate.get(result.predictions[0].executionDate);
const lastIndex = indexByDate.get(result.predictions.at(-1).exitDate);
const expectedBaseline = options.initialCapital * (1 - fee) * bars[lastIndex].open / bars[firstIndex].open * (1 - fee);
almost(result.baselineMetrics.endingEquity, expectedBaseline, 'Buy and hold includes both fees');

// Future shocks cannot rewrite predictions already made at earlier dates.
const firstSignalIndex = indexByDate.get(result.predictions[0].signalDate);
const shockIndex = firstSignalIndex + 35;
const shocked = bars.map((bar, index) => index < shockIndex ? { ...bar } : {
  ...bar, open: bar.open * 1.7, high: bar.high * 1.7, low: bar.low * 1.7, close: bar.close * 1.7,
});
const shockedResult = E.runExperiment(shocked, options);
for (let i = 0; i < result.predictions.length; i++) {
  if (indexByDate.get(result.predictions[i].signalDate) >= shockIndex) break;
  almost(result.predictions[i].probability, shockedResult.predictions[i].probability, 'Future-invariant model prediction');
  almost(result.predictions[i].position, shockedResult.predictions[i].position, 'Future-invariant allocation');
}
result.models.forEach((model, index) => almost(model.weight, shockedResult.models[index].weight, 'Final-window changes do not change validation weights'));

// CSV round trip and defensive validation exercise the actual import boundary.
const csv = 'Date,Open,High,Low,Close,Volume\n' + bars.map(bar => [bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume].join(',')).join('\n');
assert.equal(E.parseCSV(csv).bars.length, bars.length);
assert.throws(() => E.parseCSV('Date,Open,Close\n2026-01-01,100,101\n'), /./, 'Short datasets rejected');
for (const invalid of [{ initialCapital: 0 }, { feeBps: Infinity }, { threshold: -1 }, { maxExposure: 2 }]) {
  assert.throws(() => E.runExperiment(bars, invalid), /./, 'Invalid numerical options rejected');
}
assert.ok(result.audit.every(check => check.passed), 'Every reported audit check passed');
console.log(`Integration checks passed: ${result.predictions.length} predictions; ${result.models.length} models; accounting, temporal alignment, future invariance, validation isolation, and import validation.`);
