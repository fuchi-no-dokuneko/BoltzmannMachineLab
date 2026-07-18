(() => {
  "use strict";
  const lab = window.BoltzmannLab;
  const $ = (id) => document.getElementById(id);
  const inference = { input: [], distribution: [], result: null, randomState: 91573 };

  function random() { inference.randomState = (1664525 * inference.randomState + 1013904223) >>> 0; return inference.randomState / 4294967296; }
  function softplus(value) { return value > 30 ? value : value < -30 ? Math.exp(value) : Math.log1p(Math.exp(value)); }
  function temperature() { return Math.max(0.1, Number($("temperature").value) || 1); }
  function stateLabel(bits) { return bits.join(""); }
  function bitsFromMask(mask, count) { return Array.from({ length: count }, (_, bit) => (mask >> bit) & 1); }
  function normalizedRows(logRows) {
    const maximum = Math.max(...logRows.map((row) => row.logWeight));
    const weights = logRows.map((row) => Math.exp(row.logWeight - maximum));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return logRows.map((row, index) => ({ bits: row.bits, label: stateLabel(row.bits), probability: weights[index] / total })).sort((a, b) => b.probability - a.probability);
  }

  function rbmDistribution() {
    const state = lab.state, size = state.visible + state.hidden, divisor = temperature(), rows = [];
    for (let mask = 0; mask < 2 ** state.visible; mask += 1) {
      const visible = bitsFromMask(mask, state.visible);
      let logWeight = 0;
      for (let v = 0; v < state.visible; v += 1) logWeight += state.biases[v] * visible[v] / divisor;
      for (let h = 0; h < state.hidden; h += 1) {
        let activation = state.biases[state.visible + h];
        for (let v = 0; v < state.visible; v += 1) activation += visible[v] * state.weights[lab.index(v, state.visible + h, size)];
        logWeight += softplus(activation / divisor);
      }
      rows.push({ bits: visible, logWeight });
    }
    return { rows: normalizedRows(rows), method: "Exact RBM visible marginal", representedStates: rows.length };
  }

  function fullExactDistribution() {
    const state = lab.state, size = state.visible + state.hidden, divisor = temperature(), visibleWeights = new Map(), totalStates = 2 ** size;
    const logRows = [];
    for (let mask = 0; mask < totalStates; mask += 1) {
      const units = bitsFromMask(mask, size);
      logRows.push({ bits: units, logWeight: -lab.energy(units) / divisor });
    }
    const maximum = Math.max(...logRows.map((row) => row.logWeight));
    let total = 0;
    logRows.forEach((row) => { const weight = Math.exp(row.logWeight - maximum), visible = row.bits.slice(0, state.visible), label = stateLabel(visible); total += weight; const record = visibleWeights.get(label) || { bits: visible, weight: 0 }; record.weight += weight; visibleWeights.set(label, record); });
    const rows = [...visibleWeights.values()].map((row) => ({ bits: row.bits, label: stateLabel(row.bits), probability: row.weight / total })).sort((a, b) => b.probability - a.probability);
    return { rows, method: "Exact full-BM visible marginal", representedStates: totalStates };
  }

  function gibbsUnit(units, unit) {
    const state = lab.state, size = units.length; let sum = state.biases[unit];
    for (let other = 0; other < size; other += 1) if (other !== unit) sum += units[other] * state.weights[lab.index(unit, other, size)];
    units[unit] = random() < lab.probability(sum) ? 1 : 0;
  }
  function sweep(units, start = 0) { for (let unit = start; unit < units.length; unit += 1) gibbsUnit(units, unit); }
  function sampledDistribution(burnIn, sampleCount) {
    const state = lab.state, size = state.visible + state.hidden, units = Uint8Array.from({ length: size }, () => random() < 0.5 ? 1 : 0), counts = new Map();
    for (let step = 0; step < burnIn; step += 1) sweep(units);
    for (let sample = 0; sample < sampleCount; sample += 1) { sweep(units); const bits = Array.from(units.slice(0, state.visible)), label = stateLabel(bits), record = counts.get(label) || { bits, count: 0 }; record.count += 1; counts.set(label, record); }
    const rows = [...counts.values()].map((row) => ({ bits: row.bits, label: stateLabel(row.bits), probability: row.count / sampleCount })).sort((a, b) => b.probability - a.probability);
    const activationLabel = lab.state.standardActivation ? "" : " with custom activation";
    return { rows, method: "Gibbs estimate" + activationLabel, representedStates: counts.size };
  }

  function rbmConditional(input) {
    const state = lab.state, size = state.visible + state.hidden;
    const hidden = Array.from({ length: state.hidden }, (_, h) => { let sum = state.biases[state.visible + h]; for (let v = 0; v < state.visible; v += 1) sum += input[v] * state.weights[lab.index(v, state.visible + h, size)]; return lab.probability(sum); });
    const reconstructed = Array.from({ length: state.visible }, (_, v) => { let sum = state.biases[v]; for (let h = 0; h < state.hidden; h += 1) sum += hidden[h] * state.weights[lab.index(v, state.visible + h, size)]; return lab.probability(sum); });
    return { hidden, reconstructed };
  }
  function fullConditional(input, burnIn, sampleCount) {
    const state = lab.state, size = state.visible + state.hidden, units = new Uint8Array(size); units.set(input); for (let h = state.visible; h < size; h += 1) units[h] = random() < .5 ? 1 : 0;
    const hidden = new Float64Array(state.hidden), reconstructed = new Float64Array(state.visible);
    for (let step = 0; step < burnIn; step += 1) sweep(units, state.visible);
    for (let sample = 0; sample < sampleCount; sample += 1) {
      sweep(units, state.visible);
      for (let h = 0; h < state.hidden; h += 1) hidden[h] += units[state.visible + h];
      for (let v = 0; v < state.visible; v += 1) { let sum = state.biases[v]; for (let other = 0; other < size; other += 1) if (other !== v) sum += units[other] * state.weights[lab.index(v, other, size)]; reconstructed[v] += lab.probability(sum); }
    }
    return { hidden: Array.from(hidden, (value) => value / sampleCount), reconstructed: Array.from(reconstructed, (value) => value / sampleCount) };
  }

  function renderBits() {
    const container = $("inferenceBits"); container.replaceChildren();
    inference.input.forEach((bit, index) => { const button = document.createElement("button"); button.type = "button"; button.className = "inference-bit" + (bit ? " on" : ""); button.textContent = "V" + (index + 1) + "=" + bit; button.addEventListener("click", () => { inference.input[index] = 1 - inference.input[index]; renderBits(); invalidateResults("Inference input changed. Run inference again."); }); container.appendChild(button); });
  }
  function resetInput(message = "Train the model or inspect its initial distribution.") { const pattern = lab.state.patterns.find((item) => item.enabled); inference.input = pattern ? pattern.bits.slice() : Array(lab.state.visible).fill(0); inference.distribution = []; inference.result = null; renderBits(); clearResults(); $("inferenceStatus").textContent = message; }
  function renderProbabilities(id, values, prefix) {
    const container = $(id); container.replaceChildren();
    values.forEach((value, index) => { const row = document.createElement("div"); row.className = "probability-row"; const label = document.createElement("strong"); label.textContent = prefix + (index + 1); const track = document.createElement("div"); track.className = "probability-track"; const fill = document.createElement("i"); fill.style.width = (value * 100).toFixed(2) + "%"; track.appendChild(fill); const output = document.createElement("output"); output.textContent = value.toFixed(3); row.append(label, track, output); container.appendChild(row); });
  }
  function renderDistribution(rows) {
    const container = $("distribution"); container.replaceChildren();
    rows.slice(0, 12).forEach((row) => { const item = document.createElement("div"); item.className = "distribution-row"; const label = document.createElement("code"); label.textContent = row.label; const track = document.createElement("div"); track.className = "distribution-track"; const fill = document.createElement("i"); fill.style.width = Math.max(.5, row.probability * 100).toFixed(3) + "%"; track.appendChild(fill); const output = document.createElement("output"); output.textContent = (row.probability * 100).toFixed(3) + "%"; item.append(label, track, output); container.appendChild(item); });
  }
  function clearResults() { $("hiddenPosterior").innerHTML = "<p>No inference result.</p>"; $("visibleReconstruction").innerHTML = "<p>No inference result.</p>"; $("distribution").innerHTML = "<p>No learned distribution yet.</p>"; $("distributionStates").textContent = "0"; $("distributionEntropy").textContent = "0.000"; $("probabilitySum").textContent = "0.000000"; $("sampledState").textContent = "-"; $("drawInferenceSample").disabled = true; $("inferenceMethod").textContent = "weights frozen"; }
  function invalidateResults(message = "Inputs or parameters changed. Run inference again.") { inference.distribution = []; inference.result = null; clearResults(); $("inferenceStatus").textContent = message; }
  function runInference() {
    lab.stop(); inference.randomState = 91573;
    const burnIn = Math.max(0, Math.min(5000, Number($("inferenceBurnIn").value) || 0));
    const sampleCount = Math.max(100, Math.min(20000, Number($("inferenceSamples").value) || 2000));
    const conditional = $("model").value === "rbm" ? rbmConditional(inference.input) : fullConditional(inference.input, burnIn, sampleCount);
    const distribution = $("model").value === "rbm" && lab.state.standardActivation ? rbmDistribution() : (lab.state.standardActivation && lab.state.visible + lab.state.hidden <= 16 ? fullExactDistribution() : sampledDistribution(burnIn, sampleCount));
    const probabilitySum = distribution.rows.reduce((sum, row) => sum + row.probability, 0);
    const entropy = -distribution.rows.reduce((sum, row) => sum + (row.probability > 0 ? row.probability * Math.log(row.probability) : 0), 0);
    inference.distribution = distribution.rows; inference.result = { conditional, distribution, probabilitySum, entropy };
    renderProbabilities("hiddenPosterior", conditional.hidden, "H"); renderProbabilities("visibleReconstruction", conditional.reconstructed, "V"); renderDistribution(distribution.rows);
    $("distributionStates").textContent = distribution.representedStates.toLocaleString(); $("distributionEntropy").textContent = entropy.toFixed(3); $("probabilitySum").textContent = probabilitySum.toFixed(6); $("inferenceMethod").textContent = distribution.method; $("inferenceStatus").textContent = "Inference complete at epoch " + lab.state.epoch + ". Weights were not changed."; $("drawInferenceSample").disabled = false;
    return inference.result;
  }
  function drawSample() { if (!inference.distribution.length) return; let threshold = random(); let selected = inference.distribution[inference.distribution.length - 1]; for (const row of inference.distribution) { threshold -= row.probability; if (threshold <= 0) { selected = row; break; } } $("sampledState").textContent = selected.label; }

  $("runInference").addEventListener("click", runInference);
  $("drawInferenceSample").addEventListener("click", drawSample);
  $("loadPattern").addEventListener("click", () => resetInput());
  $("model").addEventListener("change", () => resetInput()); $("visibleCount").addEventListener("change", () => resetInput()); $("hiddenCount").addEventListener("change", () => resetInput());
  $("temperature").addEventListener("input", () => invalidateResults("Temperature changed. Run inference again."));
  $("inferenceBurnIn").addEventListener("input", () => invalidateResults("Sampling settings changed. Run inference again."));
  $("inferenceSamples").addEventListener("input", () => invalidateResults("Sampling settings changed. Run inference again."));
  window.addEventListener("boltzmannlab:changed", (event) => {
    const reason = event.detail && event.detail.reason;
    if (reason === "reset") resetInput();
    else if (reason === "train") invalidateResults("Weights changed through training. Run inference again.");
    else if (reason === "code") invalidateResults("Learning functions changed. Run inference again.");
  });
  resetInput();
  window.BoltzmannInference = { state: inference, runInference, rbmDistribution, fullExactDistribution, sampledDistribution, rbmConditional, fullConditional, invalidateResults };
})();
