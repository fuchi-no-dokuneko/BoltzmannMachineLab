(() => {
  "use strict";
  const ACTIVATION_SOURCE = `function activation(sum, temperature) {
  return 1 / (1 + Math.exp(-sum / temperature));
}`;
  const UPDATE_SOURCE = `function update(weight, positive, negative, learningRate) {
  return weight + learningRate * (positive - negative);
}`;
  const $ = (id) => document.getElementById(id);
  const state = { visible: 6, hidden: 4, weights: [], biases: [], units: [], patterns: [], epoch: 0, timer: null, seed: 20260630, phase: "idle", error: 0, activation: null, update: null };

  function seededRandom() { state.seed = (1664525 * state.seed + 1013904223) >>> 0; return state.seed / 4294967296; }
  function probability(sum) { return Math.max(0, Math.min(1, Number(state.activation(sum, Number($("temperature").value))) || 0)); }
  function sample(value) { return seededRandom() < value ? 1 : 0; }
  function index(a, b, size) { return a * size + b; }
  function compileFunction(source, expectedName) {
    const fn = new Function('"use strict"; return (' + source + "\n);")();
    if (typeof fn !== "function" || fn.name !== expectedName) throw new Error("Expected function " + expectedName + "(...)");
    return fn;
  }
  function applyCode(showStatus = true) {
    try {
      const activation = compileFunction($("activationCode").value, "activation");
      const update = compileFunction($("updateCode").value, "update");
      const probe = activation(0, 1); const updateProbe = update(0, 1, 0, .1);
      if (!Number.isFinite(probe) || !Number.isFinite(updateProbe)) throw new Error("Functions must return finite numbers");
      state.activation = activation; state.update = update;
      if (showStatus) { $("codeStatus").textContent = "Custom functions active."; setStatus("Learning functions applied."); }
      return true;
    } catch (error) { $("codeStatus").textContent = "Code error: " + error.message; setStatus(error.message, true); return false; }
  }
  function restoreCode() { $("activationCode").value = ACTIVATION_SOURCE; $("updateCode").value = UPDATE_SOURCE; applyCode(false); $("codeStatus").textContent = "Original functions active."; }

  function makePatterns(count) {
    const rows = Math.min(6, Math.max(4, count));
    return Array.from({ length: rows }, (_, row) => ({ enabled: true, bits: Array.from({ length: count }, (_, column) => ((column + row) % rows === 0 || column === row % count) ? 1 : 0) }));
  }
  function resetNetwork() {
    stop(); state.visible = clampInt($("visibleCount").value, 3, 12); state.hidden = clampInt($("hiddenCount").value, 2, 10);
    $("visibleCount").value = state.visible; $("hiddenCount").value = state.hidden; state.seed = 20260630; state.epoch = 0; state.phase = "idle"; state.error = 0;
    const size = state.visible + state.hidden; state.weights = new Float64Array(size * size); state.biases = new Float64Array(size); state.units = new Uint8Array(size);
    for (let a = 0; a < size; a += 1) for (let b = a + 1; b < size; b += 1) {
      if ($("model").value === "rbm" && ((a < state.visible) === (b < state.visible))) continue;
      const weight = (seededRandom() - .5) * .28; state.weights[index(a, b, size)] = weight; state.weights[index(b, a, size)] = weight;
    }
    state.patterns = makePatterns(state.visible); renderPatterns(); updateMetrics(); draw(); setStatus("Network reset with deterministic initial weights.");
  }
  function clampInt(value, min, max) { return Math.max(min, Math.min(max, Math.round(Number(value) || min))); }
  function enabledPatterns() { return state.patterns.filter((pattern) => pattern.enabled); }
  function hiddenProbabilities(visible) {
    const size = state.visible + state.hidden;
    return Array.from({ length: state.hidden }, (_, h) => {
      let sum = state.biases[state.visible + h]; for (let v = 0; v < state.visible; v += 1) sum += visible[v] * state.weights[index(v, state.visible + h, size)]; return probability(sum);
    });
  }
  function visibleProbabilities(hidden) {
    const size = state.visible + state.hidden;
    return Array.from({ length: state.visible }, (_, v) => {
      let sum = state.biases[v]; for (let h = 0; h < state.hidden; h += 1) sum += hidden[h] * state.weights[index(v, state.visible + h, size)]; return probability(sum);
    });
  }
  function rbmStep(pattern) {
    const v0 = pattern.bits.slice(); const h0p = hiddenProbabilities(v0); let hidden = h0p.map(sample); let vk = v0; let vkp = v0; let hkp = h0p;
    for (let k = 0; k < clampInt($("gibbsSteps").value, 1, 50); k += 1) { vkp = visibleProbabilities(hidden); vk = vkp.map(sample); hkp = hiddenProbabilities(vk); hidden = hkp.map(sample); }
    const size = state.visible + state.hidden; const lr = Number($("learningRate").value);
    for (let v = 0; v < state.visible; v += 1) for (let h = 0; h < state.hidden; h += 1) {
      const a = v; const b = state.visible + h; const old = state.weights[index(a, b, size)]; const next = Number(state.update(old, v0[v] * h0p[h], vk[v] * hkp[h], lr));
      if (Number.isFinite(next)) state.weights[index(a, b, size)] = state.weights[index(b, a, size)] = Math.max(-8, Math.min(8, next));
    }
    for (let v = 0; v < state.visible; v += 1) { const next = Number(state.update(state.biases[v], v0[v], vk[v], lr)); if (Number.isFinite(next)) state.biases[v] = Math.max(-8, Math.min(8, next)); }
    for (let h = 0; h < state.hidden; h += 1) { const unit = state.visible + h; const next = Number(state.update(state.biases[unit], h0p[h], hkp[h], lr)); if (Number.isFinite(next)) state.biases[unit] = Math.max(-8, Math.min(8, next)); }
    state.units.set([...vk, ...hidden]); state.error = v0.reduce((sum, bit, i) => sum + (bit - vkp[i]) ** 2, 0) / state.visible; state.phase = "CD reconstruction";
  }
  function gibbsUnit(units, unit) {
    const size = units.length; let sum = state.biases[unit]; for (let other = 0; other < size; other += 1) if (other !== unit) sum += units[other] * state.weights[index(unit, other, size)]; units[unit] = sample(probability(sum));
  }
  function bmStep(pattern) {
    const size = state.visible + state.hidden; const positiveUnits = new Uint8Array(size); positiveUnits.set(pattern.bits); for (let i = state.visible; i < size; i += 1) positiveUnits[i] = sample(.5);
    const rounds = clampInt($("gibbsSteps").value, 1, 50);
    for (let k = 0; k < rounds; k += 1) for (let i = state.visible; i < size; i += 1) gibbsUnit(positiveUnits, i);
    const negativeUnits = Uint8Array.from(state.units.length ? state.units : Array(size).fill(0));
    for (let k = 0; k < rounds; k += 1) for (let i = 0; i < size; i += 1) gibbsUnit(negativeUnits, i);
    const lr = Number($("learningRate").value);
    for (let a = 0; a < size; a += 1) for (let b = a + 1; b < size; b += 1) {
      const old = state.weights[index(a, b, size)]; const next = Number(state.update(old, positiveUnits[a] * positiveUnits[b], negativeUnits[a] * negativeUnits[b], lr));
      if (Number.isFinite(next)) state.weights[index(a, b, size)] = state.weights[index(b, a, size)] = Math.max(-8, Math.min(8, next));
    }
    for (let unit = 0; unit < size; unit += 1) { const next = Number(state.update(state.biases[unit], positiveUnits[unit], negativeUnits[unit], lr)); if (Number.isFinite(next)) state.biases[unit] = Math.max(-8, Math.min(8, next)); }
    state.units = negativeUnits; state.error = pattern.bits.reduce((sum, bit, i) => sum + (bit - negativeUnits[i]) ** 2, 0) / state.visible; state.phase = "free negative phase";
  }
  function trainStep() {
    const patterns = enabledPatterns(); if (!patterns.length) { setStatus("Enable at least one training pattern.", true); stop(); return; }
    try { const pattern = patterns[state.epoch % patterns.length]; $("model").value === "rbm" ? rbmStep(pattern) : bmStep(pattern); state.epoch += 1; updateMetrics(); draw(); }
    catch (error) { setStatus("Training stopped: " + error.message, true); stop(); }
  }
  function start() { if (state.timer) return; const period = Math.round(1000 / Number($("speed").value)); state.timer = setInterval(trainStep, period); $("start").disabled = true; $("pause").disabled = false; setStatus("Training in progress."); }
  function stop() { if (state.timer) clearInterval(state.timer); state.timer = null; $("start").disabled = false; $("pause").disabled = true; }
  function energy(units = state.units) { const size = units.length; let value = 0; for (let unit = 0; unit < size; unit += 1) value -= state.biases[unit] * units[unit]; for (let a = 0; a < size; a += 1) for (let b = a + 1; b < size; b += 1) value -= state.weights[index(a, b, size)] * units[a] * units[b]; return value; }
  function updateMetrics() { $("epoch").textContent = state.epoch; $("error").textContent = state.error.toFixed(4); $("energy").textContent = energy().toFixed(4); const active = [...state.weights].filter((value) => value !== 0); $("weightRms").textContent = Math.sqrt(active.reduce((sum, value) => sum + value * value, 0) / Math.max(1, active.length)).toFixed(4); $("phase").textContent = state.phase; }
  function setStatus(message, error = false) { $("status").textContent = message; $("status").className = "status" + (error ? " error" : ""); }

  function renderPatterns() {
    const container = $("patterns"); container.replaceChildren();
    state.patterns.forEach((pattern, row) => { const item = document.createElement("div"); item.className = "pattern"; const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = pattern.enabled; enabled.setAttribute("aria-label", "Enable pattern " + (row + 1)); enabled.addEventListener("change", () => pattern.enabled = enabled.checked); const bits = document.createElement("div"); bits.className = "bits"; bits.style.setProperty("--bits", state.visible); pattern.bits.forEach((bit, column) => { const button = document.createElement("button"); button.type = "button"; button.className = "bit" + (bit ? " on" : ""); button.textContent = bit; button.setAttribute("aria-label", "Pattern " + (row + 1) + " bit " + (column + 1)); button.addEventListener("click", () => { pattern.bits[column] = 1 - pattern.bits[column]; renderPatterns(); }); bits.appendChild(button); }); item.append(enabled, bits); container.appendChild(item); });
  }
  function draw() {
    const canvas = $("network"); const ratio = devicePixelRatio || 1; const width = Math.max(320, canvas.clientWidth); const height = canvas.clientHeight; canvas.width = width * ratio; canvas.height = height * ratio; const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height);
    const positions = []; const margin = Math.min(100, width * .12); const place = (count, y, offset) => { for (let i = 0; i < count; i += 1) positions[offset + i] = { x: count === 1 ? width / 2 : margin + i * (width - margin * 2) / (count - 1), y }; }; place(state.hidden, height * .28, state.visible); place(state.visible, height * .74, 0);
    const size = positions.length; for (let a = 0; a < size; a += 1) for (let b = a + 1; b < size; b += 1) { const weight = state.weights[index(a, b, size)]; if (!weight) continue; ctx.beginPath(); ctx.moveTo(positions[a].x, positions[a].y); ctx.lineTo(positions[b].x, positions[b].y); ctx.strokeStyle = weight >= 0 ? "rgba(11,118,93,.72)" : "rgba(189,73,63,.72)"; ctx.lineWidth = Math.min(9, .7 + Math.abs(weight) * 4); ctx.stroke(); }
    positions.forEach((point, unit) => { ctx.beginPath(); ctx.arc(point.x, point.y, Math.max(11, Math.min(20, width / (size * 3))), 0, Math.PI * 2); ctx.fillStyle = state.units[unit] ? "#e6ad2e" : "#ffffff"; ctx.strokeStyle = unit < state.visible ? "#17201f" : "#0b765d"; ctx.lineWidth = 3; ctx.fill(); ctx.stroke(); ctx.fillStyle = "#52615d"; ctx.font = "12px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.fillText((unit < state.visible ? "V" + (unit + 1) : "H" + (unit - state.visible + 1)), point.x, point.y + 38); });
  }

  $("activationCode").value = ACTIVATION_SOURCE; $("updateCode").value = UPDATE_SOURCE; restoreCode(); resetNetwork();
  $("start").addEventListener("click", start); $("pause").addEventListener("click", stop); $("step").addEventListener("click", trainStep); $("reset").addEventListener("click", resetNetwork); $("applyCode").addEventListener("click", () => applyCode(true)); $("resetCode").addEventListener("click", restoreCode);
  $("model").addEventListener("change", resetNetwork); $("visibleCount").addEventListener("change", resetNetwork); $("hiddenCount").addEventListener("change", resetNetwork); $("speed").addEventListener("input", () => { $("speedValue").value = $("speed").value; if (state.timer) { stop(); start(); } });
  $("randomPattern").addEventListener("click", () => { state.patterns.forEach((pattern) => pattern.bits = pattern.bits.map(() => seededRandom() > .5 ? 1 : 0)); renderPatterns(); }); window.addEventListener("resize", draw);
  window.BoltzmannLab = { state, compileFunction, trainStep, resetNetwork, energy, probability, hiddenProbabilities, visibleProbabilities, index, stop };
})();
