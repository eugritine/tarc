'use strict';
const $ = id => document.getElementById(id);
const state = { data: null, metrics: null, localTexts: {}, dataset: 'iemocap', view: 'examples', filter: 'all', search: '', page: 0, selected: null, manual: null };
const PAGE_SIZE = 6;
const CLASS_NAMES = ['Happy / Excited', 'Sad', 'Neutral', 'Angry'];
const CLASS_SHORT = ['Happy', 'Sad', 'Neutral', 'Angry'];
const EMOJI = ['😄', '😢', '😐', '😠'];
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fixed = (value, digits = 3) => value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
const signed = value => (Number(value) >= 0 ? '+' : '') + fixed(value);
const percent = value => fixed(Number(value) * 100, 1) + '%';
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
const argmax = values => values.indexOf(Math.max(...values));
const sentiment = value => value > 0 ? 'Positive sentiment' : value < 0 ? 'Negative sentiment' : 'Neutral';
const ds = () => state.data.datasets[state.dataset];
const sample = () => ds().samples.find(s => s.id === state.selected);
const isClass = () => ds().task === 'classification';

function scenarioName(scenario) {
  return (isClass() ? { helpful: 'Helpful correction', kept: 'Remains correct', harmful: 'Harmful correction', other: 'Remains incorrect' } : { helpful: 'Reduced error', kept: 'Unchanged error', harmful: 'Increased error', other: 'Other' })[scenario] || scenario;
}

function chooseInitial() {
  const d = ds();
  const recommended = Object.values(d.recommended_ids || {}).flat().filter(v => typeof v === 'string');
  state.selected = recommended.find(id => d.samples.some(s => s.id === id && s.scenario === 'helpful')) || d.samples.find(s => s.scenario === 'helpful')?.id || d.samples[0].id;
  state.filter = 'all'; state.search = ''; state.page = 0;
  $('search').value = '';
  const counts = Object.fromEntries(['helpful', 'kept', 'harmful', 'other'].map(k => [k, d.samples.filter(s => s.scenario === k).length]));
  $('scenario').innerHTML = '<option value="all">All outcomes</option>' + Object.entries(counts).map(([k, n]) => `<option value="${k}" ${n ? '' : 'disabled'}>${scenarioName(k)} · ${n.toLocaleString('en-US')}</option>`).join('');
  resetManual(); render();
}

function filtered() {
  const preferred = Object.values(ds().recommended_ids || {}).flat();
  const rank = new Map(preferred.map((id, index) => [id, index]));
  return ds().samples.filter(s => (state.filter === 'all' || s.scenario === state.filter) && s.id.toLowerCase().includes(state.search.toLowerCase())).sort((a, b) => (rank.get(a.id) ?? 100000) - (rank.get(b.id) ?? 100000));
}

function renderList() {
  const records = filtered(); const maxPage = Math.max(0, Math.ceil(records.length / PAGE_SIZE) - 1);
  state.page = Math.min(state.page, maxPage);
  $('case-count').textContent = `${records.length.toLocaleString('en-US')} examples / ${ds().n.toLocaleString('en-US')} validation examples`;
  $('case-list').innerHTML = records.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE).map(s => {
    const prediction = isClass() ? CLASS_SHORT[argmax(s.corrected_prob)] : signed(s.corrected);
    return `<button class="case ${s.id === state.selected ? 'active' : ''}" data-id="${esc(s.id)}" aria-pressed="${s.id === state.selected}"><div class="case-id">${esc(s.id)}</div><div class="case-meta"><span class="tag ${esc(s.scenario)}">${scenarioName(s.scenario)}</span><span>${prediction}</span></div></button>`;
  }).join('') || '<p class="muted tiny">No examples match the current filters.</p>';
  $('case-list').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { state.selected = b.dataset.id; resetManual(); renderList(); renderSample(); }));
  $('page-label').textContent = `${state.page + 1} / ${maxPage + 1}`;
  $('previous').disabled = state.page === 0; $('next').disabled = state.page === maxPage;
}

function bars(probabilities, kind = '') {
  return `<div class="bar-list ${kind}">${probabilities.map((p, i) => `<div class="bar-row"><span>${CLASS_SHORT[i]}</span><div class="bar-track"><div class="bar-fill" style="width:${clamp(p * 100, 0, 100)}%"></div></div><span class="bar-value">${fixed(p * 100, 1)}</span></div>`).join('')}</div>`;
}

function predictionCard(s, kind, title, number) {
  if (isClass()) {
    const p = s[kind === 'text' ? 'text_prob' : kind === 'evidence' ? 'evidence_prob' : 'corrected_prob']; const pred = argmax(p);
    return `<article class="panel prediction ${kind}"><div class="step-label">${number} / ${title}</div><div class="face" aria-hidden="true">${EMOJI[pred]}</div><div class="prediction-title">${CLASS_NAMES[pred]}</div><div class="prediction-subtitle">Top-class probability ${percent(p[pred])}</div>${bars(p)}</article>`;
  }
  const value = s[kind];
  return `<article class="panel prediction ${kind}"><div class="step-label">${number} / ${title}</div><div class="scalar-value">${signed(value)}</div><div class="prediction-title">${sentiment(value)}</div><div class="prediction-subtitle">Absolute error ${fixed(Math.abs(value - s.y))}</div><div class="score-axis"><span class="score-dot" style="left:${clamp((value + 3) / 6 * 100, 0, 100)}%"></span><span class="axis-left">−3</span><span class="axis-zero">0</span><span class="axis-right">+3</span></div></article>`;
}

function renderSource(s) {
  let html = '<h2 class="section-title">Source weights</h2><p class="section-description">Source weights guide evidence construction; they are not correctness probabilities.</p>';
  if (!Array.isArray(s.rho)) html += '<div class="muted tiny">Source weights were not recorded in this dataset cache.</div>';
  else html += s.rho.map((p, i) => `<div class="source-row"><span class="source-label"><i class="swatch" style="background:${['var(--text)', 'var(--audio)', 'var(--vision)'][i]}"></i>${['Text', 'Audio', 'Vision'][i]}</span><div class="bar-track"><div class="bar-fill" style="background:${['var(--text)', 'var(--audio)', 'var(--vision)'][i]};width:${clamp(p * 100, 0, 100)}%"></div></div><strong>${percent(p)}</strong></div>`).join('');
  $('source-panel').innerHTML = html;
}

function renderGate(s) {
  let html = '<h2 class="section-title">Correction strength</h2>';
  if (isClass()) {
    html += `<p class="section-description">Lower values favor the text anchor; higher values favor multimodal evidence.</p><div class="gate-value">${fixed(s.q)}</div><div class="gate-track"><span style="width:${clamp(s.q * 100, 0, 100)}%"></span></div><div class="gate-scale"><span>0 · Text anchor</span><span>1 · Multimodal evidence</span></div>`;
  } else {
    html += `<p class="section-description">Separate gates control polarity and magnitude. A value of 1 applies the full corresponding correction.</p><div class="gate-values"><div><div class="gate-value">${fixed(s.q_sign)}</div><div class="gate-caption">Polarity correction qₛ</div></div><div><div class="gate-value">${fixed(s.q_magnitude)}</div><div class="gate-caption">Magnitude adjustment qₐ</div></div></div>`;
  }
  $('gate-panel').innerHTML = html;
}

function resetManual() {
  const s = sample(); if (!s) return;
  state.manual = isClass() ? { q: s.q } : { q_sign: s.q_sign, q_magnitude: s.q_magnitude };
}

function computeManual(s, values, classification) {
  if (classification) return s.text_prob.map((p, i) => (1 - values.q) * p + values.q * s.evidence_prob[i]);
  const r = s.readout_state;
  if (!r) return null;
  const residual = (Math.abs(r.text_sign_logit) + r.beta) * Math.tanh(r.fusion_sign_logit);
  const sign = r.text_sign_logit + values.q_sign * residual;
  const magnitude = (1 - values.q_magnitude) * r.text_magnitude + values.q_magnitude * r.fused_magnitude;
  return Math.tanh(sign) * magnitude;
}

function renderInteractive() {
  const s = sample();
  const controls = isClass() ? [['q', 'Evidence contribution q', s.q]] : [['q_sign', 'Polarity correction qₛ', s.q_sign], ['q_magnitude', 'Magnitude adjustment qₐ', s.q_magnitude]];
  $('interactive').innerHTML = `<div class="interactive-head"><div><h2 class="interactive-title">Explore correction with fixed evidence</h2><p class="section-description">Adjust the coefficients to explore the readout equation. Saved predictions remain fixed; the encoder is not rerun.</p></div><button id="reset-gate" class="button outline">Reset coefficients</button></div>` + controls.map(([key, label]) => `<label class="slider-row"><span>${label}</span><input type="range" min="0" max="1" step="0.001" value="${state.manual[key]}" data-gate="${key}" aria-label="${label}"><output id="value-${key}">${fixed(state.manual[key])}</output></label>`).join('') + `<div class="formula">${isClass() ? 'p = (1 − q) × p_text + q × p_evidence' : 's = t + qₛ × (|t| + 0.5) × tanh(f)<br>a = (1 − qₐ) × a_text + qₐ × a_evidence<br>prediction = tanh(s) × a'}</div><div id="manual-output" class="manual-grid"></div>`;
  $('interactive').querySelectorAll('input').forEach(input => input.addEventListener('input', () => { state.manual[input.dataset.gate] = Number(input.value); $(`value-${input.dataset.gate}`).textContent = fixed(Number(input.value)); renderManual(); }));
  $('reset-gate').addEventListener('click', () => { resetManual(); renderInteractive(); });
  renderManual();
}

function renderManual() {
  const s = sample(); const value = computeManual(s, state.manual, isClass());
  if (value === null) { $('manual-output').innerHTML = '<div class="muted">The cache does not contain all required readout components.</div>'; return; }
  if (isClass()) {
    const actual = argmax(s.corrected_prob), manual = argmax(value);
    $('manual-output').innerHTML = `<div class="manual-card corrected"><h3>Saved model output</h3><div class="manual-label">${EMOJI[actual]} ${CLASS_NAMES[actual]}</div>${bars(s.corrected_prob)}<div class="manual-note">Model coefficient q = ${fixed(s.q)}</div></div><div class="manual-card corrected"><h3>Output at current coefficients</h3><div class="manual-label">${EMOJI[manual]} ${CLASS_NAMES[manual]}</div>${bars(value)}<div class="manual-note">Manual coefficient q = ${fixed(state.manual.q)}</div></div>`;
  } else {
    $('manual-output').innerHTML = `<div class="manual-card"><h3>Saved model output</h3><div class="manual-label">${signed(s.corrected)}</div><div class="manual-note">Absolute error ${fixed(Math.abs(s.corrected - s.y))}</div></div><div class="manual-card"><h3>Output at current coefficients</h3><div class="manual-label">${signed(value)}</div><div class="manual-note">Absolute error ${fixed(Math.abs(value - s.y))}</div></div>`;
  }
}

function renderSample() {
  const s = sample();
  const localText = state.localTexts?.[state.dataset]?.[s.id];
  $('input-panel').hidden = typeof localText !== 'string' || !localText.trim();
  $('input-text').textContent = typeof localText === 'string' ? localText : '';
  const truth = isClass() ? `${EMOJI[s.y]} ${CLASS_NAMES[s.y]}` : signed(s.y);
  $('sample-header').innerHTML = `<div><div class="sample-title">${scenarioName(s.scenario)} <span class="tag ${esc(s.scenario)}">${isClass() ? 'Emotion classification' : 'Sentiment regression'}</span></div><div class="sample-id">${esc(s.id)}</div></div><div class="truth">Ground truth<strong>${truth}</strong></div>`;
  $('predictions').innerHTML = predictionCard(s, 'text', 'Text anchor', '01') + predictionCard(s, 'evidence', 'Multimodal evidence', '02') + predictionCard(s, 'corrected', 'Corrected output', '03');
  renderSource(s); renderGate(s); renderInteractive();
  $('sample-note').textContent = isClass() ? 'Class order: happy/excited, sad, neutral, angry. Outcome labels compare the text and corrected predictions with ground truth.' : 'Outcome labels reflect changes in per-example absolute error. Scalar correction uses separate polarity and magnitude gates.';
  $('download').disabled = false;
}

function renderResults() {
  if (!state.metrics) { $('results-view').innerHTML = '<p>Results have not loaded.</p>'; return; }
  const records = state.metrics.datasets;
  $('results-view').innerHTML = '<div class="result-intro"><h2 class="section-title">Five-seed test results</h2><p class="section-description">Test-set mean and sample standard deviation across five seeds, reported separately from the seed-3 validation examples.</p></div><div class="result-grid">' + Object.entries(records).map(([name, entry]) => `<article class="panel result-card"><h2>${esc(entry.name || name.toUpperCase())}</h2><div class="muted tiny">TEST · SEEDS 1–5</div>${entry.metrics.map(m => `<div class="result-metric"><span>${esc(m.label)}</span><strong>${esc(m.mean)} <small>± ${esc(m.sd)}</small></strong></div>`).join('')}</article>`).join('') + '</div><section class="panel" style="margin-top:18px"><h2 class="section-title">A shared evidence architecture with task-specific readouts</h2><p class="section-description" style="margin:0">All three datasets use text, audio and vision. MOSI/MOSEI predict continuous sentiment; IEMOCAP predicts four emotion classes. The common evidence architecture uses task-specific readouts.</p></section>';
}

function render() {
  document.querySelectorAll('[data-dataset]').forEach(b => b.classList.toggle('active', b.dataset.dataset === state.dataset));
  document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === state.view));
  $('cohort').textContent = `VALIDATION · SEED ${ds().seed} · EPOCH ${ds().epoch} · n=${ds().n.toLocaleString('en-US')}`;
  const results = state.view === 'results';
  document.querySelector('.dataset-tabs').hidden = results;
  $('example-layout').hidden = results; $('results-view').hidden = !results;
  $('example-view').hidden = state.view !== 'examples'; $('correction-view').hidden = state.view !== 'correction';
  if (results) { $('download').disabled = true; renderResults(); }
  else { renderList(); renderSample(); }
}

document.querySelectorAll('[data-dataset]').forEach(b => b.addEventListener('click', () => { state.dataset = b.dataset.dataset; chooseInitial(); }));
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => { state.view = b.dataset.view; render(); }));
$('scenario').addEventListener('change', e => { state.filter = e.target.value; state.page = 0; const records = filtered(); if (records.length) { state.selected = records[0].id; resetManual(); } renderList(); renderSample(); });
$('search').addEventListener('input', e => { state.search = e.target.value; state.page = 0; renderList(); });
$('previous').addEventListener('click', () => { state.page--; renderList(); });
$('next').addEventListener('click', () => { state.page++; renderList(); });
$('download').addEventListener('click', () => {
  const record = { dataset: state.dataset, split: ds().split, seed: ds().seed, epoch: ds().epoch, mode: state.data.mode, sample: sample() };
  const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = `TARC_${state.dataset}_${sample().id.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});

Promise.all([fetch('data.json').then(r => { if (!r.ok) throw new Error('Could not load data.json'); return r.json(); }), fetch('results.json').then(r => { if (!r.ok) throw new Error('Could not load results.json'); return r.json(); }), fetch('local-texts.json').then(r => r.ok ? r.json() : { datasets: {} }).catch(() => ({ datasets: {} }))]).then(([data, metrics, localTexts]) => {
  state.data = data; state.metrics = metrics; state.localTexts = localTexts.datasets || {}; $('loading').hidden = true; $('workspace').hidden = false; chooseInitial();
}).catch(error => { $('loading').textContent = `Could not load demo data: ${error.message}. Start the complete demo using python start_demo.py or a static HTTP server.`; });

// The browser test reads this small interface; it cannot modify archived records.
window.TARCDemo = { getState: () => ({ dataset: state.dataset, view: state.view, selected: state.selected }), getSample: () => sample(), computeManual };
