// app.js -- the page logic: reads the machine state, starts jobs, polls their progress and draws the ranked candidates (2D picture, rotatable 3D view, docking and properties).
'use strict';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const STAGE_INFO = {
  train: ['Train exemplar model', 'Minutes to hours, depending on the GPU (about 13 minutes for 6,000 steps on the study machine). Off by default; the checkpoint chosen above is used otherwise.'],
  generate: ['Generate molecules', 'The seed-and-edit model writes analogs of the target’s known ligands.'],
  baseline: ['Random-seed baseline', 'Same model, seeds from the whole library: the control for "did the target choice help?"'],
  dock: ['Dock at the validated receptor', 'Vina at a receptor that passed calibration; scores are size-corrected against decoys.'],
  validate: ['Conditioning check', 'Measures how well the model follows its target on held-out targets.'],
};
const STAGE_ORDER = ['train', 'generate', 'baseline', 'dock', 'validate'];
let state = null, since = 0, pollTimer = null, activeJobId = null, jobOrigin = 'pipeline';

// ---------- small helpers ----------

async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-GUI': '1' }, body: JSON.stringify(body) };
  const r = await fetch('/api/' + path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const fmt = (x, d = 2) => (x == null || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);

// ---------- tabs ----------

document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('on', t === b));
  ['pipeline', 'context', 'runs'].forEach((n) => { $('tab-' + n).hidden = n !== b.dataset.tab; });
  if (b.dataset.tab === 'runs') loadRuns();
}));

// ---------- start-up: machine state ----------

async function loadState() {
  state = await api('state');
  $('demo').hidden = !state.demo;
  const missing = [];
  if (!state.demo) {
    if (!state.library) missing.push('CleanedData.csv');
    if (!state.vina) missing.push('vina.exe');
    if (!state.checkpoints.some((c) => c.role === 'exemplar')) missing.push('an exemplar checkpoint (checkpoint_exemplar_*.pt)');
  }
  $('warn').hidden = !missing.length;
  if (missing.length) $('warn').textContent = 'Missing on this machine: ' + missing.join(', ') + '. Generating or docking will fail until they are placed next to the scripts.';
  const ts = $('target_key'); ts.textContent = '';
  state.targets.forEach((t) => { const o = el('option', null, t.label); o.value = t.key; ts.appendChild(o); });
  const c = el('option', null, 'Other (type a disease and/or target)'); c.value = 'custom'; ts.appendChild(c);
  for (const id of ['checkpoint', 'c_checkpoint']) {
    const s = $(id); s.textContent = '';
    state.checkpoints.filter((k) => k.role === 'exemplar').forEach((k) => { const o = el('option', null, k.note ? k.name + '  (' + k.note + ')' : k.name); o.value = k.name; if (k.name === state.default_checkpoint) o.selected = true; s.appendChild(o); });
  }
  for (const id of ['checkpoint', 'c_checkpoint']) $(id).addEventListener('change', ckptHint);
  const pre = $('pretrained'); pre.textContent = '';
  state.checkpoints.filter((k) => k.role === 'pretrained').forEach((k) => { const o = el('option', null, k.note ? k.name + '  (' + k.note + ')' : k.name); o.value = k.name; pre.appendChild(o); });
  if (!pre.children.length) { const o = el('option', null, 'no base model found next to the project'); o.value = ''; pre.appendChild(o); }
  if (state.holdout_default) $('holdout').value = state.holdout_default;
  renderStages(); updateTarget(); ckptHint(); loadRunsSelect();
  if (state.job) attachJob(state.job, 'pipeline');
}

// The targets' trust note and the custom-target fields.
function updateTarget() {
  const key = $('target_key').value, t = state.targets.find((x) => x.key === key);
  $('customrow').hidden = key !== 'custom';
  const note = $('trustnote'); note.textContent = '';
  if (t) {
    const b = el('span', 'badge ' + t.trust, { trusted: 'docking validated', rule: 'docking validated with a rule', none: 'docking not validated' }[t.trust]);
    note.append(b, ' ', t.note);
    const miss = (t.requirements || []).filter((r) => !r.found).map((r) => r.file);
    if (miss.length) note.append(el('div', 'err', 'Files missing for docking: ' + miss.join(', ')));
  } else note.textContent = 'A request for a disease/target of your own. Docking is not offered, because only PfDHFR and DPP-4 have a validated receptor.';
  renderStages(); renderBest();
}

// The best-supported setup for the chosen target (from the study), with a button that fills in the form.  It is advice: nothing runs until you click Run.
function renderBest() {
  const box = $('bestbox'), key = $('target_key').value, t = state.targets.find((x) => x.key === key);
  const b = t ? (state.best_setups || {})[t.key] : state.best_general;
  box.textContent = '';
  if (!b) { box.hidden = true; return; }
  box.hidden = false;
  const have = state.checkpoints.some((c) => c.role === 'exemplar' && c.name === b.model);
  const head = el('div', 'bhead'); head.append(el('b', null, t ? 'Best-supported setup for this target' : 'Best-supported setup for any other target'));
  box.appendChild(head);
  box.appendChild(el('p', 'bline', b.headline));
  const kv = el('dl', 'kv bkv');
  const rows = [['Model', b.model + (have ? '' : '  (not found on this machine)')], ['Temperature', String(b.temperature)], ['Molecules', String(b.n) + (b.stages.includes('dock') ? '  (50 or more are needed for a docking summary)' : '')],
                ['Stages', b.stages.map((s) => STAGE_INFO[s][0]).join(', ')], ['Docking', b.docking]];
  rows.forEach(([k, v]) => kv.append(el('dt', null, k), el('dd', null, v)));
  box.appendChild(kv);
  const more = el('details'); more.appendChild(el('summary', null, 'Why this setup, and what it does not show'));
  more.appendChild(el('p', 'explain', b.evidence));
  const ul = el('ul', 'bcav'); b.caveats.forEach((c) => ul.appendChild(el('li', null, c))); more.appendChild(ul);
  more.appendChild(el('p', 'explain', b.seeds));
  box.appendChild(more);
  const act = el('div', 'actions'), btn = el('button', 'btn small', 'Apply these settings'), msg = el('span', 'note');
  btn.type = 'button'; btn.addEventListener('click', () => { msg.textContent = applyBest(b); }); act.append(btn, msg); box.appendChild(act);
}

// Fills the form with a best-supported setup (model, temperature, number of molecules, stages) and says what it did.  Stages that are not available for the target stay unticked.
function applyBest(b) {
  const sel = $('checkpoint'), have = [...sel.options].some((o) => o.value === b.model);
  if (have) sel.value = b.model;
  $('temperature').value = b.temperature; $('n').value = b.n;
  $('stages').querySelectorAll('input[type=checkbox]').forEach((cb) => { cb.checked = !cb.disabled && b.stages.includes(cb.value); });
  $('trainbox').hidden = !$('stages').querySelector('input[value=train]').checked;
  ckptHint();
  return have ? 'Filled in. Review the form, then click Run.' : 'Filled in, but the model ' + b.model + ' was not found next to the project, so the model list was left as it is.';
}
$('target_key').addEventListener('change', () => { updateTarget(); ckptHint(); });

// Explains the optional PfDHFR-tuned model under the model list, and warns when it is chosen for any target other than Malaria / T89515.
function ckptHint() {
  [['checkpoint', 'ckpt_hint', () => $('target_key').value === 'malaria_t89515'], ['c_checkpoint', 'c_ckpt_hint', () => false]].forEach(([id, hid, isMalaria]) => {
    const sel = $(id); if (!sel) return;
    let h = $(hid);
    if (!h) { h = el('small'); h.id = hid; sel.insertAdjacentElement('afterend', h); }
    const k = state.checkpoints.find((c) => c.name === sel.value);
    h.className = k && k.note && !isMalaria() ? 'err' : 'note';
    h.textContent = !(k && k.note) ? '' : (isMalaria() ? 'Optional model: ' + k.note + '. It keeps the antifolate pharmacophore and stays close to its seeds. The general model is the first in the list.'
                                                       : 'This model is tuned for Malaria / T89515 only. For any other target or a new context use the general model (first in the list).');
  });
}

// The stage checkboxes, with the reason when a stage is not available for the chosen target.
function renderStages() {
  const box = $('stages'), key = $('target_key').value, t = state.targets.find((x) => x.key === key);
  const keep = new Set([...box.querySelectorAll('input:checked')].map((i) => i.value));
  const first = box.children.length === 0;
  box.textContent = '';
  const unavailable = { dock: !(t && t.ready) ? (t && t.trust === 'none' ? 'not offered: the SERT receptor did not validate' : (t && !t.ready ? 'not available: receptor files are missing' : 'not offered for a custom request')) : '',
                        baseline: !t ? 'defined for the study targets only' : '' };
  STAGE_ORDER.forEach((s) => {
    const row = el('label', 'stage'), cb = el('input'); cb.type = 'checkbox'; cb.value = s;
    const why = unavailable[s] || '';
    cb.disabled = !!why;
    cb.checked = !why && (first ? ['generate', 'dock'].includes(s) && !(s === 'dock' && !(t && t.ready)) : keep.has(s));
    const body = el('div'); body.append(el('b', null, STAGE_INFO[s][0]), el('small', null, why || STAGE_INFO[s][1]));
    row.classList.toggle('off', !!why); row.append(cb, body); box.appendChild(row);
    cb.addEventListener('change', () => { $('trainbox').hidden = !box.querySelector('input[value=train]').checked; });
  });
  const pre = el('div', 'stage off'); pre.append(el('span'), (() => { const d = el('div'); d.append(el('b', null, 'Pretrain base model'), el('small', null, 'Not a GUI stage: run pretrain_base_model.py from the command line (see REBUILD_FROM_SCRATCH.md). A rebuilt base model then appears in the Train stage’s base-model list.')); return d; })());
  box.appendChild(pre);
  $('trainbox').hidden = !box.querySelector('input[value=train]').checked;
}

async function loadRunsSelect() {
  const { runs } = await api('runs');
  const s = $('resume_run'); s.textContent = ''; const o = el('option', null, 'New run'); o.value = ''; s.appendChild(o);
  runs.filter((r) => r.has_gen).forEach((r) => { const x = el('option', null, r.id + ' (' + (r.params.title || '') + ')'); x.value = r.id; s.appendChild(x); });
}

// ---------- starting a job ----------

async function startJob(form, origin) {
  jobOrigin = origin;
  const { job } = await api('start', form);
  attachJob(job, origin);
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault(); $('formerr').textContent = '';
  const stages = [...$('stages').querySelectorAll('input:checked')].map((i) => i.value);
  const form = { target_key: $('target_key').value, disease: $('disease').value, target: $('target').value, n: $('n').value, temperature: $('temperature').value, seed: $('seed').value,
                 checkpoint: $('checkpoint').value, resume_run: $('resume_run').value, stages, train_steps: $('train_steps').value, holdout: $('holdout').value, train_out: $('train_out').value, pretrained: $('pretrained').value };
  try { await startJob(form, 'pipeline'); } catch (err) { $('formerr').textContent = err.message; }
});

// New context: save the ligands, then generate with them as an extra library.
$('c_file').addEventListener('change', async () => {
  const f = $('c_file').files[0]; if (!f) return;
  const text = await f.text(), lines = text.split(/\r?\n/).filter(Boolean);
  if (/Canonical Smiles/i.test(lines[0] || '')) {
    const head = lines[0].split(','), i = head.findIndex((h) => /Canonical Smiles/i.test(h));
    $('c_smiles').value = lines.slice(1).map((l) => (l.split(',')[i] || '').replace(/"/g, '').trim()).filter(Boolean).join('\n');
  } else $('c_smiles').value = lines.join('\n');
});
$('ctxform').addEventListener('submit', async (e) => {
  e.preventDefault(); $('c_err').textContent = '';
  try {
    const lib = await api('library', { disease: $('c_disease').value, target: $('c_target').value, smiles: $('c_smiles').value });
    if (lib.n_bad) $('c_err').textContent = lib.n_bad + ' line(s) did not parse and were left out.';
    await startJob({ target_key: 'custom', disease: $('c_disease').value, target: $('c_target').value, n: $('c_n').value, temperature: $('c_temperature').value, seed: $('c_seed').value,
                     checkpoint: $('c_checkpoint').value, stages: ['generate'], extra_library: lib.file }, 'context');
  } catch (err) { $('c_err').textContent = err.message; }
});
$('cancel').addEventListener('click', () => api('cancel', {}));

// ---------- job progress ----------

function attachJob(job, origin) {
  activeJobId = job.id; jobOrigin = origin || jobOrigin;
  $('log').textContent = ''; $('c_log').textContent = '';
  drawJob(job); since = job.next;
  clearInterval(pollTimer);
  if (job.status === 'running') { pollTimer = setInterval(poll, 1200); poll(); }
}

async function poll() {
  const { job } = await api('job?since=' + since);
  if (!job) return;
  since = job.next;
  drawJob(job);
  if (job.status !== 'running') {
    clearInterval(pollTimer); pollTimer = null;
    if (job.status === 'done' && (job.params.stages || []).some((s) => s === 'generate' || s === 'dock')) showResults(job.id, jobOrigin);
    loadRunsSelect();
  }
}

function drawJob(job, append) {
  const running = job.status === 'running';
  $('jobbar').hidden = !running;
  const cur = job.stages.find((s) => s.status === 'running');
  $('jobtext').textContent = 'Running: ' + (job.params.title || job.id) + (cur ? ' – ' + cur.label : '');
  const host = jobOrigin === 'context' ? $('c_progress') : $('progress');
  host.textContent = '';
  const head = el('p', 'note', 'Run ' + job.id + ' – ' + job.status + ' (' + job.elapsed + ' s)'); host.appendChild(head);
  job.stages.forEach((s) => {
    const row = el('div', 'prow'); row.append(el('b', null, s.label), el('span', 'chip ' + s.status, s.status));
    const bar = el('div', 'bar' + (s.status === 'running' && s.progress == null ? ' indet' : '')), fill = el('i');
    fill.style.width = (s.status === 'done' ? 100 : Math.round((s.progress || 0) * 100)) + '%'; bar.appendChild(fill);
    if (s.status === 'running' || s.status === 'done' || s.status === 'failed') row.appendChild(bar);
    const last = s.note || s.last; if (last) row.appendChild(el('div', 'last', last));
    host.appendChild(row);
  });
  const log = jobOrigin === 'context' ? $('c_log') : $('log');
  if (job.log && job.log.length) { const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 30; log.textContent += job.log.join('\n') + '\n'; if (stick) log.scrollTop = log.scrollHeight; }
}

// ---------- results ----------

async function showResults(runId, origin) {
  const host = $('results-' + (origin || 'pipeline'));
  host.textContent = '';
  let data;
  try { data = await api('candidates?run=' + encodeURIComponent(runId)); } catch (e) { host.appendChild(el('p', 'err', e.message)); return; }
  host.appendChild(renderResults(data, runId));
}

function renderResults(d, runId) {
  const root = el('div', 'card');
  const head = el('div', 'rhead'); head.appendChild(el('h2', null, 'Top candidates – ' + d.title));
  if (d.trust) head.appendChild(el('span', 'badge ' + d.trust, { trusted: 'docking validated', rule: 'validated with a rule', none: 'not validated' }[d.trust]));
  [['gen', 'molecules CSV'], ['dock', 'docking CSV'], ['log', 'log']].forEach(([f, label]) => { const a = el('a', null, label); a.href = '/api/download?run=' + encodeURIComponent(runId) + '&file=' + f; head.appendChild(a); });
  root.appendChild(head);
  const why = d.basis === 'docking'
    ? 'Ranked by docking: the size-corrected percentile of each molecule among property-matched decoys at ' + (d.receptor || 'the receptor') + ' (1.0 = better than every decoy, 0.5 = a typical decoy). The mean over this set is ' + fmt(d.mean_percentile) + '. Docking shows whether molecules look like binders at a receptor that passed calibration; it does not measure the model’s conditioning.' + (d.trust === 'rule' ? ' This receptor is validated only with the pre-registered interaction rule: read the scores as rule-constrained.' : '')
    : 'Ranked by similarity to the seed (closest analogs first). That says how faithful an analog is, not how active it is: no validated receptor was docked for this request.';
  root.appendChild(el('p', 'explain', why));
  // Honest fidelity note: the model only sometimes writes close analogs; say so when most of this set is far from its seeds.
  const sims = d.rows.map((r) => r.similarity_to_seed).filter((x) => x != null).sort((a, b) => a - b);
  if (sims.length >= 3) {
    const med = sims[Math.floor(sims.length / 2)], close = sims.filter((x) => x >= 0.40).length / sims.length;
    if (close < 0.25) root.appendChild(el('p', 'err', 'Only ' + Math.round(close * 100) + '% of these molecules are close analogs of their seed ligands (similarity 0.40 or more; median ' + fmt(med) + '). The rest are only loosely related to the ligands you supplied. This is typical when there are few or very different seed ligands. Treat them as loose ideas, not analogs.'));
  }
  if (d.n_flagged) root.appendChild(el('p', 'explain', d.n_flagged + ' of these ' + d.rows.length + ' molecules carry a warning mark (very large, hard to make, or an extreme logP). These are rules of thumb, not filters: marked molecules stay in the list and keep their rank, but they are unlikely to be practical leads as written. Do not pick examples from them.'));
  const cards = el('div', 'cards');
  const shown = d.basis === 'docking' ? d.rows.filter((r) => r.docked) : d.rows;
  shown.slice(0, 5).forEach((r) => cards.appendChild(moleculeCard(r, d)));
  root.appendChild(cards);
  if (!d.rows.length) root.appendChild(el('p', 'muted', 'No candidates in this run yet.'));
  if (d.basis === 'docking') {
    let note = d.n_docked + ' of ' + d.rows.length + ' generated molecules were docked.';
    if (d.n_undocked) note += ' The other ' + d.n_undocked + ' are listed in the table below as “not docked”: docking skips a molecule whose seed lacks the antifolate pharmacophore (that is the usual reason), or one that is too small or large, or a duplicate.';
    if (d.n_docked < 15) note += ' With fewer than about 15 docked molecules the average percentile and any PASS / FAIL verdict in the log are not meaningful; use a larger run (50 or more) to judge the model.';
    root.appendChild(el('p', 'explain', note));
  }
  if (d.rows.length > 5 || d.n_undocked) {
    const w = el('div', 'tablewrap'), t = el('table'), hr = el('tr');
    ['#', 'SMILES', d.basis === 'docking' ? 'Percentile' : '', 'Score', 'Sim. to seed', 'MW', 'logP', 'SA', 'Warning'].filter((x) => x !== '').forEach((h) => hr.appendChild(el('th', null, h)));
    t.appendChild(hr);
    d.rows.forEach((r) => {
      const tr = el('tr'); const p = r.properties || {};
      const cells = [r.rank, r.smiles]; if (d.basis === 'docking') cells.push(r.docked ? fmt(r.percentile) : 'not docked');
      cells.push(fmt(r.score), fmt(r.similarity_to_seed), fmt(p.MW, 1), fmt(p.logP), fmt(p.SA), (r.flags || []).map((f) => f.code).join(', '));
      cells.forEach((c, i) => { const td = el('td', i === 1 ? 's' : null, c); if (i === cells.length - 1 && (r.flags || []).length) { td.className = 'flagcell'; td.title = r.flags.map((f) => f.text).join('; '); } tr.appendChild(td); });
      t.appendChild(tr);
    });
    w.appendChild(t); root.append(el('h2', null, 'All ' + d.rows.length + ' generated molecules'), el('p', 'explain', 'The cards above show the top 5; this table lists every molecule.'), w);
  }
  return root;
}

function moleculeCard(r, d) {
  const c = el('div', 'mcard'), pic = el('div', 'pic');
  const img = el('img'); img.alt = 'structure of ' + r.smiles; img.src = '/api/svg?smiles=' + encodeURIComponent(r.smiles);
  pic.append(el('span', 'rank', '#' + r.rank), img); c.appendChild(pic);
  const b = el('div', 'body'); b.appendChild(el('p', 'smi', r.smiles));
  (r.flags || []).forEach((f) => b.appendChild(el('div', 'flagline', '⚠ ' + f.text)));
  if (d.basis === 'docking') { const dk = el('div', 'dock'); dk.append(el('span', null, 'Docking percentile'), el('strong', null, fmt(r.percentile))); b.appendChild(dk); }
  const kv = el('dl', 'kv'); const p = r.properties || {};
  const rows = [['Vina score (kcal/mol)', r.score == null ? null : fmt(r.score)], ['Similarity to seed', fmt(r.similarity_to_seed)], ['Nearest library molecule', fmt(r.nearest_library)],
                ['Mol. weight', p.MW], ['logP', p.logP], ['H-bond donors / acceptors', p.HBD == null ? null : p.HBD + ' / ' + p.HBA], ['Polar surface area', p.TPSA], ['Rotatable bonds', p.RotB], ['Synthetic accessibility', p.SA]];
  rows.forEach(([k, v]) => { if (v == null || v === '–') return; kv.append(el('dt', null, k), el('dd', null, String(v))); });
  b.appendChild(kv); c.appendChild(b);
  const tools = el('div', 'tools');
  const three = el('button', 'btn small', '3D view'); three.type = 'button';
  three.addEventListener('click', () => toggle3D(pic, img, three, r.smiles));
  const copy = el('button', 'btn small', 'Copy SMILES'); copy.type = 'button';
  copy.addEventListener('click', () => { navigator.clipboard && navigator.clipboard.writeText(r.smiles); copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy SMILES'; }, 1200); });
  tools.append(three, copy); c.appendChild(tools);
  return c;
}

// ---------- a small rotatable 3D ball-and-stick view (no external library) ----------

const CPK = { C: '#555d6b', N: '#2f5bd6', O: '#d63a2f', S: '#c9a400', F: '#2ca05a', Cl: '#2ca05a', Br: '#8a3a2f', I: '#6a2fa0', P: '#e07a00', H: '#cccccc' };

async function toggle3D(pic, img, btn, smiles) {
  const old = pic.querySelector('canvas');
  if (old) { old.remove(); img.hidden = false; btn.textContent = '3D view'; return; }
  btn.textContent = 'Loading…';
  let m;
  try { m = await api('mol3d?smiles=' + encodeURIComponent(smiles)); } catch (e) { btn.textContent = '3D view'; alert('3D view unavailable: ' + e.message); return; }
  img.hidden = true; btn.textContent = '2D view';
  const cv = el('canvas'); pic.appendChild(cv);
  const dpr = window.devicePixelRatio || 1, W = pic.clientWidth, H = 200; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const n = m.atoms.length, cx = m.atoms.reduce((s, a) => s + a[1], 0) / n, cy = m.atoms.reduce((s, a) => s + a[2], 0) / n, cz = m.atoms.reduce((s, a) => s + a[3], 0) / n;
  const P = m.atoms.map((a) => [a[1] - cx, a[2] - cy, a[3] - cz]);
  const R = Math.max(...P.map((p) => Math.hypot(...p))) || 1, scale = Math.min(W, H) * 0.42 / R;
  let ax = 0.3, ay = 0.4, drag = null, auto = true;
  const rot = (p) => { const [x, y, z] = p, c1 = Math.cos(ay), s1 = Math.sin(ay), c2 = Math.cos(ax), s2 = Math.sin(ax); const x1 = x * c1 + z * s1, z1 = -x * s1 + z * c1; return [x1, y * c2 - z1 * s2, y * s2 + z1 * c2]; };
  const draw = () => {
    if (!cv.isConnected) return;
    g.clearRect(0, 0, W, H);
    const Q = P.map(rot), sx = (q) => W / 2 + q[0] * scale, sy = (q) => H / 2 - q[1] * scale;
    const items = [];
    m.bonds.forEach(([i, j, o]) => items.push({ z: (Q[i][2] + Q[j][2]) / 2, f: () => { g.strokeStyle = '#8a93a3'; g.lineWidth = o >= 2 ? 3.2 : 2; g.beginPath(); g.moveTo(sx(Q[i]), sy(Q[i])); g.lineTo(sx(Q[j]), sy(Q[j])); g.stroke(); } }));
    m.atoms.forEach((a, i) => items.push({ z: Q[i][2], f: () => { g.fillStyle = CPK[a[0]] || '#888'; g.beginPath(); g.arc(sx(Q[i]), sy(Q[i]), 5 + 1.5 * (Q[i][2] / R), 0, 7); g.fill(); } }));
    items.sort((a, b) => a.z - b.z).forEach((it) => it.f());
    if (auto) ay += 0.012;
    requestAnimationFrame(draw);
  };
  cv.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; auto = false; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
  cv.addEventListener('pointermove', (e) => { if (!drag) return; ay += (e.clientX - drag[0]) * 0.012; ax += (e.clientY - drag[1]) * 0.012; drag = [e.clientX, e.clientY]; });
  cv.addEventListener('pointerup', () => { drag = null; cv.style.cursor = 'grab'; });
  draw();
}

// ---------- runs tab ----------

async function loadRuns() {
  const { runs } = await api('runs'), host = $('runlist'); host.textContent = '';
  if (!runs.length) { host.appendChild(el('p', 'muted', 'No runs yet.')); return; }
  const t = el('table'), hr = el('tr'); ['Started', 'Title', 'Status', 'Stages', ''].forEach((h) => hr.appendChild(el('th', null, h))); t.appendChild(hr);
  runs.forEach((r) => {
    const tr = el('tr'); tr.append(el('td', null, new Date(r.started * 1000).toLocaleString()), el('td', null, (r.params.title || r.id) + (r.demo ? ' (demo)' : '')), el('td', null, r.status), el('td', null, (r.params.stages || []).join(', ')));
    const td = el('td'); if (r.has_gen) { const b = el('button', 'btn small', 'Show candidates'); b.type = 'button'; b.addEventListener('click', () => showResults(r.id, 'runs')); td.appendChild(b); } tr.appendChild(td); t.appendChild(tr);
  });
  host.appendChild(t);
}

loadState().catch((e) => { $('warn').hidden = false; $('warn').textContent = 'Could not reach the server: ' + e.message; });
