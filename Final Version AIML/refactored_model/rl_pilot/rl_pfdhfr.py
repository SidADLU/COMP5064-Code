#!/usr/bin/env python3
"""
rl_pfdhfr.py -- EXPERIMENT, not part of the shipped tool: a time-boxed reinforcement-learning pilot that fine-tunes the held-out exemplar generator so that its molecules
dock better at the PfDHFR receptor (1J3I, plain Vina, the one receptor trusted without a post-hoc rule).

WHAT IT DOES
  Each step: draw B seed ligands of T89515 -> the model writes one analog per seed (same sampling as generate_for_request.py) -> valid, novel, in-range molecules are docked at
  PfDHFR -> reward = percentile of the molecule's SIZE-CORRECTED score among the calibration decoys (1.0 = better than every decoy of its size, 0.5 = typical decoy; invalid,
  known, repeated, out-of-size-range, near-copy and undockable molecules get 0) -> REINFORCE update with a per-seed baseline and a KL penalty to the starting model.
  The size correction and the decoys are exactly those of the frozen calibration (read from a dock_results CSV), so the reward is the quantity the evaluation reports.

WHAT IT DOES NOT DO
  It edits no existing file.  Everything it writes goes to rl_pilot/runs/<tag>/ (log, molecules, docking cache, checkpoints), so deleting the rl_pilot folder removes the experiment.
  The model is evaluated afterwards with the ordinary scripts (generate_for_request.py, dock_generated.py, check_exemplar_conditioning.py); see RL_PILOT_README.txt.

USAGE  (run from the project folder, the one that holds config.py)
  plumbing test, no Vina  :  python rl_pilot\\rl_pfdhfr.py --smoke
  one real docking step   :  python rl_pilot\\rl_pfdhfr.py --tag smoke_real --steps 1 --batch 4 --workers 1
  pilot run               :  python rl_pilot\\rl_pfdhfr.py --tag r1
  gated reward (r3)       :  python rl_pilot\\rl_pfdhfr.py --tag r3 --reward gated --batch 64 --lr 1e-4 --max-minutes 120
  gate pass rate, no train:  python rl_pilot\\rl_pfdhfr.py --tag probe --reward gated --probe-steps 10 --batch 64

GATED REWARD (--reward gated)
  The original reward pays for any molecule that docks well, so the model can drift to molecules that Vina likes but that are no longer analogs of their seed and lack the antifolate
  pharmacophore (run r2 did exactly that).  The gated reward pays the same decoy percentile ONLY for molecules that (1) carry the pharmacophore the calibration validates, (2) have
  Tanimoto similarity >= --min-sim to their seed, and (3) are within --size-window heavy atoms of their seed.  Everything else gets --fail-reward and is not docked, which also makes a
  step cheaper.  The aim is 'dock better than the seed-like decoys while remaining an analog of a known ligand'.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
import config as cfg

CASE_KEY = 'malaria_t89515'
TARGET = 'T89515'
RECEPTOR = 'PfDHFR'
EXHAUSTIVENESS = 8
VINA_SEED = 42

# worker-process state: the receptor and docking settings, set once per process by _worker_init
_W = {}


# Makes every vina.exe call started through docking.py use at most n_cpu threads (Vina's own --cpu option).  Without this each of the parallel workers lets Vina grab all cores, the
# workers fight over them and parallel docking ends up no faster than docking one molecule at a time.  Used for TRAINING dockings only; the evaluation scripts are not touched.
def limit_vina_cpu(docking_module, n_cpu):
    orig_run = docking_module.subprocess.run

    def run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and '--cpu' not in cmd:
            cmd = list(cmd) + ['--cpu', str(n_cpu)]
        return orig_run(cmd, *args, **kwargs)
    docking_module.subprocess.run = run


# Runs once in every docking worker process: loads docking.py, points it at vina.exe, applies the thread limit (0 = leave Vina's default) and remembers the receptor and settings.
def _worker_init(rec, vina, exh, vseed, n_cpu=0):
    import docking
    docking.VINA_EXECUTABLE_PATH = vina
    if n_cpu > 0:
        limit_vina_cpu(docking, n_cpu)
    _W.update(docking=docking, rec=rec, exh=exh, vseed=vseed)


# Docks one SMILES at the receptor exactly as dock_generated.py does for a plain-Vina receptor (best of one pose).  Returns the score, None if the molecule could not be docked (no pose or
# no 3D structure), or the error text (a string) if something went wrong, so that a broken Vina set-up is reported instead of looking like many failed molecules.
def _worker_dock(smi):
    try:
        rec = _W['rec']
        r = _W['docking'].dock_ligand(smi, rec['pdbqt'], rec['center'], rec['size'], exhaustiveness=_W['exh'], n_poses=1, vina_seed=_W['vseed'])
        return None if r is None else float(r['best_score'])
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ''}"


# Made-up score for the plumbing test (--smoke): a size trend plus a fixed per-molecule offset, so the same SMILES always gets the same number.  Never interpret it.
def fake_score(smi, ha):
    return -(4.0 + 0.17 * ha) - (zlib.crc32(smi.encode()) % 1000) / 500.0


# The calibration decoys of the case, read from a dock_results CSV written by dock_generated.py (rows with role calibration_decoy).  Returns (heavy atoms, scores) as arrays.
def load_decoys(path, case_key=CASE_KEY):
    ha, sc = [], []
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r.get('role') == 'calibration_decoy' and r.get('case') == case_key and r.get('score') not in (None, ''):
                ha.append(float(r['heavy_atoms']))
                sc.append(float(r['score']))
    return np.array(ha), np.array(sc)


# The reward function: percentile among the decoys of the size-corrected score.  The size fit is the one dock_generated.py uses (score ~ heavy atoms, fitted on the decoys).
class DecoyPercentile:

    # Fits the size trend on the decoys and keeps their residuals.
    def __init__(self, ha, sc):
        from dock_generated import fit_size, resid
        self.fit = fit_size(ha, sc)
        self.decoy_resid = resid(ha, sc, self.fit)
        self.ha_min, self.ha_max = int(ha.min()), int(ha.max())

    # Percentile (0..1, higher = better) of one docked molecule.
    def __call__(self, ha, score):
        from dock_generated import resid, percentile_lower
        return float(percentile_lower(resid([ha], [score], self.fit), self.decoy_resid)[0])


# Reads the command line.  --smoke is a shortcut for a 2-step plumbing test with made-up docking scores.
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='r1', help='run name; everything is written to rl_pilot/runs/<tag>/')
    p.add_argument('--checkpoint', default=cfg.DEFAULT_HELDOUT_CKPT, help='the model to start from (default: the held-out model)')
    p.add_argument('--csv', default=cfg.DEFAULT_CSV)
    p.add_argument('--decoys-from', default='dock_results_malaria_ho70_s1.csv', help='a dock_results CSV that holds the calibration decoys of malaria_t89515')
    p.add_argument('--steps', type=int, default=60, help='maximum number of updates')
    p.add_argument('--max-minutes', type=float, default=75.0, help='stop (and save) after this many minutes, whatever the step count: the time box')
    p.add_argument('--batch', type=int, default=32, help='seeds (and samples) per step')
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--kl', type=float, default=0.05, help='weight of the KL penalty that keeps the model near its starting point')
    p.add_argument('--temperature', type=float, default=0.7, help='sampling temperature (the evaluation uses 0.7)')
    p.add_argument('--max-lib-sim', type=float, default=0.85, help='reward 0 for a molecule this similar (Tanimoto) to a library molecule: a near-copy is not a new molecule')
    p.add_argument('--repeat-window', type=int, default=2000, help='a molecule already produced within this many recent molecules gets reward 0')
    p.add_argument('--workers', type=int, default=3, help='parallel vina processes')
    p.add_argument('--vina-cpu', type=int, default=-1, help='threads per vina process; -1 = automatic (CPU cores // workers), 0 = leave Vina\'s default (all cores)')
    p.add_argument('--dock-timeout', type=float, default=300.0, help='seconds to wait for one docking before counting it as failed')
    p.add_argument('--min-validity', type=float, default=0.30, help='stop if validity stays below this for 3 steps in a row (collapse guard)')
    p.add_argument('--save-every', type=int, default=0, help='also save a checkpoint every N steps (0 = only at the end)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reward', choices=['percentile', 'gated'], default='percentile',
                   help='percentile = the original reward (decoy percentile of any molecule); gated = the same percentile, but only for molecules that stay inside the domain where Vina was validated '
                        '(see --min-sim, --size-window and the pharmacophore rule); every other molecule gets --fail-reward and is not docked')
    p.add_argument('--min-sim', type=float, default=0.40, help='gated reward: the analog must have at least this Tanimoto similarity to its seed (0.40 = the lower edge of the analog range used in the evaluation)')
    p.add_argument('--size-window', type=int, default=4, help='gated reward: the analog may differ from its seed by at most this many heavy atoms')
    p.add_argument('--fail-reward', type=float, default=-0.5, help='gated reward: reward of a molecule that fails the gate or is not valid/novel/dockable (passing molecules score 0..1)')
    p.add_argument('--probe-steps', type=int, default=0, help='no training and no docking: sample this many batches from the starting model and print how many molecules would pass the gate, then stop')
    p.add_argument('--smoke', action='store_true', help='plumbing test: 2 steps, 6 seeds, made-up docking scores (no Vina), tag "smoke"')
    a = p.parse_args()
    if a.smoke:
        a.tag, a.steps, a.batch, a.workers = 'smoke', 2, 6, 1
        a.fake_dock = True
    else:
        a.fake_dock = False
    return a


# Appends one row to a CSV file, writing the header first if the file is new.
def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# Judges every sampled molecule and returns one dict per sample (status, cleaned SMILES, heavy atoms, similarities).  Only 'ok' molecules are docked.  Statuses: invalid (does not parse),
# known (already in the library), repeat (made earlier in this batch or within the recent window), unsupported (an element Vina cannot dock), size (outside the decoys' size range),
# near_copy (too similar to a library molecule), gated (only with a gate: fails the pharmacophore, the similarity-to-seed or the size-window rule), ok.  Updates `recent` with the molecules it has seen.
# gate = None (original reward) or a dict {'smarts': pharmacophore SMARTS, 'min_sim': .., 'size_window': .., 'seed_ha': {seed: heavy atoms}}; with a gate each judged molecule also carries the
# three flags pharm / sim_ok / size_ok (None when it never reached the check).
def judge_samples(seeds, smis, chem, be, known_can, recent, lib_fps, reward_fn, max_lib_sim, gate=None):
    batch_seen, info = set(), []
    for seed, smi in zip(seeds, smis):
        d = {'seed': seed, 'smiles': smi, 'status': 'invalid', 'cc': None, 'ha': None, 'sim_seed': float('nan'), 'near': float('nan'), 'pharm': None, 'sim_ok': None, 'size_ok': None, 'ha_diff': None}
        c = chem.canon(smi) if smi else None
        if c is not None:
            if c in known_can:
                d['status'] = 'known'
            elif c in batch_seen or recent.count(c) > 0:
                d['status'] = 'repeat'
            else:
                cc, _ = be.clean(c, mixtures='largest')
                desc = be.describe(cc) if cc else None
                if desc is None:
                    d['status'] = 'unsupported'
                elif not (reward_fn.ha_min <= desc['ha'] <= reward_fn.ha_max):
                    d['status'] = 'size'
                else:
                    fp = chem.fp(cc)
                    near = max(chem.DataStructs.BulkTanimotoSimilarity(fp, lib_fps)) if fp is not None else 0.0
                    d.update(cc=cc, ha=desc['ha'], sim_seed=chem.sim(cc, seed), near=near)
                    d['status'] = 'near_copy' if near > max_lib_sim else 'ok'
                    if gate is not None:
                        sh = gate['seed_ha'].get(seed)
                        d['pharm'] = bool(be.matches(cc, gate['smarts']))
                        d['sim_ok'] = bool(d['sim_seed'] >= gate['min_sim'])
                        d['ha_diff'] = None if sh is None else abs(desc['ha'] - sh)
                        d['size_ok'] = True if sh is None else bool(d['ha_diff'] <= gate['size_window'])
                        if d['status'] == 'ok' and not (d['pharm'] and d['sim_ok'] and d['size_ok']):
                            d['status'] = 'gated'
            if d['status'] in ('ok', 'near_copy', 'size', 'gated'):
                batch_seen.add(c)
                recent.add(c)
        info.append(d)
    return info


# Docks the 'ok' molecules that are not in the cache yet, in parallel through pool_exec (or with made-up scores when fake), and stores every result, failures included.
def dock_new(info, cache, key, pool_exec, fake, timeout):
    todo = sorted({d['cc'] for d in info if d['status'] == 'ok' and cache.get(key(d['cc'])) is None})
    if not todo:
        return
    if fake:
        ha_of = {d['cc']: d['ha'] for d in info if d['status'] == 'ok'}
        res = [fake_score(s, ha_of[s]) for s in todo]
    else:
        futs = [pool_exec.submit(_worker_dock, s) for s in todo]
        res = []
        for f in futs:
            try:
                res.append(f.result(timeout=timeout))
            except Exception as e:
                res.append(f"{type(e).__name__} (timeout or crashed worker)")
    for s, r in zip(todo, res):
        if isinstance(r, str):
            cache.put(key(s), None, r)
        else:
            cache.put(key(s), r, None if r is not None else 'no pose / embedding failed')


# Turns the docked scores into rewards (percentile among the decoys; fail_reward, 0 in the original reward, for anything that was not docked) and marks each sample 'docked' or 'dock_failed'.
# Returns the reward array.
def score_samples(info, cache, key, reward_fn, fail_reward=0.0):
    rewards = np.full(len(info), float(fail_reward))
    for i, d in enumerate(info):
        if d['status'] == 'ok':
            r_ = cache.get(key(d['cc']))
            if r_ is not None and r_['score'] is not None:
                d['score'] = float(r_['score'])
                d['pct'] = reward_fn(d['ha'], d['score'])
                rewards[i] = d['pct']
                d['status'] = 'docked'
            else:
                d['status'] = 'dock_failed'
        d['reward'] = float(rewards[i])
    return rewards


# One log row for a step: counts per status, validity, mean percentile (the training signal), size, similarity to the seed and to the library, share of unique molecules, KL and loss.
def step_row(step, info, rewards, kl_v, loss_v, t_start, t_step):
    n = len(info)
    docked = [d for d in info if d['status'] == 'docked']
    count = lambda st: sum(d['status'] == st for d in info)
    mean = lambda key, nd: round(float(np.nanmean([d[key] for d in docked])), nd) if docked else ''
    checked = [d for d in info if d['pharm'] is not None]          # molecules that reached the gate check (gated reward only)
    share = lambda k: round(float(np.mean([d[k] for d in checked])), 3) if checked else ''
    return {'step': step, 'n_gated': count('gated'), 'pharm_frac': share('pharm'), 'sim_ok_frac': share('sim_ok'), 'size_ok_frac': share('size_ok'), 'minutes': round((time.time() - t_start) / 60.0, 2), 'n': n, 'validity': round(1.0 - count('invalid') / n, 3),
            'n_known': count('known'), 'n_repeat': count('repeat'), 'n_size': count('size'), 'n_near_copy': count('near_copy'),
            'n_docked': len(docked), 'n_dock_failed': count('dock_failed'),
            'pct_mean': mean('pct', 4), 'reward_mean': round(float(np.mean(rewards)), 4), 'ha_mean': mean('ha', 2), 'sim_seed_mean': mean('sim_seed', 3),
            'near_lib_mean': mean('near', 3), 'unique_frac': round(len({d['cc'] for d in docked}) / max(len(docked), 1), 3) if docked else '',
            'kl': round(kl_v, 5) if kl_v == kl_v else '', 'loss': round(loss_v, 5) if loss_v == loss_v else '', 'step_seconds': round(time.time() - t_step, 1)}


# The whole pilot: set-up, then the generate -> dock -> update loop until the step limit, the time limit or the collapse guard stops it, then the checkpoint and the summary.
def main():
    a = parse_args()
    import contextlib
    import torch
    import exemplar_data as xd
    import exemplar_model as xm
    import generate_for_request as gfr
    import dock_generated as dg
    from rl_utils import PerContextBaseline, RollingMultiset
    from concurrent.futures import ProcessPoolExecutor

    t_start = time.time()
    out_dir = os.path.join(HERE, 'runs', a.tag)
    os.makedirs(out_dir, exist_ok=True)
    log_path, mol_path = os.path.join(out_dir, 'log.csv'), os.path.join(out_dir, 'molecules.csv')
    if os.path.exists(log_path) and not a.smoke and not a.probe_steps:
        raise SystemExit(f"{log_path} exists: use a new --tag (a run is never overwritten)")
    for f in (log_path, mol_path):
        if a.smoke and os.path.exists(f):
            os.remove(f)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reward: the calibration decoys and their size fit
    dpath = cfg.find_file(a.decoys_from)
    if not dpath:
        raise SystemExit(f"{a.decoys_from} not found: pass --decoys-from <a dock_results CSV from dock_generated.py with the malaria_t89515 calibration>")
    d_ha, d_sc = load_decoys(dpath)
    if len(d_ha) < 50:
        raise SystemExit(f"only {len(d_ha)} calibration decoys of {CASE_KEY} in {dpath}")
    reward_fn = DecoyPercentile(d_ha, d_sc)
    print(f"[reward] {len(d_ha)} decoys from {os.path.basename(dpath)}; size fit slope {reward_fn.fit[0]:.3f} intercept {reward_fn.fit[1]:.2f}; decoy size range {reward_fn.ha_min}-{reward_fn.ha_max} heavy atoms (molecules outside it get reward 0)")

    # model (trained) and a frozen copy (the KL reference)
    model, pp, ckpt = xm.load_exemplar_checkpoint(a.checkpoint, dev)
    ref, _, _ = xm.load_exemplar_checkpoint(a.checkpoint, dev)
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()
    model.eval()
    k_max, ex_max_len, max_list = ckpt['max_exemplars'], ckpt['ex_max_len'], ckpt['max_list']
    print(f"[model] {os.path.basename(a.checkpoint)} on {dev}; {sum(p_.numel() for p_ in model.parameters()):,} parameters")

    # library, known molecules and seed pool (the same calls as generate_for_request.py)
    rows = xm.load_rows(a.csv, pp)
    lib = xd.Library(rows, max_list, ok=lambda k: len(k) + 2 <= ex_max_len)
    t_label, _ = gfr.resolve(TARGET, lib.pool_ok['t'].keys(), 'target')
    if t_label is None:
        raise SystemExit(f"target {TARGET} has no usable ligands in the library")
    pool = lib.ok_pool('t', t_label)
    chem = xm.Chem2()
    arr_fn = xm.make_arr_fn(pp)
    print("[library] fingerprinting the library (one-off, ~10-30 s) ...")
    known_can = {c for c in (chem.canon(r['key']) for r in rows) if c}
    lib_keys = sorted({r['key'] for r in rows})
    lib_fps = [chem.fp(k) for k in lib_keys]
    lib_fps = [f for f in lib_fps if f is not None]

    # docking set-up: receptor, backend (for cleaning / size / pharmacophore), cache
    case = next(c for c in dg.CASES if c['key'] == CASE_KEY)
    from targets import DISEASE_TARGET_CONFIGS
    cases, receptors = dg.build_cases(DISEASE_TARGET_CONFIGS, [CASE_KEY])
    rec = receptors[RECEPTOR]
    rec['sig'] = dg.receptor_signature(rec)
    import docking
    vina = cfg.DEFAULT_VINA
    if not a.fake_dock and not os.path.exists(vina):
        raise SystemExit(f"vina.exe not found at {vina}")
    docking.VINA_EXECUTABLE_PATH = vina
    be = dg.RealBackend(docking)
    bad = be.self_test()
    if bad:
        raise SystemExit("pharmacophore self-test failed: " + "; ".join(bad))
    cache = dg.DockCache(os.path.join(out_dir, 'dock_cache.jsonl'))

    # seeds: the target's ligands that the evaluation would use (carry the antifolate pharmacophore, not on the drop list)
    dlib = dg.read_library(cfg.find_file(os.path.basename(a.csv)) or a.csv)
    dropped = set(dlib.loc[dlib['name_l'].isin(set(case.get('drop_seed_names', []))), 'smi'])
    eligible, seed_ha = [], {}
    for raw in pool:
        c, _ = be.clean(raw, mixtures='largest')
        if raw not in dropped and c and be.matches(c, case['active_smarts']):
            eligible.append(raw)
            ds_ = be.describe(c)
            seed_ha[raw] = ds_['ha'] if ds_ else None
    print(f"[seeds] {len(eligible)} of the {len(pool)} {TARGET} ligands are eligible (pharmacophore, not on the drop list)")
    if len(eligible) < 5:
        raise SystemExit("too few eligible seeds")

    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.0)
    baseline = PerContextBaseline(0.9)
    recent = RollingMultiset(a.repeat_window)
    rng = random.Random(a.seed)
    BOS, PAD = pp.smiles_vocab['<BOS>'], pp.PAD_IDX
    V = model.vocab_size
    banned = torch.zeros(V, dtype=torch.bool, device=dev)
    banned[BOS] = True
    banned[PAD] = True
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev.type == 'cuda' else (lambda: contextlib.nullcontext())

    # log-probability of each sampled token under a model, with the same temperature scaling and token bans as xm.sample_tokens, so the gradient is that of the sampling distribution
    def token_logprobs(m, tokens, E, R):
        with amp():
            lg = m.target_logits(tokens[:, :-1], E, R)
        lg = lg.float() / max(a.temperature, 1e-6)
        lg = lg.masked_fill(banned, float('-inf'))
        tgt = tokens[:, 1:]
        mask = tgt != PAD
        lp = torch.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return lp.masked_fill(~mask, 0.0), mask.float()

    # saves the current model in the same format as the starting checkpoint, so generate_for_request.py and the check scripts can load it
    def save_ckpt(path, step):
        ck = dict(ckpt)
        ck['exemplar_state_dict'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ck['rl_info'] = {'script': 'rl_pfdhfr.py', 'tag': a.tag, 'step': step, 'start_checkpoint': os.path.basename(a.checkpoint), 'args': {k: v for k, v in vars(a).items()}}
        torch.save(ck, path)
        print(f"[save] {path}")

    gated = a.reward == 'gated'
    gate = {'smarts': case['active_smarts'], 'min_sim': a.min_sim, 'size_window': a.size_window, 'seed_ha': seed_ha} if gated else None
    fail_reward = a.fail_reward if gated else 0.0
    if gated:
        print(f"[reward] GATED: reward = decoy percentile only for molecules with the antifolate pharmacophore, similarity to the seed >= {a.min_sim}, and within "
              f"{a.size_window} heavy atoms of the seed; every other molecule gets {fail_reward} and is not docked")
    pool_exec = None
    if not a.fake_dock and not a.probe_steps:
        n_cpu = a.vina_cpu if a.vina_cpu >= 0 else max(1, (os.cpu_count() or a.workers) // a.workers)
        print(f"[docking] {a.workers} workers x {n_cpu if n_cpu else 'all'} Vina threads each ({os.cpu_count()} CPU cores reported)")
        pool_exec = ProcessPoolExecutor(max_workers=a.workers, initializer=_worker_init, initargs=(rec, vina, EXHAUSTIVENESS, VINA_SEED, n_cpu))
    key = lambda smi: f"{RECEPTOR}|{rec['sig']}|{EXHAUSTIVENESS}|{VINA_SEED}|{smi}"
    hist = []
    low_validity = 0
    dock_fail_streak = 0
    step = 0
    probe = []
    stop_reason = 'step limit'
    print(f"[run] tag {a.tag}: up to {a.steps} steps x {a.batch} seeds, {a.max_minutes:.0f} min time box, lr {a.lr}, KL weight {a.kl}, {'FAKE docking' if a.fake_dock else str(a.workers) + ' docking workers'}")
    try:
        while step < a.steps:
            if (time.time() - t_start) / 60.0 > a.max_minutes:
                stop_reason = 'time box'
                break
            step += 1
            t_step = time.time()
            seeds = rng.sample(eligible, a.batch) if len(eligible) >= a.batch else [rng.choice(eligible) for _ in range(a.batch)]
            items = [('C', [s], [xd.ROLE_TARGET]) for s in seeds]
            _, E, R = xm.build_arrays(items, k_max, ex_max_len, arr_fn, arr_fn, pad=PAD)
            E, R = xm.to_device([E, R], dev)
            tokens = xm.sample_tokens(model, pp, E, R, a.temperature)
            smis = xm.decode_tokens(tokens, pp)

            info = judge_samples(seeds, smis, chem, be, known_can, recent, lib_fps, reward_fn, a.max_lib_sim, gate)
            if a.probe_steps:                       # probe: judge only (no docking, no update)
                probe.extend(info)
                print(f"probe {step}/{a.probe_steps}: {len(probe)} molecules judged so far")
                if step >= a.probe_steps:
                    stop_reason = 'probe finished'
                    break
                continue
            dock_new(info, cache, key, pool_exec, a.fake_dock, a.dock_timeout)
            rewards = score_samples(info, cache, key, reward_fn, fail_reward)

            # policy-gradient update: advantage = reward - per-seed baseline, normalised over the batch
            base = np.array([baseline.get(('s', d['seed'])) for d in info])
            adv = rewards - base
            for d, r_ in zip(info, rewards):
                baseline.update(('s', d['seed']), float(r_))
            sd = adv.std()
            loss_v, kl_v = float('nan'), float('nan')
            if sd > 1e-6:
                adv_t = torch.tensor(np.clip((adv - adv.mean()) / (sd + 1e-6), -3, 3), dtype=torch.float32, device=dev)
                lp, mask = token_logprobs(model, tokens, E, R)
                with torch.no_grad():
                    lp_ref, _ = token_logprobs(ref, tokens, E, R)
                n_tok = mask.sum(1).clamp(min=1.0)
                seq_lp = lp.sum(1) / n_tok
                kl = (((lp - lp_ref) * mask).sum(1) / n_tok).mean()
                loss = -(adv_t * seq_lp).mean() + a.kl * kl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_v, kl_v = float(loss.item()), float(kl.item())

            # statistics, log files, one printed line
            row = step_row(step, info, rewards, kl_v, loss_v, t_start, t_step)
            validity, pct_mean, n, docked = row['validity'], (row['pct_mean'] if row['pct_mean'] != '' else float('nan')), row['n'], row['n_docked']
            append_csv(log_path, row)
            for d in info:
                append_csv(mol_path, {'step': step, 'seed': d['seed'], 'smiles': d['cc'] or d['smiles'], 'status': d['status'], 'heavy_atoms': d['ha'] or '',
                                      'score': d.get('score', ''), 'percentile': round(d['pct'], 4) if 'pct' in d else '', 'reward': round(d['reward'], 4),
                                      'similarity_to_seed': round(d['sim_seed'], 3) if d['sim_seed'] == d['sim_seed'] else '', 'nearest_library_similarity': round(d['near'], 3) if d['near'] == d['near'] else '',
                                      'pharmacophore': '' if d['pharm'] is None else int(d['pharm']), 'sim_ok': '' if d['sim_ok'] is None else int(d['sim_ok']),
                                      'size_ok': '' if d['size_ok'] is None else int(d['size_ok'])})
            hist.append(row)
            gate_txt = f"in-domain {(row['n_docked'] + row['n_dock_failed']) / n:4.0%} | pharm {row['pharm_frac'] or 0:>5} | " if gated else ''
            print(f"step {step:3d}/{a.steps} | valid {validity:4.0%} | {gate_txt}docked {docked:2d}/{n} | pct {pct_mean:5.3f} | reward {rewards.mean():5.3f} | HA {row['ha_mean'] or 0:>5} | "
                  f"sim-seed {row['sim_seed_mean'] or 0:>5} | KL {kl_v:7.4f} | {row['step_seconds']:.0f}s (total {row['minutes']:.1f} min)")
            n_tried = row['n_docked'] + row['n_dock_failed']
            if n_tried >= 3 and row['n_docked'] == 0:
                errs = sorted({(cache.get(key(d['cc'])) or {}).get('error') or '' for d in info if d['status'] == 'dock_failed'})
                print(f"[warn] none of {n_tried} molecules could be docked this step; errors: {errs[:3]}")
                dock_fail_streak += 1
            else:
                dock_fail_streak = 0
            if dock_fail_streak >= 2:
                stop_reason = 'docking is failing (see the errors above): check vina.exe, the receptor file and the Python packages'
                break
            low_validity = low_validity + 1 if validity < a.min_validity else 0
            if low_validity >= 3:
                stop_reason = 'collapse guard (validity below %.0f%% for 3 steps)' % (100 * a.min_validity)
                break
            if a.save_every and step % a.save_every == 0:
                save_ckpt(os.path.join(out_dir, f'checkpoint_rl_{a.tag}_step{step}.pt'), step)
    except KeyboardInterrupt:
        stop_reason = 'stopped by hand'
    finally:
        if pool_exec is not None:
            pool_exec.shutdown(wait=False, cancel_futures=True)

    # probe mode: report the share of molecules that pass each part of the gate, then stop (nothing is saved)
    if a.probe_steps:
        tot = len(probe)
        valid = [d for d in probe if d['status'] != 'invalid']
        reached = [d for d in probe if d['pharm'] is not None]
        ok_all = [d for d in probe if d['status'] == 'ok']
        pct = lambda x, base: f"{(x / base if base else 0):.1%}"
        print(f"\n[probe] {tot} samples from the starting model (reward {a.reward}); gate: pharmacophore, similarity to seed >= {a.min_sim}, within {a.size_window} heavy atoms")
        print(f"        valid {pct(len(valid), tot)} | reached the gate check (valid, new, in size range) {pct(len(reached), tot)}")
        if reached:
            n_r = len(reached)
            print(f"        of those: pharmacophore {pct(sum(d['pharm'] for d in reached), n_r)} | similarity ok {pct(sum(d['sim_ok'] for d in reached), n_r)} | size ok {pct(sum(d['size_ok'] for d in reached), n_r)}")
        print(f"        PASS ALL THREE and dockable: {len(ok_all)} of {tot} = {pct(len(ok_all), tot)}  (about {len(ok_all) / max(step, 1):.1f} per batch of {a.batch})")
        print("        rule of thumb: under ~3 per batch of 64 the reward is too sparse to learn from; lower --temperature or raise --batch.")
        # the same sample under other thresholds (pass = valid, new, in size range, pharmacophore, similarity >= S, heavy-atom difference <= W): molecules per batch
        cand = [d for d in reached if d['status'] in ('ok', 'gated')]
        print(f"\n        molecules passing per batch of {a.batch} under other thresholds (rows: min similarity to the seed; columns: heavy-atom window)")
        wins = (3, 4, 6, 8)
        print('        sim >=   ' + ''.join(f'  +-{w:<2d} atoms' for w in wins))
        for S in (0.25, 0.30, 0.35, 0.40, 0.50):
            cells = [sum(1 for d in cand if d['pharm'] and d['sim_seed'] >= S and (d['ha_diff'] is None or d['ha_diff'] <= w)) / max(step, 1) for w in wins]
            print(f'        {S:<7.2f}  ' + ''.join(f'  {c:>9.1f}' for c in cells))
        return
    # final checkpoint and summary: the training curve compares the first steps (the starting model) with the last ones
    final =os.path.join(out_dir, f'checkpoint_rl_{a.tag}.pt')
    if step > 0:
        save_ckpt(final, step)
    pcts = [h['pct_mean'] for h in hist if h['pct_mean'] != '']
    first, last = pcts[:5], pcts[-10:]
    summ = {'tag': a.tag, 'stop_reason': stop_reason, 'steps_done': step, 'minutes': round((time.time() - t_start) / 60.0, 1),
            'pct_mean_first5': round(float(np.mean(first)), 4) if first else None, 'pct_mean_last10': round(float(np.mean(last)), 4) if last else None,
            'validity_first5': round(float(np.mean([h['validity'] for h in hist[:5]])), 3) if hist else None,
            'validity_last5': round(float(np.mean([h['validity'] for h in hist[-5:]])), 3) if hist else None,
            'ha_last5': [h['ha_mean'] for h in hist[-5:]], 'kl_last5': [h['kl'] for h in hist[-5:]], 'checkpoint': final if step > 0 else None,
            'args': {k: v for k, v in vars(a).items()}}
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1)
    print(f"\n[done] stopped: {stop_reason}; {step} steps in {summ['minutes']} min")
    if first and last:
        print(f"       training reward (mean percentile of docked molecules): first 5 steps {summ['pct_mean_first5']:.3f} -> last 10 steps {summ['pct_mean_last10']:.3f}"
              f"   (0.5 = typical decoy; this is the TRAINING signal, not the evaluation)")
        print(f"       validity {summ['validity_first5']:.0%} -> {summ['validity_last5']:.0%};  mean heavy atoms of the last steps {summ['ha_last5']}")
    print(f"       log: {log_path}\n       molecules: {mol_path}\n       checkpoint: {final if step > 0 else '(none)'}")
    if a.smoke:
        print("       SMOKE TEST finished: the plumbing works (the numbers above are made up).")


if __name__ == '__main__':
    main()
