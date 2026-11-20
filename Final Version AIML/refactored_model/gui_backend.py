"""
gui_backend.py -- everything the GUI does that is not drawing the page: which study targets may be docked, which files are present, how a pipeline is turned into command lines,
the job runner that executes them, and the readers that turn the output files into ranked candidates with structures and properties.

The GUI adds no model or docking code.  Every stage runs one of the existing scripts as a separate process, with the same options as the frozen commands:
  train     train_exemplar_generator.py        generate  generate_for_request.py        baseline  generate_for_request.py --random-seeds
  dock      dock_generated.py                  validate  check_exemplar_conditioning.py
Pretraining is not a GUI stage: run pretrain_base_model.py from the command line (REBUILD_FROM_SCRATCH.md).  The Train stage can use any base model found next to the project (see find_checkpoints).

Everything a run produces is written to gui_runs/<run id>/ (run.json, log.txt, gen.csv, ...).  Uploaded ligand files go to gui_runs/uploads/.  Neither folder is part of the shipped tool.
Command lines are built as argument lists from validated values (no shell), and only the scripts named above can be started.

DEMO mode (gui_server.py --demo) replaces every script with a small stand-in that prints progress and writes made-up files, so the page can be tried without a model, Vina or RDKit.
"""
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

import config as cfg

HERE = cfg.HERE
RUNS_DIR = os.path.join(HERE, 'gui_runs')
UPLOAD_DIR = os.path.join(RUNS_DIR, 'uploads')
# set by gui_server.py --demo
DEMO = False

# Study targets the GUI knows.  'trust' follows FROZEN_TRUST_DECISIONS.md: 'trusted' = plain Vina passed calibration; 'rule' = trusted only with the pre-registered interaction rule; 'none' = not
# validated, so docking is not offered.  'dock' holds the options of the frozen docking command; 'exclude' lists same-family target codes left out of the random-seed baseline.
STUDY_TARGETS = [
    {'key': 'malaria_t89515', 'label': 'Malaria - PfDHFR (T89515)', 'target': 'T89515', 'trust': 'trusted',
     'note': 'Docking receptor PfDHFR (1J3I) passed calibration with plain Vina (AUC 0.72).',
     'exclude': ['T83386', 'T17345', 'T19229', 'T35465', 'T52227', 'T02702', 'T47077'],
     'dock': {'extra_receptors': '', 'interaction': '', 'cache': 'dock_cache_v2.jsonl', 'noise_check': None}, 'best': 'malaria_t89515'},
    {'key': 'diabetes', 'label': 'Type-2 diabetes - DPP-4 (T31391)', 'target': 'T31391', 'trust': 'rule',
     'note': 'Docking receptor DPP-4 (2ONC) is validated ONLY with a pharmacophore constraint (basic nitrogen near Glu205/206); plain Vina fails calibration.  Read scores as rule-constrained.',
     'exclude': [],
     'dock': {'extra_receptors': 'extra_dpp4_2onc.json', 'interaction': 'dpp4_amine_glu', 'cache': 'dock_cache_inter.jsonl', 'noise_check': 0}, 'best': 'diabetes'},
    {'key': 'depression', 'label': 'Depression - SERT (T27812)', 'target': 'T27812', 'trust': 'none',
     'note': 'No SERT receptor set-up reached the trust rule (best AUC 0.65-0.68, rule 0.70), so docking is not offered.  Molecules can still be generated and are ranked by similarity to their seed.',
     'exclude': ['T21945', 'T55959'], 'dock': None, 'best': 'depression'},
]
STAGES = ['train', 'generate', 'baseline', 'dock', 'validate']
STAGE_LABELS = {'train': 'Train exemplar model', 'generate': 'Generate molecules', 'baseline': 'Random-seed baseline', 'dock': 'Dock at the validated receptor', 'validate': 'Conditioning check'}
EXEMPLAR_CKPT_GLOB = 'checkpoint_exemplar*.pt'
# Shown next to the optional RL-tuned model in the model list: it was fine-tuned on PfDHFR (T89515) only, so it is an option for that target, not the general model.
TUNED_NOTE = 'PfDHFR-tuned by RL: for Malaria / T89515 only'
# Held-out targets of the Train stage.  The shipped checkpoint_exemplar_seed_v3_holdout.pt held out 17 whole targets: these five were named and the training script adds the other 12 by a
# seeded random draw, so the same five names reproduce the same 17 (checked against the shipped checkpoint on 5 Nov 2026).
HELDOUT_DEFAULT = 'T89515 T31391 T27812 T52522 T45993'
# Base models offered to the Train stage: the shipped one, and any rebuilt one made by pretrain_base_model.py.
PRETRAINED_GLOBS = [cfg.PRETRAINED_CKPT_NAME, 'checkpoint_pretrained_rebuild*.pt']
R3_CKPT_NAME = 'checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt'

# The best-supported setup per study target, as measured in the study (numbers: RESULTS_AND_LIMITS.md).  It is advice shown on the page; "Apply" only fills in the form, and nothing here
# changes what a stage does.  'model' is a checkpoint file name, 'stages' the stages to tick.
BEST_SETUP = {
    'malaria_t89515': {
        'headline': 'Best evidence of any target: the RL-tuned model r3, docked with plain Vina.',
        'model': R3_CKPT_NAME, 'temperature': 0.7, 'n': 70, 'stages': ['generate', 'baseline', 'dock'],
        'docking': 'Plain Vina at PfDHFR (1J3I): validated (AUC 0.72). Dock set: size-corrected percentile among decoys.',
        'evidence': 'After RL, generated analogs dock better than same-seed controls from the general model: size-corrected AUC 0.70 and 0.69 against 0.58 and 0.53 (two seeds), unchanged at a higher search effort. 79-81% of molecules are close analogs of their seed.',
        'caveats': ['r3 was tuned on this receptor\'s own docking reward, so the gain is partly the antifolate pharmacophore that the calibration rewards, not independent proof of binding.',
                    'Validity falls from 71% to 64%; mean ligand efficiency is 0.35 against 0.40 for known binders.',
                    'The general model (the other choice) docks no better than a random-seed baseline here (AUC 0.55 and 0.60 against 0.56 and 0.51).'],
        'seeds': 'Any seed works. Judge the model by pooling several seeds: one run of 50 to 70 molecules gives a docking AUC with a spread of about plus or minus 0.1.'},
    'diabetes': {
        'headline': 'No setup has beaten decoys here. Best available: the general held-out model, docked with the interaction rule.',
        'model': 'checkpoint_exemplar_seed_v3_holdout.pt', 'temperature': 0.7, 'n': 70, 'stages': ['generate', 'baseline', 'dock'],
        'docking': 'Vina plus the interaction rule at DPP-4 (2ONC): a basic amine within 4 A of the Glu205/206 oxygens, else +2 kcal/mol. Validated only with the rule (AUC 0.88; plain Vina 0.63).',
        'evidence': 'Generated molecules score like decoys (size-corrected AUC 0.50, interval 0.40-0.60) but beat their own random-seed baseline (0.63, interval 0.54-0.72), so choosing the target helped.',
        'caveats': ['RL was not attempted at this target (gated RL on the rule is future work).',
                    'Read scores as rule-constrained; plain Vina at DPP-4 is not validated.',
                    'Do not use the PfDHFR-tuned model r3 here.'],
        'seeds': 'Any seed works; pool several seeds to judge.'},
    'depression': {
        'headline': 'Generation only. No docking set-up is validated for SERT, so nothing can be called best for binding.',
        'model': 'checkpoint_exemplar_seed_v3_holdout.pt', 'temperature': 0.7, 'n': 50, 'stages': ['generate'],
        'docking': 'Not offered: no SERT receptor set-up reached the trust rule (best AUC 0.65-0.68, rule 0.70), including the potency filter, Na+/Cl- ions, the 7LWD structure and an Asp98 rule.',
        'evidence': 'Only the conditioning result applies: models like this one write molecules that resemble the known ligands of an unseen target (target lift 1.4 to 1.6). Generated SERT molecules were never docked or tested.',
        'caveats': ['Molecules are ranked by similarity to their seed, which says how faithful an analog is, not how active.',
                    'Do not use the PfDHFR-tuned model r3 here.'],
        'seeds': 'Any seed works.'},
}
# For any other disease or target: no receptor has been validated, so generation with the general held-out model is all that is supported.
GENERAL_BEST = {
    'headline': 'Any other target: generation with the general held-out model. Docking is not offered.',
    'model': 'checkpoint_exemplar_seed_v3_holdout.pt', 'temperature': 0.7, 'n': 50, 'stages': ['generate'],
    'docking': 'Not offered: only PfDHFR and DPP-4 (with its rule) have a validated receptor.',
    'evidence': 'The target\'s known ligands transfer to unseen targets: generated molecules resemble that target\'s ligands more than other targets\' ligands (target lift 1.4 to 1.6, against 1.0 for no conditioning).',
    'caveats': ['The disease label only selects which ligands become seeds (the exact disease-target pair\'s when it has 3 or more, else the target\'s); it does not make the model produce disease-relevant chemistry (disease lift about 1.1).',
                'With few or very different seed ligands most molecules are only loosely related to the seeds.',
                'Do not use the PfDHFR-tuned model r3 for other targets.'],
    'seeds': 'Any seed works.'}

# Rules of thumb for molecules that are unlikely to be a usable lead as written.  They only mark a molecule on the page: nothing is removed or re-ranked.
OUTLIER_RULES = [('large', 'very large (MW above 600)', lambda p: p.get('MW') is not None and p['MW'] > 600),
                 ('hard to make', 'hard to make (synthetic accessibility 5 or more)', lambda p: p.get('SA') is not None and p['SA'] >= 5),
                 ('logP', 'extreme logP (below -2 or above 6)', lambda p: p.get('logP') is not None and (p['logP'] < -2 or p['logP'] > 6))]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# files present on this machine
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# The study target record for a key, or None (a custom request).
def target_by_key(key):
    return next((t for t in STUDY_TARGETS if t['key'] == key), None)


# Stops RDKit printing a warning to the terminal for every SMILES it cannot read (the page already copes with those: no picture, no properties).  Safe to call repeatedly; no-op without RDKit.
def quiet_rdkit():
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        pass


# Exemplar and pretrained checkpoints found in the project folder or its parent, as [{'name', 'path', 'role'}].  The pilot RL checkpoints under rl_pilot/ are deliberately not listed.
def find_checkpoints():
    out, seen = [], set()
    for d in (HERE, os.path.dirname(HERE)):
        for p in sorted(glob.glob(os.path.join(d, EXEMPLAR_CKPT_GLOB))):
            if os.path.basename(p) not in seen:
                seen.add(os.path.basename(p))
                out.append({'name': os.path.basename(p), 'path': p, 'role': 'exemplar', 'note': TUNED_NOTE if 'rl_r3' in os.path.basename(p) else ''})
    out = [c for c in out if 'smoke' not in c['name']]
    first = {cfg.HELDOUT_CKPT_NAME: 0, 'checkpoint_exemplar_seed_v2.pt': 1}
    out.sort(key=lambda c: (first.get(c['name'], 2), c['name']))
    # base models for the Train stage: the shipped one first (the one the package finds), then any rebuilt one; the folder that holds the file wins, as in config.find_file
    pre = cfg.find_file(cfg.PRETRAINED_CKPT_NAME)
    if pre:
        out.append({'name': os.path.basename(pre), 'path': pre, 'role': 'pretrained', 'note': 'shipped base model'})
    for d in (HERE, os.path.dirname(HERE)):
        for g in PRETRAINED_GLOBS[1:]:
            for p in sorted(glob.glob(os.path.join(d, g))):
                if os.path.basename(p) not in seen and 'smoke' not in os.path.basename(p) and not any(c['name'] == os.path.basename(p) for c in out):
                    out.append({'name': os.path.basename(p), 'path': p, 'role': 'pretrained', 'note': 'rebuilt base model'})
    return out


# Files a docking case needs, as [(label, found)].  The receptor and ligand files come from the case's entry in targets.py; the extra receptor file named in the registry is read for its own files.
def dock_requirements(t):
    if t is None or not t.get('dock'):
        return []
    req = []
    try:
        from targets import DISEASE_TARGET_CONFIGS
        import dock_generated as dg
        case = next(c for c in dg.CASES if c['key'] == t['key'])
        cfg_entry = next(c for c in DISEASE_TARGET_CONFIGS if c['name'] == case['v13_config'])
        r = next(x for x in cfg_entry['candidate_receptors'] if x['label'] == case['receptor'])
        req.append((r['receptor_pdbqt_path'], bool(cfg.find_file(r['receptor_pdbqt_path']))))
        if case.get('crystal_ligand'):
            req.append((case['crystal_ligand'], bool(cfg.find_file(case['crystal_ligand']))))
    except Exception as e:
        req.append((f"receptor files (could not be read: {type(e).__name__})", False))
    ex = t['dock'].get('extra_receptors')
    if ex:
        p = cfg.find_file(ex)
        req.append((ex, bool(p)))
        if p:
            try:
                for e in json.load(open(p, encoding='utf-8')):
                    for k in ('pdbqt', 'crystal_ligand'):
                        if e.get(k):
                            req.append((e[k], bool(cfg.find_file(e[k]))))
            except Exception:
                req.append((ex + ' (unreadable)', False))
    return req


# Fixed values the page shows: the Train stage's default held-out targets and the best-supported setups.
def _page_constants():
    return {'holdout_default': HELDOUT_DEFAULT, 'best_setups': BEST_SETUP, 'best_general': GENERAL_BEST}


# Everything the page needs to know about this machine: files, checkpoints, targets with their trust status and docking requirements.
def detect_state():
    if DEMO:
        ck = [{'name': 'checkpoint_exemplar_seed_v3_holdout.pt', 'path': 'demo', 'role': 'exemplar'}, {'name': 'checkpoint_exemplar_seed_v2.pt', 'path': 'demo', 'role': 'exemplar'},
              {'name': R3_CKPT_NAME, 'path': 'demo', 'role': 'exemplar', 'note': TUNED_NOTE},
              {'name': cfg.PRETRAINED_CKPT_NAME, 'path': 'demo', 'role': 'pretrained', 'note': 'shipped base model'}]
        return {'demo': True, 'library': True, 'vina': True, 'checkpoints': ck, 'default_checkpoint': ck[0]['name'], **_page_constants(),
                'targets': [dict(t, requirements=[], ready=(t['trust'] != 'none')) for t in STUDY_TARGETS]}
    cks = find_checkpoints()
    names = [c['name'] for c in cks if c['role'] == 'exemplar']
    default = cfg.HELDOUT_CKPT_NAME if cfg.HELDOUT_CKPT_NAME in names else (names[0] if names else '')
    targets = []
    for t in STUDY_TARGETS:
        req = dock_requirements(t)
        targets.append(dict(t, requirements=[{'file': f, 'found': ok} for f, ok in req], ready=bool(t['dock']) and all(ok for _, ok in req)))
    return {'demo': False, 'library': bool(cfg.find_file(cfg.CSV_NAME)), 'vina': os.path.exists(cfg.DEFAULT_VINA), 'checkpoints': cks, 'default_checkpoint': default, **_page_constants(), 'targets': targets}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# validated parameters and command lines
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Accepts a name or code made of letters, digits, spaces and a few punctuation marks; raises ValueError otherwise.
def clean_text(value, what, max_len=80, allow_empty=False):
    v = (value or '').strip()
    if not v and allow_empty:
        return ''
    if not v or len(v) > max_len or not re.fullmatch(r"[A-Za-z0-9 ,.()'_/+-]+", v):
        raise ValueError(f"{what}: use letters, digits, spaces and , . ( ) ' _ / + - only (at most {max_len} characters)")
    return v


# Accepts a number within [lo, hi]; raises ValueError otherwise.
def clean_number(value, what, lo, hi, integer=False):
    try:
        x = int(value) if integer else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number")
    if not lo <= x <= hi:
        raise ValueError(f"{what} must be between {lo} and {hi}")
    return x


# A checkpoint name from the detected list (never a free path); returns its full path.
def checkpoint_path(name, role='exemplar'):
    if DEMO:
        return name
    for c in find_checkpoints():
        if c['name'] == name and c['role'] == role:
            return c['path']
    raise ValueError(f"checkpoint {name!r} was not found next to the project")


# A path with forward slashes, which the scripts and Windows both accept inside a pattern.
def fwd(p):
    return p.replace('\\', '/')


# The command line of a stage: [python, -u, script, options...].  In DEMO mode the script is this file's stand-in worker.
def command(stage, script, args):
    if DEMO:
        return [sys.executable, '-u', os.path.abspath(__file__), '--demo-worker', stage] + [str(a) for a in args]
    return [sys.executable, '-u', os.path.join(HERE, script)] + [str(a) for a in args]


# Turns the form into (run id, run folder, params, stage list).  Each stage is {'name', 'label', 'argv', 'skip'}; 'skip' is a reason (the stage is not run) or None.  Raises ValueError with a
# message for anything the study rules do not allow, for example docking at a receptor that did not validate.
def plan_pipeline(form):
    t = target_by_key(form.get('target_key'))
    stages = [s for s in STAGES if s in (form.get('stages') or [])]
    if not stages:
        raise ValueError('tick at least one stage')
    n = clean_number(form.get('n', 50), 'number of molecules', 1, 500, integer=True)
    temp = clean_number(form.get('temperature', 0.7), 'temperature', 0.1, 1.5)
    seed = clean_number(form.get('seed', 0), 'seed', 0, 10 ** 6, integer=True)
    extra_lib = form.get('extra_library') or ''
    if t is not None:
        disease, target = '', t['target']
        title = t['label']
    else:
        disease = clean_text(form.get('disease'), 'disease', allow_empty=True)
        target = clean_text(form.get('target'), 'target', allow_empty=True)
        if not disease and not target:
            raise ValueError('give a disease and/or a target')
        title = ' / '.join(x for x in (disease, target) if x)
    if 'dock' in stages:
        if t is None or not t.get('dock'):
            raise ValueError('docking is offered only for study targets whose receptor validated (PfDHFR, and DPP-4 with its rule)')
        if not DEMO:
            missing = [f for f, ok in dock_requirements(t) if not ok]
            if missing:
                raise ValueError('docking files missing: ' + ', '.join(missing))
    if extra_lib and not re.fullmatch(r'[A-Za-z0-9_.-]+\.csv', os.path.basename(extra_lib)):
        raise ValueError('bad library file name')

    resume = form.get('resume_run') or ''
    if resume:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', resume) or not os.path.isdir(os.path.join(RUNS_DIR, resume)):
            raise ValueError('the run to continue was not found')
        run_id = resume
        old = read_run(run_id) or {}
        t = target_by_key(old.get('params', {}).get('target_key')) or t
    else:
        slug = re.sub(r'[^A-Za-z0-9]+', '-', title).strip('-')[:30]
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + slug
    run_dir = os.path.join(RUNS_DIR, run_id)
    if 'dock' in stages and 'generate' not in stages and not os.path.exists(os.path.join(run_dir, 'gen.csv')):
        raise ValueError('docking needs generated molecules: tick Generate, or continue an earlier run that has them')
    gen, base, dres, dsum = (os.path.join(run_dir, f) for f in ('gen.csv', 'gen_baseline.csv', 'dock_results.csv', 'dock_summary.json'))
    ckpt_name = form.get('checkpoint') or ''
    plan = []

    if 'train' in stages:
        steps = clean_number(form.get('train_steps', 6000), 'training steps', 100, 100000, integer=True)
        out_name = re.sub(r'[^A-Za-z0-9_-]+', '_', form.get('train_out') or 'gui_trained')[:40]
        out_ckpt = os.path.join(HERE, f'checkpoint_exemplar_{out_name}.pt')
        holdout = [clean_text(x, 'held-out target', 20) for x in (form.get('holdout') or HELDOUT_DEFAULT).replace(',', ' ').split()]
        pre = checkpoint_path(form.get('pretrained') or cfg.PRETRAINED_CKPT_NAME, 'pretrained')
        skip = f"{os.path.basename(out_ckpt)} already exists" if os.path.exists(out_ckpt) and not DEMO else None
        plan.append({'name': 'train', 'label': STAGE_LABELS['train'], 'skip': skip,
                     'argv': command('train', 'train_exemplar_generator.py', ['--seed-mode', '--eval-k', 1, '--steps', steps, '--checkpoint', pre, '--out', out_ckpt, '--holdout-targets'] + holdout)})
    # the stages after training use the newly trained model; without a training stage they use the chosen checkpoint
    model = out_ckpt if 'train' in stages else (checkpoint_path(ckpt_name) if set(stages) & {'generate', 'baseline', 'validate'} else '')

    gargs = ['--checkpoint', model, '--n', n, '--temperature', temp, '--seed', seed]
    if extra_lib:
        gargs += ['--extra-library', os.path.join(UPLOAD_DIR, os.path.basename(extra_lib))]
    if 'generate' in stages:
        a = list(gargs)
        if disease:
            a += ['--disease', disease]
        if target:
            a += ['--target', target]
        skip = 'gen.csv already exists in this run' if os.path.exists(gen) and not DEMO and resume else None
        plan.append({'name': 'generate', 'label': STAGE_LABELS['generate'], 'skip': skip, 'argv': command('generate', 'generate_for_request.py', a + ['--out', gen])})
    if 'baseline' in stages:
        if t is None:
            raise ValueError('the random-seed baseline is defined for the study targets only')
        a = ['--checkpoint', model, '--n', n, '--temperature', temp, '--seed', seed, '--random-seeds', '--target', t['target']]
        if t['exclude']:
            a += ['--exclude-seed-targets', ','.join(t['exclude'])]
        plan.append({'name': 'baseline', 'label': STAGE_LABELS['baseline'], 'skip': None, 'argv': command('baseline', 'generate_for_request.py', a + ['--out', base])})
    if 'dock' in stages:
        d = t['dock']
        a = ['--cases', t['key'], '--n', n, '--gen-pattern', fwd(gen), '--cache', d['cache'], '--out', dres, '--summary', dsum]
        if d['extra_receptors']:
            a += ['--extra-receptors', d['extra_receptors']]
        if d['interaction']:
            a += ['--interaction', d['interaction']]
        if d['noise_check'] is not None:
            a += ['--noise-check', d['noise_check']]
        if 'baseline' in stages or os.path.exists(base):
            a += ['--baseline', '--baseline-pattern', fwd(base)]
        plan.append({'name': 'dock', 'label': STAGE_LABELS['dock'], 'skip': None, 'argv': command('dock', 'dock_generated.py', a)})
    if 'validate' in stages:
        plan.append({'name': 'validate', 'label': STAGE_LABELS['validate'], 'skip': None,
                     'argv': command('validate', 'check_exemplar_conditioning.py', ['--checkpoint', model, '--temperature', 0.7, '--max-targets', 17, '--k', 1])})
    params = {'target_key': form.get('target_key'), 'disease': disease, 'target': target, 'n': n, 'temperature': temp, 'seed': seed, 'checkpoint': os.path.basename(model),
              'extra_library': os.path.basename(extra_lib) if extra_lib else '', 'stages': stages, 'title': title}
    return run_id, run_dir, params, plan


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the job runner: one job at a time, stages run in order, a log the page can poll
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Reads gui_runs/<id>/run.json, or None.
def read_run(run_id):
    try:
        with open(os.path.join(RUNS_DIR, run_id, 'run.json'), encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


# Runs one job in a background thread.  Status values: running, done, failed, cancelled.  Stage values: waiting, running, done, skipped, failed, cancelled, not run (an earlier stage failed).
class Job:

    # Keeps the plan and prepares the run folder, log and the shared state the page polls.
    def __init__(self, run_id, run_dir, params, plan):
        self.id, self.dir, self.params, self.plan = run_id, run_dir, params, plan
        self.stages = [{'name': s['name'], 'label': s['label'], 'status': 'skipped' if s['skip'] else 'waiting', 'note': s['skip'] or '', 'progress': None, 'last': ''} for s in plan]
        self.status, self.lines, self.proc, self.cancelled = 'running', [], None, False
        self.lock = threading.Lock()
        self.t0 = time.time()
        os.makedirs(run_dir, exist_ok=True)
        self.log_file = open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8')

    # Appends a line to the in-memory log and the log file.
    def log(self, text):
        with self.lock:
            self.lines.append(text)
            self.log_file.write(text + '\n')
            self.log_file.flush()

    # Writes run.json (parameters, stage states, status) so the Runs tab can list the run later.
    def save(self):
        rec = {'id': self.id, 'params': self.params, 'status': self.status, 'started': self.t0, 'ended': getattr(self, 't1', None), 'stages': self.stages, 'demo': DEMO}
        with open(os.path.join(self.dir, 'run.json'), 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1)

    # Copy of the state for the page: status, stages and the log lines from index `since` on.
    def snapshot(self, since=0):
        with self.lock:
            return {'id': self.id, 'status': self.status, 'stages': self.stages, 'params': self.params, 'log': self.lines[since:], 'next': len(self.lines), 'elapsed': round(time.time() - self.t0)}

    # Starts the worker thread.
    def start(self):
        self.save()
        threading.Thread(target=self._run, daemon=True).start()

    # After a docking stage: says plainly that the job itself finished and that a PASS/FAIL verdict printed by the docking script is about the molecule set, not about the job; warns when the set is too small.
    def log_dock_note(self):
        try:
            n = sum(1 for r in read_csv(os.path.join(self.dir, 'dock_results.csv')) if (r.get('role') or '').startswith('generated_own'))
        except Exception:
            return
        self.log(f"[GUI note] the docking stage finished normally.  Any PASS / FAIL / PARTIAL verdict above judges the generated SET against the decoys, it is not a job error.  {n} generated molecule(s) were docked"
                 + ('; that is too few for the verdict to mean anything (use 50 or more).' if n < 15 else '.'))

    # Runs the stages in order; a failed stage stops the job.
    def _run(self):
        ok = True
        for i, s in enumerate(self.plan):
            st = self.stages[i]
            if st['status'] == 'skipped':
                self.log(f"--- {s['label']}: skipped ({st['note']})")
                continue
            if self.cancelled:
                st['status'] = 'cancelled'
                continue
            st['status'] = 'running'
            self.log(f"--- {s['label']}")
            self.log('$ python ' + ' '.join([os.path.basename(s['argv'][2])] + s['argv'][3:]))
            self.save()
            code = self._exec(s['argv'], st)
            st['status'] = 'cancelled' if self.cancelled else ('done' if code == 0 else 'failed')
            st['progress'] = 1.0 if code == 0 else st['progress']
            self.save()
            if code == 0 and s['name'] == 'dock':
                self.log_dock_note()
            if code != 0:
                ok = False
                self.log(f"--- {s['label']} {'cancelled' if self.cancelled else 'FAILED (exit code %s)' % code}")
                break
        for st in self.stages:
            if st['status'] == 'waiting':
                st['status'] = 'cancelled' if self.cancelled else 'not run'
        self.status = 'cancelled' if self.cancelled else ('done' if ok else 'failed')
        self.t1 = time.time()
        self.log(f"=== {self.status}")
        self.save()
        self.log_file.close()

    # Starts one script, streams its output into the log and updates the stage's progress from lines such as '[12/118]' or 'step 40/60'.
    def _exec(self, argv, st):
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        kw = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        try:
            self.proc = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env, **kw)
        except OSError as e:
            self.log(f"could not start: {e}")
            return 1
        for line in self.proc.stdout:
            line = line.rstrip('\r\n')
            self.log(line)
            m = re.search(r'\[(\d+)/(\d+)\]|(?:step|round)\s+(\d+)/(\d+)|(\d+)/(\d+) kept', line)
            if m:
                a, b = [int(x) for x in m.groups() if x is not None][:2]
                if b > 0:
                    st['progress'] = min(a / b, 0.99)
            if line.strip():
                st['last'] = line.strip()[:140]
        return self.proc.wait()

    # Stops the running script and everything it started (Vina included).
    def cancel(self):
        self.cancelled = True
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.kill()


_CURRENT = {'job': None}


# Starts a job for the form, or raises ValueError (bad input) / RuntimeError (another job is running).
def start_job(form):
    cur = _CURRENT['job']
    if cur is not None and cur.status == 'running':
        raise RuntimeError('a job is already running: wait for it or cancel it')
    run_id, run_dir, params, plan = plan_pipeline(form)
    job = Job(run_id, run_dir, params, plan)
    _CURRENT['job'] = job
    job.start()
    return job


# The current (or last) job, or None.
def current_job():
    return _CURRENT['job']


# All runs, newest first, as the small records of run.json.
def list_runs():
    out = []
    for p in glob.glob(os.path.join(RUNS_DIR, '*', 'run.json')):
        try:
            r = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        d = os.path.dirname(p)
        r['has_gen'] = os.path.exists(os.path.join(d, 'gen.csv'))
        r['has_dock'] = os.path.exists(os.path.join(d, 'dock_results.csv'))
        r.pop('stages', None)
        out.append(r)
    return sorted(out, key=lambda r: -r.get('started', 0))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# results: ranked candidates, properties, pictures
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Rows of a CSV file as dicts ([] if it is missing).
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


# Float of a CSV cell, or None.
def num(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


# The outlier marks of a molecule's properties, as [{'code', 'text'}] (empty for a normal molecule or when the properties are unknown).
def outlier_flags(p):
    return [{'code': c, 'text': t} for c, t, test in OUTLIER_RULES if p and test(p)]


# Canonical SMILES through RDKit when it is installed (so the same molecule written two ways matches), else the string itself.
def canonical(smiles):
    quiet_rdkit()
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m) if m is not None else smiles
    except ImportError:
        return smiles


# The synthetic-accessibility scorer if RDKit's Contrib copy is available, else None.
def _sa_scorer():
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        return sascorer.calculateScore
    except Exception:
        return None


# Molecular properties of a SMILES (weight, logP, H-bond donors and acceptors, polar surface area, rotatable bonds, heavy atoms, synthetic accessibility); {} if RDKit is not installed or the SMILES does not parse.
def properties(smiles):
    quiet_rdkit()
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError:
        return {}
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    sa = _sa_scorer()
    p = {'MW': round(Descriptors.MolWt(m), 1), 'logP': round(Crippen.MolLogP(m), 2), 'HBD': Lipinski.NumHDonors(m), 'HBA': Lipinski.NumHAcceptors(m),
         'TPSA': round(rdMolDescriptors.CalcTPSA(m), 1), 'RotB': rdMolDescriptors.CalcNumRotatableBonds(m), 'HeavyAtoms': m.GetNumHeavyAtoms()}
    if sa:
        try:
            p['SA'] = round(sa(m), 2)
        except Exception:
            pass
    return p


# A 2D drawing of a SMILES as SVG text; a plain placeholder if RDKit is missing.
def svg(smiles, w=260, h=190):
    quiet_rdkit()
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="none" stroke="#999"/><text x="50%" y="50%" text-anchor="middle" font-size="12" fill="#999">RDKit not installed</text></svg>'
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><text x="10" y="20" font-size="12">invalid SMILES</text></svg>'
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = d.drawOptions()
    opts.clearBackground = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText()


# A 3D conformer for the viewer: embeds the molecule with explicit hydrogens, relaxes it with MMFF (UFF if MMFF has no parameters) and returns {'atoms': [[element, x, y, z]], 'bonds': [[i, j, order]]}
# without the hydrogens.  Raises ValueError with a message if RDKit is missing or no conformer can be made.
def molecule_3d(smiles):
    quiet_rdkit()
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ValueError('RDKit is not installed')
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('invalid SMILES')
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
        raise ValueError('no 3D conformer could be generated')
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    except Exception:
        AllChem.UFFOptimizeMolecule(m, maxIters=2000)
    m = Chem.RemoveHs(m)
    conf = m.GetConformer()
    atoms = [[a.GetSymbol()] + [round(c, 3) for c in conf.GetAtomPosition(a.GetIdx())] for a in m.GetAtoms()]
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 1.5 if b.GetIsAromatic() else b.GetBondTypeAsDouble()] for b in m.GetBonds()]
    return {'atoms': atoms, 'bonds': bonds}


# The candidates of a run, ranked.  With docking results they are ordered by the size-corrected percentile among the decoys (higher = looks more like a known binder of that receptor); without
# them by similarity to the seed (closest analogs first), which says how faithful an analog is, not how active.  Returns {'basis', 'receptor', 'n', 'mean_percentile', 'rows'}.
def candidates(run_id, limit=60):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('bad run id')
    d = os.path.join(RUNS_DIR, run_id)
    gen = read_csv(os.path.join(d, 'gen.csv'))
    drows = [r for r in read_csv(os.path.join(d, 'dock_results.csv')) if (r.get('role') or '').startswith('generated_own')]
    run = read_run(run_id) or {}
    t = target_by_key(run.get('params', {}).get('target_key'))
    # dock_generated.py writes its own CLEANED form of each molecule (largest fragment, normalised), which can differ from the text in gen.csv, so a plain
    # canonical-SMILES match can lose a molecule.  Match in four steps: exact text, canonical SMILES, largest fragment (salt removed), then the (seed, similarity-to-seed) pair, which both files carry unchanged.
    by_text = {r['smiles']: r for r in drows}
    by_canon = {canonical(r['smiles']): r for r in drows}
    main_part = lambda smi: canonical(max((smi or '').split('.'), key=len))     # the generator can emit a salt ('...nc12.Cl'); docking keeps only the largest piece
    by_main = {main_part(r['smiles']): r for r in drows}
    pair = lambda seed, sim: (canonical(seed) if seed else '', None if num(sim) is None else round(num(sim), 4))
    by_pair = {pair(r.get('seed', ''), r.get('similarity_to_seed')): r for r in drows}
    rows, used = [], set()
    for g in gen:
        r = {'smiles': g['smiles'], 'seed': g.get('seed_smiles', ''), 'similarity_to_seed': num(g.get('similarity_to_seed')), 'docked': False,
             'nearest_library': num(g.get('similarity_to_nearest_library_molecule')), 'score': None, 'percentile': None, 'seed_name': ''}
        x = by_text.get(g['smiles']) or by_canon.get(canonical(g['smiles'])) or by_main.get(main_part(g['smiles'])) or by_pair.get(pair(g.get('seed_smiles', ''), g.get('similarity_to_seed')))
        if x is not None and id(x) not in used:
            used.add(id(x))
            r.update(score=num(x.get('score')), percentile=num(x.get('percentile')), seed_name=x.get('seed_name', ''), docked=True)
        rows.append(r)
    basis = 'docking' if any(r['docked'] for r in rows) else 'similarity'
    n_undocked = 0
    if basis == 'docking':
        # Docked molecules first, ranked by percentile; molecules that were not docked stay in the list (marked) instead of vanishing.
        docked = sorted([r for r in rows if r['docked']], key=lambda r: (-(r['percentile'] if r['percentile'] is not None else -1), -(r['similarity_to_seed'] or 0)))
        undocked = sorted([r for r in rows if not r['docked']], key=lambda r: -(r['similarity_to_seed'] or 0))
        n_undocked = len(undocked)
        rows = docked + undocked
    else:
        rows.sort(key=lambda r: -(r['similarity_to_seed'] or 0))
    rows = rows[:limit]
    for i, r in enumerate(rows):
        r['rank'] = i + 1
        r['properties'] = properties(r['smiles'])
        r['flags'] = outlier_flags(r['properties'])
    pcts = [r['percentile'] for r in rows if r['percentile'] is not None]
    return {'basis': basis, 'receptor': (t or {}).get('label'), 'trust': (t or {}).get('trust'), 'trust_note': (t or {}).get('note'), 'n': len(rows), 'n_docked': len(pcts), 'n_undocked': n_undocked, 'n_flagged': sum(1 for r in rows if r['flags']),
            'mean_percentile': round(sum(pcts) / len(pcts), 3) if pcts else None, 'rows': rows, 'title': (run.get('params') or {}).get('title', run_id)}


# Writes a ligand library CSV in the project's format (columns Disease Entry, Targets for Drug, Canonical Smiles) from pasted SMILES lines ('SMILES' or 'SMILES name', one per line) and returns
# its file name inside gui_runs/uploads/ with the number of ligands.  Lines that RDKit cannot parse are reported, not silently dropped.
def save_library(disease, target, text):
    disease, target = clean_text(disease, 'disease'), clean_text(target, 'target')
    smiles, bad = [], []
    try:
        from rdkit import Chem
    except ImportError:
        Chem = None
    for line in (text or '').splitlines():
        s = line.strip().split()[0] if line.strip() else ''
        if not s:
            continue
        if Chem is not None and Chem.MolFromSmiles(s) is None:
            bad.append(s)
        else:
            smiles.append(s)
    if not smiles:
        raise ValueError('no valid SMILES found' + (f' ({len(bad)} line(s) did not parse)' if bad else ''))
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = time.strftime('%Y%m%d-%H%M%S') + '-' + re.sub(r'[^A-Za-z0-9]+', '-', target)[:20] + '.csv'
    with open(os.path.join(UPLOAD_DIR, name), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['Disease Entry', 'Targets for Drug', 'Canonical Smiles'])
        for s in smiles:
            w.writerow([f"['{disease}']", f"['{target}']", s])
    return {'file': name, 'n': len(smiles), 'bad': bad[:10], 'n_bad': len(bad), 'rdkit_checked': Chem is not None}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEMO worker: stands in for every script
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DEMO_SMILES = ['Nc1nc(N)c2nc(-c3ccccc3)c(-c3ccccc3)nc2n1', 'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC', 'CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1', 'Nc1nc(N)c2cc(CNc3ccccc3)ccc2n1', 'Cc1ccc2nc(N)nc(N)c2c1',
               'Nc1nc2ncc(CNc3ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc3)nc2c(N)n1', 'Nc1nc(N)c2c(Sc3ccccc3)cccc2n1', 'CC1=NC(N)=NC(N)=C1Cc1ccccc1', 'COc1ccc(Cc2cnc(N)nc2N)cc1OC',
               'Nc1nc(N)c2cc(Cc3ccccc3)ccc2n1', 'CCc1nc(N)nc(N)c1-c1ccc(OC)cc1', 'Nc1ncnc2[nH]cc(-c3ccccc3)c12']


# The stand-in worker: prints stage-like progress and writes made-up files.  Everything it makes is labelled DEMO in the log.
def demo_worker(argv):
    import random
    stage, args = argv[0], argv[1:]
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    rnd = random.Random(int(opt('--seed', 0) or 0))
    print(f"[DEMO] stand-in for the {stage} stage: nothing real is computed")
    if stage in ('generate', 'baseline'):
        n = min(int(opt('--n', 12)), len(DEMO_SMILES))
        for r in range(1, 4):
            time.sleep(0.4)
            print(f"  round {r}: {min(n, r * n // 3 + 1)}/{n} kept")
        with open(opt('--out'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['smiles', 'seed_smiles', 'seed_source', 'similarity_to_seed', 'similarity_to_nearest_library_molecule'])
            for s in rnd.sample(DEMO_SMILES, n):
                w.writerow([s, rnd.choice(DEMO_SMILES), 'target', round(rnd.uniform(0.2, 0.8), 3), round(rnd.uniform(0.3, 0.8), 3)])
        print(f"[done] {n} made-up molecules written to {opt('--out')}")
    elif stage == 'dock':
        gen = read_csv(opt('--gen-pattern'))
        total = len(gen) + 6
        for i in range(1, total + 1):
            time.sleep(0.15)
            if i % 3 == 0 or i == total:
                print(f"      [{i}/{total}] DEMO  {rnd.uniform(-9.5, -6):.2f}")
        cols = ['case', 'receptor', 'role', 'name', 'smiles', 'heavy_atoms', 'score', 'size_corrected', 'percentile', 'seed', 'seed_name', 'similarity_to_seed']
        with open(opt('--out'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for g in gen:
                w.writerow({'case': opt('--cases'), 'receptor': 'DEMO', 'role': 'generated_own', 'smiles': g['smiles'], 'heavy_atoms': 24, 'score': round(rnd.uniform(-9.5, -6), 2),
                            'size_corrected': round(rnd.uniform(-1, 1), 2), 'percentile': round(rnd.uniform(0.2, 0.95), 3), 'seed': g['seed_smiles'], 'seed_name': 'demo seed',
                            'similarity_to_seed': g['similarity_to_seed']})
        with open(opt('--summary'), 'w', encoding='utf-8') as fh:
            json.dump({'demo': True}, fh)
        print("[DEMO] made-up docking results written")
    elif stage == 'train':
        total = 5
        for i in range(1, total + 1):
            time.sleep(0.4)
            print(f"step {i * 1000}/{total * 1000}  loss {3 - i * 0.3:.2f}")
        print("[DEMO] no checkpoint was written")
    else:
        for line in ('PART A  target conditioning (made-up)', '  target lift 1.43   disease lift 1.11   validity 71%', 'PART B  ...'):
            time.sleep(0.3)
            print(line)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--demo-worker':
        sys.exit(demo_worker(sys.argv[2:]))
    print(__doc__)
