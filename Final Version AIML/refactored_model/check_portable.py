#!/usr/bin/env python3
"""
check_portable.py -- is this copy of the project complete, and can this machine run it?  Run it on the new machine after extracting the portable zip, and on the old machine before sending it.

WHAT IT CHECKS
  1. the code files and the GUI page files that must sit in this folder
  2. every file listed in portable_manifest.json (written by make_portable_zip.py): present, and the same size (--hashes also compares the SHA-256 of each file)
  3. the shared inputs the scripts look up through config.py: library CSV, vina.exe, the exemplar checkpoints, the pretrained base model
  4. python packages (numpy, pandas, torch, rdkit, meeko, tqdm, and the optional SA scorer) and whether torch can see a GPU
  5. that vina.exe starts (Windows only)
  --deep    also runs 'gui_server.py --check' (receptor files per target, the commands the page would run)
  --smoke   also generates 3 molecules with the held-out checkpoint (needs torch and RDKit; about a minute) to prove the model loads and writes valid molecules

USAGE  (run from the project folder, the one that holds config.py)
  python check_portable.py
  python check_portable.py --hashes --deep
  python check_portable.py --smoke

It reads files and starts nothing except 'vina.exe --version', gui_server.py --check and the smoke test when asked.  It writes nothing except the smoke test's temporary file.
Exit code 0 = nothing required is missing; 1 = at least one required item is missing.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import config as cfg

MANIFEST_NAME = 'portable_manifest.json'

# Code files that must sit in the project folder (the 19 core files, the two GUI files and the two packaging tools).
CODE_FILES = ['config.py', 'targets.py', 'base_data.py', 'base_model.py', 'base_train.py', 'docking.py', 'rl_utils.py', 'exemplar_data.py', 'exemplar_model.py',
              'train_exemplar_generator.py', 'generate_for_request.py', 'check_exemplar_conditioning.py', 'dock_generated.py', 'prep_receptors.py', 'fetch_potency.py',
              'rule_gen_check.py', 'check_refactor.py', 'test_hybrid_pairs.py', 'test_new_context.py', 'gui_server.py', 'gui_backend.py', 'check_portable.py', 'make_portable_zip.py']
GUI_PAGE_FILES = ['index.html', 'style.css', 'app.js']

# (import name, pip name, required).  A missing package stops the scripts that use it.
PACKAGES = [('numpy', 'numpy', True), ('pandas', 'pandas', True), ('torch', 'torch', True), ('rdkit', 'rdkit', True), ('meeko', 'meeko', True), ('tqdm', 'tqdm', True)]


# Size in bytes as a short string.
def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024.0


# SHA-256 of a file, read in 8 MB pieces.
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Collects the lines and the problem count, so every check prints one line the same way.
class Report:
    def __init__(self):
        self.problems = []

    # ok -> OK; not ok and warn -> WARN (does not count); not ok -> MISSING (counts as a problem).
    def line(self, ok, what, detail='', warn=False):
        tag = 'OK     ' if ok else ('WARN   ' if warn else 'MISSING')
        print(f"  [{tag}] {what}" + (f"   {detail}" if detail else ''))
        if not ok and not warn:
            self.problems.append(what)


# Reads portable_manifest.json from the project folder; None if there is none or it cannot be read.
def read_manifest():
    p = os.path.join(cfg.HERE, MANIFEST_NAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# Where a manifest path lives on THIS machine: paths under the project folder map to this folder (even if it was renamed), the others sit next to it, in its parent.
def local_path(manifest, rel):
    proj = manifest.get('project_folder', os.path.basename(cfg.HERE))
    if rel.startswith(proj + '/'):
        return os.path.join(cfg.HERE, *rel[len(proj) + 1:].split('/'))
    return os.path.join(os.path.dirname(cfg.HERE), *rel.split('/'))


# Names of the receptor, structure and extra-receptor files the code looks up (from targets.py, the GUI's study targets and the extra-receptor json files they name).
# Returns (names, notes); a name is looked up with config.find_file like everywhere else.
def referenced_names():
    names, notes = set(), []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.lower().endswith(('.pdbqt', '.pdb', '.json')) and not os.path.isabs(o):
            names.add(o)
    try:
        import targets
        walk(targets.DISEASE_TARGET_CONFIGS)
    except Exception as e:
        notes.append(f"targets.py could not be read ({type(e).__name__})")
    try:
        import gui_backend as gb
        walk([t.get('dock') for t in gb.STUDY_TARGETS])
    except Exception as e:
        notes.append(f"gui_backend.py could not be read ({type(e).__name__})")
    try:
        import dock_generated as dg
        walk(dg.CASES)
    except Exception as e:
        notes.append(f"dock_generated.py could not be imported ({type(e).__name__}); its receptor and ligand names were not listed")
    for n in sorted(n for n in names if n.lower().endswith('.json')):
        path = cfg.find_file(n)
        if path:
            try:
                with open(path, encoding='utf-8') as fh:
                    walk(json.load(fh))
            except (OSError, ValueError):
                notes.append(f"{n} could not be read")
    names = {n for n in names if n.lower().endswith(('.pdbqt', '.pdb', '.json'))}
    return sorted(names), notes


# True when `path` lies inside the package root (the project folder and its parent folder, the two places the zip recreates).
def inside_package(path):
    root = os.path.dirname(cfg.HERE)
    try:
        return os.path.commonpath([os.path.abspath(path), root]) == root
    except ValueError:
        return False


# 1. code files and page files.
def check_code(rep):
    print("\ncode files in this folder")
    gone = [f for f in CODE_FILES if not os.path.isfile(os.path.join(cfg.HERE, f))]
    rep.line(not gone, f"{len(CODE_FILES) - len(gone)} of {len(CODE_FILES)} code files", ('missing: ' + ', '.join(gone)) if gone else '')
    page = os.path.join(cfg.HERE, 'gui_static')
    gone = [f for f in GUI_PAGE_FILES if not os.path.isfile(os.path.join(page, f))]
    rep.line(not gone, 'gui_static page files', ('missing: ' + ', '.join(gone)) if gone else 'index.html, style.css, app.js')


# 2. every file in the manifest: present, same size, optionally the same SHA-256.
def check_manifest(rep, hashes):
    print("\nfiles listed in the manifest")
    m = read_manifest()
    if m is None:
        rep.line(False, MANIFEST_NAME, 'not found: the file-by-file check is skipped (make_portable_zip.py writes it into the zip)', warn=True)
        return
    files = m.get('files', [])
    missing, wrong_size, wrong_hash, total = [], [], [], 0
    for i, f in enumerate(files, 1):
        p = local_path(m, f['path'])
        total += f['size']
        if not os.path.isfile(p):
            missing.append(f['path'])
        elif os.path.getsize(p) != f['size']:
            wrong_size.append(f['path'])
        elif hashes:
            if sha256_of(p) != f['sha256']:
                wrong_hash.append(f['path'])
            if i % 10 == 0:
                print(f"      hashed {i}/{len(files)}", end='\r')
    if hashes:
        print(' ' * 40, end='\r')
    rep.line(True, f"manifest of {len(files)} files, {human(total)}", f"built {m.get('built', '?')}, RL files: {m.get('rl', '?')}")
    rep.line(not missing, 'every listed file is present', ('%d missing, e.g. %s' % (len(missing), ', '.join(missing[:6]))) if missing else '')
    rep.line(not wrong_size, 'every listed file has its original size', ('%d differ, e.g. %s' % (len(wrong_size), ', '.join(wrong_size[:6]))) if wrong_size else '')
    if hashes:
        rep.line(not wrong_hash, 'every listed file matches its SHA-256', ('%d differ, e.g. %s' % (len(wrong_hash), ', '.join(wrong_hash[:6]))) if wrong_hash else '')
    else:
        print("      (sizes only; add --hashes to compare SHA-256 of every file, slow for large checkpoints)")


# 3. the shared inputs and the receptor files, found the way the scripts find them (config.find_file).  A file found OUTSIDE the package folder works on this machine but would be missing on
# another one: with a manifest present that is a problem unless the manifest lists it.
def check_inputs(rep):
    print("\nshared inputs (found the way the scripts find them)")
    m = read_manifest()
    listed = {os.path.basename(f['path']) for f in m.get('files', [])} if m else None

    def judge(label, name, req):
        p = cfg.find_file(name)
        if not p:
            rep.line(False, f"{label}: {name}", 'not found' + ('' if req else ' (only matters if that target or the training stage is used)'), warn=not req)
        elif not inside_package(p) and listed is not None and name not in listed:
            rep.line(False, f"{label}: {name}", f"found at {p}, OUTSIDE this package and not in the manifest: another machine will not have it")
        else:
            rep.line(True, f"{label}: {name}", p)
    for label, name, req in (('library CSV', cfg.CSV_NAME, True), ('vina executable', cfg.VINA_NAME, True), ('held-out exemplar model', cfg.HELDOUT_CKPT_NAME, True),
                             ('all-targets exemplar model', cfg.EXEMPLAR_CKPT_NAME, True), ('pretrained base model', cfg.PRETRAINED_CKPT_NAME, False)):
        judge(label, name, req)
    names, notes = referenced_names()
    for n in names:
        judge('receptor/structure file', n, False)
    for n in notes:
        rep.line(False, 'receptor list', n, warn=True)
    tuned = [f for f in os.listdir(cfg.HERE) if f.startswith('checkpoint_exemplar') and f.endswith('.pt')]
    print(f"      exemplar checkpoints in this folder: {len(tuned)}" + (' (' + ', '.join(sorted(tuned)[:5]) + (', ...' if len(tuned) > 5 else '') + ')' if tuned else ''))


# 4. python packages and the GPU.
def check_packages(rep):
    print(f"\npython {sys.version.split()[0]} and packages (this check installs nothing)")
    for mod, pip_name, req in PACKAGES:
        found = importlib.util.find_spec(mod) is not None
        ver = ''
        if found:
            try:
                from importlib.metadata import version
                ver = version(pip_name)
            except Exception:
                ver = ''
        rep.line(found, f"{pip_name}", ver or ('pip install ' + pip_name if not found else ''), warn=not req)
    # the GUI's 'synthetic accessibility' column uses the SA scorer that ships inside RDKit (RDKit's Contrib folder); without it only that column is missing
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        rep.line(callable(sascorer.calculateScore), 'SA scorer (RDKit Contrib)', 'for the GUI synthetic-accessibility column', warn=True)
    except Exception:
        rep.line(False, 'SA scorer (RDKit Contrib)', 'not available: the GUI shows no synthetic-accessibility value, everything else works', warn=True)
    if importlib.util.find_spec('torch') is not None:
        try:
            import torch
            if torch.cuda.is_available():
                rep.line(True, 'GPU for torch', torch.cuda.get_device_name(0))
            else:
                rep.line(False, 'GPU for torch', 'none found: generating still works on the CPU (slowly); training needs a GPU', warn=True)
        except Exception as e:
            rep.line(False, 'torch could not be loaded', f"{type(e).__name__}: {str(e)[:100]}")


# 5. vina.exe starts (a Windows program: on another system it cannot run, which only matters for docking).
def check_vina(rep):
    print("\nvina")
    p = cfg.find_file(cfg.VINA_NAME)
    if not p:
        rep.line(False, 'vina.exe not found', '', warn=True)
        return
    if os.name != 'nt' and p.lower().endswith('.exe'):
        rep.line(False, 'vina.exe cannot run on this system', 'it is a Windows program: docking needs Windows or a native AutoDock Vina 1.2.7 binary', warn=True)
        return
    try:
        out = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=30)
        first = (out.stdout or out.stderr or '').strip().splitlines()[:1]
        rep.line(out.returncode == 0 or 'Vina' in (out.stdout + out.stderr), 'vina.exe starts', first[0] if first else '')
    except Exception as e:
        rep.line(False, 'vina.exe did not start', f"{type(e).__name__}: {str(e)[:100]}")


# --deep: the GUI's own self-check (receptor files per target and the commands it would run); its output is shown as it is.
def run_deep(rep):
    print("\ngui_server.py --check")
    r = subprocess.run([sys.executable, os.path.join(cfg.HERE, 'gui_server.py'), '--check'], cwd=cfg.HERE, capture_output=True, text=True)
    for ln in (r.stdout + r.stderr).splitlines():
        print('   | ' + ln)
    rep.line(r.returncode == 0, 'gui_server.py --check reports nothing missing', '' if r.returncode == 0 else 'see the list above')


# --smoke: three molecules from the held-out checkpoint, written to a temporary file; proves torch, the model file, the library and RDKit work together.
def run_smoke(rep):
    print("\nsmoke test: generating 3 molecules (about a minute)")
    ckpt = cfg.find_file(cfg.HELDOUT_CKPT_NAME)
    if not ckpt:
        rep.line(False, 'smoke test', 'the held-out checkpoint is missing')
        return
    out = os.path.join(tempfile.mkdtemp(prefix='portable_smoke_'), 'smoke.csv')
    r = subprocess.run([sys.executable, os.path.join(cfg.HERE, 'generate_for_request.py'), '--checkpoint', ckpt, '--n', '3', '--temperature', '0.7', '--seed', '0', '--target', 'T89515',
                        '--out', out], cwd=cfg.HERE, capture_output=True, text=True)
    n = 0
    if os.path.isfile(out):
        with open(out, encoding='utf-8') as fh:
            n = max(0, sum(1 for _ in fh) - 1)
    rep.line(r.returncode == 0 and n >= 1, f"generate_for_request.py wrote {n} molecule(s)", '' if r.returncode == 0 else (r.stderr.strip().splitlines() or ['failed'])[-1][:160])


# Reads the options, runs the checks and prints the verdict.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hashes', action='store_true', help='compare the SHA-256 of every file in the manifest (slow for large checkpoints)')
    ap.add_argument('--deep', action='store_true', help='also run gui_server.py --check')
    ap.add_argument('--smoke', action='store_true', help='also generate 3 molecules with the held-out checkpoint')
    a = ap.parse_args()
    print(f"Portable package check   (folder {cfg.HERE})")
    rep = Report()
    check_code(rep)
    check_manifest(rep, a.hashes)
    check_inputs(rep)
    check_packages(rep)
    check_vina(rep)
    if a.deep:
        run_deep(rep)
    if a.smoke:
        run_smoke(rep)
    print('\n' + ('ALL REQUIRED ITEMS PRESENT' if not rep.problems else str(len(rep.problems)) + ' PROBLEM(S): ' + '; '.join(rep.problems)))
    sys.exit(1 if rep.problems else 0)


if __name__ == '__main__':
    main()
