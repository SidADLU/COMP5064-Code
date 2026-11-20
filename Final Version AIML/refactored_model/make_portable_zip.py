#!/usr/bin/env python3
"""
make_portable_zip.py -- packs the project into ONE zip that runs on another machine that has the python packages installed (pip install -r requirements.txt) and nothing else.

WHAT GOES IN (the default, 'minimal': only what the code itself uses)
  the code files and the GUI page; CleanedData.csv, vina.exe and the core models (pretrained base, all-targets, held-out) wherever config.find_file finds them; the receptor, structure and
  extra-receptor files named in targets.py, the GUI and dock_generated; the docking caches the GUI uses; the notes (FROZEN_TRUST_DECISIONS.md, README files); the optional PfDHFR-tuned r3 model
  and the RL pilot record (see --rl); portable_manifest.json (every file, its size and SHA-256), requirements.txt and PORTABLE_README.txt written by this script
  Zip layout = your folders: files the project keeps one level up (vina.exe, the pretrained model, receptor files) go at the root next to the project folder, so config.find_file finds them as here.
WHAT STAYS BEHIND
  everything else: older scripts, older checkpoints, old results and unrelated files, plus gui_runs/, rl_pilot checkpoints other than r3, __pycache__, *.pyc, *.zip, temporary files.
  The plan lists what is left behind BY NAME; add one back with --also NAME.  --everything packs every file instead (an archive).  No python packages are ever included.

--rl adopt    (the default) ships the RL pilot as an OPTIONAL PfDHFR-tuned model: rl_pilot's scripts, logs and small result files (no pilot checkpoints), and the final r3 checkpoint,
              placed in the project folder as checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt so the GUI and generate_for_request.py list it next to the general models.
--rl exclude  packs nothing from rl_pilot and no r3 model.
--everything  the archive: every file of the project folder and the top-level files of its parent folder.  --lean (with --everything) then keeps only the core checkpoints.

USAGE  (run from the project folder, the one that holds config.py)
  python make_portable_zip.py --plan                  prints what would be packed and what is left behind; writes nothing
  python make_portable_zip.py --plan --list           the same, and every file (packed or left behind) is written to portable_file_list.txt
  python make_portable_zip.py                         builds exemplar_generator_portable.zip next to the project folder (in its parent)
  python make_portable_zip.py --also notes.txt        add a file the code does not use (can be repeated)
  python make_portable_zip.py --everything --lean     an archive with every file except the extra checkpoints
  python make_portable_zip.py --rl exclude            without the r3 model and the RL pilot
  python make_portable_zip.py --skip some_file.csv    leave one named file out (can be repeated)
  python make_portable_zip.py --out D:\\ship\\pack.zip  another output file
Afterwards run  python check_portable.py  on the new machine.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import config as cfg
import check_portable as cp

DEFAULT_ZIP_NAME = 'exemplar_generator_portable.zip'
R3_SOURCE = os.path.join('rl_pilot', 'runs', 'r3', 'checkpoint_rl_r3.pt')
R3_TARGET_NAME = 'checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt'
# code files that are packed although check_portable.CODE_FILES does not list them: the command-line driver that rebuilds the base model (see REBUILD_FROM_SCRATCH.md)
EXTRA_CODE_FILES = ('pretrain_base_model.py',)

EXCLUDE_DIRS = {'gui_runs', 'rl_pilot', '__pycache__', '.git', '.vscode', '.idea', 'node_modules'}
EXCLUDE_SUFFIXES = ('.pyc', '.pyo', '.zip', '.tmp', '.partial')
EXCLUDE_NAMES = {'thumbs.db', 'desktop.ini', '.ds_store', 'portable_file_list.txt'}
# top-level files of the PARENT folder that are packed (anything else there is left alone)
PARENT_SUFFIXES = ('.py', '.pt', '.csv', '.pdbqt', '.pdb', '.json', '.jsonl', '.exe', '.md', '.txt', '.sdf', '.mol2')
# small result files of the RL pilot that --rl adopt packs (never its checkpoints, except the final r3 model)
RL_SUFFIXES = ('.py', '.txt', '.md', '.csv', '.json', '.jsonl', '.log')
CORE_CKPTS = (cfg.PRETRAINED_CKPT_NAME, cfg.EXEMPLAR_CKPT_NAME, cfg.HELDOUT_CKPT_NAME)
STORED_SUFFIXES = ('.pt', '.zip')

REQUIREMENTS = """# Python packages for the exemplar generator (tested with Python 3.13).  Nothing here is bundled in the zip.
# torch: for a GPU install the build that matches the machine's CUDA first, for example
#   pip install torch --index-url https://download.pytorch.org/whl/cu130
torch>=2.13
numpy>=2.5
pandas>=3.0
rdkit>=2026.3
meeko>=0.8
tqdm
# not needed: pretrain_base_model.py (the base-model rebuild) uses the refactored modules only; the GUI's synthetic-accessibility column uses the scorer inside RDKit
"""

README_TEXT = """EXEMPLAR GENERATOR - PORTABLE PACKAGE
Built {built} by make_portable_zip.py.  Optional PfDHFR-tuned model (RL r3): {rl}.

SET-UP ON A NEW MACHINE
  1. Extract the zip anywhere (a path with spaces is fine).  You get this layout:
       <root>/                    files the project keeps one level up: vina.exe, the pretrained base model, older scripts
       <root>/{proj}/    the project: code, GUI, receptor files, checkpoints, caches, CleanedData.csv
  2. Install Python 3.13 (3.10 or newer is likely fine) and the packages:
       cd {proj}
       pip install -r requirements.txt          (read the torch line first: GPU users install the CUDA build of torch first)
  3. Check that nothing is missing and the machine can run it:
       python check_portable.py                 files against the manifest, inputs, packages, vina
       python check_portable.py --deep --smoke  also the GUI self-check and a 3-molecule generation test
  4. Start the page:
       python gui_server.py                     opens the browser;  python gui_server.py --demo  shows made-up results without any model

WHAT IT CAN DO
  generate analogs of known ligands for a disease or target (held-out model: checkpoint_exemplar_seed_v3_holdout.pt), dock them at the receptors that passed calibration, and add
  a new disease or target from a few ligands.  Docking is validated for PfDHFR (plain Vina) and DPP-4 (only with the interaction rule); SERT is not validated, so docking is not offered.
  Two models are offered in the GUI and with --checkpoint: the GENERAL model (checkpoint_exemplar_seed_v3_holdout.pt, the default) and the optional PfDHFR-tuned model
  (checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt: fine-tuned by reinforcement learning on one target; for Malaria / T89515 only, it keeps the antifolate pharmacophore and stays close to its seeds).
  See FROZEN_TRUST_DECISIONS.md for the trust decisions and rl_pilot/RL_PILOT_README.txt for the RL record and the wording of its claims.
  RESULTS_AND_LIMITS.md says what the tool can and cannot be claimed to do.

REBUILDING THE MODELS (optional)
  REBUILD_FROM_SCRATCH.md lists every step from CleanedData.csv: pretrain_base_model.py for the base model, then the GUI's Train stage and the later stages.  A rebuild makes NEW models:
  the shipped checkpoints are the ones the reported results belong to, so keep them.

NEEDS NOTHING ELSE
  Everything the scripts read is in this package: library, models, receptor files, docking program, caches.  Generating, docking and the GUI need no internet.  Only two OPTIONAL tools go online:
  fetch_potency.py (ChEMBL potency filter) and prep_receptors.py (downloads a PDB file if it is not already here).  A Windows 64-bit machine with the python packages is enough.

NOTES
  vina.exe is a Windows program.  On Linux or macOS, put a native AutoDock Vina 1.2.7 binary under the same name in the same place.
  config.py contains one absolute fallback path from the original machine (LEGACY_ROOT); it is only used when a file is found nowhere else and can be ignored.
  The GUI writes its results to gui_runs/ (not packed).  Deleting that folder is safe.
"""


# Size in bytes as a short string.
def human(n):
    return cp.human(n)


# A path relative to `base`, with forward slashes (the form zip files use).
def rel(path, base):
    return os.path.relpath(path, base).replace(os.sep, '/')


# True when a file is never packed (cache folders, temporary and previous zip files, 'smoke' test checkpoints, editor and system files).
def excluded_file(name):
    low = name.lower()
    return low in EXCLUDE_NAMES or low.endswith(EXCLUDE_SUFFIXES) or low.startswith('~$') or ('smoke' in low and low.endswith('.pt'))


# True when a checkpoint is left out under --lean (every checkpoint_*.pt except the core ones and, with --rl adopt, the r3 model).
def lean_drops(name, a):
    if not a.lean or not (name.startswith('checkpoint') and name.endswith('.pt')):
        return False
    return name not in CORE_CKPTS and not (a.rl == 'adopt' and name == R3_TARGET_NAME)


# Packs the optional RL pilot (--rl adopt): its scripts, logs and small result files, and the final r3 model under the name the GUI lists.  `add(src, arc, cat)` and `note(label, path)` come from the caller.
def pack_rl(a, add, note):
    if a.rl != 'adopt':
        return
    proj = cfg.HERE
    proj_name = os.path.basename(proj)
    for dirpath, dirnames, filenames in os.walk(os.path.join(proj, 'rl_pilot')):
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__')
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            if fn.lower().endswith(RL_SUFFIXES) and not excluded_file(fn):
                add(p, proj_name + '/' + rel(p, proj), 'rl_pilot (adopted)')
            elif fn.lower().endswith('.pt'):
                note('rl_pilot checkpoints other than the final r3 model', p)
    src = os.path.join(proj, R3_SOURCE)
    if os.path.isfile(src):
        add(src, proj_name + '/' + R3_TARGET_NAME, 'rl_pilot (adopted)')


# --everything: lists every file of the project folder and the top-level files of its parent folder (minus the exclusions), as dicts {src, arc, cat, size}.  `skipped` collects what was left out.
def collect_everything(a, quiet=False):
    proj, root = cfg.HERE, os.path.dirname(cfg.HERE)
    proj_name = os.path.basename(proj)
    entries, seen, skipped = [], set(), {}
    skip_names = {s.lower() for s in a.skip}

    def add(src, arc, cat):
        if arc in seen:
            return
        seen.add(arc)
        entries.append({'src': src, 'arc': arc, 'cat': cat, 'size': os.path.getsize(src)})

    def note(label, path):
        d = skipped.setdefault(label, [0, 0])
        d[0] += 1
        d[1] += os.path.getsize(path)

    # 1. the project folder
    for dirpath, dirnames, filenames in os.walk(proj):
        for d in list(dirnames):
            if d in EXCLUDE_DIRS:
                dirnames.remove(d)
                if d == 'rl_pilot' and a.rl == 'adopt' and dirpath == proj:
                    continue                      # packed (selectively) in step 4
                for dp, _, fns in os.walk(os.path.join(dirpath, d)):
                    for fn in fns:
                        note(d + '/', os.path.join(dp, fn))
        dirnames.sort()
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            if excluded_file(fn):
                note('temporary / zip / smoke files', p)
            elif fn.lower() in skip_names:
                note('named with --skip', p)
            elif lean_drops(fn, a):
                note('other checkpoints (--lean)', p)
            else:
                add(p, proj_name + '/' + rel(p, proj), 'project')
    # 2. top-level files of the parent folder
    for fn in sorted(os.listdir(root)):
        p = os.path.join(root, fn)
        if not os.path.isfile(p) or not fn.lower().endswith(PARENT_SUFFIXES):
            continue
        if excluded_file(fn):
            note('temporary / zip / smoke files', p)
        elif fn.lower() in skip_names:
            note('named with --skip', p)
        elif lean_drops(fn, a):
            note('other checkpoints (--lean)', p)
        else:
            add(p, fn, 'parent folder')
    # 3. required inputs that live somewhere else (found by config.find_file): packed next to the project, where the lookup also finds them
    for name in (cfg.CSV_NAME, cfg.VINA_NAME) + CORE_CKPTS:
        p = cfg.find_file(name)
        if p and not any(e['arc'].endswith('/' + name) or e['arc'] == name for e in entries):
            add(p, name, 'found elsewhere')
    # 3b. receptor, structure and extra-receptor files named in the code, wherever config.find_file finds them
    names, _ = cp.referenced_names()
    unresolved = []
    for name in names:
        p = cfg.find_file(name)
        if not p:
            unresolved.append(name)
        elif not any(e['arc'].endswith('/' + name) or e['arc'] == name for e in entries):
            add(p, name, 'found elsewhere')
    if unresolved:
        skipped['named in targets.py / the GUI but not found on this machine (ignored)'] = [len(unresolved), 0]
        if not quiet:
            print('   not found anywhere (not packed; harmless if that target is unused): ' + ', '.join(unresolved))
    # 4. the optional RL pilot
    pack_rl(a, add, note)
    return entries, skipped


# The default package: ONLY what the scripts and the GUI use, found through the code itself: the code files, the page, the library, vina, the core models, the receptor and structure files named in
# targets.py / the GUI / dock_generated, the docking caches the GUI uses and the notes.  Everything else (older scripts, older checkpoints, old results) stays behind; --also adds a named file.
def collect_minimal(a, quiet=False):
    proj = cfg.HERE
    proj_name = os.path.basename(proj)
    entries, seen, skipped = [], set(), {}
    skip_names = {x.lower() for x in a.skip}

    def arc_for(path):
        path = os.path.abspath(path)
        return proj_name + '/' + rel(path, proj) if os.path.commonpath([path, proj]) == proj else os.path.basename(path)

    def add(src, arc, cat):
        if arc in seen or os.path.basename(src).lower() in skip_names:
            return
        seen.add(arc)
        entries.append({'src': src, 'arc': arc, 'cat': cat, 'size': os.path.getsize(src)})

    def note(label, path):
        d = skipped.setdefault(label, [0, 0])
        d[0] += 1
        d[1] += os.path.getsize(path)
    for f in list(cp.CODE_FILES) + list(EXTRA_CODE_FILES):
        p = os.path.join(proj, f)
        if os.path.isfile(p):
            add(p, arc_for(p), 'code')
    page = os.path.join(proj, 'gui_static')
    for f in (sorted(os.listdir(page)) if os.path.isdir(page) else []):
        if os.path.isfile(os.path.join(page, f)) and not excluded_file(f):
            add(os.path.join(page, f), proj_name + '/gui_static/' + f, 'GUI page')
    for f in sorted(os.listdir(proj)):
        low, p = f.lower(), os.path.join(proj, f)
        if os.path.isfile(p) and not excluded_file(f) and (low.endswith('.md') or low == 'requirements.txt' or ('readme' in low and low.endswith('.txt'))):
            add(p, arc_for(p), 'notes')
    for name in (cfg.CSV_NAME, cfg.VINA_NAME) + CORE_CKPTS:
        p = cfg.find_file(name)
        if p:
            add(p, arc_for(p), 'library, vina, core models')
    names, _ = cp.referenced_names()
    unresolved = []
    for name in names:
        p = cfg.find_file(name)
        if p:
            add(p, arc_for(p), 'receptor / structure files')
        else:
            unresolved.append(name)
    if unresolved:
        skipped['named in targets.py / the GUI but not found on this machine (ignored)'] = [len(unresolved), 0]
        if not quiet:
            print('   not found anywhere (not packed; harmless if that target is unused): ' + ', '.join(unresolved))
    try:
        import gui_backend as gb
        caches = [t['dock']['cache'] for t in gb.STUDY_TARGETS if t.get('dock')]
    except Exception:
        caches = ['dock_cache_v2.jsonl', 'dock_cache_inter.jsonl']
    for name in caches:
        p = cfg.find_file(name)
        if p:
            add(p, arc_for(p), 'docking caches')
    for name in a.also:
        p = cfg.find_file(name) or (os.path.abspath(name) if os.path.isfile(name) else None)
        if p:
            add(p, arc_for(p), 'named with --also')
        elif not quiet:
            print(f"   --also {name}: file not found, ignored")
    pack_rl(a, add, note)
    return entries, skipped


# Lists the files to pack: the minimal package by default, everything with --everything.  Also returns the files the minimal package leaves behind (None with --everything).
def collect(a):
    if a.everything:
        entries, skipped = collect_everything(a)
        return entries, skipped, None
    entries, skipped = collect_minimal(a)
    b = copy.copy(a)
    b.lean = False
    full, _ = collect_everything(b, quiet=True)
    have = {e['arc'] for e in entries}
    return entries, skipped, [e for e in full if e['arc'] not in have]


# The items that must be in the package; returns (label, ok) pairs checked against the collected entries.
def required_status(entries):
    arcs = [e['arc'] for e in entries]
    have = lambda name: any(x == name or x.endswith('/' + name) for x in arcs)
    items = [(cfg.CSV_NAME, have(cfg.CSV_NAME)), (cfg.VINA_NAME, have(cfg.VINA_NAME)), (cfg.HELDOUT_CKPT_NAME, have(cfg.HELDOUT_CKPT_NAME)), (cfg.EXEMPLAR_CKPT_NAME, have(cfg.EXEMPLAR_CKPT_NAME)),
             (cfg.PRETRAINED_CKPT_NAME, have(cfg.PRETRAINED_CKPT_NAME))]
    items += [(f, have(f)) for f in cp.CODE_FILES]
    items += [('gui_static/' + f, any(x.endswith('gui_static/' + f) for x in arcs)) for f in cp.GUI_PAGE_FILES]
    ext = lambda suffix: sum(1 for x in arcs if x.lower().endswith(suffix))
    items += [('at least one receptor file (.pdbqt)', ext('.pdbqt') > 0), ('at least one ligand/structure file (.pdb)', ext('.pdb') > 0)]
    return items


# True for a checkpoint that --lean would drop: any checkpoint_*.pt other than the core ones (and the r3 model).
def lean_candidate(e):
    name = os.path.basename(e['arc'])
    return name.startswith('checkpoint') and name.endswith('.pt') and name not in CORE_CKPTS and name != R3_TARGET_NAME


# Prints the plan: counts and sizes by category, what --lean would save, the biggest remaining files, file types, what is left out and the required items.
def print_plan(entries, skipped, a, left=None):
    total = sum(e['size'] for e in entries)
    print(f"\nPLAN   {len(entries)} files, {human(total)}   ({'everything' if a.everything else 'minimal: only what is used'}; RL pilot: {a.rl}{', lean' if a.lean else ''})")
    cats = {}
    for e in entries:
        c = cats.setdefault(e['cat'], [0, 0])
        c[0] += 1
        c[1] += e['size']
    for c, (n, s) in sorted(cats.items()):
        print(f"   packed   {c:<28} {n:>5} files  {human(s):>10}")
    drop = [e for e in entries if lean_candidate(e)] if a.everything else []
    if drop:
        print(f"\n   checkpoints beyond the core ones: {len(drop)} files, {human(sum(e['size'] for e in drop))}   (--lean leaves these out; the core ones are {', '.join(CORE_CKPTS)} and the r3 model)")
    rest = [e for e in entries if not lean_candidate(e)]
    print(f"\n   biggest files{' (without those extra checkpoints)' if drop else ''}")
    for e in sorted(rest, key=lambda e: -e['size'])[:15]:
        print(f"     {human(e['size']):>10}  {e['arc']}")
    kinds = {}
    for e in rest:
        k = os.path.splitext(e['arc'])[1].lower() or '(none)'
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        kinds[k][1] += e['size']
    print(f"\n   file types{' (without those extra checkpoints)' if drop else ''}: " + ';  '.join(f"{k} {n} files {human(sz)}" for k, (n, sz) in sorted(kinds.items(), key=lambda x: -x[1][1])[:10]))
    if a.list:
        path = os.path.join(cfg.HERE, 'portable_file_list.txt')
        with open(path, 'w', encoding='utf-8') as fh:
            for e in sorted(entries, key=lambda e: (e['cat'], -e['size'])):
                fh.write(f"{e['cat']:<20} {human(e['size']):>10}  {e['arc']}\n")
            for e in sorted(left or [], key=lambda e: -e['size']):
                fh.write(f"{'LEFT BEHIND':<20} {human(e['size']):>10}  {e['arc']}\n")
        print(f"   every file is listed in {path}")
    if left:
        code = sorted(os.path.basename(e['arc']) for e in left if e['arc'].lower().endswith(('.py', '.md', '.txt')))
        ckpt = [e for e in left if os.path.basename(e['arc']).startswith('checkpoint') and e['arc'].lower().endswith('.pt')]
        other = [e for e in left if e not in ckpt and not e['arc'].lower().endswith(('.py', '.md', '.txt'))]
        print(f"\n   LEFT BEHIND because nothing in the code uses it ({len(left)} files, {human(sum(e['size'] for e in left))}); add one back with --also NAME, or use --everything")
        if code:
            print(f"     scripts and notes ({len(code)}): " + ', '.join(code[:60]) + (' ...' if len(code) > 60 else ''))
        if ckpt:
            print(f"     older checkpoints ({len(ckpt)}, {human(sum(e['size'] for e in ckpt))}): " + ', '.join(sorted(os.path.basename(e['arc']) for e in ckpt)[:25]) + (' ...' if len(ckpt) > 25 else ''))
        kinds2 = {}
        for e in other:
            k = os.path.splitext(e['arc'])[1].lower() or '(none)'
            kinds2.setdefault(k, [0, 0])
            kinds2[k][0] += 1
            kinds2[k][1] += e['size']
        if kinds2:
            print("     other files: " + ';  '.join(f"{k} {n} files {human(sz)}" for k, (n, sz) in sorted(kinds2.items(), key=lambda x: -x[1][1])))
    print("\n   left out")
    if not skipped:
        print("     nothing")
    for label, (n, s) in sorted(skipped.items()):
        print(f"     {label:<52} {n:>5} files  {human(s):>10}")
    print("\n   required items")
    bad = []
    for label, ok in required_status(entries):
        if not ok:
            bad.append(label)
    print("     " + ('all present' if not bad else 'MISSING: ' + ', '.join(bad)))
    return bad


# Writes one file into the zip in a single pass (copy and hash together); returns its SHA-256.
def write_file(zf, e):
    zi = zipfile.ZipInfo.from_file(e['src'], e['arc'])
    zi.compress_type = zipfile.ZIP_STORED if e['arc'].lower().endswith(STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
    h = hashlib.sha256()
    with open(e['src'], 'rb') as src, zf.open(zi, 'w', force_zip64=e['size'] > 1900 * 1024 * 1024) as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            h.update(block)
            dst.write(block)
    return h.hexdigest()


# Builds the zip (to a .partial file first), writes the manifest, requirements and README into it, then checks the result.
def build(entries, a, out):
    proj_name = os.path.basename(cfg.HERE)
    total = sum(e['size'] for e in entries)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out)) or '.').free
    if free < total * 1.05:
        print(f"\nNOT ENOUGH DISK SPACE for {out}: need about {human(total)}, {human(free)} free.  Use --lean, --skip or --out on another drive.")
        return False
    partial = out + '.partial'
    files, t0, done = [], time.time(), 0
    with zipfile.ZipFile(partial, 'w', allowZip64=True) as zf:
        for i, e in enumerate(entries, 1):
            digest = write_file(zf, e)
            files.append({'path': e['arc'], 'size': e['size'], 'sha256': digest})
            done += e['size']
            if i % 5 == 0 or e['size'] > 50 * 1024 * 1024 or i == len(entries):
                print(f"   packed {i}/{len(entries)}  {human(done)} of {human(total)}  ({time.time() - t0:.0f} s)   {e['arc'][-60:]}", end='\r' if i != len(entries) else '\n')
        built = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        manifest = {'built': built, 'builder_python': sys.version.split()[0], 'project_folder': proj_name, 'rl': a.rl, 'lean': bool(a.lean), 'files': files}
        zf.writestr(proj_name + '/' + cp.MANIFEST_NAME, json.dumps(manifest, indent=1))
        if not any(e['arc'] == proj_name + '/requirements.txt' for e in entries):
            zf.writestr(proj_name + '/requirements.txt', REQUIREMENTS)
        zf.writestr('PORTABLE_README.txt', README_TEXT.format(built=built, rl=('included (checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt)' if a.rl == 'adopt' else 'not included'), proj=proj_name))
    if os.path.exists(out):
        os.remove(out)
    os.replace(partial, out)
    return True


# Opens the finished zip and compares it with the plan: every planned file is in it with its size, the required items are in it, and (with --verify-crc) every file reads back without error.
def verify(out, entries, a):
    print("\nverifying the zip")
    ok = True
    with zipfile.ZipFile(out) as zf:
        info = {i.filename: i for i in zf.infolist()}
        gone = [e['arc'] for e in entries if e['arc'] not in info]
        wrong = [e['arc'] for e in entries if e['arc'] in info and info[e['arc']].file_size != e['size']]
        print(f"   [{'OK     ' if not gone else 'MISSING'}] every planned file is in the zip" + (f"   ({len(gone)} missing, e.g. {gone[:3]})" if gone else ''))
        print(f"   [{'OK     ' if not wrong else 'MISSING'}] every file has its original size" + (f"   ({len(wrong)} differ, e.g. {wrong[:3]})" if wrong else ''))
        proj = os.path.basename(cfg.HERE)
        for need in (proj + '/' + cp.MANIFEST_NAME, proj + '/requirements.txt', 'PORTABLE_README.txt'):
            has = need in info
            print(f"   [{'OK     ' if has else 'MISSING'}] {need}")
            ok &= has
        ok &= not gone and not wrong
        if a.verify_crc:
            bad = zf.testzip()
            print(f"   [{'OK     ' if bad is None else 'MISSING'}] every file reads back without a checksum error" + (f"   (first bad file: {bad})" if bad else ''))
            ok &= bad is None
        else:
            print("   (add --verify-crc to read every file back and check the zip's own checksums; slow for large checkpoints)")
    print(f"\n{'ZIP OK' if ok else 'ZIP HAS PROBLEMS'}: {out}   ({human(os.path.getsize(out))})")
    return ok


# Reads the options, prints the plan and (unless --plan) builds and verifies the zip.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', action='store_true', help='print what would be packed and left out, write nothing')
    ap.add_argument('--rl', choices=['exclude', 'adopt'], default='adopt', help='adopt (default) = pack the optional PfDHFR-tuned r3 model and the RL pilot record; exclude = leave them out')
    ap.add_argument('--everything', action='store_true', help='pack every file of the project folder and the top-level files of its parent (the old behaviour), minus the exclusions')
    ap.add_argument('--also', action='append', default=[], help='a file name to add to the minimal package (repeatable)')
    ap.add_argument('--lean', action='store_true', help='with --everything: keep only the core checkpoints')
    ap.add_argument('--skip', action='append', default=[], help='a file name to leave out (repeatable)')
    ap.add_argument('--out', default=None, help='output zip (default: %s in the parent of the project folder)' % DEFAULT_ZIP_NAME)
    ap.add_argument('--list', action='store_true', help='also write every planned file with its size to portable_file_list.txt')
    ap.add_argument('--verify-crc', action='store_true', help='read every file back after building (slow for large checkpoints)')
    a = ap.parse_args()
    if a.rl == 'adopt' and not os.path.isfile(os.path.join(cfg.HERE, R3_SOURCE)):
        print(f"The r3 model is packed by default but {R3_SOURCE} is not in the project folder.  Put it there, or add --rl exclude to build without it.")
        sys.exit(1)
    out = a.out or os.path.join(os.path.dirname(cfg.HERE), DEFAULT_ZIP_NAME)
    print(f"Portable package   (project {cfg.HERE})")
    entries, skipped, left = collect(a)
    bad = print_plan(entries, skipped, a, left)
    if a.plan:
        print("\n(plan only: nothing was written)")
        sys.exit(1 if bad else 0)
    if bad:
        print("\nNot building: the required items above are missing.  Fix that (or run --plan to see what was found) and run again.")
        sys.exit(1)
    print(f"\nbuilding {out}")
    if not build(entries, a, out):
        sys.exit(1)
    sys.exit(0 if verify(out, entries, a) else 1)


if __name__ == '__main__':
    main()
