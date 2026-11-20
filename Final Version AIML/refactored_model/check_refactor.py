#!/usr/bin/env python3
"""
check_refactor.py -- does the refactored code (base_*.py, docking.py, targets.py, rl_utils.py, the changed scripts) still do exactly what v13 did?

  1. every definition that was moved out of legacy/Data_Preprocessing_and_model_updated_v13.py (or legacy/rl_fixes.py) is AST-identical to the original
     (docstrings and comments are ignored: they were rewritten after the move);
     the settings that moved with them have the v13 values; DISEASE_TARGET_CONFIGS equals the v13 table
  2. every module imports (needs torch, RDKit, meeko)
  3. real Vina: a few cached PfDHFR dockings and the cached redocking of the crystal ligand are repeated through the new code path and must give the same
     scores / RMSD as the cache (docking.dock_ligand, RealBackend.dock, RealBackend.redock, pose_rmsds)

  python check_refactor.py                      all three (about 2 min with vina.exe)
  python check_refactor.py --skip-docking       steps 1 and 2 only
  python check_refactor.py --n 5                repeat 5 cached dockings instead of 3
"""
import argparse
import ast
import json
import os
import sys

# the folder this script lives in; the moved modules and legacy/ (the untouched v13 and rl_fixes.py) are looked up relative to it
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LEGACY = os.path.join(HERE, 'legacy')
V13 = os.path.join(LEGACY, 'Data_Preprocessing_and_model_updated_v13.py')
RLF = os.path.join(LEGACY, 'rl_fixes.py')

# the refactored modules whose definitions were moved verbatim out of v13 (rl_utils.py out of rl_fixes.py)
MODULES = ['base_data.py', 'base_model.py', 'base_train.py', 'docking.py', 'targets.py', 'rl_utils.py']
# module-level settings that moved together with the code: each must equal its v13 value.  Paths are left out on purpose, they now come from config.py
SETTINGS = {'base_train.py': ['CONTEXT_MASKING_VERSION', 'USE_TORCH_COMPILE', 'PRETRAIN_MAX_EPOCHS', 'PRETRAIN_PATIENCE', 'PRETRAIN_LR'],
            'docking.py': ['DOCKING_EXHAUSTIVENESS', 'DOCKING_NUM_POSES', 'DOCKING_BOX_SIZE', 'LIGAND_PROTONATION_MODE', 'VINA_SUPPORTED_ELEMENTS', '_PROTONATION_SELFTEST'],
            'config.py': ['SMOKE_TEST']}
# one True/False per check, in the order they ran
RESULTS = []


# Records one check result and prints it as PASS or FAIL.
def say(ok, msg):
    RESULTS.append(ok)
    print(('  PASS  ' if ok else '  FAIL  ') + msg)


# Parses a Python source file into a syntax tree.
def parse(path):
    with open(path, encoding='utf-8') as f:
        return ast.parse(f.read())


# Text dump of a function or class syntax tree with every docstring removed.  Comments never reach a syntax tree, so two versions that differ only in
# comments or docstrings dump identically.
def nodoc_dump(node):
    import copy
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], 'value', None), ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(node)


# Maps each top-level function, class and simple assigned name of a module to its syntax-tree node (the first assignment wins).
def top_level(tree):
    d = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            d[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    d.setdefault(t.id, n)
    return d


# Value of a module-level setting: the literal if it is one, else its source text; '<missing>' if the module does not define it.
def const(tree, name):
    n = top_level(tree).get(name)
    if n is None:
        return '<missing>'
    try:
        return ast.literal_eval(n.value)
    except Exception:
        return ast.unparse(n.value)


# Check 1: every function and class in the refactored modules, and the DISEASE_TARGET_CONFIGS table, must be identical to its v13 original (ignoring docstrings), and the
# settings listed in SETTINGS must have the v13 values.
def step1():
    print("\n1. moved code is identical to the originals (AST comparison)")
    for p in (V13, RLF):
        if not os.path.exists(p):
            say(False, f"{os.path.relpath(p, HERE)} not found (legacy/ must hold the original v13 and rl_fixes.py)")
            return
    old, rl = top_level(parse(V13)), top_level(parse(RLF))
    for fn in MODULES:
        new = top_level(parse(os.path.join(HERE, fn)))
        ref = rl if fn == 'rl_utils.py' else old
        n_def = n_same = 0
        bad = []
        for name, node in new.items():
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) or name == 'DISEASE_TARGET_CONFIGS':
                if name not in ref:
                    bad.append(f"{name} (not in the original)")
                    continue
                n_def += 1
                same = (ast.literal_eval(node.value) == ast.literal_eval(ref[name].value)) if name == 'DISEASE_TARGET_CONFIGS' else nodoc_dump(node) == nodoc_dump(ref[name])
                if same:
                    n_same += 1
                else:
                    bad.append(name)
        say(not bad and n_def > 0, f"{fn}: {n_same} of {n_def} definitions identical" + (f"; PROBLEM: {', '.join(bad)}" if bad else ''))
    vt = parse(V13)
    for fn, names in SETTINGS.items():
        t = parse(os.path.join(HERE, fn))
        bad = [n for n in names if const(t, n) != const(vt, n)]
        say(not bad, f"{fn}: {len(names) - len(bad)} of {len(names)} settings equal the v13 values" + (f"; DIFFERENT: {', '.join(bad)}" if bad else ''))


# Check 2: every module can be imported (this needs torch, RDKit and meeko installed).  A failed import is reported and the loop goes on.
def step2():
    print("\n2. every module imports")
    import importlib
    for m in ['config', 'targets', 'base_data', 'base_model', 'base_train', 'docking', 'rl_utils', 'exemplar_data', 'exemplar_model', 'dock_generated']:
        try:
            importlib.import_module(m)
            say(True, f"import {m}")
        except Exception as e:
            # an import failure counts as a failed check, it does not stop the run
            say(False, f"import {m}: {type(e).__name__}: {e}")


# Reads a docking cache (one JSON record per line).  Returns (plain, redock): plain is a list of (SMILES, score) for ordinary exhaustiveness-8, seed-42 dockings of the
# labelled receptor, and redock is the first cached redocking record of its reference ligand, or None.
def read_cache(path, label):
    plain, redock = [], None
    for line in open(path, encoding='utf-8'):
        try:
            r = json.loads(line)
            key = r['key']
        except Exception:
            continue
        # redocking records have keys of the form 'redock|label|...' and carry the scores and RMSDs in 'extra'
        if key.startswith('redock|'):
            parts = key[len('redock|'):].split('|', 4)
            if len(parts) == 5 and parts[0] == label and redock is None and (r.get('extra') or {}).get('rmsd'):
                redock = (parts, r['extra'])
            continue
        # an ordinary docking key has five '|' separated parts: label, receptor signature, exhaustiveness, seed, SMILES; records that carry extra data are skipped
        parts = key.split('|', 4)
        if len(parts) == 5 and parts[0] == label and parts[2] == '8' and parts[3] == '42' and r.get('score') is not None and not r.get('extra'):
            plain.append((parts[4], float(r['score'])))
    return plain, redock


# Check 3: re-docks a few cached molecules and the crystal ligand through the refactored docking code and requires the same Vina scores (within --tol) and the
# same redocking RMSD (within 0.05 A) as the cache.  It reports a skipped check rather than a failure when the cache, vina.exe or the reference files are missing.
def step3(a):
    print(f"\n3. real Vina: repeat cached dockings of {a.label} (cache {a.cache})")
    import config as cfg
    import docking
    import dock_generated as dg
    from targets import DISEASE_TARGET_CONFIGS
    cache = dg.find_file(a.cache)
    if not cache:
        say(False, f"cache {a.cache} not found (run from the folder that holds it, or pass --cache)")
        return
    vina = docking.VINA_EXECUTABLE_PATH
    if not os.path.exists(vina):
        say(False, f"vina.exe not found at {vina}")
        return
    cases, receptors = dg.build_cases(DISEASE_TARGET_CONFIGS, [a.case], find=dg.find_file)
    rec = receptors[a.label]
    plain, redock = read_cache(cache, a.label)
    if not plain:
        say(False, f"no plain exhaustiveness-8 / seed-42 dockings of {a.label} in {cache}")
        return
    be = dg.RealBackend(docking)
    print(f"   vina: {vina}\n   receptor: {rec['pdbqt']}\n   re-docking {min(a.n, len(plain))} of {len(plain)} cached molecules (Vina scores print to 0.1 kcal/mol)")
    for smi, old in plain[:a.n]:
        new = be.dock(smi, rec, 8, 42)
        say(new is not None and abs(new - old) <= a.tol, f"{smi[:50]:<50} cached {old:7.2f}   now {new if new is None else round(new, 2)}")
    if redock is None:
        print("   (no cached redocking record for this receptor: redock check skipped)")
        return
    parts, extra = redock
    ref, cry = rec.get('reference'), rec.get('crystal')
    if not ref or not cry:
        print("   (reference ligand or crystal file missing: redock check skipped)")
        return
    r = be.redock(ref, rec, int(parts[2]), int(parts[3]), cry)
    ok_s = r['scores'] and abs(r['scores'][0] - extra['scores'][0]) <= a.tol
    ok_r = r.get('rmsd') and abs(r['rmsd'][0] - extra['rmsd'][0]) <= 0.05
    say(bool(ok_s), f"redock top score  cached {extra['scores'][0]:.2f}   now {r['scores'][0] if r['scores'] else None}")
    say(bool(ok_r), f"redock top RMSD   cached {extra['rmsd'][0]:.2f} A   now {round(r['rmsd'][0], 2) if r.get('rmsd') else r.get('rmsd_error')}")


# Runs the checks (the docking one unless --skip-docking is given) and prints the summary.  The exit status is 1 if any check failed.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-docking', action='store_true')
    ap.add_argument('--cache', default='dock_cache_v2.jsonl')
    ap.add_argument('--label', default='PfDHFR')
    ap.add_argument('--case', default='malaria_t89515')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--tol', type=float, default=0.15, help='allowed score difference (kcal/mol)')
    a = ap.parse_args()
    step1()
    step2()
    if not a.skip_docking:
        step3(a)
    bad = RESULTS.count(False)
    print(f"\n{'ALL CHECKS PASSED' if not bad else str(bad) + ' CHECK(S) FAILED'}  ({len(RESULTS)} checks)")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
