#!/usr/bin/env python3
"""
test_new_context.py -- an END-TO-END test of the stated goal: "a NEW disease and a NEW target are added to the library and the generator
works for them", run through the real request tool (generate_for_request.py) and its --extra-library path.

It makes two pretend new contexts out of two real targets that the model never saw in training:
    context A = disease 'Coronavirus' + target 'Tx12345'   <- the ligands of real target T52522
    context B = disease 'Zika'        + target 'Tx67890'   <- the ligands of real target T45993
(the names are invented and exist nowhere in the training data; the model only ever sees SMILES).
For each, half of the target's ligands are ADDED to the library as the new context (the 'known ligands'); the other half are kept
secret as the TEST set.  The generator must then produce molecules that resemble the secret half more than they resemble what it
makes for the OTHER new context, and more than random library molecules do.

  step 1  prep  : writes ctx_library.csv (CleanedData minus both targets), ctx_new_A.csv, ctx_new_B.csv, ctx_test_A.txt, ctx_test_B.txt
  step 2        : generate_for_request.py twice (one command each, see USAGE)
  step 3  score : reads gen_A.csv / gen_B.csv and prints the verdict

USAGE (one chained command; needs torch + RDKit + pandas)
  python test_new_context.py prep && python generate_for_request.py --csv ctx_library.csv --extra-library ctx_new_A.csv --disease Coronavirus --target Tx12345 --n 60 --out gen_A.csv && python generate_for_request.py --csv ctx_library.csv --extra-library ctx_new_B.csv --disease Zika --target Tx67890 --n 60 --out gen_B.csv && python test_new_context.py score
"""
import argparse
import csv
import os
import random
import re
import sys

import config as cfg

# The two pretend contexts.  Each reuses the ligands of a real target the model never saw in training ('real') under an invented disease and target name.
DEFAULT_CSV = cfg.DEFAULT_CSV
COLS = ['Disease Entry', 'Targets for Drug', 'Canonical Smiles']
CTX = {'A': dict(real='T52522', disease='Coronavirus', target='Tx12345'),
       'B': dict(real='T45993', disease='Zika', target='Tx67890')}


# True if the target code appears as a whole token in a library cell (a cell such as "['T52522', 'T1']"), so T5 does not match T52522.
def has_target(cell, name):
    return isinstance(cell, str) and re.search(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', cell) is not None


# Step 1.  Removes both real targets from the library (ctx_library.csv), splits each target's molecules into two halves, writes one half as the new context's known ligands (ctx_new_A/B.csv, under the invented names)
# and keeps the other half secret as the test set (ctx_test_A/B.txt, minus anything still in the reduced library).
def prep(a):
    import pandas as pd
    df = pd.read_csv(a.csv)
    for c in COLS:
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not found in {a.csv}: {list(df.columns)}")
    # rows that belong to each real target; they are removed from the library so the generator cannot see them
    drop = {k: df['Targets for Drug'].apply(lambda x, n=v['real']: has_target(x, n)) for k, v in CTX.items()}
    all_drop = drop['A'] | drop['B']
    rest = df[~all_drop]
    rest_smiles = set(s for s in rest['Canonical Smiles'] if isinstance(s, str))
    rest.to_csv('ctx_library.csv', index=False)
    print(f"[prep] ctx_library.csv: {len(rest)} rows ({int(all_drop.sum())} rows of the two targets removed)")
    # shuffle each target's molecules and split them in half: known ligands vs secret test molecules
    rng = random.Random(a.seed)
    for k, v in CTX.items():
        sub = df[drop[k]]
        mols = sorted({s for s in sub['Canonical Smiles'] if isinstance(s, str) and s.strip()})
        if len(mols) < 8:
            raise SystemExit(f"target {v['real']} has only {len(mols)} usable molecules: choose another one in CTX")
        rng.shuffle(mols)
        half = len(mols) // 2
        lib_m, test_m = set(mols[:half]), [m for m in mols[half:] if m not in rest_smiles]
        rows = [{'Disease Entry': str([v['disease']]), 'Targets for Drug': str([v['target']]), 'Canonical Smiles': m}
                for m in sorted(lib_m)]
        with open(f'ctx_new_{k}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        with open(f'ctx_test_{k}.txt', 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(test_m) + '\n')
        print(f"[prep] context {k}: '{v['disease']}' + '{v['target']}'  (really {v['real']}):  {len(lib_m)} ligands added to the library, "
              f"{len(test_m)} kept secret as test")


# Mean, over the query fingerprints, of the best Tanimoto similarity to any fingerprint in the pool; NaN if either is empty.
def mean_best(chem, q_fps, pool_fps):
    import numpy as np
    if not q_fps or not pool_fps:
        return float('nan')
    return float(np.mean([max(chem.DataStructs.BulkTanimotoSimilarity(f, pool_fps)) for f in q_fps]))


# 1 - mean pairwise Tanimoto similarity of a set (a random sample of at most `cap` molecules); a value far below that of the real ligands means the generator has collapsed to near-identical output.
def internal_diversity(chem, fps, cap=150, seed=0):
    import numpy as np
    rng = random.Random(seed)
    fps = list(fps)
    if len(fps) > cap:
        fps = rng.sample(fps, cap)
    if len(fps) < 2:
        return float('nan')
    d = []
    for i in range(len(fps) - 1):
        d.extend(1 - np.array(chem.DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])))
    return float(np.mean(d))


# Rows of a generated-set CSV as dicts.
def read_gen(path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


# All the numbers of the test, kept apart from printing so it can be tested without RDKit.  For each context it computes the similarity of its samples to its own secret ligands versus the other context's samples
# (lift_other) and versus random library molecules (lift_random), the 'copy the added ligands' ceiling, exact rediscoveries, internal diversity, and the lift split by how close each sample is to its seed.
def score_logic(chem, gen, test, lib_half, rand_lib, verbose=True):
    import numpy as np
    keys = sorted(gen)
    gf = {k: chem.fps([r['smiles'] for r in gen[k]]) for k in keys}
    tf = {k: chem.fps(test[k]) for k in keys}
    # S[(i, j)]: how similar context i's samples are to context j's secret ligands (the diagonal is the real signal)
    S = {(i, j): mean_best(chem, gf[i], tf[j]) for i in keys for j in keys}
    rnd = chem.fps(rand_lib)
    out = {'S': S, 'lift_other': {}, 'lift_random': {}, 'ceiling': {}, 'bins': {}, 'rediscovered': {}, 'diversity': {}}
    for k in keys:
        others = [S[(j, k)] for j in keys if j != k]
        out['lift_other'][k] = S[(k, k)] / float(np.mean(others)) if others else float('nan')
        rb = mean_best(chem, rnd, tf[k])
        out['lift_random'][k] = S[(k, k)] / rb if rb == rb and rb > 0 else float('nan')
        out['ceiling'][k] = mean_best(chem, chem.fps(lib_half[k]), tf[k]) / rb if rb == rb and rb > 0 else float('nan')
        tcan = {chem.canon(s) for s in test[k]}
        out['rediscovered'][k] = sum(1 for r in gen[k] if chem.canon(r['smiles']) in tcan)
        out['diversity'][k] = (internal_diversity(chem, gf[k]), internal_diversity(chem, chem.fps(lib_half[k])))
        # split the samples by similarity to their own seed: near-copy, analog, far
        bins = {'near-copy (>=0.80)': [], 'analog (0.40-0.80)': [], 'far (<0.40)': []}
        for r in gen[k]:
            s = float(r['similarity_to_seed'])
            f = chem.fp(r['smiles'])
            if f is None:
                continue
            v = max(chem.DataStructs.BulkTanimotoSimilarity(f, tf[k])) if tf[k] else float('nan')
            bins['near-copy (>=0.80)' if s >= 0.8 else 'analog (0.40-0.80)' if s >= 0.4 else 'far (<0.40)'].append(v)
        out['bins'][k] = {b: (len(v), float(np.mean(v)) / rb if v and rb == rb and rb > 0 else float('nan')) for b, v in bins.items()}
    return out


# Step 3.  Loads the generated sets and the secret ligands, scores them with score_logic and prints per-context lifts plus a PASS / PARTIAL / FAIL verdict (PASS: both lifts versus random library molecules >= 1.30).
def score(a):
    import exemplar_model as xm
    chem = xm.Chem2()
    gen, test, lib_half = {}, {}, {}
    for k, v in CTX.items():
        for f in (f'gen_{k}.csv', f'ctx_test_{k}.txt', f'ctx_new_{k}.csv'):
            if not os.path.exists(f):
                raise SystemExit(f"missing {f}: run the whole chained command from the top")
        gen[k] = read_gen(f'gen_{k}.csv')
        test[k] = [l.strip() for l in open(f'ctx_test_{k}.txt', encoding='utf-8') if l.strip()]
        lib_half[k] = [r['Canonical Smiles'] for r in read_gen(f'ctx_new_{k}.csv')]
    import pandas as pd
    # 400 random library molecules (without the two removed targets) as the reference level for 'no conditioning'
    lib = [s for s in pd.read_csv('ctx_library.csv')['Canonical Smiles'].dropna().unique() if isinstance(s, str)]
    rng = random.Random(a.seed)
    rand_lib = rng.sample(lib, min(400, len(lib)))
    r = score_logic(chem, gen, test, lib_half, rand_lib)
    print("\n" + "=" * 100)
    print("NEW-CONTEXT TEST  (disease and target names invented; only their ligands were added to the library)")
    print("=" * 100)
    for k, v in CTX.items():
        n = len(gen[k])
        print(f"\ncontext {k}: '{v['disease']}' + '{v['target']}'   generated {n} novel valid molecules; test set {len(test[k])} secret ligands")
        others = ', '.join('%.3f (%s samples)' % (r['S'][(j, k)], j) for j in sorted(CTX) if j != k)
        print(f"  mean best-match to the secret ligands : {r['S'][(k, k)]:.3f}   (this context's samples)   vs {others}")
        print(f"  LIFT vs the other new context's samples : {r['lift_other'][k]:.2f}     LIFT vs random library molecules : {r['lift_random'][k]:.2f}"
              f"     (copying the added ligands themselves: {r['ceiling'][k]:.2f})")
        print(f"  secret ligands re-discovered exactly   : {r['rediscovered'][k]} of {n}")
        dg, dl = r['diversity'][k]
        print(f"  internal diversity (1 - mean Tanimoto) : generated {dg:.2f}   vs the added ligands {dl:.2f}   (much lower = collapse)")
        print("  lift vs random, split by how close the sample is to its seed:")
        for b, (m, l) in r['bins'][k].items():
            print(f"      {b:<20} n={m:<3} lift {l:.2f}")
    lo = [r['lift_random'][k] for k in CTX]
    print("\nVERDICT (my thresholds):")
    ok = all(x >= 1.30 for x in lo)
    part = all(x >= 1.10 for x in lo)
    verdict = 'PASS' if ok else 'PARTIAL' if part else 'FAIL'
    print("  new-context generation vs random library molecules: %s  ->  %s   (PASS >= 1.30 for both, PARTIAL >= 1.10)"
          % (', '.join('%.2f' % x for x in lo), verdict))
    print("  The earlier check script measured the same thing through the library-split path and gave about 1.5; this run goes through "
          "the real request tool and the --extra-library file.")


# mode 'prep' builds the files, mode 'score' reads the generated sets; the generator itself is run in between with generate_for_request.py
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['prep', 'score'])
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    prep(a) if a.mode == 'prep' else score(a)
