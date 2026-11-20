#!/usr/bin/env python3
"""
rule_gen_check.py -- what is behind a generated-set result obtained with an interaction rule?  (no docking: reads dock_cache_*.jsonl)

For the generated set, the random-seed baseline and the seeds, at one receptor docked with --interaction, it reports
  * the share of molecules with a basic nitrogen, and the share that satisfy the rule (basic N within the cutoff of the rule's receptor atoms in >= 1 pose)
  * the plain Vina score (best pose, no rule) next to the rule score
  * conditioned vs baseline: P(conditioned scores better) with the rule score, with the plain score, and within the molecules that have a basic N
  * analog vs its seed: how often the seed satisfies the rule and the analog keeps it, and the mean plain-score difference (no rule)
The point: if the rule-based advantage of the conditioned set is only 'its seeds carry the amine and the analogs kept it', that is a pharmacophore result,
not evidence of better binding; the plain-score rows show what Vina alone says.

  python rule_gen_check.py --cache dock_cache_inter.jsonl --label DPP4-2ONC --gen gen_diabetes_ho.csv --baseline gen_diabetes_baseline_ho.csv
"""
import argparse
import csv
import json
import sys

import numpy as np


# Canonical SMILES of the largest fragment, the form used as the key in the docking cache.  Returns None for an unparsable SMILES; with use_rdkit False the string is returned as is (test mode).
def canon(smi, use_rdkit=True):
    if not use_rdkit:
        return smi
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if len(frags) > 1:
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(m)


# Reads a dock_cache jsonl and returns {canonical smiles: record} for one receptor label.  Only dockings at exhaustiveness 8 and Vina seed 42 that carry interaction-rule details are kept;
# each record has the rule score ('score'), the plain Vina best score ('plain'), whether the molecule has a basic nitrogen and how many poses satisfy the rule ('n_ok').
def load_cache(path, label):
    d = {}
    for line in open(path, encoding='utf-8'):
        try:
            r = json.loads(line)
            # cache key = receptor label | receptor signature | exhaustiveness | Vina seed | smiles
            lab, sig, exh, vs, smi = r['key'].split('|', 4)
        except Exception:
            continue
        if lab != label or exh != '8' or vs != '42' or r.get('score') is None or not r.get('extra') or 'plain' not in r['extra']:
            continue
        e = r['extra']
        d[smi] = {'score': float(r['score']), 'plain': float(e['plain']), 'has_basic': bool(e['has_basic']), 'n_ok': int(e['n_ok'])}
    return d


# Matches the molecules of a generated-set CSV to their cache records (and each molecule's seed, if it was docked too).  Returns the matched rows and the number of molecules not found in the cache.
def read_set(path, cache, use_rdkit):
    rows, miss = [], 0
    for r in csv.DictReader(open(path, encoding='utf-8')):
        c, s = canon(r.get('smiles'), use_rdkit), canon(r.get('seed_smiles', ''), use_rdkit)
        if c in cache:
            rows.append({'smi': c, 'rec': cache[c], 'seed': cache.get(s)})
        else:
            miss += 1
    return rows, miss


# Probability that a random member of a scores better (lower) than a random member of b, ties counting half; 0.5 means no difference.  NaN if either list is empty.
def p_better(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    return float(((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


# Prints one summary line for a set: size, share with a basic nitrogen, share satisfying the rule, and the median plain and rule scores.
def describe(name, rows):
    if not rows:
        print(f"   {name:<12} none found in the cache")
        return
    rec = [r['rec'] for r in rows]
    print(f"   {name:<12} n {len(rows):>3}   basic N {np.mean([x['has_basic'] for x in rec]):4.0%}   satisfy rule {np.mean([x['n_ok'] > 0 for x in rec]):4.0%}"
          f"   median plain score {np.median([x['plain'] for x in rec]):6.2f}   median rule score {np.median([x['score'] for x in rec]):6.2f}")


# Compares the conditioned set, the random-seed baseline and the seeds under the interaction rule, using only docking results already in the cache (no docking is run).
# It separates what the rule score rewards (an amine / Glu contact inherited from the seeds) from what plain Vina sees (overall fit), and checks whether analogs keep the pharmacophore of their seed.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--label', required=True, help='receptor label as printed by dock_generated.py (e.g. DPP4-2ONC)')
    ap.add_argument('--gen', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--no-canon', action='store_true', help='test mode: skip RDKit canonicalisation')
    a = ap.parse_args()
    use = not a.no_canon
    cache = load_cache(a.cache, a.label)
    print(f"[cache] {len(cache)} dockings with rule details at {a.label}")
    g, mg = read_set(a.gen, cache, use)
    b, mb = read_set(a.baseline, cache, use)
    print(f"[sets] generated {len(g)} found ({mg} not in the cache: failed / skipped); baseline {len(b)} found ({mb} not in the cache)")
    # each seed once, even if several analogs share it
    seeds = {}
    for r in g:
        if r['seed'] is not None:
            seeds[id(r['seed'])] = r['seed']
    seed_rows = [{'rec': s} for s in seeds.values()]
    print("\n1. WHAT EACH SET LOOKS LIKE UNDER THE RULE")
    describe('generated', g)
    describe('baseline', b)
    describe('seeds', seed_rows)
    # rule scores and plain Vina scores of the two sets
    gs, bs = [r['rec']['score'] for r in g], [r['rec']['score'] for r in b]
    gp, bp = [r['rec']['plain'] for r in g], [r['rec']['plain'] for r in b]
    print("\n2. CONDITIONED vs BASELINE (raw scores, not size-corrected; 0.5 = no difference; the size-corrected figure is in the main report)")
    print(f"   rule score : P(conditioned better) = {p_better(gs, bs):.2f}")
    print(f"   plain Vina : P(conditioned better) = {p_better(gp, bp):.2f}   (no rule at all)")
    # the same comparison restricted to molecules with a basic nitrogen, then to molecules that satisfy the rule, to see whether the plain-Vina result depends on the amine
    gb = [r['rec']['plain'] for r in g if r['rec']['has_basic']]
    bb = [r['rec']['plain'] for r in b if r['rec']['has_basic']]
    print(f"   plain Vina, only molecules with a basic N (n {len(gb)} vs {len(bb)}): {p_better(gb, bb):.2f}")
    gk = [r['rec']['plain'] for r in g if r['rec']['n_ok'] > 0]
    bk = [r['rec']['plain'] for r in b if r['rec']['n_ok'] > 0]
    print(f"   plain Vina, only molecules that satisfy the rule (n {len(gk)} vs {len(bk)}): {p_better(gk, bk):.2f}")
    # analog vs its own seed: does the analog keep the rule, and does it dock better or worse than the seed
    pairs = [r for r in g if r['seed'] is not None]
    print(f"\n3. ANALOG vs ITS SEED (n {len(pairs)} pairs with a docked seed)")
    if pairs:
        sat_seed = [r for r in pairs if r['seed']['n_ok'] > 0]
        kept = [r for r in sat_seed if r['rec']['n_ok'] > 0]
        print(f"   seeds satisfying the rule: {len(sat_seed)} of {len(pairs)};  of those, analogs that keep it: {len(kept)} ({len(kept) / max(1, len(sat_seed)):.0%})")
        dp = np.array([r['rec']['plain'] - r['seed']['plain'] for r in pairs])
        dr = np.array([r['rec']['score'] - r['seed']['score'] for r in pairs])
        print(f"   mean score change analog - seed (kcal/mol, + = analog worse):  plain Vina {dp.mean():+.2f}   rule score {dr.mean():+.2f}   (plain: analog better in {np.mean(dp < 0):.0%})")
    print("\nReading: if the rule score favours the conditioned set but the plain-Vina rows do not, the advantage is the amine / Glu contact that the seeds carry,"
          "\nnot a better fit overall.  If analogs mostly lose the rule relative to their seeds, the model is dropping the pharmacophore.")


if __name__ == '__main__':
    sys.exit(main())
