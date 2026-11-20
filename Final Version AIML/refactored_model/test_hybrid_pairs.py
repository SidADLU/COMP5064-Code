#!/usr/bin/env python3
"""
test_hybrid_pairs.py -- does the generator work for a (disease, target) request when NOTHING is known about that exact pair?
(Needs torch + RDKit; run it on your machine.  Uses the checkpoint's own held-out targets, so the model never saw them.)

THE SITUATION BEING TESTED
  A user asks for disease D at target T.  The library knows
      * target T's ligands that were found for OTHER diseases   (the 'target seeds'),
      * disease D's ligands found at OTHER targets              (the 'disease seeds'),
  but NO ligand of the pair (D, T) itself.  That is exactly what 'a new disease paired with a known target' looks like.
  The ligands of the real pair (D, T) are kept secret as the TEST set.  Four ways of choosing seeds are compared:
      target   seeds from the target's ligands only
      disease  seeds from the disease's ligands only   (these come from other targets, so this isolates DISEASE conditioning)
      mixed    seeds alternating between the two       (what generate_for_request.py does when the pair has no ligands)
      oracle   seeds from the pair's own library ligands (NOT available in the situation; the best case to compare against)
  and two ways of letting the disease STEER the target's chemistry instead of supplying seeds of its own:
      steered  target seeds only, but only the half of them that look most like the disease's ligands (the target's sub-series that
               resembles what treats this disease elsewhere)
      bridge   disease seeds only, but only the quarter of them that look most like the target's ligands (disease chemistry that fits the target)

TWO LIFTS PER MODE  (1.00 = no better than samples made for other cases; the same column-normalised idea as the other checks)
  vs other targets :  samples for this case scored against this case's test ligands, divided by what OTHER TARGETS' samples score
                      on the same test ligands.  Asks: do the molecules resemble this pair's real ligands at all?
  vs other diseases:  the same, but divided by what OTHER DISEASES AT THE SAME TARGET score.  Asks: are the molecules specific to
                      this DISEASE, not just to the target?  (only for targets with a second disease)

USAGE
  python test_hybrid_pairs.py --checkpoint checkpoint_exemplar_seed_v2.pt
  options: --n 30 (samples per case and mode)  --temperature 0.7  --min-disease-seeds 8  --min-target-seeds 5  --seed 0
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exemplar_data as xd

import config as cfg

DEFAULT_CSV = cfg.DEFAULT_CSV
# the six ways of choosing seeds that are compared (described in the module text); --modes can select a subset
MODES = ('target', 'disease', 'mixed', 'oracle', 'steered', 'bridge')


# Builds one test case for every (held-out target, disease) pair that has test ligands.  Each case holds the target's seeds (its ligands with the pair's own removed),
# the disease's seeds (its ligands at other targets), the pair's library ligands (for the oracle mode) and the secret test ligands.  Returns (cases, skipped), where
# skipped lists (target, disease, reason) for cases with too few seeds.
def build_cases(ho, lib, min_disease_seeds, min_target_seeds):
    cases, skipped = [], []
    for t, s in sorted(ho['sets'].items()):
        for d in sorted(s['lib_by_d']):
            pair = sorted(s['lib_by_d'][d])
            pset = set(pair)
            tseeds = [k for k in s['lib'] if k not in pset]
            dseeds = list(lib.ok_pool('d', d))
            if len(tseeds) < min_target_seeds:
                skipped.append((t, d, f"only {len(tseeds)} target seeds once the pair's own ligands are removed"))
            elif len(dseeds) < min_disease_seeds:
                skipped.append((t, d, f"only {len(dseeds)} ligands of this disease at other targets"))
            else:
                cases.append({'t': t, 'd': d, 'tseeds': tseeds, 'dseeds': dseeds, 'pair': pair, 'test': sorted(s['test_by_d'][d])})
    return cases, skipped


# Endless generator of (seed, role) pairs.  It alternates between the given pools and walks each pool in shuffled cycles, so every seed of a pool is used once
# before any is repeated.
def seed_stream(rng, sources):
    cyc = [[] for _ in sources]
    i = 0
    while True:
        j = i % len(sources)
        if not cyc[j]:
            cyc[j] = list(sources[j][0])
            rng.shuffle(cyc[j])
        yield cyc[j].pop(), sources[j][1]
        i += 1


# Returns the [(pool, role), ...] list a case uses in the given mode (the target's, the disease's, both, the pair's own, or the steered / bridge sub-pools).
def mode_sources(case, mode):
    if mode == 'target':
        return [(case['tseeds'], xd.ROLE_TARGET)]
    if mode == 'disease':
        return [(case['dseeds'], xd.ROLE_DISEASE)]
    if mode == 'mixed':
        return [(case['tseeds'], xd.ROLE_TARGET), (case['dseeds'], xd.ROLE_DISEASE)]
    if mode == 'steered':
        return [(case['tsteer'], xd.ROLE_TARGET)]
    if mode == 'bridge':
        return [(case['dnear'], xd.ROLE_DISEASE)]
    return [(case['pair'], xd.ROLE_PAIR)]


# Sorts the pool molecules by their best Tanimoto similarity to any molecule of the reference set, most similar first.  Molecules that cannot be fingerprinted
# (or an empty reference set) score 0.
def rank_by_similarity(chem, pool, ref):
    rf = chem.fps(ref)
    sc = []
    for k in pool:
        f = chem.fp(k)
        sc.append((max(chem.DataStructs.BulkTanimotoSimilarity(f, rf)) if (f is not None and rf) else 0.0, k))
    sc.sort(key=lambda x: -x[0])
    return [k for _, k in sc]


# Adds the two sub-pools used by the steered and bridge modes to every case: 'tsteer', the target seeds most similar to the disease's ligands (t_frac of them),
# and 'dnear', the disease seeds most similar to the target's ligands (d_frac of them).  Each keeps at least floor molecules.
def add_steered_pools(chem, cases, t_frac=0.5, d_frac=0.25, floor=3):
    for c in cases:
        rt = rank_by_similarity(chem, c['tseeds'], c['dseeds'])
        rd = rank_by_similarity(chem, c['dseeds'], c['tseeds'])
        c['tsteer'] = rt[:max(floor, int(round(t_frac * len(rt))))]
        c['dnear'] = rd[:max(floor, int(round(d_frac * len(rd))))]


# Turns the score matrix into lifts.  S[i][j] is the score of the samples made for case i against case j's test ligands, so the diagonal is a case scored on its own
# test ligands.  For each case it returns (lift over other targets' samples, lift over the other diseases' samples at the same target); a lift is nan when there is no
# comparison set.
def lifts(S, cases):
    out = []
    for i, c in enumerate(cases):
        diff = [S[r][i] for r in range(len(cases)) if cases[r]['t'] != c['t']]
        same = [S[r][i] for r in range(len(cases)) if r != i and cases[r]['t'] == c['t']]
        a = S[i][i] / (sum(diff) / len(diff)) if diff and sum(diff) > 0 else float('nan')
        b = S[i][i] / (sum(same) / len(same)) if same and sum(same) > 0 else float('nan')
        out.append((a, b))
    return out


# Mean of the values that are not nan (nan if none are).
def mean_ok(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float('nan')


# Runs the test: builds the cases, generates the requested number of novel valid molecules per case and mode, scores them against the secret test ligands, and
# prints the lift table, validity, how many secret ligands were found exactly, and a verdict.
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=cfg.DEFAULT_EXEMPLAR_CKPT)
    p.add_argument('--v6', default=None, help='ignored (kept so old command lines still work): the model code is now imported from base_model.py / base_data.py')
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--min-disease-seeds', type=int, default=8)
    p.add_argument('--min-target-seeds', type=int, default=5)
    p.add_argument('--max-rounds', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--modes', default=','.join(MODES), help='comma list; to save time e.g. target,steered,bridge')
    a = p.parse_args()
    modes = [m for m in a.modes.split(',') if m]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}; choose from {MODES}")

    # heavy imports come after argument parsing so that --help is fast
    import torch
    import exemplar_model as xm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, pp, ckpt = xm.load_exemplar_checkpoint(a.checkpoint, device)
    # the checkpoint records its own held-out targets and the exemplar limits it was trained with
    ho, k_max, ex_max_len, max_list = ckpt['holdout'], ckpt['max_exemplars'], ckpt['ex_max_len'], ckpt['max_list']
    print(f"[checkpoint] {os.path.basename(a.checkpoint)}  (saved step {ckpt.get('step')}, started from {ckpt.get('source')})")
    rows = xm.load_rows(a.csv, pp)
    ok = lambda k: len(k) + 2 <= ex_max_len
    # the library WITHOUT the held-out targets' ligands and rows: these targets are unseen by the model and their ligands must not leak in as seeds
    lib = xd.Library(rows, max_list, exclude=set(ho['target_keys']) | set(ho['row_keys']), ok=ok)
    chem = xm.Chem2()
    arr_fn = xm.make_arr_fn(pp)
    # seeded sampling (seed order and the model's random draws) so a repeat run gives the same numbers
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    cases, skipped = build_cases(ho, lib, a.min_disease_seeds, a.min_target_seeds)
    print(f"[cases] {len(cases)} (disease, unseen target) cases with no pair ligands in the library; {len(skipped)} skipped:")
    for t, d, why in skipped:
        print(f"          skipped {t} / {d}: {why}")
    n_t = len({c['t'] for c in cases})
    if len(cases) < 3 or n_t < 2:
        raise SystemExit("too few cases for a lift: lower --min-disease-seeds / --min-target-seeds.")
    for c in cases:
        print(f"   {c['t']} / {c['d']:<32} target seeds {len(c['tseeds']):3d} | disease seeds {len(c['dseeds']):4d} | pair ligands in library "
              f"(oracle) {len(c['pair']):2d} | secret test ligands {len(c['test']):2d}")

    # adds the steered and bridge sub-pools to every case: the target seeds most like the disease's ligands, and the disease seeds most like the target's
    add_steered_pools(chem, cases)
    # canonical SMILES of the whole library (new molecules must not be in it) and the fingerprints of each case's secret test ligands, computed once
    known_all = {x for x in (chem.canon(r['key']) for r in rows) if x}
    test_fps = [chem.fps(c['test']) for c in cases]
    gens = {m: [] for m in modes}
    valid = {m: [0, 0] for m in modes}
    print(f"\n[generate] {a.n} novel valid molecules per case and mode, temperature {a.temperature} ...")
    for ci, c in enumerate(cases):
        # a re-discovered secret ligand counts as a hit, not as 'known'
        known = known_all - {chem.canon(s) for s in c['test']}
        for m in modes:
            stream = seed_stream(rng, mode_sources(c, m))
            kept, seen = [], set()
            # generation rounds: sample a batch, count validity, keep novel valid molecules until n are collected
            for _ in range(a.max_rounds):
                plan = [next(stream) for _ in range(max(48, 2 * a.n))]
                smis = xm.generate(model, pp, [([s], [r]) for s, r in plan], k_max, ex_max_len, arr_fn, a.temperature)
                for smi in smis:
                    valid[m][1] += 1
                    cn = chem.canon(smi) if smi else None
                    if cn is None:
                        continue
                    valid[m][0] += 1
                    if cn in known or cn in seen:
                        continue
                    seen.add(cn)
                    kept.append(cn)
                    if len(kept) >= a.n:
                        break
                if len(kept) >= a.n:
                    break
            gens[m].append(kept)
        print(f"   case {ci + 1}/{len(cases)} done ({c['t']} / {c['d']}): " + ", ".join(f"{m} {len(gens[m][-1])}" for m in modes))

    # scoring: for each mode, S[i][j] is the mean best similarity of case i's molecules to case j's test ligands; lifts compare the diagonal with the off-diagonal
    res, rediscovered = {}, {m: 0 for m in modes}
    for m in modes:
        gf = [chem.fps(g) for g in gens[m]]
        S = [[chem.mean_best(gf[i], test_fps[j]) for j in range(len(cases))] for i in range(len(cases))]
        res[m] = lifts(S, cases)
        for i, c in enumerate(cases):
            tc = {chem.canon(s) for s in c['test']}
            rediscovered[m] += sum(1 for g in gens[m][i] if g in tc)

    print("\n" + "=" * 118)
    print("HYBRID TEST: lift per case.  'oth-T' = vs other targets (does it resemble this pair's real ligands?),  'oth-D' = vs other diseases at the same target")
    print("=" * 118)
    head = f"  {'case':<44}" + "".join(f"{m:>16}" for m in modes)
    print(head)
    print(f"  {'':<44}" + "".join(f"{'oth-T  oth-D':>16}" for _ in modes))
    for i, c in enumerate(cases):
        cells = ""
        for m in modes:
            x, y = res[m][i]
            cells += f"{x:>9.2f} {('%.2f' % y) if y == y else '  -':>6}"
        print(f"  {(c['t'] + ' / ' + c['d'])[:43]:<44}{cells}")
    print("  " + "-" * 116)
    means = {m: (mean_ok(r[0] for r in res[m]), mean_ok(r[1] for r in res[m])) for m in modes}
    print(f"  {'MEAN':<44}" + "".join(f"{means[m][0]:>9.2f} {means[m][1]:>6.2f}" for m in modes))
    print("\n  validity: " + " | ".join(f"{m} {100 * valid[m][0] / max(valid[m][1], 1):.0f}%" for m in modes))
    print("  secret ligands re-discovered exactly: " + " | ".join(f"{m} {rediscovered[m]}" for m in modes))

    # In how many cases mode x scores higher than mode y (side 0 = lift over other targets, 1 = lift over other diseases); returns (wins, cases compared).  nan cases are skipped.
    def wins(x, y, side):
        pairs = [(res[x][i][side], res[y][i][side]) for i in range(len(cases))]
        pairs = [(u, v) for u, v in pairs if u == u and v == v]
        return sum(1 for u, v in pairs if u > v), len(pairs)

    # Maps a lift to PASS, PARTIAL or FAIL using the two thresholds.
    def grade(x, hi, lo):
        return 'PASS' if x >= hi else 'PARTIAL' if x >= lo else 'FAIL'
    print("\n" + "=" * 118)
    print("VERDICT (thresholds are my judgement)")
    print("=" * 118)
    for m in modes:
        print(f"  {m:<9} resembles the pair's real ligands (oth-T) {means[m][0]:.2f}   specific to the disease (oth-D) {means[m][1]:.2f}")
    if 'mixed' in means:
        print(f"\n  mixed seeds (new disease at a known target): {means['mixed'][0]:.2f}  [{grade(means['mixed'][0], 1.30, 1.10)}]  (PASS >= 1.30, PARTIAL >= 1.10)")
    if 'disease' in means:
        print(f"  disease seeds ALONE (other targets only)     : {means['disease'][0]:.2f}  [{grade(means['disease'][0], 1.20, 1.08)}]  (PASS >= 1.20, PARTIAL >= 1.08)"
              "  <- disease conditioning with no help from the target")
    for x, y in (('mixed', 'target'), ('steered', 'target'), ('bridge', 'disease'), ('bridge', 'target')):
        if x in res and y in res:
            w0, n0 = wins(x, y, 0)
            w1, n1 = wins(x, y, 1)
            print(f"  {x} beats {y}: resembles the pair's ligands in {w0} of {n0} cases; disease-specific in {w1} of {n1} cases")
    print("  How to read it: a mode is only useful if it beats 'target' (the plain baseline).  A disease-specific lift (oth-D) clearly above 1.00 means "
          "the molecules favour THIS disease over the other diseases at the same target; the oracle shows how much is possible when pair ligands exist.")

if __name__ == '__main__':
    main()
