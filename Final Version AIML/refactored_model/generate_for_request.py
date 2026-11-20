#!/usr/bin/env python3
"""
generate_for_request.py -- ask the trained exemplar (seed-and-edit) generator for molecules for ONE (disease, target) request.
(Needs torch + RDKit; run it on your machine.)

HOW A REQUEST IS ANSWERED
  The model has no disease/target ID tables.  It is shown ONE known ligand (a 'seed') per sample and returns an analog of it, so what
  a request means is decided by WHERE THE SEEDS COME FROM -- the ligand library (the CleanedData.csv, plus an optional extra file):
    * the (disease, target) pair has >= --min-pair ligands  -> seeds from that pair's ligands                    (source 'pair')
    * otherwise (few or no ligands for the exact pair)       -> seeds from the target's ligands (plus the pair's few, if any)
                                                                ('target' / 'pair').  The disease's ligands at OTHER targets are NOT
                                                                used: test_hybrid_pairs.py showed they carry no usable signal for a
                                                                pair without its own ligands (lift 1.05) and dilute the target seeds
                                                                (mixed 1.15 vs target-only 1.23).  --use-disease-seeds brings them back.
    * only a target given                                    -> seeds from the target's ligands
    * only a disease given                                   -> seeds from the disease's ligands (weak: close to unconditioned)
  A NEW disease or target works as soon as its ligands are in the library: put them in an extra CSV (--extra-library) with the same
  three columns as CleanedData.csv ('Disease Entry', 'Targets for Drug', 'Canonical Smiles'; list cells look like "['Coronavirus']").
  No retraining is involved.

WHAT YOU GET
  A CSV of valid, UNIQUE, NOVEL molecules (not in the library), each with its seed, the seed's source pool, its similarity to the seed
  and to the closest library molecule.  Invalid and duplicate samples are generated, counted and dropped.

USAGE
  find names :  python generate_for_request.py --find asthma
  a request  :  python generate_for_request.py --disease "Asthma" --target T52522 --n 50 --out asthma_T52522.csv
  new context:  python generate_for_request.py --extra-library my_new_ligands.csv --disease Coronavirus --target Tx12345 --n 50
Defaults: --checkpoint checkpoint_exemplar_seed_v2.pt  --temperature 0.7  (lower = closer to the seeds, more valid, more copies;
          1.0 = more varied and less valid -- the check script measured both)

BASELINE (control for docking): --random-seeds draws every seed from the usable ligands of the whole library instead of from the request's
  ligands.  Same model, same filters, same temperature; only the seed choice differs, so a comparison of this set with the target-conditioned
  set isolates what the conditioning contributes.  --target (optional) is used ONLY to EXCLUDE that target's ligands from the seed pool;
  --exclude-seed-targets adds more target codes to exclude (e.g. the other DHFR codes for T89515, NET and DAT for SERT).
  python generate_for_request.py --random-seeds --target T89515 --exclude-seed-targets T83386,T17345,T19229,T35465,T52227,T02702,T47077 --n 70 --out gen_malaria_t89515_baseline.csv
NOTE: checkpoint_exemplar_seed_v2.pt was trained on almost all targets of the library; 16 targets (T52522, T45993, ...) were HELD OUT, and its conditioning
      numbers were measured on those only.  Any other target, including the docking study targets, was in its training data.  To generate for the study
      targets as UNSEEN targets use --checkpoint checkpoint_exemplar_seed_v3_holdout.pt, which holds them out as well.
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exemplar_data as xd

import config as cfg

# default ligand library and model checkpoint (paths come from config.py)
DEFAULT_CSV, DEFAULT_CKPT = cfg.DEFAULT_CSV, cfg.DEFAULT_EXEMPLAR_CKPT


# Looks a disease or target name up among the library's names: an exact match first, then a case-insensitive one.  Returns (match, []) on success, or (None, up to 12
# names that contain the text) so the caller can show close names.  The kind argument is not used.
def resolve(name, pool_keys, kind):
    if name in pool_keys:
        return name, []
    low = {str(k).lower(): k for k in pool_keys}
    if name.lower() in low:
        return low[name.lower()], []
    sugg = sorted(k for k in pool_keys if name.lower() in str(k).lower())[:12]
    return None, sugg


# Decides which ligands act as seeds for a request.  Returns (sources, notes): sources is a list of (name, ligand pool, role) and notes are messages to print.
# A pair with at least min_pair ligands is used alone; otherwise the target's ligands (plus the pair's few) are used.  The disease's ligands at other targets are
# used only when there is no target pool or use_disease_seeds is set, because in testing they did not help.
def build_sources(pair, tp, dp, min_pair, use_disease_seeds=False):
    sources, notes = [], []
    if len(pair) >= min_pair:
        return [('pair', pair, xd.ROLE_PAIR)], notes
    if pair:
        sources.append(('pair', pair, xd.ROLE_PAIR))
    if tp:
        sources.append(('target', tp, xd.ROLE_TARGET))
    if dp and (use_disease_seeds or not tp):
        sources.append(('disease', dp, xd.ROLE_DISEASE))
        if not tp:
            notes.append("no target ligands are known: the seeds come from the disease's ligands at other targets, which in testing were barely "
                         "better than unconditioned output (lift about 1.05).  Add ligands of the target (or of the pair) for a real fit.")
        else:
            notes.append("--use-disease-seeds: disease seeds are mixed in; in testing this lowered the match to the pair's real ligands.")
    elif dp and tp:
        notes.append("the disease's ligands at other targets are not used as seeds (in testing they did not help once the target's ligands "
                     "are available); add ligands of the exact pair to make the disease count.")
    return sources, notes


# Answers one request: chooses the seed pools, then asks the model for an analog of one seed per sample, round after round, until n novel valid molecules
# are collected (or max_rounds is reached), and writes them with their seed and similarities to a CSV.  With --random-seeds the seeds come from the whole
# library instead (the baseline control); with --find it only lists matching diseases and targets and stops.
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=DEFAULT_CKPT)
    p.add_argument('--v6', default=None, help='ignored (kept so old command lines still work): the model code is now imported from base_model.py / base_data.py')
    p.add_argument('--csv', default=DEFAULT_CSV, help='the ligand library')
    p.add_argument('--extra-library', default=None, help='optional CSV with the same columns: ligands of NEW diseases/targets (or more of old ones)')
    p.add_argument('--disease', default=None)
    p.add_argument('--target', default=None)
    p.add_argument('--n', type=int, default=50, help='number of novel valid molecules wanted')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--blend-disease', action='store_true', help='also use the disease\'s ligands (from other targets) as seeds when the pair has too few ligands (measured to lower the result)')
    p.add_argument('--min-pair', type=int, default=3, help='pair pool must hold at least this many ligands to be used alone')
    p.add_argument('--use-disease-seeds', action='store_true', help="also seed from the disease's ligands at other targets (tested: lowers the match)")
    p.add_argument('--max-rounds', type=int, default=12)
    p.add_argument('--min-seed-sim', type=float, default=0.0, help='drop samples less similar than this to their seed')
    p.add_argument('--out', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--find', default=None, help='list diseases and targets whose name contains this text, with their ligand counts, and exit')
    p.add_argument('--random-seeds', action='store_true',
                   help='baseline control: ignore --disease and draw seeds uniformly from the usable ligands of the whole library. --target, if given, is only used to '
                        'EXCLUDE that target\'s ligands from the seed pool')
    p.add_argument('--exclude-seed-targets', default='', help='with --random-seeds: comma list of further target codes whose ligands are not used as seeds')
    a = p.parse_args()

    # heavy imports come after argument parsing so that --help is fast
    import torch
    import exemplar_model as xm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, pp, ckpt = xm.load_exemplar_checkpoint(a.checkpoint, device)
    # limits the checkpoint was trained with: exemplars per sample, and the longest exemplar it can be shown
    k_max, ex_max_len, max_list = ckpt['max_exemplars'], ckpt['ex_max_len'], ckpt['max_list']
    rows = xm.load_rows(a.csv, pp)
    if a.extra_library:
        extra = xm.load_rows(a.extra_library, pp)
        print(f"[library] + {len(extra)} ligands from {a.extra_library}")
        rows = rows + extra
    # a library molecule can be a seed only if it fits the exemplar length limit (its tokens plus the two end markers)
    ok = lambda k: len(k) + 2 <= ex_max_len
    lib = xd.Library(rows, max_list, ok=ok)
    print(f"[library] {lib.summary()}")

    # --find: list the diseases and targets whose name contains the text, with their numbers of usable ligands, and stop
    if a.find:
        f = a.find.lower()
        for kind, title in (('d', 'DISEASES'), ('t', 'TARGETS')):
            hits = sorted(((lab, len(v)) for lab, v in lib.pool_ok[kind].items() if f in str(lab).lower()), key=lambda x: -x[1])[:40]
            print(f"\n{title} containing '{a.find}' (usable ligands):")
            for lab, n in hits:
                print(f"  {n:5d}  {lab}")
            if not hits:
                print("  (none)")
        return

    # baseline mode: the seeds come from the whole library, minus the ligands of the excluded targets
    if a.random_seeds:
        excl_codes = [x.strip() for x in ([a.target] if a.target else []) + a.exclude_seed_targets.split(',') if x and x.strip()]
        excl = set()
        for code in excl_codes:
            hit, _ = resolve(code, lib.pool_all['t'].keys(), 'target')
            if hit is None:
                print(f"[warn] target {code} has no ligands in the library: nothing to exclude for it")
            else:
                excl |= set(lib.pool_all['t'][hit])
        all_smiles = sorted({r['key'] for r in rows if ok(r['key'])} - excl)
        if not all_smiles:
            raise SystemExit("random-seed pool is empty")
        sources = [('random_library', all_smiles, xd.ROLE_TARGET)]
        print(f"[request] RANDOM-SEED BASELINE: seeds drawn from the library's usable ligands (no disease/target conditioning); excluded {len(excl)} ligands of {excl_codes}")
        for name, pool, _ in sources:
            print(f"[seeds]   {name:<20} pool: {len(pool)} ligands")
        notes = []
        d = t = None
    else:
        # resolve the requested disease and target names against the library; a miss stops with close-name suggestions
        if not a.disease and not a.target:
            raise SystemExit("Give --disease and/or --target (use --find TEXT to look up names).")
        d, t = None, None
        if a.disease:
            d, sugg = resolve(a.disease, lib.pool_ok['d'].keys(), 'disease')
            if d is None:
                raise SystemExit(f"Disease '{a.disease}' has no usable ligands in the library." +
                                 (f"  Close names: {sugg}" if sugg else "") +
                                 "  To add a new disease, put its ligands in a CSV and pass --extra-library.")
        if a.target:
            t, sugg = resolve(a.target, lib.pool_ok['t'].keys(), 'target')
            if t is None:
                raise SystemExit(f"Target '{a.target}' has no usable ligands in the library." +
                                 (f"  Close names: {sugg}" if sugg else "") +
                                 "  To add a new target, put its ligands in a CSV and pass --extra-library.")

    # seed pools for the request: the pair's ligands, the target's and the disease's; build_sources chooses among them
    if not a.random_seeds:
        pair = lib.ok_pool('dt', (d, t)) if (d and t) else []
        tp = lib.ok_pool('t', t) if t else []
        dp = lib.ok_pool('d', d) if d else []
        sources, notes = build_sources(pair, tp, dp, a.min_pair, a.use_disease_seeds)
        if not sources:
            raise SystemExit("No ligands for this request: add some with --extra-library.")
        print("[request] " + (f"disease '{d}'" if d else "no disease") + " + " + (f"target '{t}'" if t else "no target"))
        for name, pool, _ in sources:
            print(f"[seeds]   {name:<8} pool: {len(pool)} ligands")
        for n_ in notes:
            print(f"[note] {n_}")

    # one seeded random generator for the seed order and one for the model's sampling, so a repeat run gives the same file
    chem = xm.Chem2()
    arr_fn = xm.make_arr_fn(pp)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    # fingerprints of the library: used to reject molecules that are already known and to report each molecule's nearest library neighbour
    print("[library] fingerprinting the library (one-off, ~10-30 s) ...")
    known_can = {c for c in (chem.canon(r['key']) for r in rows) if c}
    lib_keys = sorted({r['key'] for r in rows})
    lib_fps = [chem.fp(k) for k in lib_keys]
    lib_keys = [k for k, f in zip(lib_keys, lib_fps) if f is not None]
    lib_fps = [f for f in lib_fps if f is not None]

    # seeds are taken from each pool in shuffled cycles, and the pools alternate
    cyc = {name: [] for name, _, _ in sources}

    def next_seed(i):
        name, pool, role = sources[i % len(sources)]
        if not cyc[name]:
            cyc[name] = list(pool)
            rng.shuffle(cyc[name])
        return name, cyc[name].pop(), role

    # generation loop: sample a batch, keep the molecules that are valid, new (not in the library or already kept) and, if requested, close enough to their seed
    out_rows, seen = [], set()
    stats = {'generated': 0, 'invalid': 0, 'known_or_duplicate': 0, 'too_far_from_seed': 0}
    batch = max(64, 3 * a.n)
    i = 0
    for rnd in range(a.max_rounds):
        plan = []
        for _ in range(batch):
            plan.append(next_seed(i))
            i += 1
        sets = [([seed], [role]) for _, seed, role in plan]
        smis = xm.generate(model, pp, sets, k_max, ex_max_len, arr_fn, a.temperature)
        for (src, seed, _), smi in zip(plan, smis):
            stats['generated'] += 1
            c = chem.canon(smi) if smi else None
            if c is None:
                stats['invalid'] += 1
                continue
            if c in known_can or c in seen:
                stats['known_or_duplicate'] += 1
                continue
            s_seed = chem.sim(c, seed)
            if s_seed < a.min_seed_sim:
                stats['too_far_from_seed'] += 1
                continue
            fp = chem.fp(c)
            near = max(chem.DataStructs.BulkTanimotoSimilarity(fp, lib_fps)) if fp is not None else float('nan')
            seen.add(c)
            out_rows.append({'smiles': c, 'seed_smiles': seed, 'seed_source': src,
                             'similarity_to_seed': round(s_seed, 3), 'similarity_to_nearest_library_molecule': round(near, 3)})
            if len(out_rows) >= a.n:
                break
        print(f"  round {rnd + 1}: {len(out_rows)}/{a.n} kept  ({stats})")
        if len(out_rows) >= a.n:
            break

    # output file: the --out name, or one built from the disease and target names
    name = a.out or (("request_random_baseline.csv") if a.random_seeds else
                    ("request_" + "_".join(x.replace(' ', '-')[:24] for x in (d, t) if x) + ".csv"))
    with open(name, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['smiles', 'seed_smiles', 'seed_source', 'similarity_to_seed', 'similarity_to_nearest_library_molecule'])
        w.writeheader()
        w.writerows(out_rows)
    n_gen = max(stats['generated'], 1)
    print(f"\n[done] {len(out_rows)} novel valid molecules written to {name}")
    print(f"       validity {100 * (1 - stats['invalid'] / n_gen):.0f}% | rejected as known or duplicate {stats['known_or_duplicate']} | "
          f"temperature {a.temperature}")
    if out_rows:
        import numpy as np
        ss = np.array([r['similarity_to_seed'] for r in out_rows])
        print(f"       similarity to seed: median {np.median(ss):.2f} (analog range 0.40-0.80: {100 * np.mean((ss >= 0.4) & (ss <= 0.8)):.0f}%)")
    if len(out_rows) < a.n:
        print(f"[warn] only {len(out_rows)} of {a.n} found in {a.max_rounds} rounds: the seed pool may be small -- add ligands or raise --temperature.")


if __name__ == '__main__':
    main()
