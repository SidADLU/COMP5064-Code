#!/usr/bin/env python3
"""
check_exemplar_conditioning.py -- does the exemplar-conditioned generator actually use the ligands it is shown, and does that
GENERALISE to targets and diseases it never saw in training?   (Needs torch + RDKit; run it on your machine.)

Everything is evaluated on data the model never trained on, taken from the split stored in the checkpoint:
  * held-out WHOLE TARGETS: their 'lib' molecules (what a user would supply for a new target) are the exemplars, their 'test'
    molecules are the yardstick.  Held-out targets with >= 2 populated diseases also give a DISEASE contrast at an unseen target.
  * held-out molecules at seen targets (teacher-forced only).

PART A  teacher-forced: loss with exemplars vs without; swap test at several exemplar counts k.  A swap test asks whether a
        molecule is more likely given the RIGHT target's (or disease's) exemplars than given another one's.
PART B  free-running at unseen targets: validity; column-normalised LIFT matrix (rows = generated for, columns = similarity to
        the test molecules of); lift of simply handing back lib molecules as the reference; copy / near-copy / analog rates;
        lift with exact copies of any known molecule removed.
PART C  free-running disease contrast at unseen multi-disease targets: (similarity to own disease's test molecules) /
        (similarity to the other disease's) for samples made from each disease's lib, vs the same ratio for lib molecules.
PART D  verdict.  (The thresholds are my own judgement, not an established standard.)
Reference points from the ID-conditioned model on the SAME kinds of contexts: target lift 1.00-1.03 on natural triples,
disease lift 1.00-1.03 everywhere.

USAGE
  python check_exemplar_conditioning.py --checkpoint checkpoint_exemplar_seed_v3_holdout.pt --temperature 0.7 --max-targets 17 --k 1
The settings above (one seed per sample, temperature 0.7, all 17 held-out targets) are the ones used for the reported numbers.
Defaults: --n 30 (samples per target), --k 5 (exemplars), --temperature 1.0, --max-targets 12, --seed 0.  Sampling is seeded, so a repeat run gives identical numbers.
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exemplar_data as xd

import config as cfg

DEFAULT_CSV = cfg.DEFAULT_CSV


# Runs the conditioning check on a checkpoint and prints PART A to D (see the module description).  Everything is measured on the split held out in the checkpoint.
# Part A is teacher-forced (no sampling), parts B and C sample molecules at the held-out targets, part D turns the numbers into PASS / PARTIAL / FAIL against fixed thresholds.
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--v6', default=None, help='ignored (kept so old command lines still work): the model code is now imported from base_model.py / base_data.py')
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-targets', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    import numpy as np
    import torch
    import exemplar_model as xm

    # free-running sampling draws from torch's random generator; seeding it makes repeat runs give identical numbers
    torch.manual_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, pp, ckpt = xm.load_exemplar_checkpoint(a.checkpoint, device)
    # the split saved in the checkpoint: held-out targets (with their lib / test molecules), held-out molecules, and the exemplar limits the model was trained with
    ho, k_max, ex_max_len, max_list = ckpt['holdout'], ckpt['max_exemplars'], ckpt['ex_max_len'], ckpt['max_list']
    print(f"[checkpoint] {os.path.basename(a.checkpoint)}  (saved step {ckpt.get('step')}, started from {ckpt.get('source')})")
    rows = xm.load_rows(a.csv, pp)
    # an exemplar must fit in the length the model was trained for
    ok = lambda k: len(k) + 2 <= ex_max_len
    # the library without the held-out targets and molecules, so exemplars never come from data the test uses
    lib = xd.Library(rows, max_list, exclude=set(ho['target_keys']) | set(ho['row_keys']), ok=ok)
    chem = xm.Chem2()
    arr_fn = xm.make_arr_fn(pp)
    PAD = pp.PAD_IDX
    sets = ho['sets']
    print(f"[holdout] {len(sets)} unseen targets, {sum(1 for t in sets if sets[t]['lib_by_d'])} of them with >= 2 populated diseases; "
          f"{len(ho['row_keys'])} unseen molecules at seen targets")
    labels = xd.label_maps(rows, max_list)
    # canonical SMILES of every library molecule, to recognise exact copies of known molecules
    known = {c for c in (chem.canon(r['key']) for r in rows) if c}
    use_amp = device == 'cuda'

    # per-molecule negative log-likelihood of each item under the model
    def nll(items):
        v, _ = xm.sequence_nll(model, items, k_max, ex_max_len, arr_fn, PAD, amp=use_amp)
        return v

    # an item: the molecule, its exemplars, and the role (target / pair) of each exemplar
    def mk(key, ex, role):
        return (key, ex, [role] * len(ex))

    # Compares the true exemplar set with a swapped one: returns the mean loss gap (positive = the true set makes the molecule more likely), the share of molecules for which it does, and n.
    def swap(true_items, swap_items):
        x, y = nll(true_items), nll(swap_items)
        return float(np.mean(y - x)), float(np.mean(x < y)), len(x)

    print("\n" + "=" * 100 + "\nPART A  teacher-forced (held-out molecules; exemplars never include the molecule itself)\n" + "=" * 100)
    # PART A: loss with and without exemplars, and swap tests, on molecules the model never trained on (exemplars never include the molecule itself)
    rng = random.Random(a.seed + 1)
    row_items = xd.row_eval_items(lib, ho['row_keys'], labels, rng, k=a.k, cap=500)
    ti = [it for it in row_items if it['tgt'] and it['tgt_swap']]
    pi = [it for it in row_items if it['pair'] and it['pair_swap']]
    if ti:
        e = nll([(it['key'], [], []) for it in ti])
        t_ = nll([mk(it['key'], it['tgt'][0], xd.ROLE_TARGET) for it in ti])
        g_, acc_, n_ = swap([mk(it['key'], it['tgt'][0], xd.ROLE_TARGET) for it in ti], [mk(it['key'], it['tgt_swap'][0], xd.ROLE_TARGET) for it in ti])
        hard = sum(1 for it in ti if it.get('tgt_swap_hard'))
        print(f"  unseen molecules, seen targets  : loss no exemplars {np.mean(e):.3f} -> target exemplars {np.mean(t_):.3f}  | "
              f"swap-TARGET gap {g_:+.4f} acc {100 * acc_:.0f}% (n={n_}, {hard} against a hard target that shares a disease)")
    if pi:
        g_, acc_, n_ = swap([mk(it['key'], it['pair'][0], xd.ROLE_PAIR) for it in pi], [mk(it['key'], it['pair_swap'][0], xd.ROLE_PAIR) for it in pi])
        print(f"  unseen molecules, seen targets  : swap-DISEASE (same target, other disease's pair pool) gap {g_:+.4f} acc {100 * acc_:.0f}% (n={n_})")
    # swap tests at unseen targets for several numbers of exemplars; the line for k = a.k also feeds the verdict and the loss comparison
    for kk in (1, 3, a.k):
        t_items, d_items = xd.heldout_target_items(sets, random.Random(a.seed + 2), k=kk)
        tt = [it for it in t_items if it['swap']]
        g_, acc_, n_ = swap([mk(it['key'], it['true'][0], xd.ROLE_TARGET) for it in tt], [mk(it['key'], it['swap'][0], xd.ROLE_TARGET) for it in tt])
        line = f"  UNSEEN TARGETS, k={kk}: swap-TARGET gap {g_:+.4f} acc {100 * acc_:.0f}% (n={n_})"
        if d_items:
            gd, ad, nd = swap([mk(it['key'], it['true'][0], xd.ROLE_PAIR) for it in d_items], [mk(it['key'], it['swap'][0], xd.ROLE_PAIR) for it in d_items])
            line += f" | swap-DISEASE gap {gd:+.4f} acc {100 * ad:.0f}% (n={nd})"
        print(line)
        if kk == a.k:
            tf_t_acc, tf_d_acc = acc_, (ad if d_items else float('nan'))
            e = nll([(it['key'], [], []) for it in tt])
            tr = nll([mk(it['key'], it['true'][0], xd.ROLE_TARGET) for it in tt])
            print(f"      loss at unseen targets: no exemplars {np.mean(e):.3f} -> with their lib exemplars {np.mean(tr):.3f}")

    print("\n" + "=" * 100 + "\nPART B  free-running at UNSEEN targets\n" + "=" * 100)
    # PART B: free-running generation at the held-out targets; S[i, j] is the similarity of target i's samples to target j's test molecules
    rng = random.Random(a.seed + 3)
    g = xm.heldout_gen_eval(model, pp, sets, rng, k_max, ex_max_len, arr_fn, chem, k=a.k, n=a.n, max_targets=a.max_targets,
                            temperature=a.temperature, keep=True)
    targets = g['targets']
    if not targets:
        raise SystemExit("No held-out target has enough lib/test molecules.")
    print(f"  {len(targets)} unseen targets, {a.n} samples each, k={a.k} exemplars per sample, temperature {a.temperature}")
    print(f"  validity: {100 * g['valid']:.0f}%   (per target: " + ", ".join(f"{100 * v:.0f}" for v in g['valid_each']) + ")")
    # validity of unconditioned generation (no exemplars), as the reference for the conditioned validity
    unc = xm.generate(model, pp, [([], [])] * a.n, k_max, ex_max_len, arr_fn, a.temperature)
    print(f"  validity with NO exemplars (unconditional): {100 * len(chem.fps(unc)) / max(len(unc), 1):.0f}%")
    # lift = a target's samples vs its own test molecules, divided by what the other targets' samples score on the same molecules (the column mean); 1.00 means no better than other targets
    S = g['S']
    col = np.nanmean(S, axis=0)
    print("\n  LIFT matrix (rows = generated from that target's lib, columns = similarity to that target's TEST molecules; 1.00 = no better than other targets' samples)")
    print("            " + " ".join(f"{t:>8}" for t in targets))
    for i, t in enumerate(targets):
        print(f"  {t:<9} " + " ".join(f"{S[i, j] / col[j]:8.2f}" for j in range(len(targets))))
    print(f"  diagonal mean lift = {g['target_lift']:.3f}     (copying lib molecules instead: {g['ref_target_lift']:.3f};  "
          f"ID-conditioned model on natural targets: 1.00-1.03)")
    print("  per target lift: " + ", ".join(f"{t} {l:.2f}" for t, l in zip(targets, g['target_lift_each'])))

    # copy and novelty statistics: how many valid samples are exact copies of an exemplar or of any library molecule, and how many are near-copies, analogs or far from their exemplars
    ex_exact = known_exact = near = analog = far = valid_n = 0
    Snov = np.full((len(targets), len(targets)), np.nan)
    test_fps = {t: chem.fps(sets[t]['test'][:150]) for t in targets}
    n_nov = []
    for i, t in enumerate(targets):
        smis = g['samples'][t]['smiles']
        exs = g['samples'][t]['exemplars']
        # samples that are not exact copies of a known molecule; used for the novel-only lift
        keep = []
        for s_, ex in zip(smis, exs):
            c = chem.canon(s_)
            if c is None:
                continue
            valid_n += 1
            ex_c = {chem.canon(e) for e in ex}
            if c in ex_c:
                ex_exact += 1
            if c in known:
                known_exact += 1
            else:
                keep.append(s_)
            best = max([chem.sim(s_, e) for e in ex] + [0.0])
            if c in ex_c:
                pass
            elif best >= 0.8:
                near += 1
            elif best >= 0.4:
                analog += 1
            else:
                far += 1
        n_nov.append(len(keep))
        fps = chem.fps(keep)
        for j in range(len(targets)):
            Snov[i, j] = chem.mean_best(fps, test_fps[targets[j]])
    vn = max(valid_n, 1)
    print(f"\n  of {valid_n} valid samples: exact copy of one of its own exemplars {100 * ex_exact / vn:.0f}% | "
          f"exact copy of ANY known molecule {100 * known_exact / vn:.0f}% | near-copy of an exemplar (0.80-0.99) {100 * near / vn:.0f}% | "
          f"analog range (0.40-0.80) {100 * analog / vn:.0f}% | far (<0.40) {100 * far / vn:.0f}%")
    lift_nov, _ = xm.column_lift(Snov)
    print(f"  NOVEL-ONLY (exact copies of any known molecule removed; kept per target: {n_nov}): diagonal mean lift {lift_nov:.3f}")

    print("\n" + "=" * 100 + "\nPART C  disease contrast at UNSEEN multi-disease targets (same target, two diseases' lib exemplars)\n" + "=" * 100)
    # PART C: for each held-out target with two diseases, similarity of samples made from one disease's lib to that disease's test molecules relative to the other disease's
    if g['dis_cases']:
        for c in g['dis_cases']:
            print(f"  {c['target']}: {c['d1'][:34]:<34} ({c['n1']} lib) vs {c['d2'][:34]:<34} ({c['n2']} lib):  "
                  f"lift {c['lift']:.2f}   (lib molecules copied: {c['ref_lift']:.2f})")
        print(f"  mean disease lift {g['disease_lift']:.3f}   (copying lib molecules: {g['ref_disease_lift']:.3f};  ID-conditioned model: 1.00-1.03)")
    else:
        print("  no held-out target has two diseases with enough lib/test molecules at this sample size.")

    print("\n" + "=" * 100 + "\nPART D  verdict (thresholds are my judgement)\n" + "=" * 100)

    # PASS at or above hi, PARTIAL at or above mid, otherwise FAIL
    def grade(v, hi, mid):
        return 'PASS' if v >= hi else ('PARTIAL' if v >= mid else 'FAIL')
    dl = g.get('disease_lift', float('nan'))
    print(f"  target conditioning at UNSEEN targets : free-running lift {g['target_lift']:.2f}  [{grade(g['target_lift'], 1.30, 1.10)}]  "
          f"(PASS >= 1.30, PARTIAL >= 1.10; novel-only {lift_nov:.2f});  teacher-forced swap acc {100 * tf_t_acc:.0f}%")
    print(f"  disease conditioning at unseen targets: free-running lift {dl:.2f}  "
          f"[{grade(dl, 1.20, 1.08) if dl == dl else 'n/a'}]  (PASS >= 1.20, PARTIAL >= 1.08);  teacher-forced swap acc {100 * tf_d_acc:.0f}%")
    print(f"  validity {100 * g['valid']:.0f}%  [{grade(g['valid'], 0.85, 0.65)}]   exact copies of exemplars {100 * ex_exact / vn:.0f}%  "
          f"[{'PASS' if ex_exact / vn <= 0.20 else 'HIGH'}]")
    # how much of the possible lift above 1.00 (the retrieval ceiling) the generator reaches
    def share(v, ref):
        return f"{100 * (v - 1) / (ref - 1):.0f}% of it" if (ref == ref and ref > 1.0 and v == v) else 'n/a'
    print(f"  against the retrieval ceiling (copying lib molecules scores {g['ref_target_lift']:.2f} on target, "
          f"{g.get('ref_disease_lift', float('nan')):.2f} on disease): generated molecules reach {share(g['target_lift'], g['ref_target_lift'])} "
          f"of the target lift above 1.00 and {share(dl, g.get('ref_disease_lift', float('nan')))} of the disease lift above 1.00")
    print("  Read lift against the 'copying lib molecules' reference: a generator that only returns its exemplars scores that number "
          "with no novelty at all; the useful region is lift above 1.1 with a low copy rate.")


if __name__ == '__main__':
    main()
