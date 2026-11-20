#!/usr/bin/env python3
"""
train_exemplar_generator.py -- train the EXEMPLAR-CONDITIONED generator (see exemplar_model.py / exemplar_data.py).

WHY
  The ID-embedding generator cannot be conditioned on a disease or target it has seen little of (median disease: 2 rows) or
  not at all.  Here the condition is a handful of KNOWN LIGANDS of the disease / target / (disease, target) pair, shown to the
  model as SMILES.  Anything with known ligands in the library can then be used, including contexts added after training.

WHAT IT TRAINS ON  (each step: a batch of episodes)
  pick a molecule x; pick exemplars from the pool of its target / disease / (disease, target) pair / a mix of the target and
  disease pools (never x itself, optionally never a near-duplicate of x); train the model to generate x from them.
  Small groups are sampled far more often than their share of rows (--balance-power).  A few episodes have no exemplars at all.
  --arch lm (default): ONE causal transformer reads  [BOS ex1 EOS][BOS ex2 EOS]...[BOS molecule EOS]  and is trained to continue it
  (loss on the molecule only).  Its layers start from the pretrained decoder's self-attention/feed-forward blocks (general SMILES
  grammar is kept) plus --extra-layers fresh ones.  --arch xattn is an encoder + cross-attention alternative.
  --seed-mode (SEED-AND-EDIT, used for the shipped generator): every episode shows ONE exemplar, a near neighbour of the molecule, drawn from its pair / target / disease pool;
  molecules with no near neighbour are skipped.  Use it with --eval-k 1; it saves the final step, because the best-validation-loss rule scores random exemplars.

WHAT IT HOLDS OUT  (never trained on, not even as an exemplar)
  * WHOLE TARGETS (--holdout-targets + random ones, some with >= 2 populated diseases).  Their molecules are split into 'lib'
    (the ligands a user would supply for a new target) and 'test'.  This is the "unseen target" test, and at the multi-disease
    ones also a disease contrast at an unseen target.
  * a random share of the remaining molecules (--row-holdout): unseen molecules at seen targets and diseases.
  Reported (teacher-forced, every --eval-every steps): loss with exemplars vs none, and swap tests (the true exemplar set must
  make the molecule more likely than another target's / another disease's exemplar set).  Reported (free-running, every
  --gen-every steps, needs RDKit): validity and column-normalised lift on the held-out targets, and the disease lift.

USAGE (folder with config.py, base_model.py, base_data.py, exemplar_data.py and exemplar_model.py; the CSV is found by config.py)
  python train_exemplar_generator.py --checkpoint checkpoint_pretrained_v11.pt --out checkpoint_exemplar_v1.pt
  quick code-path test:  add  --steps 30 --eval-every 15 --gen-every 15 --gen-n 4 --gen-targets 2
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exemplar_data as xd

import config as cfg

# defaults from config.py: the ligand library and the pretrained SMILES-model checkpoint to start from
DEFAULT_CSV, DEFAULT_CKPT = cfg.DEFAULT_CSV, cfg.DEFAULT_PRETRAINED_CKPT


# Trains the exemplar generator and saves it.  Builds the held-out sets (whole targets and random molecules, never trained on, not even as exemplars), trains on episodes
# (a molecule plus exemplars drawn from the library pools), evaluates every --eval-every steps (teacher-forced loss, swap tests, and a free-running generation test),
# then saves the best-validation or the final step.  With --require-self-below it exits with status 3 unless the model learned to copy an exemplar that holds the answer.
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=DEFAULT_CKPT, help='the PRETRAINED checkpoint (general SMILES model) to start from')
    p.add_argument('--out', default=None)
    p.add_argument('--v6', default=None, help='ignored (kept so old command lines still work): the model code is now imported from base_model.py / base_data.py')
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--arch', choices=['lm', 'xattn'], default='lm',
                   help="lm = decoder-only, exemplars written in front of the molecule (default); xattn = the earlier cross-attention design")
    p.add_argument('--extra-layers', type=int, default=1, help='(lm) freshly initialised transformer layers added after the 3 pretrained ones')
    p.add_argument('--no-pointer', action='store_true',
                   help='(lm) turn OFF the pointer-generator copy channel (on by default: copying gets a direct gradient)')
    p.add_argument('--p-near', type=float, default=0.6,
                   help='share of exemplar draws that include 1-2 NEAR NEIGHBOURS of the molecule (similarity in [--near-min-sim, --max-ex-sim]) from the '
                        'same pool: random exemplars rarely contain an analog, so without this there is little to learn from (needs RDKit)')
    p.add_argument('--near-min-sim', type=float, default=0.35)
    p.add_argument('--seed-mode', action='store_true',
                   help="SEED-AND-EDIT training: every episode is ONE exemplar (a 'seed' from the disease / target / pair pool) that is a near neighbour "
                        "of the molecule to generate (similarity in [--near-min-sim, --max-ex-sim]); molecules with no such neighbour are skipped. "
                        "Use with --eval-k 1.  At generation each sample gets one random seed from the library and returns an analog of it.")
    p.add_argument('--p-solo', type=float, default=0.3,
                   help='in a copy-curriculum episode (--p-self), probability that the molecule itself is the ONLY exemplar (easiest copy task)')
    p.add_argument('--require-self-below', type=float, default=None,
                   help='GATE: after training, exit with an error (so a chained && command stops) unless the loss with the molecule itself '
                        'among the exemplars is below this value at the held-out molecules')
    p.add_argument('--k-max', type=int, default=6, help='maximum number of exemplars shown')
    p.add_argument('--ex-max-len', type=int, default=100, help='exemplars longer than this many characters are not used')
    p.add_argument('--max-list', type=int, default=3, help='rows listing more diseases/targets than this give no label')
    p.add_argument('--p-empty', type=float, default=0.08, help='share of episodes shown NO exemplars')
    p.add_argument('--type-probs', type=float, nargs=4, default=[0.35, 0.25, 0.15, 0.25],
                   metavar=('PAIR', 'TARGET', 'DISEASE', 'MIXED'))
    p.add_argument('--balance-power', type=float, default=0.5)
    p.add_argument('--max-ex-sim', type=float, default=0.9, help='drop exemplars more similar than this to the molecule being generated (0 = off)')
    p.add_argument('--reset-cross-attn', action='store_true', help='re-initialise the decoder cross-attention (pretrained one may be stuck ignoring its memory)')
    p.add_argument('--reset-encoder', action='store_true', help='re-initialise the exemplar encoder')
    p.add_argument('--p-self', type=float, default=0.0,
                   help='COPY CURRICULUM: probability that the molecule to generate is itself one of the exemplars (teaches the model to '
                        'attend to and copy the exemplars); decays linearly to 0 over --self-steps')
    p.add_argument('--self-steps', type=int, default=0, help='steps over which --p-self decays to 0 (0 = half of --steps)')
    p.add_argument('--aug-prob', type=float, default=0.3, help='probability of a randomised SMILES spelling (needs RDKit)')
    p.add_argument('--holdout-targets', nargs='*', default=['T52522', 'T45993'])
    p.add_argument('--n-holdout-multi', type=int, default=8, help='held-out targets with >= 2 populated diseases (incl. the named ones)')
    p.add_argument('--n-holdout-plain', type=int, default=8)
    p.add_argument('--row-holdout', type=float, default=0.08)
    p.add_argument('--eval-k', type=int, default=5, help='exemplars per set in the evaluations')
    p.add_argument('--val-max', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--gen-every', type=int, default=2000, help='free-running evaluation every N steps (0 = only at the end)')
    p.add_argument('--gen-n', type=int, default=20, help='samples per held-out target in the free-running evaluation')
    p.add_argument('--gen-targets', type=int, default=8)
    p.add_argument('--no-gen-eval', action='store_true')
    p.add_argument('--save-last', action='store_true', help='save the final step instead of the best-validation step')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.seed_mode and not a.save_last:
        # seed mode: best-validation-loss selection scores random exemplars, which is the wrong yardstick for seeds (a random exemplar is not a near neighbour, so its loss
        # is meant to be poor); the final step is saved instead
        a.save_last = True
        print("[seed-mode] saving the FINAL step (best-validation-loss selection scores random exemplars, not seeds).")

    # heavy imports come after argument parsing so that --help is fast
    import numpy as np
    import torch
    import exemplar_model as xm

    # rng draws the training episodes; the fixed evaluation sets use their own generator (rng_eval), so they do not depend on how many training draws were made
    rng = random.Random(a.seed)
    rng_eval = random.Random(a.seed + 7)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    # start from the pretrained SMILES model and wrap it: lm = one causal transformer reading the exemplars and then the molecule; xattn = encoder + cross-attention
    base, pp, raw = xm.load_base_checkpoint(a.checkpoint)
    if a.arch == 'lm':
        model = xm.ExemplarLM(base, a.k_max, extra_layers=a.extra_layers, pointer=not a.no_pointer).to(device)
    else:
        model = xm.ExemplarModel(base, a.k_max, reset_cross_attn=a.reset_cross_attn, reset_encoder=a.reset_encoder).to(device)
    PAD = pp.PAD_IDX
    out_path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), 'checkpoint_exemplar_v1.pt')
    print(f"[model] exemplar generator: {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M parameters, device {device}")

    # RDKit is optional: without it there is no randomised SMILES, no near-neighbour search or near-duplicate filter, and no free-running evaluation
    try:
        from rdkit import Chem, DataStructs, RDLogger
        RDLogger.DisableLog('rdApp.*')
        chem = xm.Chem2()
    except Exception:
        chem = None
    if chem is None:
        print("[warn] RDKit not importable: no randomised SMILES, no near-duplicate filter, no free-running evaluation.")
    do_gen = chem is not None and not a.no_gen_eval

    # data: the library, the held-out sets, and the training pools built WITHOUT the held-out molecules
    rows = xm.load_rows(a.csv, pp)
    # an exemplar must fit the length limit (its characters plus the two end markers)
    ok = lambda k: len(k) + 2 <= a.ex_max_len
    ho = xd.make_holdout(rows, rng, a.holdout_targets, a.n_holdout_multi, a.n_holdout_plain, a.row_holdout, a.max_list, ok=ok)
    exclude = set(ho['target_keys']) | set(ho['row_keys'])
    lib = xd.Library(rows, a.max_list, exclude=exclude, ok=ok)
    print(f"[library] training pools: {lib.summary()}")
    n_ok = sum(1 for k in lib.keys if ok(k))
    print(f"[library] {n_ok}/{len(lib.keys)} molecules are short enough (<= {a.ex_max_len} chars) to serve as exemplars")
    print(f"[holdout] {len(ho['held_targets'])} whole targets ({len(ho['target_keys'])} molecules) and "
          f"{len(ho['row_keys'])} random molecules never used in training")
    for t in ho['held_targets']:
        s = ho['sets'][t]
        dis = ', '.join(f"{d} {len(s['lib_by_d'][d])}/{len(s['test_by_d'][d])}" for d in list(s['lib_by_d'])[:3])
        print(f"    held-out target {t}: lib {len(s['lib'])} / test {len(s['test'])}" + (f"   diseases (lib/test): {dis}" if dis else ''))

    labels = xd.label_maps(rows, a.max_list)
    arr_fn = xm.make_arr_fn(pp)
    # mol_cache holds parsed RDKit molecules for randomised spelling; memo holds the spelling chosen for each molecule in the current step
    mol_cache = {}

    memo = {}

    # Returns the encoded token array to use for a molecule in this training step: with probability --aug-prob a randomised SMILES spelling (needs RDKit), otherwise the
    # standard encoding.  The choice is made once per molecule per step (memo is cleared every step), so a molecule that is also among its own exemplars is an exact copy.
    def spell(key):
        if key not in memo:
            arr = None
            if chem is not None and rng.random() < a.aug_prob:
                if key not in mol_cache:
                    mol_cache[key] = Chem.MolFromSmiles(key)
                m = mol_cache[key]
                if m is not None:
                    smi = Chem.MolToSmiles(m, doRandom=True, canonical=False)
                    if len(smi) + 2 <= (a.ex_max_len if ok(key) else pp.max_smiles_len - 1):
                        arr = pp.encode_smiles(smi)
            memo[key] = arr if arr is not None else arr_fn(key)
        return memo[key]

    # Tells whether a candidate exemplar c is too similar to the molecule x being generated (Tanimoto above --max-ex-sim).  Always False when the filter is off or RDKit is missing.
    def reject(x, c):
        if a.max_ex_sim <= 0 or chem is None:
            return False
        fx, fc = chem.fp(x), chem.fp(c)
        return fx is not None and fc is not None and DataStructs.TanimotoSimilarity(fx, fc) > a.max_ex_sim

    # Tanimoto similarity of x to each candidate, in order (0.0 for molecules that cannot be fingerprinted).
    def sims(x, cands):
        fx = chem.fp(x)
        out = [0.0] * len(cands)
        if fx is None:
            return out
        fcs = [chem.fp(c) for c in cands]
        idx = [i for i, f in enumerate(fcs) if f is not None]
        if idx:
            for i, v in zip(idx, DataStructs.BulkTanimotoSimilarity(fx, [fcs[i] for i in idx])):
                out[i] = v
        return out

    if a.seed_mode and chem is None:
        raise SystemExit("--seed-mode needs RDKit (similarity search).")
    # near-neighbour exemplars: in seed mode every episode must have one ('require'), otherwise a share --p-near of the draws includes one; the similarity window is
    # [--near-min-sim, --max-ex-sim].  None when RDKit is missing
    near_cfg = {'sims': sims, 'p': 1.0 if a.seed_mode else a.p_near, 'lo': a.near_min_sim,
                'hi': a.max_ex_sim if a.max_ex_sim > 0 else 0.9999, 'require': bool(a.seed_mode)} \
        if (chem is not None and (a.p_near > 0 or a.seed_mode)) else None
    # episode types (pair, target, disease, mixed pool) and the numbers of exemplars to draw; seed mode uses a single exemplar and no mixed episodes
    ep_types = (0.4, 0.35, 0.25, 0.0) if a.seed_mode else tuple(a.type_probs)
    ep_k = (1,) if a.seed_mode else (1, 2, 3, 3, 4, 4, 5, 5, 6, 6)
    # draws training episodes; small groups are sampled more often than their share of rows (--balance-power)
    sampler = xd.EpisodeSampler(lib, rng, a.k_max, ep_types, a.balance_power, k_choices=ep_k,
                                reject=reject if chem else None, near=near_cfg)
    if not any(sampler.groups[t][0] for t in xd.TYPES):
        raise SystemExit("No training groups -- check the CSV columns.")

    # fixed evaluation sets: held-out molecules at seen targets, test molecules of the held-out targets (t_items), and their disease contrasts (d_items)
    row_items = xd.row_eval_items(lib, ho['row_keys'], labels, rng_eval, k=a.eval_k, cap=a.val_max)
    t_items, d_items = xd.heldout_target_items(ho['sets'], rng_eval, k=a.eval_k)

    # near_ht: test molecules of the held-out targets that have a near neighbour in their own target's library, with that neighbour as the exemplar; they give a fixed
    # teacher-forced 'near neighbour' test
    near_ht = []
    if chem is not None:
        # signal check: how similar each test molecule is to its own target's library compared with the other held-out targets' libraries
        own, oth, sizes = [], [], {}
        tl = [t for t in ho['sets'] if len(ho['sets'][t]['lib']) >= 3]
        for t in tl:
            lf = chem.fps(ho['sets'][t]['lib'])
            of = chem.fps([m for u in tl if u != t for m in ho['sets'][u]['lib'][:20]])
            for x in ho['sets'][t]['test'][:30]:
                fx = chem.fp(x)
                if fx is None or not lf or not of:
                    continue
                own.append(max(DataStructs.BulkTanimotoSimilarity(fx, lf)))
                oth.append(max(DataStructs.BulkTanimotoSimilarity(fx, of)))
        if own:
            own, oth = np.array(own), np.array(oth)
            print(f"[signal] held-out test molecules: best similarity to their own target's library median {np.median(own):.2f} "
                  f"(share >= {a.near_min_sim:.2f}: {100 * np.mean(own >= a.near_min_sim):.0f}%) vs to other targets' libraries median {np.median(oth):.2f} "
                  f"(share >= {a.near_min_sim:.2f}: {100 * np.mean(oth >= a.near_min_sim):.0f}%)")
        for it in t_items:
            cands = [c for c in ho['sets'][it['t']]['lib'] if ok(c)]
            sv = sims(it['key'], cands) if cands else []
            best = sorted(((v, c) for v, c in zip(sv, cands) if a.near_min_sim <= v <= (a.max_ex_sim or 0.9999)), reverse=True)
            if best:
                nn = best[0][1]
                others = [] if a.seed_mode else [c for c in rng_eval.sample(cands, min(len(cands), 4)) if c != nn][:2]
                near_ht.append({'key': it['key'], 'ex': [nn] + others, 'sim': best[0][0]})
        print(f"[signal] {len(near_ht)}/{len(t_items)} held-out-target test molecules have a near neighbour in their library "
              f"(similarity {a.near_min_sim:.2f}-{a.max_ex_sim:.2f}); these are used for the 'NEAR neighbour' test below")
    print(f"[eval sets] held-out molecules {len(row_items)} | held-out-target molecules {len(t_items)} "
          f"| disease-contrast molecules at held-out targets {len(d_items)}")

    # Builds one evaluation item (molecule, exemplars, roles) giving every exemplar the same role.
    def mk(key, ex, role):
        return (key, ex, [role] * len(ex))

    # Teacher-forced negative log-likelihood of each item's molecule under the current model.
    def nll(items):
        v, _ = xm.sequence_nll(model, items, a.k_max, a.ex_max_len, arr_fn, PAD, amp=use_amp)
        return v

    # Compares the loss with the TRUE exemplars against the loss with SWAPPED exemplars (another target's or disease's set).  Returns (mean loss gap swapped - true,
    # share of molecules for which the true set gives the lower loss, mean true loss, n).  A positive gap and a share above 50% mean the exemplars carry information
    # about the molecule.  All nan when there are no items.
    def swap_stat(true_items, swap_items):
        if not true_items:
            return float('nan'), float('nan'), float('nan'), 0
        x, y = nll(true_items), nll(swap_items)
        return float(np.mean(y - x)), float(np.mean(x < y)), float(np.mean(x)), len(x)

    # Teacher-forced evaluation.  Returns a dict with 'rows' (held-out molecules at seen targets), 'ht' (held-out whole targets) and, when near neighbours exist, 'near'.
    # Each entry holds the loss with no exemplars, with the true exemplars, and with the molecule itself added to them ('self'), plus the swap-target and swap-disease gaps,
    # the swap accuracies and the counts.
    def tf_eval():
        out = {}
        ti = [it for it in row_items if it['tgt'] and it['tgt_swap']]
        pi = [it for it in row_items if it['pair'] and it['pair_swap']]
        tg = swap_stat([mk(it['key'], it['tgt'][0], xd.ROLE_TARGET) for it in ti],
                       [mk(it['key'], it['tgt_swap'][0], xd.ROLE_TARGET) for it in ti])
        pg = swap_stat([mk(it['key'], it['pair'][0], xd.ROLE_PAIR) for it in pi],
                       [mk(it['key'], it['pair_swap'][0], xd.ROLE_PAIR) for it in pi])
        ei = ti
        empty = float(np.mean(nll([(it['key'], [], []) for it in ei]))) if ei else float('nan')
        # the 'self' loss adds the molecule itself to its exemplars (only molecules short enough to be exemplars): a model that has learned to copy should score far lower
        si = [it for it in ti if ok(it['key'])]
        self_rows = float(np.mean(nll([(it['key'], it['tgt'][0][:a.k_max - 1] + [it['key']], [xd.ROLE_TARGET] * (len(it['tgt'][0][:a.k_max - 1]) + 1))
                                       for it in si]))) if si else float('nan')
        out['rows'] = {'self': self_rows, 'tgt_gap': tg[0], 'tgt_acc': tg[1], 'loss_true': tg[2], 'n_t': tg[3], 'pair_gap': pg[0], 'pair_acc': pg[1],
                       'n_p': pg[3], 'loss_empty': empty}
        # held-out whole targets, which the model never saw
        tt = [it for it in t_items if it['swap']]
        tg = swap_stat([mk(it['key'], it['true'][0], xd.ROLE_TARGET) for it in tt],
                       [mk(it['key'], it['swap'][0], xd.ROLE_TARGET) for it in tt])
        empty = float(np.mean(nll([(it['key'], [], []) for it in tt]))) if tt else float('nan')
        dg = swap_stat([mk(it['key'], it['true'][0], xd.ROLE_PAIR) for it in d_items],
                       [mk(it['key'], it['swap'][0], xd.ROLE_PAIR) for it in d_items])
        sh = [it for it in tt if ok(it['key'])]
        self_ht = float(np.mean(nll([(it['key'], it['true'][0][:a.k_max - 1] + [it['key']], [xd.ROLE_TARGET] * (len(it['true'][0][:a.k_max - 1]) + 1))
                                     for it in sh]))) if sh else float('nan')
        out['ht'] = {'self': self_ht, 'tgt_gap': tg[0], 'tgt_acc': tg[1], 'loss_true': tg[2], 'n_t': tg[3], 'pair_gap': dg[0], 'pair_acc': dg[1],
                     'n_p': dg[3], 'loss_empty': empty}
        if near_ht:
            sub = [(d['key'], d['ex'], [xd.ROLE_TARGET] * len(d['ex'])) for d in near_ht]
            emp = [(d['key'], [], []) for d in near_ht]
            out['near'] = {'with': float(np.mean(nll(sub))), 'without': float(np.mean(nll(emp))), 'n': len(near_ht),
                           'sim': float(np.median([d['sim'] for d in near_ht]))}
        return out

    # Free-running evaluation on the held-out targets (validity, target lift, disease lift) through xm.heldout_gen_eval.  Returns None without RDKit or with --no-gen-eval.
    # It uses a fresh generator with a fixed seed, so every call draws the same seeds.
    def gen_eval():
        if not do_gen:
            return None
        model.eval()
        if device == 'cuda':
            torch.cuda.empty_cache()
        return xm.heldout_gen_eval(model, pp, ho['sets'], random.Random(a.seed + 99), a.k_max, a.ex_max_len, arr_fn, chem,
                                   k=a.eval_k, n=a.gen_n, max_targets=a.gen_targets)

    # Prints one evaluation block: the teacher-forced results for held-out molecules and unseen targets, the near-neighbour test, and the free-running result when it was run.
    def show(step, tf, g, tag=''):
        print(f"  step {step}{tag}")
        for nm, lab in (('rows', 'unseen molecules, seen targets'), ('ht', 'UNSEEN TARGETS (lib -> test)')):
            r = tf[nm]
            print(f"      {lab:<32} loss: no exemplars {r['loss_empty']:.3f} -> true exemplars {r['loss_true']:.3f} -> exemplars INCLUDING the molecule itself {r['self']:.3f} | "
                  f"swap-TARGET gap {r['tgt_gap']:+.4f} acc {100 * r['tgt_acc']:.0f}% (n={r['n_t']}) | "
                  f"swap-DISEASE gap {r['pair_gap']:+.4f} acc {100 * r['pair_acc']:.0f}% (n={r['n_p']})")
        if tf.get('near'):
            r = tf['near']
            print(f"      UNSEEN TARGETS, molecules WITH a near neighbour in the library (n={r['n']}, median similarity {r['sim']:.2f}): loss without exemplars "
                  f"{r['without']:.3f} -> with that neighbour as the exemplar(s) {r['with']:.3f}   (the number that should fall)")
        if g and g.get('targets'):
            print(f"      FREE-RUNNING at unseen targets: valid {100 * g['valid']:.0f}% | target lift {g['target_lift']:.2f} "
                  f"(reference = copying lib molecules: {g['ref_target_lift']:.2f}) | disease lift {g.get('disease_lift', float('nan')):.2f} "
                  f"(reference {g.get('ref_disease_lift', float('nan')):.2f}, {len(g['dis_cases'])} cases)")

    # optimiser: AdamW with the learning rate scaled by lr_at
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    # Learning-rate multiplier at step s: linear warm-up over --warmup steps, then cosine decay to 0.1 at --steps.
    def lr_at(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        prog = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # number of steps over which the copy-curriculum probability --p-self decays to 0
    self_steps = a.self_steps or max(1, a.steps // 2)

    # Draws one batch of training episodes, each (molecule, exemplars, roles).  A share --p-empty gets no exemplars; with the copy curriculum (--p-self) the molecule itself
    # is put among its exemplars (alone with probability --p-solo).  Also returns each item's category ('other', 'near' or 'self') for the training-loss report.
    def make_batch(step):
        p_self = a.p_self * max(0.0, 1.0 - step / self_steps)
        items, cat = [], []
        while len(items) < a.batch:
            ep = sampler.sample()
            if ep is None:
                continue
            if rng.random() < a.p_empty:
                items.append((ep['key'], [], []))
                cat.append('other')
                continue
            ex, roles = list(ep['ex']), list(ep['roles'])
            self_ep = False
            if p_self > 0 and ok(ep['key']) and rng.random() < p_self:
                self_ep = True
                r0 = roles[0] if roles else xd.ROLE_TARGET
                if rng.random() < a.p_solo:
                    ex, roles = [ep['key']], [r0]
                else:
                    if len(ex) >= a.k_max:
                        j = rng.randrange(len(ex))
                        ex.pop(j)
                        roles.pop(j)
                    pos = rng.randrange(len(ex) + 1)
                    ex.insert(pos, ep['key'])
                    roles.insert(pos, r0)
            items.append((ep['key'], ex, roles))
            cat.append('self' if self_ep else ('near' if ep.get('near') else 'other'))
        return items, cat

    print("\n[start] evaluating the starting point (no exemplar pathway trained yet; swap accuracies ~50%)")
    model.eval()
    tf0 = tf_eval()
    g0 = gen_eval() if do_gen else None
    show(0, tf0, g0)
    # best: the state with the lowest validation loss so far (mean of the seen-target and unseen-target losses with true exemplars); acc: training loss per episode category, reset after each report
    best = {'score': float('inf'), 'step': 0, 'state': None}
    acc = {'other': [0.0, 0], 'near': [0.0, 0], 'self': [0.0, 0]}
    t0 = time.time()
    model.train()
    # training loop: build a batch, take one optimiser step (gradient norm clipped to 1.0), and evaluate every --eval-every steps
    for step in range(1, a.steps + 1):
        items, cat = make_batch(step)
        memo.clear()
        # arrays for the molecule (S), its exemplars (E) and their roles (R); the loss is the mean negative log-likelihood per molecule token, exemplars excluded
        S, E, R = xm.to_device(xm.build_arrays(items, a.k_max, a.ex_max_len, spell, spell, pad=PAD), device)
        nl, m = xm.per_token_nll(model, S, E, R, pad=PAD, amp=use_amp)
        loss = (nl * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            row = ((nl * m).sum(1) / m.sum(1).clamp(min=1.0)).detach().cpu().tolist()
        for r_, c_ in zip(row, cat):
            acc[c_][0] += r_
            acc[c_][1] += 1
        if step == 20:
            print(f"  [speed] {(time.time() - t0) / 20:.2f} s/step -> about {(time.time() - t0) / 20 * a.steps / 60:.0f} min for {a.steps} steps")
        if step % 250 == 0:
            f = lambda k_: f"{acc[k_][0] / max(acc[k_][1], 1):.3f} (n={acc[k_][1]})"
            print(f"  step {step:>5}  TRAIN loss: random exemplars {f('other')} | exemplars incl. a near neighbour {f('near')} | "
                  f"copy-curriculum (molecule itself) {f('self')}")
            acc = {'other': [0.0, 0], 'near': [0.0, 0], 'self': [0.0, 0]}
        if step % a.eval_every == 0 or step == a.steps:
            tf = tf_eval()
            g = gen_eval() if (do_gen and ((a.gen_every and step % a.gen_every == 0) or step == a.steps)) else None
            score = (tf['rows']['loss_true'] + tf['ht']['loss_true']) / 2
            tag = ''
            if score < best['score']:
                best = {'score': score, 'step': step, 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                tag = '  <- best validation loss so far'
            show(step, tf, g, tag)
            model.train()
    # keep the final step (--save-last, or no validation step improved) or restore the best-validation step
    if a.save_last or best['state'] is None:
        best = {'score': float('nan'), 'step': a.steps, 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print('[save-last] saving the FINAL step.')
    else:
        model.load_state_dict(best['state'])
    print(f"\n[done] saving step {best['step']}")
    model.eval()
    tfb = tf_eval()
    gb = gen_eval() if do_gen else None
    print("  start:")
    show(0, tf0, g0)
    print("  saved checkpoint:")
    show(best['step'], tfb, gb)

    # checkpoint contents: the exemplar model's weights, the base model's hyperparameters and preprocessor (so the file loads on its own), the exemplar settings,
    # and the held-out sets, which the evaluation scripts use to find the unseen targets
    out = {'format': xm.FORMAT_LM if a.arch == 'lm' else xm.FORMAT, 'exemplar_state_dict': best['state'],
           'model_hyperparams': raw['model_hyperparams'],
           'preprocessor_data': raw['preprocessor_data'], 'max_exemplars': a.k_max, 'ex_max_len': a.ex_max_len,
           'max_list': a.max_list, 'holdout': ho, 'args': vars(a), 'step': best['step'],
           'extra_layers': a.extra_layers if a.arch == 'lm' else 0, 'pointer': bool(a.arch == 'lm' and not a.no_pointer),
           'source': os.path.basename(a.checkpoint)}
    torch.save(out, out_path)
    base_name = os.path.basename(out_path)
    print(f"[saved] {out_path}")
    print("\nNext:  python check_exemplar_conditioning.py --checkpoint " + base_name)
    # optional gate: exit with status 3 unless the copy test passed, so a chained command stops before a long run
    if a.require_self_below is not None:
        s_rows, s_ht = tfb['rows']['self'], tfb['ht']['self']
        base_rows = tfb['rows']['loss_empty']
        print(f"\n[gate] loss with the molecule itself among the exemplars: held-out molecules {s_rows:.3f}, unseen targets {s_ht:.3f} "
              f"(no exemplars: {base_rows:.3f}; required: below {a.require_self_below:.3f})")
        if not (s_rows < a.require_self_below):
            print("[gate] FAILED -- the model did not learn to copy an exemplar that contains the answer, so the longer run is not started.")
            sys.exit(3)
        print("[gate] PASSED -- the exemplar pathway can copy; continuing.")


if __name__ == '__main__':
    main()
