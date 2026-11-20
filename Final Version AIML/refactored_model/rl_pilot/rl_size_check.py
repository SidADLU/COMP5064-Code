#!/usr/bin/env python3
"""
rl_size_check.py -- is an RL run's gain real, or is the model just making bigger molecules that Vina scores better?  READ-ONLY: it only reads CSV files and prints.

Why: the reward is a size-corrected docking percentile, but the size correction is a straight line fitted on decoys that are all about the same size
(15-38 heavy atoms, most near 22).  Vina scores keep improving with molecule size, so a model that learns to write bigger molecules can raise its percentile
without becoming any better at binding.  This script asks the same question three ways:

  1. TRAINING LOG   (--molecules rl_pilot/runs/<tag>/molecules.csv)
        percentile, raw score and ligand efficiency (LE = -score / heavy atoms) by size band, early steps vs late steps.
        Real improvement: percentile rises at a similar LE.  Size exploit: percentile and score rise while LE falls and heavy atoms climb.
  2. EVALUATION     (--dock-results rl_pilot/runs/<tag>/dock_results_*.csv ...)
        the generated set against only the decoys of a similar size: a 'local' percentile (decoys within +-3 heavy atoms of each molecule) and
        an AUC inside one fixed size band.  These do not depend on the straight-line size fit.
  3. SCORE vs SIZE  slope of score on heavy atoms for decoys, actives and generated molecules (kcal/mol per heavy atom).  A generated slope much steeper than the decoy slope
        means the straight-line correction under-corrects for large molecules.

Run from the folder that holds config.py:
  python rl_pilot\\rl_size_check.py --molecules rl_pilot\\runs\\r2\\molecules.csv
  python rl_pilot\\rl_size_check.py --dock-results rl_pilot\\runs\\r1\\dock_results_rl.csv rl_pilot\\runs\\r1\\dock_results_pre.csv
"""
import argparse
import csv
import math
import sys

import numpy as np

BANDS = [(0, 20), (21, 24), (25, 28), (29, 99)]
DECOY_ROLE, ACTIVE_ROLE, GEN_ROLE = 'calibration_decoy', 'calibration_active', 'generated_own'


# Reads a CSV into a list of dicts (empty file -> empty list).
def read_rows(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


# A float from a CSV cell, or NaN when the cell is empty or not a number.
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


# Slope and intercept of score on heavy atoms (straight line), or (nan, nan) with too few points or one size only.
def line(ha, score):
    ha, score = np.asarray(ha, float), np.asarray(score, float)
    ok = np.isfinite(ha) & np.isfinite(score)
    if ok.sum() < 8 or len(set(ha[ok].tolist())) < 3:
        return float('nan'), float('nan')
    s, i = np.polyfit(ha[ok], score[ok], 1)
    return float(s), float(i)


# Share of the decoys that a molecule beats (lower score = better); ties count half.  0.5 = a typical decoy.
def beats(score, decoy_scores):
    d = np.asarray(decoy_scores, float)
    return float((d > score).mean() + 0.5 * (d == score).mean())


# AUC of generated scores against decoy scores (lower = better), as in dock_generated.py.  NaN if either side is empty.
def auc_lower(gen, dec):
    g, d = np.asarray(gen, float), np.asarray(dec, float)
    if len(g) == 0 or len(d) == 0:
        return float('nan')
    return float((g[:, None] < d[None, :]).mean() + 0.5 * (g[:, None] == d[None, :]).mean())


# Local percentile: each molecule is compared only with the decoys within +-window heavy atoms of its own size (needs >= min_n of them).
# Returns the mean and its standard error over the molecules that could be compared, and how many that was.
def local_percentile(ha, score, dec_ha, dec_score, window=3, min_n=10):
    dec_ha, dec_score = np.asarray(dec_ha, float), np.asarray(dec_score, float)
    vals = []
    for h, s in zip(ha, score):
        near = dec_score[np.abs(dec_ha - h) <= window]
        if len(near) >= min_n:
            vals.append(beats(s, near))
    if not vals:
        return float('nan'), float('nan'), 0
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float('nan'), len(v)


# Formats a number or a dash.
def f(x, nd=2):
    return '  -  ' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{nd}f}'


# Part 1: the training log of one run (molecules.csv): early vs late, by size band.
def check_training(path, k=10):
    rows = [r for r in read_rows(path) if r.get('status') == 'docked']
    if not rows:
        print(f'[training] {path}: no docked molecules')
        return
    step = np.array([num(r['step']) for r in rows])
    ha = np.array([num(r['heavy_atoms']) for r in rows])
    sc = np.array([num(r['score']) for r in rows])
    pc = np.array([num(r['percentile']) for r in rows])
    le = -sc / ha
    last = step.max()
    print(f'\n[training] {path}   {len(rows)} docked molecules over {int(last)} steps')
    print('  (percentile 0.5 = typical decoy.  LE = -score / heavy atoms; the decoys sit near 0.36 and the known binders near 0.40 in the evaluation tables.)')
    print(f"  {'steps':<14}{'n':>5}{'heavy atoms':>13}{'raw score':>11}{'LE':>7}{'percentile':>12}")
    for name, m in (('first 10', step <= k), ('middle', (step > k) & (step <= last - k)), ('last 10', step > last - k)):
        if m.sum():
            print(f'  {name:<14}{m.sum():>5}{np.nanmean(ha[m]):>13.1f}{np.nanmean(sc[m]):>11.2f}{np.nanmean(le[m]):>7.3f}{np.nanmean(pc[m]):>12.3f}')
    print(f"\n  by size band, all steps pooled:   {'band (heavy atoms)':<22}{'n':>5}{'raw score':>11}{'LE':>7}{'percentile':>12}")
    for lo, hi in BANDS:
        m = (ha >= lo) & (ha <= hi)
        if m.sum():
            print(f"  {'':<34}{f'{lo}-{hi}' if hi < 99 else f'{lo}+':<22}{m.sum():>5}{np.nanmean(sc[m]):>11.2f}{np.nanmean(le[m]):>7.3f}{np.nanmean(pc[m]):>12.3f}")
    s, _ = line(ha, sc)
    r = float(np.corrcoef(ha, pc)[0, 1]) if len(ha) > 3 else float('nan')
    print(f'\n  slope of raw score on heavy atoms among these molecules: {f(s, 3)} kcal/mol per atom   |   correlation of percentile with heavy atoms: {f(r)}')
    e, l = (step <= k), (step > last - k)
    if e.sum() and l.sum():
        d_pc, d_ha, d_le = np.nanmean(pc[l]) - np.nanmean(pc[e]), np.nanmean(ha[l]) - np.nanmean(ha[e]), np.nanmean(le[l]) - np.nanmean(le[e])
        print(f'  first 10 -> last 10:  percentile {d_pc:+.3f},  heavy atoms {d_ha:+.1f},  LE {d_le:+.3f}')
        if d_ha >= 4 and d_le <= -0.02:
            print('  READ: the molecules got much bigger and less efficient while the percentile rose.  That is the signature of a size exploit, not of better binding.')
        elif d_ha >= 4:
            print('  READ: the molecules got much bigger.  Efficiency held up, so this is not clearly an exploit, but check the evaluation tables below before trusting it.')
        else:
            print('  READ: no strong size drift in the training molecules.')


# Part 2 and 3: evaluation tables for one dock_results CSV.
def check_eval(path, band=(18, 26)):
    rows = read_rows(path)
    get = lambda role: [r for r in rows if r.get('role') == role and r.get('score') not in ('', None)]
    dec, act, gen = get(DECOY_ROLE), get(ACTIVE_ROLE), get(GEN_ROLE)
    col = lambda rs, k: np.array([num(r[k]) for r in rs])
    d_ha, d_sc = col(dec, 'heavy_atoms'), col(dec, 'score')
    a_ha, a_sc = col(act, 'heavy_atoms'), col(act, 'score')
    g_ha, g_sc = col(gen, 'heavy_atoms'), col(gen, 'score')
    print(f'\n[evaluation] {path}   decoys {len(dec)}, actives {len(act)}, generated {len(gen)}')
    if len(gen) == 0 or len(dec) == 0:
        print('  nothing to compare (no generated or no decoy rows)')
        return
    print(f"  {'':<12}{'n':>4}{'mean HA':>9}{'HA range':>10}{'mean score':>12}{'mean LE':>9}{'slope kcal/atom':>17}")
    for name, ha, sc in (('decoys', d_ha, d_sc), ('actives', a_ha, a_sc), ('generated', g_ha, g_sc)):
        if len(ha):
            print(f"  {name:<12}{len(ha):>4}{np.nanmean(ha):>9.1f}{f'{ha.min():.0f}-{ha.max():.0f}':>10}{np.nanmean(sc):>12.2f}{np.nanmean(-sc / ha):>9.3f}{f(line(ha, sc)[0], 3):>17}")
    lp, se, n = local_percentile(g_ha, g_sc, d_ha, d_sc)
    print(f'\n  local percentile (each generated molecule vs decoys within +-3 heavy atoms): {f(lp)} +/- {f(se)}  (n {n} of {len(gen)} had >= 10 size-neighbour decoys)')
    if len(act):
        alp, ase, an = local_percentile(a_ha, a_sc, d_ha, d_sc)
        print(f'  the same for the known actives (the ceiling):                                {f(alp)} +/- {f(ase)}  (n {an})')
    lo, hi = band
    gi, di = (g_ha >= lo) & (g_ha <= hi), (d_ha >= lo) & (d_ha <= hi)
    print(f'  AUC inside the band {lo}-{hi} heavy atoms only: generated {int(gi.sum())} vs decoys {int(di.sum())}:  {f(auc_lower(g_sc[gi], d_sc[di]))}   (no size line involved)')
    print(f"  share of generated molecules outside the decoy size range +-2: {np.mean((g_ha < d_ha.min() - 2) | (g_ha > d_ha.max() + 2)):.0%}")
    sg, sd = line(g_ha, g_sc)[0], line(d_ha, d_sc)[0]
    if np.isfinite(sg) and np.isfinite(sd) and sg < sd - 0.1:
        print(f'  NOTE: generated molecules gain {abs(sg):.2f} kcal/mol per heavy atom but the decoy line only {abs(sd):.2f}: the straight-line size correction under-corrects for them.')
    if np.isfinite(sd) and abs(sd) < 0.06 and np.nanmean(g_ha) - np.nanmean(d_ha) >= 3:
        print('  CAVEAT: these decoys show almost no score-vs-size trend, so NO decoy-based size correction (line, local or band) can separate size from binding for molecules this much bigger.')
        print('          Trust the ligand efficiency (generated mean LE vs actives and decoys) and the heavy-atom comparison with the known actives instead.')
    print(f"\n  by size band:   {'band':<8}{'decoys n':>10}{'decoy score':>13}{'gen n':>7}{'gen score':>11}{'gen - decoy':>13}")
    for lo2, hi2 in BANDS:
        dm, gm = (d_ha >= lo2) & (d_ha <= hi2), (g_ha >= lo2) & (g_ha <= hi2)
        if gm.sum():
            ds = np.nanmean(d_sc[dm]) if dm.sum() else float('nan')
            print(f"  {'':<16}{f'{lo2}-{hi2}' if hi2 < 99 else f'{lo2}+':<8}{dm.sum():>10}{f(ds):>13}{gm.sum():>7}{np.nanmean(g_sc[gm]):>11.2f}{f(np.nanmean(g_sc[gm]) - ds):>13}")


# Reads the options and runs the parts that were asked for.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--molecules', nargs='*', default=[], help='training molecules.csv file(s) from rl_pilot/runs/<tag>/')
    ap.add_argument('--dock-results', nargs='*', default=[], help='dock_results CSV file(s) written by dock_generated.py')
    ap.add_argument('--band', type=int, nargs=2, default=[18, 26], metavar=('LO', 'HI'), help='size band for the band-only AUC (default 18 26)')
    a = ap.parse_args()
    if not a.molecules and not a.dock_results:
        ap.error('give --molecules and/or --dock-results')
    for p in a.molecules:
        check_training(p)
    for p in a.dock_results:
        check_eval(p, tuple(a.band))
    return 0


if __name__ == '__main__':
    sys.exit(main())
