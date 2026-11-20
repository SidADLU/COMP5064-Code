#!/usr/bin/env python3
"""
dock_generated.py  (v3.3)  --  do the molecules generated for a (disease, target) request look like binders of THAT target's receptor?

WHAT THIS SCRIPT IS (and is not)
  Docking never sees the model: it takes a SMILES string and a receptor and returns a score.  So this script cannot say anything about whether the
  generator's conditioning works (your similarity-based lift numbers do that).  It asks a separate question: do molecules generated for a target
  score like known binders of that target's pocket, once molecule size is accounted for?  That is only answerable at receptors where Vina can
  first be shown to tell known binders from property-matched decoys (CALIBRATION).  Everything else is reported as 'not interpretable'.

PIPELINE
  1. Calibrate every receptor: known actives vs decoys matched per active on heavy atoms / logP / H-bond donors / charge class (v13's rules).
     TRUSTED only if AUC >= 0.70 (lower bound > 0.5) AND the size-corrected AUC >= 0.60.  Sub-analyses are printed (library-only actives,
     curated-only actives, without the co-crystal ligand) because a pocket built around one ligand flatters that ligand.
  2. Diagnostics for receptors that fail (all optional, all cached):
       --redock        redock the co-crystal ligand (9 poses) and report the pose RMSD to the crystal pose (PASS = top pose <= 2 A)
       --easy-decoys   a second calibration against decoys matched on SIZE ONLY.  Separates 'the pocket cannot even tell binders from random
                       drugs' (suspect receptor / box / preparation / Vina) from 'it only fails against property-matched decoys'
       --box-pad X     enlarge every docking box by X A on each side (tests whether a tight box penalises larger ligands)
       extra structures of the same target can be added in EXTRA_RECEPTORS below; each is calibrated, the best TRUSTED one is used
  3. Dock each generated set at its own target's trusted receptor (own receptor only by default), together with the SEEDS the generator used,
     so each analog can be compared with its own seed.
  3b. --baseline  the random-seed control: the SAME generator, seeds drawn from the whole library (none of this target's ligands), docked at the same receptor.
      'conditioned vs baseline' is the test of whether choosing the target's ligands as seeds yields better-docking molecules.
  4. --cross  (optional negative control): also dock every generated set at the other trusted receptors.  It works with a single trusted
     receptor: the sets generated for OTHER targets are the comparison.  Caveat: those sets differ chemically, not only in target.

SCORING  a generated molecule's raw Vina score is size-corrected with the residual of (score ~ heavy atoms) fitted on that receptor's decoys, and
  reported as a percentile among the decoys.  AUC = chance a generated molecule beats a decoy (0.5 = no better).  Intervals are bootstrap 95%.

RULES FOR PfDHFR (T89515), FIXED BEFORE THE RUN THAT USES THEM  (pre-registered; nothing below was chosen by looking at a docking score)
  * T89515 is labelled 'PDF' in TTD, but its ligands are DHFR-type antifolates and the receptor (1J3I, WR99210) matches the ligands.
  * actives = ligands of T89515 of ANY stage that carry the classical antifolate pharmacophore (exocyclic N - ring C - ring N - ring C - exocyclic N:
    2,4-diaminopyrimidine / pteridine / 4,6-diamino-dihydrotriazine; amide and sulfonamide N excluded), plus v13's 10 curated antifolates.
    Proguanil / chlorproguanil (biguanide prodrugs, no pharmacophore) and meprobamate (mis-annotated) are therefore out.
  * strata: 'curated' (v13's 10), 'clinical' (clinical-stage library ligands), 'research' (investigative / patented library ligands).  The earlier
    run saw the curated and clinical ones; the 'research' stratum was not in it.  TRUST IS DECIDED ON THE RESEARCH STRATUM ONLY (n >= 15):
    AUC >= 0.70, AUC - SE > 0.5, size-corrected AUC >= 0.60.  The other strata are printed as descriptive context.
  * decoys: clinical-stage library drugs, not ligands of any DHFR code, without the pharmacophore or a glutamate tail.
  * seeds without the pharmacophore are not used for the generated-set analysis (selected on the seed only, never on the output).
  * with --ensemble a second, equivalent site (chain B of the same crystal structure) is docked as well and the better score is used, for the
    calibration and for the generated sets alike.

POTENCY FILTER (--potency; rules fixed before use, applied identically to every receptor)
  python fetch_potency.py            (once, on a machine with internet) writes potency_<case>.csv from ChEMBL: human target (P. falciparum for PfDHFR) by UniProt
                                     accession, Ki / Kd / IC50 with a pChEMBL value; potency of a compound = the median over its measurements
  * calibration actives are kept only if their potency is <= --max-nm (default 100 nM); actives with no ChEMBL measurement are dropped and counted
  * decoys are dropped if they are KNOWN binders of the target: potency <= --decoy-nm (default 1000 nM)
  * a potency-versus-score check is printed (Spearman of pKi / pIC50 against the size-corrected score): does Vina rank potent ligands above weaker ones?
  * --max-nm 0 switches the active filter off but keeps the check and the decoy clean-up.  Compare a filtered run with the unfiltered one: the filtered result
    is NOT independent confirmation, because the filter was chosen after the first calibration had been seen.

DPP-4 INTERACTION RULE (--interaction dpp4_amine_glu; FIXED 30 Oct 2026 BEFORE ANY DOCKING WITH IT; no value below may be changed after seeing a result)
  Why: Vina has no electrostatics, and DPP-4 inhibitors anchor a basic amine on the Glu205 / Glu206 carboxylates (in the 1X70 crystal the ligand's primary amine is
  2.7-3.1 A from their oxygens).  The rule asks Vina for that interaction explicitly.  It is a separate scoring variant, always reported next to the plain Vina AUC.
  * each ligand is docked with 9 poses.  A pose SATISFIES the rule if a basic nitrogen of the ligand lies within 4.0 A of OE1/OE2 of Glu205 or Glu206 (receptor atoms
    inside the docking box).  Basic nitrogen = ammonium / protonated amine, a non-aromatic amine that is not an amide / sulfonamide / aniline / N-N / N-O, or an amidine.
  * score = the best Vina energy among the satisfying poses; a ligand with no satisfying pose (or no basic nitrogen) gets its plain best score + 2.0 kcal/mol (constant
    chosen a priori, not fitted).  The usual calibration (AUC, size-corrected AUC, trust rule) then runs on these scores.  Redocking of the crystal ligand is unaffected.
  * printed with it: the share of actives and of decoys that satisfy the rule, and the plain Vina AUC computed from the SAME dockings (so the two are comparable).
  * calibration actives / decoys are the same as the plain run (same seed), so the cache keys differ only by the rule: it never mixes with plain-Vina cache entries.
  * also printed: a CONTROL that keeps only "has a basic nitrogen" (+2 kcal/mol if not) and ignores pose geometry, to show how much of the gain is the amine alone.
  * --interaction sert_amine_asp98 (depression case): the same rule with Asp98 OD1/OD2 for SERT, every constant copied unchanged from the DPP-4 rule.

USAGE (from the folder that holds the scripts; CleanedData.csv, rec_*.pdbqt, *_ligand.pdb and vina.exe are searched in this folder and its parents)
  python generate_for_request.py --target T89515 --n 70 --out gen_malaria_t89515.csv     (target-level seeds: no --disease)
  python dock_generated.py --fake                                plumbing and report layout only (seconds); NO docking, numbers are made up
  python generate_for_request.py --random-seeds --target T89515 --exclude-seed-targets T83386,T17345,T19229,T35465,T52227,T02702,T47077 --n 70 --out gen_malaria_t89515_baseline.csv
  python generate_for_request.py --random-seeds --target T31391 --n 70 --out gen_diabetes_baseline.csv
  python generate_for_request.py --random-seeds --target T27812 --exclude-seed-targets T21945,T55959 --n 70 --out gen_depression_baseline.csv
  python dock_generated.py --cross --baseline --n 40              main run: calibrations, own-receptor dockings, random-seed baseline, negative control (resumable)
  python dock_generated.py --diagnose --cases diabetes,depression   redock + easy-decoy checks for the receptors that failed (no generated sets needed)
  python dock_generated.py --cases diabetes --box-pad 4 --cache dock_cache_box4.jsonl --out dock_results_box4.csv --summary dock_summary_box4.json
  --ensemble   also dock the equivalent second site (chain B) and keep the better score   --report-only   re-print from the cache, no docking
  --cases a,b  runs a subset (keys: see CASES).
NOTE: these targets were in the training data of checkpoint_exemplar_seed_v2.pt, so a good result means 'conditioning on a known target gives
molecules that look like binders', not generalisation to a new target.  Disease conditioning cannot be tested by docking.
      For the unseen-target version of this test generate the sets with --checkpoint checkpoint_exemplar_seed_v3_holdout.pt, which also holds the three study targets out.
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

# folder of this script; find_file also searches its parent folders
HERE = os.path.dirname(os.path.abspath(__file__))

# antifolate pharmacophore as SMARTS: exocyclic N - ring C - ring N - ring C - exocyclic N (bond orders left open, so Kekule and aromatic SMILES both match).  It covers
# 2,4-diaminopyrimidines, 2,4-diaminopteridines and 4,6-diamino-1,2-dihydro-1,3,5-triazines; amide and sulfonamide nitrogens do not count
DHFR_PHARMACOPHORE = ['[#7;!R;!$([#7]C=O);!$([#7]S(=O)=O)]~[#6;R]~[#7;R]~[#6;R]~[#7;!R;!$([#7]C=O);!$([#7]S(=O)=O)]']
# classical antifolate glutamate tail (excluded from the decoys)
GLU_TAIL = ['[#6](=O)[#7][#6]([#6][#6][#6](=O)[#8])[#6](=O)[#8]']
# start-up self-test of the pharmacophore rule, run in real mode (--skip-selftest bypasses it): (SMILES, must match the pharmacophore, name)
SELFTEST = [('COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC', True, 'trimethoprim'), ('CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1', True, 'pyrimethamine'),
            ('CC1(C)N=C(N)N=C(N)N1c1ccc(Cl)cc1', True, 'cycloguanil'), ('CC1(C)N=C(N)N=C(N)N1OCCCOc1cc(Cl)c(Cl)cc1Cl', True, 'WR99210'),
            ('CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(O)=O)C(O)=O', True, 'methotrexate'),
            ('Nc1nc(N)c2nc(-c3ccccc3)c(N)nc2n1', True, 'triamterene'),
            ('CN(C)C(=N)NC(N)=N', False, 'metformin'), ('CC(C)NC(=N)NC(=N)Nc1ccc(Cl)cc1', False, 'proguanil'), ('Nc1ncnc2[nH]cnc12', False, 'adenine'),
            ('Nc1ccc(cc1)S(=O)(=O)Nc1ncccn1', False, 'sulfadiazine'), ('Fc1ccc(cc1)C1CCNCC1COc1ccc2OCOc2c1', False, 'paroxetine'),
            ('N[C@H](CC(=O)N1CCn2c(nnc2C1)C(F)(F)F)Cc1cc(F)c(F)cc1F', False, 'sitagliptin')]
# the study cases.  key: the generated set is read from gen_<key>.csv; v13_config: the name of the entry of targets.DISEASE_TARGET_CONFIGS whose receptor, box and curated actives are used;
# decoy_exclude_codes: target codes whose ligands are never used as decoys; drop_active_names / drop_seed_names: mis-annotated compounds that are left out; active_stages and decision:
# which actives are used, and with 'fresh' the trust is decided on the 'research' stratum only (see the docstring); active_smarts / exclude_decoy_smarts: the pharmacophore required
# of library actives and forbidden in decoys; alt_sites: equivalent sites of the SAME crystal structure (the other chain, found by superposing the binding-site C-alpha atoms on the
# co-crystal ligand's site), used with --ensemble.  Each box is the rotated ligand's bounding box + 5 A per side
CASES = [
    dict(key='malaria_t89515', disease='Malaria (target-level seeds)', target='T89515', v13_config='Malaria - T89515', receptor='PfDHFR',
         crystal_ligand='malaria_ligand.pdb',
         decoy_exclude_codes=['T89515', 'T83386', 'T17345', 'T19229', 'T35465', 'T52227', 'T02702', 'T47077'],
         drop_active_names=['meprobamate', 'ly333531', 'proguanil', 'chlorproguanil'],
         drop_seed_names=['meprobamate', 'ly333531', 'proguanil', 'chlorproguanil'],
         active_stages='all', decision='fresh', active_smarts=DHFR_PHARMACOPHORE, exclude_decoy_smarts=DHFR_PHARMACOPHORE + GLU_TAIL,
         alt_sites=[dict(name='B', center=(31.633, -28.245, 5.847), size=(19.2, 17.3, 18.3))]),
    dict(key='diabetes', disease='Type-2 diabetes', target='T31391', v13_config='Type-2 Diabetes', receptor='DPP-4',
         crystal_ligand='diabetes_ligand.pdb',
         decoy_exclude_codes=['T31391'], drop_active_names=[], drop_seed_names=[],
         active_stages='clinical', decision='all', active_smarts=None, exclude_decoy_smarts=None,
         alt_sites=[dict(name='B', center=(-5.752, 64.664, 34.756), size=(23.1, 15.8, 20.1))]),
    dict(key='depression', disease='Major depressive disorder', target='T27812', v13_config='Depression', receptor='SERT',
         crystal_ligand='depression_ligand.pdb',
         decoy_exclude_codes=['T27812', 'T21945', 'T55959'], drop_active_names=[], drop_seed_names=[],
         active_stages='clinical', decision='all', active_smarts=None, exclude_decoy_smarts=None, alt_sites=[]),
]
# further structures of the same target (another PDB entry or conformation), normally loaded from a JSON file made by prep_receptors.py (--extra-receptors).  Each is calibrated on its
# own and the generated set is docked at the best TRUSTED structure.  Fields: label, case, pdbqt, center, size, crystal_ligand and reference (None = the case's own reference SMILES)
EXTRA_RECEPTORS = []

# calibration constants, unchanged since the first calibration: actives sampled per receptor, decoys per active, minimum number of actives, minimum number of decision-set actives
N_ACTIVES, DECOYS_PER_ACTIVE, MIN_ACTIVES, MIN_DECISION = 30, 3, 10, 15
# heavy-atom range allowed for calibration actives and decoys
MIN_HA, MAX_HA = 15, 45
# decoy candidates sampled per receptor; decoys with a Tanimoto similarity of MAX_DECOY_SIM or more to an active are dropped; random seed of the calibration draws
DECOY_POOL, MAX_DECOY_SIM, CAL_SEED = 3000, 0.40, 1
# trust thresholds: 'usable' needs AUC >= AUC_USABLE (and AUC - SE > 0.5); AUC_WEAK separates 'weak' from 'non_discriminating'
AUC_USABLE, AUC_WEAK = 0.70, 0.60
# a receptor is trusted only if the size-corrected AUC also reaches this
SIZE_CORRECTED_MIN = 0.60
# library stage labels that do not count as clinical-stage
NON_CLINICAL = ('investigative', 'patented', 'preclinical')
# heavy-atom range of the generated molecules that are docked
GEN_HA_RANGE = (8, 60)
# redocking: number of poses, and the RMSD (A) that the top pose must reach to PASS
REDOCK_POSES, REDOCK_PASS_A = 9, 2.0
# research-stratum actives sampled per receptor (cases that use all stages)
N_RESEARCH = 40

# pre-registered interaction rules (see the docstring): a pose satisfies a rule if a basic ligand nitrogen lies within `cutoff` A of the listed receptor oxygens; a ligand with no
# satisfying pose gets its plain best score + `penalty`; `poses` is the number of poses docked and `margin` the extra A around the docking box in which receptor atoms are looked up.
# The constants were fixed in advance and must not be changed after a result has been seen
INTERACTION_RULES = {
    # DPP-4: the amine anchors on Glu205 / Glu206 (fixed 30 Oct 2026, before any docking with it)
    'dpp4_amine_glu': dict(cases=['diabetes'], residues=(('GLU', 205), ('GLU', 206)), atoms=('OE1', 'OE2'), cutoff=4.0, penalty=2.0, poses=9, margin=3.0),
    # SERT: every SSRI / tricyclic / SNRI anchors its amine on Asp98 (the Na+/amine site).  Constants copied unchanged from the DPP-4 rule (fixed 31 Oct 2026, before any SERT docking with it)
    'sert_amine_asp98': dict(cases=['depression'], residues=(('ASP', 98),), atoms=('OD1', 'OD2'), cutoff=4.0, penalty=2.0, poses=9, margin=3.0),
}
# SMARTS for a basic nitrogen (the first atom of each pattern is the nitrogen): (1) ammonium, protonated or neutral aliphatic amine (not aromatic, not N=, N-O, an amide-type
# N, an aniline, N-N or a cyanamide); (2) an amidine or guanidine nitrogen; (3) an amidinium or iminium nitrogen
BASIC_N_SMARTS = [
    '[N;!a;!$(N=*);!$(N~[#8]);!$(N-[#6,#16,#15]=[#8,#16,#7]);!$(N-a);!$(N-[#7]);!$(N-C#N)]',
    '[N;!a;!$(N=*);!$(N-C=O)]-[#6]=[N;!a]',
    '[N;+;!a;$(N=[#6])]',
]
# self-test of the basic-nitrogen rule (used with --interaction): (SMILES, should have a basic nitrogen, name)
INTER_SELFTEST = [('N[C@H](CC(=O)N1CCn2c(nnc2C1)C(F)(F)F)Cc1cc(F)c(F)cc1F', True, 'sitagliptin (primary amine)'),
                  ('Cn1c(=O)cc(N2CCCC(N)C2)n(Cc2ccccc2C#N)c1=O', True, 'alogliptin (primary amine)'),
                  ('CNCCC(c1ccccc1)Oc1ccc(cc1)C(F)(F)F', True, 'fluoxetine (secondary amine)'),
                  ('[NH3+]CCc1ccccc1', True, 'protonated phenethylamine'), ('CN1CCCCC1', True, 'N-methylpiperidine (tertiary amine)'),
                  ('NC(=N)c1ccccc1', True, 'benzamidine'),
                  ('CC(=O)Nc1ccccc1', False, 'acetanilide (amide)'), ('Nc1ccccc1', False, 'aniline'), ('c1ccncc1', False, 'pyridine'),
                  ('Cn1cnc2c1c(=O)n(C)c(=O)n2C', False, 'caffeine'), ('CS(N)(=O)=O', False, 'methanesulfonamide'),
                  ('O=[N+]([O-])c1ccccc1', False, 'nitrobenzene'), ('Nc1ncccn1', False, '2-aminopyrimidine'), ('NC(=O)N', False, 'urea'),
                  ('c1c[nH]cn1', False, 'imidazole')]


import config as cfg

DEFAULT_CSV = cfg.DEFAULT_CSV


# Looks for a file: the path as given, then the script folder, its parents and the working directory (and its parent), then extra_dirs.  Returns the absolute path or None.
def find_file(name, extra_dirs=()):
    if os.path.isabs(name) and os.path.exists(name):
        return name
    cands = [name, os.path.join(os.getcwd(), name)]
    for d in [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)), os.getcwd(), os.path.dirname(os.getcwd())] + list(extra_dirs):
        cands.append(os.path.join(d, name))
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# In-place RMSD (no alignment), minimum over the atom mappings.  D is the docked heavy-atom coordinates (n, 3) in docked-atom order, C the crystal coordinates (n, 3), and each
# mapping m in matches has m[i] = index of the crystal atom that docked atom i maps to (all symmetry-equivalent mappings).
def min_rmsd(D, C, matches):
    D, C = np.asarray(D, float), np.asarray(C, float)
    best = float('inf')
    for m in matches:
        r = float(np.sqrt(((D - C[list(m)]) ** 2).sum(axis=1).mean()))
        if r < best:
            best = r
    return best


# cache for rule_atoms
_RULE_ATOMS = {}


# Coordinates (n, 3) of the receptor atoms an interaction rule refers to (for example OE1 / OE2 of Glu205 and Glu206) that lie inside this receptor's docking box plus the rule's
# margin, read from the receptor pdbqt.  Box membership picks the right chain in a dimer.  Results are cached.
def rule_atoms(rec, rule):
    key = (rec['pdbqt'], tuple(rec['center']), tuple(rec['size']), tuple(rule['residues']), tuple(rule['atoms']))
    if key in _RULE_ATOMS:
        return _RULE_ATOMS[key]
    want = {(rn, int(i)) for rn, i in rule['residues']}
    c, s = np.asarray(rec['center'], float), np.asarray(rec['size'], float)
    xyz = []
    with open(rec['pdbqt'], encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            try:
                rn, ri, nm = line[17:20].strip(), int(line[22:26]), line[12:16].strip()
                p = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if (rn, ri) in want and nm in rule['atoms'] and np.all(np.abs(p - c) <= s / 2 + rule['margin']):
                xyz.append(p)
    _RULE_ATOMS[key] = np.array(xyz, float).reshape(-1, 3)
    return _RULE_ATOMS[key]


# Smallest distance (A) between the ligand atoms with indices n_idx (pos = all ligand coordinates of one pose) and any point of target_xyz; inf if either is empty.
def pose_min_dist(pos, n_idx, target_xyz):
    if len(n_idx) == 0 or len(target_xyz) == 0:
        return float('inf')
    P = np.asarray(pos, float)[list(n_idx)]
    return float(np.sqrt(((P[:, None, :] - np.asarray(target_xyz, float)[None, :, :]) ** 2).sum(axis=-1)).min())


# Interaction-rule score from the per-pose Vina energies and the per-pose rule distances: the best energy among the poses within cutoff, or the plain best energy + penalty if there is none.
# Returns (score, number of satisfying poses).
def rule_score(scores, dists, cutoff, penalty):
    ok = [s for s, d in zip(scores, dists) if d <= cutoff]
    return (min(ok) if ok else min(scores) + penalty), len(ok)


# AutoDock atom type -> element symbol
_AD_ELEMENT = {'C': 'C', 'A': 'C', 'N': 'N', 'NA': 'N', 'NS': 'N', 'O': 'O', 'OA': 'O', 'OS': 'O', 'S': 'S', 'SA': 'S', 'P': 'P', 'F': 'F',
               'CL': 'Cl', 'BR': 'Br', 'I': 'I', 'H': 'H', 'HD': 'H', 'HS': 'H'}


# Heavy atoms of a PDB ligand file.  Returns (element list, coordinates (n, 3)).
def read_pdb_ligand(path):
    el, xyz = [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        if line[:6].strip() not in ('ATOM', 'HETATM'):
            continue
        e = line[76:78].strip().capitalize() or re.sub(r'[^A-Za-z]', '', line[12:16])[:1].upper()
        if e in ('H', 'D'):
            continue
        el.append(e)
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return el, np.array(xyz, float)


# Splits Vina output (several MODEL blocks) into poses.  Returns a list of (elements, heavy-atom coordinates (n, 3)); hydrogens are skipped and AutoDock types are mapped to elements.
def parse_pdbqt_poses(text):
    poses, el, xyz = [], None, None
    for line in text.splitlines():
        if line.startswith('MODEL'):
            el, xyz = [], []
        elif line.startswith('ENDMDL'):
            if el is not None:
                poses.append((el, np.array(xyz, float)))
            el, xyz = None, None
        elif line[:6].strip() in ('ATOM', 'HETATM'):
            if el is None:
                el, xyz = [], []
            e = _AD_ELEMENT.get(line.split()[-1].upper(), line.split()[-1][:1].upper())
            if e == 'H':
                continue
            el.append(e)
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if el:
        poses.append((el, np.array(xyz, float)))
    return poses


# Topology-free, element-matched, in-place RMSD (optimal one-to-one assignment per element, no alignment).  It is optimistic, because it may pair atoms that the chemistry would not.
# Returns (rmsd, number of matched atoms).  Used only as a fallback when the graph mapping between the docked and the crystal ligand fails.
def assignment_rmsd(D_el, D, C_el, C):
    from scipy.optimize import linear_sum_assignment
    D, C = np.asarray(D, float), np.asarray(C, float)
    tot, n = 0.0, 0
    for e in set(D_el) & set(C_el):
        di = [i for i, x in enumerate(D_el) if x == e]
        ci = [i for i, x in enumerate(C_el) if x == e]
        cost = ((D[di][:, None, :] - C[ci][None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        tot += float(cost[r, c].sum())
        n += len(r)
    return (float(np.sqrt(tot / n)) if n else float('nan')), n


# Assignment RMSD of every docked pose to the crystal ligand.  Raises ValueError if fewer than 80% of the atoms can be matched by element.
def pose_assignment_rmsds(poses_text, crystal_pdb):
    cel, C = read_pdb_ligand(crystal_pdb)
    res = []
    for el, xyz in parse_pdbqt_poses(poses_text):
        r, n = assignment_rmsd(el, xyz, cel, C)
        if n < 0.8 * max(len(el), len(cel)):
            raise ValueError(f'only {n} atoms could be matched by element (docked {len(el)}, crystal {len(cel)})')
        res.append(r)
    return res


# Everything that needs RDKit or Vina: cleaning SMILES, describing molecules (size, logP, donors, charge class, fingerprint), SMARTS tests, docking (with or without an interaction
# rule), pose RMSD and redocking.
class RealBackend:

    # ns is the docking module (docking.py: dock_ligand, unsupported_elements, PDBQTMolecule, ...).  Imports RDKit and creates the salt remover and the Morgan fingerprint generator.
    def __init__(self, ns):
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem, Crippen, Lipinski
        from rdkit.Chem.SaltRemover import SaltRemover
        RDLogger.DisableLog('rdApp.*')
        self.ns, self.Chem, self.DataStructs, self.Crippen, self.Lipinski = ns, Chem, DataStructs, Crippen, Lipinski
        self.remover = SaltRemover()
        self.gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        self._pat = {}

    # Returns (canonical single-component SMILES or None, reason).  Counterions are stripped; mixtures='skip' drops records that are still multi-component (the rule for library records),
    # mixtures='largest' keeps the largest fragment (used for generated molecules).  Molecules with elements Vina cannot dock are rejected.
    def clean(self, smi, mixtures='skip'):
        Chem = self.Chem
        if not isinstance(smi, str) or not smi.strip():
            return None, 'empty'
        m = Chem.MolFromSmiles(smi.strip())
        if m is None:
            return None, 'unparsable'
        if len(Chem.GetMolFrags(m)) > 1:
            try:
                m2 = self.remover.StripMol(m, dontRemoveEverything=True)
            except Exception:
                m2 = m
            if len(Chem.GetMolFrags(m2)) > 1:
                if mixtures == 'skip':
                    return None, 'multi_component'
                m2 = max(Chem.GetMolFrags(m2, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
            m = m2
        bad = self.ns.unsupported_elements(m)
        if bad:
            return None, 'unsupported:' + ','.join(sorted(bad))
        return Chem.MolToSmiles(m), 'ok'

    # Canonical SMILES without stereochemistry, to recognise the same compound whatever the stereo annotation.
    def flat(self, smi):
        m = self.Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        return self.Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else smi

    # Identity of a compound for the potency table: largest fragment, neutralised, no stereo (ChEMBL and the library differ in salts, charges and stereo).  None if it cannot be parsed.
    def potency_key(self, smi):
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            m = self.Chem.MolFromSmiles(smi)
            if m is None:
                return None
            if len(self.Chem.GetMolFrags(m)) > 1:
                m = max(self.Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms())
            m = rdMolStandardize.Uncharger().uncharge(m)
            return self.Chem.MolToSmiles(m, isomericSmiles=False)
        except Exception:
            return None

    # Properties used to match decoys to actives: heavy atoms, Crippen logP, H-bond donors, charge class (sign of the net charge after protonation) and the Morgan fingerprint.  None if the SMILES does not parse.
    def describe(self, smi):
        m = self.Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return {'ha': m.GetNumHeavyAtoms(), 'logp': self.Crippen.MolLogP(m), 'hbd': self.Lipinski.NumHDonors(m),
                'cls': int(np.sign(self.ns._net_charge_after_protonation(m))), 'fp': self.gen.GetFingerprint(m)}

    # Tanimoto similarity of one fingerprint to a list of fingerprints.
    def similarities(self, fp, fps):
        return list(self.DataStructs.BulkTanimotoSimilarity(fp, fps))

    # Exits with a message if a SMARTS pattern cannot be parsed.
    def check_smarts(self, patterns):
        for p in patterns or []:
            if self.Chem.MolFromSmarts(p) is None:
                raise SystemExit(f"SMARTS pattern could not be parsed by RDKit: {p}")

    # True if the molecule contains any of the SMARTS patterns (patterns are parsed once and kept).
    def matches(self, smi, patterns):
        m = self.Chem.MolFromSmiles(smi)
        if m is None:
            return False
        for p in patterns:
            q = self._pat.get(p)
            if q is None:
                q = self._pat[p] = self.Chem.MolFromSmarts(p)
            if q is not None and m.HasSubstructMatch(q):
                return True
        return False

    # Indices of the basic nitrogen atoms of a ligand (BASIC_N_SMARTS).
    def basic_n(self, mol):
        if getattr(self, '_basic', None) is None:
            self._basic = [self.Chem.MolFromSmarts(p) for p in BASIC_N_SMARTS]
            if any(q is None for q in self._basic):
                raise SystemExit('BASIC_N_SMARTS could not be parsed by RDKit')
        mol.UpdatePropertyCache(strict=False)
        idx = set()
        for q in self._basic:
            for m in mol.GetSubstructMatches(q):
                idx.add(m[0])
        return sorted(idx)

    # Checks the basic-nitrogen rule on the known compounds in INTER_SELFTEST.  Returns failure messages (empty = as intended).
    def inter_selftest(self):
        bad = []
        for smi, want, name in INTER_SELFTEST:
            m = self.Chem.MolFromSmiles(smi)
            got = m is not None and len(self.basic_n(m)) > 0
            if got != want:
                bad.append(f"{name}: basic nitrogen found {got}, expected {want}")
        return bad

    # Docks one SMILES at a receptor and returns its score (None if docking failed).  Plain: one pose, the best Vina score.  With an interaction rule on the receptor (rec['inter']): 9 poses are
    # docked and rebuilt as RDKit molecules, the smallest distance from the ligand's basic nitrogens to the rule's receptor atoms is measured in every pose, and the score is the best energy
    # among the poses within the cutoff, or the plain best + penalty if there is none (rule_score).  The details (plain best score, number of satisfying poses, whether there is a basic
    # nitrogen, closest distance) are left in self.last_extra, which the cache stores.
    def dock(self, smi, rec, exhaustiveness, vina_seed):
        self.last_extra = None
        rule = INTERACTION_RULES.get(rec.get('inter')) if rec.get('inter') else None
        if rule is None:
            r = self.ns.dock_ligand(smi, rec['pdbqt'], rec['center'], rec['size'], exhaustiveness=exhaustiveness, n_poses=1, vina_seed=vina_seed)
            return None if r is None else float(r['best_score'])
        r = self.ns.dock_ligand(smi, rec['pdbqt'], rec['center'], rec['size'], exhaustiveness=exhaustiveness, n_poses=rule['poses'], vina_seed=vina_seed)
        if r is None:
            return None
        scores = [float(s) for s in r['all_scores']]
        pm = self.ns.PDBQTMolecule(r['poses_pdbqt'], skip_typing=True)
        mols = self.ns.RDKitMolCreate.from_pdbqt_mol(pm)
        mol = mols[0] if mols and mols[0] is not None else None
        if mol is None:
            raise RuntimeError('interaction rule: could not rebuild the docked poses')
        try:
            mol = self.Chem.RemoveHs(mol)
        except Exception:
            mol = self.Chem.RemoveHs(mol, sanitize=False)
        nidx = self.basic_n(mol)
        glu = rule_atoms(rec, rule)
        dists = [pose_min_dist(conf.GetPositions(), nidx, glu) for conf in mol.GetConformers()]
        if len(dists) != len(scores):
            raise RuntimeError(f'interaction rule: {len(dists)} poses rebuilt but {len(scores)} scores')
        score, n_ok = rule_score(scores, dists, rule['cutoff'], rule['penalty'])
        self.last_extra = {'rule': rec['inter'], 'plain': min(scores), 'n_ok': n_ok, 'n_poses': len(scores), 'has_basic': bool(nidx),
                           'dmin': (None if min(dists) == float('inf') else round(min(dists), 2))}
        return score

    # Element-and-connectivity graph with every bond single and no charges: robust to bond-order, charge and aromaticity differences between the docked and the crystal ligand.
    def _skeleton(self, m):
        Chem = self.Chem
        rw = Chem.RWMol()
        for a in m.GetAtoms():
            rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
        for b in m.GetBonds():
            rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
        sk = rw.GetMol()
        sk.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(sk)
        return sk

    # RMSD (A) of every docked pose to the crystal ligand, in place with no alignment.  The poses are rebuilt as an RDKit molecule, the docked atoms are mapped onto the crystal atoms through the
    # connectivity skeleton (all symmetry-equivalent mappings) and the best mapping is taken per pose.  Raises ValueError if the poses or the crystal file cannot be read, or the atom counts
    # or the connectivity differ.
    def pose_rmsds(self, poses_pdbqt, crystal_pdb):
        Chem, ns = self.Chem, self.ns
        pm = ns.PDBQTMolecule(poses_pdbqt, skip_typing=True)
        mols = ns.RDKitMolCreate.from_pdbqt_mol(pm)
        mol = mols[0] if mols and mols[0] is not None else None
        if mol is None:
            raise ValueError('could not rebuild the docked poses as an RDKit molecule')
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            mol = Chem.RemoveHs(mol, sanitize=False)
        ref = Chem.MolFromPDBFile(crystal_pdb, removeHs=True, sanitize=False)
        if ref is None or ref.GetNumConformers() == 0:
            raise ValueError(f'could not read the crystal ligand {crystal_pdb}')
        sk_ref, sk_dock = self._skeleton(ref), self._skeleton(mol)
        if sk_ref.GetNumAtoms() != sk_dock.GetNumAtoms():
            raise ValueError(f'heavy-atom count differs: crystal ligand {sk_ref.GetNumAtoms()} vs docked ligand {sk_dock.GetNumAtoms()}')
        matches = sk_ref.GetSubstructMatches(sk_dock, uniquify=False, maxMatches=20000)
        if not matches:
            raise ValueError('docked and crystal ligand have different connectivity (no atom mapping found)')
        C = ref.GetConformer().GetPositions()
        return [min_rmsd(conf.GetPositions(), C, matches) for conf in mol.GetConformers()]

    # Redocks a ligand (9 poses) and compares the poses with the crystal pose.  Returns {'scores', 'rmsd', 'rmsd_error', 'method'}.  If the graph-based RMSD fails, the optimistic
    # element-assignment RMSD is used instead and the method records that.
    def redock(self, smi, rec, exhaustiveness, vina_seed, crystal_pdb):
        r = self.ns.dock_ligand(smi, rec['pdbqt'], rec['center'], rec['size'], exhaustiveness=exhaustiveness, n_poses=REDOCK_POSES, vina_seed=vina_seed)
        if r is None:
            raise RuntimeError('no pose returned')
        out = {'scores': [float(s) for s in r['all_scores']], 'rmsd': None, 'rmsd_error': None, 'method': None}
        try:
            out['rmsd'], out['method'] = [float(x) for x in self.pose_rmsds(r['poses_pdbqt'], crystal_pdb)], 'graph'
        except Exception as e:
            out['rmsd_error'] = (str(e).splitlines()[0][:200] if str(e) else type(e).__name__)
            # fallback: topology-free assignment (optimistic)
            try:
                out['rmsd'] = [float(x) for x in pose_assignment_rmsds(r['poses_pdbqt'], crystal_pdb)]
                out['method'] = 'assignment (graph mapping failed: ' + out['rmsd_error'][:80] + ')'
                out['rmsd_error'] = None
            except Exception as e2:
                out['rmsd_error'] += ' | fallback: ' + (str(e2).splitlines()[0][:120] if str(e2) else type(e2).__name__)
        return out

    # Checks the pharmacophore and glutamate-tail rules on the known compounds in SELFTEST.  Returns failure messages (empty = as intended).
    def self_test(self):
        bad = []
        self.check_smarts(DHFR_PHARMACOPHORE + GLU_TAIL)
        for smi, want, name in SELFTEST:
            c, _ = self.clean(smi, mixtures='largest')
            got = bool(c) and self.matches(c, DHFR_PHARMACOPHORE)
            if got != want:
                bad.append(f"{name}: pharmacophore match {got}, expected {want}")
        if not self.matches(self.clean(SELFTEST[4][0], mixtures='largest')[0], GLU_TAIL):
            bad.append('methotrexate: glutamate tail not recognised')
        if self.matches(self.clean(SELFTEST[0][0], mixtures='largest')[0], GLU_TAIL):
            bad.append('trimethoprim: glutamate tail wrongly recognised')
        return bad


# crude SMILES tokenizer that FakeBackend uses to count heavy atoms
_TOK = re.compile(r'\[[^\]]+\]|Br|Cl|[BCNOPSFI]|[bcnops]')


# No RDKit and no Vina: crude SMILES tokenising and a made-up scoring rule.  ONLY for checking the plumbing and the report layout (--fake); the numbers mean nothing.
class FakeBackend:

    # binders: receptor label -> SMILES that get a made-up bonus
    def __init__(self):
        self.binders = {}

    # Minimal stand-in for RealBackend.clean: rejects empty and 'failed' strings, mixtures (or keeps the longest part) and boron.
    def clean(self, smi, mixtures='skip'):
        if not isinstance(smi, str) or not smi.strip() or smi.strip().lower() == 'failed':
            return None, 'unparsable'
        parts = smi.strip().split('.')
        if len(parts) > 1:
            if mixtures == 'skip':
                return None, 'multi_component'
            smi = max(parts, key=len)
        else:
            smi = parts[0]
        if re.search(r'\[B|B\(|B[^r]', smi):
            return None, 'unsupported:B'
        return smi, 'ok'

    # Same string back (no stereochemistry handling).
    def flat(self, smi):
        return smi

    # Same string back.
    def potency_key(self, smi):
        return smi

    # Crude heavy-atom count: the number of SMILES tokens.
    def _ha(self, smi):
        return sum(1 for t in _TOK.findall(smi))

    # Made-up properties derived from the string (a hash gives logP) and a set of 3-grams as the 'fingerprint'.
    def describe(self, smi):
        h = int(hashlib.md5(smi.encode()).hexdigest(), 16)
        return {'ha': self._ha(smi), 'logp': (h % 400) / 100 - 1, 'hbd': smi.count('N') % 4, 'cls': 0,
                'fp': {smi[i:i + 3] for i in range(max(1, len(smi) - 2))}}

    # Jaccard similarity of the 3-gram sets.
    def similarities(self, fp, fps):
        return [len(fp & f) / max(1, len(fp | f)) for f in fps]

    # Nothing to check in fake mode.
    def check_smarts(self, patterns):
        pass

    # crude stand-in for the DHFR head-group test (fake mode only)
    def matches(self, smi, patterns):
        return smi.count('N') >= 3 and ('C(N)=N' in smi or 'N)N' in smi or 'NC(N)' in smi or '(N)N' in smi)

    # Made-up score: more negative with size, plus noise from a hash of molecule, receptor and seed, and a bonus of 3 for the made-up binders so that calibration has something to find.
    def dock(self, smi, rec, exhaustiveness, vina_seed):
        h = int(hashlib.md5((smi + rec['label'] + str(vina_seed)).encode()).hexdigest(), 16)
        noise = ((h % 2000) / 1000 - 1) * 0.9
        sc = -0.30 * self._ha(smi) - 1.0 + noise
        if smi in self.binders.get(rec['label'].split('@')[0], ()):
            sc -= 3.0
        return sc

    # Made-up redocking scores and RMSDs.
    def redock(self, smi, rec, exhaustiveness, vina_seed, crystal_pdb):
        h = int(hashlib.md5((smi + rec['label']).encode()).hexdigest(), 16)
        sc = sorted([-9.0 - (h % 7) / 10.0 + i * 0.3 for i in range(REDOCK_POSES)])
        rm = [0.8 + ((h >> (3 * i)) % 50) / 10.0 for i in range(REDOCK_POSES)]
        return {'scores': sc, 'rmsd': rm, 'rmsd_error': None, 'method': 'fake'}

    # Nothing to test in fake mode.
    def self_test(self):
        return []


# AUC where lower is better: P(a < d) + 0.5 P(a == d), the chance that a random 'a' scores better (lower) than a random 'd'.  Returns (auc, Hanley-McNeil standard error), nan if either group is empty.
def auc_lower(a, d):
    a, d = np.asarray(a, float), np.asarray(d, float)
    na, nd = len(a), len(d)
    if na == 0 or nd == 0:
        return float('nan'), float('nan')
    auc = float((a[:, None] < d[None, :]).mean() + 0.5 * (a[:, None] == d[None, :]).mean())
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (na - 1) * (q1 - auc ** 2) + (nd - 1) * (q2 - auc ** 2)) / (na * nd)
    return auc, float(np.sqrt(max(var, 0.0)))


# Bootstrap 95% interval (2.5th and 97.5th percentiles over B resamples of both groups) of the statistic fn(a, d); nan if either group has fewer than 2 values.
def boot_ci(fn, a, d, rng, B=2000):
    a, d = np.asarray(a, float), np.asarray(d, float)
    if len(a) < 2 or len(d) < 2:
        return float('nan'), float('nan')
    v = [fn(a[rng.integers(0, len(a), len(a))], d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# The AUC alone, without its standard error (for boot_ci).
def _auc_only(a, d):
    return auc_lower(a, d)[0]


# Fits score ~ slope * heavy_atoms + intercept on the decoys.  Falls back to 'no correction' (slope 0, intercept = median score) if there are fewer than 10 points or fewer than 3 distinct sizes.
def fit_size(ha, score):
    ha, score = np.asarray(ha, float), np.asarray(score, float)
    if len(ha) >= 10 and len(set(ha.tolist())) >= 3:
        slope, icpt = np.polyfit(ha, score, 1)
        return float(slope), float(icpt)
    return 0.0, float(np.median(score)) if len(score) else 0.0


# Size-corrected score: the score minus the fitted line.  Below 0 means better than a decoy of the same size.
def resid(ha, score, fit):
    return np.asarray(score, float) - (fit[0] * np.asarray(ha, float) + fit[1])


# Share of the decoys that this residual beats (lower = better); 0.5 = typical decoy, 1.0 = better than every decoy.
def percentile_lower(r, decoy_r):
    d = np.asarray(decoy_r, float)
    r = np.asarray(r, float)
    if len(d) == 0:
        return np.full(r.shape, np.nan)
    return ((d[None, :] > r[:, None]).mean(axis=1) + 0.5 * (d[None, :] == r[:, None]).mean(axis=1))


# Ranks (from 0) with ties given the average rank.
def _rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):
        idx = x == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


# Spearman rank correlation; nan if there are fewer than 5 valid pairs or one of the variables is constant.
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float('nan')
    return float(np.corrcoef(_rank(x[ok]), _rank(y[ok]))[0, 1])


# Parses a stringified list from the CSV; [] for a missing or empty cell, a single literal is wrapped in a list.
def parse_list(v):
    if v is None or (isinstance(v, float) and v != v) or v in ('', '[]'):
        return []
    try:
        p = ast.literal_eval(v)
        return p if isinstance(p, list) else [p]
    except Exception:
        return []


# Reads the ligand CSV into a table with columns smi, name, stage, targets (a list), clinical (True if the stage is non-empty and not investigative / patented / preclinical) and
# name_l (lower-case name).  Rows without a usable SMILES are dropped.
def read_library(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'Canonical Smiles': 'smi', 'Drug Name': 'name', 'Approved or Clinical': 'stage'})
    df['targets'] = df['Targets for Drug'].apply(parse_list)
    df = df[df['smi'].apply(lambda s: isinstance(s, str) and s.strip() != '' and s.strip().lower() != 'failed')].copy()
    df['clinical'] = df['stage'].apply(lambda s: isinstance(s, str) and s.strip() != '' and not any(k in s.lower() for k in NON_CLINICAL))
    df['name_l'] = df['name'].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


# SMILES -> drug name (the first non-empty name wins).
def name_table(lib):
    names = {}
    for s, n in zip(lib['smi'], lib['name']):
        n = str(n)
        if s not in names or names[s] in ('nan', ''):
            names[s] = n
    return names


# Builds the actives and decoys used to calibrate one receptor.  Actives: library ligands of the target codes (clinical-stage only, or all stages), cleaned, within MIN_HA-MAX_HA heavy atoms,
# required to carry active_smarts, minus drop_names and (with a potency table) only those with potency <= max_nm, plus the curated actives in extra_smiles.  Each active has a
# 'stratum': 'curated', 'clinical' or 'research' (investigative / patented, all-stage mode only); in all-stage mode curated actives take precedence over a library duplicate and the
# clinical and research actives are sampled separately (n_actives and n_research).  Decoys: clinical-stage library drugs that are not ligands of the target or of decoy_exclude_codes, do
# not carry exclude_decoy_smarts, are not known binders (potency <= decoy_nm) and are not close analogs of an active; each active gets up to dpa decoys matched on heavy atoms, logP,
# H-bond donors and charge class, with progressively looser tiers.  All draws use a generator seeded with `seed`, so a rerun rebuilds the same set and cached dockings stay valid.
# Returns (actives, decoys, pool), where pool is every decoy candidate that survived the filters (re-used by easy_check).
def build_calibration_set(be, lib, codes, extra_smiles, drop_names, decoy_exclude_codes, n_actives=N_ACTIVES, dpa=DECOYS_PER_ACTIVE, seed=CAL_SEED,
                          all_stages=False, active_smarts=None, exclude_decoy_smarts=None, n_research=N_RESEARCH,
                          potency=None, max_nm=None, decoy_nm=None, keep_unknown=False):
    rng = np.random.default_rng(seed)
    codes, excl = set(codes), set(decoy_exclude_codes)
    stats = {'multi_component': 0, 'unsupported': 0, 'unparsable': 0, 'dropped_by_name': 0, 'no_head_group': 0, 'no_potency': 0, 'weak': 0, 'potent_decoys': 0}
    has_code = lib['targets'].apply(lambda t: bool(codes & set(t)))
    has_excl = lib['targets'].apply(lambda t: bool(excl & set(t)))
    dropped = lib['name_l'].isin(set(drop_names))

    # Cleans one SMILES and describes it.  Returns the description (plus smi, flat form and potency) or None if the molecule is dropped: it cannot be cleaned, lacks a required pharmacophore,
    # carries a forbidden one, or is outside the size range.  The reasons are counted in stats.
    def prep(smi, size_filter=True, require=None, forbid=None):
        c, why = be.clean(smi, mixtures='skip')
        if c is None:
            key = 'multi_component' if why == 'multi_component' else 'unsupported' if why.startswith('unsupported') else 'unparsable'
            stats[key] += 1
            return None
        if require and not be.matches(c, require):
            stats['no_head_group'] += 1
            return None
        if forbid and be.matches(c, forbid):
            return None
        d = be.describe(c)
        if d is None or (size_filter and not (MIN_HA <= d['ha'] <= MAX_HA)):
            return None
        d['smi'] = c
        d['flat'] = be.flat(c)
        d['nM'] = potency.get(be.potency_key(c)) if potency is not None else None
        return d

    # Potency filter for an active: True (and counted) if it has no ChEMBL measurement or is weaker than max_nm.  With keep_unknown an active without a measurement is kept.
    def too_weak(p):
        if potency is None or not max_nm:
            return False
        if p['nM'] is None:
            if keep_unknown:
                return False
            stats['no_potency'] += 1
            return True
        if p['nM'] > max_nm:
            stats['weak'] += 1
            return True
        return False

    seen, valid = set(), []
    n_rows = int(has_code.sum())
    n_clin = int((has_code & lib['clinical']).sum())
    # all_stages also takes investigative and patented ligands
    act_mask = has_code if all_stages else (has_code & lib['clinical'])
    # curated actives, which take precedence over a library duplicate in all-stage mode
    cur_set = set()
    if all_stages:
        for s_ in extra_smiles:
            c_, _ = be.clean(s_, mixtures='skip')
            if c_:
                cur_set.add(c_)
    # usable library actives, in library order
    for i in lib.index[act_mask]:
        if dropped[i]:
            stats['dropped_by_name'] += 1
            continue
        p = prep(lib.at[i, 'smi'], require=active_smarts)
        if p is None or p['smi'] in seen or p['smi'] in cur_set or too_weak(p):
            continue
        seen.add(p['smi'])
        p['name'] = str(lib.at[i, 'name'])
        p['stratum'] = 'clinical' if bool(lib.at[i, 'clinical']) else 'research'
        valid.append(p)
    # sample the actives at random from the usable ones (the clinical and the research ones separately in all-stage mode)
    if all_stages:
        clin = [p for p in valid if p['stratum'] == 'clinical']
        resv = [p for p in valid if p['stratum'] == 'research']
        ch_c = list(rng.choice(len(clin), size=min(n_actives, len(clin)), replace=False)) if clin else []
        ch_r = list(rng.choice(len(resv), size=min(n_research, len(resv)), replace=False)) if resv else []
        actives = [dict(clin[i], source='library') for i in ch_c] + [dict(resv[i], source='library') for i in ch_r]
    else:
        chosen = list(rng.choice(len(valid), size=min(n_actives, len(valid)), replace=False)) if valid else []
        actives = [dict(valid[i], source='library') for i in chosen]
    # curated actives are added after the sampling and are not size-filtered
    n_extra = 0
    for s in extra_smiles:
        p = prep(s, size_filter=False)
        if p is None or p['smi'] in seen or too_weak(p):
            continue
        if active_smarts and not be.matches(p['smi'], active_smarts):
            print(f"   [calibration set] NOTE: curated active lacks the head group: {s[:60]}")
        seen.add(p['smi'])
        actives.append(dict(p, source='curated', stratum='curated', name='(curated active)'))
        n_extra += 1
    print(f"   [calibration set] {n_rows} library rows carry {sorted(codes)}; {n_clin} clinical-stage; {len(valid)} usable{' (ALL stages)' if all_stages else ' (clinical-stage only)'}; "
          f"{stats['dropped_by_name']} dropped by name"
          + (f"; {stats['no_head_group']} lack the pharmacophore" if active_smarts else '')
          + f"; using {len(actives) - n_extra} + {n_extra}/{len(extra_smiles)} curated actives"
          + ((f"; potency filter <= {max_nm:g} nM: {stats['weak']} measured-weaker dropped, actives without ChEMBL data KEPT" if keep_unknown else f"; potency filter <= {max_nm:g} nM: {stats['no_potency']} without ChEMBL data and {stats['weak']} weaker dropped") if potency is not None and max_nm else '')
          + ("  [" + ', '.join(f"{k} {sum(1 for x in actives if x['stratum'] == k)}" for k in ('curated', 'clinical', 'research')) + "]" if all_stages else ''))

    # decoy candidates: clinical-stage library drugs that are not ligands of this target or of the excluded codes, at most DECOY_POOL of them, sampled at random
    pool_idx = np.array(lib.index[(~has_code) & (~has_excl) & lib['clinical']])
    if len(pool_idx) > DECOY_POOL:
        pool_idx = rng.choice(pool_idx, size=DECOY_POOL, replace=False)
    pool = []
    for i in pool_idx:
        p = prep(lib.at[int(i), 'smi'], forbid=exclude_decoy_smarts)
        if p is None or p['smi'] in seen:
            continue
        if potency is not None and decoy_nm and p['nM'] is not None and p['nM'] <= decoy_nm:
            stats['potent_decoys'] += 1
            continue
        p['name'] = str(lib.at[int(i), 'name'])
        seen.add(p['smi'])
        pool.append(p)
    if potency is not None and decoy_nm:
        print(f"   [calibration set] {stats['potent_decoys']} candidate decoys dropped as known binders of the target (potency <= {decoy_nm:g} nM)")
    if actives and pool:
        afps = [a['fp'] for a in actives]
        before = len(pool)
        pool = [p for p in pool if max(be.similarities(p['fp'], afps)) < MAX_DECOY_SIM]
        print(f"   [calibration set] {before - len(pool)} candidate decoys dropped as close analogs of an active; pool {len(pool)}")
    # matching tiers (heavy-atom window, logP window, donor window, same charge class required), tried in order until an active has enough candidates
    tiers = ((2, 1.0, 1, True), (4, 2.0, 2, True), (4, 2.0, 2, False))
    used, decoys = set(), []
    for a in actives:
        cands = []
        for ha_t, lp_t, hbd_t, same in tiers:
            cands = [i for i, p in enumerate(pool) if i not in used and abs(p['ha'] - a['ha']) <= ha_t and abs(p['logp'] - a['logp']) <= lp_t
                     and abs(p['hbd'] - a['hbd']) <= hbd_t and (not same or p['cls'] == a['cls'])]
            if len(cands) >= dpa:
                break
        if cands:
            for i in rng.choice(cands, size=min(dpa, len(cands)), replace=False):
                used.add(int(i))
                decoys.append(pool[int(i)])
    if actives and decoys:
        ma, md = np.median([x['ha'] for x in actives]), np.median([x['ha'] for x in decoys])
        print(f"   [calibration set] {len(actives)} actives / {len(decoys)} decoys; median heavy atoms {ma:.1f} vs {md:.1f}"
              + ("  <-- size mismatch > 1 atom" if abs(ma - md) > 1 else ""))
    return actives, decoys, pool


# Reads a potency CSV (columns smiles, nM, ...) into {compound key: median nM over its measurements}; rows without a positive nM are skipped.
def load_potency(be, path):
    import csv as _csv
    import statistics
    vals = {}
    for r in _csv.DictReader(open(path, encoding='utf-8')):
        try:
            nm = float(r['nM'])
        except (KeyError, TypeError, ValueError):
            continue
        k = be.potency_key(r.get('smiles', ''))
        if k and nm > 0:
            vals.setdefault(k, []).append(nm)
    return {k: float(statistics.median(v)) for k, v in vals.items()}


# Append-only cache of docking results in a JSON-lines file (key -> score, error and optional extra record).  Earlier records are loaded at start and every new one is written and flushed
# at once, so a run can be stopped and resumed.
class DockCache:
    # Loads the existing records (unreadable lines are ignored).
    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            for line in open(path, encoding='utf-8'):
                try:
                    r = json.loads(line)
                    self.d[r['key']] = r
                except Exception:
                    pass
        self.fh = None

    # Returns the cached record for a key, or None.
    def get(self, key):
        return self.d.get(key)

    # Stores a record in memory and appends it to the file.
    def put(self, key, score, error=None, extra=None):
        r = {'key': key, 'score': score, 'error': error}
        if extra is not None:
            r['extra'] = extra
        self.d[key] = r
        if self.fh is None:
            self.fh = open(self.path, 'a', encoding='utf-8')
        self.fh.write(json.dumps(r) + '\n')
        self.fh.flush()


# Short hash that identifies a receptor: its pdbqt content and box and, when an interaction rule is on, the rule and its constants.  It is part of every cache key, so a changed receptor or
# rule never reuses old dockings, and plain-Vina entries keep their keys.
def receptor_signature(rec):
    h = hashlib.md5()
    try:
        h.update(open(rec['pdbqt'], 'rb').read())
    except Exception:
        h.update(b'missing')
    h.update(json.dumps([rec['center'], rec['size']]).encode())
    if rec.get('inter'):
        h.update(('|rule:' + str(rec['inter']) + json.dumps(INTERACTION_RULES.get(rec['inter']), sort_keys=True, default=list)).encode())
    return h.hexdigest()[:8]


# Docks molecules at receptors through the backend, with caching, resumable progress output and per-molecule error handling.  members maps a receptor label to its ensemble members;
# a molecule's score at that label is the best (lowest) over them.
class Docker:

    # Stores the backend, cache, receptor table and settings; members (ensemble label -> [label, label@B, ...]) starts empty.
    def __init__(self, be, cache, receptors, exhaustiveness, retry_failed=False, report_only=False):
        self.be, self.cache, self.R, self.exh = be, cache, receptors, exhaustiveness
        self.retry, self.report_only = retry_failed, report_only
        self.errors, self.n_done, self.t0 = {}, 0, time.time()
        self.members = {}

    # The ensemble members of a receptor label (just the label itself if it has none).
    def mem(self, lab):
        return self.members.get(lab, [lab])

    # Expands (SMILES, label) jobs so that every ensemble member of a label is docked.
    def expand(self, items):
        return [(smi, m) for smi, lab in items for m in self.mem(lab)]

    # Cache key: label | receptor signature | exhaustiveness | Vina seed | SMILES.
    def key(self, smi, lab, exh=None, vseed=42):
        r = self.R[lab]
        return f"{lab}|{r['sig']}|{exh or self.exh}|{vseed}|{smi}"

    # Cached score of a molecule at a receptor: the best over the ensemble members, or None if any member is missing or failed.
    def lookup(self, smi, lab, exh=None, vseed=42):
        sc = []
        for m in self.mem(lab):
            r = self.cache.get(self.key(smi, m, exh, vseed))
            if r is None or r['score'] is None:
                return None
            sc.append(r['score'])
        return min(sc)

    # The 'extra' record stored with a cached docking (interaction-rule details), or None.
    def extra_of(self, smi, lab, exh=None, vseed=42):
        r = self.cache.get(self.key(smi, lab, exh, vseed))
        return None if r is None else r.get('extra')

    # The (SMILES, label) jobs that are not in the cache yet (failed ones too with --retry-failed), ensemble members expanded, without duplicates.
    def pending(self, items, exh=None, vseed=42):
        out, seen = [], set()
        for smi, lab in self.expand(items):
            k = self.key(smi, lab, exh, vseed)
            r = self.cache.get(k)
            if (r is None or (self.retry and r['score'] is None)) and k not in seen:
                out.append((smi, lab))
                seen.add(k)
        return out

    # Docks the jobs that are not cached yet and stores each result at once, so a stopped run resumes.  A failure is cached with its error message and listed at the end.  In report-only
    # mode only the number of missing dockings is reported.  exh and vseed override the exhaustiveness and the Vina seed (used by the reproducibility check).
    def run(self, items, what, exh=None, vseed=42):
        items = self.expand(items)
        todo = self.pending(items, exh, vseed)
        if self.report_only:
            if todo:
                print(f"   [{what}] {len(todo)} dockings missing from the cache (report-only: skipped)")
            return
        if not todo:
            return
        print(f"   [{what}] {len(todo)} dockings to do ({len(items) - len(todo)} already cached)")
        t0 = time.time()
        for i, (smi, lab) in enumerate(todo, 1):
            t1 = time.time()
            self.be.last_extra = None
            try:
                sc, err = self.be.dock(smi, self.R[lab], exh or self.exh, vseed), None
            except Exception as e:
                sc, err = None, str(e).splitlines()[0][:160] if str(e) else type(e).__name__
            if sc is None and err is None:
                err = 'no pose / embedding failed'
            if err:
                self.errors.setdefault(err, []).append(smi)
            self.cache.put(self.key(smi, lab, exh, vseed), sc, err, getattr(self.be, 'last_extra', None) if sc is not None else None)
            if i <= 2 or i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"      [{i}/{len(todo)}] {lab:<9} {'FAILED' if sc is None else '%.2f' % sc:>7}  ({time.time() - t1:.0f}s; ~{el / i * (len(todo) - i) / 60:.0f} min left)")

    # Redocks the co-crystal ligand (cached).  Returns (result dict or None, error text or None); the result is {'scores': [...], 'rmsd': [...] or None, 'rmsd_error': str or None}.  With
    # --retry-failed a cached failure is tried again; in report-only mode nothing is docked.
    def redock(self, lab, smi, crystal, exh=None, vseed=42):
        k = 'redock|' + self.key(smi, lab, exh, vseed)
        r = self.cache.get(k)
        bad = r is not None and (r.get('error') or (r.get('extra') or {}).get('rmsd') is None)
        if r is not None and not (self.retry and bad):
            return r.get('extra'), r.get('error')
        if self.report_only:
            return None, 'not in the cache (report-only)'
        t1 = time.time()
        try:
            extra, err = self.be.redock(smi, self.R[lab], exh or self.exh, vseed, crystal), None
        except Exception as e:
            extra, err = None, (str(e).splitlines()[0][:200] if str(e) else type(e).__name__)
        self.cache.put(k, None, err, extra)
        print(f"   [redock {lab}] {'FAILED: ' + err if err else 'done'}  ({time.time() - t1:.0f}s)")
        return extra, err

    # Prints the most frequent docking error messages with an example molecule.
    def failure_summary(self):
        if self.errors:
            print("\n[docking failures] (cached as failed; use --retry-failed to try again)")
            for e, l in sorted(self.errors.items(), key=lambda kv: -len(kv[1]))[:8]:
                print(f"   {len(l):>3} x {e}   e.g. {l[0][:60]}")


# Easy-decoy diagnostic: decoys matched on heavy atoms only (+-2), a few per active, to separate 'cannot tell binders from random drugs' from 'fails only against property-matched
# decoys'.  Docks them and returns (heavy atoms, score) for those that docked.
def easy_check(dock, label, actives, pool, a):
    rng = np.random.default_rng(a.seed + 1000)
    used, easy = set(), []
    for x in actives:
        cands = [i for i, p in enumerate(pool) if i not in used and abs(p['ha'] - x['ha']) <= 2]
        if cands:
            for i in rng.choice(cands, size=min(a.easy_per_active, len(cands)), replace=False):
                used.add(int(i))
                easy.append(pool[int(i)])
    dock.run([(p['smi'], label) for p in easy], f"easy-decoy diagnostic {label}")
    es = [(p['ha'], dock.lookup(p['smi'], label)) for p in easy]
    return [(h, s_) for h, s_ in es if s_ is not None]


# Enrichment factor at 20%: the share of actives among the best-scoring 20% of actives plus decoys, divided by their share overall (1 = no enrichment).
def ef20_of(sa, sd):
    allsc = np.concatenate([sa, sd])
    isa = np.concatenate([np.ones(len(sa)), np.zeros(len(sd))])
    top = np.argsort(allsc)[:max(1, int(round(0.2 * len(allsc))))]
    return float(isa[top].mean() / isa.mean())


# The pre-registered trust rule.  Status 'usable' = AUC >= 0.70 with AUC - SE > 0.5; otherwise 'weak' if AUC >= 0.60, else 'non_discriminating'.  Trusted = usable and size-corrected
# AUC >= 0.60.  n must reach `need`, otherwise 'too_few_actives'.  Returns (status, trusted).
def decision_status(auc, se, auc_sc, n, need):
    if n < need:
        return 'too_few_actives', False
    status = 'usable' if (auc >= AUC_USABLE and auc - se > 0.5) else 'weak' if auc >= AUC_WEAK else 'non_discriminating'
    return status, bool(status == 'usable' and auc_sc >= SIZE_CORRECTED_MIN)


# Calibrates one receptor: builds the actives and decoys, docks them (and the reference ligand) and measures how well Vina separates them: AUC with its standard error, the
# size-corrected AUC (residuals of score ~ heavy atoms fitted on the decoys), EF20, sub-analyses by stratum and without the co-crystal ligand, an optional potency-versus-score check, and
# when requested the interaction-rule comparison, a reproducibility check and the easy-decoy diagnostic.  The pre-registered trust rule (decision_status) is applied.  Returns a dict with
# the status, trusted flag, arrays and statistics the report needs, or an 'uncalibrated' dict if too few actives or decoys were docked.
def calibrate(be, dock, lib, case, rec, a, rng):
    label = rec['label']
    stages = a.active_stages if a.active_stages in ('clinical', 'all') else case.get('active_stages', 'clinical')
    fresh = case.get('decision', 'all') == 'fresh'
    print(f"\n[calibration] {label} (case {case['key']}, target {case['target']}; actives: {stages} stages"
          + (f"; box padded +{a.box_pad:g} A" if a.box_pad else '') + (f"; ensemble {dock.mem(label)}" if len(dock.mem(label)) > 1 else '')
          + (f"; INTERACTION RULE {rec['inter']}" if rec.get('inter') else '') + ")")
    codes = rec.get('dataset_target_codes') or [case['target']]
    actives, decoys, pool = build_calibration_set(be, lib, codes, rec.get('calibration_active_smiles') or [], set(case.get('drop_active_names', [])),
                                                  case['decoy_exclude_codes'], n_actives=a.n_actives, all_stages=(stages == 'all'),
                                                  active_smarts=case.get('active_smarts'), exclude_decoy_smarts=case.get('exclude_decoy_smarts'),
                                                  n_research=a.n_research, potency=case.get('_potency'), max_nm=(a.max_nm if a.max_nm > 0 else None),
                                                  decoy_nm=(a.decoy_nm if a.decoy_nm > 0 else None), keep_unknown=a.keep_no_potency)
    pot_on = case.get('_potency') is not None
    out = {'label': label, 'case': case['key'], 'status': 'uncalibrated', 'trusted': False, 'actives': actives, 'decoys': decoys}
    if len(actives) < MIN_ACTIVES or len(decoys) < 2 * MIN_ACTIVES:
        out['reason'] = f"only {len(actives)} actives / {len(decoys)} decoys (need >= {MIN_ACTIVES} / {2 * MIN_ACTIVES})"
        print(f"   -> UNCALIBRATED: {out['reason']}")
        return out
    dock.run([(x['smi'], label) for x in actives + decoys], f"calibration {label}")
    if rec.get('reference'):
        dock.run([(rec['reference'], label)], f"reference {label}")
    for x in actives + decoys:
        x['score'] = dock.lookup(x['smi'], label)
    act = [x for x in actives if x['score'] is not None]
    dec = [x for x in decoys if x['score'] is not None]
    out.update(n_act=len(act), n_dec=len(dec), reference=dock.lookup(rec['reference'], label) if rec.get('reference') else None)
    if len(act) < MIN_ACTIVES or len(dec) < 2 * MIN_ACTIVES:
        out['reason'] = f"too few docked ({len(act)} actives, {len(dec)} decoys)" + (' (docking not run)' if dock.report_only else '')
        print(f"   -> UNCALIBRATED: {out['reason']}")
        return out
    sa, sd = np.array([x['score'] for x in act]), np.array([x['score'] for x in dec])
    ha_a, ha_d = np.array([x['ha'] for x in act], float), np.array([x['ha'] for x in dec], float)
    # the size correction is learned from the decoys only
    fit = fit_size(ha_d, sd)
    ra, rd = resid(ha_a, sa, fit), resid(ha_d, sd, fit)
    strat = np.array([x.get('stratum', 'clinical') for x in act])
    # the actives that the trust decision is based on: the research stratum for 'fresh' cases, all actives otherwise
    dmask = (strat == 'research') if fresh else np.ones(len(act), bool)
    nd = int(dmask.sum())
    auc, se = auc_lower(sa[dmask], sd) if nd else (float('nan'), float('nan'))
    auc_sc, se_sc = auc_lower(ra[dmask], rd) if nd else (float('nan'), float('nan'))
    auc_all, _ = auc_lower(sa, sd)
    auc_sc_all, _ = auc_lower(ra, rd)
    # pre-registered trust rule (at least MIN_DECISION decision-set actives for 'fresh' cases, MIN_ACTIVES otherwise)
    status, trusted = decision_status(auc, se, auc_sc, nd, MIN_DECISION if fresh else MIN_ACTIVES)
    ef20 = ef20_of(sa[dmask], sd) if nd else float('nan')
    cc = be.flat(rec['reference']) if rec.get('reference') else None
    is_cc = np.array([bool(cc) and x['flat'] == cc for x in act])

    # Size-corrected AUC and count for the actives selected by a mask (nan if fewer than 3).
    def sub(mask):
        return (auc_lower(ra[mask], rd)[0], int(mask.sum())) if mask.sum() >= 3 else (float('nan'), int(mask.sum()))
    strata = {k: sub(strat == k) for k in ('curated', 'clinical', 'research')}
    auc_nocc, n_nocc = sub(~is_cc & dmask)
    le_act = float(np.mean([-x['score'] / x['ha'] for x, m in zip(act, dmask) if m])) if nd else float('nan')
    rho, n_rho = float('nan'), 0
    if pot_on:
        nm_ = np.array([x['nM'] if x.get('nM') is not None else np.nan for x in act], float)
        okp = dmask & ~np.isnan(nm_)
        n_rho = int(okp.sum())
        if n_rho >= 8:
            # positive = the more potent a ligand, the better (more negative) its size-corrected score
            rho = spearman(9.0 - np.log10(nm_[okp]), -ra[okp])
    out.update(status=status, trusted=trusted, decision='research stratum only' if fresh else 'all actives', n_decision=nd, auc=auc, se=se,
               auc_sc=auc_sc, se_sc=se_sc, ef20=ef20, fit=fit, resid_decoys=rd, resid_actives=ra, auc_all=auc_all, auc_sc_all=auc_sc_all,
               act_med=float(np.median(sa[dmask])) if nd else float('nan'), dec_med=float(np.median(sd)),
               ha_a=float(np.median(ha_a[dmask])) if nd else float('nan'), ha_d=float(np.median(ha_d)), le_act=le_act,
               underpowered=bool(nd < 20 or se > 0.08), strata={k: {'auc_sc': v[0], 'n': v[1]} for k, v in strata.items()},
               auc_nocc=auc_nocc, has_cc=bool(is_cc.any()), potency_rho=rho, potency_n=n_rho, potency_filter_nm=(a.max_nm if (pot_on and a.max_nm > 0) else None))
    pct_a = percentile_lower(ra, rd)
    print("   known actives, best score first   (score | heavy atoms | percentile among decoys, size-corrected | stratum | "
          + ("potency | " if pot_on else "") + "name):")
    for x, p_ in sorted(zip(act, pct_a), key=lambda t: t[0]['score'])[:70]:
        tag = '   <- co-crystal ligand' if cc and x['flat'] == cc else ''
        pn = (('%7.1f nM | ' % x['nM']) if x.get('nM') is not None else '    n/a    | ') if pot_on else ''
        print(f"      {x['score']:7.2f} | {x['ha']:3d} | {p_:.2f} | {x.get('stratum', '?')[:4]} | {pn}{x.get('name', '')[:30]}{tag}")
    print(f"   decoys: score min {sd.min():.2f}, median {np.median(sd):.2f}, max {sd.max():.2f}   (an active that scores worse than the decoy median is a miss)")
    print(f"   -> DECISION SET: {out['decision']} (n {nd}, {len(dec)} decoys): AUC {auc:.2f} +/- {se:.2f} ({status}); size-corrected AUC {auc_sc:.2f} +/- {se_sc:.2f}; "
          f"EF20 {ef20:.2f}; median score {out['act_med']:.2f} vs decoys {np.median(sd):.2f}")
    if fresh:
        print(f"      (all {len(act)} actives together: AUC {auc_all:.2f}, size-corrected {auc_sc_all:.2f}; rule: >= {MIN_DECISION} research actives, AUC >= {AUC_USABLE:.2f} "
              f"with AUC - SE > 0.5, size-corrected >= {SIZE_CORRECTED_MIN:.2f})")
    print("   -> size-corrected AUC by stratum: " + ', '.join(f"{k} {v[0]:.2f} (n {v[1]})" for k, v in strata.items() if v[1])
          + f";  decision set without the co-crystal ligand {auc_nocc:.2f} (n {n_nocc})")
    if pot_on:
        if n_rho < 8:
            print(f"   -> POTENCY CHECK: only {n_rho} actives with potency data: too few for a Spearman check")
        elif rho != rho:
            print(f"   -> POTENCY CHECK: n/a over {n_rho} actives (potencies identical, nothing to rank)")
        else:
            print(f"   -> POTENCY CHECK: Spearman(pKi/pIC50, size-corrected score) = {rho:.2f} over {n_rho} actives with data "
                  "(+ = Vina ranks the more potent ligands better; about 0 = it does not rank potency; run once with --max-nm 0 to see it over the full potency range)")
    # interaction rule: compare with the plain Vina scores from the same dockings (every cached rule docking also stores its plain best score) and with a control that only asks
    # whether the ligand has a basic nitrogen, so the gain can be attributed
    if rec.get('inter'):
        rule = INTERACTION_RULES[rec['inter']]
        exa, exd = [dock.extra_of(x['smi'], label) for x in act], [dock.extra_of(x['smi'], label) for x in dec]
        if exa and exd and all(e and 'plain' in e for e in exa + exd):
            pa_, pd_ = np.array([e['plain'] for e in exa], float), np.array([e['plain'] for e in exd], float)
            fit_p = fit_size(ha_d, pd_)
            plain_auc = auc_lower(pa_[dmask], pd_)[0] if nd else float('nan')
            plain_sc = auc_lower(resid(ha_a, pa_, fit_p)[dmask], resid(ha_d, pd_, fit_p))[0] if nd else float('nan')
            sat_a, sat_d = float(np.mean([e['n_ok'] > 0 for e in exa])), float(np.mean([e['n_ok'] > 0 for e in exd]))
            bas_a, bas_d = float(np.mean([e['has_basic'] for e in exa])), float(np.mean([e['has_basic'] for e in exd]))
            hb_a, hb_d = np.array([bool(e['has_basic']) for e in exa]), np.array([bool(e['has_basic']) for e in exd])
            ctl_a, ctl_d = pa_ + rule['penalty'] * (~hb_a), pd_ + rule['penalty'] * (~hb_d)
            fit_c = fit_size(ha_d, ctl_d)
            ctl_auc = auc_lower(ctl_a[dmask], ctl_d)[0] if nd else float('nan')
            ctl_sc = auc_lower(resid(ha_a, ctl_a, fit_c)[dmask], resid(ha_d, ctl_d, fit_c))[0] if nd else float('nan')
            out.update(inter_rule=rec['inter'], inter_plain_auc=plain_auc, inter_plain_auc_sc=plain_sc, inter_sat_actives=sat_a, inter_sat_decoys=sat_d,
                       inter_basic_actives=bas_a, inter_basic_decoys=bas_d, inter_control_auc=ctl_auc, inter_control_auc_sc=ctl_sc)
            print(f"   -> INTERACTION RULE {rec['inter']} (basic N within {rule['cutoff']:g} A of {'/'.join(rule['atoms'])} of {', '.join(f'{rn}{ri}' for rn, ri in rule['residues'])} in >= 1 of {rule['poses']} poses; otherwise +{rule['penalty']:g} kcal/mol):")
            print(f"      share of ACTIVES satisfying it {sat_a:.0%}, of DECOYS {sat_d:.0%}   (ligands with a basic nitrogen: actives {bas_a:.0%}, decoys {bas_d:.0%})")
            print(f"      same dockings WITHOUT the rule: AUC {plain_auc:.2f}, size-corrected {plain_sc:.2f}   |   WITH the rule (decision set above): AUC {auc:.2f}, size-corrected {auc_sc:.2f}"
                  + ("   [ensemble: rule details from the main structure only]" if len(dock.mem(label)) > 1 else ''))
            print(f"      CONTROL (has a basic nitrogen, +{rule['penalty']:g} if not; NO geometry): AUC {ctl_auc:.2f}, size-corrected {ctl_sc:.2f}"
                  "   (if close to the WITH-rule figures, the gain is just 'has an amine'; if near the plain figures, the pose geometry carries it)")
        else:
            print("      (interaction details missing from the cache for some ligands: run without --report-only once)")
    print(f"   -> receptor {'TRUSTED' if trusted else 'NOT TRUSTED (generated molecules will not be interpreted here)'}"
          + ("  [underpowered: wide error bar]" if out['underpowered'] else ''))
    # reproducibility check: re-dock a few actives at exhaustiveness 32 with another seed and report the mean score difference
    if a.noise_check > 0 and not dock.report_only:
        pool_n = [x for x, m in zip(act, dmask) if m] or act
        pick = random.Random(0).sample(pool_n, min(a.noise_check, len(pool_n)))
        dock.run([(x['smi'], label) for x in pick], f"noise check {label} (exhaustiveness 32, other seed)", exh=32, vseed=7)
        diffs = [abs(x['score'] - dock.lookup(x['smi'], label, 32, 7)) for x in pick if dock.lookup(x['smi'], label, 32, 7) is not None]
        if diffs:
            out['noise'] = float(np.mean(diffs))
            print(f"   -> score reproducibility: mean |difference| {np.mean(diffs):.2f} kcal/mol over {len(diffs)} re-docked actives "
                  f"({'sampling noise could blur small effects' if np.mean(diffs) > 0.5 else 'noise is small'})")
    # easy-decoy diagnostic: AUC against decoys matched on size only
    if a.easy_decoys:
        act_d = [x for x, m in zip(act, dmask) if m]
        es = easy_check(dock, label, act_d, pool, a)
        if len(es) >= 15 and act_d:
            he, se_ = np.array([h for h, _ in es], float), np.array([s_ for _, s_ in es], float)
            fe = fit_size(he, se_)
            auc_e_sc, sd_e = auc_lower(resid(ha_a[dmask], sa[dmask], fe), resid(he, se_, fe))
            auc_e_raw = auc_lower(sa[dmask], se_)[0]
            out.update(easy_auc_sc=auc_e_sc, easy_auc_raw=auc_e_raw, easy_n=len(es))
            print(f"   -> EASY-DECOY check (decoys matched on size only, n {len(es)}): AUC {auc_e_raw:.2f}, size-corrected {auc_e_sc:.2f} +/- {sd_e:.2f}")
            if auc_e_sc >= 0.65 and auc_sc < 0.60:
                print("      reading: the pocket separates binders from random drugs but not from property-matched ones, so what it rewards is generic (polarity, charge, "
                      "flexibility), not specific recognition.")
            elif auc_e_sc < 0.60:
                print("      reading: even size-matched random drugs are not separated: suspect the receptor structure / box / preparation, or a Vina limitation for this chemotype.")
        else:
            print(f"   -> EASY-DECOY check: only {len(es)} decoys docked (need >= 15)")
    return out


# Redocks the co-crystal ligand (the receptor's reference SMILES) and compares its poses with the crystal pose.  Returns a dict with status ('ok', 'skipped', 'error' or 'no_rmsd'),
# the top-pose and best-pose RMSD and a verdict: PASS if the top pose is within REDOCK_PASS_A of the crystal pose, PARTIAL if another pose is, else FAIL.
def run_redock(be, dock, case, rec):
    label, ref, cry = rec['label'], rec.get('reference'), rec.get('crystal')
    out = {'label': label, 'status': 'skipped'}
    if not ref:
        out['reason'] = 'no reference SMILES for this receptor'
    elif not cry:
        out['reason'] = f"crystal ligand file not found (expected {rec.get('crystal_name')}; copy it next to the scripts)"
    if 'reason' in out:
        print(f"\n[redock] {label}: skipped - {out['reason']}")
        return out
    extra, err = dock.redock(label, ref, cry)
    if extra is None:
        out.update(status='error', reason=err)
        print(f"\n[redock] {label}: FAILED - {err}")
        return out
    sc = np.array(extra['scores'], float)
    out.update(top_score=float(sc[0]), n_poses=int(len(sc)))
    if extra.get('rmsd') is None:
        out.update(status='no_rmsd', reason=extra.get('rmsd_error'))
        print(f"\n[redock] {label}: reference scored {sc[0]:.2f}, but the pose RMSD could not be computed: {extra.get('rmsd_error')}")
        return out
    rm = np.array(extra['rmsd'], float)
    n = min(len(rm), len(sc))
    bi = int(np.argmin(rm[:n]))
    verdict = 'PASS' if rm[0] <= REDOCK_PASS_A else 'PARTIAL' if rm[:n].min() <= REDOCK_PASS_A else 'FAIL'
    method = extra.get('method') or 'graph'
    out.update(status='ok', top_rmsd=float(rm[0]), best_rmsd=float(rm[:n].min()), best_pose=bi + 1, best_pose_score=float(sc[bi]), verdict=verdict, method=method)
    print(f"\n[redock] {label}: reference scored {sc[0]:.2f}; top pose RMSD to the crystal pose {rm[0]:.2f} A; closest of {n} poses {rm[:n].min():.2f} A "
          f"(pose {bi + 1}, score {sc[bi]:.2f})  -> {verdict}   [RMSD by {method.split(' (')[0]}, in place, no re-alignment]")
    if not method.startswith(('graph', 'fake')):
        print("         NOTE: atoms were paired by element and distance only (the connectivity of the docked and the crystal ligand could not be mapped): "
              "this RMSD is optimistic.  " + method)
    print("         " + {'PASS': 'the box and receptor reproduce the crystal pose.',
                          'PARTIAL': 'a pose near the crystal one is found but not ranked first: sampling is fine, the scoring function prefers another pose.',
                          'FAIL': 'no pose within 2 A of the crystal pose: suspect the receptor preparation or box (or a ligand Vina cannot place).'}[verdict])
    return out


# Reads a generated set (gen_<key>.csv, or the random-seed baseline file with baseline=True), checks that it fits the case and returns up to a.n dockable molecules as dicts (smi, ha, seed,
# sim, seed_name, ph).  A generated set must come from the case's target: at least 50% of its seeds must be ligands of the target (--allow-seed-mismatch overrides), a baseline at most 10%.
# Seeds of dropped-by-name compounds are excluded; where the case has a pharmacophore rule the seed must carry it (generated set) or must not (baseline), so selection is on the seed only,
# never on the output.  Molecules are cleaned (largest fragment), size-filtered (GEN_HA_RANGE) and de-duplicated; the reasons for skipping are printed.
def load_generated(be, case, a, lib, names, baseline=False):
    import csv as _csv
    pat = a.baseline_pattern if baseline else a.gen_pattern
    f = find_file(pat.format(key=case['key']))
    if not f:
        raise SystemExit(f"{pat.format(key=case['key'])} not found: run step 1 of the USAGE (generate_for_request.py{' --random-seeds' if baseline else ''}) first")
    rows = list(_csv.DictReader(open(f, encoding='utf-8')))
    # library ligands of the case's target, to check where the seeds came from
    tgt_smiles = set(lib.loc[lib['targets'].apply(lambda t: case['target'] in t), 'smi'])
    seeds = [r.get('seed_smiles', '') for r in rows if r.get('seed_smiles')]
    frac = np.mean([s in tgt_smiles for s in seeds]) if seeds else float('nan')
    if baseline:
        if seeds and frac > 0.1:
            raise SystemExit(f"{os.path.basename(f)}: {frac:.0%} of the seeds are ligands of {case['target']}: this is not a random-seed baseline "
                             f"(generate it with --random-seeds --target {case['target']}).")
    elif seeds and frac < 0.5 and not a.allow_seed_mismatch:
        raise SystemExit(f"{os.path.basename(f)}: only {frac:.0%} of the seed molecules are ligands of {case['target']} in the library. "
                         f"This file was generated for a different target; regenerate it (USAGE step 1) or pass --allow-seed-mismatch.")
    # seeds whose target annotation looks wrong (drop_seed_names)
    bad_seeds = set(lib.loc[lib['name_l'].isin(set(case.get('drop_seed_names', []))), 'smi'])
    pats = case.get('active_smarts')
    ok_cache = {}

    # True if the seed may be used: always when the case has no pharmacophore rule, otherwise the seed must carry the pharmacophore (generated set) or must not (baseline).  Cached.
    def seed_ok(raw):
        if not pats:
            return True
        if raw not in ok_cache:
            c, _ = be.clean(raw, mixtures='largest') if raw else (None, '')
            ok_cache[raw] = bool(c) and (not be.matches(c, pats) if baseline else be.matches(c, pats))
        return ok_cache[raw]
    out, seen, why = [], set(), {}
    for r in rows:
        raw = r.get('seed_smiles')
        if raw in bad_seeds:
            why['analog of a dropped seed'] = why.get('analog of a dropped seed', 0) + 1
            continue
        if not seed_ok(raw):
            why['seed carries the pharmacophore' if baseline else 'seed lacks the pharmacophore'] = why.get('seed carries the pharmacophore' if baseline else 'seed lacks the pharmacophore', 0) + 1
            continue
        c, reason = be.clean(r['smiles'], mixtures='largest')
        d = be.describe(c) if c else None
        if c is None or d is None:
            why[reason] = why.get(reason, 0) + 1
            continue
        if not (GEN_HA_RANGE[0] <= d['ha'] <= GEN_HA_RANGE[1]):
            why['size'] = why.get('size', 0) + 1
            continue
        if c in seen:
            why['duplicate'] = why.get('duplicate', 0) + 1
            continue
        seen.add(c)
        sc, _ = be.clean(raw, mixtures='largest') if raw else (None, '')
        try:
            sim = float(r.get('similarity_to_seed', 'nan'))
        except ValueError:
            sim = float('nan')
        out.append({'smi': c, 'ha': d['ha'], 'seed': sc, 'sim': sim, 'seed_name': names.get(raw, '') if raw else '',
                    'ph': (bool(be.matches(c, pats)) if pats else None)})
    out = out[:a.n]
    print(f"[set] {'RANDOM-SEED BASELINE' if baseline else 'generated'} {case['key']}: {len(out)} dockable of {len(rows)} in {os.path.basename(f)}"
          + (f"  (skipped: {', '.join(f'{k} {v}' for k, v in why.items())})" if why else '')
          + f";  seeds that are {case['target']} ligands: {frac:.0%}")
    return out


# Runs the docking pipeline up to the report.  The generated sets are loaded first, so a missing or mismatched file fails BEFORE the long calibration.  Then it calibrates every receptor
# structure, redocks the co-crystal ligands if asked, picks for each case the working structure (the best trusted one) and docks the generated molecules, the baseline set and the seeds
# there (and at the other trusted receptors with --cross).  A --diagnose run stops before any generated set is used.  Returns (cal, redock, gens, work, dock_at).
def run(a, be, dock, lib, cases, receptors):
    rng = np.random.default_rng(a.seed)
    names = name_table(lib)
    gens = {} if a.diagnose else {c['key']: load_generated(be, c, a, lib, names) for c in cases}
    if a.baseline and not a.diagnose:
        for c in cases:
            gens[c['key'] + '#baseline'] = load_generated(be, c, a, lib, names, baseline=True)
    cal = {}
    # 1. calibrate every receptor (every structure of every case)
    for c in cases:
        for lab in c['variants']:
            cal[lab] = calibrate(be, dock, lib, c, receptors[lab], a, rng)
    redock = {}
    # 2. redock the co-crystal ligands
    if (a.redock or a.diagnose) and not a.calibration_only:
        print("\n[redock] co-crystal ligand redocking (pose RMSD to the crystal pose)")
        for c in cases:
            for lab in c['variants']:
                redock[lab] = run_redock(be, dock, c, receptors[lab])
    if a.diagnose:
        return cal, redock, gens, {c['key']: c['variants'][0] for c in cases}, []
    # which structure carries each case: the best TRUSTED one by size-corrected AUC; structures that are already ensemble members are shown as diagnostics only
    work = {}
    for c in cases:
        in_ens = [l for l in c['variants'][1:] if l in dock.members.get(c['receptor'], [])]
        cand = [l for l in c['variants'] if l not in in_ens]
        tl = [l for l in cand if cal[l]['trusted']]
        work[c['key']] = max(tl, key=lambda l: cal[l]['auc_sc']) if tl else cand[0]
        if len(c['variants']) > 1:
            print(f"\n[structure choice] {c['key']}: " + (f"{work[c['key']]} (best trusted of {cand})" if tl else f"no structure trusted; {work[c['key']]} shown")
                  + (f"; {in_ens} are members of the ensemble {c['receptor']} and shown as diagnostics only" if in_ens else ''))
    # receptors where generated molecules are docked: the working structures that are trusted (all of them with --dock-untrusted)
    dock_at = []
    for c in cases:
        lab = work[c['key']]
        if (cal[lab]['trusted'] or a.dock_untrusted) and lab not in dock_at:
            dock_at.append(lab)
    skipped = [work[c['key']] for c in cases if work[c['key']] not in dock_at]
    if skipped:
        print(f"\n[plan] receptors not trusted, generated molecules will NOT be docked there: {skipped}  (--dock-untrusted to override)")
    # 3. docking jobs: a case's generated molecules at its own receptor (at every docked receptor with --cross), and its baseline set and seeds at the own receptor
    jobs = []
    for c in cases:
        own = work[c['key']]
        for lab in dock_at:
            if lab == own or a.cross:
                jobs += [(g['smi'], lab) for g in gens[c['key']]]
        if own in dock_at and a.baseline:
            jobs += [(g['smi'], own) for g in gens[c['key'] + '#baseline']]
        if own in dock_at and not a.no_seeds:
            jobs += [(g['seed'], own) for g in gens[c['key']] if g['seed']]
    dock.run(jobs, 'generated molecules, seeds' + (', cross-target' if a.cross else ''))
    dock.failure_summary()
    return cal, redock, gens, work, dock_at


# For a list of molecules ({'smi', 'ha', ...}) returns arrays of the raw score, the size-corrected residual, the percentile among the receptor's decoys and the ligand efficiency
# (score per heavy atom), for the molecules that were docked; None if none were.
def gen_stats(dock, lab, mols, cal):
    rows = [(m, dock.lookup(m['smi'], lab)) for m in mols]
    rows = [(m, s) for m, s in rows if s is not None]
    if not rows:
        return None
    ha = np.array([m['ha'] for m, _ in rows], float)
    sc = np.array([s for _, s in rows], float)
    r = resid(ha, sc, cal['fit'])
    return {'smi': [m['smi'] for m, _ in rows], 'ha': ha, 'score': sc, 'resid': r, 'pct': percentile_lower(r, cal['resid_decoys']), 'le': -sc / ha,
            'mols': [m for m, _ in rows]}


# Prints a per-seed breakdown of the generated set: how many analogs each seed gave, their median percentile, the seed's own percentile and the median similarity to the seed.
def seed_table(g, sg):
    by = {}
    for m, p_ in zip(g['mols'], g['pct']):
        by.setdefault(m['seed'], {'name': m.get('seed_name') or '?', 'p': [], 'sim': []})
        by[m['seed']]['p'].append(p_)
        by[m['seed']]['sim'].append(m['sim'])
    seed_pct = dict(zip(sg['smi'], sg['pct'])) if sg is not None else {}
    print(f"      by seed:   {'seed':<28}{'analogs':>8}{'median pct':>11}{'seed pct':>9}{'median sim':>11}")
    for s, v in sorted(by.items(), key=lambda kv: -len(kv[1]['p']))[:14]:
        sp = seed_pct.get(s)
        print(f"                 {v['name'][:27]:<28}{len(v['p']):>8}{np.median(v['p']):>11.2f}{('%.2f' % sp) if sp is not None else '   n/a':>9}{np.nanmedian(v['sim']):>11.2f}")


# Prints the report in four parts and writes the outputs: 1 receptor calibration, 2 each generated set at its own receptor (size-corrected AUC with a bootstrap interval, the seeds, the paired
# analog-versus-seed comparison, the random-seed baseline and conditioned-versus-baseline), 3 the cross-target negative control (--cross), 4 the verdict and the roadmap outcome per case.
# A --diagnose run stops after part 1.
def report(a, cases, receptors, cal, redock, gens, work, dock_at, dock, lib):
    import csv as _csv
    nrng = np.random.default_rng(a.seed)
    W = 118
    print("\n" + "=" * W)
    print("1. RECEPTOR CALIBRATION   (can Vina tell known binders from property-matched decoys in this pocket?  AUC 0.5 = no)")
    print("=" * W)
    print(f"   {'receptor':<10}{'status':<19}{'AUC':>12}{'size-corr. AUC':>17}{'EF20':>6}{'n act/dec':>11}{'redock ref':>11}   trusted")
    for c in cases:
        for lab in c['variants']:
            k = cal[lab]
            star = ' *' if len(c['variants']) > 1 and work[c['key']] == lab else ''
            if 'auc' not in k:
                print(f"   {lab + star:<10}{k['status']:<19}  {k.get('reason', '')}")
                continue
            ref = f"{k['reference']:.1f}" if k.get('reference') is not None else 'n/a'
            print(f"   {lab + star:<10}{k['status']:<19}{k['auc']:>7.2f}+/-{k['se']:.2f}{k['auc_sc']:>11.2f}+/-{k['se_sc']:.2f}{k['ef20']:>6.2f}"
                  f"{str(k['n_act']) + '/' + str(k['n_dec']):>11}{ref:>11}   {'YES' if k['trusted'] else 'no'}")
            print(f"        decision set: {k['decision']} (n {k['n_decision']});  size-corrected AUC by stratum: "
                  + ' | '.join(f"{n_} {v['auc_sc']:.2f} (n {v['n']})" for n_, v in k['strata'].items() if v['n'])
                  + f";  without co-crystal ligand {k['auc_nocc']:.2f}" + ("" if k.get('has_cc') else " (co-crystal ligand not among the actives)"))
            if k.get('potency_filter_nm') is not None or k.get('potency_n'):
                r_ = k['potency_rho']
                print(("        potency: actives kept at <= %g nM" % k['potency_filter_nm'] if k.get('potency_filter_nm') else "        potency: no active filter")
                      + ";  Spearman(pKi/pIC50, size-corrected score) = " + ('n/a' if r_ != r_ else '%.2f' % r_) + f" (n {k['potency_n']})")
            if k['decision'].startswith('research'):
                print(f"        all {k['n_act']} actives together: AUC {k['auc_all']:.2f}, size-corrected {k['auc_sc_all']:.2f}  (descriptive only; the decision uses the research stratum)")
            if k.get('easy_auc_sc') is not None:
                print(f"        easy decoys (size-matched only, n {k['easy_n']}): AUC {k['easy_auc_raw']:.2f}, size-corrected {k['easy_auc_sc']:.2f}")
            r = redock.get(lab)
            if r and r.get('status') == 'ok':
                print(f"        redock of the co-crystal ligand: top pose {r['top_rmsd']:.2f} A from the crystal pose (closest of {r['n_poses']}: {r['best_rmsd']:.2f} A)  -> {r['verdict']}")
            elif r and r.get('reason'):
                print(f"        redock: {r['status']} - {r['reason']}")
    if a.box_pad:
        print(f"   (all boxes enlarged by {a.box_pad:g} A on each side)")
    if a.ensemble:
        print("   (ensemble: score = better of the crystal site and the equivalent second site, for actives, decoys and generated molecules alike)")
    if a.diagnose:
        write_outputs(a, cases, cal, redock, work, {}, [])
        return

    print("\n" + "=" * W)
    print("2. EACH GENERATED SET AT ITS OWN RECEPTOR   (size-corrected: residual of score ~ heavy atoms, fitted on that receptor's decoys)")
    print("   AUC = chance a generated molecule beats a decoy (0.5 = no better).  'actives' row = the same number for the known binders (the ceiling).")
    print("=" * W)
    verdicts, all_rows, extras = {}, [], {}
    for c in cases:
        lab = work[c['key']]
        k = cal[lab]
        print(f"\n  {c['disease']} / {c['target']} -> {lab}")
        if not k['trusted'] and not (a.dock_untrusted and 'fit' in k):
            print(f"      receptor NOT TRUSTED ({k['status']}): nothing is interpreted for this case.")
            verdicts[c['key']] = 'not interpretable'
            continue
        g = gen_stats(dock, lab, gens[c['key']], k)
        if g is None or len(g['score']) < 3:
            print("      too few docked generated molecules (run without --report-only, or check the failures list)")
            verdicts[c['key']] = 'not enough data'
            continue
        if not k['trusted']:
            print("      WARNING: receptor not trusted (--dock-untrusted): read nothing into the numbers below.")
        auc, se = auc_lower(g['resid'], k['resid_decoys'])
        lo, hi = boot_ci(_auc_only, g['resid'], k['resid_decoys'], nrng)
        auc_raw = auc_lower(g['score'], [x['score'] for x in k['decoys'] if x['score'] is not None])[0]
        le_dec = float(np.mean([-x['score'] / x['ha'] for x in k['decoys'] if x['score'] is not None]))
        le_act = k['le_act']
        print(f"      n docked {len(g['score'])};  mean heavy atoms {g['ha'].mean():.1f} (decoys {k['ha_d']:.1f}, actives {k['ha_a']:.1f})")
        print(f"      {'':<11}{'mean score':>11}{'size-corr. AUC':>16}{'raw AUC':>9}{'mean LE':>9}")
        print(f"      {'generated':<11}{g['score'].mean():>11.2f}{auc:>9.2f} [{lo:.2f}-{hi:.2f}]{auc_raw:>6.2f}{g['le'].mean():>9.3f}")
        print(f"      {'actives':<11}{k['act_med']:>11.2f}{k['auc_sc']:>9.2f}{'':<14}{k['auc']:>5.2f}{le_act:>9.3f}   <- ceiling (the known binders)")
        print(f"      {'decoys':<11}{k['dec_med']:>11.2f}{0.5:>9.2f}{'':<14}{0.5:>5.2f}{le_dec:>9.3f}")
        print(f"      median percentile of a generated molecule among the decoys: {np.median(g['pct']):.2f}  (0.5 = typical decoy)")
        phf = np.array([m.get('ph') for m in g['mols']], dtype=object)
        if any(x is not None for x in phf):
            wp = np.array([x is True for x in phf])
            parts = []
            for nm, mk in (('with the pharmacophore', wp), ('without it', ~wp)):
                if mk.sum() >= 5:
                    parts.append(f"{nm}: n {int(mk.sum())}, size-corrected AUC {auc_lower(g['resid'][mk], k['resid_decoys'])[0]:.2f}")
                else:
                    parts.append(f"{nm}: n {int(mk.sum())}")
            print("      descriptive split (the pharmacophore is the one thing the calibration rewards):  " + ';  '.join(parts))
        dha = [x['ha'] for x in k['decoys'] if x['score'] is not None]
        inr = (g['ha'] >= min(dha) - 2) & (g['ha'] <= max(dha) + 2)
        if (~inr).any():
            print(f"      SIZE WARNING: the decoys span {min(dha)}-{max(dha)} heavy atoms; {int((~inr).sum())} of {len(inr)} generated molecules lie outside that "
                  f"(+-2), where the size correction is an extrapolation.")
            if inr.sum() >= 5:
                a_in, _ = auc_lower(g['resid'][inr], k['resid_decoys'])
                l_in, h_in = boot_ci(_auc_only, g['resid'][inr], k['resid_decoys'], nrng)
                print(f"      in-range molecules only (n={int(inr.sum())}): size-corrected AUC {a_in:.2f} [{l_in:.2f}-{h_in:.2f}]")
        # verdict thresholds (my judgement): PASS = size-corrected AUC >= 0.65 with the 95% interval above 0.5; PARTIAL = AUC >= 0.55; otherwise FAIL
        v = 'PASS' if (auc >= 0.65 and lo > 0.5) else 'PARTIAL' if auc >= 0.55 else 'FAIL'
        verdicts[c['key']] = f"{v} (size-corrected AUC {auc:.2f}, CI {lo:.2f}-{hi:.2f})" + ('' if k['trusted'] else ' [receptor not trusted]')
        # paired comparison: each analog against its own seed, docked at the same receptor
        seed_mols = {}
        for m in gens[c['key']]:
            if m['seed'] and m['seed'] not in seed_mols:
                seed_mols[m['seed']] = {'smi': m['seed'], 'ha': be_ha(m['seed'], lib)}
        sg = gen_stats(dock, lab, list(seed_mols.values()), k) if seed_mols else None
        if sg is not None and len(sg['score']) >= 3:
            sa_auc = auc_lower(sg['resid'], k['resid_decoys'])[0]
            print(f"      seeds ({len(sg['score'])} distinct): mean score {sg['score'].mean():.2f}, size-corrected AUC vs decoys {sa_auc:.2f}, "
                  f"median percentile {np.median(sg['pct']):.2f}   (do the seeds themselves look like binders here?)")
            sres = dict(zip(sg['smi'], sg['resid']))
            d_res, sims = [], []
            for m, r_ in zip(g['mols'], g['resid']):
                if m['seed'] in sres:
                    d_res.append(r_ - sres[m['seed']])
                    sims.append(m['sim'])
            if len(d_res) >= 3:
                d_res = np.array(d_res)
                rho = spearman(sims, np.abs(d_res))
                print(f"      PAIRED analog - seed (n={len(d_res)}): mean size-corrected difference {d_res.mean():+.2f} kcal/mol "
                      f"[{np.percentile(d_res, 25):+.2f} to {np.percentile(d_res, 75):+.2f} IQR],  analogs better than their seed: {np.mean(d_res < 0):.0%};  "
                      f"Spearman(similarity to seed, |difference|) = {rho:.2f}  (negative = closer analogs score more like their seed)")
        seed_table(g, sg)
        # random-seed baseline: the same model with seeds from the whole library.  The size-corrected AUC of conditioned over baseline tests whether choosing the target's ligands as seeds helps
        bl = gens.get(c['key'] + '#baseline')
        if bl is not None:
            b = gen_stats(dock, lab, bl, k)
            if b is None or len(b['score']) < 3:
                print("      RANDOM-SEED BASELINE: too few docked molecules")
            else:
                ab, _ = auc_lower(b['resid'], k['resid_decoys'])
                lob, hib = boot_ci(_auc_only, b['resid'], k['resid_decoys'], nrng)
                acb, _ = auc_lower(g['resid'], b['resid'])
                loc, hic = boot_ci(_auc_only, g['resid'], b['resid'], nrng)
                print(f"      RANDOM-SEED BASELINE (same model, seeds from the whole library, none of this target's): n docked {len(b['score'])}, mean heavy atoms {b['ha'].mean():.1f}, "
                      f"mean score {b['score'].mean():.2f}, size-corrected AUC vs decoys {ab:.2f} [{lob:.2f}-{hib:.2f}]")
                print(f"      CONDITIONED vs BASELINE: chance a target-conditioned molecule beats a random-seed one (size-corrected): {acb:.2f} [{loc:.2f}-{hic:.2f}]  -> "
                      + ('conditioning gives better-docking molecules here (the seeds are known binders: see the paired analog-vs-seed line for what the model adds)' if loc > 0.5
                         else 'no detectable docking advantage of conditioning at this receptor' if hic >= 0.5 else 'the random-seed set docks BETTER than the conditioned one'))
                extras[c['key']] = {'baseline_auc_vs_decoys': ab, 'baseline_ci': [lob, hib], 'conditioned_vs_baseline_auc': acb, 'conditioned_vs_baseline_ci': [loc, hic],
                                    'n_baseline': int(len(b['score']))}
                for m, s_, r_, p_ in zip(b['mols'], b['score'], b['resid'], b['pct']):
                    all_rows.append({'case': c['key'], 'receptor': lab, 'role': 'baseline_random_seed', 'name': '', 'smiles': m['smi'], 'heavy_atoms': m['ha'], 'score': s_,
                                     'size_corrected': r_, 'percentile': p_, 'seed': m['seed'], 'seed_name': m.get('seed_name', ''), 'similarity_to_seed': m['sim']})
        for m, s_, r_, p_ in zip(g['mols'], g['score'], g['resid'], g['pct']):
            all_rows.append({'case': c['key'], 'receptor': lab, 'role': 'generated_own', 'name': '', 'smiles': m['smi'], 'heavy_atoms': m['ha'], 'score': s_,
                             'size_corrected': r_, 'percentile': p_, 'seed': m['seed'], 'seed_name': m.get('seed_name', ''), 'similarity_to_seed': m['sim']})
    # 3. cross-target comparison (works with a single trusted receptor)
    if a.cross:
        print("\n" + "=" * W)
        print("3. SPECIFICITY / NEGATIVE CONTROL   (every generated set docked at each trusted receptor)")
        print("=" * W)
        tl = [l for l in dock_at if cal[l]['trusted']]
        if not tl:
            print("   no trusted receptor: nothing to compare.")
        else:
            G = {}
            for c in cases:
                for l in tl:
                    G[(c['key'], l)] = gen_stats(dock, l, gens[c['key']], cal[l])
            if len(tl) >= 2:
                print("   cell = median percentile among that receptor's decoys, size-corrected (0.5 = typical decoy, higher = better);  * = the set's own receptor")
                print(f"      {'generated for':<26}" + ''.join(f"{l:>11}" for l in tl) + "     own receptor ranks")
                for c in cases:
                    cells, meds = '', []
                    for l in tl:
                        g = G[(c['key'], l)]
                        m = float(np.median(g['pct'])) if g is not None and len(g['pct']) else float('nan')
                        meds.append(m)
                        cells += f"{('%.2f' % m) + ('*' if l == work[c['key']] else ' '):>11}"
                    if work[c['key']] in tl and not all(np.isnan(meds)):
                        order = list(np.argsort(-np.nan_to_num(meds, nan=-1)))
                        rk = f"{order.index(tl.index(work[c['key']])) + 1} of {len(tl)}"
                    else:
                        rk = 'n/a'
                    print(f"      {c['disease'][:18] + ' ' + c['target']:<26}{cells}     {rk}")
                print()
            print("   specificity at each trusted receptor: AUC that a molecule generated for THIS receptor's target beats one generated for another target\n"
                  "   (size-corrected, same receptor).  The other sets are docked here although this is not their target's pocket: they are the negative control.")
            for l in tl:
                own_cases = [c for c in cases if work[c['key']] == l]
                if not own_cases:
                    continue
                c_own = own_cases[0]
                mine = G[(c_own['key'], l)]
                others = [G[(c['key'], l)] for c in cases if c['key'] != c_own['key'] and G[(c['key'], l)] is not None]
                if mine is None or not others:
                    print(f"      {l:<9} not enough docked molecules for a comparison")
                    continue
                oth = np.concatenate([o['resid'] for o in others])
                auc_, _ = auc_lower(mine['resid'], oth)
                lo, hi = boot_ci(_auc_only, mine['resid'], oth, nrng)
                print(f"      {l:<9} AUC {auc_:.2f} [{lo:.2f}-{hi:.2f}]  (n {len(mine['resid'])} own vs {len(oth)} other)   "
                      + ('specific: the own-target set scores better here than the other sets' if lo > 0.5 else 'not distinguishable from the other sets'))
            print("   caution: the sets differ chemically as well as by target (e.g. guanidines vs basic amines), and the size correction does not remove charge or polarity effects.")
            for (ck, l), g in G.items():
                if g is not None and ck != next((c['key'] for c in cases if work[c['key']] == l), None):
                    for m, s_, r_, p_ in zip(g['mols'], g['score'], g['resid'], g['pct']):
                        all_rows.append({'case': ck, 'receptor': l, 'role': 'generated_cross', 'name': '', 'smiles': m['smi'], 'heavy_atoms': m['ha'], 'score': s_,
                                         'size_corrected': r_, 'percentile': p_, 'seed': m['seed'], 'seed_name': m.get('seed_name', ''), 'similarity_to_seed': m['sim']})
    # 4. verdict
    print("\n" + "=" * W)
    print("4. VERDICT  (thresholds are my judgement: PASS = size-corrected AUC >= 0.65 with the 95% interval above 0.5; PARTIAL >= 0.55)")
    print("=" * W)
    for c in cases:
        print(f"   {c['disease']:<28}{c['target']} / {work[c['key']]:<9}{verdicts.get(c['key'], 'n/a')}")
    print("\n   Roadmap outcome per case  (RL is only needed if generated molecules cannot be told apart from random ones at a TRUSTED receptor):")
    for c in cases:
        v = verdicts.get(c['key'], 'n/a')
        o = ('A: receptor not trusted -> fix Phase 1 first (no RL decision possible)' if v == 'not interpretable' else
             'not enough docked molecules: rerun (resumable) before deciding' if v in ('not enough data', 'n/a') else
             'C: generated molecules dock like binders -> RL not needed' if v.startswith('PASS') else
             'B: generated molecules do not beat the decoys -> RL candidate' if v.startswith('FAIL') else
             'borderline: add molecules / rerun before deciding on RL')
        print(f"   {c['disease']:<28}{c['target']:<8}{o}")
    print("   Caveats: Vina is a crude ranking tool; decoys are 'unrelated by label', not guaranteed non-binders; the generated molecules come from a model\n"
          "   that saw these targets in training; n per set is small, so read the intervals, not the point estimates.  Docking says nothing about the model's\n"
          "   conditioning itself, only about whether its output looks like binders at a receptor that passed calibration.")
    write_outputs(a, cases, cal, redock, work, verdicts, all_rows, extras)


# Writes the per-molecule CSV (generated, baseline, cross-target and calibration rows) to a.out and the JSON summary (settings, verdict per case, calibration scalars, redock results) to a.summary.
def write_outputs(a, cases, cal, redock, work, verdicts, all_rows, extras=None):
    import csv as _csv
    for c in cases:
        for lab in c['variants']:
            k = cal[lab]
            if 'actives' in k:
                for role, key in (('calibration_active', 'actives'), ('calibration_decoy', 'decoys')):
                    for x in k[key]:
                        if x.get('score') is not None:
                            all_rows.append({'case': c['key'], 'receptor': lab, 'role': role, 'name': x.get('name', ''), 'smiles': x['smi'], 'heavy_atoms': x['ha'],
                                             'score': x['score'], 'seed_name': x.get('stratum', '')})
    cols = ['case', 'receptor', 'role', 'name', 'smiles', 'heavy_atoms', 'score', 'size_corrected', 'percentile', 'seed', 'seed_name', 'similarity_to_seed']
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    # Calibration results without the arrays, lists and dicts, so that they fit in the JSON summary (the strata are kept).
    def scalars(k):
        d = {k2: v for k2, v in k.items() if not isinstance(v, (np.ndarray, list, dict)) and k2 not in ('fit',)}
        d['strata'] = k.get('strata')
        return d
    summ = {'box_pad_A': a.box_pad, 'cross': bool(a.cross), 'ensemble': bool(a.ensemble), 'diagnose': bool(a.diagnose),
            'cases': {c['key']: {'working_receptor': work[c['key']], 'verdict': verdicts.get(c['key']), 'conditioned_vs_baseline': (extras or {}).get(c['key']),
                                 'receptors': {lab: {'calibration': scalars(cal[lab]), 'redock': redock.get(lab)} for lab in c['variants']}} for c in cases}}
    with open(a.summary, 'w', encoding='utf-8') as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(f"\n   per-molecule results: {a.out}   summary: {a.summary}   cache: {a.cache}")


# the backend in use (RealBackend or FakeBackend), set by main(); be_ha reads it
BE = None


# Heavy-atom count of a SMILES through the active backend (lib is not used).
def be_ha(smi, lib=None):
    return BE.describe(smi)['ha']


# Builds the case list and the receptor table from CASES and targets.DISEASE_TARGET_CONFIGS.  For each wanted case it finds the receptor file (with `find`), pads the box by pad A on each
# side and adds the case's EXTRA_RECEPTORS as further structures (case['variants'] lists the receptor labels, the main one first).  Exits with a message if an entry or a file is missing.
# Returns (cases, receptors).
def build_cases(configs, wanted, find=find_file, pad=0.0):
    by_name = {c['name']: c for c in configs}
    cases, receptors = [], {}
    for case0 in CASES:
        if wanted and case0['key'] not in wanted:
            continue
        case = dict(case0)
        cfg = by_name.get(case['v13_config'])
        if cfg is None:
            raise SystemExit(f"v13 DISEASE_TARGET_CONFIGS has no entry named '{case['v13_config']}': {list(by_name)}")
        r = next((x for x in cfg['candidate_receptors'] if x['label'] == case['receptor']), None)
        if r is None:
            raise SystemExit(f"v13 entry '{case['v13_config']}' has no receptor '{case['receptor']}'")
        path = find(r['receptor_pdbqt_path'])
        if not path:
            raise SystemExit(f"receptor file {r['receptor_pdbqt_path']} ({r['label']}) not found: put it beside the scripts or in a parent folder")
        size = tuple(round(float(s) + 2 * pad, 3) for s in r['box_size'])
        receptors[r['label']] = {'label': r['label'], 'pdbqt': path, 'center': tuple(r['box_center']), 'size': size, 'reference': r.get('reference_smiles'),
                                 'calibration_active_smiles': r.get('calibration_active_smiles') or [], 'dataset_target_codes': r.get('dataset_target_codes'),
                                 'crystal': find(case['crystal_ligand']) if case.get('crystal_ligand') else None, 'crystal_name': case.get('crystal_ligand')}
        case['variants'] = [r['label']]
        for ex in EXTRA_RECEPTORS:
            if ex['case'] != case['key']:
                continue
            p = find(ex['pdbqt'])
            if not p:
                raise SystemExit(f"extra receptor file {ex['pdbqt']} ({ex['label']}) not found")
            receptors[ex['label']] = {'label': ex['label'], 'pdbqt': p, 'center': tuple(ex['center']), 'size': tuple(round(float(s) + 2 * pad, 3) for s in ex['size']),
                                      'reference': ex.get('reference') or r.get('reference_smiles'), 'calibration_active_smiles': r.get('calibration_active_smiles') or [],
                                      'dataset_target_codes': r.get('dataset_target_codes'),
                                      'crystal': find(ex['crystal_ligand']) if ex.get('crystal_ligand') else None, 'crystal_name': ex.get('crystal_ligand')}
            case['variants'].append(ex['label'])
        cases.append(case)
    if not cases:
        raise SystemExit("no cases selected")
    return cases, receptors


# Command-line entry point: reads the options, sets up the backend (real, or fake with --fake), the receptors (with ensemble, extra structures, interaction rule, potency tables), the
# library and the docking cache, then runs the pipeline and prints the report.  Everything is cached, so an interrupted run can be repeated.
def main():
    global BE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v13', default=None, help='ignored (kept so old command lines still work): the docking code is now imported from docking.py')
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--vina', default=None, help='path to vina.exe (default: docking.VINA_EXECUTABLE_PATH, resolved by config.py)')
    ap.add_argument('--cases', default='', help='comma list of case keys to run (default all): ' + ', '.join(c['key'] for c in CASES))
    ap.add_argument('--n', type=int, default=30, help='generated molecules docked per set')
    ap.add_argument('--n-actives', type=int, default=N_ACTIVES)
    ap.add_argument('--exhaustiveness', type=int, default=8)
    ap.add_argument('--noise-check', type=int, default=3, help='re-dock this many actives per receptor at exhaustiveness 32 / another seed (0 = off); each takes ~1-1.5 min')
    ap.add_argument('--cross', action='store_true', help="ALSO dock each generated set at the other targets' trusted receptors (negative control for specificity; works with one trusted receptor). "
                    "Default off: each set is docked at its own target's receptor only. Adds only the missing dockings if you run it later.")
    ap.add_argument('--active-stages', choices=['auto', 'clinical', 'all'], default='auto',
                    help="calibration actives for ALL cases: 'clinical' = clinical-stage ligands only (v13's rule); 'all' = also investigative/patented. Default auto: each case's own setting")
    ap.add_argument('--n-research', type=int, default=N_RESEARCH, help='research-stratum actives sampled in all-stage cases (default %d)' % N_RESEARCH)
    ap.add_argument('--redock', '--redock-check', dest='redock', action='store_true', help='redock each co-crystal ligand (9 poses) and report the pose RMSD to the crystal pose')
    ap.add_argument('--diagnose', action='store_true', help='calibration + redock + easy decoys for the selected cases only: no generated sets are read or docked')
    ap.add_argument('--calibration-only', action='store_true', help='calibration ONLY (no generated sets, no redock, no easy decoys): the shortest run that gives a trust decision for the selected cases')
    ap.add_argument('--ensemble', action='store_true', help="also dock the equivalent second site of the same crystal structure (CASES alt_sites) and keep the better score")
    ap.add_argument('--potency', action='store_true', help='use potency_<case>.csv (fetch_potency.py): keep actives <= --max-nm, drop known-binder decoys <= --decoy-nm, print a potency-vs-score check')
    ap.add_argument('--extra-receptors', default='', help="JSON list of further receptor structures (made by prep_receptors.py); each is calibrated on its own, see EXTRA_RECEPTORS in the script")
    ap.add_argument('--ensemble-structures', action='store_true', help="with --extra-receptors: the case's main receptor becomes the BEST score over its own structure and every extra structure flagged in_ensemble (decoys get the same treatment)")
    ap.add_argument('--potency-cases', default='', help='with --potency: apply it only to these case keys (default: all cases).  Do NOT include malaria_t89515: most PfDHFR research ligands have no ChEMBL measurement')
    ap.add_argument('--keep-no-potency', action='store_true', help='with --potency: actives with no ChEMBL measurement are KEPT (only measured weak ones are dropped); sensitivity check for the default, which drops them')
    ap.add_argument('--potency-pattern', default='potency_{key}.csv')
    ap.add_argument('--max-nm', type=float, default=100.0, help='with --potency: actives must be at least this potent (nM); 0 = no active filter')
    ap.add_argument('--decoy-nm', type=float, default=1000.0, help='with --potency: decoys with a ChEMBL potency <= this (nM) are dropped as known binders; 0 = off')
    ap.add_argument('--interaction', default='', choices=[''] + sorted(INTERACTION_RULES),
                    help="score with a pre-registered interaction rule (see the docstring): dpp4_amine_glu = a basic N within 4 A of Glu205/206 (diabetes case only). Uses its own cache keys; "
                         "prints the plain-Vina AUC from the same dockings next to the rule's")
    ap.add_argument('--baseline', action='store_true', help='also load and dock the random-seed baseline set of each case (gen_<key>_baseline.csv, made with generate_for_request.py --random-seeds) at its own receptor')
    ap.add_argument('--baseline-pattern', default='gen_{key}_baseline.csv')
    ap.add_argument('--no-seeds', action='store_true', help='do not dock the seed molecules')
    ap.add_argument('--skip-selftest', action='store_true', help='skip the start-up check of the pharmacophore SMARTS')
    ap.add_argument('--easy-decoys', action='store_true', help='extra calibration against decoys matched on size only (diagnostic for receptors that fail)')
    ap.add_argument('--easy-per-active', type=int, default=2, help='easy decoys per active (default 2)')
    ap.add_argument('--box-pad', type=float, default=0.0, help='enlarge every docking box by this many A on each side (use separate --cache/--out/--summary names)')
    ap.add_argument('--dock-untrusted', action='store_true', help='also dock generated sets at receptors that failed calibration (numbers are then NOT interpretable)')
    ap.add_argument('--allow-seed-mismatch', action='store_true')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='no docking: report from the cache')
    ap.add_argument('--fake', action='store_true', help='plumbing test: made-up scores, no RDKit/Meeko/Vina needed (never interpret the numbers)')
    ap.add_argument('--gen-pattern', default='gen_{key}.csv')
    ap.add_argument('--cache', default='dock_cache_v2.jsonl')
    ap.add_argument('--out', default='dock_results_v3.csv')
    ap.add_argument('--summary', default='dock_summary_v3.json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    # --calibration-only and --diagnose share the diagnose path (no generated sets are read); --diagnose also switches the easy-decoy check on
    if a.calibration_only:
        a.diagnose = True
    elif a.diagnose:
        a.easy_decoys = True
    # fake runs use their own cache and output names, so made-up numbers never mix with real ones
    if a.fake:
        a.cache = a.cache.replace('.jsonl', '_FAKE.jsonl')
        a.out, a.summary = a.out.replace('.csv', '_FAKE.csv'), a.summary.replace('.json', '_FAKE.json')

    csv_path = find_file(a.csv)
    if not csv_path:
        raise SystemExit(f"cannot find {a.csv}")
    from targets import DISEASE_TARGET_CONFIGS as configs
    # extra receptor structures from one or several JSON files (made by prep_receptors.py)
    if a.extra_receptors:
        EXTRA_RECEPTORS[:] = []
        for nm_ in [x.strip() for x in a.extra_receptors.split(',') if x.strip()]:
            ef = find_file(nm_)
            if not ef:
                raise SystemExit(f"{nm_} not found")
            EXTRA_RECEPTORS.extend(json.load(open(ef, encoding='utf-8')))
        for ex in EXTRA_RECEPTORS:
            for k in ('label', 'case', 'pdbqt', 'center', 'size'):
                if k not in ex:
                    raise SystemExit(f"{a.extra_receptors}: entry {ex.get('label', '?')} lacks '{k}'")
        print(f"[extra receptors] {len(EXTRA_RECEPTORS)} from {a.extra_receptors}: " + ', '.join(f"{e['label']} ({e['case']})" for e in EXTRA_RECEPTORS))
    cases, receptors = build_cases(configs, [x for x in a.cases.split(',') if x], find=(lambda n: find_file(n) or (n if a.fake else None)), pad=a.box_pad)
    # interaction rule: switch it on for the receptors of the cases it applies to and check that the rule's receptor atoms lie in the docking box
    if a.interaction:
        irule = INTERACTION_RULES[a.interaction]
        n_on = 0
        for c in cases:
            if c['key'] in irule['cases']:
                for lab in c['variants']:
                    receptors[lab]['inter'] = a.interaction
                    n_on += 1
                    if not a.fake:
                        ng = len(rule_atoms(receptors[lab], irule))
                        print(f"[interaction] {a.interaction} on {lab}: {ng} receptor atoms found in the box (expect {len(irule['residues']) * len(irule['atoms'])}: "
                              f"{'/'.join(irule['atoms'])} of {', '.join(f'{rn}{ri}' for rn, ri in irule['residues'])})")
                        if ng < 2:
                            raise SystemExit(f"{lab}: fewer than 2 of the rule's receptor atoms {irule['residues']} lie in the docking box: wrong residue numbering or box? (nothing was docked)")
        if not n_on:
            raise SystemExit(f"--interaction {a.interaction} applies to case(s) {irule['cases']}: add one with --cases (nothing was docked)")
    if a.fake:
        BE = FakeBackend()
        print("[FAKE MODE] made-up scores, no docking: only the plumbing and report layout are being tested")
    else:
        import docking as ns
        vina = a.vina or ns.VINA_EXECUTABLE_PATH
        vf = find_file(vina) if not os.path.isabs(vina) else (vina if os.path.exists(vina) else find_file(os.path.basename(vina)))
        if not vf and not a.report_only:
            raise SystemExit(f"vina.exe not found at {vina}: pass --vina <path>")
        ns.VINA_EXECUTABLE_PATH = vf or vina
        BE = RealBackend(ns)
    for c in cases:
        BE.check_smarts(c.get('active_smarts'))
        BE.check_smarts(c.get('exclude_decoy_smarts'))
    # start-up self-tests of the pharmacophore rule (and of the basic-nitrogen rule with --interaction): a failure stops the run before anything is docked
    if not a.fake and not a.skip_selftest:
        bad = BE.self_test()
        if bad:
            raise SystemExit("self-test of the pharmacophore rule FAILED (nothing was docked):\n   " + "\n   ".join(bad)
                             + "\n   Send this output back; use --skip-selftest only to override deliberately.")
        print(f"[self-test] pharmacophore rule OK on {len(SELFTEST)} known compounds")
        if a.interaction:
            bad = BE.inter_selftest()
            if bad:
                raise SystemExit("self-test of the basic-nitrogen rule FAILED (nothing was docked):\n   " + "\n   ".join(bad) + "\n   Send this output back.")
            print(f"[self-test] basic-nitrogen rule OK on {len(INTER_SELFTEST)} known compounds")
    # potency tables (potency_<case>.csv from fetch_potency.py) for the chosen cases
    if a.potency:
        pot_cases = {x.strip() for x in a.potency_cases.split(',') if x.strip()}
        for c in cases:
            if pot_cases and c['key'] not in pot_cases:
                print(f"[potency] {c['key']}: not applied (not in --potency-cases)")
                continue
            pf = find_file(a.potency_pattern.format(key=c['key']))
            if not pf:
                raise SystemExit(f"{a.potency_pattern.format(key=c['key'])} not found: run fetch_potency.py first (or drop --potency)")
            c['_potency'] = load_potency(BE, pf)
            print(f"[potency] {c['key']}: {len(c['_potency'])} compounds with a ChEMBL potency ({os.path.basename(pf)})")
    # ensemble members per receptor label: --ensemble adds the equivalent second site (chain B), --ensemble-structures adds the extra structures flagged in_ensemble; a molecule's score is the best over the members
    members = {}
    if a.ensemble:
        for c in cases:
            alts = c.get('alt_sites') or []
            if not alts:
                print(f"[ensemble] {c['key']}: no second site defined (single chain): docked as before")
            for alt in alts:
                lab = c['receptor']
                nm = f"{lab}@{alt['name']}"
                receptors[nm] = dict(receptors[lab], label=nm, center=tuple(alt['center']), size=tuple(round(float(x) + 2 * a.box_pad, 3) for x in alt['size']))
                members.setdefault(lab, [lab]).append(nm)
    if a.ensemble_structures:
        for c in cases:
            lab = c['receptor']
            for ex in EXTRA_RECEPTORS:
                if ex['case'] == c['key'] and ex.get('in_ensemble', True):
                    members.setdefault(lab, [lab]).append(ex['label'])
            if len(members.get(lab, [lab])) > 1:
                print(f"[ensemble-structures] {c['key']}: {lab} = best score over {members[lab]}")
            else:
                print(f"[ensemble-structures] {c['key']}: no extra structures flagged in_ensemble: docked as before")
    # receptor signatures go into the cache keys
    for r in receptors.values():
        r['sig'] = receptor_signature(r)
    lib = read_library(csv_path)
    print(f"[library] {len(lib)} rows with a usable SMILES, {int(lib['clinical'].sum())} clinical-stage")
    if a.fake:
        for c in cases:
            for lab in c['variants']:
                r = receptors[lab]
                BE.binders[lab] = set(lib.loc[lib['targets'].apply(lambda t: c['target'] in t), 'smi']) | set(r['calibration_active_smiles'])
    cache = DockCache(a.cache)
    print(f"[cache] {len(cache.d)} earlier dockings in {a.cache}")
    dock = Docker(BE, cache, receptors, a.exhaustiveness, a.retry_failed, a.report_only)
    dock.members = members
    cal, redock, gens, work, dock_at = run(a, BE, dock, lib, cases, receptors)
    report(a, cases, receptors, cal, redock, gens, work, dock_at, dock, lib)


if __name__ == '__main__':
    main()
