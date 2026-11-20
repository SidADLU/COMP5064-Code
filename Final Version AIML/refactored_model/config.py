#!/usr/bin/env python3
"""
config.py -- the single place for file names, default locations and shared switches.

Every script imports its default paths from here (generate_for_request, train_exemplar_generator, check_exemplar_conditioning, dock_generated, docking, base_train).
Nothing in the scripts hard-codes a path.  Change a name or a location HERE, once.

Lookup rule (the same for every file): the folder that holds this file wins, then its parent folders, then the legacy absolute location below, then the bare
file name (so an error message shows what was wanted).  Command-line options still override every default.

  python config.py        prints where each default resolves to, and flags anything that cannot be found
"""
import os

# Folder that holds this file; it is searched first for every default file.
HERE = os.path.dirname(os.path.abspath(__file__))
# The only absolute location left in the project: the folder the files lived in before the project folder existed.  Used as a last-resort fallback.
LEGACY_ROOT = r'C:\Users\sidha\Desktop\AI ML code'

# File names of the shared inputs (the bare names that find_file looks for).  CleanedData.csv is the ligand library (columns Disease Entry, Targets for Drug, Canonical Smiles, ...).  The pretrained checkpoint is the base SMILES language model;
# the two exemplar checkpoints are the generator trained on all targets but 16 (v2) and with the study targets held out (v3_holdout, used for evaluation).
CSV_NAME = 'CleanedData.csv'
VINA_NAME = 'vina.exe'
PRETRAINED_CKPT_NAME = 'checkpoint_pretrained_v11.pt'
EXEMPLAR_CKPT_NAME = 'checkpoint_exemplar_seed_v2.pt'
HELDOUT_CKPT_NAME = 'checkpoint_exemplar_seed_v3_holdout.pt'


# Returns the absolute path of `name`, searching the name as given, the current folder, this folder, its parent and grandparent, the parent of the current folder and any extra_dirs, in that order.
# Returns None if it is nowhere.
def find_file(name, extra_dirs=()):
    # an absolute path that exists is used as is
    if os.path.isabs(name) and os.path.exists(name):
        return name
    cands = [name, os.path.join(os.getcwd(), name)]
    for d in [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)), os.getcwd(), os.path.dirname(os.getcwd())] + list(extra_dirs):
        cands.append(os.path.join(d, name))
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# Resolves a default file: where find_file finds it, else the given fallback, else the bare name (so a later 'file not found' message names the file that was wanted).
def default_path(name, fallback=None):
    return find_file(name) or fallback or name


# Switch shared by base_data and base_train.  True = wiring check only: a 200-row sample and a tiny model.  Leave False for real work.
SMOKE_TEST = False


# Default locations, resolved once at import time (found next to this file or in a parent folder, else the fallback).
DEFAULT_CSV = default_path(CSV_NAME, os.path.join(LEGACY_ROOT, 'Bionic Code', CSV_NAME))
DEFAULT_VINA = default_path(VINA_NAME, os.path.join(LEGACY_ROOT, VINA_NAME))
DEFAULT_PRETRAINED_CKPT = default_path(PRETRAINED_CKPT_NAME, os.path.join(LEGACY_ROOT, PRETRAINED_CKPT_NAME))
DEFAULT_EXEMPLAR_CKPT = default_path(EXEMPLAR_CKPT_NAME, EXEMPLAR_CKPT_NAME)
DEFAULT_HELDOUT_CKPT = default_path(HELDOUT_CKPT_NAME, HELDOUT_CKPT_NAME)


# Prints where each default file resolves to (ok / MISSING) and whether the code modules next to this file are all present.  Run as: python config.py
def report():
    rows = [('library CSV', DEFAULT_CSV), ('vina executable', DEFAULT_VINA), ('pretrained base model', DEFAULT_PRETRAINED_CKPT),
            ('exemplar model (all targets)', DEFAULT_EXEMPLAR_CKPT), ('exemplar model (study targets held out)', DEFAULT_HELDOUT_CKPT)]
    missing = 0
    for label, p in rows:
        ok = os.path.exists(p)
        missing += (not ok)
        print(f"  {'ok     ' if ok else 'MISSING'}  {label:<40} {p}")
    print(f"\n{len(rows) - missing} of {len(rows)} found" + ('' if not missing else '  (a MISSING file is only a problem for the scripts that use it)'))
    mods = ['base_data.py', 'base_model.py', 'base_train.py', 'docking.py', 'targets.py', 'rl_utils.py', 'exemplar_data.py', 'exemplar_model.py']
    gone = [m for m in mods if not os.path.exists(os.path.join(HERE, m))]
    print('code modules: ' + ('all ' + str(len(mods)) + ' present' if not gone else 'MISSING ' + ', '.join(gone)))


if __name__ == '__main__':
    report()
