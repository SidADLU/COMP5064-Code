#!/usr/bin/env python3
"""
pretrain_base_model.py -- pretrain the base language model with the refactored modules only (base_data.py, base_model.py, base_train.py, config.py).
No legacy file is needed.

It repeats what the old v13 script did at the top level, step by step:
  1. read CleanedData.csv and drop rows whose SMILES is longer than 254 characters,
  2. build the preprocessor and the dataset (base_data.preprocess_data),
  3. torch.manual_seed(123), then a random 80/10/10 split of ALL rows (train / validation / test) -- the same call v13 made, so the split is the same
     for the same CSV,
  4. data loaders with batch size 32 (training shuffled, 0 worker processes),
  5. build the model and the agent (base_train.initialize_model, DrugGeneratorAgent),
  6. base_train.pretrain_model: up to 10 epochs, early stopping after 3 epochs without a better validation loss, learning rate 1e-3, best-validation weights kept,
  7. save the checkpoint, then (unless --no-verify) re-load it with base_train.load_trained_agent and compare it with the shipped checkpoint_pretrained_v11.pt
     (same weight names, shapes and architecture settings; the weights themselves will differ, since this is a new training run).

The base model sees ALL targets, including the ones held out later in the seed-and-edit stage ("unseen" means unseen by that stage, not by this one).

  python pretrain_base_model.py --smoke        100 training rows, 20 validation rows, 1 epoch: a wiring check.  Writes checkpoint_pretrain_smoketest.pt
  python pretrain_base_model.py                full pretraining.  Writes checkpoint_pretrained_rebuild.pt (it refuses to write the shipped name)
  python pretrain_base_model.py --dry-run      reads the data and builds the model and loaders; trains nothing

Options: --csv PATH   --out NAME   --epochs N   --no-verify
Run it from the project folder (the folder that holds config.py).  ASCII only.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split

import config as cfg
from base_data import preprocess_data, collate_fn
from base_train import DrugGeneratorAgent, initialize_model, pretrain_model, load_trained_agent, PRETRAIN_MAX_EPOCHS

SHIPPED_BASE = cfg.PRETRAINED_CKPT_NAME     # checkpoint_pretrained_v11.pt
MAX_SMILES_CHARS = 254                      # v13 dropped longer SMILES so nothing is silently cut at the 256-token limit
SPLIT_SEED = 123                            # torch.manual_seed(123) at the top of v13
BATCH_SIZE = 32
NUM_WORKERS = 0                             # 0: no worker processes (they cause trouble on Windows)


def shapes_of(ck):
    return {k: tuple(v.shape) for k, v in ck.get('model_state_dict', {}).items()}


def verify(new_path, shipped_path):
    print('\n--- verify: %s ---' % new_path)
    print('size: %.1f MB' % (os.path.getsize(new_path) / 1e6))
    load_trained_agent(new_path, device='cpu')          # the refactored loader must accept it
    print('re-loaded with base_train.load_trained_agent: OK')
    ck = torch.load(new_path, map_location='cpu', weights_only=False)
    print('top-level keys:', sorted(ck.keys()))
    for k, v in ck.items():
        if k in ('model_state_dict', 'model_hyperparams', 'agent_hyperparams'):
            continue
        if not hasattr(v, 'shape') and len(str(v)) < 200:
            print('  %s = %s' % (k, v))
    if not os.path.isfile(shipped_path):
        print('(shipped %s not found; no comparison made)' % os.path.basename(shipped_path))
        return
    old = torch.load(shipped_path, map_location='cpu', weights_only=False)
    a, b = shapes_of(ck), shapes_of(old)
    print('weight tensors: new %d, shipped %d; same names and shapes: %s' % (len(a), len(b), a == b))
    if a != b:
        print('  only in new:', sorted(set(a) - set(b))[:8], ' only in shipped:', sorted(set(b) - set(a))[:8],
              ' shape differences:', sorted(k for k in set(a) & set(b) if a[k] != b[k])[:8])
    print('model_hyperparams equal to shipped: %s' % (ck.get('model_hyperparams') == old.get('model_hyperparams')))
    print('(the weights themselves are expected to differ: a new run is a new model)')


def main():
    ap = argparse.ArgumentParser(description='Pretrain the base model with the refactored modules.')
    ap.add_argument('--smoke', action='store_true', help='100 training rows, 20 validation rows, 1 epoch')
    ap.add_argument('--csv', default=None, help='path of CleanedData.csv (default: found by config.py)')
    ap.add_argument('--out', default=None, help='output checkpoint name (default checkpoint_pretrained_rebuild.pt; checkpoint_pretrain_smoketest.pt with --smoke)')
    ap.add_argument('--epochs', type=int, default=PRETRAIN_MAX_EPOCHS, help='maximum epochs (default %d, as v13)' % PRETRAIN_MAX_EPOCHS)
    ap.add_argument('--no-verify', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='build data, model and loaders, then stop')
    args = ap.parse_args()

    csv_path = args.csv or cfg.DEFAULT_CSV
    if not os.path.isfile(csv_path):
        sys.exit('CleanedData.csv not found at %s.  Give the file with --csv.' % csv_path)
    out_name = args.out or ('checkpoint_pretrain_smoketest.pt' if args.smoke else 'checkpoint_pretrained_rebuild.pt')
    if os.path.basename(out_name) == SHIPPED_BASE:
        sys.exit('Refusing to write the shipped name %s.  Choose another --out.' % SHIPPED_BASE)
    out_path = os.path.abspath(out_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('data:    %s' % csv_path)
    print('output:  %s' % out_path)
    print('mode:    %s' % ('SMOKE (100 training rows, 20 validation rows, 1 epoch)' if args.smoke else 'FULL pretraining (up to %d epochs)' % args.epochs))
    print('device:  %s' % device)

    # 1. data
    df = pd.read_csv(csv_path)
    long_mask = df['Canonical Smiles'].fillna('').str.len() > MAX_SMILES_CHARS
    print('Dropping %d rows with SMILES > %d characters (of %d).' % (long_mask.sum(), MAX_SMILES_CHARS, len(df)))
    df = df[~long_mask].reset_index(drop=True)

    # 2. preprocessor and dataset
    dataset, preprocessor = preprocess_data(df)
    print('rows used: %d' % len(dataset))

    # 3. split (same seed and call as v13)
    torch.manual_seed(SPLIT_SEED)
    total = len(dataset)
    n_train, n_val = int(0.8 * total), int(0.1 * total)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, total - n_train - n_val])
    print('split: train %d, validation %d, test %d' % (len(train_ds), len(val_ds), len(test_ds)))

    # 4. loaders
    pin = (device == 'cuda')
    if args.smoke:
        train_ds, val_ds = Subset(train_ds, range(100)), Subset(val_ds, range(20))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_fn)

    # 5. model and agent
    model = initialize_model(preprocessor)
    agent = DrugGeneratorAgent(model, preprocessor, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print('model built: %.2f M parameters' % (n_params / 1e6))
    if args.dry_run:
        print('dry run finished: data, model and loaders are fine; nothing was trained.')
        return

    # 6. pretrain and save (pretrain_model saves the best-validation weights to out_path)
    t0 = time.time()
    pretrain_model(agent, train_loader, val_loader, device=device,
                   max_epochs=1 if args.smoke else args.epochs, patience=1 if args.smoke else 3, checkpoint_path=out_path)
    print('\nDone in %.1f minutes: %s' % ((time.time() - t0) / 60.0, out_path))

    # 7. check the file
    if not args.no_verify:
        verify(out_path, cfg.find_file(SHIPPED_BASE) or SHIPPED_BASE)


if __name__ == '__main__':
    main()
