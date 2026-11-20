# Rebuilding everything from scratch

Draft of 5 Nov 2026. Assumes the portable package is extracted in `C:\Users\sidha\Desktop\TEMP` and all commands run in cmd.exe from `C:\Users\sidha\Desktop\TEMP\refactored_model` (called "the project folder" below).

Read this first:

- A rebuild makes NEW models. They will not be identical to the shipped checkpoints, so none of the validation numbers in `RESULTS_AND_LIMITS.md` carry over to them. Rebuild to show the pipeline works end to end, not to reproduce the reported results. The reported results belong to the shipped checkpoints.
- Do the rebuild in a spare copy of TEMP. New files are saved under new names, but keep the shipped ones safe.
- Only part of the chain runs from the GUI. The table says which. The start point is `CleanedData.csv`, which was made by hand.

## 0. Where each step runs

Start point: `CleanedData.csv` is made by hand and is the input (no cleaning script is needed). Put it in the project folder.

| Step | What | Where it runs |
| --- | --- | --- |
| 1 | Pretrain the base model | Command line: `pretrain_base_model.py` (new file, uses the refactored modules) |
| 2 | Train the seed-and-edit model | GUI, Pipeline tab, Train stage |
| 3 | Conditioning check | GUI, Validate stage |
| 4 | Generate molecules | GUI, Generate stage (or New context tab for a custom request) |
| 5 | Random-seed baseline | GUI, Baseline stage (study targets only) |
| 6 | Dock | GUI, Dock stage (PfDHFR, and DPP-4 with the rule) |
| 7 | RL tuning for PfDHFR | Command line only (`rl_pilot\rl_pfdhfr.py`) |
| 8 | Check the package | Command line (`check_portable.py`) |
| Separate | Receptor files (only if you also rebuild the docking inputs) | Command line, section 1 |

## 1. Receptor files (only for rebuilding the docking inputs)

The package already contains the receptor files for docking (`rec_*.pdbqt` and the box files). You do not need this section for steps 1 to 6. It is here because `Ligandsfiltering.py` and `E64filter.py` are the scripts that split the downloaded PDB files:

- `Ligandsfiltering.py` reads `1J3I.pdb`, `1X70.pdb` and `5I6X.pdb` from the current folder and writes `malaria_ligand.pdb`, `malaria_receptor_raw.pdb`, `diabetes_ligand.pdb`, `diabetes_receptor_raw.pdb`, `depression_ligand.pdb` and `depression_receptor_raw.pdb` (ligand WRA in chain A of 1J3I, 715 in 1X70, 8PR in 5I6X; waters, sugars and ions stripped; for 5I6X only chain A kept).
- `E64filter.py` reads `3BPF.pdb` and writes `falcipain2_ligand.pdb` and `falcipain2_receptor_raw.pdb` (ligand E64).

Run each from a folder holding the PDB files: `python Ligandsfiltering.py` and `python E64filter.py`. These scripts only split the PDB files. How the `_receptor_raw.pdb` files become `rec_*.pdbqt` and the box files for the three original receptors is not in the scripts I have; `prep_receptors.py` (in the package) does that conversion and writes the box for the extra receptors (2ONC, 2RGU and others), but I have not checked it against the original malaria receptor. For the rebuild test, use the shipped receptor files.

## 2. Step 1: pretrain the base model

`pretrain_base_model.py` pretrains the base model using only the refactored modules in the project folder (`config.py`, `base_data.py`, `base_model.py`, `base_train.py`). No legacy file is involved. It repeats what the old v13 script did at the top level:

1. reads `CleanedData.csv` and drops rows whose SMILES is longer than 254 characters,
2. builds the preprocessor and dataset,
3. sets `torch.manual_seed(123)` and splits ALL rows 80/10/10 at random,
4. batch size 32, training shuffled,
5. builds the model (embed 256, 8 heads, 3 encoder and 3 decoder layers, feed-forward 1024) and trains up to 10 epochs, stopping after 3 epochs without a better validation loss, learning rate 1e-3, keeping the best-validation weights.

Files needed in the project folder: `CleanedData.csv` and `pretrain_base_model.py` (the package already has the rest). Nothing needs editing and nothing extra needs installing.

Run, in this order:

    cd /d C:\Users\sidha\Desktop\TEMP\refactored_model
    python pretrain_base_model.py --dry-run
    python pretrain_base_model.py --smoke
    python pretrain_base_model.py

- `--dry-run` reads the data, builds the model and loaders, and trains nothing (a minute or less).
- `--smoke` trains on 100 rows for 1 epoch and writes `checkpoint_pretrain_smoketest.pt`. It proves the whole route works before the long run.
- The last command is the full run. It writes `checkpoint_pretrained_rebuild.pt`. The script refuses to write `checkpoint_pretrained_v11.pt`, so the shipped base model cannot be overwritten.
- After a real run the script re-loads the new file with the package's own loader and compares it with the shipped base model (found by the package's usual lookup, so the one in `TEMP`): same weight names and shapes, same architecture settings. The weights themselves will differ, since it is a new training run.

Facts about this step:

- The split is seeded, so a rerun with the same `CleanedData.csv` gives the same split. GPU training is still not bit-for-bit repeatable, so the weights can differ.
- The base model sees the molecules of every target, including those held out later. Results mean "unseen by the seed-and-edit stage", as in `RESULTS_AND_LIMITS.md`.
- Tested on 5 Nov 2026 (study machine, GPU): the full run took 19.8 minutes. It ran all 10 epochs; the validation loss was still falling slowly (0.6671 at epoch 9, 0.6588 at epoch 10), so early stopping did not trigger and the cap of 10 epochs ended the run. Final validation accuracy 75.9%. By epoch 2 the sample check gave 5 of 5 valid molecules. The new file loads with the package's loader and has the same 98 weight tensors, shapes and architecture settings as the shipped base model.
- The data step printed: 328 rows dropped (SMILES over 254 characters) of 42,938, leaving 42,610 rows (34,088 train, 4,261 validation, 4,261 test).
- Compared with the shipped base model: its recorded best validation loss is 0.6546; the rebuilt one is 0.6588 (0.6% higher). Both are single runs on a seeded split of the same kind, so a gap this small is within what a rerun can change; the rebuilt model is as good as the shipped one for practical purposes. Print the shipped value with: `python -c "import torch; c=torch.load(r'..\checkpoint_pretrained_v11.pt', map_location='cpu', weights_only=False); print(c.get('extra_info'))"` from `refactored_model` (the shipped file is one folder up, in TEMP).
- A first launch of the full run was stopped by hand during start-up (a Ctrl+C); the second launch ran normally. Nothing to fix.

## 3. Step 2: train the seed-and-edit model (GUI)

In the Train stage the GUI now has a **Base model** list. It shows the shipped base model `checkpoint_pretrained_v11.pt` (which sits one level up, in `TEMP` itself, next to `vina.exe`) and any rebuilt one made by `pretrain_base_model.py` (`checkpoint_pretrained_rebuild*.pt` in `refactored_model`). To train on your rebuilt base model, pick `checkpoint_pretrained_rebuild.pt` in that list. Nothing has to be copied or renamed, and the shipped file is never touched. (With an older GUI that has no Base model list, copy the rebuilt file to `checkpoint_pretrained_v11.pt` inside `refactored_model`; the code looks in the current folder first. Delete that copy afterwards.)

Start the GUI:

    cd /d C:\Users\sidha\Desktop\TEMP\refactored_model
    python gui_server.py

The browser opens by itself (otherwise open http://127.0.0.1:8765). `python gui_server.py --check` first prints a pass/fail list of what the page needs. In the Pipeline tab, Train stage:

- Model name: use a new name, for example `rebuild`. The output is `checkpoint_exemplar_rebuild.pt` in the project folder. The stage is skipped if that file already exists, so a new name is needed each time.
- Steps: 6000 (the value used for the shipped models).
- Base model: the rebuilt one, or the shipped one for the quick test.
- Held-out targets: enter `T89515 T31391 T27812 T52522 T45993`. The shipped `v3_holdout` model holds out 17 whole targets: these five (the three study targets plus the two GUI defaults) are the ones that were named, and the script adds the other 12 by a seeded random draw, so the same five names should give the same 17. Entering only the three study targets gave a different draw (10 of the 17 matched, 7 did not), so use all five. Check the log's `[holdout]` list against the shipped model's list, printed by: `python -c "import torch; a=torch.load('checkpoint_exemplar_seed_v3_holdout.pt', map_location='cpu', weights_only=False); print(sorted(a['holdout']['held_targets']))"`. The shipped list is T06773 T12499 T14912 T27812 T30082 T31391 T45993 T52522 T54128 T56915 T58921 T66665 T68782 T76369 T77913 T83145 T89515.
- The script holds out whole targets and a further 8% of molecules, runs 6,000 steps at batch 32 and keeps the final step.

Tested on 5 Nov 2026 (study machine, GPU), with the rebuilt base model and held-out targets `T89515 T31391 T27812 T52522 T45993`:

- Time: about 13 minutes for 6,000 steps by the clock (started about 16:57, finished by 17:10). The early "[speed]" line in the log (0.49 s/step, about 49 min) was a pessimistic estimate; the real run was much faster.
- The log's `[holdout]` list matched the shipped `v3_holdout` list exactly (the same 17 targets).
- Training loss fell steadily (with a near-neighbour exemplar: 1.668 at step 250, 0.644 at step 6000). At unseen targets, for molecules with a near neighbour in the library, loss without exemplars was 0.663 and with the neighbour as the exemplar 0.431 (at step 0 it was 1.775): the model learned to use the seed.
- The in-training free-running check at step 6000 (small, one sample per target; it varies between readings): valid 61% in one reading and 54% in the saved re-evaluation, target lift 1.25 to 1.27, disease lift 0.94 to 0.98. These are NOT comparable with the Validate stage figures in `RESULTS_AND_LIMITS.md` (general model, three-seed means: target lift 1.47, disease lift 1.07, validity 70%); compare the rebuilt model only through the Validate stage (section 4).
- The disease swap gap at unseen targets stayed near zero or negative (-0.017 at step 6000, n = 75), in line with the finding that disease conditioning is weak.

## 4. Steps 3 to 6 in the GUI, using the new model

The later stages use the newly trained model automatically. In the Pipeline tab, in this order:

1. Validate (conditioning check): 17 unseen targets, temperature 0.7. Run it twice; the two runs must give identical numbers.
2. Generate: for each study target, with temperature 0.7 and seed 0, n = 70.
3. Baseline: random-seed baseline for each study target, same n and seed.
4. Dock: PfDHFR (plain Vina, receptor 1J3I) and DPP-4 (rule `dpp4_amine_glu`, receptor 2ONC). SERT is not docked because no receptor calibrates. Docking is cached; a repeat run is fast.

Expected on a rebuilt model: numbers of the same kind as the shipped general held-out model, but not the same values.

Tested on 5 Nov 2026 (Validate stage on `checkpoint_exemplar_rebuild2.pt`, 17 unseen targets, temperature 0.7, k = 1): target lift 1.41 (novel-only 1.38), disease lift 1.02, validity 71%, exact copies of a seed 3%. The shipped general held-out model gave 1.49, 1.05 and 69% at the same seed (0), and 1.47, 1.07 and 70% averaged over seeds 0 to 2. The check was run twice and the two outputs were identical, line for line. Target conditioning passes the script's threshold in both models; disease conditioning is below its pass mark in both (1.02 here, 1.05 to 1.09 shipped), based on only six multi-disease contrasts per run, so the difference is small relative to the noise. Docking was not rebuilt in this test.

## 5. Step 7: RL for PfDHFR (command line only)

Only if you want an RL-tuned model. The package contains `rl_pilot\rl_pfdhfr.py` and the shipped r3 model. The r3 settings (with a new tag, because the script refuses to reuse a tag whose run folder exists, and `r3` is already in the package):

    cd /d C:\Users\sidha\Desktop\TEMP\refactored_model
    python rl_pilot\rl_pfdhfr.py --tag r3_rebuild --lr 5e-5 --kl 0.15 --max-minutes 90 --checkpoint checkpoint_exemplar_rebuild.pt

By default the script starts from the shipped held-out model (`--checkpoint`); the added `--checkpoint` above points it at your rebuilt model instead. It writes its checkpoint under `rl_pilot\`. Docking inside it uses the PfDHFR receptor, so `vina.exe` and the receptor files must be present (the package has them). A rebuilt r3 will not reproduce the shipped one exactly. Nothing shipped is overwritten.

## 6. Step 8: check the result

    python check_portable.py --deep --smoke

(`--hashes` fails for new files, because the hashes are for the shipped files.)

## 7. Which route to pick

- Quick test of the whole chain: run `--smoke` in step 1, then steps 2 to 6 with the shipped base model. Minutes for the smoke run, plus the Train stage and docking.
- Full rebuild: the full step 1 run, then steps 2 to 6 on `checkpoint_pretrained_rebuild.pt`.
- Do not replace the shipped checkpoints with rebuilt ones in the final package. All reported results are tied to the shipped files.
