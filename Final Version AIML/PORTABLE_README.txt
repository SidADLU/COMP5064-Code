EXEMPLAR GENERATOR - PORTABLE PACKAGE
Built 2026-11-06 07:43 by make_portable_zip.py.  Optional PfDHFR-tuned model (RL r3): included (checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt).

SET-UP ON A NEW MACHINE
  1. Extract the zip anywhere (a path with spaces is fine).  You get this layout:
       <root>/                    files the project keeps one level up: vina.exe, the pretrained base model, older scripts
       <root>/refactored_model/    the project: code, GUI, receptor files, checkpoints, caches, CleanedData.csv
  2. Install Python 3.13 (3.10 or newer is likely fine) and the packages:
       cd refactored_model
       pip install -r requirements.txt          (read the torch line first: GPU users install the CUDA build of torch first)
  3. Check that nothing is missing and the machine can run it:
       python check_portable.py                 files against the manifest, inputs, packages, vina
       python check_portable.py --deep --smoke  also the GUI self-check and a 3-molecule generation test
  4. Start the page:
       python gui_server.py                     opens the browser;  python gui_server.py --demo  shows made-up results without any model

WHAT IT CAN DO
  generate analogs of known ligands for a disease or target (held-out model: checkpoint_exemplar_seed_v3_holdout.pt), dock them at the receptors that passed calibration, and add
  a new disease or target from a few ligands.  Docking is validated for PfDHFR (plain Vina) and DPP-4 (only with the interaction rule); SERT is not validated, so docking is not offered.
  Two models are offered in the GUI and with --checkpoint: the GENERAL model (checkpoint_exemplar_seed_v3_holdout.pt, the default) and the optional PfDHFR-tuned model
  (checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt: fine-tuned by reinforcement learning on one target; for Malaria / T89515 only, it keeps the antifolate pharmacophore and stays close to its seeds).
  See FROZEN_TRUST_DECISIONS.md for the trust decisions and rl_pilot/RL_PILOT_README.txt for the RL record and the wording of its claims.
  RESULTS_AND_LIMITS.md says what the tool can and cannot be claimed to do.

REBUILDING THE MODELS (optional)
  REBUILD_FROM_SCRATCH.md lists every step from CleanedData.csv: pretrain_base_model.py for the base model, then the GUI's Train stage and the later stages.  A rebuild makes NEW models:
  the shipped checkpoints are the ones the reported results belong to, so keep them.

NEEDS NOTHING ELSE
  Everything the scripts read is in this package: library, models, receptor files, docking program, caches.  Generating, docking and the GUI need no internet.  Only two OPTIONAL tools go online:
  fetch_potency.py (ChEMBL potency filter) and prep_receptors.py (downloads a PDB file if it is not already here).  A Windows 64-bit machine with the python packages is enough.

NOTES
  vina.exe is a Windows program.  On Linux or macOS, put a native AutoDock Vina 1.2.7 binary under the same name in the same place.
  config.py contains one absolute fallback path from the original machine (LEGACY_ROOT); it is only used when a file is found nowhere else and can be ignored.
  The GUI writes its results to gui_runs/ (not packed).  Deleting that folder is safe.
