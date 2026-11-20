# Results and limits

Draft of 4 Nov 2026. This page says what the tool can and cannot be claimed to do. All numbers come from runs on the study machine (AutoDock Vina 1.2.7, Python 3.13, torch 2.13); the run records are in `rl_pilot/RL_PILOT_README.txt` and `FROZEN_TRUST_DECISIONS.md`.

## 1. What the tool is

A seed-and-edit molecule generator. For a chosen disease and target it takes the known ligands of that target (or of the exact disease-target pair, when there are 3 or more) from `CleanedData.csv`, shows the model one such ligand at a time (the seed) and keeps the valid, unique, new analogs it writes. A new disease or target is added by supplying its known ligands; nothing is retrained.

Models shipped:

| File | Use it for |
| --- | --- |
| `checkpoint_exemplar_seed_v3_holdout.pt` | Default. Study targets were held out of training, so results describe unseen targets. |
| `checkpoint_exemplar_seed_v2.pt` | Trained on all targets except 16. For requests where you want the strongest conditioning on a library target. |
| `checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt` | Optional. RL-tuned for PfDHFR (T89515) only. Do not use it for other targets. |

## 2. Docking trust, per receptor

A receptor is trusted only if known actives separate from property-matched decoys: AUC at least 0.70 with AUC minus its standard error above 0.5, and size-corrected AUC at least 0.60. The rule was fixed before the generated molecules were docked.

| Target (dataset code) | Receptor | Set-up | Calibration AUC | Trusted |
| --- | --- | --- | --- | --- |
| Malaria, PfDHFR (T89515) | 1J3I | Plain Vina | 0.72 (size-corrected 0.73), research stratum n = 35 | Yes |
| Type-2 diabetes, DPP-4 (T31391) | 2ONC | Vina plus the interaction rule (basic amine within 4 A of Glu205/206 oxygens, else +2 kcal/mol) | 0.88 (plain Vina 0.63); 1X70 0.77 | Yes, only with the rule |
| Depression, SERT (T27812) | 5I6X | None reached 0.70 (potency filter, Na+/Cl- ions, 7LWD structure, Asp98 rule: best 0.65 to 0.68) | - | No |

Plain Vina at DPP-4 gave 0.47 to 0.64 on single structures, 0.59 on a 4-structure ensemble and 0.54 with a potency filter. The amine-only control (no geometry) gave 0.55 and 0.69 against 0.77 and 0.88 for the full rule, so most of the gain comes from the geometry.

## 3. What the results show

Conditioning (17 unseen targets, one sample per context, k = 1, temperature 0.7, check_exemplar_conditioning.py). Each run draws 30 samples per target, so single runs vary. The table gives three runs at seeds 0, 1 and 2 (mean, with the range in brackets), made on the shipped package on 6 Nov 2026:

| Model | Target lift | Disease lift | Validity |
| --- | --- | --- | --- |
| General held-out model (`checkpoint_exemplar_seed_v3_holdout.pt`) | 1.47 (1.44 to 1.49) | 1.07 (1.05 to 1.09) | 70% (68 to 72%) |
| r3, PfDHFR-tuned (`checkpoint_exemplar_rl_r3_pfdhfr_tuned.pt`) | 1.62 (1.60 to 1.66) | 1.15 (1.09 to 1.18) | 63% (58 to 66%) |

Notes on these numbers:
- Target lift passes the script's threshold (1.30) at every seed in both models. Disease lift is below the pass mark (1.20) at every seed in both models, and for the general model it sits around the partial mark (1.08): 1.05, 1.06 and 1.09. Call disease conditioning weak, not absent.
- Validity is below 70% in most runs, and for r3 it is 58 to 66%. For the general model the unconditional validity (no exemplars) is 87 to 90%, so conditioning costs about 20 points.
- Seed 0 of r3 reproduces the figures first reported for it (1.60, 1.18, 64%). The general model's first-reported figures (1.43, 1.11, 71%) were not reproduced at seed 0 (1.49, 1.05, 69%). Target lift and validity are close to the seed 1 run (1.44, 72%), but the disease lift is above all three runs. The cause is not known, so those figures are superseded by the table above.
- The disease figure rests on six multi-disease contrasts per run, and one of them (T54128, Alzheimer disease vs Anaesthesia) is much higher than the rest in every run. Treat the disease lift as imprecise.
- At PfDHFR (T89515), the Malaria study target, the lift was 1.46, 1.74 and 1.55 for the general model and 2.25, 2.37 and 2.46 for r3. T89515 was held out of the supervised training of both models, but r3 was then RL-tuned on PfDHFR, so its T89515 figure is not an unseen-target result and its mean target lift is inflated by that one target.
- The check is repeatable: running a seed twice gives identical output, line for line.

Docking of generated molecules (size-corrected AUC against decoys, 0.5 is chance):

| Target | General held-out model | After RL (r3) |
| --- | --- | --- |
| PfDHFR | 0.55 and 0.60 at two seeds; random-seed baselines 0.56 and 0.51 | 0.70 and 0.69; same-seed controls 0.58 and 0.53; 0.70 and 0.69 again at higher search effort |
| DPP-4 | 0.50 [0.40-0.60], equal to decoys; beats its random-seed baseline (0.37): 0.63 [0.54-0.72] | Not attempted |
| SERT | Not evaluated: no receptor calibrates | Not attempted |

For PfDHFR, r3 puts 79 to 81% of its molecules in the analog range (similarity to seed 0.40 to 0.80; before RL, 9 to 19%) and keeps the pharmacophore in 94 to 96% (before, 23 to 27%). Its analogs dock better than their own seed in 57% and 52% of cases (before RL, 21 to 36%).

## 4. Claims that are supported

1. Given the known ligands of a target, the model writes molecules that resemble that target's ligands more than other targets' ligands, including for targets it was not trained on (target lift 1.44 to 1.66 across runs, against 1.0 for no conditioning).
2. For PfDHFR, the RL-tuned model r3 produces analogs that dock better than a same-seed control from the model before RL, at two fresh seeds and at a second, independent search effort.
3. Receptor trust was decided by a rule fixed in advance. Plain Vina is trusted at PfDHFR; at DPP-4 only with the interaction rule; SERT is not validated.

Suggested wording: "Conditioning on the known ligands of a target transfers to unseen targets (lift 1.4 to 1.7). Molecules from the general model do not dock better than random-seed controls at PfDHFR or decoys at DPP-4. After RL tuning on PfDHFR, generated analogs dock better than same-seed controls (AUC 0.70 and 0.69 against 0.58 and 0.53), largely because the tuned model keeps the antifolate pharmacophore the receptor's calibration rewards."

## 5. Claims that are not supported

1. **Disease conditioning.** The disease label works only by selecting the exact pair's ligands as seeds (disease lift 1.05 to 1.18, below the 1.20 pass mark). The model cannot produce disease-relevant chemistry from a disease name alone. The docking studies use `--target` only and test target conditioning, not disease conditioning.
2. **Binding.** Docking is a computed proxy with no experimental test. Trusted receptors have AUC 0.72 to 0.88, which is useful but not accurate for single molecules.
3. **DPP-4 and SERT molecules.** DPP-4 generated molecules score like decoys. SERT molecules were not evaluated.
4. **A general RL gain.** r3 was tuned on one target. It does not show that RL improves unseen targets.

## 6. Known limits

- **r3 is not independent of its reward.** The gain is largely the pharmacophore that the calibration rewards, and the evaluation uses the same receptor and scoring. Mean ligand efficiency is 0.35, against 0.40 for known binders. Validity fell from about 70% to about 63% (three-seed means).
- **Pretraining and "unseen".** The base model (`checkpoint_pretrained_v11.pt`) was pretrained by the v13 driver on a random 80/10/10 split of all library rows, so it saw the held-out targets' molecules, with their target IDs. "Unseen target" therefore means unseen by the seed-and-edit stage, not by the base model. The size of any leak is not measured; a clean test would repeat the pretraining without the held-out targets. Say "unseen by the seed-and-edit stage" in a write-up.
- **Dataset naming.** T89515 is the dataset's target code. Its library ligands are antifolates and the receptor used is PfDHFR (1J3I). A target lookup lists T89515 as polypeptide deformylase; the two do not agree.
- **Many generated molecules are not docked.** Only molecules whose seed carries the antifolate pharmacophore are docked at PfDHFR (selected on the seed, never on the output). About 25 to 45% of a set can show "not docked".
- **Poor analogs without good seeds.** With few or very different seed ligands, most molecules are only loosely related to the seeds. The GUI shows a red note when fewer than 25% of molecules have similarity 0.40 or more to their seed.
- **Platform.** `vina.exe` is a Windows build. On another system, supply a Vina 1.2.7 binary and set `--vina`.
- **Sampling noise.** A single run of 50 to 70 molecules gives a docking AUC with a spread of about plus or minus 0.1. Judge the model by pooling several seeds, as the adoption checks did.

## 7. Reproducibility

- The same checkpoint, `--seed`, n, temperature and seed pool give the same molecules on the same machine and library versions (checked: GUI and command-line runs gave 70 of 70 identical molecules). A different GPU or CPU can change a sampled token and so the set.
- Docking is cached by receptor, search effort, Vina seed (42) and SMILES, so a repeat gives identical scores.
- `python check_portable.py --hashes --deep --smoke` verifies every shipped file against its SHA-256 and runs a 3-molecule generation.

## 8. Future work

- DPP-4 gated RL: train with the interaction rule on 2ONC and evaluate on 1X70 as an independent check, with the pass criteria written down before the run.
- A SERT receptor set-up that reaches the trust rule.
- Check whether RL gains transfer to a target other than the one tuned.
