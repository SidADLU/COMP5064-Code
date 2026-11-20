# Frozen trust decisions (Sat 31 Oct 2026)

Docking is used only at receptors that passed calibration (Phase 1). This file records the decision, the rules it rests on, the numbers, and the exact commands.
Nothing in this file may be changed after the fact; a later change goes in a new dated section.

## Receptor status

| Target | Receptor used | Scoring | Phase 1 result | Status |
|---|---|---|---|---|
| PfDHFR (T89515) | 1J3I | plain Vina | AUC 0.72 +/- 0.05 (research stratum), size-corrected 0.73, redock 1.59 A | TRUSTED |
| DPP-4 (T31391) | 2ONC | plain Vina + interaction rule `dpp4_amine_glu` | AUC 0.88 +/- 0.05, size-corrected 0.89 (plain Vina on the same dockings 0.63; amine-only control 0.69); redock 1.56 A | TRUSTED ONLY WITH THE RULE |
| DPP-4 | 1X70 | same rule | AUC 0.77 +/- 0.07 (plain 0.49; control 0.55); redock 9.59 A fails | not used |
| SERT (T27812) | 5I6X | plain Vina | 0.51 / 0.54 | NOT VALIDATED |
| SERT | 5I6X / 7LWD | rule `sert_amine_asp98` | 0.65 +/- 0.06 (size-corrected 0.67, weak) / 0.58 +/- 0.06 (non-discriminating) | NOT VALIDATED |

Other SERT / DPP-4 variants tried and not sufficient: potency filter, Na+ ion, ensemble of 4 DPP-4 structures (0.59), imipramine structure 7LWD with plain Vina (0.64).

## Rules (fixed before the runs that use them)

* Trust rule: AUC >= 0.70 with AUC - SE > 0.5, and size-corrected AUC >= 0.60. PfDHFR is decided on the research stratum only (n >= 15).
* Interaction rule: 9 poses per ligand; a pose satisfies the rule if a basic nitrogen is within 4.0 A of OE1/OE2 of Glu205 or Glu206 (DPP-4) / OD1/OD2 of Asp98 (SERT); score = best Vina energy among satisfying poses, otherwise the plain best score + 2.0 kcal/mol. Constants chosen a priori, not fitted. The rule was introduced after the plain DPP-4 test failed, so it is a post-hoc fix: report DPP-4 as "validated with a pharmacophore constraint", never as plain-Vina validation.
* SERT rule constants are copied unchanged from the DPP-4 rule.

## Phase 2 results with the held-out model (checkpoint_exemplar_seed_v3_holdout.pt; study targets not seen in training)

| | n generated / baseline | generated size-corr. AUC | random-seed baseline | conditioned vs baseline | seeds | outcome |
|---|---|---|---|---|---|---|
| PfDHFR | 44 / 64 | 0.55 [0.46-0.64] | 0.56 | 0.48 [0.37-0.59] | 0.73 | B (generated ~ decoys) |
| DPP-4 (rule) | 65 / 68 | 0.50 [0.40-0.60] | 0.37 [0.27-0.47] | 0.63 [0.54-0.72] | 0.80 | B (generated ~ decoys) |

* DPP-4 diagnostics (rule_gen_check.py): basic nitrogen 60% generated vs 37% baseline vs 100% seeds; satisfy the rule 48% vs 25% vs 96%; conditioned vs baseline with plain Vina 0.50 (with the rule 0.60); analogs keep the rule of a satisfying seed 51% of the time; analog - seed plain Vina +0.34 kcal/mol (36% better).
* The conditioned set carries the amine pharmacophore more often than random seeds, but plain Vina sees no better binders. The earlier model that had seen PfDHFR (outcome C, 0.72 vs 0.57) does not carry over to unseen targets.
* Similarity to the seed shows only a weak, non-significant link with docking percentile (Spearman +0.13 DPP-4, +0.12 PfDHFR; 6-12 molecules at similarity >= 0.4-0.5), so a similarity threshold is not supported as a fix.
* Still open at the time of writing: PfDHFR re-sample confirmation (different sampling seed; done Mon 2 Nov, see the next section) and the Phase 3 (RL) decision on Fri 6 Nov.

## Commands used

    python dock_generated.py --cases diabetes --extra-receptors extra_dpp4_2onc.json --interaction dpp4_amine_glu --calibration-only --noise-check 0 --cache dock_cache_inter.jsonl --out dock_results_inter.csv --summary dock_summary_inter.json
    python generate_for_request.py --checkpoint checkpoint_exemplar_seed_v3_holdout.pt --target T31391 --n 70 --out gen_diabetes_ho.csv
    python generate_for_request.py --checkpoint checkpoint_exemplar_seed_v3_holdout.pt --random-seeds --target T31391 --n 70 --out gen_diabetes_baseline_ho.csv
    python dock_generated.py --cases diabetes --extra-receptors extra_dpp4_2onc.json --interaction dpp4_amine_glu --baseline --n 70 --noise-check 0 --gen-pattern gen_{key}_ho.csv --baseline-pattern gen_{key}_baseline_ho.csv --cache dock_cache_inter.jsonl --out dock_results_dpp4_gen.csv --summary dock_summary_dpp4_gen.json
    python dock_generated.py --cases depression --extra-receptors extra_sert_7lwd.json --interaction sert_amine_asp98 --calibration-only --noise-check 0 --cache dock_cache_inter_sert.jsonl --out dock_results_inter_sert.csv --summary dock_summary_inter_sert.json
    python dock_generated.py --cases malaria_t89515 --baseline --n 70 --gen-pattern gen_{key}_ho.csv --baseline-pattern gen_{key}_baseline_ho.csv --out dock_results_malaria_ho70.csv --summary dock_summary_malaria_ho70.json
    python rule_gen_check.py --cache dock_cache_inter.jsonl --label DPP4-2ONC --gen gen_diabetes_ho.csv --baseline gen_diabetes_baseline_ho.csv

## Phase 2 PfDHFR re-sample (Mon 2 Nov 2026; held-out model, --seed 1; added after the freeze, earlier sections unchanged)

| | first run (seed 0) | re-sample (seed 1) |
|---|---|---|
| generated, n docked | 44 | 47 (51 dockable of 70; 15 skipped: seed lacks the pharmacophore, 4: seed dropped; 6 dockings failed, no pose) |
| generated size-corrected AUC | 0.55 [0.46-0.64] | 0.60 [0.51-0.69] (in-range only, n 44: 0.63 [0.54-0.71]) |
| random-seed baseline, n docked | 64 | 65; size-corrected AUC 0.51 [0.41-0.60] (first run 0.56) |
| conditioned vs baseline | 0.48 [0.37-0.59] | 0.55 [0.45-0.66] |
| seeds (ceiling for the analogs) | 0.73 | 0.74 (32 distinct seeds) |
| analog - seed (kcal/mol, positive = analog worse) | +0.59 | +0.43; analogs better than their seed 38% |
| script verdict | outcome B | PARTIAL / "borderline" (PASS needs >= 0.65 with the interval above 0.5) |

* Reading: generated molecules score between the decoys (0.50) and the seeds / known binders (0.73-0.74), more likely above decoys than not but below the PASS line in both samples. The random-seed baseline sits at decoy level (0.51-0.56). Conditioning shows no detectable docking advantage over the baseline in either sample (0.48, 0.55; both intervals include 0.5).
* Split in the re-sample: molecules whose seed carries the antifolate pharmacophore (n 5) 0.73, the rest (n 42) 0.59.
* The sentence in the script's verdict caveats "the generated molecules come from a model that saw these targets in training" does not apply to this run (held-out model).
* Command: generate with --seed 1 (target and random-seed sets), then dock_generated.py --cases malaria_t89515 --baseline --n 70 --gen-pattern gen_{key}_ho_s1.csv --baseline-pattern gen_{key}_baseline_ho_s1.csv --out dock_results_malaria_ho70_s1.csv --summary dock_summary_malaria_ho70_s1.json
