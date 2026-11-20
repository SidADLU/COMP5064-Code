RL PILOT (PfDHFR)  --  experiment folder, NOT part of the shipped tool
=======================================================================
Started Mon 2 Nov 2026.  HARD STOP: Fri 6 Nov 2026, 18:00 (Adelaide).  The GUI starts after that regardless of the result.

WHERE THINGS LIVE
  Save this folder as:        C:\Users\sidha\Desktop\AI ML code\refactored_model\rl_pilot\
                              (so rl_pfdhfr.py sits next to this README, and config.py is one level up)
  Run everything from:        C:\Users\sidha\Desktop\AI ML code\refactored_model     (the folder that holds config.py)
  Everything the pilot writes goes to  rl_pilot\runs\<tag>\   (log.csv, molecules.csv, dock_cache.jsonl, checkpoint_rl_<tag>.pt, summary.json)
  The pilot edits NO existing file.  The 19 project files, the caches and the checkpoints in refactored_model are untouched.

FINAL ZIP RULE
  The portable zip for a new machine EXCLUDES the whole rl_pilot\ folder, unless RL is adopted (see "adopting" below).
  If RL does not work: delete the rl_pilot folder (or leave it out of the zip).  Nothing else needs cleaning up.
  Files outside this folder that an RL run could touch: none.  (The evaluation commands below copy dock_cache_v2.jsonl INTO rl_pilot\runs\<tag>\ and write there.)

STEP 1  PLUMBING TEST (a few minutes, no Vina; scores are made up)
  python rl_pilot\rl_pfdhfr.py --smoke
  Expected: "[done] ... SMOKE TEST finished".  If it fails, send the full output; nothing else has been changed.

STEP 2  ONE REAL STEP (checks Vina inside the worker processes; ~1-2 min)
  python rl_pilot\rl_pfdhfr.py --tag smoke_real --steps 1 --batch 4 --workers 1
  Expected: one "step 1/1" line with "docked" at least 1 of 4.  (A tag can be used once; use a new tag for a repeat.)

STEP 3  THE PILOT RUN r1 (time box 75 min; stops earlier at 60 steps)
  python rl_pilot\rl_pfdhfr.py --tag r1
  Watch "pct" (mean percentile of docked molecules; the pre-RL model sits at about 0.55) and "valid" (pre-RL about 65-70%).  Ctrl+C stops safely and saves.
  Defaults: batch 32, lr 3e-5, KL weight 0.05, temperature 0.7, 3 docking workers (--workers N to change).

STEP 4  EVALUATE r1 ON A FRESH SAMPLE (about 20 min; the real test, not the training curve)
  python generate_for_request.py --checkpoint rl_pilot\runs\r1\checkpoint_rl_r1.pt --target T89515 --n 70 --seed 2 --out rl_pilot\runs\r1\gen_malaria_t89515_rl.csv
  python generate_for_request.py --checkpoint rl_pilot\runs\r1\checkpoint_rl_r1.pt --random-seeds --target T89515 --exclude-seed-targets T83386,T17345,T19229,T35465,T52227,T02702,T47077 --n 70 --seed 2 --out rl_pilot\runs\r1\gen_malaria_t89515_baseline_rl.csv
  copy dock_cache_v2.jsonl rl_pilot\runs\r1\eval_cache.jsonl
  python dock_generated.py --cases malaria_t89515 --baseline --n 70 --gen-pattern rl_pilot/runs/r1/gen_{key}_rl.csv --baseline-pattern rl_pilot/runs/r1/gen_{key}_baseline_rl.csv --cache rl_pilot/runs/r1/eval_cache.jsonl --out rl_pilot/runs/r1/dock_results.csv --summary rl_pilot/runs/r1/dock_summary.json
  Compare with the held-out model before RL:  generated 0.55 [0.46-0.64] (seed 0) and 0.60 [0.51-0.69] (seed 1); baseline 0.56 / 0.51.

STEP 5  ONLY IF STEP 4 LOOKS PROMISING: the two checks that can still sink it
  (a) independent re-docking (different search effort, so Vina-specific exploitation shows up; ~40 min, calibration is redone at the same setting):
      python dock_generated.py --cases malaria_t89515 --baseline --n 70 --exhaustiveness 16 --noise-check 0 --gen-pattern rl_pilot/runs/r1/gen_{key}_rl.csv --baseline-pattern rl_pilot/runs/r1/gen_{key}_baseline_rl.csv --cache rl_pilot/runs/r1/eval_cache_ex16.jsonl --out rl_pilot/runs/r1/dock_results_ex16.csv --summary rl_pilot/runs/r1/dock_summary_ex16.json
  (b) the conditioning check on the held-out targets (RL must not break target conditioning; before RL: target lift 1.43):
      python check_exemplar_conditioning.py --checkpoint rl_pilot\runs\r1\checkpoint_rl_r1.pt --temperature 0.7 --max-targets 17 --k 1

PRE-REGISTERED CRITERIA (fixed on 2 Nov, before any run; do not change after seeing results)
  Training curve (from summary.json / log.csv), run is "alive" if ALL hold:
      pct_mean_last10 - pct_mean_first5 >= 0.08;  validity_last5 >= 50%;  mean heavy atoms stays within 15-38.
  Evaluation (step 4), run is "promising" if:
      generated size-corrected AUC >= 0.65 with the 95% interval above 0.5 (the script's own PASS line), at the fresh seed.
  Adopted only if "promising" AND:
      the exhaustiveness-16 re-dock gives >= 0.60 with the interval above 0.5;  the conditioning lift is still >= 1.3;  median similarity to the seed is <= 0.6 (not copying seeds).
  Reading rule: the pre-RL held-out model scored 0.55 and 0.60 at two seeds, so a single sample is read against that spread, not against 0.5.

STOP RULES (the time box)
  * At most THREE training runs in total (r1, r2, r3).  r2 and r3 change ONE thing each, chosen from the previous run's log
    (training reward flat -> raise --lr 3x; validity or similarity to seed collapsing -> raise --kl 3x and lower --lr; reward still rising at the limit -> more --steps).
  * After r1: training curve not alive AND no change in step-4 result -> one retry (r2) only.  After r2 shows the same -> STOP RL.
  * A run that is alive in training but not promising at evaluation counts as a failure of that run.
  * Fri 6 Nov 18:00 Adelaide: stop whatever the state is.  Report: the pre-RL numbers (outcome B / borderline) and what RL did or did not change.
  * No new ideas after r3 (no reward redesign, no second receptor, no DPP-4): those are future work.

ADOPTING (only if the criteria above are met)
  The RL checkpoint becomes an OPTIONAL "PfDHFR-tuned" model, reported as tuned for PfDHFR (it has seen PfDHFR through its reward, so it is no longer an unseen-target model).
  The pre-RL held-out model stays the general model.  Then: move rl_pfdhfr.py out of this folder into the project, re-comment it in the project style, add the checkpoint and
  the script to the zip.  Until then nothing in this folder ships.

-----------------------------------------------------------------------
LOG  (what has been run, newest last)
-----------------------------------------------------------------------
Mon 2 Nov, evening: smoke (ok), smoke_real (ok, Vina works in the workers), r1 (default settings).
  r1: stopped by the 75-minute time box after 17 of 60 steps (4.4 min per step; 19 molecules docked per step).  329 docked molecules in total.
  Training reward (mean percentile): first 5 steps 0.675, last 10 steps 0.684; fitted slope 0.0002 +/- 0.0024 per step -> flat.  Criterion "alive" (+0.08) NOT met.
  Validity 69% -> 80%; heavy atoms 21-23.5; KL to the start only 0.034 (the model moved very little; no collapse).
  Speed: ~14 s of wall time per docked molecule with 3 workers, no faster than docking one at a time: the workers all let Vina use every core.
  Fix in this version: --vina-cpu (default automatic = cores // workers).  Training dockings only; evaluation docking is unchanged.
  Step 4 (fresh-sample evaluation) not yet run for r1; run it together with a same-seed control of the model BEFORE RL (see below).

STEP 4 FOR r1, WITH CONTROL  (generated sets only; about 25 min)
  python generate_for_request.py --checkpoint rl_pilot\runs\r1\checkpoint_rl_r1.pt --target T89515 --n 70 --seed 2 --out rl_pilot\runs\r1\gen_malaria_t89515_rl.csv
  python generate_for_request.py --checkpoint checkpoint_exemplar_seed_v3_holdout.pt --target T89515 --n 70 --seed 2 --out rl_pilot\runs\r1\gen_malaria_t89515_pre.csv
  copy dock_cache_v2.jsonl rl_pilot\runs\r1\eval_cache.jsonl
  python dock_generated.py --cases malaria_t89515 --n 70 --gen-pattern rl_pilot/runs/r1/gen_{key}_rl.csv --cache rl_pilot/runs/r1/eval_cache.jsonl --out rl_pilot/runs/r1/dock_results_rl.csv --summary rl_pilot/runs/r1/dock_summary_rl.json
  python dock_generated.py --cases malaria_t89515 --n 70 --gen-pattern rl_pilot/runs/r1/gen_{key}_pre.csv --cache rl_pilot/runs/r1/eval_cache.jsonl --out rl_pilot/runs/r1/dock_results_pre.csv --summary rl_pilot/runs/r1/dock_summary_pre.json

r2 (the registered retry: training reward flat -> lr x3; the time box is a budget, not a learning setting)
  python rl_pilot\rl_pfdhfr.py --tag r2 --lr 1e-4 --max-minutes 120

Tue 3 Nov, morning: STEP 4 for r1 (with same-seed control) and r2 (the registered retry) were run.
  Commands were the registered ones.  Calibration was identical in both docking runs (PfDHFR 46 actives / 138 decoys, size-corrected AUC 0.73, trusted).
  r1 fresh sample (seed 2):  size-corrected AUC 0.71 [0.63-0.79], median percentile 0.75, validity 83%, median similarity to seed 0.22, mean HA 22.6 (decoys 22.0)
  pre-RL control (seed 2):   size-corrected AUC 0.58 [0.48-0.67], median percentile 0.59, validity 69%, median similarity to seed 0.28, mean HA 21.4
  (pre-RL at the other seeds: 0.55 and 0.60.)  r1 is above all three but the intervals overlap; NOT yet independent of the training reward (same decoys, same Vina setting).
  r1 training curve was flat (0.675 -> 0.684, KL 0.034), so a +0.13 evaluation gain from it is surprising: read as "small real effect plus sampling luck" until replicated.
  r2 (lr 1e-4, 120 min box, 49 steps, 2 min/step after the --vina-cpu fix):  training percentile 0.637 -> 0.922, validity 68% -> 98%,
  mean heavy atoms 20.6 -> 30.0 (known actives 21, decoys 22), similarity to seed 0.26 -> 0.12, KL 0.5 and still rising.  Literal "alive" criteria MET.
  Reading: the percentile rose in step with molecule size and the molecules stopped resembling their seeds; the reward's size line has a slope of only -0.029 kcal/mol per atom,
  so it cannot remove Vina's real size advantage.  Suspected size exploit.  NOT adopted; judged by the checks below.

NEXT (order matters; do not start r3 before these are read):
  A. rl_size_check.py on what already exists (seconds, no docking):
     python rl_pilot\rl_size_check.py --molecules rl_pilot\runs\r1\molecules.csv rl_pilot\runs\r2\molecules.csv --dock-results rl_pilot\runs\r1\dock_results_rl.csv rl_pilot\runs\r1\dock_results_pre.csv
  B. r2 on a fresh sample, same seed as the control (about 20 min):
     python generate_for_request.py --checkpoint rl_pilot\runs\r2\checkpoint_rl_r2.pt --target T89515 --n 70 --seed 2 --out rl_pilot\runs\r2\gen_malaria_t89515_rl.csv
     copy rl_pilot\runs\r1\eval_cache.jsonl rl_pilot\runs\r2\eval_cache.jsonl
     python dock_generated.py --cases malaria_t89515 --n 70 --gen-pattern rl_pilot/runs/r2/gen_{key}_rl.csv --cache rl_pilot/runs/r2/eval_cache.jsonl --out rl_pilot/runs/r2/dock_results_rl.csv --summary rl_pilot/runs/r2/dock_summary_rl.json
     python rl_pilot\rl_size_check.py --dock-results rl_pilot\runs\r2\dock_results_rl.csv
  C. Replicate r1 on a second fresh seed, with its control (about 25 min):
     python generate_for_request.py --checkpoint rl_pilot\runs\r1\checkpoint_rl_r1.pt --target T89515 --n 70 --seed 3 --out rl_pilot\runs\r1\gen_malaria_t89515_rl3.csv
     python generate_for_request.py --checkpoint checkpoint_exemplar_seed_v3_holdout.pt --target T89515 --n 70 --seed 3 --out rl_pilot\runs\r1\gen_malaria_t89515_pre3.csv
     python dock_generated.py --cases malaria_t89515 --n 70 --gen-pattern rl_pilot/runs/r1/gen_{key}_rl3.csv --cache rl_pilot/runs/r1/eval_cache.jsonl --out rl_pilot/runs/r1/dock_results_rl3.csv --summary rl_pilot/runs/r1/dock_summary_rl3.json
     python dock_generated.py --cases malaria_t89515 --n 70 --gen-pattern rl_pilot/runs/r1/gen_{key}_pre3.csv --cache rl_pilot/runs/r1/eval_cache.jsonl --out rl_pilot/runs/r1/dock_results_pre3.csv --summary rl_pilot/runs/r1/dock_summary_pre3.json
  D. The registered independent checks (exhaustiveness 16, about 60-75 min for the three sets; one shared cache so the calibration is docked once):
     python dock_generated.py --cases malaria_t89515 --n 70 --exhaustiveness 16 --noise-check 0 --gen-pattern rl_pilot/runs/r1/gen_{key}_rl.csv --cache rl_pilot/runs/r1/eval_cache_ex16.jsonl --out rl_pilot/runs/r1/dock_results_rl_ex16.csv --summary rl_pilot/runs/r1/dock_summary_rl_ex16.json
     python dock_generated.py --cases malaria_t89515 --n 70 --exhaustiveness 16 --noise-check 0 --gen-pattern rl_pilot/runs/r1/gen_{key}_pre.csv --cache rl_pilot/runs/r1/eval_cache_ex16.jsonl --out rl_pilot/runs/r1/dock_results_pre_ex16.csv --summary rl_pilot/runs/r1/dock_summary_pre_ex16.json
     python dock_generated.py --cases malaria_t89515 --n 70 --exhaustiveness 16 --noise-check 0 --gen-pattern rl_pilot/runs/r2/gen_{key}_rl.csv --cache rl_pilot/runs/r1/eval_cache_ex16.jsonl --out rl_pilot/runs/r2/dock_results_rl_ex16.csv --summary rl_pilot/runs/r2/dock_summary_rl_ex16.json
  E. Conditioning (RL must not break target conditioning; before RL: lift 1.43; needs >= 1.3):
     python check_exemplar_conditioning.py --checkpoint rl_pilot\runs\r1\checkpoint_rl_r1.pt --temperature 0.7 --max-targets 17 --k 1
     python check_exemplar_conditioning.py --checkpoint rl_pilot\runs\r2\checkpoint_rl_r2.pt --temperature 0.7 --max-targets 17 --k 1
  r3 (only if still warranted after A-E; ONE registered change): if r2 is a size exploit, the registered response is "similarity to seed collapsing -> KL x3, lower lr":
     python rl_pilot\rl_pfdhfr.py --tag r3 --lr 5e-5 --kl 0.15 --max-minutes 90
  Hard stop Fri 6 Nov 18:00 Adelaide unchanged.

Tue 3 Nov, ~08:00: STEP A and STEP B results (rl_size_check.py; r2 fresh sample seed 2)
  r1 training: no size drift (HA 22.7 -> 22.3, LE 0.383 -> 0.389, percentile 0.685 -> 0.684).  r1 is effectively the untouched model with higher validity (69% -> 83%).
  r1 evaluation vs same-seed control, size-robust views:  size-corrected 0.71 vs 0.58 (control in-range only: 0.60);  local percentile 0.70 vs 0.60;
     AUC inside 18-26 heavy atoms 0.61 vs 0.58.  The gain shrinks to about 0.03-0.10 and sits mostly in the <= 20 heavy-atom molecules (n 22).  Not established; seed-3 replication (C) decides.
  r2 fresh sample: size-corrected 0.94 [0.90-0.97] (the known actives, the ceiling, score 0.73), local percentile 0.94, band 18-26 AUC 0.91, mean HA 27.4, LE 0.369,
     mean score -9.81 vs actives -8.50;  median similarity to seed 0.11, 0% in the analog range 0.40-0.80;  0 of 46 contain the pharmacophore (r1 8 of 47, pre-RL 12 of 45).
  READ: r2 beats the known binders at every size band (also at 21-24 heavy atoms: -9.76 vs decoys -8.17), contains none of the one feature the calibration validates, and no
     longer resembles its seeds.  That is exploitation of the Vina score, not a better binder generator.  r2 is REJECTED (a judgement on integrity grounds; the literal
     pre-registered step-4 line, >= 0.65 with CI above 0.5, was met, which shows that the line alone cannot tell a hacked set from a good one).
  Gap in the pre-registered criteria (my omission): no check that molecules stay in the domain where Vina was validated (pharmacophore present, analogs of their seeds).

REVISED NEXT (replaces D and E above):
  C (seed-3 replication of r1 with its control) is the only run still to do.  D and E are for r1 and the control only, and only if C replicates (r1 above control at seed 3 on the
  size-robust views).  D and E are DROPPED for r2.  If C does not replicate: stop RL, report "RL either did not move the model (r1) or exploited the scoring function (r2)".
  Optional r3 (one run, 90 min, only if the user wants it; registered here before running):  python rl_pilot\rl_pfdhfr.py --tag r3 --lr 5e-5 --kl 0.15 --max-minutes 90

Tue 3 Nov, evening: REWARD REDESIGN (requested by the user; replaces the "no reward redesign" rule for ONE final run)
  Why: r2 showed that the original reward (decoy percentile of ANY molecule) is satisfied by molecules that Vina likes but that are not analogs of their seed and lack the pharmacophore.
  The optimiser did what it was told; the reward was under-specified.  The calibration only validated Vina on pharmacophore-bearing antifolates, so outside that domain the score is not trusted.
  Receptor check (PDB 1J3I = WILD-TYPE PfDHFR-TS with WR99210, NADPH and dUMP, 2.33 A): the receptor is the right one for generic antifolates.
  Literature (other groups): RL on docking rewards is known to be exploited (arXiv 2310.02902: long C/N chains, scores below -20, mitigated but not removed by multiplying in QED and SA);
  docking scores favour generic molecules that score well in many pockets (arXiv 2311.12035: random ZINC molecules beat the ground-truth ligand at 22% of targets; their Delta Score subtracts the
  score at other pockets).  Both support gating the reward and checking target specificity.
  New option:  --reward gated  (default stays the original reward, so r1/r2 remain reproducible).  Reward = the SAME decoy percentile, but only for a molecule that
     (1) carries the antifolate pharmacophore, (2) has similarity >= 0.40 to its seed (the lower edge of the analog range), (3) is within 4 heavy atoms of its seed.
     Everything else gets -0.5 and is not docked.  --probe-steps N measures how many molecules pass the gate with no training and no docking.
  r3 = r2's optimiser settings (lr 1e-4, KL 0.05) with the gated reward and batch 64 (more samples per step because few pass the gate).

PRE-REGISTERED CRITERIA FOR r3 (written before the run; not to be changed afterwards)
  Evaluation = 70 fresh molecules at seed 2 AND seed 3, each against the pre-RL control at the same seed, same scripts as before (dock_generated.py + rl_size_check.py).
  r3 is "promising" only if ALL hold at BOTH seeds:
    1. Domain: median similarity to seed >= 0.30; share with the pharmacophore >= 40% (pre-RL about 27%); mean heavy atoms <= 25.5 (decoys 22, seeds about 22).
    2. Docking: size-corrected AUC >= 0.65 with the interval above 0.5 AND at least 0.08 above the same-seed control; band-only AUC (18-26 heavy atoms) >= 0.60.
    3. Paired: analogs better than their seed >= 50% of the time (pre-RL 33-36%).
    A result above the known-binder ceiling (0.73) by more than 0.10 is FLAGGED as suspect, not celebrated.
  Adopted only if promising AND: exhaustiveness-16 re-dock >= 0.60 (interval above 0.5) AND conditioning lift >= 1.3.
  If r3 is not promising: STOP RL, report "an unconstrained Vina reward is exploited (r2); a domain-gated reward (r3) did / did not improve docking of analogs".  No further runs.
  Hard stop unchanged: Fri 6 Nov 18:00 Adelaide.  GUI polish and the portable zip start Sat 7 Nov regardless.

Tue 3 Nov, 18:30: STEP C RESULT (r1 replicated at seed 3, with the same-seed pre-RL control)
  seed 3:  r1 0.63 [0.54-0.72] (in-range 0.64) vs control 0.53 [0.42-0.63] (in-range 0.56);  local percentile 0.67 vs 0.60;  band 18-26 AUC 0.58 vs 0.59;  analogs better than their seed 30% vs 21%.
  seed 2:  r1 0.71 vs control 0.58;  local 0.70 vs 0.60;  band 0.61 vs 0.58;  analogs better than seed 36% vs 33%.
  pre-RL model at four seeds: 0.55, 0.60, 0.58, 0.53 (mean 0.57, stable).  r1 at two seeds: 0.71, 0.63 (mean 0.67).
  READ: r1 is above its control at both seeds on the size-corrected and local measures (+0.07 to +0.13) but NOT in the size-matched band 18-26 (+0.03, -0.01) and not in the 21-24 band;
     the gain sits in molecules of <= 20 and >= 25 heavy atoms (small n).  By the registered rule r1 is NOT promising (0.63 < 0.65 at seed 3).  r1 is closed: no exhaustiveness-16 run, no conditioning run for r1.
  Also: only 9-19% of generated molecules are in the analog range (similarity 0.40-0.80) and 70-80% of analogs dock worse than their seed.  So a gate of similarity >= 0.40 AND pharmacophore may pass
     only a few molecules per batch.  The probe now prints a table of pass counts per batch for other thresholds (similarity 0.25-0.50, heavy-atom window 3-8).

Tue 3 Nov, 18:45: PROBE RESULT and FINAL r3 COMMAND (registered before the run)
  Probe (960 samples from the starting model, batch 64): 3.8% pass the gate (2.4 per batch of 64) at similarity >= 0.40 and +-4 heavy atoms; pharmacophore 19%, similarity ok 16%, size ok 70%.
  Passes barely fall from similarity 0.40 to 0.50 (2.4 -> 2.1 per batch): the gate selects close analogs that keep the pharmacophore.
  Decision: gate unchanged; batch raised from 64 to 128 (about 4.8 passing molecules per step); --steps 150 (steps are cheap); --save-every 25 (earlier checkpoints if the late steps drift).
  r3 command (from the folder that holds config.py):
     python rl_pilot\rl_pfdhfr.py --tag r3 --reward gated --batch 128 --lr 1e-4 --steps 150 --max-minutes 120 --save-every 25
  Training-curve criterion "alive" for r3 (registered now): in-domain share in the last 10 steps >= 2x the probe baseline (7.6%) AND mean percentile of docked molecules rises >= 0.05 from the first 5 steps to the last 10.
  The evaluation criteria and the adoption criteria above are unchanged.

Tue 3 Nov, 23:20: r3 TRAINING RESULT (gated reward; --batch 128 --lr 1e-4 --steps 150 --max-minutes 120 --save-every 25)
  Stopped by the time box at step 55 of 150 (125.6 min): steps slowed from about 1 to 4-6 min as the share of molecules that pass the gate (and so get docked) rose.
  In-domain share 8% (steps 1-3: 5-9%) -> 37-41% (steps 50-55).  Pharmacophore share 31% -> 89-93%.  Validity 72% -> 86%.  KL 0.29 (flat after step 30).
  Mean heavy atoms 20.6 -> 23-24.7 (inside the +-4 window of the seed; decoys 22).  Similarity to seed (docked molecules) 0.60 -> 0.54.
  Mean percentile of docked (in-domain) molecules 0.777 (first 5) -> 0.722 (last 10).  Registered "alive" criterion: in-domain share >= 7.6% MET; percentile +0.05 NOT met (it fell).
  READ: the model learned the gate (stay a pharmacophore-bearing close analog) but not better Vina scores among in-domain molecules; the early in-domain molecules were near-copies of the seeds
  and scored like them (about 0.78), so the share rise dilutes the mean.  Evaluation of the FINAL checkpoint (checkpoint_rl_r3.pt) at seeds 2 and 3 decides, as registered.

Wed 4 Nov, 06:10: r3 EVALUATION (final checkpoint, fresh seeds 2 and 3, 70 molecules each, same-seed pre-RL controls)
                                   seed 2      control 2   seed 3      control 3   registered criterion
  size-corrected AUC               0.70        0.58        0.69        0.53        >= 0.65, CI above 0.5, and >= control + 0.08   MET (+0.12, +0.16)
  AUC inside 18-26 heavy atoms     0.70        0.58        0.65        0.59        >= 0.60                                         MET
  analogs better than their seed   57%         33%         52%         21%         >= 50%                                          MET (seed 3 by 2 points)
  median similarity to seed        0.49        0.28        0.51        0.22        >= 0.30                                         MET
  share with the pharmacophore     96%         27%         94%         23%         >= 40%                                          MET
  mean heavy atoms                 25.4        21.4        24.3        20.7        <= 25.5                                         MET (seed 2 by 0.1)
  analog range 0.40-0.80           79%         17%         81%         11%
  mean LE                          0.349       0.402       0.357       0.404       (actives 0.395, decoys 0.359)
  Not above the known-binder ceiling by > 0.10 (ceiling 0.73): no flag.
  READ: r3 meets every registered 'promising' criterion at both seeds.  Cautions: (1) the calibration rewards the pharmacophore (all actives have it, all decoys lack it) and r3 keeps it in about 95%
  of its molecules, so much of the gain is 'kept the pharmacophore'; (2) r3 molecules are 2-3 heavy atoms bigger than decoys/actives with lower ligand efficiency; (3) evaluation uses the same decoys,
  Vina setting and seed pool as the reward, so it is not independent; (4) the model is PfDHFR-tuned, no longer an unseen-target model.
  STILL REQUIRED FOR ADOPTION: exhaustiveness-16 re-dock >= 0.60 (interval above 0.5) and conditioning lift >= 1.3.

=====================================================================================================
4 NOV 2026 - r3 ADOPTION CHECKS (exhaustiveness 16, conditioning) AND DECISION
=====================================================================================================
All commands were run as registered (shared cache rl_pilot/runs/r3/eval_cache_ex16.jsonl; cache counts 0 -> 238 -> 292 -> 338 add up).
Calibration re-docked at exhaustiveness 16: unchanged (size-corrected AUC 0.73, TRUSTED).

Size-corrected AUC at exhaustiveness 16 (default effort in brackets):
  r3 seed 2        0.70 [0.62-0.77]   in-range only 0.72 [0.64-0.79]     (0.70)
  r3 seed 3        0.69 [0.61-0.77]   in-range only 0.71 [0.63-0.78]     (0.69)
  control seed 2   0.57 [0.48-0.67]   PARTIAL                            (0.58)
  control seed 3   0.52 [0.42-0.62]   FAIL (this is the pre-RL model; expected)  (0.53)
Paired analog - seed (positive = analog scores worse than its seed):
  r3 +0.10 / +0.10 kcal/mol, analogs better than seed 58% / 50%;  controls +0.57 / +0.75, 33% / 21%.
Docking failures (no pose, excluded, not penalised): r3 0 and 1; controls 1 and 3.

Conditioning check at 17 unseen targets (k=1, T=0.7):
  target lift 1.60 [PASS], novel-only 1.56;  T89515 itself 2.25 (it was RL-trained, so that cell is no longer 'unseen');
  mean of the other 16 targets 1.56 (before RL, same check: about 1.42).  Disease lift 1.18 [PARTIAL] (before 1.11).
  Validity 64% [FAIL by the script's 0.65 cut; before RL 71%], unconditional validity 70%.  Exact copies of exemplars 5%.

REGISTERED ADOPTION GATES: (1) ex16 >= 0.60 with interval above 0.5: MET at both seeds (lower bounds 0.62 / 0.61).
(2) conditioning lift >= 1.3: MET (1.60; 1.56 without the tuned target).
DECISION: adopt r3 as an OPTIONAL, labelled 'PfDHFR-tuned' model. The default model stays checkpoint_exemplar_seed_v3_holdout.pt.

CLAIM WORDING (use this, not more):
  'With a reward gated on the antifolate pharmacophore, similarity to the seed and size, RL fine-tuning made the model write close analogs of known
   PfDHFR binders that keep the pharmacophore and dock like their seeds (size-corrected AUC about 0.70 vs 0.52-0.57 before; 50-58% of analogs
   score better than their seed vs 21-33%).  An unconstrained Vina reward was exploited (r2: 0.94, 0/46 molecules with the pharmacophore).'
CAVEATS TO STATE WITH IT: (a) most of the gain is the pharmacophore the calibration rewards; r3 is not shown to beat its seeds;
  (b) molecules are 2-4 heavy atoms larger with lower ligand efficiency (0.35 vs 0.40 for known binders);
  (c) the evaluation is not independent of the reward (same decoys, Vina setting and seed pool); no experimental binding data;
  (d) validity at unseen targets fell 71% -> 64% and the model is tuned to one target;
  (e) the script's roadmap label 'RL not needed' printed for r3 is mechanical and must not be quoted.
RL IS CLOSED: no further runs.  Remaining RL work is packaging only (see the project plan).
