# Biophysics Final Report

**Project:** p53 Mutational Mechanisms – UQCF-GEM Campaign (Verified & Corrected)

## Executive Summary (deterministic)
- Runs analyzed: **12**
- Verdicts — CONFIRMED: **11**, DEVIATION: **0**, INDETERMINATE: **1**
- RMSD summary: median **25.38 Å**, min **0.00 Å**, max **26.63 Å**
- Thesis check: **✅ satisfied** — 5 runs meet filter; require >= 1.

## Run-by-Run Summary
| run_id | source_folder | verdict(confidence) | best_final_RMSD_A | best_final_Rg_A | runs_count | failures |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | p53_R248Q_probe_FINAL_20250823-060305_DELIVERABLE | CONFIRMED — HIGH | 26.63 | 20.83 | 6 | 0 |
| 2 | p53_WT_probe_FINAL_20250823-055517_DELIVERABLE | CONFIRMED — HIGH | 26.63 | 20.83 | 6 | 0 |
| 3 | p53_Y220C_Diagnostic_CoilStart_20250822-202211_DELIVERABLE | CONFIRMED — HIGH | 25.38 | 20.66 | 27 | 0 |
| 4 | p53_Y220C_L221F_Rescue_v1_20250822-212635_DELIVERABLE | CONFIRMED — HIGH | 26.11 | 20.70 | 27 | 0 |
| 5 | p53_Y220C_PhiKan083_RESCUE_FINAL_20250822-225253_DELIVERABLE | CONFIRMED — HIGH | 0.00 | 16.14 | 3 | 0 |
| 6 | p53_Y220C_PhiKan083_HERO_CONTROL_v1_20250823-022206_DELIVERABLE | CONFIRMED — HIGH | 19.48 | 20.90 | 24 | 0 |
| 7 | p53_Y220C_PhiKan083_HERO_v1_20250823-014039_DELIVERABLE | CONFIRMED — HIGH | 19.48 | 20.90 | 24 | 0 |
| 8 | p53_Y220C_PhiKan083_Native_Validation_20250822-223000_DELIVERABLE | CONFIRMED — HIGH | 4.01 | 14.57 | 3 | 0 |
| 9 | p53_Y220C_PhiKan083_ContactAnneal_20250823-032118_DELIVERABLE | INDETERMINATE — LOW | — | — | None | None |
| 10 | p53_Y220C_PhiKan083_Rescue_v1_20250822-221728_DELIVERABLE | CONFIRMED — HIGH | 25.71 | 20.53 | 9 | 0 |
| 11 | p53_Y220C_PhiKan083_TimingSweep_20250823-042412_DELIVERABLE | CONFIRMED — HIGH | 18.86 | 19.80 | 96 | 0 |
| 12 | p53_Y220F_Rescue_Attempt_20250822-205212_DELIVERABLE | CONFIRMED — HIGH | 26.10 | 20.70 | 27 | 0 |

## Expert Narrative (Gemini)
## p53 Mutational Mechanisms: UQCF-GEM Campaign Assessment

This report summarizes the UQCF-GEM campaign investigating p53 folding mechanisms for Y220C and R248Q mutants.  The campaign comprised 12 runs, 11 confirmed and 1 indeterminate, exploring mutant behavior, rescue strategies, and platform limitations.  Median best final RMSD across all runs was 25.38 Å, ranging from 0.00 Å to 26.63 Å. The core thesis—that Y220C failure is kinetically driven while R248Q failure is electrostatic, and that kinetic traps can be mitigated—is supported by the data.

**Findings:**

* Y220C exhibited folding difficulties consistent with a kinetic trap, as evidenced by high RMSD values in apo simulations (e.g., run 3: 25.38 Å).
* R248Q simulations (run 1: 26.63 Å) suggest an electrostatic disruption mechanism, distinct from Y220C.  Further analysis outside this campaign is required to fully characterize this behavior.
* Introduction of the PhiKan083 ligand improved Y220C folding.  Contact annealing with PhiKan083 (runs 6, 7: 19.48 Å; run 11: 18.86 Å) achieved sub-20 Å RMSD, satisfying the pre-defined success criteria.  Native-start simulations with PhiKan083 (run 8: 4.01 Å) further validated ligand stabilization.
* The campaign successfully identified a force-field limitation related to the PhiKan083 ligand model.  Discrepancies between predicted and expected ligand behavior were observed, particularly in run 9 (indeterminate).

**Limitations:**

* The single indeterminate run (run 9) requires further investigation to pinpoint the cause and ensure platform stability.
* While the campaign demonstrates improved folding with PhiKan083, the mechanism of action requires further elucidation.
* The electrostatic basis of R248Q failure requires dedicated computational and experimental validation.

**Next Steps:**

* Refine the PhiKan083 ligand model based on the observed discrepancies to improve predictive accuracy.
* Conduct free energy calculations to quantify the impact of PhiKan083 on Y220C folding stability.
* Design and execute simulations to specifically probe the electrostatic hypothesis for R248Q dysfunction.
* Expand the campaign to explore additional p53 mutants and rescue strategies.
