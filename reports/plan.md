# Plan and status

This file tracks the project plan and its status. Results live in [summary.md](summary.md), every change to the preregistered plan in [deviations.md](deviations.md), and the preregistration itself in [preregistration.md](preregistration.md).

## Stages (order of work)

| Stage | Content | Status |
|---|---|---|
| 1 | Literature catalog, preregistration, core code, README | done (2026-10-31) |
| 2 | First model `main_s0`, with the controls and baselines that need no training; paper, reports, site, Hugging Face | done (2026-10-31); fact-checks 2026-11-01 and 2026-11-04 |
| 2a | Independent soundness audit of stage 2 and its fixes (below) | **in progress** (2026-11-14) |
| 3 | Trained ablations A0, A1, A2, A4, A6 (`bash scripts/sweep.sh ablations`), including H5a | postponed by the author |
| 4 | Seeds 1–2 and hyperparameter sensitivity (`bash scripts/sweep.sh sensitivity`); hierarchical bootstrap over seeds | postponed by the author |
| 5 | OOD synthetic set (4D flat torus, two-frequency Fourier circles, multi-period number helix) | postponed |
| 6 | Next models suggested by the failures: coverage check, Fourier-curve family, more cluster centroids, helix test against circles and clusters, abstention | not started |

## Stage 2a: audit of 2026-11-14

**Question.** Is any methodological error invalidating the findings of `main_s0`?

**Findings.** None flips a preregistered verdict. Details are in [deviations.md](deviations.md), section "Found in the independent audit of 2026-11-14".
1. Geometric AIC, as implemented, can never select the blob (B1, A5-G-AIC). This inflates the H1 effect size and the share of the A5 drop attributed to model selection.
2. The test clouds are clustered: chunks of 25 share N and the R^D-embedding draw. The i.i.d. CIs were too narrow.
3. The "unambiguous" subset is defined by B1's own fits, which favours B1.
4. H3 had no trivial baseline. A constant "curve" prediction does at least as well as the method. Fine-level agreement was never stated.
5. H4's "supported" depends on averaging over datasets; the per-dataset criterion fails for the smallest clouds.

**Fixes.**
- [x] Cluster bootstrap for every synthetic CI. Wilcoxon on per-chunk mean differences.
- [x] Post-hoc outputs: blob recall, non-blob H1/H2, H3 constant baseline and fine agreement, H4 per-dataset count.
- [x] B1-fixed implemented with tests: per-cloud cluster candidates, exact point-to-shape J, and a blob null test with τ fitted on `val`. Smoke-tested.
- [x] Deviations written, including the B1-fixed pre-specification (before the test run).
- [ ] Push the code and pre-specification before the long runs.
- [ ] `run.py tune-b1-fixed` on 780 `val` clouds (about 30–40 min GPU).
- [ ] `run.py eval-b1-fixed` on the 3,120-cloud baseline subset (about 2 h GPU, resumable per cell).
- [ ] Generators: headline lines, H4 verdict text, unambiguous-table caption, LaTeX macros, error bars on the by-N and by-noise figures.
- [ ] Text: README, summary, exp01, exp02, methodology, paper. Covers the "strongest" wording, the A5 attribution, "Successes" → "Coarse agreements", the B1-conditioned unambiguous table, the cluster bootstrap and the limitations.
- [ ] Regenerate the report, site and paper PDF. Publish the Hugging Face card. PR and merge.

## Open to-do after stage 2a
- Re-run the real-data B1 column with B1-fixed (currently the preregistered B1).
- Recompute the ambiguity tag with B1-fixed fits, or with a ground-truth-based definition.
- Stages 3–6 above.
