# Validation record

Local preparation checked on 2026-10-01, using Python 3.12.14, NumPy 2.5.3, and Matplotlib 3.11.2.

| Check | Result | Scope |
|---|---|---|
| Numerical unit tests | 11 passed | Ridge normal equations, minimum-norm geometry, quadratic GD, Gaussian risk domains, scalar SGD and CLT formulas, and equivalent momentum recursions |
| Experiment execution | Five experiments completed | Fixed synthetic-data seeds; measurements and runtime versions recorded in [the report](labs/RESULTS.md) |
| Formula syntax | 806 expressions parsed with KaTeX 0.16.22 | Syntax check of inline and display math, not a mathematical proof or a GitHub rendering test |
| Figure inspection | Six PNGs visually inspected | Axis ranges, legends, labels, and readability; no page screenshots used |
| Repository checks | Passed | Local Markdown file links, matched code/math delimiters, identifying local paths, transcript-role markers, required artifacts, PDF allowlist |
| Original source | Byte-for-byte copy | SHA-256 recorded in [reference/SHA256SUMS](reference/SHA256SUMS) |

The formula checker was a preparation-time tool; it is not a runtime dependency of the labs. The repository check does not validate external links or Markdown fragment anchors. Automated privacy heuristics are supplemented by editorial review of the public learning ledger and documentation.

To repeat the maintained checks, use the commands in [README.md](README.md). This record describes local preparation; hosted check results are recorded separately in the repository's GitHub Actions history. Mathematical exposition has been reviewed during preparation but has not undergone independent peer review.

## Tail-diagnostic revision: 2026-10-02

Local execution used Python 3.13.15, NumPy 2.5.3, and Matplotlib 3.11.2. CI remains configured for Python 3.12. The original five experiments were rerun; their numerical metrics match the previous recorded run. The new diagnostic adds two independent streams (n = 30 / 60), seven dimensions per stream and 1,200 repetitions, plus nested budgets of 40 / 240 / 1,200. This is a synthetic finite-budget diagnostic, not a validation of the infinite-expectation theorem by simulation.

- 14 tests passed: the 11 existing numerical checks, an exact discrete isotropic test-population check of conditional risk, an extreme-draw diagnostic check, and a cross-check of saved raw risks, running means, checkpoints and JSON summaries.
- All 16,800 new least-squares fits had full numerical rank. The saved smallest singular values and ranks expose the solver's finite-precision boundary; this does not rule out rarer rank truncation on future draws.
- The new six-panel running-mean figure was visually inspected for labels, finite-target lines, scales and legibility. Existing figures were unchanged by regeneration.
- Repository checks and `git diff --check` passed. Mean confidence intervals were intentionally not inferred from these heavy-tail samples.
