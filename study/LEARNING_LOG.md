# Learning ledger

This public record tracks the mathematical route and evidence of learning. It does not reproduce private dialogue or personal questions. Snapshot: 2026-10-01.

## What the status labels mean

- **Reading covered:** material has been worked through in the documented reading sequence. This does not certify independent problem-solving.
- **Companion prepared:** a self-contained chapter is available for future study; reading completion is not claimed.
- **Evidence pending:** the learner has not yet recorded an independently produced derivation, solution, or experiment interpretation.

## The journey so far

| Stage | Mathematical focus | Reading status | Evidence still to produce |
|---|---|---|---|
| 1 · Establish the foundations | Risk, model classes, validation, ridge | Foundation recap prepared; earlier independent coverage not documented | Derive the three-error bound and ridge normal equation |
| 2 · Understand interpolation | §1.5, minimum norm, Gaussian test risk, critical dimensions | Reading covered | Solve E05–E06 without the solution; interpret the singularity in the risk formula |
| 3 · Move from fits to algorithms | §2.1, quadratic GD, curvature, stable steps | Reading covered | Reconstruct the scalar multiplier and matrix eigencoordinate argument |
| 4 · Learn reusable proof patterns | §§2.2–2.3, contraction and telescoping | Reading covered | Reproduce the convex $1/T$ proof and identify where monotonicity enters |
| 5 · Introduce random updates | §2.4, conditional expectations, noise floors, averaging | Reading covered | Derive the conditional recursion and distinguish final from averaged output |
| 6 · Consolidate through problems | Worked example, original problem set, reproducible labs | Study materials prepared | Attach independent solutions and short lab interpretations |
| 7 · Study uncertainty and other optimizers | §§2.5–2.7, CLTs, momentum, Adam | Companion prepared; onward study | Verify local CLT assumptions and derive the heavy-ball roots |
| 8 · Strengthen probability tools | Appendix A | Companion prepared; onward study | Explain convergence counterexamples and solve the delta-method exercise |

The current reading endpoint is §2.4. The next useful activity is consolidation, not automatically increasing the page count. The repository contains explanations and solutions as learning resources; their existence is not evidence that the learner has completed them.

## Conceptual milestones

**From class size to a fitted procedure.** The route through interpolation separates a class's best possible risk from the risk of the algorithmically selected fit. This distinction motivates minimum-norm selection and validation.

**From a theorem to its mechanism.** Quadratic GD reduces to scalar multipliers. General convex proofs instead organize descent inequalities around a potential. Recognizing the mechanism makes a new exercise approachable even when its notation changes.

**From deterministic progress to random progress.** SGD requires conditioning on the history. The same update now combines contraction with injected noise, making the learning-rate schedule and reported output part of the result.

**From reading to evidence.** Subsequent entries should record what can be reproduced, the assumption that was essential, and the next problem that tests transfer. This is the transition the companion is designed to support.

## Add a future entry

```text
Date:
Topic and chapter:
Status: read / reconstructed / independently solved / experimentally checked
Problem or claim:
Assumptions used:
Derivation or solution file:
Experiment command and result, if relevant:
What the evidence establishes:
What it does not establish:
Next verification task:
```

## Mastery checks — deliberately not pre-completed

- [ ] Explain population versus empirical risk without notation.
- [ ] Derive the normal equation and handle a deficient-rank design.
- [ ] Reconstruct the bias–variance identity using conditional expectation.
- [ ] Explain why an almost-surely finite quantity can have infinite expectation.
- [ ] Derive $0<\eta<2/\beta$ for a positive-definite quadratic.
- [ ] Prove the strongly convex and convex GD objective bounds.
- [ ] Derive the SGD noise recursion and a decreasing-step induction.
- [ ] State the output and randomness in every row of the rate sheet.
- [ ] Distinguish finite-time averaging from the PR covariance CLT.
- [ ] Explain why the frozen-preconditioner theorem is not an Adam theorem.
