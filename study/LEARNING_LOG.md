# Learning ledger

This public record tracks the mathematical route and evidence of learning. It does not reproduce private dialogue or personal questions. Snapshot: 2026-10-26.

## What the status labels mean

- **Reading covered:** material has been worked through in the documented reading sequence. This does not certify independent problem-solving.
- **Companion prepared:** a self-contained chapter is available for future study; reading completion is not claimed.
- **Evidence pending:** the learner has not yet recorded an independently produced derivation, solution, or experiment interpretation.

## Consolidated study

| Material | Documented coverage | Public notes | Independent evidence |
|---|---|---|---|
| Chapter 1 | Risk, ERM, ridge, training/test error and double descent | [Chinese 01–02](../notes/zh/README.md), including inverse-Gram and projection calculations | Reconstruction without the notes remains to be recorded |
| Chapter 2, GD | Spectral formula, descent lemma, convex and strongly convex rates | [Chinese 03](../notes/zh/03-gradient-descent.md) | Independent solutions remain to be recorded |
| Chapter 2, SGD | Minibatch covariance, finite-time averaging, CLTs, batch means and self-normalization | [Chinese 04–06](../notes/zh/README.md) | Derivations are supplied resources, not evidence of independent mastery |
| Chapter 2, optimizers | Momentum, adaptive updates, AdamW and Muon | [Chinese 09](../notes/zh/09-momentum-and-adaptation.md) | Distinguish quadratic proofs from general algorithm claims |
| Chapter 3 | Concentration, covering, VC dimension and Rademacher bounds | [Chinese 07–08](../notes/zh/README.md) | Independent proof reconstruction remains to be recorded |
| Homework 4–5 | Complete worked solutions prepared from the mathematical study | [HW4](../exercises/zh/homework04.md), [HW5](../exercises/zh/homework05.md) | Submission, grading and independent completion are not claimed |
| Probability appendix | Existing English reference | [Probability toolkit](../notes/10-probability.md) | Separate reference, not a new reading-completion claim |

The 2026-10-26 update covers the mathematical discussion through Chapter 3 of the later source. Chapter 4 and reinforcement-learning topics have not been added. No personal evaluations or dialogue transcripts are part of this ledger.

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
