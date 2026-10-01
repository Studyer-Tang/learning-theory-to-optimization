# Qualifications that matter

[Home](../README.md) · [Source crosswalk](../reference/SECTION_MAP.md)

These notes distinguish explanatory corrections from extensions. They are not a claim of independent peer review of the entire source.

1. **High probability is not all-or-nothing.** Source Remark 2.18 suggests stronger tails are needed for a high-probability guarantee. A nonnegative expected-gap bound already implies the weak Markov bound $\Pr(\Delta>B/\delta)\le\delta$. Stronger tail control is needed for sharper confidence dependence, not for the existence of any probability statement.
2. **Bounded full gradients versus bounded noise.** Jensen gives $\|\nabla\Phi(\theta)\|^2\le\mathbb E\|g(\theta)\|^2$. A global uniform full-gradient second-moment bound on all of $\mathbb R^d$ is incompatible with global strong convexity. Read the source's moment assumption along a controlled trajectory or bounded region; noise-only bounds avoid this specific conflict.
3. **A proof sentence after (2.4.9).** Passing to (2.4.10) only needs $1-(1-\mu\eta)^{T-1}\le1$. The wording about dropping “two factors” is unnecessary.
4. **Objective convergence and parameter convergence.** The source's convex-only warning concerns what follows from an objective bound alone. It should not be read as a general claim that finite-dimensional smooth convex GD iterates cannot converge to some optimizer.
5. **Finite-time and asymptotic averaging use different schedules.** The finite-time $1/T$ result in §2.4 uses a $1/t$-scale schedule. The covariance-efficient PR CLT in §2.5 uses $t^{-\alpha}$ with $1/2<\alpha<1$. Do not equate those assumptions.
6. **Local CLTs need probabilistic control.** Stability, mean-square bounds, localization, and uniform integrability/Lindeberg conditions must justify negligible remainders and averaged conditional covariance limits. Pointwise convergence in probability alone does not license arbitrary Cesàro or expectation manipulations.
7. **Heavy-ball root factors are asymptotic.** Repeated roots can yield $tq^t$. The source correctly states a root-rate claim; a simplified pure-geometric finite-time bound could be false at the optimal endpoints.
8. **Noise floors are upper-bound scales in general.** The scalar quadratic lab has an exact stationary calculation; an upper bound for a general SGD problem does not establish a matching positive lower bound.
9. **Critical dimensions cannot be interpolated numerically into a theorem.** Finite inverse matrices on individual Gaussian samples do not imply a finite inverse expectation. The double-descent plot excludes dimensions where the expected test MSE is infinite with positive effective noise.
10. **Source numbering.** The supplied document goes from §2.7 to §2.9. The companion covers the actual content without manufacturing an absent section.

The original source is preserved unchanged so these annotations remain distinguishable from the author's text.
