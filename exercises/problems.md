# Original practice problems

[Home](../README.md) · [Worked example](worked-example.md) · [Solutions](solutions.md)

These are original practice exercises, not supplied course homework or exam questions. Try the task, use the hint only if needed, then compare with the solution. “Exit evidence” means a derivation you can reproduce, not merely a read checkmark.

## E01 · Risk decomposition — foundation

Prove that the conditional mean minimizes squared loss. Identify the irreducible term. **Hint:** add and subtract $m(X)$ and condition on $X$.

## E02 · Why the factor two? — proof

Derive the three-error excess-risk bound without assuming a minimizer in $\mathcal F$ exists. **Hint:** compare against arbitrary $f\in\mathcal F$ before taking an infimum.

## E03 · Ridge normalization — calculation

Derive the estimator for $\|y-X\theta\|^2+\lambda\|\theta\|^2$ and compare it with the normalized objective in Chapter 02. **Hint:** differentiate before canceling constants.

## E04 · Penalty selection — explanation

Show that ridge training residual is nondecreasing in $\lambda$. Explain why that does not imply test error is nondecreasing. **Hint:** use the singular-vector residual decomposition.

## E05 · Minimum norm — linear algebra

Let $X=(1\;1)$ and $y=2$. Find all interpolators, the minimum-norm one, and the limit of stable GD initialized at $(3,-1)$. **Hint:** separate the row-space and null-space components.

## E06 · One useful feature — calculation

In the Gaussian double-descent model, take $n=12,d=3,\tau_d^2=8$. What signal energy $b_4^2$ makes the next step decrease risk? What happens if $b_4=0$? **Hint:** compare the finite adjacent risks.

## E07 · Stability from scratch — calculation

For $\Phi(x)=5(x-2)^2/2$, derive the exact iterates, stable interval, and objective gap. **Hint:** use $e_t=x_t-2$.

## E08 · Optimized step — calculation

For $H=\operatorname{diag}(2,8)$, compare steps $1/8$ and $2/(2+8)$. Find each worst parameter contraction factor. **Hint:** inspect both endpoint eigenvalues.

## E09 · Separate geometric assumptions — counterexample

Give examples of a smooth nonconvex function and a convex function with a unique minimizer that is not strongly convex. Explain the failure, not just the names. **Hint:** use $-x^2$ and a fourth power.

## E10 · Strong convexity to gradient domination — proof

Prove $\|\nabla\Phi(\theta)\|^2\ge2\mu\Delta(\theta)$ by minimizing a quadratic lower model. **Hint:** complete the square in $v-\theta$.

## E11 · Convex GD — proof

Derive $\Delta_T\le\|e_0\|^2/(2\eta T)$ for a convex smooth objective and $\eta\le1/\beta$. Mark the exact line that uses monotonic objective values. **Hint:** expand a squared distance, then telescope.

## E12 · Conditional expectation — probability

Suppose $g_t=\nabla\Phi(\theta_t)+\xi_t$ and $\mathbb E[\xi_t\mid\mathcal H_t]=0$. Derive the conditional second-moment identity without claiming $\theta_t$ is independent of past noise. **Hint:** the current gradient is measurable given the history.

## E13 · An exact noise floor — calculation

For $x_{t+1}=(1-\eta h)x_t-\eta\xi_t$ with iid mean-zero noise of variance $s^2$, compute the stationary second moment and objective expectation. **Hint:** solve a scalar affine recurrence.

## E14 · Decreasing-step induction — proof

If $u_{t+1}\le(1-2/(t+\tau))u_t+C/(t+\tau)^2$ with $\tau\ge2$, prove $u_t\le A/(t+\tau-1)$ for $A=\tau u_1+C$. **Hint:** set $s=t+\tau$.

## E15 · Weighted convex SGD — proof

Derive the weighted-average bound from a distance expansion and Jensen. Optimize the fixed-step bound over $\eta$. **Hint:** write $S_T=\sum\eta_t$ and $Q_T=\sum\eta_t^2$.

## E16 · Scalar SGD CLT — calculation

Solve $2(ah-1/2)V=a^2s^2$, state the domain, and find the optimal scalar gain. **Hint:** differentiate the rational function; inspect the boundary.

## E17 · Heavy-ball — roots

For curvature endpoints $1$ and $9$, calculate optimal heavy-ball parameters and both endpoint roots. Explain why $tq^t$ can appear. **Hint:** the discriminant vanishes at the endpoints.

## E18 · Adam bias correction — calculation and qualification

Let every gradient equal $g=(2,-4)$ and initialize moments at zero. Find corrected moments and the first update. Does the same correction make the estimate unbiased for a changing current gradient? **Hint:** sum geometric weights.

## E19 · Modes of convergence — counterexample

Construct a sequence that converges almost surely to zero but has constant second moment. **Hint:** use a rare event of probability $1/n$ and amplitude $\sqrt n$ on a single uniform random variable.

## E20 · Delta method — transfer

If $\sqrt n(T_n-m)\Rightarrow N(0,s^2)$ with $m>0$, derive the limit for $\sqrt n(\log T_n-\log m)$. Then consider $m=0$ and $g(x)=x^2$. **Hint:** first check whether the derivative vanishes.

## A mixed review

Choose E05, E11, E12, and E16. Before solving, label the target as parameter selection, deterministic objective rate, conditional identity, or asymptotic distribution. This classification is part of the task. Afterward, write one changed assumption under which your proof would fail.
