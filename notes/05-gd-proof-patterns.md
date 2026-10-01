# 05 · Two reusable proofs for gradient descent

[Previous](04-geometry-and-gd.md) · [Home](../README.md) · [Next](06-sgd.md)

**Goal:** recognize when to contract a gap and when to telescope a distance. Source connection: §§2.2–2.3. A broader collection of proof patterns is in [Garrigos and Gower](https://arxiv.org/abs/2301.11235).

## Pattern A: strong convexity turns descent into contraction

The descent lemma provides progress proportional to $\|\nabla\Phi(\theta)\|^2$. We need to relate that gradient to the error we care about. Strong convexity says

$$\Phi(v)\ge\Phi(\theta)+g^\top(v-\theta)+\frac\mu2\|v-\theta\|^2.$$

Complete the square in $v-\theta$. The quadratic lower model has minimum $\Phi(\theta)-\|g\|^2/(2\mu)$, so

$$\|g\|^2\ge2\mu[\Phi(\theta)-\Phi(\theta^\star)].$$

This is gradient domination, also called a Polyak–Łojasiewicz inequality. Strong convexity implies it; the converse is not needed and is not generally true.

For $0<\eta\le1/\beta$, combine this with descent:

$$\Delta_{t+1}\le\Delta_t-\frac\eta2\|g_t\|^2
\le(1-\mu\eta)\Delta_t.$$

Iteration gives $\Delta_T\le(1-\mu\eta)^T\Delta_0$. At $\eta=1/\beta$, the factor is $(1-1/\kappa)^T\le e^{-T/\kappa}$. To reduce the original gap by a factor $\epsilon$, it suffices to take $T$ on the scale of $\kappa\log(1/\epsilon)$.

“Linear convergence” names this geometric reduction, not a straight-line decrease in error. This general proof is less sharp than exact quadratic spectral analysis: it gives exponent $T$ where that special calculation can give a squared factor with exponent $2T$.

## Convert an objective claim into a parameter claim

At an unconstrained differentiable optimum, $\nabla\Phi(\theta^\star)=0$. Strong convexity and smoothness then sandwich the objective gap:

$$\frac\mu2\|\theta-\theta^\star\|^2
\le\Delta(\theta)
\le\frac\beta2\|\theta-\theta^\star\|^2.$$

Thus $\|e_T\|^2\le2\Delta_T/\mu$ and

$$\|e_T\|\le\sqrt{\beta/\mu}(1-\mu\eta)^{T/2}\|e_0\|.$$

The square root appears because the inequality controlled squared distance. If a problem asks for parameter accuracy $r$, first target a gap of at most $\mu r^2/2$, not $r$.

## Pattern B: convexity gives a telescoping potential

Without strong convexity there is no uniform positive $\mu$ to insert above. Start instead with distance to an attained minimizer, and write $g_t=\nabla\Phi(\theta_t)$:

$$\|e_{t+1}\|^2=\|e_t\|^2-2\eta g_t^\top e_t+\eta^2\|g_t\|^2.$$

Rearrange:

$$g_t^\top e_t=\frac{\|e_t\|^2-\|e_{t+1}\|^2}{2\eta}
+\frac\eta2\|g_t\|^2.$$

Convexity bounds $\Delta_t\le g_t^\top e_t$. Descent bounds $\frac\eta2\|g_t\|^2\le\Phi(\theta_t)-\Phi(\theta_{t+1})$. Combining them cancels the current objective value and leaves

$$\Delta_{t+1}\le\frac{\|e_t\|^2-\|e_{t+1}\|^2}{2\eta}.$$

Sum from zero to $T-1$. All intermediate squared distances cancel. Since the objective values are nonincreasing, the final gap is at most every earlier gap:

$$T\Delta_T\le\sum_{t=0}^{T-1}\Delta_{t+1}
\le\frac{\|e_0\|^2}{2\eta},\qquad
\Delta_T\le\frac{\|e_0\|^2}{2\eta T}.$$

Notice exactly where deterministic descent is used. In SGD a realized objective may increase, so the last-gap step is not automatically available.

## Variable learning rates

The same calculation before division gives
$2\eta_t\Delta_{t+1}\le\|e_t\|^2-\|e_{t+1}\|^2$. With $0<\eta_t\le1/\beta$,

$$\Delta_T\le\frac{\|e_0\|^2}{2\sum_{t<T}\eta_t}.$$

The largest permitted fixed step maximizes this denominator. For $\eta_t=c(t+1)^{-\alpha}$, integration of the decreasing function $x^{-\alpha}$ gives

$$\sum_{t<T}\eta_t\ge
\begin{cases}
c[(T+1)^{1-\alpha}-1]/(1-\alpha),&0<\alpha<1,\\
c\log(T+1),&\alpha=1.
\end{cases}$$

Consequently this proof gives $O(T^{-(1-\alpha)})$ or $O(1/\log T)$, respectively. These are guarantees from one argument, not statements that every function attains the worst case. Under strong convexity, multiply the separate contraction factors to obtain
$\Delta_T\le\Delta_0\exp(-\mu\sum_{t<T}\eta_t)$.

## What the convex result does not say

For $\Phi(x,y)=x^2$, every $(0,y)$ is optimal. A small objective gap cannot bound distance to a preselected $(0,0)$ uniformly over all $y$. This does not assert that GD iterates never converge: under standard finite-dimensional smooth convex assumptions they can converge to an optimizer. It says the objective bound alone does not provide a universal distance rate to a designated one.

**Checkpoint:** hide this page and write the three lines that connect convexity, the distance identity, and descent. Attempt E10–E11. If the proof fails, identify the missing inequality rather than rereading all theorem statements.
