# 09 · Adam and what a preconditioner actually proves

[Previous](08-momentum.md) · [Home](../README.md) · [Next](10-probability.md)

**Goal:** understand coordinatewise scaling without importing an invalid convergence guarantee. Source connection: §2.7.

## Unpack one update

Adam maintains two exponential averages:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t),$$

with zero initialization, then computes

$$\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\qquad
D_t=\operatorname{diag}\left(\frac1{\sqrt{\widehat v_t}+\epsilon_{\rm adam}}\right),$$

$$\theta_{t+1}=\theta_t-\alpha_tD_t\widehat m_t.$$

Squares, roots, and division are coordinatewise. The $\beta_1,\beta_2$ averaging parameters are not the smoothness constant $\beta$. The small positive $\epsilon_{\rm adam}$ prevents division by zero and affects scaling.

Unroll the first average: $m_t=(1-\beta_1)\sum_{s=1}^t\beta_1^{t-s}g_s$. Its weights sum to $1-\beta_1^t$, so bias correction normalizes the missing weight from zero initialization. If every gradient equals a constant vector $g$, then $\widehat m_t=g$ and $\widehat v_t=g\odot g$ exactly. If gradient means change along the trajectory, this normalization does not make $\widehat m_t$ an unbiased estimate of the current gradient. The update originates in [Kingma and Ba](https://arxiv.org/abs/1412.6980).

## Where the SGD proof stops working

Smoothness still gives a pathwise inequality:

$$\Phi(\theta_{t+1})\le\Phi(\theta_t)
-\alpha_t\langle\nabla\Phi(\theta_t),D_t\widehat m_t\rangle
+\frac{\beta\alpha_t^2}{2}\|D_t\widehat m_t\|^2.$$

For ordinary SGD, conditional unbiasedness simplifies the alignment term. Here the diagonal matrix and momentum estimate both depend on current and past gradients. They cannot be pulled out of conditional expectation as if they were a fixed independent matrix. Strong convexity supplies gradient domination, but does not establish that this adaptive direction is sufficiently aligned with the gradient.

Nonconvergence examples in convex online settings are given by [Reddi, Kale, and Kumar](https://arxiv.org/abs/1904.09237). This is a limitation on broad claims, not a claim that every use of Adam fails. A positive theorem must specify additional assumptions and the exact algorithm variant.

## Prove the frozen-matrix benchmark

Suppose instead that $D$ is fixed, positive definite, diagonal, and $d_-I\preceq D\preceq d_+I$. Use the exact-gradient update $\theta^+=\theta-\alpha Dg$ with $0<\alpha\le1/(\beta d_+)$. Since $D^2\preceq d_+D$,

$$
\begin{aligned}
\Phi(\theta^+)
&\le\Phi(\theta)-\alpha g^\top Dg+\frac{\beta\alpha^2}{2}g^\top D^2g\\
&\le\Phi(\theta)-\frac\alpha2g^\top Dg\\
&\le\Phi(\theta)-\frac{\alpha d_-}{2}\|g\|^2.
\end{aligned}
$$

With $\mu$-strong convexity, this yields $\Delta_{t+1}\le(1-\alpha\mu d_-)\Delta_t$. That is a deterministic preconditioning theorem. It does not prove the same rate for Adam's adaptive matrix and momentum.

For a diagonal quadratic with $H=\operatorname{diag}(1,100)$, choosing $D=H^{-1}$ and $\alpha=1$ solves the noiseless problem in one step. This illustrates the potential of curvature-aware scaling. Adam's running second moments are not in general a Hessian inverse, so the example is a geometric benchmark rather than an explanation of an exact Adam identity.

## What to compare in an experiment

Match initialization, objective, gradient noise, iteration count, and measurement cost. Label a plot “one quadratic example” rather than treating it as a general optimizer ranking. An algorithm may make fast early progress while a theorem gives a conservative bound; conversely, an attractive curve does not verify the assumptions of a global theorem.

**Checkpoint:** calculate Adam's first corrected update for a constant two-dimensional gradient, then explain why a random $D_t$ prevents the simple SGD conditional-expectation step. Attempt E18. The rate sheet intentionally leaves vanilla Adam without a universal rate under strong convexity and smoothness alone.
