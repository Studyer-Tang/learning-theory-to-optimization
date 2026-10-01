# 07 · What remains random near the optimum?

[Previous](06-sgd.md) · [Home](../README.md) · [Next](08-momentum.md)

**Goal:** distinguish a finite-time error bound from a limiting distribution. Source connection: §2.5. Read [the probability toolkit](10-probability.md) first if $\Rightarrow$ or $o_P$ is unfamiliar. This chapter is prepared onward material, not a recorded completed learning milestone.

## An error scale is not yet a distribution

A bound $\mathbb E\|\theta_T-\theta^\star\|^2=O(1/T)$ says how fast an average squared error decreases. It does not identify the shape of the fluctuations or supply a covariance matrix. A central limit theorem studies $\sqrt T(\theta_T-\theta^\star)$, whose scale can remain nonzero as the unscaled error vanishes.

Work locally with

$$e_{t+1}=e_t-\eta_t(He_t+\xi_{t+1}+r_t),\qquad
H=\nabla^2\Phi(\theta^\star)\succ0,\quad\|r_t\|\le C\|e_t\|^2.$$

The required assumptions include convergence to this neighborhood, a local mean-square rate $\mathbb E\|e_t\|^2=O(\eta_t)$, a locally Lipschitz Hessian, martingale-difference noise, convergence of its conditional covariance to a deterministic matrix $S$, and conditional moment/Lindeberg control. Localization must also control excursions. These are prerequisites of the local argument, not consequences of writing a Taylor expansion. In particular, convergence in probability alone does not justify every expectation or averaged-covariance step; integrability control matters.

## Final iterate with a $1/t$ step

Take $\eta_t=a/t$. Provided $2a\lambda_{\min}(H)>1$ and the preceding regularity conditions hold,

$$\sqrt T(\theta_T-\theta^\star)\Rightarrow N(0,V_a),$$

where

$$\left(aH-\frac12I\right)V_a+V_a\left(aH-\frac12I\right)^\top=a^2S.$$

Why a Lyapunov equation? Iterating the linearized recursion gives a weighted sum of past noise. The transition from time $t$ to $T$ behaves like $(t/T)^{aH}$. Its limiting covariance has the integral representation

$$V_a=a^2\int_0^1x^{aH-I}Sx^{aH^\top-I}\,dx.$$

Set $x=e^{-s}$ and $A=aH-I/2$ to obtain $V_a=a^2\int_0^\infty e^{-As}Se^{-A^\top s}ds$. The condition on the minimum eigenvalue makes the integral finite. Differentiating the integrand and evaluating its endpoints yields $AV_a+V_aA^\top=a^2S$.

The initialization term scales as $T^{1/2-a\lambda_{\min}(H)}$ and vanishes under the strict condition. The quadratic remainder is negligible using the local mean-square rate. A martingale triangular-array CLT then treats the remaining weighted noise; this is an external probability theorem, not something established by the covariance calculation alone.

In one dimension,

$$V_a=\frac{a^2\sigma_g^2}{2ah-1},\qquad ah>1/2.$$

Differentiation shows the minimum at $a=1/h$, where $V_a=\sigma_g^2/h^2$. A gain close to the stability threshold can yield a very large covariance. In several dimensions a single scalar gain cannot generally be optimal for all curvatures.

## Why averaging removes leading gain dependence

Use $\eta_t=at^{-\alpha}$ with $1/2<\alpha<1$ and average $\bar\theta_T=T^{-1}\sum_{t=1}^T\theta_t$. Reorganize the recursion and sum:

$$H(\bar\theta_T-\theta^\star)
=-\frac1T\sum_{t=1}^T\xi_{t+1}
-\frac1T\sum_{t=1}^T\frac{e_{t+1}-e_t}{\eta_t}
-\frac1T\sum_{t=1}^Tr_t.$$

Summation by parts makes the middle numerator

$$\frac{e_{T+1}}{\eta_T}-\frac{e_1}{\eta_1}
+\sum_{t=2}^Te_t\left(\frac1{\eta_{t-1}}-\frac1{\eta_t}\right).$$

The mean-square rate bounds its expected norm by $O(T^{\alpha/2})$. After division by $\sqrt T$ it vanishes because $\alpha<1$. The accumulated Taylor remainder is $O_P(T^{1/2-\alpha})$ and vanishes because $\alpha>1/2$. Both endpoints of the interval have a role.

We therefore obtain the asymptotic linear representation

$$\sqrt T(\bar\theta_T-\theta^\star)
=-H^{-1}\frac1{\sqrt T}\sum_{t=1}^T\xi_{t+1}+o_P(1).$$

With conditional covariance averaging and Lindeberg conditions sufficient for the martingale CLT, the limit is

$$\sqrt T(\bar\theta_T-\theta^\star)
\Rightarrow N(0,H^{-1}S(H^{-1})^\top).$$

The covariance no longer depends on $a$ or $\alpha$ within this range. Transient performance still does. This is the averaging principle associated with [Polyak and Juditsky](https://doi.org/10.1137/0330046); it should not be silently transferred to the endpoint $\alpha=1$.

## Online least squares and interpretation

For population least squares, $\widehat g(\theta;X,Y)=(X^\top\theta-Y)X$. With $\varepsilon=Y-X^\top\theta^\star$,

$$H=\mathbb E[XX^\top],\qquad S=\mathbb E[\varepsilon^2XX^\top].$$

If $\mathbb E[\varepsilon^2\mid X]=\sigma_y^2$, then $S=\sigma_y^2H$ and the sandwich covariance becomes $\sigma_y^2H^{-1}$. More generally it retains the heteroskedastic sandwich form.

If a covariance estimator is consistent and the required nondegeneracy holds, asymptotic coordinate intervals use standard errors $\sqrt{\widehat V_{jj}/T}$. They are approximate large-sample statements, not finite-time certificates. Here $T$ counts stochastic-approximation updates. Resampling a fixed training dataset gives algorithmic uncertainty around an empirical optimizer; it does not create additional independent observations or remove statistical uncertainty about the population.

**Checkpoint:** solve the scalar Lyapunov equation, minimize its variance, and identify the two places where the PR exponent restrictions are used. Attempt E16. The [uncertainty lab](../labs/README.md) reports finite-horizon histograms and their discrepancy from the asymptotic reference instead of presenting a simulation as a CLT proof.
