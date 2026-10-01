# 03 · Interpolation, minimum norm, and double descent

[Previous](02-capacity-and-ridge.md) · [Home](../README.md) · [Next](04-geometry-and-gd.md)

**Goal:** derive the prediction-error curve of one specified learning procedure, and understand why it is not a universal law. Source connection: §1.5. Related research: [Hastie et al.](https://arxiv.org/abs/1903.08560).

## Fix the data-generating problem first

Let independent standard-normal coordinates generate $Y=\sum_{j\ge1}b_jX_j+\varepsilon$, with $\sum_jb_j^2<\infty$ and independent noise $\varepsilon\sim N(0,\sigma_y^2)$. Keep the sample size $n$ fixed and fit only the first $d$ coordinates. Define

$$a_d=\sum_{j>d}b_j^2,\quad B_d^2=\sum_{j\le d}b_j^2,
\quad\tau_d^2=\sigma_y^2+a_d.$$

The omitted coordinates act as additional independent Gaussian noise. Thus $y=X_d\theta_d+\eta_d$, where $\theta_d=(b_1,\ldots,b_d)^\top$ and $\eta_d\sim N(0,\tau_d^2I_n)$ is independent of $X_d$. This reduction depends on the independent isotropic Gaussian design; correlated features require a different analysis.

## Why the solution changes at $d=n$

Almost surely $X_d$ has rank $\min(n,d)$. Below the threshold,
$\widehat\theta=(X_d^\top X_d)^{-1}X_d^\top y$ is unique. At $d=n$, the square design is invertible. Above it,

$$\widehat\theta=X_d^\top(X_dX_d^\top)^{-1}y$$

is the minimum-norm interpolator. Every interpolator equals $\widehat\theta+v$ for $v\in\ker X_d$. Since $\widehat\theta$ lies in the row space, it is orthogonal to $v$, so
$\|\widehat\theta+v\|^2=\|\widehat\theta\|^2+\|v\|^2$. This proves the minimum-norm claim.

Zero-initialized GD selects this row-space solution with an appropriate stable step. A nonzero initialization can preserve a null-space component, so “GD finds minimum norm” needs an initialization condition.

## Training loss is the easier curve

For $d<n$, the fitted values are the orthogonal projection $H_dy$ onto the design column space. The residual projector has rank $n-d$. Therefore

$$\mathbb E\widehat L_n(\widehat\theta)
=\frac{\tau_d^2}{2}(1-d/n),\qquad d<n.$$

For $d\ge n$, training loss is zero. With nonzero effective noise, $n$ is the first dimension that can fit arbitrary responses. This establishes the training interpolation threshold, not a statement about good test predictions.

## Convert prediction error into coefficient error

For an independent test input, let $\Delta=\widehat\theta-\theta_d$. Isotropy and independence give

$$\mathbb E[(Y'-X_{1:d}'{}^\top\widehat\theta)^2\mid\text{training data}]
=\tau_d^2+\|\Delta\|^2.$$

Below the threshold, $\Delta=(X_d^\top X_d)^{-1}X_d^\top\eta_d$. Taking a trace of its conditional covariance yields

$$\mathbb E[\|\Delta\|^2\mid X_d]
=\tau_d^2\operatorname{tr}((X_d^\top X_d)^{-1}).$$

The inverse-Wishart first moment is
$\mathbb E[(X_d^\top X_d)^{-1}]=I_d/(n-d-1)$ **only when $n>d+1$**. Hence
$\mathbb E\|\Delta\|^2=\tau_d^2d/(n-d-1)$.

Above the threshold define $\Pi=X_d^\top(X_dX_d^\top)^{-1}X_d$ and $A=X_d^\top(X_dX_d^\top)^{-1}$. Now

$$\Delta=-(I-\Pi)\theta_d+A\eta_d.$$

The two terms lie in orthogonal spaces on every realization. Rotational symmetry gives $\mathbb E\Pi=(n/d)I_d$: its trace is $n$ and no direction is preferred. Thus the expected missing-signal contribution is $(1-n/d)B_d^2$. The noise contribution uses
$\mathbb E[(X_dX_d^\top)^{-1}]=I_n/(d-n-1)$ for $d>n+1$.

## The complete test MSE

$$\mathcal E_d=
\begin{cases}
\tau_d^2\left(1+\dfrac d{n-d-1}\right),&d<n-1,\\[5pt]
B_d^2(1-n/d)+\tau_d^2\left(1+\dfrac n{d-n-1}\right),&d>n+1.
\end{cases}$$

Population risk under half-squared loss is $\mathcal E_d/2$. When effective noise is positive, the expectation is infinite at $d=n-1,n,n+1$. A realized fit can have finite risk almost surely while the mean is infinite: rare almost-singular designs create a heavy tail. Never “repair” the denominator with a small numerical constant and call it this theorem.

## A peak does not guarantee two descents

For finite adjacent underparameterized dimensions,

$$\mathcal E_{d+1}<\mathcal E_d
\iff b_{d+1}^2>\frac{\tau_d^2}{n-d-1}.$$

Adding a zero-signal feature increases risk there. Strong early coordinates can create an initial descent; their order matters. Beyond interpolation, $\tau_d^2(d-1)/(d-n-1)$ is nonincreasing, while $B_d^2(1-n/d)$ is nondecreasing. The second descent occurs only when noise reduction dominates the growing unobserved-signal term.

The bias–variance identity still holds. In this model, for $d>n+1$,

$$\operatorname{Bias}^2=a_d+(1-n/d)^2B_d^2,$$
$$\operatorname{Var}=(n/d)(1-n/d)B_d^2+\tau_d^2n/(d-n-1).$$

Their sum is $\mathcal E_d-\sigma_y^2$. Random row-space orientation itself contributes variance, even before considering label noise. Double descent is a possible shape of the same decomposition, not a new error category.

**Checkpoint:** establish the minimum-norm solution and explain why inverse existence is weaker than finite inverse expectation. Attempt E05–E06. The [double-descent lab](../labs/README.md) keeps the true signal fixed as $d$ varies and marks the excluded critical dimensions. Finite Monte Carlo averages near the threshold are not reliable estimates of an infinite expectation.
