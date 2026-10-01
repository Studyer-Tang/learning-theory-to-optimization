# 04 · Geometry and exact gradient descent

[Previous](03-double-descent.md) · [Home](../README.md) · [Next](05-gd-proof-patterns.md)

**Goal:** derive a stable step-size interval instead of memorizing one. Source connection: §2.1.

## Begin in one dimension

For $\Phi(x)=\frac12hx^2$ with $h>0$, the optimizer is zero and GD gives $x_{t+1}=(1-\eta h)x_t$. Everything follows from that multiplier. Values between zero and one shrink without sign changes; values between minus one and zero shrink while alternating signs; magnitude greater than one produces growth whenever the initial error is nonzero.

Thus convergence from every starting point requires $0<\eta<2/h$. At $\eta=1/h$ the error disappears in one step. At $2/h$ it oscillates without shrinking. This is the small problem behind the matrix theorem.

## The exact least-squares recursion

For $\Phi(\theta)=\frac12\|A\theta-b\|^2$, an optimizer satisfies $A^\top A\theta^\star=A^\top b$. Subtract its equation from the GD update:

$$e_{t+1}=(I-\eta H)e_t,\qquad H=A^\top A,\quad e_t=\theta_t-\theta^\star.$$

With full column rank, $H\succ0$. Diagonalize it and define $\mu=\lambda_{\min}(H)$, $\beta=\lambda_{\max}(H)$. Each coordinate satisfies the scalar recursion above, so

$$q(\eta)=\max_j|1-\eta\lambda_j|
=\max\{|1-\eta\mu|,|1-\eta\beta|\}.$$

The endpoint formula holds because the absolute value of an affine function is convex on the interval and both endpoints are eigenvalues. If $0<\eta<2/\beta$, then $q<1$ and

$$\|e_T\|\le q^T\|e_0\|,\qquad
\Phi(\theta_T)-\Phi(\theta^\star)\le q^{2T}[\Phi(\theta_0)-\Phi(\theta^\star)].$$

The objective bound follows from $\Phi(\theta)-\Phi(\theta^\star)=\frac12e^\top He$ and the same coordinate factors. The $1/\beta$ choice is a convenient nonoscillatory step, not the exact stability boundary.

## Balance the slow and fast directions

At $\eta=1/\beta$, $q=1-\mu/\beta$. Increasing the step helps the low-curvature direction but eventually increases the alternating high-curvature factor. Equalize them:

$$1-\eta\mu=\eta\beta-1
\quad\Rightarrow\quad
\eta_{\rm opt}=\frac2{\mu+\beta},\quad
q_{\rm opt}=\frac{\kappa-1}{\kappa+1},\quad\kappa=\beta/\mu.$$

For $H=\operatorname{diag}(1,9)$, $1/\beta=1/9$ gives multipliers $8/9,0$; the optimized step $0.2$ gives $0.8,-0.8$. Allowing controlled oscillation speeds up the slow direction. Optimality here is among fixed scalar steps for the worst eigendirection, not among all possible algorithms.

## What deficient rank changes

When $H$ has a zero eigenvalue, its error multiplier is one. GD cannot change the initial null-space component. Under a stable step on the positive spectrum,

$$\theta_\infty=A^\dagger b+P_{\ker A}\theta_0.$$

From zero initialization, the second term vanishes. All these limits minimize the same least-squares objective. Consequently, objective convergence alone does not select a unique parameter vector. In the full-row-rank overparameterized case, the minimum-norm solution also interpolates; in a general inconsistent least-squares problem, it need not.

## Geometry beyond quadratics

For a differentiable function on $\mathbb R^d$, convexity means

$$\Phi(v)\ge\Phi(u)+\langle\nabla\Phi(u),v-u\rangle.$$

Strong convexity with $\mu>0$ adds $\frac\mu2\|v-u\|^2$ to the right. It yields a unique minimizer for a finite differentiable function on all of $\mathbb R^d$. Smoothness means the gradient is Lipschitz:

$$\|\nabla\Phi(v)-\nabla\Phi(u)\|\le\beta\|v-u\|.$$

It does not by itself imply convexity. If twice differentiable, smoothness bounds the Hessian's operator norm; in the convex case its eigenvalues lie in $[0,\beta]$. Strong convexity places them above $\mu$.

The function $x^4$ has a unique minimizer but no global positive strong-convexity constant. The function $x^2$ on $(x,y)$ is convex and smooth but flat in $y$. The concave function $-x^2$ is smooth but not convex. These counterexamples keep the definitions separate.

## Derive the descent lemma once

Set $d=v-u$ and integrate the gradient along the segment:

$$\Phi(v)-\Phi(u)-\langle\nabla\Phi(u),d\rangle
=\int_0^1\langle\nabla\Phi(u+sd)-\nabla\Phi(u),d\rangle ds
\le\frac\beta2\|d\|^2.$$

Cauchy–Schwarz and smoothness bound the integrand by $\beta s\|d\|^2$. Substituting $v=u-\eta\nabla\Phi(u)$ gives

$$\Phi(v)\le\Phi(u)-\eta(1-\beta\eta/2)\|\nabla\Phi(u)\|^2.$$

For $0<\eta\le1/\beta$, the decrease is at least $\frac\eta2\|\nabla\Phi(u)\|^2$. A descent guarantee needs smoothness, not convexity. Global optimality requires additional structure.

**Checkpoint:** reproduce the scalar stability calculation, then generalize it to a singular diagonal matrix. Attempt E07–E09 and run the [GD lab](../labs/README.md).
