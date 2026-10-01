# 10 · A probability toolkit for learning and optimization

[Previous](09-adam.md) · [Home](../README.md) · [Return to uncertainty](07-asymptotic-uncertainty.md)

**Goal:** know which limit operation is justified, and which extra assumption it needs. Source connection: Appendix A.

## Conditional expectation before asymptotics

In SGD, condition on the information available before the next draw. If $a_t$ is determined by that history and $\mathbb E[\xi_t\mid\mathcal H_t]=0$, then
$\mathbb E[a_t^\top\xi_t\mid\mathcal H_t]=0$. Taking total expectation removes the conditioning afterward. This is the tower property; it does not require the current iterate to be independent of past noise.

For an integrable scalar variable, $\mathbb E[\mathbb E[X\mid\mathcal H]]=\mathbb EX$. For a nonnegative variable, Markov gives $\Pr(X>c)\le\mathbb EX/c$. Applied to a squared error, it converts a mean-square bound into a probability bound, albeit usually a loose one.

## Four meanings of convergence

| Mode | Definition | What is compared |
|---|---|---|
| Probability | $\Pr(\Vert X_n-X\Vert>\epsilon)\to0$ for each $\epsilon>0$ | Error at each large index |
| Distribution | CDFs converge at continuity points of the limit CDF | Laws, not paired sample values |
| $L^p$ | $\mathbb E\Vert X_n-X\Vert^p\to0$ | Average powered error |
| Almost surely | $\Pr(\lim_n\Vert X_n-X\Vert=0)=1$ | Entire sample paths |

Almost-sure convergence implies convergence in probability, which implies convergence in distribution. $L^p$ convergence implies convergence in probability by Markov. For $p\ge q>0$, $L^p$ convergence implies $L^q$ convergence on a probability space. Almost-sure and moment convergence are not generally ordered.

Test the reverse arrows with examples:

1. If $X$ is equally likely to be $1$ or $-1$ and $X_n=-X$, then $X_n$ has the same law as $X$ for every $n$, but $|X_n-X|=2$. Distributional convergence need not imply probability convergence to this coupled $X$.
2. Independent events with probabilities $1/n$ occur infinitely often almost surely by Borel–Cantelli. Their indicators converge to zero in every finite $L^p$ and in probability, but not almost surely.
3. For fixed $p>0$, let $U$ be uniform on $(0,1)$ and $X_n=n^{1/p}\mathbf1\{U\le1/n\}$. Almost every fixed $U$ eventually leaves the event, so $X_n\to0$ almost surely. Yet $\mathbb E|X_n|^p=1$.

The third example also warns against exchanging expectation and a limit without uniform integrability or another sufficient condition.

## Laws of large numbers and the CLT answer different questions

For iid scalar observations with $\mathbb E|X_1|<\infty$, the sample mean converges almost surely to $\mathbb EX_1$ (and hence in probability). With iid vector observations, mean $m$, and finite covariance $\Sigma$,

$$\sqrt n(\bar X_n-m)\Rightarrow N(0,\Sigma).$$

The law of large numbers identifies the destination. The CLT describes remaining fluctuations after enlarging the vanishing error by $\sqrt n$. It does not say the unscaled error approaches a nondegenerate Gaussian. These classical theorems are used here as probability prerequisites; proving the general LLN or CLT is beyond this companion's elementary derivations.

## Continuous mapping and Slutsky

If $g$ is measurable and continuous at the values taken by the limit with probability one, continuous mapping transfers probability, almost-sure, or distributional convergence through $g$. Matrix inversion is continuous away from singular matrices; the qualification is essential.

If $X_n\Rightarrow X$ and $Y_n\to c$ in probability for a constant $c$, then $(X_n,Y_n)\Rightarrow(X,c)$. Consequently sums and products converge to $X+c$ and $cX$, and quotients to $X/c$ if $c\ne0$. A remainder tending to zero in probability can also be added without changing a distributional limit.

Knowing two nonconstant marginal limits is insufficient to infer independence or a joint limit. If $Y_n=-X_n$ and $X_n$ is standard normal, both marginals are standard normal but their sum is exactly zero. In contrast, replacing a positive scalar standard deviation by a consistent estimator in a standardized CLT is a valid Slutsky step.

## Tightness prevents probability mass from escaping

A family is uniformly tight when for every $\epsilon>0$ some finite $M$ satisfies $\sup_\alpha\Pr(\|X_\alpha\|>M)<\epsilon$. A uniform $p$th-moment bound proves this using $\Pr(\|X\|>M)\le\mathbb E\|X\|^p/M^p$.

Weakly convergent sequences are tight: choose a large continuity radius for the limit, transfer its tail bound to all sufficiently late terms, and enlarge the radius for finitely many early terms. Conversely, Prokhorov's theorem in finite dimensions says a tight sequence has weakly convergent subsequences. Tightness alone does not identify a unique limit.

The deterministic sequence $X_n=n$ is not tight. Its CDF tends pointwise to zero, which is not a probability CDF on the real line. Pointwise-looking convergence can lose all mass.

## Stochastic orders are bookkeeping, not deterministic bounds

For positive deterministic $a_n$, $X_n=O_P(a_n)$ means $X_n/a_n$ is bounded in probability; $X_n=o_P(a_n)$ means it tends to zero in probability. Useful rules are

$$O_P(a_n)O_P(b_n)=O_P(a_nb_n),\qquad
o_P(a_n)O_P(b_n)=o_P(a_nb_n),$$

and sums have order $O_P(a_n+b_n)$, or $o_P(a_n+b_n)$ when both are small-order terms. No independence is required for these rules. Bound one factor on a high-probability compact set, then control the other. To take reciprocals safely, use a nonzero probability limit: if $X_n/a_n\to c\ne0$, then $X_n^{-1}=a_n^{-1}(c^{-1}+o_P(1))$. $X_n=O_P(1)$ alone does not justify an $O_P(1)$ reciprocal.

A CLT implies $\widehat\theta_n-\theta=O_P(n^{-1/2})$ by tightness of the scaled sequence. Adding $o_P(n^{-1/2})$ does not alter its $\sqrt n$-scale limit.

## Derive the delta method

Suppose $r_n(T_n-\theta)\Rightarrow Z$, $r_n\to\infty$, and $g$ is differentiable at $\theta$ with Jacobian $G$. Tightness first gives $T_n-\theta=o_P(1)$. Differentiability gives

$$g(T_n)-g(\theta)=G(T_n-\theta)+R_n,\qquad
\|R_n\|=o_P(\|T_n-\theta\|).$$

Multiplying by $r_n$ makes the remainder $o_P(1)$. Slutsky therefore gives $r_n[g(T_n)-g(\theta)]\Rightarrow GZ$. If $Z$ is $N(0,\Sigma)$, the transformed covariance is $G\Sigma G^\top$.

For $g(x)=\log x$ at $m>0$, the derivative is $1/m$, so a scalar sample-mean CLT transforms its variance to $\sigma^2/m^2$. Positivity holds with probability tending to one. If the derivative vanishes, a different scale may be needed: $\sqrt nT_n\Rightarrow N(0,\sigma^2)$ implies $nT_n^2\Rightarrow\sigma^2\chi_1^2$ by continuous mapping. The limit is not a centered normal.

**Checkpoint:** supply a counterexample to reversing one convergence arrow, then derive a transformed variance without memorizing it. Attempt E19–E20. These tools let you read Chapter 07 without conflating convergence of means, convergence of distributions, and almost-sure convergence.
