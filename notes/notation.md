# Notation and a short mathematical bridge

[Home](../README.md) · [Study map](00-study-map.md)

| Symbol | Meaning |
|---|---|
| $S=\{(X_i,Y_i)\}_{i=1}^n$ | Training sample; randomness from drawing data |
| $P$, $P_n$ | Population distribution and empirical measure |
| $L(f)$, $\widehat L_n(f)$ | Population and empirical risk |
| $\Phi(\theta)$ | Objective being optimized; may be empirical or population |
| $n,d,T$ | Sample size, parameter dimension, iteration count |
| $\theta^\star$ | An optimizer; uniqueness is stated when available |
| $\Delta_t=\Phi(\theta_t)-\Phi(\theta^\star)$ | Objective gap, random for stochastic algorithms |
| $\delta_t=\mathbb E\Delta_t$ | Expected objective gap |
| $\mu,\beta$ | Strong-convexity and gradient-Lipschitz constants |
| $\eta_t$ | Learning rate |
| $\mathcal H_t$ | History before drawing the current stochastic gradient |
| $\xi_t$, $\sigma_g^2$ | Gradient noise and a bound on its conditional second moment |
| $\sigma_y^2$ | Irreducible observation-noise variance in regression |
| $\kappa=\beta/\mu$ | Curvature condition number when $\mu>0$ |

The source reuses symbols across sections. This companion distinguishes observation noise from gradient noise. In the double-descent chapter, regression coefficients are $b_j$ rather than the smoothness constant $\beta$. The $S$ in a sandwich covariance is a noise covariance matrix, not the training set; the chapter states that switch explicitly.

## Linear algebra you will actually use

For $A\in\mathbb R^{n\times d}$, $\theta\in\mathbb R^d$, and $b\in\mathbb R^n$,

$$\nabla_\theta\frac12\|A\theta-b\|^2=A^\top(A\theta-b).$$

Expand the scalar expression as $\frac12\theta^\top A^\top A\theta-b^\top A\theta+\frac12b^\top b$ to check the derivative. Dimensions are a first debugging tool: each term in the gradient must lie in $\mathbb R^d$.

If $H$ is symmetric, write $H=Q\Lambda Q^\top$ with orthonormal columns in $Q$. For an error $e$, the coordinates $z=Q^\top e$ separate directions. The recursion $e^+=(I-\eta H)e$ becomes $z_j^+=(1-\eta\lambda_j)z_j$.

A symmetric matrix is positive definite when $v^\top Hv>0$ for every nonzero $v$. Positive semidefinite permits zero. For a symmetric projector $P$, $P^2=P$, so $\|Pv\|^2=v^\top Pv$. The row space of $A$ is orthogonal to its null space: vectors invisible to the data lie in $\ker A$.

The trace is the sum of diagonal entries. Since $\|v\|^2=\operatorname{tr}(vv^\top)$, covariance matrices turn into mean squared norms by taking a trace. This is why inverse Gram matrices appear in least-squares prediction risk.

## Expectations and rates

“Given $\mathcal H_t$” means the current parameter is known while the new sample remains random. If $\mathbb E[\xi_t\mid\mathcal H_t]=0$, then $\mathbb E[\langle a_t,\xi_t\rangle\mid\mathcal H_t]=0$ for a history-measurable $a_t$. Independence from the entire trajectory is not needed.

$a_T=O(T^{-1})$ is an asymptotic upper bound with a constant independent of $T$; it does not specify that constant. A rate in expectation, one holding with high probability, and one holding on every path are distinct. $O_P$ and $o_P$ are explained in [Chapter 10](10-probability.md).
