# Rate sheet with assumptions attached

[Home](../README.md) · [Notation](../notes/notation.md)

Let $\Delta_T=\Phi(\theta_T)-\Phi(\theta^\star)$ and $e_T=\theta_T-\theta^\star$. Constants hidden below may depend on initial distance, curvature, and noise. A displayed rate is an upper-bound order, not an exact trajectory.

| Method | Assumptions and output | Guarantee |
|---|---|---|
| Quadratic GD | $H\succ0$, fixed $0<\eta<2/\beta$, final | $\Vert e_T\Vert\le q^T\Vert e_0\Vert$; $\Delta_T\le q^{2T}\Delta_0$ |
| General GD | $\mu$-strong convexity, $\beta$-smoothness, $\eta=1/\beta$, final | $\Delta_T\le(1-1/\kappa)^T\Delta_0$ |
| Convex GD | Convex, smooth, attained minimum, fixed $\eta\le1/\beta$, final | $\Delta_T\le\Vert e_0\Vert^2/(2\eta T)$ |
| Constant-step SGD | Strongly convex, smooth, unbiased bounded noise, final | $\mathbb E\Delta_T\le(1-\mu\eta)^{T-1}\delta_1+\beta\eta\sigma_g^2/(2\mu)$ |
| Decreasing-step SGD | Same; specific $2/[\mu(t+\tau)]$ schedule, final | $\mathbb E\Delta_T=O(T^{-1})$, $\mathbb E\Vert e_T\Vert^2=O(T^{-1})$ |
| Arithmetic SGD average | Same smooth strongly convex setting and schedule | Expected gap and mean squared parameter error $O(T^{-1})$ |
| Nonsmooth-bound argument | Differentiable strongly convex, full second moment bounded along iterates; $1/(\mu t)$ and equal weights | Expected averaged gap $O(\log T/T)$ |
| Weighted strongly convex average | Same moment setting; $2/[\mu(t+1)]$ and weights $t$ | Expected averaged gap $O(T^{-1})$ |
| Convex SGD | Unbiased bounded second moment, horizon-tuned constant step, equal-weight output | Expected gap $O(T^{-1/2})$ |
| Convex SGD | Same; $c/\sqrt t$ and step-size-weighted output | Expected gap $O(\log T/\sqrt T)$ under this bound |
| Final SGD CLT | Local stability, regularity, moments; $a/t$, $2a\lambda_{\min}(H)>1$ | $\sqrt T e_T\Rightarrow N(0,V_a)$ |
| PR CLT | Local assumptions; $at^{-\alpha}$, $1/2<\alpha<1$, arithmetic average | Sandwich covariance $H^{-1}S(H^{-1})^\top$ |
| Heavy-ball | Deterministic positive-definite quadratic, optimal parameters | Asymptotic parameter root factor $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ |
| Stochastic momentum | Convex smooth components, specified time-dependent schedule | Final expected gap $O((\eta T)^{-1})+O(\eta\sigma_\star^2)$ |
| Fixed preconditioner | Fixed positive diagonal bounds, exact gradients, permitted step | Geometric objective contraction |
| Vanilla Adam | Only strong convexity and smoothness | No general rate justified by these assumptions alone |

Under strong convexity, $\mathbb E\|e_T\|^2\le2\mathbb E\Delta_T/\mu$. A parameter-norm bound requires a square root. In the merely convex case there is no comparable uniform distance rate to a designated optimizer from the objective bound alone.
