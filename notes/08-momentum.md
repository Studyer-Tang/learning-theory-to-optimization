# 08 · Momentum: a second-order recursion

[Previous](07-asymptotic-uncertainty.md) · [Home](../README.md) · [Next](09-adam.md)

**Goal:** understand the acceleration mechanism and its domain of validity. Source connection: §2.6.

## Memory changes the algorithm

Heavy-ball uses $m_t=\rho m_{t-1}+\nabla\Phi(\theta_t)$ and $\theta_{t+1}=\theta_t-\eta m_t$. Eliminating $m_t$ gives

$$\theta_{t+1}=\theta_t-\eta\nabla\Phi(\theta_t)+\rho(\theta_t-\theta_{t-1}).$$

The new point depends on the last displacement as well as the gradient. This changes the iterates themselves. Reporting their average after training, as in Polyak–Ruppert averaging, is a different operation.

## Analyze a quadratic one direction at a time

For $\Phi(\theta)=\frac12(\theta-\theta^\star)^\top H(\theta-\theta^\star)$ with $\mu I\preceq H\preceq\beta I$, an eigendirection obeys

$$z_{t+1}=(1+\rho-\eta\lambda)z_t-\rho z_{t-1}.$$

Try a solution $z_t=r^t$. The characteristic roots are

$$r_\pm=\frac{1+\rho-\eta\lambda\pm
\sqrt{(1+\rho-\eta\lambda)^2-4\rho}}2.$$

Stability requires both roots inside the unit disk. For a real monic quadratic $r^2-ar+\rho$, the second-order Jury conditions are $1-a+\rho>0$, $1+a+\rho>0$, and $1-\rho>0$. Substituting $a=1+\rho-\eta\lambda$, with $\rho\ge0$, yields

$$0\le\rho<1,\qquad 0<\eta<\frac{2(1+\rho)}\beta.$$

The largest root magnitude across eigenvalues controls the asymptotic parameter rate. The quadratic-optimal choices are

$$\eta_{\rm HB}=\frac4{(\sqrt\beta+\sqrt\mu)^2},\qquad
\rho_{\rm HB}=\left(\frac{\sqrt\beta-\sqrt\mu}{\sqrt\beta+\sqrt\mu}\right)^2.$$

They give the root factor $q_{\rm HB}=(\sqrt\kappa-1)/(\sqrt\kappa+1)$, compared with $(\kappa-1)/(\kappa+1)$ for optimized fixed-step GD. If a root is repeated, a solution can contain $tq^t$. Therefore the precise statement is a limsup root rate, not automatically a finite-time bound $Cq^t$ with a fixed constant.

For $\mu=1,\beta=9$, the optimal parameters are $\eta=1/4$, $\rho=1/4$, and the endpoint roots are $1/2$ and $-1/2$, each repeated. GD's optimized root factor is $0.8$. This comparison is exact for a deterministic quadratic. A nonlinear strongly convex smooth function needs a separate global analysis; its changing Hessian is not the same fixed linear recurrence.

## A distinct stochastic momentum theorem

Now let $\Phi=n^{-1}\sum_i\phi_i$, each component convex and $L_{\max}$-smooth. Define $\sigma_\star^2=\mathbb E_I\|\nabla\phi_I(\theta^\star)\|^2$. Component smoothness yields

$$\mathbb E_I\|\nabla\phi_I(\theta)\|^2
\le4L_{\max}[\Phi(\theta)-\Phi(\theta^\star)]+2\sigma_\star^2.$$

One way to see the constant is to split the gradient into its difference from the optimum plus the optimum gradient, bound the squared sum by twice each squared norm, and use the convex smooth gradient-difference inequality. Averaging cancels the linear term at the optimum because $\nabla\Phi(\theta^\star)=0$.

Use time-dependent momentum

$$m_t=\rho_tm_{t-1}+\nabla\phi_{I_t}(\theta_t),\quad
\theta_{t+1}=\theta_t-\gamma_tm_t,\quad
\gamma_t=\frac{2\eta}{t+3},\quad\rho_t=\frac t{t+2},\quad
0<\eta\le\frac1{4L_{\max}}.$$

The resulting final-iterate guarantee is

$$\mathbb E\Delta_T\le\frac{R^2}{\eta(T+1)}+2\eta\sigma_\star^2,
\quad R=\|\theta_0-\theta^\star\|.$$

This is not the constant-parameter heavy-ball theorem with noise inserted. To see the proof structure, define $\lambda_t=t/2$, $z_{-1}=\theta_0$ and the equivalent recursion

$$z_t=z_{t-1}-\eta g_t,\qquad
\theta_{t+1}=\frac{\lambda_{t+1}\theta_t+z_t}{\lambda_{t+1}+1}.$$

The previous relation gives $z_{t-1}=\theta_t+\lambda_t(\theta_t-\theta_{t-1})$. Convexity bounds the inner product with the first term by $\Delta_t$ and with the displacement by $\Delta_t-\Delta_{t-1}$. Expand $\|z_t-\theta^\star\|^2$, condition, and insert the gradient second-moment bound. Since $1+\lambda_t-2\eta L_{\max}\ge\lambda_{t+1}$,

$$\mathbb E[\|z_t-\theta^\star\|^2\mid\mathcal H_t]
\le\|z_{t-1}-\theta^\star\|^2
-2\eta\lambda_{t+1}\Delta_t+2\eta\lambda_t\Delta_{t-1}
+2\eta^2\sigma_\star^2.$$

The term with $\lambda_0=0$ is omitted. Summing leaves only $-\eta(T+1)\mathbb E\Delta_T$ and the accumulated noise term. Drop the nonnegative final squared distance to obtain the stated result.

If the objective is strongly convex, multiply the objective bound by $2/\mu$ to bound expected squared parameter distance. Balancing the two terms with a horizon-tuned base step gives $O(T^{-1/2})$, subject to the step restriction. If $\sigma_\star=0$, the same schedule gives $O(1/T)$. The noisy guarantee is not a strong-convexity acceleration result: strong convexity was used only to convert the error metric. The source connects this argument to [the gradient-method handbook](https://arxiv.org/abs/2301.11235).

**Checkpoint:** derive the scalar roots and explain why two algorithms both called “momentum” need not share a theorem. Attempt E17. The numerical checks verify the moving-average and momentum recursions agree on a deterministic quadratic.
