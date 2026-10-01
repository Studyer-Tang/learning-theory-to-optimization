# 06 · Stochastic gradient descent: progress plus noise

[Previous](05-gd-proof-patterns.md) · [Home](../README.md) · [Next](07-asymptotic-uncertainty.md)

**Goal:** distinguish fixed-step noise floors, decreasing-step convergence, and output averaging. Source connection: §2.4.

## A random gradient, with the history held fixed

For $\Phi(\theta)=n^{-1}\sum_i\phi_i(\theta)$, let $\mathcal H_t$ contain the data and all previous algorithmic draws. Before drawing the next sample, $\theta_t$ is known conditional on this history. A minibatch of $b$ independent uniform indices gives

$$g_t=\frac1b\sum_{j=1}^b\nabla\phi_{I_{t,j}}(\theta_t),\quad
\mathbb E[g_t\mid\mathcal H_t]=\nabla\Phi(\theta_t),\quad
\operatorname{Cov}(g_t\mid\mathcal H_t)=\frac1b\operatorname{Cov}(\nabla\phi_I(\theta_t)\mid\mathcal H_t).$$

This covariance formula uses independent sampling with replacement. It reduces algorithmic noise, not uncertainty about the data-generating distribution. Unbiased gradients do not imply unbiased iterates or unbiased test errors.

Two assumptions must be distinguished:

- **Noise bound:** $g_t=\nabla\Phi(\theta_t)+\xi_t$, with conditional mean-zero $\xi_t$ and $\mathbb E[\|\xi_t\|^2\mid\mathcal H_t]\le\sigma_g^2$.
- **Full second-moment bound:** $\mathbb E[\|g_t\|^2\mid\mathcal H_t]\le G^2$.

The latter implies a noise bound but is stronger. On unbounded parameter space it should not be silently assumed uniformly at every point of a globally strongly convex objective, whose gradient grows without bound. Use it along the trajectory under appropriate boundedness assumptions, or use a noise-based theorem instead.

## Derive the fixed-step noise floor

Smoothness applied to $\theta_{t+1}=\theta_t-\eta g_t$ gives, before any expectation,

$$\Phi(\theta_{t+1})\le\Phi(\theta_t)-\eta\nabla\Phi(\theta_t)^\top g_t+\frac{\beta\eta^2}{2}\|g_t\|^2.$$

Condition on the history. The cross term in $\|\nabla\Phi+\xi_t\|^2$ has conditional expectation zero. Therefore, for $\eta\le1/\beta$,

$$\mathbb E[\Phi(\theta_{t+1})\mid\mathcal H_t]
\le\Phi(\theta_t)-\frac\eta2\|\nabla\Phi(\theta_t)\|^2+\frac{\beta\eta^2\sigma_g^2}{2}.$$

Under $\mu$-strong convexity, gradient domination and total expectation produce

$$\delta_{t+1}\le(1-\mu\eta)\delta_t+\frac{\beta\eta^2\sigma_g^2}{2}.$$

Let $r=1-\mu\eta$. Starting at $t=1$, summing the geometric series yields

$$\delta_T\le r^{T-1}\delta_1+\frac{\beta\eta\sigma_g^2}{2\mu}(1-r^{T-1}).$$

A larger step accelerates the transient and raises the residual upper-bound scale. A smaller step lowers that scale but slows the transient. An upper bound with a floor is not proof that every instance has a positive limiting error.

The scalar quadratic makes the phenomenon exact. For $\Phi(x)=hx^2/2$ and iid additive gradient noise of variance $\sigma_g^2$,

$$\mathbb E x_{t+1}^2=(1-\eta h)^2\mathbb E x_t^2+\eta^2\sigma_g^2,$$

so its stationary objective expectation is $\eta\sigma_g^2/[2(2-\eta h)]$ when $0<\eta<2/h$. This is the analytic reference in the [SGD lab](../labs/README.md).

## Decreasing steps control the final iterate

Take $\eta_t=2/[\mu(t+\tau)]$ with $\tau\ge2\beta/\mu$. Then

$$\delta_{t+1}\le\left(1-\frac2{t+\tau}\right)\delta_t+
\frac C{(t+\tau)^2},\qquad C=2\beta\sigma_g^2/\mu^2.$$

Set $A=\tau\delta_1+C$. The inductive claim $\delta_t\le A/(t+\tau-1)$ is true at $t=1$. With $s=t+\tau$, the next upper bound is $A(s-2)/[s(s-1)]+C/s^2\le A/s$, since $A\ge C$. Thus

$$\delta_T\le\frac A{T+\tau-1},\qquad
\mathbb E\|\theta_T-\theta^\star\|^2\le\frac{2A}{\mu(T+\tau-1)}.$$

Expected parameter norm is at most the square root of the last bound by Cauchy–Schwarz. This controls the final iterate, without averaging. It is a finite-time statement in expectation; almost-sure convergence needs its own argument.

## What arithmetic averaging changes

Let $\bar\theta_T=T^{-1}\sum_{t=1}^T\theta_t$. Under the preceding assumptions, Minkowski's inequality gives

$$\left(\mathbb E\|\bar\theta_T-\theta^\star\|^2\right)^{1/2}
\le\frac1T\sum_t\sqrt{\mathbb E\|\theta_t-\theta^\star\|^2}
\le\sqrt{\frac{8A}{\mu T}}.$$

We used $\sum_{t=1}^Tt^{-1/2}\le2\sqrt T$. Smoothness at the optimum then gives $\mathbb E\Delta(\bar\theta_T)\le4\beta A/(\mu T)$. These conservative constants do not claim averaging always beats the last iterate at every finite horizon.

Without smoothness, the quadratic **upper** model is missing. Under strong convexity and a bounded full gradient second moment, a distance expansion instead yields

$$2\eta_t\delta_t\le(1-\mu\eta_t)d_t-d_{t+1}+\eta_t^2G^2,
\qquad d_t=\mathbb E\|\theta_t-\theta^\star\|^2.$$

For $\eta_t=1/(\mu t)$, multiply by $\mu t$ and sum: the distance terms telescope, and equal-weight averaging gives $G^2H_T/(2\mu T)=O(\log T/T)$. For $\eta_t=2/[\mu(t+1)]$, multiplying by $\mu t(t+1)/4$ produces telescoping coefficients $t(t-1)$ and $t(t+1)$. Weights proportional to $t$ then give $2G^2/[\mu(T+1)]$. An averaging rule is part of the theorem.

## Convex SGD: the weighted-average theorem

Assume convex differentiability, an attained optimum, unbiased gradients, and the full second-moment bound along the iterates. No smoothness is needed for this calculation. Expand the squared distance, condition, and use convexity:

$$2\eta_t\mathbb E\Delta(\theta_t)
\le d_t-d_{t+1}+\eta_t^2G^2.$$

Let $S_T=\sum\eta_t$, $Q_T=\sum\eta_t^2$, $d_1\le R^2$, and $\bar\theta_T^{(\eta)}=S_T^{-1}\sum\eta_t\theta_t$. Summation plus convexity of $\Phi$ gives

$$\mathbb E\Delta(\bar\theta_T^{(\eta)})\le\frac{R^2+G^2Q_T}{2S_T}.$$

With fixed $\eta$, this is $R^2/(2\eta T)+\eta G^2/2$. Choose $\eta=R/(G\sqrt T)$ to balance the terms and obtain $RG/\sqrt T$. This fixed step depends on the intended horizon and stays constant within that run.

For a horizon-free schedule $\eta_t=ct^{-\alpha}$, write $H_{T,p}=\sum t^{-p}$. The same bound becomes $(R^2+c^2G^2H_{T,2\alpha})/(2cH_{T,\alpha})$. It gives $O(T^{-\alpha})$ for $\alpha<1/2$, $O(\log T/\sqrt T)$ at $1/2$, $O(T^{-(1-\alpha)})$ for $1/2<\alpha<1$, and $O(1/\log T)$ at one. If $\alpha>1$, cumulative steps stay finite and this bound need not vanish.

## Read the guarantee literally

An expectation bound does not automatically become a sharp high-probability theorem. A weak one does follow from Markov: if $\mathbb E\Delta\le B$, then $\Pr(\Delta>B/\delta)\le\delta$. Better confidence dependence requires further analysis or assumptions. Likewise, a theorem for an averaged output does not automatically describe the last iterate.

**Checkpoint:** derive the conditional squared-gradient identity and the weighted-average bound. Attempt E12–E15. Exit skill: label the output, error metric, randomness, and step schedule before quoting an SGD rate.
