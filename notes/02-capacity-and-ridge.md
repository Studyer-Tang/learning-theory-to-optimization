# 02 · Capacity, validation, and ridge regression

[Previous](01-risk-and-representation.md) · [Home](../README.md) · [Next](03-double-descent.md)

**Goal:** explain why training fit, population performance, and model-class quality can move differently. Source connection: §1.4.

## Larger classes create opportunities, not guarantees

For nested classes $\mathcal F_q\subseteq\mathcal F_{q+1}$, both $\inf_{f\in\mathcal F_q}\widehat L_n(f)$ and $\inf_{f\in\mathcal F_q}L(f)$ are nonincreasing in $q$. You are minimizing over more choices. But $L(\widehat f_{q,S})$, the risk of an actual fitted model, need not decrease: fitting can exploit sample-specific structure.

Capacity is not simply the number of written parameters. Redundant parameterizations, norm constraints, shared weights, initialization, and stopping rules all affect the functions an algorithm can effectively select. A degree-nine polynomial with only six interpolation constraints has many exact fits even when labels have no noise. A tie-breaking rule selects one of them; interpolation alone does not determine its behavior elsewhere.

## Bias and variance describe repeated training

For a fresh $X$ independent of the training sample, let $\bar f(x)=\mathbb E_S\widehat f_S(x)$. Add and subtract $m(X)$ and $\bar f(X)$ inside $Y-\widehat f_S(X)$. Conditional mean-zero residuals and averaging over independent training samples eliminate the cross terms:

$$\mathbb E_{S,X,Y}(Y-\widehat f_S(X))^2
=\mathbb E_X\operatorname{Var}(Y\mid X)
+\mathbb E_X(\bar f(X)-m(X))^2
+\mathbb E_X\operatorname{Var}_S(\widehat f_S(X)).$$

With the half-squared loss, divide the entire expression by two. Variance here means variability of the learned predictor across repeated datasets, not variability of observed labels within one dataset. A single train–validation split does not identify these three terms.

High training and validation errors suggest checking representation, optimizer progress, preprocessing, and regularization. Low training loss with much higher validation loss suggests poor transfer, but leakage, distribution shift, and repeated tuning can imitate patterns. These are diagnostic clues, not logical definitions. A U-shaped curve is one possible outcome, not a consequence of nesting.

## Ridge as a complete worked model

Use the convention

$$J_\lambda(\theta)=\frac1{2n}\|y-X\theta\|^2+\frac\lambda2\|\theta\|^2.$$

Differentiate and set the gradient to zero:

$$\frac1nX^\top(X\theta-y)+\lambda\theta=0,
\qquad\widehat\theta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y.$$

The $n\lambda$ is a consequence of the normalization, not a new kind of regularization. An unpenalized intercept can be handled by centering with training-set statistics. At $\lambda=0$ and deficient rank, use $X^\dagger y$; the ordinary inverse is unavailable.

Write a thin SVD $X=U\operatorname{diag}(s_j)V^\top$. Then

$$\widehat\theta_\lambda=
\sum_{j=1}^r\frac{s_j}{s_j^2+n\lambda}(u_j^\top y)v_j,
\qquad
\widehat y_\lambda=
\sum_{j=1}^r a_j(\lambda)(u_j^\top y)u_j,
\quad a_j=\frac{s_j^2}{s_j^2+n\lambda}.$$

Small singular values produce large inverse factors in unregularized fitting. Ridge damps those directions most strongly. Its fitted-value effective degrees of freedom are $\sum_j a_j$, which decrease with $\lambda$.

To see why training loss cannot select regularization, decompose the residual:

$$\|y-\widehat y_\lambda\|^2
=\|(I-UU^\top)y\|^2+
\sum_j\left(\frac{n\lambda}{s_j^2+n\lambda}\right)^2(u_j^\top y)^2.$$

Each variable term is nondecreasing in $\lambda$, so training residual favors no regularization, up to ties. Comparing minimized **penalized** values across $\lambda$ is also not a common prediction-risk comparison: the objective itself changes.

## The tradeoff inside one direction

Under $y=X\theta^\star+\varepsilon$ with conditional covariance $\sigma_y^2I$, let $\alpha_j=v_j^\top\theta^\star$. The estimated coefficient has mean $a_j\alpha_j$ and variance $\sigma_y^2s_j^2/(s_j^2+n\lambda)^2$. Consequently,

$$\operatorname{Bias}_j^2=(1-a_j)^2\alpha_j^2,
\qquad\operatorname{Var}_j=\frac{\sigma_y^2s_j^2}{(s_j^2+n\lambda)^2}.$$

Larger regularization increases shrinkage bias and lowers noise amplification. Turning this into prediction error requires the future covariate covariance; summing coordinate errors directly is appropriate for isotropic test inputs, with an additional null-space term if needed.

## A protocol you can implement

Fit preprocessing and model parameters on training data. Select a penalty using validation loss on held-out data. Freeze choices before evaluating the test set. With limited data, cross-validation reuses observations efficiently, but overlapping training folds make naive independent-fold standard errors suspect. If the same cross-validation selects and reports, use an outer evaluation layer for an honest performance estimate.

Regularization can change representation, statistics, and optimization simultaneously. It is not a fourth independent term in the preceding chapter's bound. If you optimize a penalized objective, its optimization gap belongs to that objective.

**Checkpoint:** derive the ridge formula under $\|y-X\theta\|^2+\lambda\|\theta\|^2$ and explain why its denominator differs. Attempt E03–E04. The [ridge lab](../labs/README.md) displays training and validation losses without using validation to fit coefficients.
