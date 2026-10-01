# 01 · What is a learning algorithm trying to achieve?

[Previous](00-study-map.md) · [Home](../README.md) · [Next](02-capacity-and-ridge.md)

**Goal:** distinguish a good representation, a well-solved training problem, and a predictor that transfers to new data. Source connection: §§1.1–1.3.

## Start with a prediction problem

Suppose $Y=m(X)+\varepsilon$, with $\mathbb E[\varepsilon\mid X]=0$. The conditional mean $m(x)=\mathbb E[Y\mid X=x]$ is the best unrestricted predictor under squared loss, assuming finite second moments. To see why, fix $x$ and expand

$$\mathbb E[(Y-a)^2\mid X=x]=\operatorname{Var}(Y\mid X=x)+(m(x)-a)^2.$$

The first term cannot be changed by choosing $a$; the second is minimized at $a=m(x)$. Gaussian noise is unnecessary for this argument. This also explains why zero error may be impossible even with a perfect representation.

A linear class contains $f_\theta(x)=\theta^\top x$. An intercept gives an affine model. A fixed response map such as the logistic sigmoid gives $f_\theta(x)=\psi(\theta^\top x)$, but a full generalized linear model also specifies an observation law and loss. A one-hidden-layer ReLU network learns multiple features, $a_0+\sum_j a_j\max(w_j^\top x+b_j,0)$. More flexible representation changes both statistical and computational difficulty; it does not promise a better fitted predictor.

The word “bias” in a neural-network intercept is unrelated to estimator bias across repeated datasets.

## Match the loss to the output

| Loss | Prediction and response | Population interpretation |
|---|---|---|
| $\frac12(a-y)^2$ | Real prediction and response | Conditional mean |
| $|a-y|$ | Real prediction and response | A conditional median |
| $\mathbf1\{ya\le0\}$ | Real score; signed label $y\in\{-1,1\}$ | Classification error with a stated tie rule |
| $-y\log p-(1-y)\log(1-p)$ | Probability $p\in(0,1)$; $y\in\{0,1\}$ | Conditional class probability |
| $\log(1+e^{-ya})$ | Real score; signed label | Logistic form of binary cross-entropy |

Huber loss uses $r=a-y$: $\frac12r^2$ for $|r|\le\delta$ and $\delta|r|-\frac12\delta^2$ outside. Its derivative stops growing with residual magnitude after the threshold. That reduces individual outlier influence relative to squared loss, without implying immunity to all contamination.

Only the classification loss above is automatically bounded. Do not apply a bounded-loss concentration inequality to unrestricted squared loss without extra assumptions.

## Population versus empirical risk

For a fresh observation $Z=(X,Y)\sim P$,

$$L(f)=\mathbb E_P[\ell(f(X),Y)],\qquad
\widehat L_n(f)=\frac1n\sum_{i=1}^n\ell(f(X_i),Y_i).$$

The empirical measure $P_n=\frac1n\sum_i\delta_{Z_i}$ places mass $1/n$ on each observation. It makes $\widehat L_n(f)$ an expectation under a random distribution. The optimizer can evaluate that quantity; it cannot directly evaluate the unknown population expectation.

For a fixed predictor independent of the data, an empirical average may estimate its population loss without bias. After selecting the predictor on the same observations, that independence is lost. This is the central reason training performance is not a substitute for test performance.

Training data fit ordinary parameters. Validation data select procedures or hyperparameters. An untouched test set evaluates the frozen procedure. Repeatedly using test results to change the model turns that set into validation data.

## The three sources of excess risk

Let $\mathcal F\subseteq\mathcal F_{\rm all}$ be the chosen representation and define

$$L^\star=\inf_{f\in\mathcal F_{\rm all}}L(f),\quad
L^\star_{\mathcal F}=\inf_{f\in\mathcal F}L(f),\quad
D_n(\mathcal F)=\sup_{f\in\mathcal F}|L(f)-\widehat L_n(f)|.$$

Suppose the algorithm returns $\widetilde f\in\mathcal F$ with empirical optimization gap at most $\epsilon_{\rm opt}$. For any comparator $f\in\mathcal F$,

$$
\begin{aligned}
L(\widetilde f)
&\le\widehat L_n(\widetilde f)+D_n\\
&\le\inf_{g\in\mathcal F}\widehat L_n(g)+\epsilon_{\rm opt}+D_n\\
&\le\widehat L_n(f)+\epsilon_{\rm opt}+D_n\\
&\le L(f)+\epsilon_{\rm opt}+2D_n.
\end{aligned}
$$

Take the infimum over $f$ and subtract $L^\star$:

$$L(\widetilde f)-L^\star\le
\underbrace{L^\star_{\mathcal F}-L^\star}_{\text{approximation}}
+\underbrace{2D_n(\mathcal F)}_{\text{statistical comparison}}
+\underbrace{\epsilon_{\rm opt}}_{\text{computation}}.$$

The factor two comes from crossing from population to empirical loss and then back. This is an upper bound, not an identity assigning three observable error bars. A uniform deviation can be infinite for a badly controlled class, making the bound valid but useless. A pointwise law of large numbers for one fixed $f$ does not automatically bound the supremum over a data-selected class.

## A small example

If $m(x)=x^2$ but the class contains only lines, training forever cannot eliminate representation error. Moving to quadratics may reduce that error. With a tiny sample, a still larger polynomial class can interpolate but behave poorly between observations. Within any chosen class, stopping the numerical solver early contributes optimization error. These are separate interventions, not three names for the same issue.

**Checkpoint:** Why does the proof use an arbitrary comparator rather than assume a population minimizer exists? Attempt E01–E02 in [the problem set](../exercises/problems.md). Exit skill: reproduce the four inequalities and identify which assumption justifies each.
