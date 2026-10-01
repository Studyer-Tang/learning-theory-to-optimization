# A proof-selection playbook

[Home](../README.md) · [Exercises](../exercises/problems.md)

## Read the question before selecting a theorem

Write five labels: objective, geometry, update, requested output, and requested metric. Then identify whether the randomness is in the dataset, in the optimizer, or both. A theorem that matches four of these labels may still be unusable.

| Question pattern | First line to write | Main tool | Common missing condition |
|---|---|---|---|
| Solve least squares | $A^\top(A\theta-b)=0$ | Rank and pseudoinverse | Invertibility or a selection rule |
| Stable GD step on a quadratic | $e^+=(I-\eta H)e$ | Eigenvalues | Strict boundary; null-space initialization |
| Strongly convex GD rate | $\Phi^+\le\Phi-\eta\Vert g\Vert^2/2$ | Gradient domination | Smoothness and a permitted step |
| Convex GD rate | Expand $\Vert e-\eta g\Vert^2$ | Telescoping and descent | Attained minimum; final gap monotonicity |
| SGD noise floor | Condition the smoothness inequality | Zero conditional cross term | Noise bound, not independent iterates |
| Convex SGD average | $2\eta_t\mathbb E\Delta_t\le d_t-d_{t+1}+\eta_t^2G^2$ | Sum, then Jensen | Correct averaging weights |
| A transformed CLT | First-order Taylor expansion | Tightness and Slutsky | Differentiability and a negligible remainder |
| Heavy-ball stability | Characteristic polynomial | Roots inside the unit disk | Quadratic structure |
| Adam rate claim | Expand the alignment term | Inspect adaptive dependence | Cannot replace a correlated direction by the gradient |

## Four algebra moves to practice

**Complete a square.** To minimize $g^\top d+\mu\|d\|^2/2$, write $\mu\|d+g/\mu\|^2/2-\|g\|^2/(2\mu)$. This turns strong convexity into gradient domination.

**Expand a distance.** $\|e-\eta g\|^2=\|e\|^2-2\eta g^\top e+\eta^2\|g\|^2$. This is the workhorse for convex GD and SGD. Do not discard the last term without an inequality that controls it.

**Unroll a recursion.** If $u_{t+1}\le ru_t+c$, then $u_T\le r^{T-1}u_1+c(1-r^{T-1})/(1-r)$ for $0\le r<1$. Identify a transient and a residual scale before introducing big-O notation.

**Telescope.** Sum $d_t-d_{t+1}$ before bounding terms individually. Bounding both distances separately destroys the cancellation that produces the rate.

## A four-pass practice session

First try the problem without notes. Then consult only the hint. Next compare the decisive inequality with the solution. Finally change one condition and explain which step fails. Record the changed problem, not merely “reviewed theorem.”

This is a practice method, not a prediction about a particular exam. No actual homework sheet or exam specification is included in the source, so these exercises are original training problems rather than official course answers.
