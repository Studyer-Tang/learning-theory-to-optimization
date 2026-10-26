# A proof-selection playbook

[Home](../README.md) · [Exercises](../exercises/problems.md)

## 中文证明索引

| 结论或计算 | 完整推导所在篇目 |
|---|---|
| 条件均值、三项误差界、偏差—方差与岭回归 SVD | [风险与线性模型](../notes/zh/01-risk-and-linear-models.md) |
| 随机投影、Schur 补、逆卡方矩及两侧风险 | [双下降](../notes/zh/02-double-descent.md) |
| 二次目标谱递推、下降引理与两种 GD 速率 | [梯度下降](../notes/zh/03-gradient-descent.md) |
| 条件小批量方差、递减步长、两种加权求和 | [SGD 有限时间界](../notes/zh/04-sgd-finite-time.md) |
| 传播权重、协方差积分、Lindeberg、Lyapunov 与 PR | [SGD 渐近分布](../notes/zh/05-sgd-asymptotics.md) |
| 分块余项、联合路径极限与自归一化区间 | [SGD 统计推断](../notes/zh/06-sgd-inference.md) |
| 指数矩、McDiarmid 条件区间、偶数矩与双网 | [集中不等式](../notes/zh/07-concentration.md) |
| 覆盖数体积、Sauer–Shelah、幽灵样本、对称化及收缩 | [一致收敛](../notes/zh/08-uniform-convergence.md) |
| Heavy-ball 稳定性与最优根半径、经典极分解 | [动量与自适应方法](../notes/zh/09-momentum-and-adaptation.md) |
| 迹、条件二次型、行空间/零空间正交与 SVD | [线性代数工具](../notes/zh/linear-algebra.md) |

“完整推导”指这些学习结论的条件与计算已写出；鞅差 CLT、函数型 CLT 等基础定理仍在使用处作为明确输入，不声称从公理重新证明。数值检验不替代数学证明。

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

This is a practice method, not a prediction about a particular exam. The original E01–E20 exercises are training problems. The later Chinese Homework 4–5 solution sets are attributed separately and do not claim to be official course answers. No exam specification is inferred.
