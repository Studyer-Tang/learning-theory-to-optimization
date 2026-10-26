# 机器学习与强化学习笔记

[仓库首页](../../README.md) · [符号](../notation.md) · [证明索引](../../study/PROOF_PLAYBOOK.md) · [速率表](../../study/RATE_SHEET.md)

当前内容集中在统计学习与优化：从总体风险与线性回归出发，讨论插值、梯度算法、SGD 的统计推断，再回到有限样本的一致收敛。强化学习部分尚未展开。

## 正文

| 篇目 | 主要内容 |
|---|---|
| [01 · 风险与线性模型](01-risk-and-linear-models.md) | 条件均值、三项误差界、偏差—方差、岭回归及零空间偏差 |
| [02 · 双下降](02-double-descent.md) | 高斯随机设计、最小范数插值、随机投影、逆 Gram 矩与临界维数 |
| [03 · 梯度下降](03-gradient-descent.md) | 精确谱递推、下降引理、梯度支配、强凸收缩与凸目标求和 |
| [04 · SGD 有限时间界](04-sgd-finite-time.md) | 条件无偏、小批量方差、噪声邻域、递减步长、算术及加权平均 |
| [05 · SGD 渐近分布](05-sgd-asymptotics.md) | 最后迭代 CLT、矩阵传播权重、Lyapunov 方程、PR 平均 |
| [06 · SGD 统计推断](06-sgd-inference.md) | 代入估计、bootstrap 前提、分块均值、自归一化与置信区间 |
| [07 · 集中不等式](07-concentration.md) | Hoeffding、McDiarmid、Bernstein、偶数矩、指数矩与随机矩阵 |
| [08 · 一致收敛](08-uniform-convergence.md) | 覆盖数、VC 维、Sauer–Shelah、幽灵样本与 Rademacher 复杂度 |
| [09 · 动量与自适应方法](09-momentum-and-adaptation.md) | 二次 heavy-ball、Adam 偏差修正、AdamW、Muon 与极分解迭代 |

01–02 对应课程第一章，03–06 与 09 对应第二章，07–08 对应第三章。03、04、09 中保留了课程第二章的公式编号；其余篇目按各篇编号。分块均值部分给出了一组便于证明的较强充分条件，并明确区分单条轨迹的分块与梯度小批量。

[线性代数工具](linear-algebra.md) 汇集迹运算、条件二次型、投影与 SVD 的计算。收敛方式、随机阶与 delta 方法可查阅现有的[概率工具篇](../10-probability.md)。若只需短篇回顾，可从[英文阅读路线](../../README.md#reading-path)进入。

## 习题

每篇正文末尾附少量练习与核对线索。两个较完整的解答集如下：

- [作业 4](../../exercises/zh/homework04.md)：5 道大题、15 个小问，集中界、尾积分、最大值、重尾截断与覆盖数。
- [作业 5](../../exercises/zh/homework05.md)：4 道大题、15 个小问，VC 构造、函数类运算及范数约束网络的复杂度。
- [原有练习](../../exercises/problems.md)及[解答](../../exercises/solutions.md)：从一维计算到完整证明。

推导按结论的实际需要保留条件。鞅差 CLT、函数型 CLT 等概率基础定理在使用处明确列出；正文展开验证所需条件与余项，不另行重建整套概率论。数值实验见 [labs](../../labs/README.md)，用于检验具体计算及展示有限样本现象。

整理范围与来源版本见[来源说明](../../SOURCES.md)和[章节对照](../../reference/SECTION_MAP.md)。阅读记录见[学习记录](../../study/LEARNING_LOG.md)；解答已写入笔记，不代表已完成独立作答。
