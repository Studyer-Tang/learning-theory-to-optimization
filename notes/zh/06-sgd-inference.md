# 06 · SGD 的协方差估计、分块与自归一化

[中文目录](README.md) · [English](../../README.md)

沿用[上一篇](05-sgd-asymptotics.md)的固定维数模型。取 $\eta_t=at^{-\alpha}$、$1/2<\alpha<1$，并设

$$
\sqrt T(\bar\theta_T-\theta^*)\Rightarrow N(0,\Sigma),
\qquad \Sigma=H^{-1}SH^{-\top},
\qquad \bar\theta_T=\frac1T\sum_{t=1}^T\theta_t.
$$

本篇处理公式中的未知尺度。代入法与分块均值法试图一致估计 $\Sigma$；bootstrap 近似整个误差分布；自归一化则让相同的未知尺度在分子与分母中抵消。它们各自需要的条件不同。

## 1. 从协方差一致性到区间

固定非零向量 $c$，记 $u^*=c^\top\theta^*$、$\bar u_T=c^\top\bar\theta_T$、$v=c^\top\Sigma c>0$。若 $\widehat\Sigma_T\xrightarrow{p}\Sigma$，则

$$
\widehat v_T=c^\top\widehat\Sigma_Tc\xrightarrow{p}v,
\qquad \sqrt{\widehat v_T}\xrightarrow{p}\sqrt v>0.
$$

于是由线性映射及 Slutsky 定理，

$$
\frac{\sqrt T(\bar u_T-u^*)}{\sqrt{\widehat v_T}}
\Rightarrow N(0,1).
$$

令 $z_q$ 是标准正态分布的 $q$ 分位数。正态分布在分位点连续，故

$$
\Pr\left\{-z_{1-\delta/2}\le
\frac{\sqrt T(\bar u_T-u^*)}{\sqrt{\widehat v_T}}
\le z_{1-\delta/2}\right\}\longrightarrow1-\delta.
$$

对上述事件移项，得到区间

$$
\boxed{
\bar u_T\ \pm\ z_{1-\delta/2}\sqrt{\frac{\widehat v_T}{T}}.
}
$$

这是一条渐近覆盖率结论。它既没有给出有限样本覆盖率误差，也不适用于 $v=0$ 的退化方向；后一情形需要另找尺度。将 $T$ 改成训练集样本量 $n$，也必须先说明二者在具体数据机制中的关系。

对于 $a/t$ 步长的**最后一次迭代**，应估计上一篇中的 $V_a$，而非直接使用 $\Sigma$。若 $\widehat H_T\to H$、$\widehat S_T\to S$，可解

$$
(a\widehat H_T-I/2)\widehat V_T
+\widehat V_T(a\widehat H_T-I/2)^\top=a^2\widehat S_T.
$$

在 $a\lambda_{\min}(H)>1/2$ 下，估计矩阵最终以概率趋于 1 保持稳定，Lyapunov 线性算子的逆连续，因而 $\widehat V_T\to V_a$。区间的中心改为 $c^\top\theta_T$，方差改为 $c^\top\widehat V_Tc$。

## 2. 代入式协方差及其大数定律

设 $Z_{t+1}$ 为独立在线观测，损失二阶可微，

$$
g_{t+1}=\nabla_\theta\ell(\theta_t;Z_{t+1}),
\qquad J_{t+1}=\nabla_\theta^2\ell(\theta_t;Z_{t+1}).
$$

沿轨迹累计

$$
\widehat H_T=\frac1T\sum_{t=1}^TJ_{t+1},
\qquad \widehat S_T=\frac1T\sum_{t=1}^Tg_{t+1}g_{t+1}^\top,
$$

再令

$$
\boxed{
\widehat\Sigma_T^{\rm plug}
=\widehat H_T^{-1}\widehat S_T\widehat H_T^{-\top}.
}
$$

最优点处 $Eg(\theta^*;Z)=0$，所以 $S=E[g(\theta^*;Z)g(\theta^*;Z)^\top]$。沿轨迹的梯度不必均值为零；其均值趋于零所带来的附加二阶矩也必须在大数定律中控制。

下面给出足以推出一致性的一组条件。对 $q_{t+1}=J_{t+1}$ 和 $q_{t+1}=g_{t+1}g_{t+1}^\top$ 分别要求

$$
\sup_tE\|q_{t+1}-E[q_{t+1}\mid\mathcal F_t]\|_F^2<\infty,
$$

以及

$$
E\|E[J_{t+1}\mid\mathcal F_t]-H\|_F\to0,
\qquad
E\|E[g_{t+1}g_{t+1}^\top\mid\mathcal F_t]-S\|_F\to0.
$$

这些条件可由独立新样本、条件均值对参数的连续性、轨迹收敛与适当的一致可积性验证。对梯度外积的平方可积控制通常要求梯度四阶矩，不能只检查梯度二阶矩。

为证明大数定律，写 $q_{t+1}=m_t+d_{t+1}$，其中 $m_t=E[q_{t+1}\mid\mathcal F_t]$、$d_{t+1}$ 为矩阵鞅差。将矩阵看成有限维向量，不同时刻的鞅差正交，因而

$$
E\left\|\frac1T\sum_{t=1}^Td_{t+1}\right\|_F^2
=\frac1{T^2}\sum_{t=1}^TE\|d_{t+1}\|_F^2
\le\frac CT\to0.
$$

若目标均值为 $q^*$，另一部分满足

$$
E\left\|\frac1T\sum_{t=1}^Tm_t-q^*\right\|_F
\le\frac1T\sum_{t=1}^TE\|m_t-q^*\|_F\to0.
$$

所以 $\widehat H_T\to H$、$\widehat S_T\to S$ 依概率成立。由于 $H$ 非奇异，求逆与矩阵乘积在此处连续，便有 $\widehat\Sigma_T^{\rm plug}\to\Sigma$。

实际只需一个方向的方差时，可解线性方程 $\widehat H_T^\top w=c$，然后计算 $w^\top\widehat S_Tw$，无需显式形成逆矩阵。有限样本下若采用特征值截断，应保证截断阈值最终小于真实最小特征值，避免改变要估计的极限矩阵。

## 3. 在线乘子 bootstrap：需要证明到哪一步

在同一份观测流上，独立产生权重 $U_t^{(b)}$，满足

$$
U_t^{(b)}\ge0,\qquad EU_t^{(b)}=1,\qquad
\operatorname{Var}(U_t^{(b)})=1,
$$

并维护额外轨迹

$$
\theta_{t+1}^{*(b)}=\theta_t^{*(b)}
-\eta_tU_{t+1}^{(b)}\nabla\ell(\theta_t^{*(b)};Z_{t+1}),
\qquad \bar\theta_T^{*(b)}=\frac1T\sum_{t=1}^T\theta_t^{*(b)}.
$$

所有副本从与原轨迹相同的初值开始。均值为 1 的指数权重或 Poisson(1) 权重满足前两个矩要求。权重应与观测流独立，各副本也使用独立权重。仅凭 $EU=\operatorname{Var}(U)=1$，尚不能断言这个非线性递推的 bootstrap 有效。

令 $\psi_{t+1}=\nabla\ell(\theta^*;Z_{t+1})$。以下给出能核对证明责任的充分路线。首先，原估计量需要有表示

$$
\sqrt T(\bar\theta_T-\theta^*)
=-H^{-1}\frac1{\sqrt T}\sum_{t=1}^T\psi_{t+1}+o_p(1).
$$

其次，扰动轨迹需满足对应的**条件渐近线性表示**

$$
\sqrt T(\bar\theta_T^*-\bar\theta_T)
=-H^{-1}\frac1{\sqrt T}\sum_{t=1}^T(U_{t+1}-1)\psi_{t+1}
+r_T^*,
$$

其中对任意 $\varepsilon,\gamma>0$，

$$
\Pr\{\Pr^*(\|r_T^*\|>\varepsilon)>\gamma\}\to0.
$$

$\Pr^*$ 表示给定全部原始观测后，只对乘子随机性取概率。第一条表示与第二条表示都需要模型正则性、矩界、迭代稳定性和余项控制。上一篇的普通 SGD 证明不能未经修改就用于扰动轨迹，因为随机权重也会改变它的线性恢复项。

假设这两条表示已经成立，$\psi_t$ 独立同分布、$E\psi_t=0$、$E\psi_t\psi_t^\top=S$、$E\|\psi_t\|^{2+\rho}<\infty$，并且 $E|U-1|^{2+\rho}<\infty$，其中 $\rho>0$。下面可以完整验证线性乘子和的条件极限。

给定原始观测，各项 $(U_{t+1}-1)\psi_{t+1}/\sqrt T$ 独立、均值零，其条件协方差和为

$$
\frac1T\sum_{t=1}^T\psi_{t+1}\psi_{t+1}^\top\xrightarrow{p}S.
$$

对任意 $\varepsilon>0$，条件 Lindeberg 和不超过

$$
\begin{aligned}
&\varepsilon^{-\rho}T^{-1-\rho/2}
\sum_{t=1}^TE|U-1|^{2+\rho}\|\psi_{t+1}\|^{2+\rho}\\
&\qquad=C\varepsilon^{-\rho}T^{-\rho/2}
\left(\frac1T\sum_{t=1}^T\|\psi_{t+1}\|^{2+\rho}\right)
\xrightarrow{p}0.
\end{aligned}
$$

独立三角阵 CLT 的条件版本及条件 Slutsky 定理于是给出

$$
\mathcal L^*\!\left(\sqrt T(\bar\theta_T^*-\bar\theta_T)\right)
\Rightarrow N(0,\Sigma)
\quad\text{依概率成立}.
$$

此处调用的条件 CLT 是外部概率工具；上述证明补齐的是它在乘子线性和上的方差与 Lindeberg 检验。完整在线 bootstrap 有效性还包括前面明确列出的条件渐近线性表示。原始方法的模型条件与证明可参阅 Fang、Xu、Yang 的 [Online Bootstrap Confidence Intervals for the Stochastic Gradient Descent Estimator](https://www.jmlr.org/papers/v19/17-370.html)。

令 $q_{\gamma,T}^*$ 为 $\sqrt T\,c^\top(\bar\theta_T^*-\bar\theta_T)$ 的条件 $\gamma$ 分位数，则基本型区间为

$$
\boxed{
\left[\bar u_T-\frac{q_{1-\delta/2,T}^*}{\sqrt T},\quad
\bar u_T-\frac{q_{\delta/2,T}^*}{\sqrt T}\right].
}
$$

分位点的顺序在移项后反转。有限个副本只能估计条件分位数，另有 Monte Carlo 误差；若要把这层误差也在渐近结论中消去，副本数还需增长。把相关的 $\theta_t$ 当作独立样本直接重抽，不等于上述过程。

## 4. 分块均值：估计量与一组可证明的充分条件

这里的 batch 指一段**连续参数迭代**，用于方差估计；它不是计算随机梯度的 mini-batch。丢弃初始阶段后，将保留轨迹分为不重叠的块 $\mathcal B_1,\ldots,\mathcal B_M$，块长为 $m_k$。设

$$
\widetilde\theta_k=\frac1{m_k}\sum_{t\in\mathcal B_k}\theta_t,
\qquad
\widetilde\theta=\frac{\sum_{k=1}^Mm_k\widetilde\theta_k}{\sum_{k=1}^Mm_k}.
$$

加权分块估计量为

$$
\boxed{
\widehat\Sigma_T^{\rm BM}
=\frac1M\sum_{k=1}^Mm_k
(\widetilde\theta_k-\widetilde\theta)
(\widetilde\theta_k-\widetilde\theta)^\top.
}
$$

长块的均值若有协方差约为 $\Sigma/m_k$，乘上 $m_k$ 才能把不同块置于相同尺度。但这个近似本身也需要证明；“块很长”不是一个可以省略速率的假设。

下面证明一个具体方案，使用比一般理论更强的条件，目的在于把所有余项写清楚。这是本笔记的一组充分条件，并不声称与课件或参考论文的条件完全等价。

取步长 $at^{-\alpha}$，$1/2<\alpha<1$。令 $b_T=\lfloor T/2\rfloor$，选整数 $M=M_T$，并定义

$$
m=m_T=\left\lfloor\frac{T-b_T}{M}\right\rfloor,
\qquad \mathcal B_k=\{b_T+(k-1)m+1,\ldots,b_T+km\},
\quad 1\le k\le M.
$$

假设

$$
M\to\infty,\qquad M=o(T^{1-\alpha}).
$$

这样所有块严格等长且不重叠，实际使用 $N_T=Mm$ 个迭代；末尾丢弃 $T-b_T-N_T<M$ 个迭代。由于 $M=o(T)$，有 $N_T/T\to1/2$、$m\asymp T/M$，不会把不整除的尾部误当成一个完整块。下文的 $\widetilde\theta$ 只对这 $N_T$ 个保留迭代平均。

假设误差递推可以写为

$$
e_{t+1}=e_t-\eta_t(He_t+\psi_{t+1}+\Delta_{t+1}+r_t),
$$

并满足

$$
E\|e_t\|^2\le Ct^{-\alpha},\quad
E\|r_t\|^2\le Ct^{-2\alpha},\quad
E[\Delta_{t+1}\mid\mathcal F_t]=0,\quad
E\|\Delta_{t+1}\|^2\le Ct^{-\alpha}.
$$

其中 $\psi_t$ 独立同分布、$E\psi_t=0$、$E\psi_t\psi_t^\top=S$、$E\|\psi_t\|^4<\infty$。在线独立数据下可取 $\psi_{t+1}=g(\theta^*;Z_{t+1})$，$\Delta_{t+1}=\xi_{t+1}-\psi_{t+1}$。若噪声关于参数在 $L^2$ 意义下 Lipschitz，则均方速率给出 $\Delta$ 的界；若 $\|r_t\|\le C\|e_t\|^2$ 且 $E\|e_t\|^4\le Ct^{-2\alpha}$，则给出 $r_t$ 的二阶矩界。这些比上一篇 PR 终点 CLT 所用条件更强。

对任一块 $\mathcal B_k=\{l,\ldots,u\}$，从递推移项并求和，得到

$$
Y_k:=\sqrt m(\widetilde\theta_k-\theta^*)
=-H^{-1}\frac1{\sqrt m}\sum_{t=l}^u\psi_{t+1}+\varepsilon_k,
$$

其中

$$
\varepsilon_k=-\frac{H^{-1}}{\sqrt m}
\left[D_{l,u}+\sum_{t=l}^u\Delta_{t+1}+\sum_{t=l}^ur_t\right],
$$

$$
D_{l,u}=\frac{e_{u+1}}{\eta_u}-\frac{e_l}{\eta_l}
+\sum_{t=l+1}^ue_t(\eta_{t-1}^{-1}-\eta_t^{-1}).
$$

所有保留时刻均满足 $t\asymp T$。用 $\|X\|_{L^2}=(E\|X\|^2)^{1/2}$ 表示随机向量的二阶范数，Minkowski 不等式给

$$
\begin{aligned}
\|D_{l,u}\|_{L^2}
&\le CT^{\alpha/2}+CT^{\alpha/2}
+C\sum_{t=l+1}^uT^{-\alpha/2}T^{\alpha-1}\\
&\le CT^{\alpha/2}+CmT^{\alpha/2-1}
\le CT^{\alpha/2}.
\end{aligned}
$$

$\Delta_{t+1}$ 是鞅差，不同时刻正交，因此

$$
E\left\|\frac1{\sqrt m}\sum_{t=l}^u\Delta_{t+1}\right\|^2
=\frac1m\sum_{t=l}^uE\|\Delta_{t+1}\|^2
\le CT^{-\alpha}.
$$

对非线性项不需要独立性，直接用 Minkowski：

$$
\left\|\frac1{\sqrt m}\sum_{t=l}^ur_t\right\|_{L^2}
\le C\sqrt m\,T^{-\alpha}.
$$

合并得一致于所有块的界

$$
\boxed{
\max_{k\le M}E\|\varepsilon_k\|^2
\le C\left(\frac{T^\alpha}{m}+T^{-\alpha}+mT^{-2\alpha}\right)
=:\rho_T.
}
$$

代入 $m\asymp T/M$，第一项为 $O(MT^{\alpha-1})\to0$，第二项显然趋于零，第三项为 $O(T^{1-2\alpha}/M)\to0$。因此 $\rho_T\to0$。这一步解释了块长必须显著大于后期局部相关尺度 $T^\alpha$，即 $m/T^\alpha\to\infty$。

令

$$
Z_k=-H^{-1}\frac1{\sqrt m}\sum_{t\in\mathcal B_k}\psi_{t+1}.
$$

不重叠块中的 $Z_k$ 独立，$EZ_k=0$、$EZ_kZ_k^\top=\Sigma$。有限四阶矩保证 $\sup_{T,k}E\|Z_k\|^4<\infty$：逐坐标展开四次方，独立性与均值零使非配对项消失，剩下 $m$ 个四阶项与 $O(m^2)$ 个二阶配对项，除以 $m^2$ 后有界。故对外积的每个元素使用方差界，得到

$$
E\left\|\frac1M\sum_{k=1}^MZ_kZ_k^\top-\Sigma\right\|_F^2
\le C/M\to0.
$$

另一方面，Markov 不等式给

$$
\frac1M\sum_k\|\varepsilon_k\|^2\xrightarrow{p}0,
\qquad \frac1M\sum_k\|Z_k\|^2=O_p(1).
$$

因为 $Y_k=Z_k+\varepsilon_k$，Cauchy–Schwarz 给出

$$
\begin{aligned}
&\left\|\frac1M\sum_k(Y_kY_k^\top-Z_kZ_k^\top)\right\|\\
&\quad\le2\left(\frac1M\sum_k\|Z_k\|^2\right)^{1/2}
\left(\frac1M\sum_k\|\varepsilon_k\|^2\right)^{1/2}
+\frac1M\sum_k\|\varepsilon_k\|^2
\xrightarrow{p}0.
\end{aligned}
$$

最后必须检查用总体块均值代替 $\theta^*$ 所引入的中心化。等长块时

$$
\widehat\Sigma_T^{\rm BM}
=\frac1M\sum_kY_kY_k^\top-\bar Y\bar Y^\top,
\qquad \bar Y=\frac1M\sum_kY_k.
$$

独立性给 $E\|M^{-1}\sum_kZ_k\|^2=O(1/M)$，而 Jensen 不等式给

$$
E\left\|\frac1M\sum_k\varepsilon_k\right\|^2
\le\frac1M\sum_kE\|\varepsilon_k\|^2\le\rho_T\to0.
$$

因此 $\bar Y\to0$ 依概率，中心化项消失。综上，

$$
\boxed{\widehat\Sigma_T^{\rm BM}\xrightarrow{p}\Sigma.}
$$

该证明估计的是同一个 PR 渐近协方差，尽管为估计协方差只使用了后半段迭代；在完整平均的 CLT 已成立时，可以把它代入第 1 节的区间。这个方案需要预先给定运行长度，不声称它是一种无需调整的全在线算法。

原课件还给出累计步长分块方案：

$$
e_k=\left\lfloor((k+1)D)^{1/(1-\alpha)}\right\rfloor,
\qquad \mathcal B_k=\{e_{k-1}+1,\ldots,e_k\}.
$$

忽略取整后，

$$
\sum_{t\in\mathcal B_k}\eta_t
\approx\frac a{1-\alpha}(e_k^{1-\alpha}-e_{k-1}^{1-\alpha})
\approx\frac{aD}{1-\alpha}.
$$

所以这些块覆盖近似相同的算法时间。取 $D\asymp T^{1-\alpha}/(M+1)$ 可使末块约在 $T$ 结束。要进一步证明该不等长方案一致，仍须针对每块的线性表示余项、初始块和中心化分别给界，不能只写 $M,D\to\infty$ 就宣布证明完成。相关更一般的结果见 Chen、Lee、Tong、Zhang 的 [Statistical Inference for Model Parameters in Stochastic Gradient Descent](https://arxiv.org/abs/1610.08637)。

固定块数也不够。即使各块已经完全独立、$Y_k\sim N(0,\sigma^2)$，固定 $M$ 时仍有

$$
\frac1M\sum_{k=1}^M(Y_k-\bar Y)^2
\ \overset{d}{=}\ \frac{\sigma^2}{M}\chi^2_{M-1},
$$

其随机性不会随块长增加而消失。因此有限个“很长的块”不能自动产生一致协方差估计。相反，固定很短的块虽然数量很多，也可能没有消去 SGD 的时间相关性。

## 5. 自归一化：从部分和路径推到区间

定义所有前缀平均及随机归一化矩阵

$$
\bar\theta_s=\frac1s\sum_{t=1}^s\theta_t,
\qquad
\widehat R_T=\frac1{T^2}\sum_{s=1}^Ts^2
(\bar\theta_s-\bar\theta_T)(\bar\theta_s-\bar\theta_T)^\top.
$$

这里只要求观察参数轨迹，不需要 Hessian 或 bootstrap 副本。但论证需要比终点 CLT 更强的前提：在 $D([0,1],\mathbb R^d)$ 的 Skorokhod $J_1$ 拓扑下，

$$
U_T(r):=\frac1{\sqrt T}\sum_{t=1}^{\lfloor Tr\rfloor}(\theta_t-\theta^*)
\Rightarrow\Sigma^{1/2}W_d(r),\qquad0\le r\le1,
$$

其中 $W_d$ 为标准 $d$ 维布朗运动。**本节以这条函数型 CLT 为前提**，证明其后的归一化结论；不把上一篇只在 $r=1$ 处的 CLT 当成整条路径的证明。

一种证明函数型 CLT 的路线是，另行验证一致的渐近线性表示

$$
\sup_{0\le r\le1}\left\|
\frac1{\sqrt T}\sum_{t=1}^{\lfloor Tr\rfloor}e_t
+H^{-1}\frac1{\sqrt T}\sum_{t=1}^{\lfloor Tr\rfloor}\xi_{t+1}
\right\|\xrightarrow{p}0,
$$

再对噪声部分和调用鞅的函数型 CLT。这里需要的最大部分和界、紧性和路径余项控制比固定终点的界更强；它们应作为单独的证明任务，不能用“同理可得”略过。相关 SGD 路径条件可参考 [Fast and Robust Online Inference with Stochastic Gradient Descent via Random Scaling](https://arxiv.org/abs/2106.03156)。

现在固定 $c$，令

$$
P_s=\sum_{t=1}^sc^\top(\theta_t-\theta^*),\qquad
\nu_T(r)=P_{\lfloor Tr\rfloor}/\sqrt T.
$$

投影后的函数型 CLT 为 $\nu_T\Rightarrow\sqrt v\,W$，其中 $v=c^\top\Sigma c>0$，$W$ 是一维标准布朗运动。终点正好是

$$
\nu_T(1)=\sqrt T\,c^\top(\bar\theta_T-\theta^*).
$$

分母也能写成同一条路径的函数。直接展开，

$$
\begin{aligned}
s\,c^\top(\bar\theta_s-\bar\theta_T)
&=\sum_{t=1}^sc^\top\theta_t-\frac sT\sum_{t=1}^Tc^\top\theta_t\\
&=P_s-\frac sT P_T.
\end{aligned}
$$

未知的 $\theta^*$ 在右边抵消，因为 $-sc^\top\theta^*+(s/T)Tc^\top\theta^*=0$。因此有精确恒等式

$$
\begin{aligned}
\widehat r_T:=c^\top\widehat R_Tc
&=\frac1{T^2}\sum_{s=1}^T(P_s-sP_T/T)^2\\
&=\frac1T\sum_{s=1}^T
[\nu_T(s/T)-(s/T)\nu_T(1)]^2.
\end{aligned}
$$

接下来证明它确实收敛到路径积分，而不只停留在形式类比。假设确定路径 $f_T$ 在 $J_1$ 拓扑下收敛到连续路径 $f$。因为极限连续，这种收敛蕴含一致收敛。令

$$
b_T(r)=f_T(r)-rf_T(1),\qquad b(r)=f(r)-rf(1).
$$

则 $\|b_T-b\|_\infty\le2\|f_T-f\|_\infty\to0$，并且 $\|b_T\|_\infty$ 有界。平方和的替换误差满足

$$
\begin{aligned}
&\left|\frac1T\sum_{s=1}^Tb_T(s/T)^2
-\frac1T\sum_{s=1}^Tb(s/T)^2\right|\\
&\qquad\le\|b_T-b\|_\infty
(\|b_T\|_\infty+\|b\|_\infty)\to0.
\end{aligned}
$$

剩下的是真正连续函数的黎曼和，且可以用连续模量明确控制：

$$
\left|\frac1T\sum_{s=1}^Tb(s/T)^2-\int_0^1b(r)^2dr\right|
\le2\|b\|_\infty\,\omega_b(1/T)\to0,
$$

其中 $\omega_b(h)=\sup_{|r-q|\le h}|b(r)-b(q)|$。同时 $f_T(1)\to f(1)$。由连续极限下的扩展连续映射定理，分子与分母联合收敛：

$$
\boxed{
(\nu_T(1),\widehat r_T)
\Rightarrow\left(\sqrt v\,W(1),\quad
v\int_0^1[W(r)-rW(1)]^2dr\right).
}
$$

亦可在 Skorokhod 表示下把收敛过程放在同一概率空间，利用上面的确定路径计算得到几乎处处联合收敛，再转回原分布。这里调用表示定理或扩展连续映射定理均属于外部概率工具；黎曼和误差已在上面给出。

记布朗桥 $B(r)=W(r)-rW(1)$，并令

$$
A=\int_0^1B(r)^2\,dr.
$$

首先 $A<\infty$ 几乎处处，因为布朗路径连续。其次 $A>0$ 几乎处处：若 $A=0$，连续非负函数 $B(r)^2$ 必须处处为零，特别是 $B(1/2)=0$。而

$$
B(1/2)=W(1/2)-\tfrac12W(1)\sim N(0,1/4)
$$

为非退化正态变量，取零的概率为零。因此 $\Pr(A=0)=0$。联合收敛与 $v>0$ 还说明 $\Pr(\widehat r_T=0)\to0$；在有限样本出现零分母时，应将区间记为未定义，而不能擅自除零。

映射 $(x,y)\mapsto x/\sqrt y$ 在 $y>0$ 处连续，故可以再次应用连续映射定理，得到

$$
\begin{aligned}
\frac{\sqrt T\,c^\top(\bar\theta_T-\theta^*)}
{\sqrt{c^\top\widehat R_Tc}}
&\Rightarrow
\frac{\sqrt v\,W(1)}{\sqrt{vA}}\\
&=\boxed{\frac{W(1)}{\sqrt{\int_0^1[W(r)-rW(1)]^2dr}}}
=:T_{\rm SN}.
\end{aligned}
$$

这里消去的是同一个 $v$；分母并没有趋于确定常数。因而这是一个不含未知模型尺度的极限分布，但不是标准正态分布。

其对称性和分位点连续性也可以验证。对任意 $r$，

$$
\operatorname{Cov}(W(1),B(r))=r-r=0.
$$

布朗运动的任意有限维向量为联合 Gaussian，因此 $W(1)$ 独立于任意有限维布朗桥向量；再由连续路径和有理点的稠密性，独立于整条布朗桥，故也独立于 $A$。给定 $A>0$ 后，$W(1)/\sqrt A$ 是中心正态的尺度混合，分布连续、关于零对称，并在任意非空区间赋正概率。因此可以使用其唯一分位数 $\kappa_q$。

它也不可能是标准正态分布。由独立性、非负积分的 Tonelli 定理以及 Jensen 不等式，

$$
E[T_{\rm SN}^2]=E[W(1)^2]E[A^{-1}]
=E[A^{-1}]\ge(EA)^{-1}=6>1,
$$

其中 $EA=1/6$ 的计算见下文，$E[A^{-1}]$ 允许为正无穷。标准正态的二阶矩为 1，与此矛盾。

由

$$
\Pr\{-\kappa_{1-\delta/2}\le T_{\rm SN}\le\kappa_{1-\delta/2}\}=1-\delta,
$$

以及极限分布在边界连续，有限样本统计量落在同一区间的概率趋于 $1-\delta$。对这个事件移项，得到

$$
\boxed{
 c^\top\bar\theta_T
\ \pm\ \kappa_{1-\delta/2}
 \sqrt{\frac{c^\top\widehat R_Tc}{T}}.
}
$$

临界值应由这个布朗运动比值的分布给出，可以一次模拟后重复使用。上述随机尺度论文的[表 2](https://cdn.aaai.org/ojs/20701/20701-13-24714-1-2-20220628.pdf)给出 $\kappa_{0.975}\approx6.747$，对应双侧 95% 区间；不能换成标准正态的 1.96。用错临界值会改变渐近覆盖率。

最后，$\widehat R_T$ **不是** $\Sigma$ 的一致估计。在方向 $c$ 上，其极限为随机变量 $vA$，而非 $v$。例如

$$
EA=\int_0^1r(1-r)\,dr=\frac16.
$$

不仅它的均值尺度不同，极限本身也不退化：Gaussian 四阶矩公式给

$$
\operatorname{Var}(A)
=2\int_0^1\int_0^1[\min(r,q)-rq]^2\,dr\,dq
=\frac1{45}>0.
$$

最后的积分可把单位方形按 $r\le q$ 分成两半：被积函数为 $r^2(1-q)^2$，因此双积分等于
$2\int_0^1(1-q)^2q^3/3\,dq=1/90$，再乘 Gaussian 公式前的 2，得 $1/45$。所以乘以 6 也不会使它变成一致协方差估计。方法成立依赖的是分子与随机分母的**联合**极限。

若只关心固定方向，可以完全在标量序列 $u_t=c^\top\theta_t$ 上累计归一化量。令 $A_s=\sum_{t=1}^su_t$，则

$$
\widehat r_T=\frac1{T^2}\left[
\sum_{s=1}^TA_s^2
-\frac{2A_T}{T}\sum_{s=1}^TsA_s
+\frac{A_T^2}{T^2}\sum_{s=1}^Ts^2\right].
$$

因此只需在线维护 $A_T$、$\sum A_s^2$ 和 $\sum sA_s$；$\sum s^2=T(T+1)(2T+1)/6$ 可直接计算。这个代数递推节省存储，但不会免除函数型 CLT 的理论前提；数值实现中还应注意大数相减带来的精度损失。

## 6. 两个检查题

1. 设同样有终点 CLT，但没有部分和过程的紧性或函数型极限，能否推出上面的自归一化区间？指出证明中具体缺少哪一步。

   *线索：* 终点的分布不决定所有前缀偏差的平方和；无法推出 $(\nu_T(1),\widehat r_T)$ 的联合极限。仅对终点使用 Slutsky 也无效，因为分母没有依概率收敛到常数。

2. 对第 4 节的充分条件，取 $\alpha=3/4$、$M=\lfloor T^\gamma\rfloor$。求允许的 $\gamma$ 范围，并逐项检查 $\rho_T$ 的三个幂次。

   *线索：* 需 $0<\gamma<1/4$；三项分别为 $T^{\gamma-1/4}$、$T^{-3/4}$、$T^{-1/2-\gamma}$。块数增加过快时，第一个边界余项不会消失。
