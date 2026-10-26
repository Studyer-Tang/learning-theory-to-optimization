# 04 · SGD 的有限时间界

[中文目录](README.md) · [上一章](03-gradient-descent.md) · [下一章](05-sgd-asymptotics.md)

设 $\theta_{t+1}=\theta_t-\eta_tg_t$，$t\ge1$。$\mathcal H_t$ 表示产生本次随机梯度之前的历史，$\theta_t$ 已由该历史确定。以下期望针对算法的抽样；若目标是固定数据的经验风险，期望并不额外平均训练集。

记 $\delta_t=\mathbb E\Phi(\theta_t)-\Phi^*$、$d_t=\mathbb E\|\theta_t-\theta^*\|^2$。$G^2$ 控制完整随机梯度的二阶矩，$\sigma^2$ 控制噪声的二阶矩，两者不可替换。每条速率都同时指定几何条件、步长及输出。有限时间界不等于渐近正态性；后者另见[下一章](05-sgd-asymptotics.md)。

## 2.4 随机梯度下降：噪声、步长与平均

本节各结论的关系是：先研究固定步长留下的噪声影响，再用递减步长控制最后迭代，接着研究返回平均参数的保证；最后去掉强凸性，检查较弱条件下还能保留什么速率。

### 2.4.1 随机梯度与条件于历史的期望

经验目标通常是有限和

$$
\Phi(\theta)=\frac1n\sum_{i=1}^n\phi_i(\theta). \tag{2.4.1}
$$

完整梯度需要处理全部 $n$ 个样本；SGD 用较便宜的随机向量 $g_t$ 替代它：

$$
\theta_{t+1}=\theta_t-\eta_tg_t. \tag{2.4.2}
$$

本节沿用讲义从 $t=1$ 开始的编号。$\mathcal H_t$ 表示抽取当前随机梯度之前的全部历史；当前参数 $\theta_t$ 已由这段历史确定。固定训练集以后，期望主要针对抽样梯度和可能的随机初始化。

$\theta_t$ 依赖过去抽过哪些样本，不能当成与整个随机过程无关的固定常数。条件于 $\mathcal H_t$ 相当于先暂停在当前位置，只对“下一批样本怎么抽”取平均。此时 $\theta_t$ 和 $\nabla\Phi(\theta_t)$ 都固定，无偏性才能在当前点正确使用。不同步骤的 $g_t$ 通常并不独立。

### 2.4.2 小批量的条件均值与协方差

条件于历史，独立、有放回、均匀抽取 $b$ 个索引，令

$$
g_t=\frac1b\sum_{j=1}^b\nabla\phi_{I_{t,j}}(\theta_t). \tag{2.4.3}
$$

则

$$
\mathbb E[g_t\mid\mathcal H_t]=\nabla\Phi(\theta_t), \tag{2.4.4}
$$

$$
\operatorname{Cov}(g_t\mid\mathcal H_t)
=\frac1b\operatorname{Cov}(\nabla\phi_I(\theta_t)\mid\mathcal H_t). \tag{2.4.5}
$$

**无偏性的证明。** 给定历史后，$\theta_t$ 固定。由于 $I$ 对 $1,\ldots,n$ 均匀抽样，

$$
E[\nabla\phi_I(\theta_t)\mid\mathcal H_t]
=\sum_{i=1}^n\frac1n\nabla\phi_i(\theta_t)
=\nabla\Phi(\theta_t).
$$

对批内的 $b$ 项用期望线性性，再除以 $b$，即得式（2.4.4）。

**协方差的证明。** 记 $V_j=\nabla\phi_{I_{t,j}}(\theta_t)$、$m=\nabla\Phi(\theta_t)$。由定义，

$$
\begin{aligned}
\operatorname{Cov}(g_t\mid\mathcal H_t)
&=E[(g_t-m)(g_t-m)^\top\mid\mathcal H_t]\\
&=\frac1{b^2}\sum_{j,k=1}^bE[(V_j-m)(V_k-m)^\top\mid\mathcal H_t].
\end{aligned}
$$

当 $j\ne k$，批内条件独立且 $E[V_j-m\mid\mathcal H_t]=0$，因此交叉项为零；当 $j=k$，每项都是单次抽样的协方差。剩下 $b$ 项，故系数为 $b/b^2=1/b$。这里独立的是同一批内的抽样，不是不同时刻的迭代。

**这个公式用来判断批量的作用。** 批量扩大到四倍，条件协方差降为原来的四分之一，噪声标准差降为一半；代价是本次梯度需要处理更多样本。它没有消除训练集本身的统计不确定性，也不是“批量越大总体训练一定越快”的结论。

抽样方式可以不同，证明真正需要的只有两点：平均方向是否正确，以及单步随机量是否大到无法控制。下面的定义把算法实现概括成这两个可检查的条件。

### 2.4.3 定义 2.11 完整梯度矩界与噪声方差界

$$
\mathbb E[g_t\mid\mathcal H_t]=\nabla\Phi(\theta_t),\qquad
\mathbb E[\|g_t\|^2\mid\mathcal H_t]\le G^2. \tag{2.4.6}
$$

另一种更细的假设将真实梯度与噪声分开：

$$
g_t=\nabla\Phi(\theta_t)+\xi_t,\qquad
\mathbb E[\xi_t\mid\mathcal H_t]=0,\qquad
\mathbb E[\|\xi_t\|^2\mid\mathcal H_t]\le\sigma^2. \tag{2.4.7}
$$

$G^2$ 控制完整随机梯度，$\sigma^2$ 只控制噪声。由条件均值为零，

$$
\mathbb E[\|g_t\|^2\mid\mathcal H_t]
=\|\nabla\Phi(\theta_t)\|^2+\mathbb E[\|\xi_t\|^2\mid\mathcal H_t].
$$

因此前一种假设推出噪声方差不超过 $G^2$，反向一般不成立。这里的矩条件是沿迭代路径使用的；不要额外理解为强凸函数在整个 $\mathbb R^d$ 上梯度都有界。无偏性也只描述梯度估计，不意味着参数或测试误差无偏。

### 2.4.4 定理 2.12：强凸、光滑时的固定步长 SGD

证明继续沿用强凸 GD 的目标间隙递推。先对一次具体的随机梯度应用光滑上界，再取条件期望；这样既保留了平均下降，也不会漏掉平方项中的噪声。

假设 $\Phi$ 为 $\mu$-强凸、$\beta$-光滑且有最小点，噪声满足式（2.4.7）。取

$$
0<\eta\le1/\beta,\qquad\theta_{t+1}=\theta_t-\eta g_t. \tag{2.4.8}
$$

记 $\delta_t=\mathbb E[\Phi(\theta_t)]-\Phi^*$，则

$$
\delta_T\le(1-\mu\eta)^{T-1}\delta_1
+\frac{\beta\eta\sigma^2}{2\mu}\bigl[1-(1-\mu\eta)^{T-1}\bigr], \tag{2.4.9}
$$

从而

$$
\delta_T\le\underbrace{(1-\mu\eta)^{T-1}\delta_1}_{\text{初始误差逐渐消失}}
+\underbrace{\frac{\beta\eta\sigma^2}{2\mu}}_{\text{持续噪声造成的邻域上界}}. \tag{2.4.10}
$$

**证明。** 对每次随机梯度的具体取值，光滑性先给出

$$
\Phi(\theta_{t+1})\le\Phi(\theta_t)-\eta\langle\nabla\Phi(\theta_t),g_t\rangle
+\frac{\beta\eta^2}{2}\|g_t\|^2.
$$

再条件于历史取期望，用上面的二阶矩分解：

$$
\begin{aligned}
\mathbb E[\Phi(\theta_{t+1})\mid\mathcal H_t]
&\le\Phi(\theta_t)-\eta(1-\beta\eta/2)\|\nabla\Phi(\theta_t)\|^2
+\frac{\beta\eta^2\sigma^2}{2}\\
&\le\Phi(\theta_t)-\frac\eta2\|\nabla\Phi(\theta_t)\|^2
+\frac{\beta\eta^2\sigma^2}{2}.
\end{aligned}
$$

强凸性给出 $\|\nabla\Phi(\theta_t)\|^2\ge2\mu(\Phi(\theta_t)-\Phi^*)$。再取全期望：

$$
\delta_{t+1}\le(1-\mu\eta)\delta_t+\frac{\beta\eta^2\sigma^2}{2}. \tag{2.4.11}
$$

令 $q=1-\mu\eta$，逐次代入得到 $\delta_T\le q^{T-1}\delta_1+(\beta\eta^2\sigma^2/2)\sum_{j=0}^{T-2}q^j$。等比求和即为结论。

**含义。** 大步长更快消除初始误差，却增大噪声邻域；小步长则相反。这里是误差的上界，不是严格为正的误差下界，不能仅凭“噪声方差至多为 $\sigma^2$”断言每个问题都存在正的误差平台。下一结果用递减步长让这个上界趋于零。

**一个能精确算出的例子。** 取一维目标 $\Phi(\theta)=h\theta^2/2$，$h>0$，最优点为零；随机梯度为 $h\theta_t+\xi_t$，其中条件均值为零、条件方差恰为 $\sigma^2>0$。更新为

$$
\theta_{t+1}=(1-\eta h)\theta_t-\eta\xi_t.
$$

平方后取期望，交叉项因条件均值为零而消失。令 $d_t=\mathbb E\theta_t^2$，则

$$
d_{t+1}=(1-\eta h)^2d_t+\eta^2\sigma^2.
$$

若 $0<\eta h<2$，等比求和给出

$$
d_t\longrightarrow\frac{\eta^2\sigma^2}{1-(1-\eta h)^2}
=\frac{\eta\sigma^2}{h(2-\eta h)},\qquad
\mathbb E\Phi(\theta_t)\longrightarrow\frac{\eta\sigma^2}{2(2-\eta h)}.
$$

这个例子中，误差确实不会降为零；当步长很小时，目标误差平台与 $\eta$ 近似成正比。更值得注意的是，$\mathbb E\theta_t=(1-\eta h)^{t-1}\mathbb E\theta_1\to0$，但均方误差不趋于零：**平均参数位置正确，不等于每次运行都准确。**

### 2.4.5 命题 2.13：递减步长使最后一次误差达到 $O(1/T)$

上一定理的附加噪声项是 $O(\eta_t^2)$，而用于消除旧误差的项是 $\mu\eta_t\delta_t$。若希望 $\delta_t$ 达到 $1/t$ 量级，选择 $\eta_t$ 为 $1/t$ 量级，会让两项都落在 $1/t^2$ 量级，正好对应 $1/t$ 序列相邻两项的差。这提示了下面的步长与归纳形式；具体常数仍需要证明。

分母中的 $\tau$ 用来让前几步也满足光滑性要求，避免 $t$ 小时步长过大。该命题的价值是：不必依靠平均输出，也能证明最后一次迭代的期望目标误差趋于零。

保持上述假设，取

$$
\eta_t=\frac2{\mu(t+\tau)},\qquad \tau\ge\frac{2\beta}{\mu}. \tag{2.4.12}
$$

则

$$
\delta_T\le\frac{\tau\delta_1+2\beta\sigma^2/\mu^2}{T+\tau-1}. \tag{2.4.13}
$$

**证明。** 步长满足 $\eta_t\le1/\beta$，所以式（2.4.11）的证明仍适用：

$$
\delta_{t+1}\le\left(1-\frac2{t+\tau}\right)\delta_t
+\frac C{(t+\tau)^2},\qquad C=\frac{2\beta\sigma^2}{\mu^2}. \tag{2.4.14}
$$

令

$$
\mathcal A=\tau\delta_1+C. \tag{2.4.15}
$$

归纳证明 $\delta_t\le\mathcal A/(t+\tau-1)$。$t=1$ 时，$\mathcal A/\tau=\delta_1+C/\tau\ge\delta_1$，所以成立。设第 $t$ 步成立，令 $s=t+\tau$，则

$$
\delta_{t+1}\le\frac{\mathcal A(s-2)}{s(s-1)}+\frac C{s^2}
\le\frac{\mathcal A}s.
$$

最后一个不等式因为

$$
\frac{\mathcal A}s-\frac{\mathcal A(s-2)}{s(s-1)}
=\frac{\mathcal A}{s(s-1)}\ge\frac C{s^2},
$$

其中用了 $\mathcal A\ge C$。归纳完成。

式（2.4.13）写成 $\delta_T\le\mathcal A/(T+\tau-1)$ 后，只要 $T+\tau-1\ge\mathcal A/\varepsilon$，就能保证期望目标间隙不超过 $\varepsilon$。这里的 $\mathcal A$ 同时记录了初值和噪声代价。

接下来仍需回答参数距离的问题。与定理 2.9 一样，强凸性把目标误差换成平方参数误差；区别是现在需要取期望。

### 2.4.6 推论 2.14 从目标误差换算参数误差

对每次实现，在最小点处用强凸性与 $\nabla\Phi(\theta^*)=0$，有

$$
\Phi(\theta_T)-\Phi^*\ge\frac\mu2\|\theta_T-\theta^*\|^2.
$$

先移项再取期望，得到 $E\|\theta_T-\theta^*\|^2\le2\delta_T/\mu$；代入命题 2.13：

$$
\mathbb E\|\theta_T-\theta^*\|^2
\le\frac{2\mathcal A}{\mu(T+\tau-1)}=O(T^{-1}). \tag{2.4.16}
$$

由 Jensen 和 Cauchy–Schwarz 不等式，

$$
\|\mathbb E\theta_T-\theta^*\|
\le\mathbb E\|\theta_T-\theta^*\|
\le\sqrt{\frac{2\mathcal A}{\mu(T+\tau-1)}}=O(T^{-1/2}). \tag{2.4.17}
$$

### 2.4.7 命题 2.15：对参数作算术平均

不能把连续迭代当成独立样本，直接声称“平均后方差除以 $T$”。实际上

$$
\mathbb E\left\|\frac1T\sum_{t=1}^Te_t\right\|^2
=\frac1{T^2}\sum_{t=1}^T\sum_{s=1}^T\mathbb E\langle e_t,e_s\rangle,
$$

不同时间的交叉项通常不为零。下面选择 $L^2$ 三角不等式，就是为了在不假设独立的情况下控制这些相关性。

定义 Polyak 平均

$$
\bar\theta_T^{\rm PR}=\frac1T\sum_{t=1}^T\theta_t. \tag{2.4.18}
$$

在命题 2.13 的假设和步长下，

$$
\mathbb E\|\bar\theta_T^{\rm PR}-\theta^*\|^2\le\frac{8\mathcal A}{\mu T}, \tag{2.4.19}
$$

$$
\mathbb E[\Phi(\bar\theta_T^{\rm PR})]-\Phi^*\le\frac{4\beta\mathcal A}{\mu T}. \tag{2.4.20}
$$

**证明。** 这里 $e_t=\theta_t-\theta^*$，所以平均参数的误差为 $T^{-1}\sum_te_t$。强凸下界与光滑上界说明 $\mu\le\beta$，故 $\tau\ge2\beta/\mu\ge2$。于是

$$
E\|e_t\|^2\le\frac{2\mathcal A}{\mu(t+\tau-1)}\le\frac{2\mathcal A}{\mu t}.
$$

用随机向量的 $L^2$ 三角不等式：

$$
\begin{aligned}
\left(\mathbb E\left\|\frac1T\sum_{t=1}^T e_t\right\|^2\right)^{1/2}
&\le\frac1T\sum_{t=1}^T\bigl(\mathbb E\|e_t\|^2\bigr)^{1/2}\\
&\le\frac1T\sqrt{\frac{2\mathcal A}{\mu}}\sum_{t=1}^Tt^{-1/2}
\le\sqrt{\frac{8\mathcal A}{\mu T}}.
\end{aligned}
$$

**补全两处不等式。** 对任意二阶可积随机向量 $U_t$，

$$
E\left\|\sum_tU_t\right\|^2
=\sum_{t,s}E\langle U_t,U_s\rangle
\le\sum_{t,s}\sqrt{E\|U_t\|^2}\sqrt{E\|U_s\|^2}
=\left(\sum_t\sqrt{E\|U_t\|^2}\right)^2.
$$

开平方就是上面使用的 $L^2$ 三角不等式，无需独立性。而

$$
\sum_{t=1}^Tt^{-1/2}\le1+\int_1^Tx^{-1/2}dx=2\sqrt T-1\le2\sqrt T.
$$

所以平均误差的 $L^2$ 范数至多为 $\sqrt{8\mathcal A/(\mu T)}$，平方得到式（2.4.19）。再用光滑上界并取期望，得到式（2.4.20）。

本命题证明的是“平均迭代保留同阶收敛保证”，没有证明其常数一定优于最后迭代。实际实现平均也不必保存所有参数，只需递推 $\bar\theta_t=\bar\theta_{t-1}+(\theta_t-\bar\theta_{t-1})/t$。

### 2.4.8 备注 2.16：不依赖光滑性的两种平均

上一证明的最后一步用了光滑性，把参数平方距离变成目标上界。去掉它后，证明必须直接控制目标误差。这里比较两种平均的目的，是说明**平均权重本身也参与决定可以证明的速率**：权重不是随便加上的，而是为了让前后两步的距离系数在求和时相消。

现在只保留可微、$\mu$-强凸及完整梯度二阶矩界 $G^2$，同时保留随机梯度的条件无偏性。记 $d_t=E\|\theta_t-\theta^*\|^2$、$\delta_t=E[\Phi(\theta_t)]-\Phi^*$。

**先证明基本递推。** 展开 $e_{t+1}=e_t-\eta_tg_t$ 的平方并取条件期望：

$$
E[\|e_{t+1}\|^2\mid\mathcal H_t]
\le\|e_t\|^2-2\eta_t\langle\nabla\Phi(\theta_t),e_t\rangle+\eta_t^2G^2.
$$

把强凸性中的 $u,v$ 分别取为 $\theta_t,\theta^*$，移项得到

$$
\langle\nabla\Phi(\theta_t),\theta_t-\theta^*\rangle
\ge\Phi(\theta_t)-\Phi^*+\frac\mu2\|\theta_t-\theta^*\|^2,
$$

代入、取全期望：$d_{t+1}\le(1-\mu\eta_t)d_t-2\eta_t\delta_t+\eta_t^2G^2$。再移项便是

$$
2\eta_t\delta_t\le(1-\mu\eta_t)d_t-d_{t+1}+\eta_t^2G^2. \tag{2.4.21}
$$

这控制的是误差之和，不能再用光滑上界把平均参数的平方误差转换为目标误差。

**算术平均。** 取 $\eta_t=1/(\mu t)$，除以 $\eta_t$ 后，

$$
2\delta_t\le\mu(t-1)d_t-\mu t d_{t+1}+\frac{G^2}{\mu t}.
$$

求和时，距离项具体消为

$$
\mu\sum_{t=1}^T[(t-1)d_t-td_{t+1}]
=-\mu Td_{T+1}\le0.
$$

其中 $d_1$ 的系数为零，$d_2$ 的系数为 $1-1$，$d_3$ 的系数为 $2-2$，依此类推。因此 $2\sum_t\delta_t\le(G^2/\mu)\sum_t1/t$。由凸性，

$$
\mathbb E\left[\Phi\left(\frac1T\sum_{t=1}^T\theta_t\right)\right]-\Phi^*
\le\frac1T\sum_{t=1}^T\delta_t
\le\frac{G^2 H_T}{2\mu T},\qquad H_T=\sum_{t=1}^T\frac1t.
$$

这给出 $O(\log T/T)$。

**线性加权平均。** 取 $\eta_t=2/[\mu(t+1)]$，令 $\bar\theta_T^{\rm lin}=2\sum_{t=1}^Tt\theta_t/[T(T+1)]$。将式（2.4.21）乘以 $\mu t(t+1)/2$：

$$
2t\delta_t\le\frac\mu2\{t(t-1)d_t-t(t+1)d_{t+1}\}
+\frac{2G^2t}{\mu(t+1)}.
$$

求和时，距离项为

$$
\sum_{t=1}^T[t(t-1)d_t-t(t+1)d_{t+1}]
=-T(T+1)d_{T+1}\le0.
$$

原因是第二组的第 $t$ 项系数 $-t(t+1)$，恰好与第一组第 $t+1$ 项的系数 $(t+1)t$ 抵消；第一项系数 $1\cdot0$ 为零。于是

$$
2\sum_{t=1}^Tt\delta_t
\le\frac{2G^2}{\mu}\sum_{t=1}^T\frac{t}{t+1}
\le\frac{2G^2T}{\mu}.
$$

权重 $w_t=2t/[T(T+1)]$ 非负且和为 $1$。凸性给 $E\Phi(\sum_tw_t\theta_t)-\Phi^*\le\sum_tw_t\delta_t$，因此

$$
\mathbb E[\Phi(\bar\theta_T^{\rm lin})]-\Phi^*
\le\frac{2G^2}{\mu(T+1)}=O(T^{-1}).
$$

没有光滑性时，合理选择平均权重仍能恢复 $O(1/T)$。这些求和证明没有直接给出最后一次迭代的同阶界；随机目标值一般也不单调下降。

### 2.4.9 定理 2.17：一般凸 SGD 的平均误差

现在连强凸性也不再假设。我们仍能用凸性把平方距离的减少量与目标间隙联系起来，但随机目标值不再保证逐步下降，无法像定理 2.10 那样从累计误差直接控制最后一步。因此改为返回一个平均参数，并借凸性把“平均目标误差界”转成“这个输出点的目标误差界”。

设 $\Phi$ 凸、可微且有最小点，随机梯度满足定义 2.11，并且 $\mathbb E\|\theta_1-\theta^*\|^2\le R^2$。取确定的正步长，定义

$$
S_T=\sum_{t=1}^T\eta_t,\qquad Q_T=\sum_{t=1}^T\eta_t^2,\qquad
\bar\theta_T^{(\eta)}=\frac1{S_T}\sum_{t=1}^T\eta_t\theta_t. \tag{2.4.22}
$$

则

$$
\mathbb E[\Phi(\bar\theta_T^{(\eta)})]-\Phi^*
\le\frac{R^2+G^2Q_T}{2S_T}. \tag{2.4.23}
$$

**证明。** 平方距离展开、无偏性与凸性依次给出

$$
\begin{aligned}
\mathbb E[\|\theta_{t+1}-\theta^*\|^2\mid\mathcal H_t]
&\le\|\theta_t-\theta^*\|^2
-2\eta_t\langle\nabla\Phi(\theta_t),\theta_t-\theta^*\rangle+\eta_t^2G^2\\
&\le\|\theta_t-\theta^*\|^2-2\eta_t(\Phi(\theta_t)-\Phi^*)+\eta_t^2G^2.
\end{aligned}
$$

取全期望并求和，$2\sum\eta_t\delta_t\le R^2+G^2Q_T$。最后利用凸性的 Jensen 不等式，$\Phi(\sum\eta_t\theta_t/S_T)\le\sum\eta_t\Phi(\theta_t)/S_T$，即得结论。

求和控制的是 $\sum\eta_t\delta_t$，所以选择权重 $\eta_t/S_T$ 就能直接使用这个结果。如果随意换一个输出平均，前面的证明并没有自动为它提供保证。

式（2.4.23）也把步长的两种作用拆开：$S_T$ 衡量累计前进量，$Q_T$ 衡量随机梯度平方项的累计代价。最终上界是初始距离贡献 $R^2/(2S_T)$ 加上 $G^2Q_T/(2S_T)$。这里 $S_T$ 是步长之和，与第一章表示训练集的 $S$ 不同。

固定步长时，步长加权平均就是算术平均：

$$
\bar\theta_T=\frac1T\sum_{t=1}^T\theta_t. \tag{2.4.24}
$$

由 $S_T=\eta T$、$Q_T=\eta^2T$，

$$
\mathbb E[\Phi(\bar\theta_T)]-\Phi^*
\le\frac{R^2}{2\eta T}+\frac{\eta G^2}{2}. \tag{2.4.25}
$$

第一项要求步长别太小，第二项要求别太大。给定运行长度 $T$，取 $\eta=R/(G\sqrt T)$（$R,G>0$），使两项相等：

$$
\mathbb E[\Phi(\bar\theta_T)]-\Phi^*\le\frac{RG}{\sqrt T}. \tag{2.4.26}
$$

这里“固定”指本次运行中不变；换一个预定长度 $T$，所选步长也会改变。

### 2.4.10 命题 2.18：光滑性将梯度大小换成噪声方差

若进一步假设 $\beta$-光滑，使用式（2.4.7）的噪声方差界，并取 $0<\eta_t\le1/\beta$，则

$$
\mathbb E[\Phi(\bar\theta_T^{(\eta)})]-\Phi^*
\le\frac{R^2+\sigma^2 Q_T}{2S_T}. \tag{2.4.27}
$$

**先补全关键不等式。** 光滑凸性给出比普通凸性更强的结论：

$$
\langle\nabla\Phi(\theta),\theta-\theta^*\rangle
\ge\Phi(\theta)-\Phi^*+\frac1{2\beta}\|\nabla\Phi(\theta)\|^2. \tag{2.4.31}
$$

为证明它，固定 $g=\nabla\Phi(\theta)$，定义 $h(z)=\Phi(z)-\langle g,z\rangle$。$h$ 仍然凸且光滑，$\nabla h(\theta)=0$，故 $\theta$ 是 $h$ 的全局最小点；而 $\nabla h(\theta^*)=-g$。这个辅助函数把当前点变成了一个已知最小点，所以能用其最小值比较一步下降的结果。

具体地，对 $h$ 从 $\theta^*$ 做一次步长 $1/\beta$ 的 GD，下降引理给出

$$
h(\theta)\le h(\theta^*+g/\beta)
\le h(\theta^*)-\frac{\|g\|^2}{2\beta}.
$$

代回 $h$ 并移项，得到所需不等式。

**完成命题证明。** 在平方距离递推中使用它及噪声分解：

$$
\begin{aligned}
\mathbb E[\|\theta_{t+1}-\theta^*\|^2\mid\mathcal H_t]
\le{}&\|\theta_t-\theta^*\|^2-2\eta_t(\Phi(\theta_t)-\Phi^*)\\
&-\eta_t(1/\beta-\eta_t)\|\nabla\Phi(\theta_t)\|^2+\eta_t^2\sigma^2.
\end{aligned} \tag{2.4.32}
$$

因为 $\eta_t\le1/\beta$，梯度平方项的系数非正，删掉它得到

$$
2\eta_t\delta_t\le d_t-d_{t+1}+\eta_t^2\sigma^2.
$$

从 $1$ 到 $T$ 求和，距离项消为 $d_1-d_{T+1}\le R^2$，所以 $2\sum_t\eta_t\delta_t\le R^2+\sigma^2Q_T$。再由凸性，平均参数的目标间隙不超过 $\sum_t\eta_t\delta_t/S_T$，即得式（2.4.27）。

固定步长时，

$$
\mathbb E[\Phi(\bar\theta_T)]-\Phi^*\le\frac{R^2}{2\eta T}+\frac{\eta\sigma^2}{2}. \tag{2.4.28}
$$

当 $R>0$，选择

$$
\eta=\frac1{\beta+\sigma\sqrt T/R}\le\frac1\beta, \tag{2.4.29}
$$

代入第一项：

$$
\frac{R^2}{2\eta T}
=\frac{R^2}{2T}\left(\beta+\frac{\sigma\sqrt T}{R}\right)
=\frac{\beta R^2}{2T}+\frac{R\sigma}{2\sqrt T}.
$$

若 $\sigma>0$，分母 $\beta+\sigma\sqrt T/R\ge\sigma\sqrt T/R$，所以 $\eta\sigma^2/2\le R\sigma/(2\sqrt T)$。若 $\sigma=0$，该项直接为零。两项相加得到

$$
\mathbb E[\Phi(\bar\theta_T)]-\Phi^*
\le\underbrace{\frac{\beta R^2}{2T}}_{\text{确定性部分}}
+\underbrace{\frac{R\sigma}{\sqrt T}}_{\text{随机噪声部分}}. \tag{2.4.30}
$$

若小批量含 $b$ 个独立样本，单样本噪声方差界为 $\sigma_1^2$，则 $\sigma^2=\sigma_1^2/b$，随机项变成 $R\sigma_1/\sqrt{bT}$。

SGD 的证明仍是“更新式 → 误差递推 → 求和或收缩”，新增的工作只有条件期望与噪声项控制。但步长是否递减、返回最后一步还是平均值，已经成为定理结论的一部分。

使用本节定理时，应先确定输出对象和矩条件，再比较速率。不能只记“SGD 是 $1/\sqrt T$”：强凸时可以得到 $1/T$ 的期望目标界；光滑一般凸时能分离 $1/T$ 的确定性项与 $1/\sqrt T$ 的随机项；固定步长与递减步长的结论也不同。

## 练习

1. 对单样本噪声方差 $\sigma_1^2$，比较独立有放回小批量大小 $b$ 与 $4b$ 的方差、标准差。
2. 推导一维二次目标的固定步长稳态均方误差，检查 $\eta\downarrow0$ 与稳定性边界两端的行为。
3. 从式（2.4.21）的距离递推出发，重做线性权重的求和，明确首项和末项。

核对：第一题分别缩小为四分之一、二分之一；第二题在正噪声方差下，小步长极限为零，接近 $2/h$ 时发散；第三题的终端距离项非正，首项系数为零，不应多出初值常数。
