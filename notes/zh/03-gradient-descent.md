# 03 · 梯度下降：谱分解、收缩与望远镜求和

[中文目录](README.md) · [上一章](02-double-descent.md) · [下一章](04-sgd-finite-time.md)

固定训练数据以后，记优化目标为 $\Phi$，最小点为 $\theta^*$，最小值为 $\Phi^*$。优化误差 $\Phi(\theta_T)-\Phi^*$ 与总体预测风险不同。本篇从二次目标的精确递推出发，再证明一般强凸及凸目标的收敛界。

记 $e_t=\theta_t-\theta^*$，$\Delta_t=\Phi(\theta_t)-\Phi^*$；$\mu$ 为强凸常数，$\beta$ 为梯度的 Lipschitz 常数，$\kappa=\beta/\mu$。范数默认为欧氏范数；矩阵范数为算子范数。所讨论的目标在整个欧氏空间上可微，最小值取得，故 $\nabla\Phi(\theta^*)=0$。公式编号沿用课程第二章，便于对照。

## 2.1 梯度下降：最小二乘与基本几何

### 2.1.1 从更新式得到误差递推

在当前点 $\theta$ 附近，沿位移 $d$ 的目标变化一阶近似为 $\langle\nabla\Phi(\theta),d\rangle$。若固定移动长度，Cauchy–Schwarz 不等式表明，沿负梯度方向能使这个一阶变化最小；因此采用 $d=-\eta\nabla\Phi(\theta)$。但线性近似没有控制大步移动的误差，允许走多远还需要曲率信息。

先研究能够精确计算曲率的最小二乘目标

$$
\Phi(\theta)=\frac12\|A\theta-b\|^2. \tag{2.1.1}
$$

设 $A\in\mathbb R^{n\times d}$、$b\in\mathbb R^n$、$\theta\in\mathbb R^d$。这里没有除以样本数；若整体再乘 $1/n$，曲率常数和允许步长也随之改变。先展开：

$$
\Phi(\theta)=\frac12\theta^\top A^\top A\theta-\theta^\top A^\top b+\frac12b^\top b.
$$

因为 $A^\top A$ 对称，二次项对 $\theta$ 的梯度为 $A^\top A\theta$；线性项梯度为 $-A^\top b$。所以

$$
\nabla\Phi(\theta)=A^\top A\theta-A^\top b. \tag{2.1.2}
$$

固定步长 $\eta$ 的梯度下降为

$$
\theta_{t+1}=(I-\eta A^\top A)\theta_t+\eta A^\top b. \tag{2.1.3}
$$

最小点满足正规方程 $A^\top A\theta^*=A^\top b$。令 $e_t=\theta_t-\theta^*$，相减得到

$$
e_{t+1}=(I-\eta A^\top A)e_t. \tag{2.1.4}
$$

原更新式含有固定项 $\eta A^\top b$，直接看不出是否收敛。最优点满足同一个平衡关系，相减后固定项消失，只剩“旧误差乘一个矩阵”。虽然算法运行时不知道 $\theta^*$，证明中仍可拿它作为比较基准；更新本身并不需要它。

先假设 $A$ 列满秩，记

$$
H=A^\top A\succ0,\qquad
\mu=\lambda_{\min}(H),\qquad \beta=\lambda_{\max}(H). \tag{2.1.5}
$$

### 2.1.2 命题 2.1：最小二乘 GD 的精确收敛界

若 $0<\eta<2/\beta$，定义

$$
q(\eta)=\max_j|1-\eta\lambda_j(H)|<1. \tag{2.1.6}
$$

则

$$
\|\theta_T-\theta^*\|\le q(\eta)^T\|\theta_0-\theta^*\|, \tag{2.1.7}
$$

$$
\Phi(\theta_T)-\Phi^*
\le q(\eta)^{2T}\bigl(\Phi(\theta_0)-\Phi^*\bigr). \tag{2.1.8}
$$

**补全谱分解证明。** 因为 $H$ 对称正定，可写为 $H=Q\Lambda Q^\top$，其中 $Q^\top Q=I$，$\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)$。在特征向量坐标 $z_t=Q^\top e_t$ 下，

$$
z_{t+1,j}=(1-\eta\lambda_j)z_{t,j},\qquad
z_{T,j}=(1-\eta\lambda_j)^Tz_{0,j}.
$$

$0<\eta<2/\beta$ 保证每个 $|1-\eta\lambda_j|<1$。正交变换保持范数，所以

$$
\|e_T\|^2=\sum_j(1-\eta\lambda_j)^{2T}z_{0,j}^2
\le q(\eta)^{2T}\|e_0\|^2.
$$

另一方面，将 $\theta=\theta^*+e$ 代入目标。由 $A\theta-b=(A\theta^*-b)+Ae$，平方展开并减去 $\Phi^*$：

$$
\Phi(\theta)-\Phi^*
=e^\top A^\top(A\theta^*-b)+\frac12e^\top He
=\frac12e^\top He.
$$

第一项由正规方程为零。因此

$$
\Phi(\theta_T)-\Phi^*
=\frac12\sum_j\lambda_j(1-\eta\lambda_j)^{2T}z_{0,j}^2
\le q(\eta)^{2T}\bigl(\Phi(\theta_0)-\Phi^*\bigr).
$$

两个结论均得证。参数范数的收缩因子是 $q^T$，二次目标间隙的收缩因子是 $q^{2T}$。

若已知 $q=0.9$，运行 $T$ 步后，参数误差至多剩初始的 $0.9^T$，目标间隙至多剩 $0.9^{2T}$。这不是每条轨迹的精确等式：初始误差若没有落在最慢方向，实际收敛可以更快。命题提供的是不依赖初始方向的保证。

这与第一章的 SVD 是同一种思路：若 $A$ 的奇异值为 $s_j$，则 $H=A^\top A$ 的正特征值为 $s_j^2$。第一章研究小奇异值如何放大估计噪声；这里研究小特征值如何使某些方向的优化变慢。

### 2.1.3 步长、振荡与条件数

在特征方向 $j$ 上，更新只做一件事：把误差乘以 $1-\eta\lambda_j$。步长增大，起初能让这个因子更接近零；超过一定范围后会跨过最小点，再继续增大就会越走越远。下面的区间划分正是把这个标量因子逐一检查。

由于 $\lambda\mapsto|1-\eta\lambda|$ 是凸函数，区间上的最大值在端点取得，而 $\mu,\beta$ 都是特征值，故

$$
q(\eta)=\max\{|1-\eta\mu|,|1-\eta\beta|\}. \tag{2.1.9}
$$

| 步长范围 | 各特征方向上的行为 |
|---|---|
| $0<\eta<1/\beta$ | 所有乘子为正且小于 1；误差不换符号地缩小 |
| $\eta=1/\beta$ | 最大曲率方向的误差一步消除；其余方向继续缩小 |
| $1/\beta<\eta<2/\beta$ | 部分乘子为负；对应误差左右交替，但幅度缩小 |
| $\eta=2/\beta$ | 最大曲率方向的乘子为 $-1$；该方向一般不收敛 |
| $\eta>2/\beta$ | 最大曲率方向的乘子绝对值大于 1；有该分量时发散 |

前两种情形满足

$$
q=1-\eta\mu. \tag{2.1.10}
$$

第三种情形满足

$$
q=\max\{1-\eta\mu,\eta\beta-1\}. \tag{2.1.11}
$$

因此，**出现振荡不等于发散；$1/\beta$ 也不是二次目标的稳定性边界。** 边界 $2/\beta$ 指对任意初值的保证，特殊初值可能没有不稳定方向的分量。

定义条件数 $\kappa=\beta/\mu$。取 $\eta=1/\beta$ 时，

$$
\Phi(\theta_T)-\Phi^*
\le (1-1/\kappa)^{2T}\bigl(\Phi(\theta_0)-\Phi^*\bigr). \tag{2.1.12}
$$

$\kappa$ 大，表示不同方向的曲率相差很大：步长受最陡方向限制，最平方向却前进缓慢。若要最小化最坏方向收缩因子，就让两个端点一样大：

$$
1-\eta\mu=\eta\beta-1
\quad\Longrightarrow\quad
\eta_{\rm opt}=\frac2{\mu+\beta},\qquad
q_{\rm opt}=\frac{\kappa-1}{\kappa+1}. \tag{2.1.13}
$$

这一选择确实是全局最优：当 $\eta\le\eta_{\rm opt}$ 时，$q(\eta)\ge1-\eta\mu\ge q_{\rm opt}$；当 $\eta\ge\eta_{\rm opt}$ 时，$q(\eta)\ge\eta\beta-1\ge q_{\rm opt}$。而在 $\eta_{\rm opt}$ 处两个端点的绝对值都等于 $q_{\rm opt}$，达到这个下界。

这给出相对误差达到 $\varepsilon$ 所需的 $O(\kappa\log(1/\varepsilon))$ 迭代量级；$\kappa=1$ 时取 $1/\beta$ 一步到达最小点。

例如 $H=\operatorname{diag}(1,100)$，取 $\eta=1/100$。第二个方向的误差一步变为零，第一个方向却每次只乘以 $0.99$。因此“某些方向已经拟合好，整体训练仍很慢”可以单纯由曲率不均衡造成。后面动量和缩放方法的动机之一，就是改善这种方向之间的不均衡；它们是否有效仍需各自分析。

### 2.1.4 秩亏时为什么零初始化选择最小范数解

**与第一章的连接。** 若 $A$ 不满列秩且 $A\ne0$，取 $0<\eta<2/\|A\|_{\rm op}^2$。零特征值方向的乘子为 1，因而初值在 $\ker A$ 中的分量保留；其余方向的乘子绝对值小于 1，最终

$$
\theta_\infty=A^\dagger b+P_{\ker A}\theta_0.
$$

**补全推导。** 取 $\theta^\dagger=A^\dagger b$ 作为比较点，它满足正规方程，误差仍满足

$$
\theta_{t+1}-\theta^\dagger=(I-\eta A^\top A)(\theta_t-\theta^\dagger).
$$

在 $A^\top A$ 的正特征值方向，乘子绝对值小于 $1$，误差趋于零；在零特征值方向，乘子等于 $1$，误差保持不变。零特征空间就是 $\ker A$，而 $\theta^\dagger$ 位于与其正交的 $\operatorname{range}(A^\top)$，所以最后只保留 $P_{\ker A}\theta_0$。

任意最小二乘解都可写成 $\theta^\dagger+z$，$z\in\ker A$。事实上，两个解的正规方程相减给 $A^\top Az=0$，左乘 $z^\top$ 得 $\|Az\|^2=0$。又由于 $\theta^\dagger\perp z$，

$$
\|\theta^\dagger+z\|^2=\|\theta^\dagger\|^2+\|z\|^2\ge\|\theta^\dagger\|^2.
$$

因此零初始化消除了全部零空间分量，选择最小范数解。这是在多个相同训练目标值的解之间作出的算法选择。 若 $A=0$，目标是常量，梯度始终为零、$\theta_t=\theta_0$；上面的极限公式仍成立，但不使用含 $1/\|A\|_{\rm op}^2$ 的步长条件。

### 2.1.5 定义 2.2–2.4 凸性、强凸性与光滑性

二次目标有固定矩阵 $H$，一般函数却未必能通过一次对角化解决。接下来把刚才用到的结构改写成适用于一般函数的不等式：**凸性负责连接当前位置与全局最优；光滑性负责保证一步不会走坏；强凸性负责保证离最优点较远时有足够的下降动力。**

**定义 2.2：凸性。** 对可微函数，凸性等价于

$$
\Phi(v)\ge\Phi(u)+\langle\nabla\Phi(u),v-u\rangle. \tag{2.1.14}
$$

即每一点的切平面都是全局下界。

**定义 2.3：$\mu$-强凸性。** 存在 $\mu>0$，使

$$
\Phi(v)\ge\Phi(u)+\langle\nabla\Phi(u),v-u\rangle
+\frac\mu2\|v-u\|^2. \tag{2.1.15}
$$

它给出统一的二次下界，排除了过于平坦的方向；若最小点存在，则唯一。

**唯一性证明。** 若 $u,v$ 都是无约束最小点，在 $u$ 的强凸下界中用 $\nabla\Phi(u)=0$，得到 $\Phi(v)\ge\Phi(u)+\mu\|v-u\|^2/2$。两点目标值相同，只能有 $u=v$。

**定义 2.4：$\beta$-光滑性。** 梯度满足 Lipschitz 条件

$$
\|\nabla\Phi(v)-\nabla\Phi(u)\|\le\beta\|v-u\|. \tag{2.1.16}
$$

它限制梯度变化速度，防止局部曲率过大。

**与常见凸性定义的联系。** 线段定义为 $\Phi((1-s)u+sv)\le(1-s)\Phi(u)+s\Phi(v)$。它推出

$$
\frac{\Phi(u+s(v-u))-\Phi(u)}s\le\Phi(v)-\Phi(u).
$$

让 $s\downarrow0$，左边趋于 $\langle\nabla\Phi(u),v-u\rangle$，得到式（2.1.14）。反过来，在 $z=(1-s)u+sv$ 分别对 $u,v$ 用一阶下界，再乘 $1-s,s$ 相加，梯度项因 $(1-s)(u-z)+s(v-z)=0$ 消失，恢复线段定义。因此以后对迭代平均使用凸性是合法的。

**用 Hessian 看几何。** 若 $\Phi$ 二阶连续可微，沿方向 $d=v-u$ 的二次余项为

$$
\Phi(v)-\Phi(u)-\langle\nabla\Phi(u),d\rangle
=\int_0^1(1-s)d^\top\nabla^2\Phi(u+sd)d\,ds.
$$

若处处有 $\nabla^2\Phi\succeq0$，右边非负，得到凸性；若处处有 $\nabla^2\Phi\succeq\mu I$，积分至少为 $\mu\|d\|^2/2$，得到强凸性。反向则令 $v=u+td$，对二次余项除以 $t^2/2$ 后令 $t\to0$，得到相应的 Hessian 下界。

光滑性对应 $\|\nabla^2\Phi\|_{\rm op}\le\beta$：一个方向由梯度差商取极限得到；反向由 $\nabla\Phi(v)-\nabla\Phi(u)=\int_0^1\nabla^2\Phi(u+sd)d\,ds$ 取范数得到。因此同时强凸、光滑时，曲率处处夹在 $\mu$ 与 $\beta$ 之间。只检查最小点处的 Hessian 不足以验证全局条件。

一般 GD 的更新是

$$
\theta_{t+1}=\theta_t-\eta_t\nabla\Phi(\theta_t). \tag{2.1.17}
$$

### 2.1.6 引理 2.5 与推论 2.6 光滑性怎样保证一步下降

**引理 2.5：下降引理。** $\beta$-光滑性推出

$$
\Phi(v)\le\Phi(u)+\langle\nabla\Phi(u),v-u\rangle
+\frac\beta2\|v-u\|^2. \tag{2.1.18}
$$

**证明第一步，把多元函数变成一元函数。** 记 $d=v-u$，定义 $\psi(s)=\Phi(u+sd)$，$0\le s\le1$。链式法则给

$$
\psi'(s)=\langle\nabla\Phi(u+sd),d\rangle.
$$

微积分基本定理给 $\Phi(v)-\Phi(u)=\psi(1)-\psi(0)=\int_0^1\psi'(s)ds$。又因区间长度为 $1$，$\langle\nabla\Phi(u),d\rangle=\int_0^1\langle\nabla\Phi(u),d\rangle ds$。相减得到

$$
\begin{aligned}
\Phi(v)-\Phi(u)-\langle\nabla\Phi(u),d\rangle
&=\int_0^1\langle\nabla\Phi(u+sd)-\nabla\Phi(u),d\rangle\,ds\\
&\le\int_0^1\beta s\|d\|^2\,ds
=\frac\beta2\|d\|^2.
\end{aligned}
$$

**证明第二步，控制被积函数。** 具体使用

$$
\langle\nabla\Phi(u+sd)-\nabla\Phi(u),d\rangle
\le\|\nabla\Phi(u+sd)-\nabla\Phi(u)\|\|d\|
\le\beta s\|d\|^2.
$$

积分后 $\int_0^1s\,ds=1/2$，即得结论。整个证明不需要凸性。

这也解释了步长 $1/\beta$ 的来源。固定当前点 $u$，令 $g=\nabla\Phi(u)$，把上界看成位移 $d$ 的函数：

$$
Q_u(d)=\Phi(u)+\langle g,d\rangle+\frac\beta2\|d\|^2
=\Phi(u)-\frac{\|g\|^2}{2\beta}+\frac\beta2\|d+g/\beta\|^2.
$$

这个二次上界在 $d=-g/\beta$ 处最小，所以选步长 $1/\beta$ 是在最小化一个有保证的局部上界，不是凭空规定的常数。

代入 $v=u-\eta\nabla\Phi(u)$：

$$
\Phi(u-\eta\nabla\Phi(u))
\le\Phi(u)-\eta\left(1-\frac{\beta\eta}{2}\right)\|\nabla\Phi(u)\|^2. \tag{2.1.19}
$$

特别地，当 $0<\eta\le1/\beta$，

$$
\Phi(u-\eta\nabla\Phi(u))
\le\Phi(u)-\frac\eta2\|\nabla\Phi(u)\|^2. \tag{2.1.20}
$$

光滑性给出二次上界，强凸性给出二次下界。一般凸函数可能有多个最小点；但反过来，最小点唯一也不意味着强凸性，例如 $\Phi(x)=x^4$。

## 2.2 强凸目标：把每步下降变成几何收敛

### 2.2.1 引理 2.7：梯度大小控制目标间隙

下降引理右边出现 $\|\nabla\Phi(\theta)\|^2$，我们需要用目标间隙给它一个**下界**，这样才知道每步至少消除多少误差。强凸二次下界可以直接取最小值，因此证明的策略是：先用它估计“真实最优值最小能低到哪里”，再反过来约束当前间隙。

若 $\Phi$ 是 $\mu$-强凸的，则

$$
\|\nabla\Phi(\theta)\|^2\ge2\mu\bigl(\Phi(\theta)-\Phi^*\bigr). \tag{2.2.1}
$$

**证明。** 固定 $\theta$，记 $g=\nabla\Phi(\theta)$。强凸性及配方给出

$$
\Phi(v)\ge\Phi(\theta)+\langle g,v-\theta\rangle
+\frac\mu2\|v-\theta\|^2
=\Phi(\theta)-\frac{\|g\|^2}{2\mu}
+\frac\mu2\left\|v-\theta+\frac g\mu\right\|^2.
$$

对 $v$ 取下确界，得到 $\Phi^*\ge\Phi(\theta)-\|g\|^2/(2\mu)$，移项即可。这个不等式意味着：在强凸条件下，只要目标还远离最优值，梯度就不能太小。

改写为 $\Phi(\theta)-\Phi^*\le\|\nabla\Phi(\theta)\|^2/(2\mu)$。如果强凸常数 $\mu$ 已知，梯度范数不超过 $\sqrt{2\mu\varepsilon}$ 就能证明目标间隙不超过 $\varepsilon$。这里不需要知道最优值本身。但对一般非凸函数，小梯度可能出现在鞍点附近，不能直接作同样判断。

### 2.2.2 定理 2.8：目标值的几何收敛

现在两条工具可以接上了：光滑性保证“下降量至少是梯度平方的倍数”，强凸性保证“梯度平方至少是剩余误差的倍数”。所以证明只需把两者代入，得到一个只含前后两次目标间隙的递推。

若目标同时 $\mu$-强凸、$\beta$-光滑，固定 $0<\eta\le1/\beta$，则

$$
\Phi(\theta_T)-\Phi^*
\le(1-\mu\eta)^T\bigl(\Phi(\theta_0)-\Phi^*\bigr). \tag{2.2.2}
$$

**证明。** 记 $\Delta_t=\Phi(\theta_t)-\Phi^*$。将引理 2.7 代入下降引理：

$$
\Delta_{t+1}\le\Delta_t-\frac\eta2\|\nabla\Phi(\theta_t)\|^2
\le(1-\mu\eta)\Delta_t.
$$

迭代 $T$ 次即得结果。取 $\eta=1/\beta$ 时，

$$
\Delta_T\le(1-\mu/\beta)^T\Delta_0. \tag{2.2.3}
$$

这叫几何收敛，也常叫线性收敛，指每一步乘以固定的小于 1 的因子，不是指误差与 $T$ 成线性关系。

### 2.2.3 定理 2.9：从目标误差转成参数误差

一般情况下不一定：很平的方向上，参数移动很远也可能只改变很少的目标值。强凸性给出二次增长，排除了这种情况。本定理的作用是将已经证明的目标误差界换算成参数距离，而不必重新分析一遍算法。

无约束最小点满足 $\nabla\Phi(\theta^*)=0$。分别在 $\theta^*$ 使用强凸下界和光滑上界，可得

$$
\Phi(\theta)-\Phi^*\ge\frac\mu2\|\theta-\theta^*\|^2. \tag{2.2.7}
$$

光滑性同样在最小点给出上界

$$
\Phi(\theta)-\Phi^*\le\frac\beta2\|\theta-\theta^*\|^2.
$$

所以

$$
\|\theta_T-\theta^*\|^2
\le\frac2\mu(1-\mu\eta)^T\Delta_0, \tag{2.2.4}
$$

$$
\|\theta_T-\theta^*\|
\le\sqrt{\frac{2\Delta_0}{\mu}}(1-\mu\eta)^{T/2}. \tag{2.2.5}
$$

再用 $\Delta_0\le\beta\|\theta_0-\theta^*\|^2/2$，得到

$$
\|\theta_T-\theta^*\|
\le\sqrt{\frac\beta\mu}(1-\mu\eta)^{T/2}\|\theta_0-\theta^*\|. \tag{2.2.6}
$$

这也完成了定理证明。注意“范数”与“范数平方”的速率相差一个平方根。

定义条件数

$$
\kappa=\frac\beta\mu. \tag{2.2.8}
$$

取 $\eta=1/\beta$，由 $1-x\le e^{-x}$，

$$
\Delta_T\le e^{-T/\kappa}\Delta_0.
$$

因此 $T\ge\kappa\log(\Delta_0/\varepsilon)$ 足以保证 $\Delta_T\le\varepsilon$。一般定理覆盖的函数更多，界也比 §2.1 利用完整二次结构得到的精确界保守。

## 2.3 一般凸目标：用求和代替收缩

### 2.3.1 定理 2.10：固定步长 GD 的 $O(1/T)$ 速率

设 $\Phi$ 凸、$\beta$-光滑，存在最小点 $\theta^*$，并取

$$
\theta_{t+1}=\theta_t-\eta\nabla\Phi(\theta_t),\qquad 0<\eta\le1/\beta. \tag{2.3.1}
$$

则对 $T\ge1$，

$$
\Phi(\theta_T)-\Phi^*\le\frac{\|\theta_0-\theta^*\|^2}{2\eta T}. \tag{2.3.2}
$$

**完整证明。** 记 $g_t=\nabla\Phi(\theta_t)$、$D_t=\|\theta_t-\theta^*\|^2$。展开更新后的平方距离：

$$
D_{t+1}=D_t-2\eta\langle g_t,\theta_t-\theta^*\rangle+\eta^2\|g_t\|^2,
$$

即

$$
\langle g_t,\theta_t-\theta^*\rangle
=\frac{D_t-D_{t+1}}{2\eta}+\frac\eta2\|g_t\|^2. \tag{2.3.3}
$$

凸性给出 $\Delta_t\le\langle g_t,\theta_t-\theta^*\rangle$；下降引理给出 $\Delta_{t+1}\le\Delta_t-\eta\|g_t\|^2/2$。将两式相接，梯度平方项恰好消去：

$$
\Delta_{t+1}\le\frac{D_t-D_{t+1}}{2\eta}.
$$

求和时相邻距离相消：

$$
\sum_{t=0}^{T-1}\Delta_{t+1}\le\frac{D_0-D_T}{2\eta}\le\frac{D_0}{2\eta}.
$$

又因 GD 的目标值单调不增，$T\Delta_T\le\sum_{t=0}^{T-1}\Delta_{t+1}$，定理得证。

**关键结构：先控制误差总和，再利用单调性控制最后一次误差。** 后面 SGD 通常没有这种逐步单调性，所以会转向平均迭代。

令 $R^2=\|\theta_0-\theta^*\|^2$。要保证目标间隙不超过 $\varepsilon$，足够取 $T\ge R^2/(2\eta\varepsilon)$。相比强凸情形对 $1/\varepsilon$ 的对数依赖，这里是线性依赖：精度要求越高，缺少强凸性带来的代价越明显。这个定理控制的是最后一次迭代，不要求实际返回平均参数。

### 2.3.2 递减步长的作用

很多训练过程会逐渐降低学习率，但在没有梯度噪声时，这样做也会减少累计前进量。本段保留任意步长，就是为了看清这个代价：最后的界主要由 $\sum_t\eta_t$ 决定，而不只是由迭代次数决定。

若使用确定步长 $0<\eta_t\le1/\beta$，同一证明给出

$$
\theta_{t+1}=\theta_t-\eta_t\nabla\Phi(\theta_t),\qquad 0<\eta_t\le1/\beta. \tag{2.3.4}
$$

逐步有 $2\eta_t\Delta_{t+1}\le D_t-D_{t+1}$。求和后利用 $\Delta_{t+1}\ge\Delta_T$：

$$
2\Delta_T\sum_{t=0}^{T-1}\eta_t
\le D_0-D_T\le D_0.
$$

所以

$$
\Delta_T\le\frac{D_0}{2\sum_{t=0}^{T-1}\eta_t}. \tag{2.3.5}
$$

取 $\eta_t=c(t+1)^{-\alpha}$，$0<c\le1/\beta$。由于递减函数的矩形和不小于其下方积分，

$$
\sum_{k=1}^T k^{-\alpha}\ge\int_1^{T+1}x^{-\alpha}\,dx.
$$

于是，当 $0<\alpha<1$，

$$
\Delta_T\le\frac{(1-\alpha)D_0}{2c[(T+1)^{1-\alpha}-1]}
=O(T^{-(1-\alpha)}); \tag{2.3.6}
$$

当 $\alpha=1$，

$$
\Delta_T\le\frac{D_0}{2c\log(T+1)}. \tag{2.3.7}
$$

若还具有强凸性，则逐步应用 §2.2 的收缩式：

$$
\Delta_T\le\Delta_0\prod_{t=0}^{T-1}(1-\mu\eta_t)
\le\Delta_0\exp\left(-\mu\sum_{t=0}^{T-1}\eta_t\right). \tag{2.3.8}
$$

**本节结论。** 对这里的确定性、光滑凸问题，固定合适步长已经能收敛，递减步长并非必需。只根据目标间隙趋于零，不能在一般凸情形下直接推出到某个指定最小点的统一参数速率；这不等于断言 GD 迭代本身一定不收敛。

## 练习

1. 取 $H=\operatorname{diag}(1,9)$，比较步长 $1/9$ 与 $1/5$ 的最坏方向收缩因子；分别写出参数误差和目标间隙的界。
2. 对 $\Phi(x,y)=(x-2)^2/2$，从 $(0,3)$ 开始运行 GD。说明目标达到最优为何不代表趋于最小范数解。
3. 在一般凸的证明中，指出哪一步把平均间隙界变成最后一步的间隙界。

核对：第一题的参数因子分别为 $8/9$ 与 $4/5$；第二题的 $y$ 坐标始终为 3；第三题使用目标值的单调性，不能直接搬到有噪声的 SGD。
