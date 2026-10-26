# 07 · 集中不等式：从样本均值到随机矩阵

[中文目录](README.md) · [English](../../README.md) · [下一节：一致收敛](08-uniform-convergence.md)

独立观测的平均值通常比单次观测稳定。集中不等式说明这种稳定性依赖哪些分布条件，以及偏差、样本量和失败概率之间如何换算。有限方差给出多项式尾界；有界性或指数矩条件可以给出指数尾界。不能只比较公式中的收敛速度而忽略假设。

本节中 $E$、$P$ 分别表示期望和概率，$\|X\|_p=(E|X|^p)^{1/p}$，$\log$ 为自然对数。随机变量均为实值，出现的期望须存在；除非特别说明，不要求同分布。常数 $C,c>0$ 可在不同公式中改变，但不依赖样本量、维数及概率参数。退化的零方差情形单独按常量理解。

## 1. Markov、Chebyshev 与均值误差

若 $V\ge0$，则对 $a>0$，

$$
P(V\ge a)\le\frac{EV}{a}.
$$

证明只需对逐点不等式 $V\ge a\mathbf1_{\{V\ge a\}}$ 取期望。令 $V=(X-EX)^2$ 即得 Chebyshev 不等式

$$
P(|X-EX|\ge t)\le\frac{\operatorname{Var}(X)}{t^2}.
$$

设 $X_1,\ldots,X_n$ 独立，$EX_i=\mu$、$\operatorname{Var}(X_i)=\sigma^2$，记 $\bar X_n=n^{-1}\sum_iX_i$。独立性使不同中心变量的交叉项为零，故

$$
E(\bar X_n-\mu)^2=\frac{\sigma^2}{n},\qquad
E|\bar X_n-\mu|\le\frac{\sigma}{\sqrt n},\qquad
P(|\bar X_n-\mu|\ge t)\le\frac{\sigma^2}{nt^2}.
$$

中间一式来自 Cauchy–Schwarz。给定 $\delta\in(0,1)$，最后一式可改写为：以至少 $1-\delta$ 的概率，$|\bar X_n-\mu|\le\sigma/\sqrt{n\delta}$。这里均方误差为 $n^{-1}$ 阶，绝对误差半径为 $n^{-1/2}$ 阶，两者不能混称。

## 2. Hoeffding 引理与有界独立和

**Hoeffding 引理。** 若 $X\in[a,b]$ 几乎处处，则对所有 $\lambda\in\mathbb R$，

$$
\log E e^{\lambda(X-EX)}\le\frac{\lambda^2(b-a)^2}{8}.
\tag{1}
$$

**证明。** 令 $h(\lambda)=\log E e^{\lambda X}$。有界性允许在期望内求导。在以 $e^{\lambda X}/Ee^{\lambda X}$ 为密度的倾斜概率下，

$$
h'(\lambda)=E_\lambda X,\qquad h''(\lambda)=\operatorname{Var}_\lambda(X).
$$

任意取值于 $[a,b]$ 的变量 $W$ 都满足

$$
\operatorname{Var}(W)=\min_c E(W-c)^2
\le E\left(W-\frac{a+b}{2}\right)^2\le\frac{(b-a)^2}{4}.
$$

倾斜不改变取值区间，因此 $h''(\lambda)\le(b-a)^2/4$。对 $h(\lambda)-h(0)-\lambda h'(0)$ 使用带积分余项的 Taylor 公式，正负 $\lambda$ 均得 (1)。$a=b$ 时两边为零。

**Hoeffding 不等式。** 若 $X_i$ 独立且 $X_i\in[a_i,b_i]$ 几乎处处，令 $D=\sum_i(b_i-a_i)^2>0$，则

$$
P\left(\left|\sum_i(X_i-EX_i)\right|\ge t\right)
\le2\exp\left(-\frac{2t^2}{D}\right).
\tag{2}
$$

**证明。** 独立性和 (1) 给出

$$
E\exp\left(\lambda\sum_i(X_i-EX_i)\right)
\le\exp(\lambda^2D/8).
$$

对 $\lambda>0$ 使用 Markov，右尾概率不超过 $\exp(-\lambda t+\lambda^2D/8)$；取 $\lambda=4t/D$ 得 $\exp(-2t^2/D)$。对负和重复一次，再合并两侧事件。

若共同区间为 $[a,b]$，则

$$
P(|\bar X_n-E\bar X_n|\ge t)
\le2\exp\left(-\frac{2nt^2}{(b-a)^2}\right).
$$

这一步只需要独立，不需要同分布。区间长度而非端点绝对值决定方差代理；对 $[0,1]$ 变量有代理 $1/4$。

## 3. McDiarmid：条件区间长度为何仍是 $c_i$

**有界差分不等式。** 设 $X_1,\ldots,X_n$ 独立，$f$ 可测且可积。对每个 $i$，仅改变第 $i$ 个坐标时都有

$$
|f(x_1,\ldots,x_i,\ldots,x_n)-f(x_1,\ldots,x_i',\ldots,x_n)|\le c_i.
\tag{3}
$$

则对 $t>0$，

$$
P(f(X)-Ef(X)\ge t)\le
\exp\left(-\frac{2t^2}{\sum_i c_i^2}\right),
\tag{4}
$$

双侧界右端乘 $2$。假设可在各坐标的取值支持集上成立；若仅几乎处处成立，需要选择满足相应条件区间界的版本。

**证明。** 令 $\mathcal F_i=\sigma(X_1,\ldots,X_i)$，定义 Doob 鞅

$$
M_i=E[f(X)\mid\mathcal F_i],\qquad D_i=M_i-M_{i-1}.
$$

于是 $f(X)-Ef(X)=\sum_iD_i$，且 $E[D_i\mid\mathcal F_{i-1}]=0$。固定过去 $u=(x_1,\ldots,x_{i-1})$，对未来坐标积分，记

$$
q_i(u,z)=E_{X_{i+1:n}}f(u,z,X_{i+1:n}).
$$

由独立性，未来的积分分布不随 $z$ 改变；用同一组未来坐标比较两项，可得

$$
|q_i(u,z)-q_i(u,z')|\le E_{X_{i+1:n}}
|f(u,z,X_{i+1:n})-f(u,z',X_{i+1:n})|\le c_i.
$$

固定 $u$ 后，任意两个可能值之差都不超过 $c_i$，所以

$$
\sup_z q_i(u,z)-\inf_z q_i(u,z)
=\sup_{z,z'}\{q_i(u,z)-q_i(u,z')\}\le c_i.
$$

即使上下确界没有取到，也可分别用逼近上下确界的值验证这一等式；对条件分布则使用本质上下确界。因此 $q_i(u,X_i)$ 在给定过去后的本质取值区间长度至多为 $c_i$。减去条件均值只平移这个区间，不改变长度。换言之，$D_i\mid\mathcal F_{i-1}$ 落在某个长度至多 $c_i$、可随过去改变的区间中。虽然粗界 $|D_i|\le c_i$ 也成立，但若直接套区间 $[-c_i,c_i]$，会无端损失常数。

条件 Hoeffding 引理给出

$$
E[e^{\lambda D_i}\mid\mathcal F_{i-1}]
\le e^{\lambda^2c_i^2/8}.
$$

逐次条件化，

$$
E e^{\lambda\sum_iD_i}
\le e^{\lambda^2c_n^2/8}E e^{\lambda\sum_{i<n}D_i}
\le e^{\lambda^2\sum_i c_i^2/8}.
$$

再作与 (2) 相同的 Chernoff 优化即得 (4)。鞅差一般并不独立；上面的逐次条件化代替了错误的无条件期望乘积。

**例。** 对 $g:\mathcal Z\to[0,1]$ 的预先固定函数类，令

$$
\Delta(S)=\sup_g\left|\frac1n\sum_i g(Z_i)-Eg(Z)\right|.
$$

只替换一个观测时，各个经验均值最多改变 $1/n$。利用 $|\sup a_g-\sup b_g|\le\sup|a_g-b_g|$，得 $|\Delta(S)-\Delta(S^{(i)})|\le1/n$。若上确界可测，McDiarmid 因而给出 $P(\Delta-E\Delta\ge t)\le e^{-2nt^2}$。剩下的难点是控制 $E\Delta$，留待下一节处理。

## 4. Bernstein：保留实际方差

设 $Y_i$ 独立、$EY_i=0$、$|Y_i|\le b$，并令 $v=\sum_iEY_i^2$。当 $v>0$ 时，

$$
P\left(\left|\sum_iY_i\right|\ge t\right)
\le2\exp\left[-\frac{t^2}{2(v+bt/3)}\right].
\tag{5}
$$

**证明。** 对 $k\ge2$，$E|Y_i|^k\le b^{k-2}EY_i^2$，且 $k!\ge2\cdot3^{k-2}$；后者由 $k=2,3$ 的等式及逐次乘以至少 $3$ 的因子得到。于是当 $0\le\lambda<3/b$，

$$
\begin{aligned}
Ee^{\lambda Y_i}
&\le1+EY_i^2\sum_{k=2}^{\infty}\frac{\lambda^kb^{k-2}}{k!}\\
&\le1+\frac{\lambda^2EY_i^2}{2(1-\lambda b/3)},
\end{aligned}
$$

故 $\log Ee^{\lambda\sum_iY_i}\le\lambda^2v/[2(1-\lambda b/3)]$。Markov 后选取 $\lambda=t/(v+bt/3)<3/b$，得到右尾指数 $-t^2/[2(v+bt/3)]$。对 $-Y_i$ 重复即可。$v=0$ 时各项几乎处处为零。

令 $q=\log(2/\delta)$，解二次不等式可得一个便于使用的版本：以至少 $1-\delta$ 的概率，

$$
\left|\sum_iY_i\right|\le\sqrt{2vq}+\frac{2bq}{3}.
\tag{6}
$$

因为右端 $t$ 满足 $t^2\ge2q(v+bt/3)$，代回 (5) 即可。对同分布 $X_i$，若 $|X_i-EX_i|\le b$、方差为 $\sigma^2$，则均值半径为

$$
\sigma\sqrt{\frac{2\log(2/\delta)}n}
+\frac{2b\log(2/\delta)}{3n}.
$$

Hoeffding 不需要实际方差；Bernstein 在方差远小于范围平方时更精细。若变量没有有界性，不能仅凭有限方差套用 (5)。

## 5. 只有有限高阶矩时：偶数阶 Rosenthal 界

对固定整数 $m\ge1$，独立中心变量 $Y_i$ 若满足 $E|Y_i|^{2m}<\infty$，则存在只依赖 $m$ 的常数 $R_m$，使

$$
E\left|\sum_iY_i\right|^{2m}
\le R_m\left\{\sum_iE|Y_i|^{2m}+\left(\sum_iEY_i^2\right)^m\right\}.
\tag{7}
$$

这里证明固定偶数阶版本，不声称给出所有实数 $p\ge2$ 的最优常数。

**证明。** $m=1$ 时由方差相加直接成立。设 $m\ge2$，记

$$
A=\sum_iEY_i^2,\quad B=\sum_iE|Y_i|^{2m},\quad
M_r=\sum_iE|Y_i|^r.
$$

对 $2\le r\le2m$，令 $\theta_r=(r-2)/(2m-2)$。先对随机变量用 Hölder，再对指标求和用 Hölder，得到

$$
M_r\le A^{1-\theta_r}B^{\theta_r}.
\tag{8}
$$

端点直接成立；若 $A=0$ 或 $B=0$，所有变量为零。展开 $E(\sum_iY_i)^{2m}$。只出现一次的指标因独立性和中心化贡献为零。把 $2m$ 个位置按相同指标分组，剩余项对应每组至少两个位置的集合划分。

固定一个划分，其组大小为 $r_1,\ldots,r_k\ge2$，$\sum_jr_j=2m$，故 $k\le m$。对其贡献取绝对值，并删去不同组所用指标须互异的限制，得上界 $\prod_jM_{r_j}$。由 (8)，

$$
\begin{aligned}
\prod_jM_{r_j}
&\le A^{k-\sum_j\theta_{r_j}}B^{\sum_j\theta_{r_j}}\\
&=(A^m)^{(k-1)/(m-1)}B^{(m-k)/(m-1)}\le A^m+B.
\end{aligned}
$$

末步用加权算术—几何平均不等式。这样的集合划分只有有限种，数目只依赖 $m$；取 $R_m$ 为它们的总数，求和即得 (7)。例如

$$
E\left(\sum_iY_i\right)^4
=\sum_iEY_i^4+6\sum_{i<j}EY_i^2EY_j^2.
$$

若另外 $\|Y_i\|_{2m}\le K_m$，则 Jensen 给出 $EY_i^2\le K_m^2$，所以

$$
P\left(\left|\frac1n\sum_iY_i\right|\ge t\right)
\le\frac{2R_mK_m^{2m}}{n^mt^{2m}}.
\tag{9}
$$

这是 (7) 除以 $n^{2m}t^{2m}$ 后的 Markov 界。固定 $\delta$ 时误差半径仍为 $n^{-1/2}$ 阶；固定 $t$ 时失败概率至多为 $n^{-m}$ 阶。常数依赖 $m$，不能让 $m$ 随 $n$ 增长而忽略这种依赖。

## 6. 次高斯与次指数

### 6.1 定义及尾界

对中心变量 $Z=X-EX$，若存在 $\sigma>0$ 使

$$
\log Ee^{\lambda Z}\le\frac{\sigma^2\lambda^2}{2},
\qquad\lambda\in\mathbb R,
\tag{10}
$$

称 $Z$ 次高斯，$\sigma^2$ 为方差代理。若存在 $\nu,b>0$，同一界只要求在 $|\lambda|<1/b$ 成立，称其为参数 $(\nu^2,b)$ 的次指数变量。本节“次指数”指 $\psi_1$ 类型的指数尾，不指重尾分布文献中同名的次指数分布族。

指数 Markov 和参数优化分别给出

$$
P(|Z|\ge t)\le2e^{-t^2/(2\sigma^2)},
\tag{11}
$$

$$
P(|Z|\ge t)\le2\exp\left[-\frac12\min\left(\frac{t^2}{\nu^2},\frac tb\right)\right].
\tag{12}
$$

对 (11)，取 $\lambda=t/\sigma^2$。对 (12)，当 $t<\nu^2/b$ 取 $\lambda=t/\nu^2$；当 $t\ge\nu^2/b$，让 $\lambda\uparrow1/b$，所得指数 $-t/b+\nu^2/(2b^2)\le-t/(2b)$。只对数值上界取极限，不需要边界指数矩有限。对负变量重复并合并两尾。

正态变量 $N(0,\sigma^2)$ 的指数矩恰好等于 $e^{\sigma^2\lambda^2/2}$；Hoeffding 引理说明有界中心变量也是次高斯。反向不成立：若 $E\sim\operatorname{Exp}(1)$，则

$$
Ee^{\lambda(E-1)}=\frac{e^{-\lambda}}{1-\lambda},\quad\lambda<1.
$$

当 $|\lambda|<1/2$，$-\lambda-\log(1-\lambda)\le\lambda^2$，故它次指数；$\lambda\ge1$ 时指数矩发散，故不是次高斯。

### 6.2 尾、矩与 Orlicz 刻画的等价性

对 $\alpha\in\{1,2\}$ 定义

$$
\|W\|_{\psi_\alpha}
=\inf\{K>0:E\exp(|W|^\alpha/K^\alpha)\le2\}.
$$

下面三种条件等价，尺度只相差依赖于 $\alpha$ 的绝对常数：

1. $\|W\|_{\psi_\alpha}\le K$；
2. $P(|W|>t)\le2\exp[-(t/K)^\alpha]$；
3. 对所有 $p\ge1$，$\|W\|_p\le Kp^{1/\alpha}$。

此处“等价”不表示三个条件可原封不动使用同一个 $K$。以下证明给出常数变化的来源。

**Orlicz 推尾界。** 对 $K$ 略大于范数先用 Markov，再令其下降到范数。范数为零时 $W=0$ 几乎处处。

**尾界推矩界。** Tonelli 给出尾积分公式

$$
E|W|^p=p\int_0^\infty t^{p-1}P(|W|>t)\,dt
\le2K^p\Gamma(1+p/\alpha).
$$

对 $q>0$，$u^qe^{-u/2}$ 在 $u=2q$ 处最大，故

$$
\Gamma(q+1)=\int_0^\infty u^qe^{-u}\,du
\le2(2q/e)^q.
$$

取 $p$ 次方根便得 $\|W\|_p\le CKp^{1/\alpha}$。

**矩界推 Orlicz。** 展开非负幂级数，使用 $k!\ge(k/e)^k$：

$$
E e^{|W|^\alpha/A^\alpha}
\le1+\sum_{k=1}^\infty
\frac{K^{\alpha k}(\alpha k)^k}{A^{\alpha k}k!}
\le1+\sum_{k=1}^\infty\left(\frac{\alpha eK^\alpha}{A^\alpha}\right)^k.
$$

取 $A^\alpha=2\alpha eK^\alpha$，右端为 $2$。阶乘下界也可直接由 $\sum_{j=1}^k\log j\ge\int_1^k\log x\,dx\ge k\log k-k$ 验证。

对中心变量，这些条件还分别等价于 (10) 的全局二次指数矩界和其局部版本；这里补上反方向，避免把等价性当作额外假设。

**$\psi_2$ 推全局指数矩。** 设 $EW=0$ 且 $Ee^{W^2/K^2}\le2$，取独立同分布副本 $W'$，令 $V=W-W'$。由 $(W-W')^2\le2W^2+2(W')^2$ 及 Cauchy–Schwarz，

$$
Ee^{V^2/(4K^2)}\le(Ee^{W^2/(2K^2)})^2\le2.
$$

$V$ 对称，且上式确保所有线性指数矩有限。非负级数给出 $EV^{2k}\le2(4K^2)^kk!$。利用 $(2k)!\ge(k!)^2$，

$$
Ee^{\lambda V}=E\cosh(\lambda V)
\le1+2\sum_{k\ge1}\frac{(4K^2\lambda^2)^k}{k!}
=2e^{4K^2\lambda^2}-1\le e^{8K^2\lambda^2}.
$$

最后 $e^{\lambda W}=e^{\lambda E_{W'}(W-W')}\le E_{W'}e^{\lambda(W-W')}$，故 $Ee^{\lambda W}\le e^{8K^2\lambda^2}$。这给出一个非最优但统一的代理 $16K^2$。反方向由 (11) 和前述尾界等价性得到。

**$\psi_1$ 推局部指数矩。** 若 $EW=0$、$Ee^{|W|/K}\le2$，则 $E|W|^k\le2K^kk!$。当 $|\lambda|<1/(2K)$，

$$
Ee^{\lambda W}\le1+2\sum_{k\ge2}(K|\lambda|)^k
\le1+4K^2\lambda^2\le e^{4K^2\lambda^2}.
$$

因此可取 $(\nu^2,b)=(8K^2,2K)$。反方向，(12) 在 $t\ge\nu$ 时不超过 $2e^{-t/(2\max(\nu,b))}$；在 $t<\nu$ 时该表达式大于 $1$，尾界仍成立。再用上述等价性，得 $\|W\|_{\psi_1}\le C\max(\nu,b)$。

### 6.3 平方、乘积与中心化

不要求 $U,V$ 独立，也有

$$
\|U^2\|_{\psi_1}=\|U\|_{\psi_2}^2,
\qquad
\|UV\|_{\psi_1}\le\|U\|_{\psi_2}\|V\|_{\psi_2}.
\tag{13}
$$

第一式直接比较定义。对第二式，取稍大于相应范数的 $K_U,K_V$，用 $2ab\le a^2+b^2$ 及 Cauchy–Schwarz，

$$
Ee^{|UV|/(K_UK_V)}
\le E e^{U^2/(2K_U^2)+V^2/(2K_V^2)}
\le\sqrt{Ee^{U^2/K_U^2}Ee^{V^2/K_V^2}}\le2.
$$

再让尺度下降到范数。不能在没有独立性时把两个指数的期望直接相乘。

若 $Ee^{|W|/K}\le2$，Jensen 给 $|EW|\le K\log2$，于是

$$
Ee^{|W-EW|/(2K)}
\le e^{|EW|/(2K)}(Ee^{|W|/K})^{1/2}\le2.
$$

所以 $\|W-EW\|_{\psi_1}\le2\|W\|_{\psi_1}$。特别地，次高斯变量的平方、两个次高斯变量的乘积，中心化后都是次指数；不能据此断言它们仍次高斯。

## 7. 独立加权和与最大值

若 $Z_i$ 独立、中心化、次高斯代理为 $\sigma_i^2$，$a_i$ 为确定实数，则

$$
\log Ee^{\lambda\sum_i a_iZ_i}
=\sum_i\log Ee^{\lambda a_iZ_i}
\le\frac{\lambda^2}{2}\sum_i a_i^2\sigma_i^2.
$$

故

$$
P\left(\left|\sum_i a_iZ_i\right|\ge t\right)
\le2\exp\left[-\frac{t^2}{2\sum_i a_i^2\sigma_i^2}\right].
\tag{14}
$$

若各项为次指数参数 $(\nu_i^2,b_i)$，同一证明在 $|\lambda|<1/\max_i|a_i|b_i$ 有效，因此和的参数可取

$$
\nu_*^2=\sum_i a_i^2\nu_i^2,\qquad b_*=\max_i|a_i|b_i.
$$

把它们代入 (12) 即得对应尾界。独立同分布均值的参数为 $(\nu^2/n,b/n)$，从而以至少 $1-\delta$ 的概率，

$$
|\bar X_n-EX_1|
\le\nu\sqrt{\frac{2\log(2/\delta)}n}
+\frac{2b\log(2/\delta)}n.
$$

两个阈值之和同时不小于二者，保证 (12) 的两个指数项均达到 $\log(2/\delta)$。

**最大值。** 若中心变量 $Z_1,\ldots,Z_M$ 的次高斯代理均不超过 $\sigma^2$，不要求相互独立，则

$$
E\max_{j\le M}|Z_j|\le\sigma\sqrt{2\log(2M)}.
\tag{15}
$$

对 $\lambda>0$，Jensen 和最大值不超过和给出

$$
e^{\lambda E\max_j|Z_j|}
\le\sum_j(Ee^{\lambda Z_j}+Ee^{-\lambda Z_j})
\le2M e^{\lambda^2\sigma^2/2}.
$$

取对数后最小化 $\log(2M)/\lambda+\lambda\sigma^2/2$ 即得 (15)。若不取绝对值，$M\ge2$ 时同理得 $E\max_jZ_j\le\sigma\sqrt{2\log M}$。

对 $M$ 个独立标准正态变量，这个后一个上界渐近精确。记最大值为 $M_M$。正态尾在 $t\ge1$ 时满足

$$
1-\Phi(t)\ge\int_t^{t+1/t}\varphi(u)\,du
\ge\frac{c}{t}e^{-t^2/2}.
$$

取 $t_M=(1-\eta)\sqrt{2\log M}$，$0<\eta<1$，则 $M(1-\Phi(t_M))\to\infty$。独立性给 $P(M_M<t_M)=\Phi(t_M)^M\to0$。又

$$
E(-M_M)_+\le E(-Z_1)_+\,2^{-(M-1)},
$$

因为负最大值要求所有变量均为负。于是 $EM_M\ge t_MP(M_M\ge t_M)-E(-M_M)_+$。除以 $\sqrt{2\log M}$，让 $M\to\infty$、再令 $\eta\downarrow0$，得 $EM_M\sim\sqrt{2\log M}$。这个下界需要独立性。

若仅知 $\|Z_j\|_{\psi_1}\le K$，并集界给 $P(\max_j|Z_j|>t)\le\min(1,2Me^{-t/K})$。在 $a=K\log(2M)$ 处分开尾积分，得

$$
E\max_j|Z_j|\le a+2M\int_a^\infty e^{-t/K}\,dt
=K\{\log(2M)+1\}.
$$

因此次高斯与次指数最大值的代价分别为 $\sqrt{\log M}$ 与 $\log M$。例如独立 $\operatorname{Exp}(1)$ 变量的最大值期望恰为 $\sum_{k=1}^M1/k\sim\log M$：将其分布函数 $(1-e^{-t})^M$ 代入尾积分，再作 $u=e^{-t}$ 的换元与有限等比展开即可验证。

## 8. 应用：次高斯随机矩阵的算子范数

设 $A\in\mathbb R^{m\times d}$ 的元素独立、中心化，次高斯代理均为 $\sigma^2$。对任意 $\delta\in(0,1)$，以至少 $1-\delta$ 的概率，

$$
\|A\|_{\mathrm{op}}
\le2\sigma\sqrt{2\{(m+d)\log9+\log(1/\delta)\}}.
\tag{16}
$$

**证明。** 算子范数等于

$$
\|A\|_{\mathrm{op}}=\sup_{\|x\|_2=\|y\|_2=1}y^\top Ax.
$$

分别在 $\mathbb S^{d-1}$ 和 $\mathbb S^{m-1}$ 取极大的 $1/4$-分离集 $N_d,N_m$。以其点为中心的半径 $1/8$ 欧氏开球互不相交，且都位于半径 $9/8$ 的大球内；比较整个环境空间中的体积，得 $|N_d|\le9^d$、$|N_m|\le9^m$。有限体积上界保证选点过程终止，极大性保证它们是 $1/4$-网。

对任意单位向量 $x,y$，取网点 $x_0,y_0$ 使距离均不超过 $1/4$，则

$$
\begin{aligned}
y^\top Ax-y_0^\top Ax_0
&=(y-y_0)^\top Ax+y_0^\top A(x-x_0),\\
|y^\top Ax-y_0^\top Ax_0|&\le\tfrac12\|A\|_{\mathrm{op}}.
\end{aligned}
$$

对所有方向取上确界并移项，得到

$$
\|A\|_{\mathrm{op}}\le2\max_{x\in N_d,y\in N_m}y^\top Ax.
$$

固定网点后，$y^\top Ax=\sum_{i,j}y_ix_jA_{ij}$。由独立加权和公式，其代理为

$$
\sigma^2\sum_{i,j}y_i^2x_j^2
=\sigma^2\left(\sum_i y_i^2\right)\left(\sum_jx_j^2\right)=\sigma^2.
$$

故右尾概率至多 $e^{-s^2/(2\sigma^2)}$。对至多 $9^{m+d}$ 对网点取并集界，

$$
P(\|A\|_{\mathrm{op}}>2s)\le9^{m+d}e^{-s^2/(2\sigma^2)}.
$$

解右侧等于 $\delta$ 得 (16)。这里既没有对不可数方向直接求和，也没有要求不同网点方向的随机变量相互独立。若 $A_{ij}\sim N(0,1/m)$，上界为 $C[1+\sqrt{d/m}+\sqrt{\log(1/\delta)/m}]$ 阶。

## 9. 练习

1. 独立 $X_i\in[0,1]$，求使 $P(|\bar X_n-E\bar X_n|>0.05)\le0.01$ 的充分样本量。若只知 $\operatorname{Var}(X_i)\le1/4$，Chebyshev 给出多大的数？
2. 证明对任意非空有限集合 $T$，函数 $f(x_1,\ldots,x_n)=\max_{t\in T}n^{-1}\sum_i g_t(x_i)$，$g_t\in[0,1]$，具有有界差分常数 $1/n$。说明这一事实为什么尚不能控制 $Ef$。
3. 若独立中心变量有共同次高斯代理 $\sigma^2$，对 $\sum_i a_iX_i$ 给出误差半径。把系数改为依赖同一组 $X_i$ 的随机数后，证明的哪一步失效？
4. 令 $X\sim N(0,1)$。用 (13) 证明 $X^2-1$ 次指数，并说明它不次高斯。

**答案线索。** 1. Hoeffding 给 $n\ge\log(200)/(2\cdot0.05^2)$，取整为 $1060$；Chebyshev 给 $n\ge10000$。2. 用最大值之差不超过各项差的最大值；期望还受候选类丰富程度影响。3. 半径为 $\sigma\sqrt{2\sum_i a_i^2\log(2/\delta)}$；随机系数通常不能提出期望，也破坏原来的独立加权和分解。4. $Ee^{\lambda X^2}$ 在 $\lambda\ge1/2$ 时发散，中心化不改变这一事实。
