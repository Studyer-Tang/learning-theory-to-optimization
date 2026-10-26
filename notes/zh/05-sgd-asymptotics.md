# 05 · SGD 的渐近分布与迭代平均

[中文目录](README.md) · [English](../../README.md)

均方误差趋于零说明估计逐渐接近目标，却不能直接给出置信区间。区间还需要知道误差在什么尺度下具有何种分布。本篇分别证明 $a/t$ 步长下最后一次迭代的正态极限，以及 $at^{-\alpha}$、$1/2<\alpha<1$ 步长下算术平均的正态极限。两组步长条件不可互换。

## 1. 局部模型及证明所用条件

设总体目标为 $\Phi:\mathbb R^d\to\mathbb R$，维数 $d$ 固定，$\theta^*$ 为驻点。SGD 写为

$$
\theta_{t+1}=\theta_t-\eta_t\widehat g_{t+1}(\theta_t),
\qquad e_t=\theta_t-\theta^*.
$$

$\mathcal F_t$ 包含第 $t$ 次更新之前的全部信息，$\theta_t$ 为 $\mathcal F_t$ 可测。定义

$$
\xi_{t+1}=\widehat g_{t+1}(\theta_t)-\nabla\Phi(\theta_t),
\qquad C_t=E[\xi_{t+1}\xi_{t+1}^\top\mid\mathcal F_t].
$$

以下用一组较强但便于检查的充分条件，避免把局部展开误当作全局收敛证明：

1. $H=\nabla^2\Phi(\theta^*)=H^\top\succ0$，$\nabla\Phi(\theta^*)=0$，沿轨迹有
   $\nabla\Phi(\theta_t)=He_t+r_t$。
2. 对各自使用的步长，$E\|e_t\|^2\le K\eta_t$、$E\|r_t\|\le K\eta_t$；初值不随运行长度 $T$ 改变且 $E\|e_1\|^2<\infty$。
3. $E[\xi_{t+1}\mid\mathcal F_t]=0$，$C_t\xrightarrow{p}S$，其中 $S$ 为确定的对称半正定矩阵。
4. 存在 $\delta>0$ 和确定常数 $K<\infty$，使所有 $t$ 满足

$$
E[\|\xi_{t+1}\|^{2+\delta}\mid\mathcal F_t]\le K
\quad\text{a.s.}
$$

这里与后文的 $K,C$ 均可随模型改变，但不随 $t,T$ 改变。向量用 Euclidean 范数，矩阵用算子范数。$X_T=o_p(1)$ 表示 $X_T\xrightarrow{p}0$。

若 Hessian 在相关邻域内 Lipschitz 连续，则

$$
\begin{aligned}
r_t
&=\nabla\Phi(\theta^*+e_t)-\nabla\Phi(\theta^*)-He_t\\
&=\int_0^1[\nabla^2\Phi(\theta^*+se_t)-H]e_t\,ds,
\end{aligned}
$$

从而 $\|r_t\|\le L\|e_t\|^2/2$。**只有当轨迹受到适当控制时**，这个局部界与均方速率才能推出条件 2 中的期望余项界。仅有 $\theta_t\to\theta^*$ 依概率，不足以控制离开邻域的低概率大误差。本文直接假设上述可用的期望界；不另外声称已经证明稳定性或局部化步骤。

最后需要以下概率定理，本文将它作为外部基础结果使用。

> **鞅差三角阵 CLT。** 设每一行 $(Z_{T,t})_t$ 是相应滤过下的平方可积向量鞅差。若
>
> $$
> \sum_tE[Z_{T,t}Z_{T,t}^\top\mid\mathcal F_t]\xrightarrow{p}V,
> $$
>
> 且任意 $\varepsilon>0$ 都有
>
> $$
> \sum_tE[\|Z_{T,t}\|^2\mathbf1_{\{\|Z_{T,t}\|>\varepsilon\}}
> \mid\mathcal F_t]\xrightarrow{p}0,
> $$
>
> 其中 $V$ 是确定矩阵，则 $\sum_t Z_{T,t}\Rightarrow N(0,V)$。

下文验证这两个条件。计算极限协方差只是其中一步，不能单独代替 CLT 的证明。

## 2. 最后一次迭代：展开递推并控制余项

取 $\eta_t=a/t$，$a>0$，并记

$$
h=a\lambda_{\min}(H)>\frac12.
$$

在第 1 节条件下，结论为

$$
\boxed{\sqrt T(\theta_T-\theta^*)\Rightarrow N(0,V_a),}
$$

其中

$$
\boxed{
V_a=a^2\int_0^1x^{aH-I}Sx^{aH^\top-I}\,dx.
}
$$

矩阵幂定义为 $x^A=\exp(A\log x)$，$x>0$。后面还将把这个积分写成可直接求解的矩阵方程。

由局部模型，

$$
e_{t+1}=A_te_t-\frac at\xi_{t+1}-\frac at r_t,
\qquad A_t=I-\frac{aH}{t}.
$$

前两步为

$$
e_2=A_1e_1-a(\xi_2+r_1),
$$

$$
e_3=A_2A_1e_1-aA_2(\xi_2+r_1)-\frac a2(\xi_3+r_2).
$$

因此第 $t$ 次加入的扰动在到达 $e_T$ 之前，还要经过 $A_{t+1},\ldots,A_{T-1}$。定义

$$
\Gamma_{T,t}=A_{T-1}\cdots A_{t+1},\qquad
\Gamma_{T,0}=A_{T-1}\cdots A_1,
$$

并约定空乘积为 $I$。归纳代入即可得到精确恒等式

$$
\begin{aligned}
\sqrt T e_T
&=\sqrt T\Gamma_{T,0}e_1
-\sum_{t=1}^{T-1}W_{T,t}\xi_{t+1}
-\sum_{t=1}^{T-1}W_{T,t}r_t,\\
W_{T,t}&=\frac{a\sqrt T}{t}\Gamma_{T,t}.
\end{aligned}
$$

先给出传播矩阵的界。将 $H$ 正交对角化为

$$
H=Q\operatorname{diag}(\lambda_1,\ldots,\lambda_d)Q^\top.
$$

由于所有 $A_j$ 都是 $H$ 的多项式，它们可以同时对角化，故

$$
\|\Gamma_{T,t}\|
=\max_i\left|\prod_{j=t+1}^{T-1}(1-a\lambda_i/j)\right|.
$$

选固定整数 $J>2a\lambda_{\max}(H)$。当 $t+1\ge J$ 时，每个因子都在 $(1/2,1)$ 内；利用 $\log(1-u)\le-u$ 及积分比较，

$$
\begin{aligned}
\log\prod_{j=t+1}^{T-1}(1-a\lambda_i/j)
&\le-a\lambda_i\sum_{j=t+1}^{T-1}\frac1j\\
&\le-a\lambda_i\log\frac{T}{t+1}.
\end{aligned}
$$

于是乘积不超过 $((t+1)/T)^{a\lambda_i}\le((t+1)/T)^h$。若 $t+1<J$，只需把 $j<J$ 的固定有限个因子并入常数，剩下的乘积由同一估计控制。再扩大常数覆盖有限个 $T<J$，得到

$$
\|\Gamma_{T,t}\|\le C\left(\frac{t+1}{T}\right)^h,
\qquad \|\Gamma_{T,0}\|\le CT^{-h}.
$$

因此

$$
E\|\sqrt T\Gamma_{T,0}e_1\|
\le CT^{1/2-h}E\|e_1\|\longrightarrow0.
$$

阈值 $h>1/2$ 正是要求最慢方向的初值影响比 $T^{-1/2}$ 衰减更快。

因为 $t+1\le2t$，权重满足

$$
\|W_{T,t}\|\le CT^{1/2-h}t^{h-1}.
$$

又由 $E\|r_t\|\le C/t$，

$$
E\left\|\sum_{t=1}^{T-1}W_{T,t}r_t\right\|
\le CT^{1/2-h}\sum_{t=1}^{T-1}t^{h-2}.
$$

若 $1/2<h<1$，右边的级数有界，总量为 $O(T^{1/2-h})$；若 $h=1$，总量为 $O(T^{-1/2}\log T)$；若 $h>1$，幂和为 $O(T^{h-1})$，总量为 $O(T^{-1/2})$。三个情形都趋于零。Markov 不等式给出初值项及总非线性项均为 $o_p(1)$，故

$$
\sqrt T e_T=-M_T+o_p(1),\qquad
M_T=\sum_{t=1}^{T-1}W_{T,t}\xi_{t+1}.
$$

上述计算只用三角不等式，没有要求不同时刻的 $r_t$ 独立。

## 3. 噪声和：条件协方差、矩阵积分与 Lindeberg 条件

由权重界以及 $2h-2>-1$，有

$$
\sum_{t<T}\|W_{T,t}\|^2
\le CT^{1-2h}\sum_{t<T}t^{2h-2}\le C.
$$

同时，

$$
\max_{t<T}\|W_{T,t}\|\le
\begin{cases}
CT^{1/2-h},&1/2<h<1,\\
CT^{-1/2},&h\ge1,
\end{cases}
\quad\longrightarrow0.
$$

更精确地，对 $0<\varepsilon<1$，

$$
\limsup_{T\to\infty}\sum_{t<\varepsilon T}\|W_{T,t}\|^2
\le C\varepsilon^{2h-1}.
$$

以下分别使用这三个估计控制总量、单步贡献和靠近时间零点的贡献。

鞅差结构保证 $E[W_{T,t}\xi_{t+1}\mid\mathcal F_t]=0$。若 $s<t$，则

$$
E[\xi_{s+1}\xi_{t+1}^\top]
=E[\xi_{s+1}E(\xi_{t+1}^\top\mid\mathcal F_t)]=0.
$$

这给出不同时刻的二阶正交性，并不说明独立。特别地，CLT 要验证的矩阵是随机的可预测二次变差

$$
Q_T=\sum_{t<T}W_{T,t}C_tW_{T,t}^\top,
$$

而普通无条件协方差为

$$
\operatorname{Cov}(M_T)=\sum_{t<T}W_{T,t}E[C_t]W_{T,t}^\top.
$$

二者不能混写。

先把 $C_t\xrightarrow{p}S$ 加强为 $L^1$ 收敛。令 $p=1+\delta/2>1$。条件 Jensen 不等式给

$$
\begin{aligned}
E\|C_t\|^p
&\le E\left(E[\|\xi_{t+1}\|^2\mid\mathcal F_t]\right)^p\\
&\le E\|\xi_{t+1}\|^{2p}\le K.
\end{aligned}
$$

从而 $Y_t=\|C_t-S\|$ 有一致有界的 $p$ 阶矩。对 $R>\varepsilon>0$，

$$
EY_t\le\varepsilon+R\Pr(Y_t>\varepsilon)
+E[Y_t\mathbf1_{\{Y_t>R\}}]
\le\varepsilon+R\Pr(Y_t>\varepsilon)+CR^{1-p}.
$$

先令 $t\to\infty$，再令 $R\to\infty$、$\varepsilon\downarrow0$，便有 $d_t:=E\|C_t-S\|\to0$。高阶矩在这里排除了小概率下协方差爆大的情形。

记

$$
D_T=\sum_{t<T}W_{T,t}SW_{T,t}^\top.
$$

任取 $\gamma>0$，选固定 $N$ 使 $t\ge N$ 时 $d_t\le\gamma$，则

$$
\begin{aligned}
E\|Q_T-D_T\|
&\le\sum_{t<T}\|W_{T,t}\|^2d_t\\
&\le\left(\max_{t<T}\|W_{T,t}\|^2\right)\sum_{t<N}d_t
+\gamma\sum_{N\le t<T}\|W_{T,t}\|^2.
\end{aligned}
$$

第一项趋于零，第二项至多为 $C\gamma$。最后令 $\gamma\downarrow0$，得到 $Q_T-D_T\to0$ 于 $L^1$，因而也依概率趋于零。

接下来计算 $D_T$。先固定 $\varepsilon>0$，只考虑 $t/T\ge\varepsilon$。记 $\beta_i=a\lambda_i$。当 $T$ 足够大时，所有 $\beta_i/j\le1/2$；由

$$
|\log(1-u)+u|
=\int_0^u\frac{v}{1-v}\,dv\le u^2,
\qquad 0\le u\le1/2,
$$

得到

$$
\log\prod_{j=t+1}^{T-1}(1-\beta_i/j)
=-\beta_i\sum_{j=t+1}^{T-1}\frac1j
+O\left(\sum_{j>t}\frac1{j^2}\right).
$$

调和和位于 $\log(T/(t+1))$ 与 $\log(T/t)$ 之间；平方和为 $O(1/t)$。因此一致地有

$$
\prod_{j=t+1}^{T-1}(1-\beta_i/j)
=(t/T)^{\beta_i}\exp\{O(1/t)\},
$$

即

$$
\sup_{t/T\ge\varepsilon}
\|\Gamma_{T,t}-(t/T)^{aH}\|\le C_\varepsilon/T.
$$

令 $x_t=t/T$，便可把权重写成

$$
W_{T,t}=\frac a{\sqrt T\,x_t}\Gamma_{T,t},
\qquad
\widetilde W_{T,t}=\frac a{\sqrt T}x_t^{aH-I}.
$$

在 $x_t\ge\varepsilon$ 上，两种权重均为 $O_\varepsilon(T^{-1/2})$，差为 $O_\varepsilon(T^{-3/2})$。利用恒等式

$$
WSW^\top-\widetilde WS\widetilde W^\top
=(W-\widetilde W)SW^\top
+\widetilde WS(W-\widetilde W)^\top,
$$

可知单项误差为 $O_\varepsilon(T^{-2})$，所有项之和为 $O_\varepsilon(T^{-1})$。故普通矩阵逐元素黎曼和给出

$$
\sum_{t=\lceil\varepsilon T\rceil}^{T-1}W_{T,t}SW_{T,t}^\top
\longrightarrow
 a^2\int_\varepsilon^1x^{aH-I}Sx^{aH^\top-I}\,dx.
$$

尚不能立即令 $\varepsilon=0$：被积矩阵在零点可能无界。离散和的早期部分由权重平方界控制为 $C\varepsilon^{2h-1}$；积分端则有

$$
\left\|\int_0^\varepsilon x^{aH-I}Sx^{aH^\top-I}\,dx\right\|
\le\|S\|\int_0^\varepsilon x^{2h-2}\,dx
=\frac{\|S\|}{2h-1}\varepsilon^{2h-1}.
$$

两端都能随 $\varepsilon\downarrow0$ 一致消失。因此先令 $T\to\infty$，再令 $\varepsilon\downarrow0$，得

$$
Q_T\xrightarrow{p}V_a
=a^2\int_0^1x^{aH-I}Sx^{aH^\top-I}\,dx.
$$

最后验证 Lindeberg 条件。令 $Z_{T,t}=W_{T,t}\xi_{t+1}$。对任意 $\varepsilon>0$，由 $x^2\mathbf1_{\{x>\varepsilon\}}\le\varepsilon^{-\delta}x^{2+\delta}$，

$$
\begin{aligned}
&\sum_{t<T}E[\|Z_{T,t}\|^2\mathbf1_{\{\|Z_{T,t}\|>\varepsilon\}}
\mid\mathcal F_t]\\
&\quad\le K\varepsilon^{-\delta}
\sum_{t<T}\|W_{T,t}\|^{2+\delta}\\
&\quad\le K\varepsilon^{-\delta}
\left(\max_{t<T}\|W_{T,t}\|\right)^\delta
\sum_{t<T}\|W_{T,t}\|^2\longrightarrow0.
\end{aligned}
$$

鞅差三角阵 CLT 因而给出 $M_T\Rightarrow N(0,V_a)$。再由 $\sqrt T e_T=-M_T+o_p(1)$，应用 Slutsky 定理，即得最后迭代的 CLT。负号不改变零均值正态分布的协方差。

## 4. Lyapunov 方程与逐元素解

令 $B=aH-I/2\succ0$。对积分作变量代换 $x=e^{-s}$，注意 $dx=-e^{-s}ds$，有

$$
\begin{aligned}
V_a
&=a^2\int_0^\infty e^{-s(aH-I)}S
 e^{-s(aH^\top-I)}e^{-s}\,ds\\
&=a^2\int_0^\infty e^{-Bs}Se^{-B^\top s}\,ds.
\end{aligned}
$$

其中 $e^{-s}$ 分别并入左右指数，各贡献一个 $-I/2$。记 $F(s)=e^{-Bs}Se^{-B^\top s}$，则

$$
F'(s)=-BF(s)-F(s)B^\top.
$$

因为 $\|F(s)\|\le\|S\|e^{-2\lambda_{\min}(B)s}$，无穷远处 $F(s)\to0$，在零点 $F(0)=S$。积分得到

$$
\boxed{\left(aH-\frac12I\right)V_a
+V_a\left(aH-\frac12I\right)^\top=a^2S.}
$$

这个方程的存在性来自积分。积分还保证 $V_a$ 对称半正定：对任意 $v$，

$$
v^\top V_av
=a^2\int_0^\infty
(e^{-B^\top s}v)^\top S(e^{-B^\top s}v)\,ds\ge0.
$$

在 $H$ 的特征基中，令 $\widetilde S=Q^\top SQ$、$\widetilde V=Q^\top V_aQ$，则

$$
(a\lambda_i+a\lambda_j-1)\widetilde V_{ij}
=a^2\widetilde S_{ij}.
$$

因此

$$
\boxed{
\widetilde V_{ij}
=\frac{a^2\widetilde S_{ij}}{a\lambda_i+a\lambda_j-1},
\qquad V_a=Q\widetilde VQ^\top.
}
$$

所有分母严格为正，因为 $a\lambda_i+a\lambda_j\ge2h>1$。若两个解之差为 $D$，其特征坐标满足相同方程但右边为零，只能 $D=0$，所以解唯一。整个推导不要求 $H$ 与 $S$ 交换。

一维时，设 $H=h_0>0$、$S=\sigma_g^2$，则

$$
V_a=\frac{a^2\sigma_g^2}{2ah_0-1},\qquad ah_0>\frac12.
$$

若 $\sigma_g^2>0$，

$$
\frac{dV_a}{da}
=\frac{2a\sigma_g^2(ah_0-1)}{(2ah_0-1)^2}.
$$

最小值在 $a=1/h_0$ 处取得，为 $\sigma_g^2/h_0^2$。这里调节的是递减步长 $a/t$ 中的常数；并非把整个步长固定为 $1/h_0$。多维时各方向曲率不同，同一个标量通常不能同时完成所有方向的最佳匹配。

## 5. Polyak–Ruppert 平均的渐近线性表示

现在改用

$$
\eta_t=at^{-\alpha},\qquad a>0,\quad\frac12<\alpha<1,
\qquad \bar\theta_T=\frac1T\sum_{t=1}^T\theta_t.
$$

第 1 节中的矩与余项条件也应针对这组步长成立，即 $E\|e_t\|^2=O(t^{-\alpha})$、$E\|r_t\|=O(t^{-\alpha})$。结论是

$$
\boxed{
\sqrt T(\bar\theta_T-\theta^*)
\Rightarrow N(0,\Sigma),\qquad \Sigma=H^{-1}SH^{-\top}.
}
$$

证明不必再次展开全部传播矩阵。由误差更新，

$$
He_t=-\xi_{t+1}-r_t-\frac{e_{t+1}-e_t}{\eta_t}.
$$

对 $t=1,\ldots,T$ 求和后除以 $\sqrt T$，得到

$$
H\sqrt T(\bar\theta_T-\theta^*)
=-\frac1{\sqrt T}\sum_{t=1}^T\xi_{t+1}
-\frac{D_T}{\sqrt T}
-\frac1{\sqrt T}\sum_{t=1}^Tr_t,
$$

这里本节另记

$$
D_T=\sum_{t=1}^T\frac{e_{t+1}-e_t}{\eta_t}.
$$

把第一组求和换下标，

$$
\sum_{t=1}^T\frac{e_{t+1}}{\eta_t}
=\sum_{t=2}^{T+1}\frac{e_t}{\eta_{t-1}},
$$

与 $\sum_{t=1}^Te_t/\eta_t$ 相减，即得离散分部求和恒等式

$$
D_T=\frac{e_{T+1}}{\eta_T}-\frac{e_1}{\eta_1}
+\sum_{t=2}^Te_t\left(\frac1{\eta_{t-1}}-\frac1{\eta_t}\right).
$$

由 Cauchy–Schwarz，$E\|e_t\|\le Ct^{-\alpha/2}$。均值定理给

$$
\left|\eta_t^{-1}-\eta_{t-1}^{-1}\right|
=a^{-1}[t^\alpha-(t-1)^\alpha]\le Ct^{\alpha-1}
\quad(t\ge2).
$$

于是

$$
\begin{aligned}
E\|D_T\|
&\le CT^{\alpha/2}+C
+C\sum_{t=2}^Tt^{-\alpha/2}t^{\alpha-1}\\
&\le CT^{\alpha/2}.
\end{aligned}
$$

因此 $D_T/\sqrt T\to0$ 于 $L^1$，所用条件恰为 $\alpha<1$。另一方面，

$$
E\left\|\frac1{\sqrt T}\sum_{t=1}^Tr_t\right\|
\le\frac C{\sqrt T}\sum_{t=1}^Tt^{-\alpha}
=O(T^{1/2-\alpha})\longrightarrow0,
$$

所用条件恰为 $\alpha>1/2$。由此获得渐近线性表示

$$
\boxed{
\sqrt T(\bar\theta_T-\theta^*)
=-H^{-1}\frac1{\sqrt T}\sum_{t=1}^T\xi_{t+1}+o_p(1).
}
$$

剩下的是噪声和的 CLT。第 3 节已经证明 $E\|C_t-S\|\to0$，因而

$$
E\left\|\frac1T\sum_{t=1}^TC_t-S\right\|
\le\frac1T\sum_{t=1}^TE\|C_t-S\|\longrightarrow0.
$$

权重现为统一的 $T^{-1/2}$。其 Lindeberg 和不超过

$$
K\varepsilon^{-\delta}\sum_{t=1}^TT^{-1-\delta/2}
=K\varepsilon^{-\delta}T^{-\delta/2}\longrightarrow0.
$$

因此鞅差 CLT 给 $T^{-1/2}\sum_t\xi_{t+1}\Rightarrow N(0,S)$。左乘 $-H^{-1}$ 的协方差为 $H^{-1}SH^{-\top}$，再由 Slutsky 定理完成证明。

$\alpha=1$ 时，上面 $D_T/\sqrt T$ 的界只有 $O(1)$，并没有趋于零。不能把这一端点代入已经证明的 PR 定理。即便 $a/t$ 下也计算算术平均，其极限协方差一般需要另外求解。

例如在一维线性模型 $H=1$、$a=1$ 中，令噪声 $\xi_{t+1}$ 独立同分布、均值零、方差 $\sigma^2$。从递推可直接算出，对 $t\ge2$，

$$
e_t=-\frac1{t-1}\sum_{s=1}^{t-1}\xi_{s+1}.
$$

因此平均误差中 $\xi_{s+1}$ 的系数为

$$
-\frac1T\sum_{t=s+1}^T\frac1{t-1}.
$$

它在 $s/T=x$ 处约为 $T^{-1}\log(1/x)$，所以 $\sqrt T\bar e_T$ 的渐近方差为

$$
\sigma^2\int_0^1[\log(1/x)]^2\,dx=2\sigma^2,
$$

而非直接套用 $H^{-1}SH^{-\top}$ 所得的 $\sigma^2$。此处积分等于 2 可令 $x=e^{-u}$ 化为 $\int_0^\infty u^2e^{-u}du$，两次分部积分即得。这个计算只说明端点的区别；其 CLT 还需验证这些确定权重满足最大项消失及 Lindeberg 条件。

## 6. 在线最小二乘与适用范围

设每步获得新的独立观测 $(X,Y)$，

$$
\Phi(\theta)=\frac12E(Y-X^\top\theta)^2,
\qquad \widehat g(\theta;X,Y)=(X^\top\theta-Y)X.
$$

记 $\varepsilon=Y-X^\top\theta^*$。一阶条件为 $E[\varepsilon X]=0$，故

$$
H=E[XX^\top],\qquad
S=E[\varepsilon^2XX^\top].
$$

在非奇异性、矩与速率条件满足时，PR 平均的极限协方差为

$$
\Sigma=H^{-1}E[\varepsilon^2XX^\top]H^{-1}.
$$

若进一步有 $E[\varepsilon^2\mid X]=\sigma^2$，才可将它化为 $\sigma^2H^{-1}$。异方差情形下不能省去中间矩阵。

以上结论是固定维数、非奇异 Hessian 附近的局部渐近结果。深度网络的参数对称性、平坦方向、非孤立极小点均可能破坏该条件。对固定训练集反复抽样时，$T\to\infty$ 描述的可能是围绕经验最小点的算法随机性；它不自动成为独立样本量 $n\to\infty$ 的总体统计结论。

**练习。** 设 $H=\operatorname{diag}(1,4)$，$S=I$。求 $a>1/2$ 时最后迭代的 $V_a$，并说明为什么无法选同一个 $a$，同时使两个对角元素都达到对应方向的最小值。再求 $1/2<\alpha<1$ 时 PR 平均的协方差。

*线索：* 两个方向分别使用 $a^2/(2a\lambda_i-1)$；最佳常数为 $1/\lambda_i$，而 PR 协方差为 $\operatorname{diag}(1,1/16)$。比较的是渐近协方差，不能据此断言任意有限运行长度下的误差排序。

迭代平均的经典来源为 Polyak 与 Juditsky 的 [Acceleration of Stochastic Approximation by Averaging](https://doi.org/10.1137/0330046)。本文给出一组明确充分条件下的局部计算；鞅差 CLT、Slutsky 定理和谱定理作为所引用的概率与线性代数基础使用。
