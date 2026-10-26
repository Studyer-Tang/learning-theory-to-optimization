# 08 · 一致收敛：覆盖数、VC 维度与 Rademacher 复杂度

[中文目录](README.md) · [English](../../README.md) · [上一节：集中不等式](07-concentration.md)

设 $Z_1,\ldots,Z_n\overset{\mathrm{iid}}\sim P$，$S=(Z_1,\ldots,Z_n)$。对损失函数 $g:\mathcal Z\to[0,1]$，记

$$
Pg=Eg(Z),\qquad P_ng=\frac1n\sum_{i=1}^ng(Z_i),\qquad
\Delta_n(\mathcal G)=\sup_{g\in\mathcal G}|P_ng-Pg|.
$$

函数类 $\mathcal G$ 非空，在抽样前确定。一个固定函数的集中界不能直接用于从同一份样本选出的函数；一致收敛要求在一个事件上，对类中所有函数同时控制偏差。

**可测性约定。** 为使上确界和条件期望均有通常的概率意义，本节假设函数可测，且函数类有一个可数子类，使每个函数都是该子类某个序列的逐点极限。有界性和控制收敛定理保证：涉及有限样本均值及总体均值的上确界可用这一可数子类代替。有限类自动满足要求；常见连续参数类也可据参数可分性核实。更一般的类需另用可测性条件或外概率表述，不能仅因每个成员可测，就断言上确界可测。计算未必有界的输出函数类复杂度时，另假定所需可积包络存在。

## 1. 数据选择为什么改变问题

对预先固定的 $g\in[0,1]^{\mathcal Z}$，Hoeffding 给出

$$
P(|P_ng-Pg|>\varepsilon)\le2e^{-2n\varepsilon^2}.
\tag{1}
$$

如果训练算法输出 $\widehat g_S$，则 $\widehat g_S(Z_i)$ 一般不再是同一固定函数作用于独立观测后得到的独立变量。不能把 (1) 中的 $g$ 直接换成 $\widehat g_S$。但逐样本总有

$$
|P_n\widehat g_S-P\widehat g_S|\le\Delta_n(\mathcal G),
$$

所以一个对全部函数同时成立的事件可以控制任何类内数据选择规则。

**有限类。** 若 $|\mathcal G|=M<\infty$，并集界给出

$$
P(\Delta_n(\mathcal G)>\varepsilon)\le2M e^{-2n\varepsilon^2}.
$$

因此以至少 $1-\delta$ 的概率，

$$
\Delta_n(\mathcal G)\le\sqrt{\frac{\log(2M/\delta)}{2n}}.
\tag{2}
$$

各函数偏差事件不必独立。这里有限类必须在抽样前固定；事后选出的随机代表集不能未经处理套用同一个并集界。

**经验风险最小化。** 设模型 $f$ 的总体风险为 $L(f)$，经验风险为 $\widehat L_n(f)$，损失类为 $\mathcal G$。若近似经验极小化器满足

$$
\widehat L_n(\widetilde f_n)\le\inf_{f\in\mathcal F}\widehat L_n(f)+\eta_n,
\qquad\eta_n\ge0,
$$

则逐样本有

$$
L(\widetilde f_n)-\inf_{f\in\mathcal F}L(f)
\le2\Delta_n(\mathcal G)+\eta_n.
\tag{3}
$$

对任意比较函数 $f$，依次插入经验风险，得

$$
L(\widetilde f_n)\le\widehat L_n(\widetilde f_n)+\Delta_n
\le\widehat L_n(f)+\eta_n+\Delta_n
\le L(f)+\eta_n+2\Delta_n.
$$

对 $f$ 取下确界即得，不要求类内最优函数存在。若比较对象改为所有可测预测器的最优风险 $L^*$，还须加上近似误差 $\inf_{f\in\mathcal F}L(f)-L^*$。

## 2. 覆盖数与两次逼近误差

度量空间 $(T,d)$ 的 $r$-网是一个有限集合 $T_r\subseteq T$，使每个 $t\in T$ 都有 $s\in T_r$ 满足 $d(t,s)\le r$。所需的最少点数记为 $N(r,T,d)$；若不存在有限网，定义为无穷。以下采用网中心在原集合中的内部覆盖。

对函数类使用一致距离

$$
\|g-h\|_\infty=\sup_{z\in\mathcal Z}|g(z)-h(z)|.
$$

若 $\mathcal G_r\subseteq\mathcal G$ 为一个预先固定的有限 $r$-网，则

$$
\Delta_n(\mathcal G)\le2r+\max_{h\in\mathcal G_r}|P_nh-Ph|.
\tag{4}
$$

**证明。** 对每个 $g$，选取 $h$ 使一致距离至多 $r$，分解为

$$
P_ng-Pg=P_n(g-h)+(P_nh-Ph)+P(h-g).
$$

第一项和第三项的绝对值各不超过 $r$，再对 $g$ 取上确界。两个 $r$ 分别控制经验均值和总体均值的近似误差，不能删去其中一个。仅在训练点上接近的样本网无法直接控制总体项。

结合 (2)，以至少 $1-\delta$ 的概率，

$$
\Delta_n(\mathcal G)
\le2r+\sqrt{\frac{\log(2N(r,\mathcal G,\|\cdot\|_\infty)/\delta)}{2n}}.
\tag{5}
$$

若最小覆盖数的定义以严格半径或下确界给出，先用略大的网，再对半径取极限即可；不应无条件假定最小网必然存在。

### 2.1 参数球的体积法

设 $\Theta\subseteq\{\theta\in\mathbb R^d:\|\theta\|_2\le B\}$ 非空，且

$$
\|g_\theta-g_{\theta'}\|_\infty\le L\|\theta-\theta'\|_2.
$$

则当 $B,L>0$ 时，

$$
N(r,\{g_\theta:\theta\in\Theta\},\|\cdot\|_\infty)
\le\left(1+\frac{2LB}{r}\right)^d.
\tag{6}
$$

**证明。** 令 $\rho=r/L$，在 $\Theta$ 中逐个选取两两距离严格大于 $\rho$ 的点 $\theta_1,\ldots,\theta_M$。以这些点为中心的半径 $\rho/2$ 开球互不相交，并都包含于半径 $B+\rho/2$ 的球中。欧氏 $d$ 维球体积随半径的 $d$ 次方缩放，因此

$$
M\,\operatorname{Vol}(B_d(\rho/2))
\le\operatorname{Vol}(B_d(B+\rho/2)),\qquad
M\le(1+2B/\rho)^d.
$$

这个有限上界保证过程必会终止。极大性保证任何未选点距离某个已选点至多 $\rho$，否则仍可添加，故得到参数网。再用 Lipschitz 条件转成函数网。若 $L=0$ 或 $B=0$，函数类只有一个不同函数，用一点覆盖即可。

注意比较的是环境空间中小球的体积，不要求 $\Theta$ 自身有正体积，也不要求小球落在 $\Theta$ 内。球面、低维曲面和不规则参数集合均可使用这一上界。

### 2.2 如何选择覆盖半径

由 (5)–(6)，对每个确定的 $r>0$，有概率至少 $1-\delta$ 满足

$$
\Delta_n\le F(r):=2r+
\sqrt{\frac{d\log(1+2LB/r)+\log(2/\delta)}{2n}}.
\tag{7}
$$

可先根据 $n,d,L,B,\delta$ 最小化这个确定函数，再使用相应固定网；这不属于数据选择。若半径也由样本决定，则需另做同时控制。

例如取 $r=\sqrt{d/n}$，得到

$$
\Delta_n\le2\sqrt{d/n}+
\sqrt{\frac{d\log(1+2LB\sqrt{n/d})+\log(2/\delta)}{2n}}.
\tag{8}
$$

固定 $LB>0$，在 $1\le d\le n$ 范围内，这是
$O(\sqrt{[d\{1+\log(n/d)\}+\log(1/\delta)]/n})$ 的单尺度界，常数可依赖固定的 $LB$。固定维数时即 $O(\sqrt{\log n/n})$。

也可精确优化 (7)。令 $A=2LB$、$H(r)=d\log(1+A/r)+\log(2/\delta)$，则内部极小点满足

$$
\frac{dA}{r(r+A)}=4\sqrt{2nH(r)}.
$$

固定 $d,A,\delta>0$、$n\to\infty$ 时，最优半径为 $r\asymp\sqrt{d/[n\log(n/d)]}$，所得 $F(r)$ 仍为 $\sqrt{d\log(n/d)/n}$ 阶。单次覆盖和并集界通常带来这个额外对数；下一部分的 Rademacher 线性类计算可以得到不含它的界。这里的 $r$ 是证明中的离散化误差，不是模型类相对真实规律的近似误差。

## 3. VC 维度与标签模式

设 $\mathcal H$ 为输入空间 $\mathcal X$ 上取值于 $\{0,1\}$ 的分类器类。固定互异点 $x_1,\ldots,x_m$，它们的预测模式集合为

$$
\mathcal H|_{x_{1:m}}=\{(h(x_1),\ldots,h(x_m)):h\in\mathcal H\}.
$$

若包含所有 $2^m$ 个二值向量，则称 $\mathcal H$ 打散这组点。VC 维度 $d_{\mathrm{VC}}(\mathcal H)$ 是可被打散点集的最大大小；若大小无上界，记为无穷。增长函数定义为

$$
\Pi_{\mathcal H}(m)=\sup_{x_1,\ldots,x_m}
|\mathcal H|_{x_{1:m}}|.
$$

允许输入重复也不影响上界；重复点不能独立赋予不同标签。VC 维度需要“存在某组点被打散”，并非要求所有同样大小的点集都被打散。

### 3.1 阈值、区间与半空间

**实线阈值。** 对 $h_a(x)=\mathbf1_{\{x\le a\}}$，一个点可任意标记；若 $x_1<x_2$，不能得到标签 $(0,1)$，故 VC 维度为 $1$。在 $m$ 个排好序的互异点上只有 $m+1$ 个模式，对应前 $0,1,\ldots,m$ 个点取 $1$。

**实线区间。** 对 $h_{a,b}(x)=\mathbf1_{\{a\le x\le b\}}$，任意两个互异点可被打散；三个有序点不能得到 $(1,0,1)$，故 VC 维度为 $2$。在 $m$ 个互异有序点上，正标签只能是一段连续非空区块，或全为空，所以恰有

$$
1+\sum_{k=1}^m(m-k+1)=1+\frac{m(m+1)}2
$$

种模式。长度为 $k$ 的连续正标签块有 $m-k+1$ 个起点，这是求和中每一项的来源。例如三个点时：

| 正标签块长度 | 模式 | 数目 |
| --- | --- | --- |
| $0$ | $000$ | $1$ |
| $1$ | $100,010,001$ | $3$ |
| $2$ | $110,011$ | $2$ |
| $3$ | $111$ | $1$ |

总数为 $7$，缺少 $101$。两个区间可能诱导同一个模式；数的是标签向量，不是参数对数目。若允许至多 $k$ 个区间的并，VC 维度为 $2k$：$2k$ 个点的正标签最多有 $k$ 段；$2k+1$ 个点上的交替标签 $(1,0,1,\ldots,0,1)$ 却需要 $k+1$ 段。

**$\mathbb R^d$ 的仿射半空间。** 令 $\mathcal H=\{x\mapsto\mathbf1_{\{w^\top x+b\ge0\}}\}$，其 VC 维度为 $d+1$。

下界：取 $d+1$ 个仿射无关点，它们的增广向量 $(x_i,1)$ 线性无关。对任意标签 $y_i\in\{-1,1\}$，线性方程 $w^\top x_i+b=y_i$ 有解，故可打散。

上界：任取 $d+2$ 个点，存在不全为零的系数 $a_i$ 满足 $\sum_i a_ix_i=0$、$\sum_i a_i=0$。正负系数均非空。把 $a_i>0$ 的点标为 $1$，$a_i<0$ 的点标为 $0$，零系数点随意。若某个半空间实现这些标签，则对 $q_i=w^\top x_i+b$，正系数项 $a_iq_i\ge0$，负系数项因 $q_i<0$ 而严格为正，故 $\sum_i a_iq_i>0$。但线性关系给出该和等于 $0$，矛盾。

### 3.2 Sauer–Shelah 引理：递推证明

若 $d=d_{\mathrm{VC}}(\mathcal H)<\infty$，则对所有整数 $m\ge0$，

$$
\Pi_{\mathcal H}(m)\le\sum_{j=0}^d\binom mj,
\tag{9}
$$

约定 $j>m$ 时二项系数为零。当 $1\le d\le m$ 时，还有

$$
\sum_{j=0}^d\binom mj\le(em/d)^d.
\tag{10}
$$

**证明 (9)。** 将问题限制在固定的 $m$ 个点上，得到有限布尔向量类 $\mathcal A\subseteq\{0,1\}^m$。按最后一位的标签，记前 $m-1$ 位可能出现的模式为 $\mathcal A_0,\mathcal A_1$，令

$$
\mathcal U=\mathcal A_0\cup\mathcal A_1,
\qquad\mathcal V=\mathcal A_0\cap\mathcal A_1.
$$

同一前缀若在两支都出现，对应原类中两个不同向量，因此

$$
|\mathcal A|=|\mathcal A_0|+|\mathcal A_1|
=|\mathcal U|+|\mathcal V|.
$$

$\mathcal U$ 的 VC 维度不超过 $d$。若 $\mathcal V$ 非空，它的 VC 维度不超过 $d-1$：若它打散前 $m-1$ 点中的一个 $d$ 点集，则每个模式均可接上最后一位 $0$ 或 $1$，原类便打散 $d+1$ 点，矛盾。若 $\mathcal V$ 为空，其计数贡献直接为零。

对点数归纳。$m=0$ 时非空类至多一个空向量；$d=0$ 时若有两个不同向量，它们某个坐标取值不同，就可打散一个点，矛盾，故至多一个向量。对 $1\le d<m$，归纳假设与 Pascal 恒等式给出

$$
|\mathcal A|
\le\sum_{j=0}^d\binom{m-1}j+
\sum_{j=0}^{d-1}\binom{m-1}j
=\sum_{j=0}^d\binom mj.
$$

当 $d\ge m$ 时右端为 $2^m$，直接成立。对所有点集取上确界即得 (9)。

**证明 (10)。** 令 $t=d/m\in(0,1]$。因为 $t^d\le t^j$ 对 $j\le d$ 成立，

$$
t^d\sum_{j=0}^d\binom mj
\le\sum_{j=0}^d\binom mjt^j
\le(1+t)^m\le e^{mt}=e^d.
$$

除以 $(d/m)^d$ 即得。当 $d=0$，使用 $\Pi_{\mathcal H}(m)\le1$，不代入含 $1/d$ 的公式。当 $d>m$，不能把 (10) 当作已证明的有效上界。

## 4. VC 一致收敛的完整证明

对零一损失 $\ell_h(x,y)=\mathbf1_{\{h(x)\ne y\}}$，记 $L(h)=P\ell_h$、$\widehat L_n(h)=P_n\ell_h$。则对任意 $\varepsilon>0$，

$$
P\left(\sup_{h\in\mathcal H}|\widehat L_n(h)-L(h)|>\varepsilon\right)
\le4\Pi_{\mathcal H}(2n)e^{-n\varepsilon^2/8}.
\tag{11}
$$

证明不能直接对当前样本上的 $\Pi_{\mathcal H}(n)$ 个模式使用固定函数集中界：同模式分类器在样本外可以不同，且模式代表由样本选择。需要先消除总体风险，使事件只依赖有限点上的标签。

### 4.1 幽灵样本

取与 $S$ 独立同分布的 $S'=(Z_1',\ldots,Z_n')$，经验均值记作 $P_n'$。先设 $n\varepsilon^2\ge2$，并令

$$
A=\{\sup_h|P_n\ell_h-P\ell_h|>\varepsilon\},\qquad
B=\{\sup_h|P_n\ell_h-P_n'\ell_h|>\varepsilon/2\}.
$$

对固定的 $S\in A$，从约定的可数子类依次选取第一个满足偏差严格超过 $\varepsilon$ 的 $h_S$，得到可测选择。相对于独立的 $S'$，它是一个固定函数，因而

$$
E_{S'}P_n'\ell_{h_S}=P\ell_{h_S},\qquad
\operatorname{Var}_{S'}(P_n'\ell_{h_S})\le\frac1{4n}.
$$

由 Chebyshev，

$$
P_{S'}\left(|P_n'\ell_{h_S}-P\ell_{h_S}|\le\varepsilon/2\mid S\right)
\ge1-\frac1{n\varepsilon^2}\ge\frac12.
$$

在该事件上，反向三角不等式给 $|P_n\ell_{h_S}-P_n'\ell_{h_S}|>\varepsilon/2$。所以 $P(B\mid S)\ge\frac12\mathbf1_A(S)$，取期望即得

$$
P(A)\le2P(B).
\tag{12}
$$

幽灵样本只用于证明，学习算法无需额外获取它。

### 4.2 随机交换

取独立符号 $\xi_i\in\{-1,1\}$，各以概率 $1/2$ 出现。若 $\xi_i=-1$ 就交换 $Z_i,Z_i'$，否则不交换。由于每一对观测独立同分布，交换不改变全部样本的联合分布。因此

$$
P(B)=E_{S,S'}P_\xi\left(
\sup_h\left|\frac1n\sum_i\xi_i\{\ell_h(Z_i)-\ell_h(Z_i')\}\right|>\varepsilon/2
\,\middle|\,S,S'\right).
\tag{13}
$$

这是对样本分布平均后的恒等式，不是在固定两份样本后把确定差值与符号化差值宣称相等。

### 4.3 对模式计数

固定 $S,S'$ 及其真实标签。分类器在 $2n$ 个输入上的预测向量至多有 $\Pi_{\mathcal H}(2n)$ 种，每个预测向量确定相应损失差向量

$$
a_i=\ell_h(Z_i)-\ell_h(Z_i')\in\{-1,0,1\}.
$$

对其中一个确定向量，$\xi_i a_i$ 独立、中心化，取值区间长度至多为 $2$，故 Hoeffding 给出

$$
P_\xi\left(\left|\frac1n\sum_i\xi_i a_i\right|>\varepsilon/2\right)
\le2e^{-n\varepsilon^2/8}.
$$

对全部模式取并集界，再对样本平均，结合 (12)–(13)，得到 (11)。若 $n\varepsilon^2<2$，(11) 的右端至少为 $4e^{-1/4}>1$，由概率至多为 $1$ 直接成立。这就补齐全部 $\varepsilon>0$，并非漏掉小偏差区间。

### 4.4 高概率界与 ERM

若 $1\le d=d_{\mathrm{VC}}(\mathcal H)\le2n$，由 (9)–(11)，以至少 $1-\delta$ 的概率，

$$
\sup_h|\widehat L_n(h)-L(h)|
\le\min\left\{1,
\sqrt{\frac8n\left[d\log\frac{2en}{d}+\log\frac4\delta\right]}\right\}.
\tag{14}
$$

令平方根为 $\varepsilon$ 代入 (11)，指数正好抵消模式数。$d=0$ 时删去 $d\log(2en/d)$ 项；$d>2n$ 时使用 $\Pi_{\mathcal H}(2n)\le2^{2n}$ 或直接用平凡界 $1$，不套超出范围的二项和估计。

把 (14) 的右端记作 $a_n(\delta)$，由 (3)，同一事件上近似 ERM 满足

$$
L(\widetilde h_n)-\inf_{h\in\mathcal H}L(h)\le2a_n(\delta)+\eta_n.
$$

固定有限 $d$ 时统计项趋于零。VC 维度限制的是可实现的标签模式数量，不保证模型已包含真实规律；类内最优风险仍可能很高。

## 5. Rademacher 复杂度与对称化

覆盖数需要度量离散化，VC 维度适用于二分类标签计数。Rademacher 复杂度直接衡量函数类在观测点上与独立随机符号的最大相关程度。

对固定样本 $S$，本节采用**上确界内带绝对值**的约定：

$$
\widehat{\mathfrak R}_S(\mathcal G)
=E_\xi\sup_{g\in\mathcal G}\left|\frac1n\sum_i\xi_i g(Z_i)\right|,
\qquad
\mathfrak R_n(\mathcal G)=E_S\widehat{\mathfrak R}_S(\mathcal G).
\tag{15}
$$

经验复杂度只对符号平均，样本保持固定，因此它仍是样本的函数。另有教材使用不带绝对值的定义；对关于取负对称的类，两种约定相同，对一般损失类未必相同。比如常量单函数类 $\{g\equiv1\}$，不带绝对值的复杂度为零，本节复杂度则为 $n^{-1}E|\sum_i\xi_i|$。

### 5.1 期望对称化

对 $\mathcal G\subseteq[0,1]^{\mathcal Z}$，

$$
E\Delta_n(\mathcal G)\le2\mathfrak R_n(\mathcal G).
\tag{16}
$$

**证明。** 固定 $S$，以独立的 $S'$ 代替总体均值。对每个 $g$，

$$
|P_ng-Pg|=|E_{S'}(P_ng-P_n'g)|
\le E_{S'}\sup_{u\in\mathcal G}|P_nu-P_n'u|.
$$

右端对 $g$ 相同，故可对左端取上确界，再对 $S$ 平均，得

$$
E\Delta_n\le E_{S,S'}\sup_g\left|\frac1n\sum_i(g(Z_i)-g(Z_i'))\right|.
$$

用上一部分的随机交换引入符号，再用三角不等式拆开两份样本：

$$
\begin{aligned}
E\Delta_n
&\le E_{S,S',\xi}\sup_g
\left|\frac1n\sum_i\xi_i(g(Z_i)-g(Z_i'))\right|\\
&\le E_{S,\xi}\sup_g\left|\frac1n\sum_i\xi_i g(Z_i)\right|
+E_{S',\xi}\sup_g\left|\frac1n\sum_i\xi_i g(Z_i')\right|\\
&=2\mathfrak R_n(\mathcal G).
\end{aligned}
$$

两个上确界可以选择不同函数，所以第二步为不等式。也没有把“期望的上确界”与“上确界的期望”当作相等。

### 5.2 从期望到高概率

仍设 $g\in[0,1]$。替换一个样本点时，$P_ng$ 改变至多 $1/n$，由绝对值和上确界的差分不等式，$\Delta_n$ 也改变至多 $1/n$。McDiarmid 给出

$$
P(\Delta_n-E\Delta_n\ge t)\le e^{-2nt^2}.
$$

结合 (16)，以至少 $1-\delta$ 的概率，

$$
\Delta_n(\mathcal G)
\le2\mathfrak R_n(\mathcal G)+\sqrt{\frac{\log(1/\delta)}{2n}}.
\tag{17}
$$

经验复杂度本身也具有有界差分常数 $1/n$：对每组固定符号，其绝对符号和的上确界变化至多 $1/n$，再对符号平均仍然成立。对负的经验复杂度用 McDiarmid，得到

$$
P\left(\mathfrak R_n(\mathcal G)-\widehat{\mathfrak R}_S(\mathcal G)\ge t\right)
\le e^{-2nt^2}.
$$

在两次应用中各取失败概率 $\delta/2$，令 $t=\sqrt{\log(2/\delta)/(2n)}$，并集界保证同时有
$\Delta_n\le2\mathfrak R_n+t$、$\mathfrak R_n\le\widehat{\mathfrak R}_S+t$。因而以至少 $1-\delta$ 的概率，

$$
\Delta_n(\mathcal G)
\le2\widehat{\mathfrak R}_S(\mathcal G)
+3\sqrt{\frac{\log(2/\delta)}{2n}}.
\tag{18}
$$

将 (18) 代入 (3)，即得近似 ERM 的经验复杂度界

$$
L(\widetilde f_n)-\inf_{f\in\mathcal F}L(f)
\le4\widehat{\mathfrak R}_S(\mathcal G)
+6\sqrt{\frac{\log(2/\delta)}{2n}}+\eta_n.
\tag{19}
$$

这里必须使用损失函数类 $\mathcal G$ 的复杂度；输出函数类的复杂度需经损失复合关系才能接入。

## 6. 收缩引理：从输出类到损失类

设 $T\subseteq\mathbb R^n$，$\phi_i:\mathbb R\to\mathbb R$ 均为 $L$-Lipschitz 且 $\phi_i(0)=0$，则

$$
E_\xi\sup_{t\in T}\left|\sum_i\xi_i\phi_i(t_i)\right|
\le2L\,E_\xi\sup_{t\in T}\left|\sum_i\xi_it_i\right|.
\tag{20}
$$

常数 $2$ 适用于本节的绝对值约定；不在此断言该常数最优。

**逐坐标证明。** 先证明不带绝对值的收缩式。设 $T$ 有限，固定除第 $j$ 个以外的符号，把其余项记为任意实数 $A_t$。则

$$
\begin{aligned}
E_{\xi_j}\sup_t[A_t+\xi_j\phi_j(t_j)]
&=\frac12\sup_{s,t}[A_s+A_t+\phi_j(s_j)-\phi_j(t_j)]\\
&\le\frac12\sup_{s,t}[A_s+A_t+L|s_j-t_j|]\\
&=\frac12\sup_{s,t}[A_s+A_t+L(s_j-t_j)]\\
&=E_{\xi_j}\sup_t[A_t+L\xi_jt_j].
\end{aligned}
$$

第三步用交换 $s,t$ 后 $A_s+A_t$ 不变，所以取绝对差与取有向差的上确界相同。依次替换每个坐标并对其他符号平均，得到

$$
E_\xi\sup_{t\in T}\sum_i\xi_i\phi_i(t_i)
\le L E_\xi\sup_{t\in T}\sum_i\xi_i t_i.
$$

对绝对值版本，先将 $0$ 加入 $T$，这不改变绝对值上确界，且因 $\phi_i(0)=0$，正负两个非绝对上确界都非负。于是

$$
\sup_t\left|\sum_i\xi_i\phi_i(t_i)\right|
\le\sup_t\sum_i\xi_i\phi_i(t_i)
+\sup_t\left(-\sum_i\xi_i\phi_i(t_i)\right).
$$

两项的符号期望相同。各用非绝对收缩式，再将右端非绝对上确界用绝对上确界控制，得到 (20)。可数类通过递增有限子集和单调收敛推广；一般可分类由本节可测性约定及适当可积性推广。

**损失复合。** 若 $a\mapsto\ell(a,y)$ 对每个 $y$ 均为 $L$-Lipschitz，固定样本后取

$$
\phi_i(a)=\ell(a,Y_i)-\ell(0,Y_i).
$$

由 (20) 和三角不等式，

$$
\widehat{\mathfrak R}_S(\ell\circ\mathcal F)
\le2L\widehat{\mathfrak R}_{S_X}(\mathcal F)
+\frac1n\sqrt{\sum_i\ell(0,Y_i)^2}.
\tag{21}
$$

最后一项来自 $E_\xi|\sum_i\xi_i b_i|\le\sqrt{\sum_i b_i^2}$。在绝对值约定下，常量基线不能无故消掉。若 $\ell(0,y)=0$ 则该项消失；若损失值在 $[0,1]$，该项至多为 $n^{-1/2}$。零一损失不满足通常的 Lipschitz 条件，不能直接套 (21)。

## 7. 常用函数类的复杂度计算

### 7.1 有限类

固定样本，若 $|\mathcal G|=M$、$|g(Z_i)|\le A$，则

$$
\widehat{\mathfrak R}_S(\mathcal G)
\le A\sqrt{\frac{2\log(2M)}n}.
\tag{22}
$$

**证明。** 令 $T_g=n^{-1}\sum_i\xi_i g(Z_i)$。条件于样本，独立符号使

$$
E_\xi e^{\lambda T_g}
\le\exp\left(\frac{\lambda^2}{2n^2}\sum_i g(Z_i)^2\right)
\le e^{\lambda^2A^2/(2n)}.
$$

同理控制 $-T_g$。Jensen 及指数最大值不超过指数和给出

$$
\widehat{\mathfrak R}_S
\le\frac{\log(2M)}\lambda+\frac{\lambda A^2}{2n},\qquad\lambda>0.
$$

对 $\lambda$ 最小化即得 (22)。$A=0$ 时直接成立。$2M$ 来自同时考虑每个函数的两个符号方向。

### 7.2 欧氏范数约束的线性类

令 $\mathcal F_B=\{x\mapsto w^\top x:\|w\|_2\le B\}$。固定输入样本后，

$$
\begin{aligned}
\widehat{\mathfrak R}_{S_X}(\mathcal F_B)
&=\frac BnE_\xi\left\|\sum_i\xi_iX_i\right\|_2\\
&\le\frac Bn\sqrt{\sum_i\|X_i\|_2^2}.
\end{aligned}
\tag{23}
$$

第一步由欧氏范数对偶性：$\sup_{\|w\|_2\le B}|w^\top v|=B\|v\|_2$；$v\ne0$ 时取 $w=Bv/\|v\|_2$ 达到上界。第二步用 Cauchy–Schwarz 和

$$
E_\xi\left\|\sum_i\xi_iX_i\right\|_2^2
=\sum_{i,j}E(\xi_i\xi_j)X_i^\top X_j
=\sum_i\|X_i\|_2^2.
$$

若 $\|X_i\|_2\le R$，则复杂度至多 $BR/\sqrt n$。计算中输入已固定，不能再把 $X_i$ 作为符号期望针对的随机性。界中不显式含维数，但数据半径 $R$ 本身可能随维数增长。

### 7.3 $\ell_1$ 约束的线性类

令 $\mathcal F_B^{(1)}=\{x\mapsto w^\top x:\|w\|_1\le B\}$，若 $\|X_i\|_\infty\le R$，则

$$
\widehat{\mathfrak R}_{S_X}(\mathcal F_B^{(1)})
=\frac BnE_\xi\left\|\sum_i\xi_iX_i\right\|_\infty
\le BR\sqrt{\frac{2\log(2d)}n}.
\tag{24}
$$

对偶等式 $\sup_{\|w\|_1\le B}|w^\top v|=B\|v\|_\infty$ 的上界来自逐项绝对值求和；把全部权重放在绝对值最大的坐标上即可达到。最后把 $d$ 个坐标函数组成有限类，应用 (22)。比较 (23) 与 (24) 时需同时检查输入使用哪一种范数限制，不能仅看最后是否出现 $d$。

### 7.4 能实现任意标签的类

若在固定的 $n$ 个互异样本点上，$\mathcal G$ 实现全部 $2^n$ 个二值向量，则给定符号，最大的绝对符号和等于正号数与负号数的较大者。因此

$$
\widehat{\mathfrak R}_S(\mathcal G)
=\frac12+\frac1{2n}E_\xi\left|\sum_i\xi_i\right|,
\qquad
\frac12\le\widehat{\mathfrak R}_S(\mathcal G)\le\frac12+\frac1{2\sqrt n}.
$$

这样的复杂度不会趋于零，因而 (18) 不能给出消失的统一偏差。这不意味着每个具体分布下每个分类器都必然预测失败，而是说明过于丰富的类无法从这一分布无关复杂度界得到保证。

## 8. 练习

1. 一个预先固定的函数类有 $M=1000$ 个 $[0,1]$ 值函数。写出使 $\Delta_n\le0.05$ 以概率至少 $0.99$ 成立的充分样本量。若先看数据再保留 $1000$ 个函数，为什么不能原样使用答案？
2. 逐一列出三个有序点上的区间标签向量，并与 $1+m(m+1)/2$ 核对。若将区间及其补集一起收入函数类，VC 维度是否仍为 $2$？
3. 解释 VC 证明中为何数 $2n$ 点而非 $n$ 点的模式，为何在条件于样本后对符号使用 Hoeffding 是合法的。
4. 对 $\|w\|_2\le B$ 的线性类，若只知每个坐标 $|X_{ij}|\le1$，把 (23) 写成显式含 $d$ 的上界。它是否仍与维数无关？
5. 在带绝对值的约定下，证明给函数类统一加一个常量可能改变复杂度；再说明 (21) 中保留基线项的必要性。

**答案线索。** 1. $n\ge\log(2M/0.01)/(2\cdot0.05^2)$，取整为 $2442$；随机筛选的代表不再是抽样前固定的有限类。2. 除 $(1,0,1)$ 外七种；加入补集后可打散三个点，但四点的交替标签 $(1,0,1,0)$ 仍不可实现，故 VC 维度为 $3$。3. 幽灵样本把总体均值转为第二份经验均值；固定全部输入与标签后，各模式差向量成为确定系数，符号仍独立。4. $R\le\sqrt d$，上界为 $B\sqrt{d/n}$。5. 比较单函数类 $\{0\}$ 和 $\{1\}$；无绝对值的另一种定义具有不同的平移性质，不能混用。
