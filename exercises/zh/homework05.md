# 作业 5 · VC 维与 Rademacher 复杂度

[中文目录](../../notes/zh/README.md) · [一致收敛](../../notes/zh/08-uniform-convergence.md) · [作业 4](homework04.md)

共 4 道大题、15 个小问。分类规则一律把零函数值标为正类；端点处理、上确界与函数类的对称性分别在证明中说明。本篇使用不预先取绝对值的 Rademacher 复杂度，与正文带绝对值的版本相区别。收缩引理的证明见正文。

## 第 1 题 · VC dimension

记集合类在有限点集上的标签为对应的指示函数值。一个含 $m$ 个不同点的集合被打散，指它的全部 $2^m$ 种二元标签都能实现。证明 VC 维等于 $d$，需要同时构造可打散的 $d$ 个点，并证明任意 $d+1$ 个点都不能被打散。

### (a) Unions of intervals：证明 $\operatorname{VCdim}(\mathcal I_k)=2k$

**下界。** 任取 $2k$ 个有序点 $x_1<\cdots<x_{2k}$，并任意指定它们的标签。把标签为 1 的点按顺序分成极大的连续块。若共有 $q$ 个正标签块，则相邻块之间至少有一个负标签点，因此至少需要 $2q-1$ 个点，故
$$
2q-1\le 2k\quad\Longrightarrow\quad q\le k.
$$
对每个正标签块 $x_a,\ldots,x_b$，选择闭区间 $[x_a,x_b]$。这些区间的并恰好包含所有正标签样本，不包含任何负标签样本；单点块可用退化区间 $[x_a,x_a]$。全零标签用空集实现。故任意标签都可由至多 $k$ 个闭有界区间实现，得到
$$
\operatorname{VCdim}(\mathcal I_k)\ge 2k.
$$

**上界。** 任取 $2k+1$ 个有序点，赋予交替标签
$$
1,0,1,0,\ldots,0,1.
$$
其中有 $k+1$ 个正标签点。一个区间如果包含两个正标签点，也必须包含它们之间的负标签点，因此每个允许的区间至多覆盖其中一个正标签点。实现该标签至少需要 $k+1$ 个区间，与类的定义矛盾。因而没有 $2k+1$ 个点能被打散。
$$
\boxed{\operatorname{VCdim}(\mathcal I_k)=2k.}
$$

### (b) A one-parameter family：证明 VC 维为 3

考虑
$$
I_\alpha=[\alpha,\alpha+1]\cup[\alpha+2,\infty).
$$
对固定点 $x$，把成员条件改写为参数 $\alpha$ 的条件：
$$
\mathbf 1_{\{x\in I_\alpha\}}=1
\quad\Longleftrightarrow\quad
\alpha\in(-\infty,x-2]\cup[x-1,x].
\tag{HW5.1}
$$

**下界：构造可打散的三个点。** 取
$$
x_1=0,\qquad x_2=\frac34,\qquad x_3=\frac32.
$$
下面列出全部八种标签及其参数。逐行代入式 （HW5.1） 即可核对。

| $\alpha$ | 标签 $(h(x_1),h(x_2),h(x_3))$ | 正标签点 |
|---|---|---|
| $-3$ | $(1,1,1)$ | $x_1,x_2,x_3$ |
| $-3/2$ | $(0,1,1)$ | $x_2,x_3$ |
| $-9/8$ | $(0,0,1)$ | $x_3$ |
| $-3/4$ | $(1,0,1)$ | $x_1,x_3$ |
| $-3/8$ | $(1,0,0)$ | $x_1$ |
| $-1/8$ | $(1,1,0)$ | $x_1,x_2$ |
| $1/4$ | $(0,1,0)$ | $x_2$ |
| $2$ | $(0,0,0)$ | 无 |

例如 $\alpha=-3/8$ 时，$I_\alpha=[-3/8,5/8]\cup[13/8,\infty)$，只包含三个样本中的 $x_1=0$。因此这三个点被打散，VC 维至少为 3。

**上界：包括临界参数本身的标签计数。** 固定任意 $m$ 个不同样本点。对每个 $x_i$，当 $\alpha$ 从 $-\infty$ 增大到 $+\infty$，式 （HW5.1） 给出的标签依次为
$$
\begin{array}{c|cccc}
\alpha & (-\infty,x_i-2] & (x_i-2,x_i-1) & [x_i-1,x_i] & (x_i,\infty)\\
\hline
\mathbf 1_{\{x_i\in I_\alpha\}} & 1&0&1&0.
\end{array}
$$
所以该坐标总共变化三次。准确地说：在 $x_i-2$ 处仍为 1，越过后才变成 0；在 $x_i-1$ 处已经变成 1；在 $x_i$ 处仍为 1，越过后才变成 0。

按顺序扫描所有临界值，在每个临界值同时检查其左侧、该点本身和右侧。如果某个临界值同时对应某些坐标的上升与另一些坐标的下降，临界点本身确实可能产生额外标签；但从左侧到临界点、再从临界点到右侧的每次标签向量变化，都至少消耗一次坐标变化。全部 $m$ 个坐标只有 $3m$ 次变化，因此不同标签向量的总数至多为
$$
1+3m.
$$
这个计数已经包括临界点标签，没有把闭区间端点遗漏掉。对 $m=4$，有
$$
1+3m=13<16=2^4.
$$
故任意四个点都不能被打散。结合下界，
$$
\boxed{\operatorname{VCdim}(\{I_\alpha:\alpha\in\mathbb R\})=3.}
$$
一个实参数并不意味着 VC 维至多为 1；关键是参数变化时能够产生多少种标签模式。

### (c) Closed balls：证明 $\operatorname{VCdim}(\mathcal B_p)\le p+2$

定义提升映射
$$
T:\mathbb R^p\longrightarrow\mathbb R^{p+1},\qquad T(x)=(x,\|x\|_2^2).
$$
球的成员条件等价于
$$
\begin{aligned}
\|x-c\|_2\le r
&\Longleftrightarrow
\|x\|_2^2-2c^\top x+\|c\|_2^2-r^2\le0\\
&\Longleftrightarrow
(2c,-1)^\top T(x)+(r^2-\|c\|_2^2)\ge0.
\end{aligned}
$$
因此每个球在原空间上的标签，都由提升空间中的某个仿射半空间实现。若原空间的一组点被球打散，其提升像也就被仿射半空间打散。

提升空间的维数是 $q=p+1$，其仿射半空间 VC 维上界为 $q+1=p+2$。为完整说明这个上界，任取 $q+2$ 个点 $z_1,\ldots,z_{q+2}\in\mathbb R^q$。增广向量 $(z_i,1)\in\mathbb R^{q+1}$ 线性相关，故存在不全为零的 $a_i$，使
$$
\sum_i a_i(z_i,1)=0.
$$
必要时把所有系数反号，使某个 $a_i>0$。给 $a_i>0$ 的点标签 0，给 $a_i<0$ 的点标签 1，零系数点任取标签。若仿射函数 $g(z)=w^\top z+b$ 以 $g\ge0$ 为正标签实现了它，则
$$
\sum_i a_i g(z_i)<0,
$$
因为正系数对应严格负函数值，负系数对应非负函数值，而且至少一个负贡献严格小于零。但线性相关关系同时给出
$$
\sum_i a_i g(z_i)=(w,b)^\top\sum_i a_i(z_i,1)=0,
$$
矛盾。故提升空间中的仿射半空间不能打散 $p+3$ 个点，从而
$$
\boxed{\operatorname{VCdim}(\mathcal B_p)\le p+2.}
$$
本问只要求这个上界，证明不需要判断它是否为球类的精确 VC 维。

### (d) A vector space of functions：证明 VC 维至多为 $r$

设 $F$ 是实值函数构成的线性空间，$\dim F=r<\infty$。任取 $r+1$ 个不同点 $x_1,\ldots,x_{r+1}$，考虑线性映射
$$
L:F\to\mathbb R^{r+1},\qquad L(f)=(f(x_1),\ldots,f(x_{r+1})).
$$
由于 $\dim\operatorname{im}L\le\dim F=r$，像空间是 $\mathbb R^{r+1}$ 的真线性子空间。故存在非零向量 $a=(a_1,\ldots,a_{r+1})$ 与整个像空间正交，即
$$
\sum_{i=1}^{r+1}a_i f(x_i)=0\qquad\text{对所有 }f\in F.
\tag{HW5.2}
$$
将 $a$ 整体反号（若需要），可保证至少一个 $a_i>0$。指定标签
$$
y_i=\begin{cases}
0,&a_i>0,\\
1,&a_i\le0.
\end{cases}
$$
如果某个 $f\in F$ 实现它，那么正系数位置满足 $f(x_i)<0$；负系数位置满足 $f(x_i)\ge0$。因此所有 $a_if(x_i)\le0$，并至少一个严格小于零，得到
$$
\sum_i a_if(x_i)<0,
$$
与式 （HW5.2） 矛盾。所以任意 $r+1$ 个点都存在无法实现的标签，故
$$
\boxed{\operatorname{VCdim}(\{\{x:f(x)\ge0\}:f\in F\})\le r.}
$$
该证明也覆盖 $r=0$。使用严格负标签是为了正确处理分类规则中的边界 $f=0$：零函数值只能标为 1。

### (e) Sine classifiers

设 $h_\omega(x)=\mathbf 1_{\{\sin(\omega x)\ge0\}}$。

**(i) 四个等间距倍数点上的限制。** 对 $x\ne0$，令 $t=\omega x$。因 $\omega$ 可取任意实数，$t$ 也可取任意实数。只需证明四点标签
$$
(h_\omega(x),h_\omega(2x),h_\omega(3x),h_\omega(4x))=(1,0,0,0)
$$
不可能实现。

利用周期性，可将 $t$ 限制在 $[0,2\pi)$。若第一个标签为 1，则 $\sin t\ge0$，从而 $t\in[0,\pi]$。若第二个标签为 0，则 $\sin(2t)<0$，进一步得到
$$
\frac\pi2<t<\pi.
$$
若第四个标签也为 0，则 $\sin(4t)<0$。在 $2\pi<4t<4\pi$ 内，这要求 $3\pi<4t<4\pi$，即
$$
\frac{3\pi}4<t<\pi.
$$
但此时
$$
\frac{9\pi}4<3t<3\pi,
$$
该区间位于 $(2\pi,3\pi)$ 内，故 $\sin(3t)>0$，第三个标签必须为 1，矛盾。由于负标签对应严格负的正弦值，上述开区间也已经排除了所有零点边界。

$A_x=\{x,2x,3x,4x\}$ 只有四个不同点，且整个四点集不能被打散，所以
$$
\boxed{\operatorname{VCdim}(\mathcal H_{\sin}|_{A_x})\le3\quad(x\ne0).}
$$
若 $x=0$，则 $A_0=\{0\}$，而 $h_\omega(0)=1$ 对所有 $\omega$ 成立，单点的零标签无法实现。因此
$$
\boxed{\operatorname{VCdim}(\mathcal H_{\sin}|_{A_0})=0.}
$$

**(ii) 整个实数轴上的 VC 维无穷大。** 任取整数 $m\ge1$，考虑不同点
$$
x_i=2^{-i},\qquad 1\le i\le m.
$$
给定任意目标标签 $y_1,\ldots,y_m\in\{0,1\}$，令 $b_i=1-y_i$，将其编码成整数
$$
N=\sum_{j=1}^m b_j2^{j-1},\qquad
\omega=2\pi\left(N+\frac14\right).
$$
我们证明这个 $\omega$ 同时实现全部标签。固定 $i$，作带余除法
$$
N=q_i2^i+r_i,\qquad 0\le r_i\le2^i-1.
$$
由二进制展开，
$$
r_i=\sum_{j=1}^i b_j2^{j-1}.
$$
所以
$$
\omega x_i=2\pi q_i+2\pi\frac{r_i+1/4}{2^i}.
$$
若 $b_i=0$，则 $0\le r_i\le2^{i-1}-1$，因此
$$
0<\frac{r_i+1/4}{2^i}<\frac12.
$$
相应正弦值严格为正，$h_\omega(x_i)=1=y_i$。若 $b_i=1$，则 $2^{i-1}\le r_i\le2^i-1$，因此
$$
\frac12<\frac{r_i+1/4}{2^i}<1.
$$
相应正弦值严格为负，$h_\omega(x_i)=0=y_i$。

这里加入 $1/4$ 是为了避免相位恰落在正弦零点，确保目标零标签也能正确实现。由于任意标签都能实现，每个 $m$ 都存在可打散的 $m$ 点集，故
$$
\boxed{\operatorname{VCdim}(\mathcal H_{\sin})=\infty.}
$$
(i) 只讨论一类特定的四点配置，(ii) 为任意规模构造另一类配置，两者没有矛盾。VC 维取决于是否存在可打散的点集。

## 第 2 题 · Union of hypothesis classes

令 $\mathcal A,\mathcal B$ 是非空二元函数类，其有限 VC 维分别为 $d_A,d_B$；设 $\mathcal C=\mathcal A\cup\mathcal B$。对固定样本序列 $S=(x_1,\ldots,x_m)$，记标签迹集
$$
\mathcal H|_S=\{(h(x_1),\ldots,h(x_m)):h\in\mathcal H\}.
$$
则 $\Pi_{\mathcal H}(m)=\sup_S|\mathcal H|_S|$。

### (a) 增长函数的并集上界

对每个固定 $S$，
$$
\mathcal C|_S=(\mathcal A|_S)\cup(\mathcal B|_S).
$$
由有限集合的基数不等式，
$$
|\mathcal C|_S|
\le |\mathcal A|_S|+|\mathcal B|_S|
\le\Pi_{\mathcal A}(m)+\Pi_{\mathcal B}(m).
$$
右侧不依赖 $S$，故对左侧取上确界可得
$$
\boxed{\Pi_{\mathcal C}(m)\le\Pi_{\mathcal A}(m)+\Pi_{\mathcal B}(m).}
$$
这里不要求两个迹集不相交；重复的标签只会使并集更小。

### (b) 用二项式和证明严格小于 $2^m$

设整数 $m\ge d_A+d_B+2$。特别地 $d_A,d_B<m$，Sauer--Shelah 引理给出
$$
\Pi_{\mathcal A}(m)\le\sum_{j=0}^{d_A}\binom mj,\qquad
\Pi_{\mathcal B}(m)\le\sum_{j=0}^{d_B}\binom mj.
$$
利用二项式系数对称性，把第二个和移到高阶端：
$$
\sum_{j=0}^{d_B}\binom mj
=\sum_{j=m-d_B}^{m}\binom mj.
$$
由 $m\ge d_A+d_B+2$，有
$$
m-d_B\ge d_A+2.
$$
因此两段下标区间 $\{0,\ldots,d_A\}$ 与 $\{m-d_B,\ldots,m\}$ 不相交，而且至少遗漏下标 $d_A+1$。该处二项式系数严格为正，于是
$$
\begin{aligned}
\Pi_{\mathcal C}(m)
&\le\sum_{j=0}^{d_A}\binom mj+
\sum_{j=m-d_B}^{m}\binom mj\\
&\le\sum_{j=0}^{m}\binom mj-\binom m{d_A+1}\\
&=2^m-\binom m{d_A+1}<2^m.
\end{aligned}
$$
故对所有所给范围内的 $m$，
$$
\boxed{\Pi_{\mathcal C}(m)<2^m.}
$$
若 $\mathcal C$ 能打散 $d_A+d_B+2$ 个点，其增长函数在该点数处就应等于 $2^{d_A+d_B+2}$，与上式矛盾。因此
$$
\boxed{\operatorname{VCdim}(\mathcal A\cup\mathcal B)\le d_A+d_B+1.}
$$
这个论证也包括 $d_A=0$ 或 $d_B=0$；类非空时 Sauer--Shelah 的零维二项式和为 1。
使用题目要求的二项式和，能看出两端之间必有一个遗漏项，从而得到严格不等式。若过早改成粗略的多项式增长上界，就不易保留这个精确的维数常数。

## 第 3 题 · Rademacher identities

固定任意样本 $S=(x_1,\ldots,x_m)$，令 $\epsilon_1,\ldots,\epsilon_m$ 为独立 Rademacher 符号，即各以概率 $1/2$ 取 $+1$ 或 $-1$。记
$$
Z_\epsilon(h)=\frac1m\sum_{i=1}^m\epsilon_i h(x_i),\qquad
\widehat{\mathfrak R}_S(\mathcal H)=\mathbb E_\epsilon\sup_{h\in\mathcal H}Z_\epsilon(h).
$$
按题设，相关类非空，上确界与期望可测且有限。以下先证明固定 $S$ 的结论，再对 $S\sim\mu^m$ 取平均。

### (a) 数乘齐次性

当 $\alpha>0$，正数可直接从上确界中提出：
$$
\widehat{\mathfrak R}_S(\alpha\mathcal H)
=\mathbb E_\epsilon\sup_{h\in\mathcal H}\alpha Z_\epsilon(h)
=\alpha\widehat{\mathfrak R}_S(\mathcal H).
$$
当 $\alpha=0$ 时，$\alpha\mathcal H=\{0\}$，两侧均为零。

当 $\alpha<0$ 时，不能直接把负数从上确界中提出。令 $\epsilon'_i=-\epsilon_i$，则整个符号向量 $\epsilon'$ 与 $\epsilon$ 同分布。因 $\alpha=-|\alpha|$，
$$
\begin{aligned}
\widehat{\mathfrak R}_S(\alpha\mathcal H)
&=\mathbb E_\epsilon\sup_{h\in\mathcal H}
\frac1m\sum_i|\alpha|(-\epsilon_i)h(x_i)\\
&=|\alpha|\mathbb E_{\epsilon'}\sup_{h\in\mathcal H}Z_{\epsilon'}(h)
=|\alpha|\widehat{\mathfrak R}_S(\mathcal H).
\end{aligned}
$$
所以对每个固定样本，$\widehat{\mathfrak R}_S(\alpha\mathcal H)=|\alpha|\widehat{\mathfrak R}_S(\mathcal H)$。再对样本取期望：
$$
\boxed{\mathfrak R_m(\alpha\mathcal H)=|\alpha|\mathfrak R_m(\mathcal H).}
$$
这里利用的是随机符号分布的对称性，并不要求函数类 $\mathcal H$ 本身关于取负封闭。

### (b) 函数类之和的可加性

对固定的 $\epsilon$，利用 $h$ 与 $h'$ 可以独立选择，
$$
\begin{aligned}
\sup_{f\in\mathcal H+\mathcal H'}Z_\epsilon(f)
&=\sup_{h\in\mathcal H,\,h'\in\mathcal H'}
\{Z_\epsilon(h)+Z_\epsilon(h')\}\\
&=\sup_{h\in\mathcal H}Z_\epsilon(h)
+\sup_{h'\in\mathcal H'}Z_\epsilon(h').
\end{aligned}
$$
第二个等号即使上确界没有取得也成立：一方面任意两项之和不超过两个上确界之和；另一方面，对任意 $\delta>0$，分别选取距各自上确界不足 $\delta/2$ 的函数，就能把总和逼近到不足 $\delta$。

先对 $\epsilon$ 取期望，再对样本取期望，得到
$$
\widehat{\mathfrak R}_S(\mathcal H+\mathcal H')=\widehat{\mathfrak R}_S(\mathcal H)+\widehat{\mathfrak R}_S(\mathcal H'),
$$
$$
\boxed{\mathfrak R_m(\mathcal H+\mathcal H')=\mathfrak R_m(\mathcal H)+\mathfrak R_m(\mathcal H').}
$$
这里的“独立选择”是优化变量互不约束的意思，不是要求函数类或随机变量之间存在概率独立性。

### (c) 逐点最大值类的上界

设
$$
\mathcal M=\{x\mapsto\max\{h(x),h'(x)\}:h\in\mathcal H,\ h'\in\mathcal H'\},
\qquad
\mathcal D=\mathcal H-\mathcal H'.
$$
逐点恒等式 $\max\{a,b\}=(a+b+|a-b|)/2$ 给出
$$
\begin{aligned}
\widehat{\mathfrak R}_S(\mathcal M)
&=\frac12\mathbb E_\epsilon\sup_{h,h'}
\left\{Z_\epsilon(h)+Z_\epsilon(h')+Z_\epsilon(|h-h'|)\right\}\\
&\le\frac12\widehat{\mathfrak R}_S(\mathcal H)+\frac12\widehat{\mathfrak R}_S(\mathcal H')
+\frac12\widehat{\mathfrak R}_S(|\mathcal D|),
\end{aligned}
$$
其中 $|\mathcal D|=\{x\mapsto|d(x)|:d\in\mathcal D\}$。不等号来自“同一对函数之和的上确界，不超过各部分分别优化的上确界之和”；不能把这里误写成等号。

对 $\varphi(t)=|t|$，其 Lipschitz 常数为 1，且 $\varphi(0)=0$。使用[收缩引理](../../notes/zh/08-uniform-convergence.md)：
$$
\widehat{\mathfrak R}_S(|\mathcal D|)\le\widehat{\mathfrak R}_S(\mathcal D).
$$
再由 (a)、(b)，
$$
\widehat{\mathfrak R}_S(\mathcal D)
=\widehat{\mathfrak R}_S(\mathcal H+(-\mathcal H'))
=\widehat{\mathfrak R}_S(\mathcal H)+\widehat{\mathfrak R}_S(\mathcal H').
$$
合并得到
$$
\widehat{\mathfrak R}_S(\mathcal M)
\le\frac12\{\widehat{\mathfrak R}_S(\mathcal H)+\widehat{\mathfrak R}_S(\mathcal H')\}
+\frac12\{\widehat{\mathfrak R}_S(\mathcal H)+\widehat{\mathfrak R}_S(\mathcal H')\}
=\widehat{\mathfrak R}_S(\mathcal H)+\widehat{\mathfrak R}_S(\mathcal H').
$$
这是每个固定样本上的结论。对样本取期望，得到
$$
\boxed{\mathfrak R_m(\mathcal M)\le\mathfrak R_m(\mathcal H)+\mathfrak R_m(\mathcal H').}
$$

## 第 4 题 · Regularized neural networks

固定样本 $S=(x_1,\ldots,x_m)\subseteq\mathbb R^{n_1}$，隐层宽度 $n_2\ge1$，约束半径 $\Lambda_0,\Lambda\ge0$。激活函数 $\sigma$ 为 $L$-Lipschitz 且 $\sigma(0)=0$。函数类为
$$
\mathcal H=\left\{x\mapsto\sum_{j=1}^{n_2}w_j\sigma(u_j^\top x):
\|w\|_1\le\Lambda_0,\ \|u_j\|_2\le\Lambda\ \text{对每个 }j\right\}.
$$
网络没有偏置项。沿用第 3 题不预先取绝对值的 $\widehat{\mathfrak R}_S$ 定义，并令
$$
\mathcal G=\{x\mapsto u^\top x:\|u\|_2\le\Lambda\},\qquad
\mathcal H_{\rm lin}=\{x\mapsto s u^\top x:\|u\|_2\le\Lambda,\ s\in\{-1,+1\}\}.
$$
事实上 $\mathcal H_{\rm lin}=\mathcal G$：因为 $u$ 与 $-u$ 都在同一个欧氏球内，额外符号 $s$ 可以吸收到 $u$ 中。

### (a) 先优化输出权重

固定符号向量 $\epsilon$ 和隐层权重 $u_1,\ldots,u_{n_2}$，记
$$
a_j=\sum_{i=1}^m\epsilon_i\sigma(u_j^\top x_i).
$$
把有限求和交换次序，得到
$$
\sum_{i=1}^m\epsilon_i\sum_{j=1}^{n_2}w_j\sigma(u_j^\top x_i)
=\sum_{j=1}^{n_2}w_ja_j.
$$
由 $\ell_1$ 与 $\ell_\infty$ 对偶关系，
$$
\sup_{\|w\|_1\le\Lambda_0}\sum_jw_ja_j
=\Lambda_0\max_{1\le j\le n_2}|a_j|.
$$
具体地，上界来自
$$
\sum_jw_ja_j\le\sum_j|w_j||a_j|
\le\|w\|_1\max_j|a_j|\le\Lambda_0\max_j|a_j|.
$$
若 $j_*$ 取得最大的 $|a_j|$，令 $w_{j_*}=\Lambda_0\operatorname{sign}(a_{j_*})$、其余权重为零，即可达到该上界；当所有 $a_j=0$ 或 $\Lambda_0=0$ 时取 $w=0$ 即可。

再对隐层权重优化。因为 $n_2\ge1$ 且每个神经元允许使用同一个参数球，
$$
\sup_{\|u_j\|_2\le\Lambda\ \forall j}
\max_j\left|\sum_i\epsilon_i\sigma(u_j^\top x_i)\right|
=\sup_{\|u\|_2\le\Lambda}
\left|\sum_i\epsilon_i\sigma(u^\top x_i)\right|.
$$
左侧不超过右侧是直接的；反过来可把任意候选 $u$ 放在第一个神经元上，其余神经元任取合法权重。对任意逼近上确界的 $u$ 都成立，所以即使不预先假设最优点取得，等式也成立。

除以 $m$，再对符号取期望，得到
$$
\boxed{
\widehat{\mathfrak R}_S(\mathcal H)
=\frac{\Lambda_0}{m}\mathbb E_\epsilon
\sup_{\|u\|_2\le\Lambda}
\left|\sum_{i=1}^m\epsilon_i\sigma(u^\top x_i)\right|.
}
$$
这里出现绝对值，是因为输出权重允许正负号；它来自优化，不是改变了 Rademacher 复杂度的定义。

### (b) 使用绝对值收缩并说明对称性

将 (a) 写成
$$
\widehat{\mathfrak R}_S(\mathcal H)
=\Lambda_0\mathbb E_\epsilon\sup_{g\in\mathcal G}
\left|\frac1m\sum_i\epsilon_i\sigma(g(x_i))\right|.
$$
对 $\varphi=\sigma$ 使用[带绝对值的收缩引理](../../notes/zh/08-uniform-convergence.md)。其条件由 Lipschitz 性和 $\sigma(0)=0$ 满足，故
$$
\widehat{\mathfrak R}_S(\mathcal H)
\le2L\Lambda_0\mathbb E_\epsilon\sup_{g\in\mathcal G}
\left|\frac1m\sum_i\epsilon_i g(x_i)\right|.
$$
线性类 $\mathcal G$ 是对称的：$g\in\mathcal G$ 蕴含 $-g\in\mathcal G$。因此对每个固定的 $\epsilon$，
$$
\begin{aligned}
\sup_{g\in\mathcal G}|Z_\epsilon(g)|
&=\sup_{g\in\mathcal G}\max\{Z_\epsilon(g),Z_\epsilon(-g)\}\\
&=\sup_{g\in\mathcal G}Z_\epsilon(g).
\end{aligned}
$$
这一步正是使用类对称性的位置：它把**带绝对值的线性类表达式**改写成本节定义的**不带绝对值的复杂度**。由 $\mathcal G=\mathcal H_{\rm lin}$，得到
$$
\boxed{\widehat{\mathfrak R}_S(\mathcal H)\le2L\Lambda_0\widehat{\mathfrak R}_S(\mathcal H_{\rm lin}).}
$$
本论证不要求 $\sigma$ 是奇函数；ReLU 等非奇激活函数也满足所用条件。这里保留一般绝对值收缩引理中的系数 2。

### (c) 线性类复杂度的精确表达式

固定 $\epsilon$，令
$$
z_\epsilon=\sum_{i=1}^m\epsilon_i x_i\in\mathbb R^{n_1}.
$$
则
$$
\begin{aligned}
\sup_{h\in\mathcal H_{\rm lin}}Z_\epsilon(h)
&=\frac1m\sup_{\|u\|_2\le\Lambda,\ s\in\{-1,+1\}}
 s u^\top z_\epsilon\\
&=\frac1m\sup_{\|v\|_2\le\Lambda}v^\top z_\epsilon
=\frac\Lambda m\|z_\epsilon\|_2.
\end{aligned}
$$
最后一个等号的上界来自 Cauchy--Schwarz 不等式
$$
v^\top z_\epsilon\le\|v\|_2\|z_\epsilon\|_2\le\Lambda\|z_\epsilon\|_2.
$$
若 $z_\epsilon\ne0$，取 $v=\Lambda z_\epsilon/\|z_\epsilon\|_2$ 达到上界；若 $z_\epsilon=0$，两侧都为零。故
$$
\boxed{
\widehat{\mathfrak R}_S(\mathcal H_{\rm lin})
=\frac\Lambda m\mathbb E_\epsilon
\left\|\sum_{i=1}^m\epsilon_i x_i\right\|_2.
}
$$

### (d) 计算随机符号和的二阶矩

由 Cauchy--Schwarz（或平方根的 Jensen 不等式），
$$
\mathbb E_\epsilon\|z_\epsilon\|_2
\le\left(\mathbb E_\epsilon\|z_\epsilon\|_2^2\right)^{1/2}.
$$
展开平方范数：
$$
\begin{aligned}
\mathbb E_\epsilon\left\|\sum_{i=1}^m\epsilon_i x_i\right\|_2^2
&=\mathbb E_\epsilon\sum_{i=1}^m\sum_{j=1}^m
\epsilon_i\epsilon_j\langle x_i,x_j\rangle\\
&=\sum_{i,j}\mathbb E_\epsilon[\epsilon_i\epsilon_j]\langle x_i,x_j\rangle.
\end{aligned}
$$
当 $i=j$ 时，$\mathbb E\epsilon_i^2=1$；当 $i\ne j$ 时，由符号独立且均值为零，
$$
\mathbb E[\epsilon_i\epsilon_j]=\mathbb E\epsilon_i\,\mathbb E\epsilon_j=0.
$$
因此所有交叉项在期望下消失，留下
$$
\mathbb E_\epsilon\|z_\epsilon\|_2^2=\sum_{i=1}^m\|x_i\|_2^2.
$$
代回 (c)，得到
$$
\boxed{
\widehat{\mathfrak R}_S(\mathcal H_{\rm lin})
\le\frac\Lambda m\sqrt{\sum_{i=1}^m\|x_i\|_2^2}.
}
$$
这里样本已经固定，不要求样本点彼此正交；交叉项消失来自随机符号的独立性。

### (e) 有界输入下的网络复杂度及宽度依赖

若 $\|x_i\|_2\le r$ 对每个样本成立，则
$$
\sum_{i=1}^m\|x_i\|_2^2\le mr^2.
$$
由 (d)，
$$
\widehat{\mathfrak R}_S(\mathcal H_{\rm lin})\le\frac\Lambda m\sqrt{mr^2}
=\frac{\Lambda r}{\sqrt m}.
$$
再用 (b)，得到
$$
\boxed{
\widehat{\mathfrak R}_S(\mathcal H)
\le\frac{2L\Lambda_0\Lambda r}{\sqrt m}.
}
$$
该上界**不依赖隐层宽度 $n_2$**。原因已经在 (a) 中体现：输出层的总 $\ell_1$ 预算固定时，线性目标的最优权重可集中在一个最有利的神经元上，所以复杂度由单个隐层单元的上确界控制，没有出现对 $n_2$ 个单元求和的额外代价。

这个结论针对题设的无偏置网络和固定范数预算；它表示所证上界中没有宽度因子，不是对所有不受约束网络的泛化结论。

## 题目出处

题目对应 Mohri、Rostamizadeh 与 Talwalkar 的 *Foundations of Machine Learning*，第二版（2018）：第 1 题对应习题 3.13、3.15、3.17、3.19、3.20，第 2、3、4 题分别对应习题 3.24、3.8、3.11。这里重述所用条件并给出独立推导，作为学习笔记；不作为官方答案。
