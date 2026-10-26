# 02 · 高斯线性回归中的双下降

[中文目录](README.md) · [English](../../README.md) · [上一篇：风险与线性模型](01-risk-and-linear-models.md)

在插值阈值附近，最小二乘可能把很小的噪声放大成很大的系数误差。进入过参数区以后，最小范数选择又改变了噪声的传播方式。本篇按讲义 §1.5 的高斯模型计算风险，补全随机投影和逆 Gram 矩阵的期望。所得公式解释双下降的一种机制，不是所有模型的曲线形状定理。

## 1. 固定真实分布，只增加使用的特征

设输入 $x=(x_1,x_2,\ldots)$ 的坐标独立服从 $N(0,1)$，响应为

$$
y=m(x)+\varepsilon,\qquad
m(x)=\sum_{j\ge1}\beta_jx_j,\qquad
\sum_{j\ge1}\beta_j^2<\infty,
$$

其中 $\varepsilon\sim N(0,\sigma^2)$ 与全部输入独立。信号级数按均方收敛理解。真实系数序列固定不变；改变 $d$ 只表示学习者使用前 $d$ 个坐标。

取 $n$ 个独立训练观测，令 $X_d\in\mathbb R^{n\times d}$ 为保留特征的设计矩阵，$y\in\mathbb R^n$ 为响应向量。记

$$
\theta_d=(\beta_1,\ldots,\beta_d)^\top,\quad
B_d^2=\sum_{j\le d}\beta_j^2,\quad
a_d=\sum_{j>d}\beta_j^2,\quad
\tau_d^2=\sigma^2+a_d.
$$

保留坐标与遗漏坐标独立，因此

$$
y=X_d\theta_d+\eta_d,\qquad
\eta_d\sim N(0,\tau_d^2I_n),\qquad \eta_d\perp X_d.
\tag{2.1}
$$

$a_d$ 是遗漏的信号能量，$B_d^2$ 是保留的信号能量。即使原始标签噪声 $\sigma^2=0$，遗漏特征仍会产生有效噪声 $a_d$。

以下取 $n\ge2$、$d\ge1$，不拟合截距；估计量固定为最小范数最小二乘解

$$
\widehat\beta_d=X_d^\dagger y,\qquad
\widehat f_d(x)=x_{1:d}^\top\widehat\beta_d.
$$

## 2. 最小二乘解与行空间、零空间

独立连续高斯矩阵几乎必然具有秩 $\min(n,d)$。例如按顺序加入各列时，一列落入已有较低维子空间的概率为零；行满秩同理。因此

$$
\widehat\beta_d=
\begin{cases}
(X_d^\top X_d)^{-1}X_d^\top y,&d<n,\\
X_d^{-1}y,&d=n,\\
X_d^\top(X_dX_d^\top)^{-1}y,&d>n.
\end{cases}
\tag{2.2}
$$

$d\ge n$ 时可插值任意响应向量，但 $d>n$ 时必须指定从无穷多个插值解中选择哪一个。

**最小范数性质。** 设 $d>n$，记

$$
A_d=X_d^\top(X_dX_d^\top)^{-1},\qquad
\Pi_d=A_dX_d.
$$

$X_dA_d=I_n$，所以 $b_0=A_dy$ 满足 $X_db_0=y$。任意其他解都写成 $b_0+v$，其中 $v\in\ker X_d$。由于 $b_0\in\operatorname{range}(X_d^\top)$，而

$$
\langle X_d^\top u,v\rangle=\langle u,X_dv\rangle=0,
$$

行空间与零空间正交。二者维数分别是 $n$ 和 $d-n$，故互为正交补。于是 $\|b_0+v\|^2=\|b_0\|^2+\|v\|^2$，唯一最小范数解就是 $b_0$。

矩阵 $\Pi_d$ 满足

$$
\Pi_d^\top=\Pi_d,\qquad
\Pi_d^2=\Pi_d,\qquad
X_d(I_d-\Pi_d)=0.
$$

此外对任意行空间向量 $X_d^\top u$，$\Pi_dX_d^\top u=X_d^\top u$。因此 $\Pi_d$ 是行空间的正交投影，$I_d-\Pi_d$ 是零空间的正交投影。设计矩阵来自随机抽样，行空间和 $\Pi_d$ 也随样本变化；给定 $X_d$ 之后，它们才成为固定的矩阵。

### 零初始化梯度下降为何选中伪逆解

对 $\|y-X_db\|^2/(2n)$ 用固定步长 $0<\eta<2n/s_{\max}^2$。在薄 SVD 的右奇异向量 $v_j$ 方向，$c_{j,t}=v_j^\top b_t$ 满足

$$
c_{j,t+1}=(1-\eta s_j^2/n)c_{j,t}
+\eta s_j(u_j^\top y)/n.
$$

递推系数的绝对值小于 $1$，所以该坐标趋于 $(u_j^\top y)/s_j$。零空间没有梯度，其坐标始终等于初始化值。零初始化便使极限恰为 $X_d^\dagger y$；一般初始化若有非零的零空间分量，极限不再是最小范数解。

## 3. 期望训练误差

$d<n$ 时令 $H_d=X_d(X_d^\top X_d)^{-1}X_d^\top$。它是列空间的秩 $d$ 正交投影，因此

$$
y-X_d\widehat\beta_d=(I_n-H_d)y=(I_n-H_d)\eta_d.
$$

给定 $X_d$，利用投影对称、幂等以及条件协方差 $\tau_d^2I_n$，

$$
\begin{aligned}
\mathbb E[\|(I_n-H_d)\eta_d\|^2\mid X_d]
&=\mathbb E[\eta_d^\top(I_n-H_d)\eta_d\mid X_d]\\
&=\operatorname{tr}\big((I_n-H_d)\mathbb E[\eta_d\eta_d^\top\mid X_d]\big)\\
&=\tau_d^2\operatorname{tr}(I_n-H_d)=\tau_d^2(n-d).
\end{aligned}
$$

第二个等号也可逐坐标核对：$\mathbb E\sum_{i,j}B_{ij}\eta_i\eta_j=\sum_{i,j}B_{ij}\tau_d^2\mathbf1\{i=j\}=\tau_d^2\sum_iB_{ii}$。迹的使用没有增加额外概率假设。

于是平均半平方训练风险为

$$
\boxed{
T(d)=\mathbb E\frac1{2n}\|y-X_d\widehat\beta_d\|^2
=\begin{cases}
\frac{\tau_d^2}{2}(1-d/n),&d<n,\\
0,&d\ge n.
\end{cases}}
\tag{2.3}
$$

有效噪声为正时，$d<n$ 的残差几乎必然非零，$d=n$ 是任意响应的插值阈值。若无噪声且低维模型已包含全部信号，训练误差可以更早归零。

## 4. 测试风险先化为系数误差

取独立测试观测 $(x',y')$，令 $\Delta_d=\widehat\beta_d-\theta_d$。同（2.1），可写

$$
y'=x_{1:d}'{}^\top\theta_d+\eta_d',\qquad
\mathbb E\eta_d'=0,\quad\mathbb E(\eta_d')^2=\tau_d^2,
$$

其中 $x'_{1:d}$、$\eta'_d$、训练样本相互独立。给定训练样本后 $\Delta_d$ 固定，故

$$
\begin{aligned}
\mathbb E[(y'-\widehat f_d(x'))^2\mid S]
&=\mathbb E[(\eta_d'-x_{1:d}'{}^\top\Delta_d)^2\mid S]\\
&=\tau_d^2+\Delta_d^\top I_d\Delta_d
=\tau_d^2+\|\Delta_d\|^2.
\end{aligned}
\tag{2.4}
$$

以下计算测试 MSE $\mathcal E_d=\mathbb E(y'-\widehat f_d(x'))^2$。若使用前篇的半平方损失，风险为 $\mathcal E_d/2$。

### 欠参数区

$d<n$ 时，

$$
\Delta_d=(X_d^\top X_d)^{-1}X_d^\top\eta_d.
$$

给定 $X_d$，均值为零、协方差为 $\tau_d^2(X_d^\top X_d)^{-1}$，因此

$$
\mathbb E[\|\Delta_d\|^2\mid X_d]
=\tau_d^2\operatorname{tr}((X_d^\top X_d)^{-1}).
\tag{2.5}
$$

当 $n>d+1$，第 6 节将证明 $\mathbb E(X_d^\top X_d)^{-1}=I_d/(n-d-1)$。代入（2.4）得到

$$
\mathcal E_d=\tau_d^2\left(1+\frac d{n-d-1}\right),\qquad d<n-1.
\tag{2.6}
$$

### 过参数区

$d>n$ 时，将（2.1）代入最小范数解：

$$
\widehat\beta_d=\Pi_d\theta_d+A_d\eta_d,\qquad
\Delta_d=-(I_d-\Pi_d)\theta_d+A_d\eta_d.
$$

第一项在零空间，第二项在行空间。对每次实现，交叉内积本身就为零，不需要先取期望。又有

$$
A_d^\top A_d
=(X_dX_d^\top)^{-1}X_dX_d^\top(X_dX_d^\top)^{-1}
=(X_dX_d^\top)^{-1}.
$$

于是

$$
\mathbb E[\|\Delta_d\|^2\mid X_d]
=\|(I_d-\Pi_d)\theta_d\|^2
+\tau_d^2\operatorname{tr}((X_dX_d^\top)^{-1}).
\tag{2.7}
$$

第一项是当前随机设计无法观察的信号，第二项是行空间内的噪声放大。

## 5. 随机行空间的平均投影

**命题。** 在标准高斯设计、$d>n$ 的条件下，

$$
\mathbb E\Pi_d=\frac ndI_d,
\qquad
\mathbb E\|(I_d-\Pi_d)\theta_d\|^2
=\left(1-\frac nd\right)B_d^2.
\tag{2.8}
$$

**证明。** 对任意确定的正交矩阵 $Q$，$X_dQ$ 与 $X_d$ 同分布，且其行空间投影为 $Q^\top\Pi_dQ$。投影的算子范数为 $1$，所有元素可积，故 $M=\mathbb E\Pi_d$ 满足 $Q^\top MQ=M$。

先取只翻转一个坐标符号的对角矩阵 $Q$。对应行列的非对角元变号，分布却不变，所以 $M$ 的非对角元全为零。再取交换两个坐标的置换矩阵，得到所有对角元相等，故 $M=cI_d$。投影秩为 $n$，其特征值为 $n$ 个 $1$ 和 $d-n$ 个 $0$，因此

$$
cd=\operatorname{tr}M
=\mathbb E\operatorname{tr}\Pi_d=n.
$$

最后，$\Pi_d^2=\Pi_d$ 意味着 $(I_d-\Pi_d)^2=I_d-\Pi_d$，所以

$$
\begin{aligned}
\mathbb E\|(I_d-\Pi_d)\theta_d\|^2
&=\mathbb E[\theta_d^\top(I_d-\Pi_d)^\top(I_d-\Pi_d)\theta_d]\\
&=\theta_d^\top(I_d-\mathbb E\Pi_d)\theta_d
=(1-n/d)\|\theta_d\|^2.
\end{aligned}
$$

$\square$

$n/d$ 是随机行空间平均保留的能量比例，不意味着每一个实现都恰好保留这个比例。固定方向 $\theta_d$ 与随机投影作用后再平均，才得到（2.8）。

结合第 6 节的逆矩阵均值，对 $d>n+1$，

$$
\boxed{
\mathcal E_d
=\left(1-\frac nd\right)B_d^2
+\tau_d^2\left(1+\frac n{d-n-1}\right).
}
\tag{2.9}
$$

## 6. 逆 Gram 矩阵的首矩及其存在条件

**引理。** 设 $Z\in\mathbb R^{m\times p}$ 的元素独立服从 $N(0,1)$，且 $m\ge p\ge1$。当 $m>p+1$，

$$
\mathbb E[(Z^\top Z)^{-1}]=\frac1{m-p-1}I_p.
\tag{2.10}
$$

当 $m=p$ 或 $m=p+1$，矩阵仍几乎必然可逆，但每个对角元的期望都是正无穷。

**证明。** 写 $Z=[z,W]$，其中 $W$ 有 $p-1$ 列。为求 $B=(Z^\top Z)^{-1}$ 的第一个对角元，解 $(Z^\top Z)b=e_1$：

$$
\begin{cases}
z^\top z\,b_1+z^\top Wb_{2:p}=1,\\
W^\top z\,b_1+W^\top Wb_{2:p}=0.
\end{cases}
$$

第二式给出 $b_{2:p}=-(W^\top W)^{-1}W^\top z\,b_1$，代回第一式得到 Schur 补公式

$$
B_{11}=\frac1{z^\top Mz},\qquad
M=I_m-W(W^\top W)^{-1}W^\top.
\tag{2.11}
$$

$p=1$ 时 $W$ 为空矩阵，定义 $M=I_m$。$M$ 是秩 $k=m-p+1$ 的正交投影。给定 $W$ 后，$z$ 仍为独立的标准高斯向量；存在正交变换把 $M$ 变成 $\operatorname{diag}(I_k,0)$，因而 $z^\top Mz$ 的条件分布为 $\chi_k^2$，且该分布不依赖 $W$。

若 $Q\sim\chi_k^2$，由密度直接积分：

$$
\begin{aligned}
\mathbb E(Q^{-1})
&=\frac1{2^{k/2}\Gamma(k/2)}
\int_0^\infty q^{k/2-2}e^{-q/2}\,dq\\
&=\frac{\Gamma(k/2-1)}{2\Gamma(k/2)}
=\frac1{k-2},\qquad k>2.
\end{aligned}
\tag{2.12}
$$

最后一步用 $\Gamma(a)=(a-1)\Gamma(a-1)$。当 $k\le2$，被积函数在零点附近的幂指数不超过 $-1$，所以积分发散。由对称性，各个 $B_{jj}$ 都有相同的期望。

当 $k>2$，正定性给出 $|B_{ij}|\le\sqrt{B_{ii}B_{jj}}\le(B_{ii}+B_{jj})/2$，故非对角元也绝对可积。翻转 $Z$ 的第 $i$ 列符号不改变其分布，却使 $B_{ij}$（$j\ne i$）变号，因此 $\mathbb EB_{ij}=0$。这便得到（2.10）。$\square$

在欠参数区取 $(m,p)=(n,d)$，在过参数区对 $Z=X_d^\top$ 取 $(m,p)=(d,n)$，分别得到（2.6）与（2.9）的分母。

### 插值阈值附近为何出现无限均值

当 $d=n-1,n,n+1$ 时，上述逆卡方变量的自由度为 $1$ 或 $2$。只要该维数下 $\tau_d^2>0$，条件噪声误差的非负期望就为无穷，因此 $\mathcal E_d=\infty$。

几乎必然可逆不等于逆矩阵具有有限均值。对几乎每份样本，估计值和条件测试误差都是有限的；但接近奇异的矩阵会产生足够重的尾部，使重复抽样的平均发散。有限次模拟看见的有限均值，并不推翻这个结论。

若 $\tau_d^2=0$，不应写成 $0\times\infty$：此时 $\eta_d=0$，噪声项直接消失。$d\le n$ 时可精确恢复 $\theta_d$，测试 MSE 为零；$d>n$ 时仍可能有零空间信号误差，其平均为 $(1-n/d)B_d^2$。

## 7. 偏差—方差分解仍然成立

在 $d>n+1$ 区域，令 $c=n/d$。由 $\mathbb E\Pi_d=cI_d$ 和条件噪声均值为零，

$$
\mathbb E_S\widehat\beta_d=c\theta_d.
$$

将平均预测与完整信号 $m(x)$ 比较，保留坐标上的偏差为 $(c-1)\theta_d$，遗漏坐标的偏差为 $-\beta_j$。独立标准输入使交叉项消失，因此积分后的预测偏差平方为

$$
\operatorname{Bias}_d^2=a_d+(1-c)^2B_d^2.
\tag{2.13}
$$

系数围绕均值的变动为

$$
\widehat\beta_d-c\theta_d=(\Pi_d-cI_d)\theta_d+A_d\eta_d.
$$

噪声项条件均值为零，因此两项的期望交叉内积为零。再由 $\Pi_d^2=\Pi_d$，

$$
\mathbb E(\Pi_d-cI_d)^2
=\mathbb E\Pi_d-2c\mathbb E\Pi_d+c^2I_d
=c(1-c)I_d.
$$

故预测方差为

$$
\operatorname{Var}_d=c(1-c)B_d^2
+\tau_d^2\frac n{d-n-1}.
\tag{2.14}
$$

于是 $\mathcal E_d=\sigma^2+\operatorname{Bias}_d^2+\operatorname{Var}_d$，与（2.9）一致：$(1-c)^2+c(1-c)=1-c$，且 $\sigma^2+a_d=\tau_d^2$。

（2.13）的遗漏信号属于相对完整条件均值的偏差；（2.14）的第一项来自随机行空间本身，即使没有噪声也可能存在。偏差—方差恒等式并未失效，失效的是“增加参数一定使方差上升”的额外直觉。

## 8. 风险公式并不保证固定形状

欠参数且相邻两项均有限，即 $d<n-2$ 时，利用 $\tau_{d+1}^2=\tau_d^2-\beta_{d+1}^2$，有

$$
\mathcal E_{d+1}-\mathcal E_d
=\frac{n-1}{(n-d-2)(n-d-1)}
\left[\tau_d^2-(n-d-1)\beta_{d+1}^2\right].
$$

因此

$$
\mathcal E_{d+1}<\mathcal E_d
\quad\Longleftrightarrow\quad
\beta_{d+1}^2>\frac{\tau_d^2}{n-d-1}.
\tag{2.15}
$$

新增特征必须含有足够信号，才足以抵消估计变动。加入完全无关的特征会增加这一侧的风险；仅仅嵌套函数类并不保证先有一段下降。

过参数区可写成

$$
\mathcal E_d=(1-n/d)B_d^2+\tau_d^2v_d,
\qquad v_d=\frac{d-1}{d-n-1}.
$$

其中

$$
v_{d+1}-v_d=-\frac n{(d-n)(d-n-1)}<0.
$$

有效噪声贡献 $\tau_d^2v_d$ 非增，但零空间信号项 $(1-n/d)B_d^2$ 非减；两者竞争，不能只看噪声项就断言总风险下降。特征顺序、信号强度、噪声、正则化及最小范数选择都会影响形状。

**例：纯噪声。** 若全部 $\beta_j=0$ 且 $\sigma^2>0$，欠参数风险从 $\sigma^2(n-1)/(n-d-1)$ 随 $d$ 上升；过参数风险为 $\sigma^2(d-1)/(d-n-1)$，随 $d$ 下降。存在插值峰，却没有“先下降”的第一段。

**固定样本量的远端极限。** 在本模型 $\sum_j\beta_j^2<\infty$、固定 $n$ 的条件下，$d\to\infty$ 时（2.9）趋于

$$
\sigma^2+\sum_{j\ge1}\beta_j^2=\mathbb Ey^2.
$$

这恰是零预测器的 MSE。第二次下降不意味着任意增加无关特征最终可以恢复全部信号，也不意味着无限宽优于所有有限宽度。

这些结论依赖各向同性独立高斯特征、固定的真实问题、平方损失和指定的最小范数规则。对相关特征、非高斯设计、非线性网络或非零正则化，不能不加验证地沿用相同分母及风险曲线。

## 练习

1. 设 $n=10,d=3,\tau_d^2=2$，计算平均训练半平方风险与测试 MSE。答案分别为 $0.7$ 和 $3$；二者使用了不同的归一化，且前者含拟合数据的乐观偏差。
2. 证明任意满行秩矩阵的最小范数插值解都在行空间内；若加入一个非零的零空间分量，说明训练误差和参数范数分别如何变化。
3. 不借助 Wishart 密度，证明 $Z\in\mathbb R^{m\times1}$ 时 $\mathbb E(Z^\top Z)^{-1}=1/(m-2)$，并解释 $m=1,2$ 为何发散。提示：使用（2.12）。
4. 在纯噪声例子中验证两侧的单调性，再说明为什么有限次 Monte Carlo 不能据此给出临界维数风险的有限真值。
5. 从（2.13）和（2.14）独立相加恢复（2.9），标明不可约噪声、遗漏信号和行空间随机性各出现在哪一项。

矩阵细节可对照[线性代数工具](linear-algebra.md)；风险分解的统一约定见[上一篇](01-risk-and-linear-models.md)。
