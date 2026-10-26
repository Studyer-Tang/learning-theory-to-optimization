# 线性代数工具：从矩阵公式到可计算的标量

[中文目录](README.md) · [English](../../README.md)

本页集中证明风险与优化推导中反复使用的矩阵恒等式。向量均为列向量，$\|x\|$ 是 Euclidean 范数，$\|A\|_{\rm op}=\sup_{\|x\|=1}\|Ax\|$ 是算子范数。条件期望中的矩阵若依赖训练设计，先给定该设计再把它视作常量。

## 1. 迹循环与二次型的期望

若 $A\in\mathbb R^{m\times n}$、$B\in\mathbb R^{n\times m}$，则

$$
\operatorname{tr}(AB)
=\sum_{i=1}^m\sum_{j=1}^n A_{ij}B_{ji}
=\sum_{j=1}^n\sum_{i=1}^m B_{ji}A_{ij}
=\operatorname{tr}(BA).
\tag{L.1}
$$

所以 $AB$ 和 $BA$ 即使大小不同，其迹仍相等。把多个因子中的第一个看成 $A$、其余整体看成 $B$，即可循环移位，例如 $\operatorname{tr}(ABC)=\operatorname{tr}(BCA)$。这不允许任意交换因子；一般 $\operatorname{tr}(ABC)\ne\operatorname{tr}(ACB)$。

$x^\top Bx$ 是一个 $1\times1$ 矩阵，其迹就是它自身。由（L.1），

$$
x^\top Bx=\operatorname{tr}(x^\top Bx)
=\operatorname{tr}(Bxx^\top).
$$

假设 $x$ 有有限二阶矩，$\mu=\mathbb Ex$、$\Sigma=\operatorname{Cov}(x)$，而 $B$ 为确定矩阵。有限求和与期望可交换，得到

$$
\begin{aligned}
\mathbb E[x^\top Bx]
&=\operatorname{tr}(B\mathbb E[xx^\top])\\
&=\operatorname{tr}(B(\Sigma+\mu\mu^\top))\\
&=\operatorname{tr}(B\Sigma)+\mu^\top B\mu.
\end{aligned}
\tag{L.2}
$$

不要求 $x$ 高斯，也不要求其坐标独立。仅当均值为零时，二阶矩矩阵 $\mathbb E[xx^\top]$ 才等于协方差。

**条件版本。** 若 $B$ 对条件信息 $\mathcal G$ 可测，则

$$
\mathbb E[x^\top Bx\mid\mathcal G]
=\operatorname{tr}\big(B\operatorname{Cov}(x\mid\mathcal G)\big)
+\mathbb E[x\mid\mathcal G]^\top B\mathbb E[x\mid\mathcal G],
$$

其中需保证相关量可积或用非负二次型的扩展条件期望解释。若 $B$ 和 $x$ 相关，却没有先条件化，不能将 $B$ 擅自移到期望之外。

例如给定设计矩阵 $X$ 后，$\mathbb E[\eta\mid X]=0$、$\operatorname{Cov}(\eta\mid X)=\tau^2I$，对由 $X$ 决定的投影 $H$，

$$
\mathbb E[\eta^\top(I-H)\eta\mid X]
=\tau^2\operatorname{tr}(I-H).
$$

再利用投影的秩，才能把右侧化成 $\tau^2(n-\operatorname{rank}H)$。

## 2. 正交投影与子空间分解

一个实矩阵 $P$ 是正交投影，当且仅当 $P^\top=P$ 且 $P^2=P$。其作用把每个向量拆成

$$
x=Px+(I-P)x.
$$

两部分正交，因为

$$
\langle Px,(I-P)x\rangle=x^\top P(I-P)x=0.
$$

因此

$$
\|x\|^2=\|Px\|^2+\|(I-P)x\|^2,
\qquad
\|(I-P)x\|^2=x^\top(I-P)x.
\tag{L.3}
$$

第二个式子来自 $(I-P)^\top(I-P)=(I-P)^2=I-P$。若 $Pv=\lambda v$ 且 $v\ne0$，则 $P^2v=Pv$ 给出 $\lambda^2=\lambda$，所以特征值仅为 $0$ 或 $1$。于是 $\operatorname{tr}P=\operatorname{rank}P$。

对任意矩阵 $X\in\mathbb R^{n\times d}$，

$$
\operatorname{range}(X^\top)=(\ker X)^\perp.
\tag{L.4}
$$

先由 $\langle X^\top u,v\rangle=\langle u,Xv\rangle=0$ 证明左侧包含于右侧；再用秩—零化度公式得到两者维数都等于 $\operatorname{rank}X$，故相等。

若 $X$ 满列秩，则 $X(X^\top X)^{-1}X^\top$ 投影到列空间；若 $X$ 满行秩，则 $X^\top(XX^\top)^{-1}X$ 投影到行空间。可直接验证对称、幂等，再验证它们在对应子空间上为恒等。秩不足时应使用伪逆，不可继续写普通逆矩阵。

## 3. SVD：把矩阵作用逐方向展开

设秩 $r$ 的矩阵有薄奇异值分解

$$
X=U_rSV_r^\top=\sum_{j=1}^r s_ju_jv_j^\top,
\qquad s_j>0,
\qquad U_r^\top U_r=V_r^\top V_r=I_r.
$$

直接相乘可得

$$
Xv_j=s_ju_j,\qquad X^\top u_j=s_jv_j,\qquad
X^\top Xv_j=s_j^2v_j.
$$

因此右奇异向量给出参数空间中的可观察方向，左奇异向量给出数据空间中的相应方向。$V_rV_r^\top$ 是行空间投影，$U_rU_r^\top$ 是列空间投影。伪逆为

$$
X^\dagger=V_rS^{-1}U_r^\top
=\sum_{j=1}^r\frac1{s_j}v_ju_j^\top.
$$

小奇异值对应大的反演因子 $1/s_j$，这正是无正则化拟合容易放大噪声的原因。岭回归则将 $1/s_j$ 替换为 $s_j/(s_j^2+n\lambda)$。

**逐方向计算示例。** 若 $X^\top Xv_j=s_j^2v_j$，则

$$
(X^\top X+n\lambda I)v_j=(s_j^2+n\lambda)v_j,
$$

故逆矩阵作用于 $v_j$ 时只乘 $1/(s_j^2+n\lambda)$。先把 $X^\top y$ 展开成 $\sum_js_j(u_j^\top y)v_j$，再对每一项应用逆矩阵，即得岭解。这里没有把矩阵当成普通数相除，而是先选取一组使它对角化的基。

## 4. 谱分解与二次优化

设 $H=H^\top\succeq0$，取正交谱分解 $H=Q\operatorname{diag}(\lambda_j)Q^\top$。若 $e=Qz$，则

$$
e^\top He=\sum_j\lambda_jz_j^2,
\qquad
\|e\|^2=\sum_jz_j^2.
\tag{L.5}
$$

因正交变换保持长度，矩阵二次型可以逐坐标计算。若 $\mu I\preceq H\preceq\beta I$，便有 $\mu\|e\|^2\le e^\top He\le\beta\|e\|^2$。

对二次目标 $\Phi(\theta)=\frac12\theta^\top H\theta-b^\top\theta+c$，假设存在极小点 $\theta^\star$，则 $H\theta^\star=b$。展开并消去一次项：

$$
\Phi(\theta)-\Phi(\theta^\star)=\frac12(\theta-\theta^\star)^\top H(\theta-\theta^\star).
$$

梯度下降的误差满足 $e_{t+1}=(I-\eta H)e_t$，在特征基下就是 $z_{t+1,j}=(1-\eta\lambda_j)z_{t,j}$。所以

$$
\Phi(\theta_T)-\Phi(\theta^\star)
=\frac12\sum_j\lambda_j(1-\eta\lambda_j)^{2T}z_{0,j}^2.
\tag{L.6}
$$

这是目标误差精确公式的来源。若 $\mu>0$ 且 $0<\eta<2/\beta$，令 $q=\max_j|1-\eta\lambda_j|<1$，便得 $\Phi(\theta_T)-\Phi(\theta^\star)\le q^{2T}(\Phi(\theta_0)-\Phi(\theta^\star))$。若存在零特征值，相应参数分量不会收缩，但也不出现在目标误差中；不能再从目标收敛推出到某个指定极小点的距离收敛。

## 5. 把多元函数限制到一条线段

设 $\Phi$ 连续可微，$d=v-u$，定义一元函数 $h(s)=\Phi(u+sd)$，$s\in[0,1]$。链式法则和微积分基本定理给出

$$
\Phi(v)-\Phi(u)=\int_0^1h'(s)\,ds
=\int_0^1\langle\nabla\Phi(u+sd),d\rangle\,ds.
$$

减去常量 $\langle\nabla\Phi(u),d\rangle$ 后，若梯度是 $\beta$-Lipschitz，

$$
\begin{aligned}
\left|\Phi(v)-\Phi(u)-\langle\nabla\Phi(u),d\rangle\right|
&\le\int_0^1\|\nabla\Phi(u+sd)-\nabla\Phi(u)\|\,\|d\|\,ds\\
&\le\int_0^1\beta s\|d\|^2\,ds
=\frac\beta2\|d\|^2.
\end{aligned}
\tag{L.7}
$$

这里依次使用链式法则、微积分基本定理、Cauchy–Schwarz 不等式和梯度 Lipschitz 条件。它将“梯度变化不太快”转为目标函数的二次上界，是光滑优化下降引理的直接证明。

## 检查练习

1. 用坐标求和证明 $x^\top Bx=\operatorname{tr}(Bxx^\top)$，并注明两边的矩阵尺寸。
2. 取 $x\sim N(\mu,I_d)$，计算 $\mathbb E\|Ax\|^2$。答案为 $\operatorname{tr}(A^\top A)+\|A\mu\|^2$；若漏掉均值项，结果不完整。
3. 设 $X=[1\ \ 1]$。求行空间投影、零空间和响应 $y=2$ 的最小范数解。答案依次为 $\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}$、$\operatorname{span}\{(1,-1)^\top\}$ 和 $(1,1)^\top$。
4. 对 $H=\operatorname{diag}(1,4)$、$\eta=1/4$ 写出（L.6）。检查大特征值方向经过一步消失，而小特征值方向的参数误差每步乘 $3/4$。
