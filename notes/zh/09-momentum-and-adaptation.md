# 09 · 动量、自适应缩放与矩阵更新

[中文目录](README.md) · [梯度下降](03-gradient-descent.md) · [SGD 有限时间界](04-sgd-finite-time.md)

本篇区分三类事实：更新公式、二次目标上可证明的性质、一般训练中的设计动机。动量平均梯度并改变参数轨迹；PR 平均已生成的参数作为输出。自适应缩放或奇异值变换也需要各自的收敛分析，不能直接继承 GD 的定理。

## 2.6 动量：把梯度历史带进更新方向

从这里起，重点由前面 GD、SGD 的收敛定理转向优化器的构造。理解更新式为何这样设计，并不等于已经获得其在任意目标上的收敛保证。

从本节起，$g_t$ 表示在 $\theta_{t-1}$ 处计算的完整梯度或小批量梯度。初始化 $m_0=0$，取 $0\le\rho<1$，更新

$$
m_t=\rho m_{t-1}+(1-\rho)g_t,\qquad
\theta_t=\theta_{t-1}-\eta m_t. \tag{2.6.1}
$$

展开递推即可看清动量的含义：

$$
m_t=(1-\rho)\sum_{k=1}^t\rho^{t-k}g_k.
$$

越早的梯度权重越小；持续同向的梯度相互加强，快速交替的分量则相互抵消。这有助于减少狭长谷底两侧的来回摆动。$\rho$ 越大，记忆越长，也可能带来更明显的冲过头，因此要与步长共同选择。

看一维梯度序列就能验证这种作用。若每步都为同一个 $g$，展开式给出 $m_t=(1-\rho^t)g\to g$；若梯度始终在 $g$ 与 $-g$ 间交替，忽略衰减的初始过渡后，令 $m_t=cg_t$ 代入递推，因 $g_{t-1}=-g_t$，有

$$
cg_t=-\rho cg_t+(1-\rho)g_t
\quad\Longrightarrow\quad c=\frac{1-\rho}{1+\rho}.
$$

同向分量最终得到保留，交替分量则被缩小。这是在给定梯度序列下解释平均器的作用；真实训练的梯度又受参数轨迹影响，所以它本身不是整体加速定理。

对 $t\ge2$，利用 $\theta_{t-1}-\theta_{t-2}=-\eta m_{t-1}$，还可写成

$$
\theta_t=\theta_{t-1}+\rho(\theta_{t-1}-\theta_{t-2})-\eta(1-\rho)g_t.
$$

因此动量相当于让上一步位移也参与本次更新。有些定义省略 $1-\rho$，固定 $\rho$ 时可以通过重设学习率转换。

**与 PR 平均区分。** PR 平均是把已经产生的参数取平均作为输出；动量则平均梯度，并改变后续参数轨迹。本节介绍机制，没有给出动量在一般非凸目标上必然加速或收敛的定理。

### 二次目标上的稳定性与最优根半径

改用不含 $1-\rho$ 的 heavy-ball 记法，令步长为 $\gamma=\eta(1-\rho)$：

$$
\theta_{t+1}=\theta_t-\gamma H(\theta_t-\theta^*)
+\rho(\theta_t-\theta_{t-1}),\qquad \mu I\preceq H\preceq\beta I,\qquad 0<\mu\le\beta.
$$

在特征值为 $\lambda$ 的方向上，误差满足

$$
z_{t+1}=a_\lambda z_t-\rho z_{t-1},\qquad
p_\lambda(r)=r^2-a_\lambda r+\rho,\qquad
a_\lambda=1+\rho-\gamma\lambda.
$$

若 $0\le\rho<1$，两个根都在单位圆内，当且仅当

$$
0<\gamma\lambda<2(1+\rho).
$$

证明分两种情况。非实根互为共轭，模长为 $\sqrt\rho<1$，此时 $|a_\lambda|<2\sqrt\rho\le1+\rho$。实根若同时大于等于 1 或同时小于等于 $-1$，乘积至少为 1，与 $\rho<1$ 矛盾。因此 $p_\lambda(1)>0$ 排除根落在 $[1,\infty)$，$p_\lambda(-1)>0$ 排除根落在 $(-\infty,-1]$；反方向由两根在 $(-1,1)$ 直接得到。代入 $p_\lambda(1)=\gamma\lambda$、$p_\lambda(-1)=2(1+\rho)-\gamma\lambda$ 即得所述条件。全谱同时稳定因而要求 $0<\gamma<2(1+\rho)/\beta$。

当 $\mu<\beta$ 时，对整段 $\lambda\in[\mu,\beta]$ 统一保证的最坏根半径，其最小值为

$$
q_* =\frac{\sqrt\beta-\sqrt\mu}{\sqrt\beta+\sqrt\mu},\qquad
\rho_*=q_*^2,\qquad
\gamma_* =\frac4{(\sqrt\beta+\sqrt\mu)^2}.
$$

先证明没有更小的统一根半径。假设所有 $\lambda\in[\mu,\beta]$ 的根模长都不超过 $q\in(0,1)$，则根的乘积给出 $\rho\le q^2$。把根缩放为 $r/q$，它们位于闭单位圆内，故对应实系数二次式在 $1$、$-1$ 处均非负，得到

$$
|a_\lambda|\le q+\frac\rho q.
$$

这也可由实根同号、乘积为 $\rho$ 时最大根和为 $q+\rho/q$，以及共轭根时 $|a_\lambda|\le2\sqrt\rho$ 直接看出。分别代入谱的两端：

$$
\gamma\mu\ge(1-q)(1-\rho/q),\qquad
\gamma\beta\le(1+q)(1+\rho/q).
$$

因 $0\le\rho/q\le q<1$，可相除并使用 $(1+x)/(1-x)$ 单调递增：

$$
\kappa=\frac\beta\mu
\le\frac{1+q}{1-q}\frac{1+\rho/q}{1-\rho/q}
\le\left(\frac{1+q}{1-q}\right)^2.
$$

解此不等式得 $q\ge q_*$。再验证能达到：取上述 $\gamma_*,\rho_*$ 时，$a_\mu=2q_*$、$a_\beta=-2q_*$，中间的 $a_\lambda$ 位于两者之间。特征根为重根 $\pm q_*$ 或模长为 $q_*$ 的共轭根，因此最坏根半径正好为 $q_*$。若 $\mu=\beta$，取 $\rho=0,\gamma=1/\mu$ 即可一步消除误差。

端点可能产生重根，解会含 $tq_*^t$。因此这里证明的是渐近根速率：

$$
\limsup_{t\to\infty}\|e_t\|^{1/t}\le q_*;
$$

不能把它无条件写成某个固定 $C$ 满足的 $\|e_t\|\le Cq_*^t$。例如 $\mu=1,\beta=9$ 时 $\gamma_*=1/4,\rho_*=1/4,q_*=1/2$；最优固定步长 GD 的根因子为 $4/5$。这项比较依赖固定二次 Hessian；一般非线性目标的 Hessian 沿轨迹改变，需要另外分析。

## 2.7 自适应方法：AdaGrad、RMSprop 与 Adam

这类更新可以看作 $\theta_t=\theta_{t-1}-\eta D_tg_t$，其中 $D_t$ 是由历史梯度构成的对角缩放矩阵；Adam 还会把分子换成平均方向。这里的 $D_t$ 并不是已经证明准确的逆 Hessian，前面的曲率分析只是解释为何要关心尺度不均衡。

这些方法给不同坐标使用不同的缩放比例。以下乘法、平方、开方和除法都按坐标进行，$\odot$ 表示逐坐标乘法，$\epsilon>0$ 用于数值稳定，$\eta$ 是基础学习率。

### 2.7.1 AdaGrad：累计所有历史梯度平方

**设计选择：用历史平方梯度作为每个坐标的尺度。** 平方使正负梯度不会抵消，累加记录这个坐标过去被激活的程度；取平方根让分母恢复到梯度的量纲，再把当前梯度除以它。

初始化 $r_0=0$，更新

$$
r_t=r_{t-1}+g_t\odot g_t,\qquad
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{r_t}+\epsilon}. \tag{2.7.1}
$$

第 $j$ 个坐标的有效学习率为 $\eta/(\sqrt{r_{t,j}}+\epsilon)$。历史梯度频繁且大的坐标受到更强缩小；较少激活的坐标保留相对大的步长。由于 $r_t$ 只累加、不遗忘，后期有效步长可能很小。

### 2.7.2 RMSprop：让平方梯度统计遗忘旧信息

AdaGrad 的分母只增不减，早期很大的梯度可能长期压低之后的步长。RMSprop 把“全部历史”改成“近期历史”：使用指数衰减，使旧尺度的影响逐渐消失，更新能跟上训练阶段的变化。

初始化 $r_0=0$，取 $0\le\rho<1$：

$$
r_t=\rho r_{t-1}+(1-\rho)(g_t\odot g_t),\qquad
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{r_t}+\epsilon}. \tag{2.7.2}
$$

它把累计和换成指数移动平均，使分母跟随最近的梯度尺度变化。不同实现可能将 $\epsilon$ 放在根号内部；复现公式时需要确认这一约定。

### 2.7.3 Adam：方向平均与尺度平均相结合

RMSprop 的分母已经能估计近期尺度，但分子仍是单次梯度。Adam 在分子也引入动量，于是同时回答“最近通常往哪边走”和“这个坐标的梯度通常有多大”。两种统计的目标不同，因此分别使用 $\beta_1$ 与 $\beta_2$ 控制记忆长度。

初始化 $m_0=v_0=0$，取 $0\le\beta_1,\beta_2<1$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t). \tag{2.7.3}
$$

作零初始化偏差修正：

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}. \tag{2.7.4}
$$

然后更新

$$
\theta_t=\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}. \tag{2.7.5}
$$

$m_t$ 提供平滑后的方向，$v_t$ 提供每个坐标的平方梯度尺度；$v_t$ 是二阶原点矩的移动统计，不是减去均值后的方差。

对通用递推 $u_t=\beta u_{t-1}+(1-\beta)a_t$、$u_0=0$，展开得

$$
u_t=(1-\beta)\sum_{k=1}^t\beta^{t-k}a_k,
\qquad (1-\beta)\sum_{k=1}^t\beta^{t-k}=1-\beta^t.
$$

权重和小于 1，因此开始几步被零初值压低。除以 $1-\beta^t$ 后权重和正好为 1。若 $\mathbb Ea_k=m$ 不随时间改变，便有

$$
\mathbb Eu_t=(1-\beta^t)m,\qquad
\mathbb E\left[\frac{u_t}{1-\beta^t}\right]=m.
$$

分别令 $a_k=g_k$ 与 $a_k=g_k\odot g_k$，就得到 Adam 的两项修正。实际训练中梯度分布随参数变化，不能据此说 $\widehat m_t$ 在一般情况下就是“当前梯度均值的无偏估计”，更不能说参数因此无偏。

分子除以分母主要是在调节相对尺度，没有决定移动多远才安全。例如第一步有 $\widehat m_1=g_1$、$\widehat v_1=g_1\odot g_1$；忽略 $\epsilon$ 时，非零坐标的更新大小近似为 $\eta$。所以“自适应”并没有免除对整体步幅的选择。

AdaGrad 使用累计平方梯度；RMSprop 使用近期平方梯度；Adam 再加入方向的移动平均。自适应缩放不等于不需要调整基础学习率，也不自动继承前面 GD、SGD 的收敛定理。采用完整梯度时，这些更新在固定初始化后可以是确定性的；小批量抽样才引入相应随机性。

## 2.8 AdamW 与 Muon

### 2.8.1 AdamW：把权重衰减与自适应梯度更新分开

加入权重衰减，是希望每一步对参数施加直接的缩小作用。但在 Adam 中，把正则项梯度与数据梯度混在一起，会同时改变动量与平方梯度统计，于是衰减也会受到历史和坐标尺度影响。AdamW 的设计目标，就是让这种乘法衰减保持为一个可以单独控制的操作。

先用数据拟合损失的梯度计算 Adam 的 $\widehat m_t,\widehat v_t$，再作

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},\qquad\lambda\ge0. \tag{2.8.1}
$$

这称为解耦的权重衰减。普通 GD 对 $\Phi(\theta)+\lambda\|\theta\|^2/2$ 更新时，

$$
\theta_t=\theta_{t-1}-\eta\bigl(\nabla\Phi(\theta_{t-1})+\lambda\theta_{t-1}\bigr)
=(1-\eta\lambda)\theta_{t-1}-\eta\nabla\Phi(\theta_{t-1}).
$$

因此在普通 GD 中，两种写法相同。但若把 $\lambda\theta$ 加入 Adam 的输入梯度，它还会进入两个移动平均，并受到坐标缩放，通常不再等于独立乘上 $1-\eta\lambda$。AdamW 保持这两步分开。实际使用中常对权重矩阵施加衰减，而不对偏置、归一化参数等施加同样处理。

只看一个固定的对角缩放 $D$ 就能发现差别：把正则项送进缩放，更新中的参数项为 $(I-\eta\lambda D)\theta$；独立衰减则为 $(1-\eta\lambda)\theta$。前者每个坐标缩小多少取决于 $D$，后者直接由 $\eta\lambda$ 控制。完整 Adam 还会改变统计历史，因此差别更不止这一项。这里的公式说明两种操作为何不等价，并不保证任何衰减系数都改善泛化。

### 2.8.2 Muon：对矩阵动量的奇异值作变换

对矩阵而言，重要方向未必与某个单独的元素坐标重合。SVD 描述的是一组输入方向怎样被映射到输出方向，奇异值描述相应的放大程度。Muon 选择保留这些方向、调整它们的相对大小，以免动量矩阵中少数很大的奇异值主导更新。

考虑矩阵参数 $W\in\mathbb R^{p\times q}$，其梯度为 $G_t$。先得到动量矩阵

$$
B_t=\rho B_{t-1}+(1-\rho)G_t,\qquad B_0=0. \tag{2.8.2}
$$

设其秩为 $r$，紧致奇异值分解为 $B_t=U_t\Sigma_tV_t^\top$，其中只保留 $r$ 个正奇异值。定义极分解的部分等距因子

$$
O_t=\operatorname{Ortho}(B_t)=U_tV_t^\top. \tag{2.8.3}
$$

这个操作保留左右奇异方向，把正奇异值都替换成 1，零奇异值仍为零。因此它是在矩阵的奇异方向上平衡更新，而不是像 Adam 那样分别缩放每个坐标。具体地，紧致 SVD 满足 $U_t^\top U_t=V_t^\top V_t=I_r$，因此

$$
O_tO_t^\top=U_tU_t^\top,\qquad O_t^\top O_t=V_tV_t^\top.
$$

这两者一般是投影矩阵；只有相应方向满秩时才是整个空间的单位矩阵。所以对矩形或秩亏矩阵，不能笼统说行、列都组成正交单位基。

例如 $B=\operatorname{diag}(100,1)$，则其极因子是 $O=\operatorname{diag}(1,1)$：原来的两个方向强度相差一百倍，变换后相同。这个操作只说明更新的方向结构怎样改变，不代表所有被放大的弱方向都是可靠信号；最终训练效果还与噪声、步长和目标有关。

直接做 SVD 较贵，接下来用多项式迭代近似这个变换。先理解标量版本，再推广到矩阵。

### 2.8.3 标量推导：从 Newton 法得到符号归一化

这里的计算任务已经变成“给定一个动量矩阵，近似求它的极因子”。对标量，这就是把幅度除掉、只留下符号；矩阵中对应把每个正奇异值变成 1。先找到能完成标量归一化的多项式，再把它作用到矩阵，就有机会用矩阵乘法代替每步 SVD。

这里的 $k$ 是一次参数更新内部的数值迭代次数，不是训练步数 $t$。下面证明的是内部极因子计算是否准确，不是在直接证明训练损失怎样下降。

对非零标量 $b$，极因子为 $\operatorname{sign}(b)=b/\sqrt{b^2}$。核心计算是逆平方根。给定 $a>0$，令 $h(y)=y^{-2}-a$，Newton 法给出

$$
\begin{aligned}
y_{k+1}
&=y_k-\frac{h(y_k)}{h'(y_k)}
=y_k+\frac{y_k^{-2}-a}{2y_k^{-3}}\\
&=\frac12y_k(3-ay_k^2).
\end{aligned} \tag{2.8.4}
$$

令 $s_k=\sqrt a\,y_k$，即得

$$
s_{k+1}=\phi_{\rm NS}(s_k),\qquad
\phi_{\rm NS}(s)=\frac32s-\frac12s^3. \tag{2.8.5}
$$

目标变成使 $s_k\to1$。假设 $0<s_0\le1$，定义 $e_k=1-s_k^2$。由于 $3-s_k^2=2+e_k$，

$$
\begin{aligned}
e_{k+1}
&=1-\frac14s_k^2(3-s_k^2)^2\\
&=1-\frac14(1-e_k)(2+e_k)^2\\
&=\frac14e_k^2(3+e_k)\le e_k^2.
\end{aligned} \tag{2.8.6}
$$

该式保持 $0\le e_{k+1}<1$；同时更新式保持 $s_{k+1}>0$。所以

$$
0\le e_k\le e_0^{2^k}\to0,\qquad s_k\to1.
$$

这就是局部二次收敛：误差小的时候，下一步大约是原误差的平方。若 $a=b^2$、$x_k=by_k$ 且 $y_0>0$，则 $|x_k|=s_k$，符号保持不变，因此 $x_k\to\operatorname{sign}(b)$。零初值会一直是零；很小的正初值对应 $e_0$ 接近 1，不能仅凭“二次收敛”就断言前几步已经很准。

### 2.8.4 命题 2.22：经典 Newton–Schulz 矩阵极分解迭代

矩阵证明将 SVD 代入递推，逐个追踪奇异值；左右奇异方向保持不变。

设非零矩阵 $C\in\mathbb R^{m\times n}$，$m\le n$，紧致 SVD 为

$$
C=P\operatorname{diag}(\sigma_1,\ldots,\sigma_r)Q^\top,
\qquad\sigma_1\ge\cdots\ge\sigma_r>0.
$$

选 $\alpha>0$ 使 $\alpha\sigma_1\le1$，令 $X_0=\alpha C$，迭代

$$
X_{k+1}=\frac32X_k-\frac12(X_kX_k^\top)X_k. \tag{2.8.7}
$$

则 $X_k\to PQ^\top$。若 $s_i^{(k)}$ 是其正奇异值，令

$$
E_k=\max_{1\le i\le r}\{1-(s_i^{(k)})^2\},
$$

则 $0\le E_{k+1}\le E_k^2$，从而 $E_k\le E_0^{2^k}\to0$。

**证明。** 初始 $X_0=P\operatorname{diag}(\alpha\sigma_i)Q^\top$。若 $X_k=P\operatorname{diag}(s_i^{(k)})Q^\top$，由 $P^\top P=Q^\top Q=I_r$，

$$
(X_kX_k^\top)X_k=P\operatorname{diag}((s_i^{(k)})^3)Q^\top.
$$

代回更新式，可知左右奇异向量不变，每个正奇异值都满足标量递推

$$
s_i^{(k+1)}=\frac32s_i^{(k)}-\frac12(s_i^{(k)})^3.
$$

初始值位于 $(0,1]$，上一小节的证明逐个适用。每个正奇异值趋于 1，零方向不变，所以 $X_k\to PI_rQ^\top=PQ^\top$，且最大误差也满足同样的平方递推。证毕。

若 $C$ 满行秩，$P$ 为方阵、$CC^\top=P\operatorname{diag}(\sigma_i^2)P^\top$，因此

$$
(CC^\top)^{-1/2}C
=P\operatorname{diag}(\sigma_i^{-1})P^\top
P\operatorname{diag}(\sigma_i)Q^\top
=PQ^\top.
$$

秩亏时紧致 SVD 的表述仍然成立，无需对奇异矩阵求普通逆平方根。

这条结论说明可以用反复矩阵乘法逼近极因子，并给出误差递推。它控制的是正奇异值到 1 的偏离程度；不是说训练目标每做一次内部迭代就下降某个比例。初始归一化条件也属于结论的一部分，不能直接把任意大矩阵代入后仍沿用证明。

### 2.8.5 从经典迭代到 Muon 的有限步近似

精确极因子在很小的奇异值方向上可能需要较多内部迭代。下面的多项式针对有限步近似，并不沿用经典迭代的渐近结论。

先令 $C_t=B_t$（$p\le q$），否则令 $C_t=B_t^\top$，保证行数不多于列数。使用 Frobenius 范数归一化：

$$
X_{t,0}=\frac{C_t}{\|C_t\|_F+\epsilon_{\rm NS}},\qquad\epsilon_{\rm NS}>0.
$$

由于最大奇异值不超过全部奇异值平方和的平方根，即 $\|C_t\|_{\rm op}\le\|C_t\|_F$，每个正的初始奇异值满足 $\sigma_i(C_t)/(\|C_t\|_F+\epsilon_{\rm NS})<1$。若采用经典三次迭代，命题 2.22 就适用；若 $C_t=0$，迭代结果也始终为零。

课程介绍的 Muon 使用经过调节的**五次多项式**，执行五步（$k=0,\ldots,4$）：

$$
A_{t,k}=X_{t,k}X_{t,k}^\top,\qquad
X_{t,k+1}=aX_{t,k}+(bA_{t,k}+cA_{t,k}^2)X_{t,k}, \tag{2.8.8}
$$

其中 $(a,b,c)=(3.4445,-4.7750,2.0315)$。在奇异值上的对应标量变换为

$$
\phi_M(s)=as+bs^3+cs^5.
$$

在原点附近，$\phi_M(s)\approx3.4445s$，比经典迭代的 $1.5s$ 更快抬高极小奇异值。在线性近似范围内，五次复合的放大倍数约为 $3.4445^5\approx485$，经典迭代则约为 $1.5^5\approx7.6$；较大值受到高次项调节。

这里有两个不同的“五”：**每一步的多项式次数为五；一共执行五步。** 五步是计算量与经验效果的选择，不是“五步精确得到极因子”的定理。直接代入也能看出

$$
\phi_M(1)=3.4445-4.7750+2.0315=0.701\ne1.
$$

所以不能把命题 2.22 的渐近收敛结论搬给这组系数。内部近似误差与训练效果也属于不同的评价指标。

必要时转置回原形状，得到 $\widetilde O_t$。带解耦衰减的示意更新为

$$
W_t=(1-\eta\lambda)W_{t-1}-\eta s_{p,q}\widetilde O_t,
\qquad\widetilde O_t\approx O_t. \tag{2.8.9}
$$

$s_{p,q}>0$ 是依赖矩阵形状的缩放因子；具体缩放依实现而定。Muon 主要用于隐藏层二维权重及适当重排后的卷积核；嵌入、输出头、偏置等参数常交给 AdamW 一类辅助优化器。

AdamW 将权重收缩与自适应更新解耦；Muon 先平均矩阵梯度，再用有限步多项式压平奇异值尺度。经典 Newton–Schulz 的收敛证明解释了极因子计算，但并不等于 Muon 训练过程的收敛证明。


## 计算与阅读

1. 对 $\mu=1,\beta=9$，求 heavy-ball 的两个端点重根，并写出各自的一般解。核对：$(c_1+c_2t)(1/2)^t$ 与 $(c_3+c_4t)(-1/2)^t$。
2. 设 $D=\operatorname{diag}(1,10)$，比较正则化梯度经过 $D$ 缩放与解耦衰减的两个坐标系数。
3. 取 $C=\operatorname{diag}(1,0)$，证明经典 Newton–Schulz 极限不是满秩正交矩阵；再计算调节多项式在 $s=1$ 处的值。

AdamW 的定义参见 [Loshchilov 与 Hutter 的论文](https://arxiv.org/abs/1711.05101)。Muon 的有限步系数、矩阵形状处理及辅助优化器参见[作者实现](https://github.com/KellerJordan/Muon/blob/master/muon.py)。这些链接对应算法定义；本篇的二次根分析与经典极分解证明不构成通用的深度网络训练保证。
