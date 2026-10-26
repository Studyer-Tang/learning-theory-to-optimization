# 01 · 风险、模型类与线性回归

[中文目录](README.md) · [English](../../README.md) · [下一篇：双下降](02-double-descent.md)

给定带标签的样本，训练过程在一组函数中选择预测器。模型类决定可以表示哪些规律，损失决定学习的目标，而有限样本和优化算法共同决定实际选出的函数。本篇整理讲义 §1.1–1.4，补全风险分解及岭回归的推导。除另有说明，平方损失约定为误差平方的 $1/2$。

## 1. 总体风险与经验风险

设 $Z_i=(X_i,Y_i)$ 独立同分布于 $P$，训练样本为 $S=(Z_1,\ldots,Z_n)$。对可测预测函数 $f$ 和非负损失 $\ell$，定义

$$
L(f)=\mathbb E_{(X,Y)\sim P}\ell(f(X),Y),\qquad
\widehat L_n(f)=\frac1n\sum_{i=1}^n\ell(f(X_i),Y_i).
$$

$L(f)$ 评价新观测上的平均表现；$\widehat L_n(f)$ 由当前数据计算。记经验测度 $P_n=n^{-1}\sum_i\delta_{Z_i}$，并令 $\ell_f(x,y)=\ell(f(x),y)$，则二者也写作 $P\ell_f$ 和 $P_n\ell_f$。对事先固定且损失可积的 $f$，$\mathbb E_S\widehat L_n(f)=L(f)$；训练所得 $\widehat f_S$ 依赖同一份样本，不能直接代入这条无偏性结论。

常见损失必须与输出和标签编码配套：

| 损失 | 定义 | 预测对象 |
|---|---|---|
| 平方损失 | $\ell(a,y)=\frac12(a-y)^2$ | 实值响应 |
| 绝对损失 | $\ell(a,y)=\lvert a-y\rvert$ | 实值响应 |
| 0–1 损失 | $\ell(a,y)=\mathbf1\{ya\le0\}$，$y\in\{-1,1\}$ | 分类分数；此约定把零分数记为错误 |
| Logistic 损失 | $\log(1+e^{-ya})$，$y\in\{-1,1\}$ | 分类分数 |
| 二元交叉熵 | $-y\log p-(1-y)\log(1-p)$，$y\in\{0,1\}$ | 概率 $p\in(0,1)$ |

若 $p=(1+e^{-a})^{-1}$、$t=(y+1)/2$，则交叉熵等于 $\log(1+e^a)-ta=\log(1+e^{-ya})$。标签从 $0/1$ 改为 $\pm1$ 后，公式也要相应变换。

Huber 损失在残差 $r=a-y$ 处定义为

$$
h_\delta(r)=
\begin{cases}
\frac12r^2,&|r|\le\delta,\\
\delta|r|-\frac12\delta^2,&|r|>\delta.
\end{cases}
$$

其导数为截断后的残差 $\max(-\delta,\min(r,\delta))$。导数有界，损失仍然无界；不能因此对 Huber 损失直接套用要求随机变量有界的集中不等式。

### 平方损失为什么以条件均值为目标

**命题。** 假设 $\mathbb E Y^2<\infty$，令 $m(X)=\mathbb E[Y\mid X]$。对任意平方可积的 $f(X)$，

$$
L(f)=\frac12\mathbb E\operatorname{Var}(Y\mid X)
+\frac12\mathbb E(m(X)-f(X))^2.
\tag{1.1}
$$

因此所有平方可积函数中的最优预测为 $m$，唯一性按 $P_X$ 几乎处处相等理解。

**证明。** 条件 Jensen 不等式给出 $\mathbb E m(X)^2\le\mathbb E Y^2$，故以下交叉项可积。写 $\varepsilon=Y-m(X)$，则 $\mathbb E[\varepsilon\mid X]=0$，并且

$$
\mathbb E[\varepsilon(m(X)-f(X))]
=\mathbb E[(m(X)-f(X))\mathbb E(\varepsilon\mid X)]=0.
$$

将 $Y-f(X)=\varepsilon+m(X)-f(X)$ 展开平方，即得（1.1）。第二项非负，且为零当且仅当 $f(X)=m(X)$ 几乎处处成立。$\square$

$m$ 是分布决定的未知函数，不随本次训练样本变化；$\widehat f_S$ 是由样本选出的随机函数。若写 $Y=f^\dagger(X)+\varepsilon$ 并假设条件噪声均值为零，则 $f^\dagger=m$。只有无条件均值 $\mathbb E\varepsilon=0$ 不够：例如 $X$ 等概率取 $\pm1$、$\varepsilon=X$ 时，$\mathbb E[\varepsilon\mid X]=X$。

绝对损失的对应目标是条件中位数。固定 $X=x$，令 $G(a)=\mathbb E[|a-Y|\mid X=x]$，假设条件一阶绝对矩有限。由于绝对值函数的差商不超过 $1$，可将单侧导数移入期望，得到

$$
G'_-(a)=2F_x(a-)-1,\qquad G'_+(a)=2F_x(a)-1.
$$

凸函数的极小条件是 $G'_-(a)\le0\le G'_+(a)$，恰好等价于 $F_x(a-)\le1/2\le F_x(a)$，即条件中位数条件。

## 2. 函数类与参数化

线性函数类为 $\{x\mapsto\theta^\top x:\theta\in\mathbb R^d\}$；加入常数特征可表示截距。采用固定特征 $\phi(x)$ 后，$\theta^\top\phi(x)$ 仍对参数线性，却可以对原始输入非线性。例如 $\phi(x)=(1,x,x^2)^\top$ 能表示二次曲线。

给线性分数加固定响应映射，得到 $f_\theta(x)=\psi(\theta^\top x)$。Logistic 映射用于概率输出，指数映射用于正值输出；完整的广义线性模型还包括响应分布及配套损失。固定 $\psi$ 时，输入只通过一个线性分数进入模型，不能据此声称可以逼近任意函数。

单隐藏层 ReLU 网络为

$$
f(x)=a_0+\sum_{j=1}^m a_j\rho(w_j^\top x+b_j),\qquad
\rho(u)=\max(u,0).
$$

隐藏特征也参与训练。例如 $\rho(x)+\rho(-x)=|x|$。若固定全部隐藏特征，只训练输出层，问题又成为固定特征上的线性拟合。

预测关于输入非线性，并不意味着目标关于参数非凸。Logistic 回归配交叉熵时，

$$
\nabla^2J(\theta)=\frac1n\sum_i p_i(1-p_i)X_iX_i^\top\succeq0,
$$

所以目标凸，但有限极小点未必存在。相反，一个隐藏单元即可给出非凸例子：$J(a,w)=\frac12(2-a\rho(w))^2$ 在 $(a,w)=(1,2),(2,1)$ 处均为零，而在两点中点 $(3/2,3/2)$ 处等于 $1/32>0$，违反凸性。

如果函数类嵌套 $\mathcal F_q\subseteq\mathcal F_{q+1}$，则

$$
\inf_{f\in\mathcal F_{q+1}}L(f)\le\inf_{f\in\mathcal F_q}L(f),\qquad
\inf_{f\in\mathcal F_{q+1}}\widehat L_n(f)\le\inf_{f\in\mathcal F_q}\widehat L_n(f).
$$

这是对更大的集合取下确界的直接结果。它不保证算法输出的 $L(\widehat f_{q,S})$ 下降。网络增宽可通过加入输出权重为零的单元保留旧函数，但参数数量、有效容量和实际泛化能力并不等同。

## 3. ERM 与三项风险上界

经验风险最小化（ERM）寻找类内最低经验风险。实际输出 $\widetilde f_n\in\mathcal F$ 允许存在优化间隙：

$$
\widehat L_n(\widetilde f_n)\le
\inf_{f\in\mathcal F}\widehat L_n(f)+\varepsilon_{\rm opt},\qquad
\varepsilon_{\rm opt}\ge0.
$$

令基准类 $\mathcal F_{\rm all}\supseteq\mathcal F$，定义

$$
L^\star=\inf_{f\in\mathcal F_{\rm all}}L(f),\quad
L^\star_\mathcal F=\inf_{f\in\mathcal F}L(f),\quad
\Delta_n(\mathcal F)=\sup_{f\in\mathcal F}|L(f)-\widehat L_n(f)|.
$$

以下假设涉及的风险与下确界有限。在 $\Delta_n(\mathcal F)$ 可测的通常设定下，后续可以研究其概率界；本处先给出对每个已实现样本成立的代数不等式：

$$
\boxed{
L(\widetilde f_n)-L^\star
\le (L^\star_\mathcal F-L^\star)
+2\Delta_n(\mathcal F)+\varepsilon_{\rm opt}.
}
\tag{1.2}
$$

**证明。** 对任意 $f\in\mathcal F$，

$$
\begin{aligned}
L(\widetilde f_n)
&\le\widehat L_n(\widetilde f_n)+\Delta_n\\
&\le\inf_{g\in\mathcal F}\widehat L_n(g)+\varepsilon_{\rm opt}+\Delta_n\\
&\le\widehat L_n(f)+\varepsilon_{\rm opt}+\Delta_n\\
&\le L(f)+\varepsilon_{\rm opt}+2\Delta_n.
\end{aligned}
$$

对右侧的 $f$ 取下确界，再减 $L^\star$ 即得结论。不要求总体风险或经验风险的下确界实际达到。$\square$

第一项是表示限制产生的近似误差；第二项控制从总体平均到样本平均的差别；第三项衡量计算没有充分最小化经验风险的代价。这里的“统计项”是上界，三项并非超额风险的精确恒等分解。两次总体—经验转换分别付出一个 $\Delta_n$，所以系数为 $2$。若 $\Delta_n=\infty$，不等式仍可形式成立，但没有有限的控制作用。

一致控制整个类是必要的：算法在看过样本之后选函数，事先固定某个函数的误差界未必能约束这种选择。扩大类可能降低近似误差，却增大一致偏差的控制难度；把优化间隙降到很小，也不会自动消除前两项。

### 线性最小二乘的精确 ERM

设计矩阵 $X\in\mathbb R^{n\times d}$ 的第 $i$ 行为 $X_i^\top$，响应向量为 $y$。目标与正规方程为

$$
J(\theta)=\frac1{2n}\|y-X\theta\|^2,\qquad
\nabla J(\theta)=\frac1nX^\top(X\theta-y),\qquad
X^\top X\widehat\theta=X^\top y.
$$

若 $r=y-X\widehat\theta$，正规方程意味着 $X^\top r=0$。于是对任意 $h$，

$$
J(\widehat\theta+h)=J(\widehat\theta)+\frac1{2n}\|Xh\|^2\ge J(\widehat\theta).
$$

故正规方程的解就是全局极小点。满列秩时解唯一且等于 $(X^\top X)^{-1}X^\top y$；秩不足时取 Moore–Penrose 解 $X^\dagger y$ 可得到唯一的最小范数最小二乘解。

## 4. 偏差、方差与不可约噪声

设测试观测 $(X,Y)$ 独立于训练样本；训练算法的额外随机性也一并计入 $S$。记

$$
\bar f(x)=\mathbb E_S\widehat f_S(x),\qquad m(x)=\mathbb E[Y\mid X=x].
$$

假设以下二阶矩有限，则

$$
\boxed{
\mathbb E_SL(\widehat f_S)
=\frac12\mathbb E_X\operatorname{Var}(Y\mid X)
+\frac12\mathbb E_X(\bar f(X)-m(X))^2
+\frac12\mathbb E_X\operatorname{Var}_S(\widehat f_S(X)).
}
\tag{1.3}
$$

**证明。** 固定 $X=x$，分解

$$
Y-\widehat f_S(x)
=(Y-m(x))+(m(x)-\bar f(x))+(\bar f(x)-\widehat f_S(x)).
$$

第一、三项的相应条件均值均为零。它们的乘积条件期望也为零，因为训练样本独立于测试观测。中间项固定，故其他交叉项同样消失。展开平方，再对 $X$ 积分并乘 $1/2$，即得（1.3）。$\square$

估计偏差比较的是平均学习结果 $\bar f$ 与 $m$，近似误差比较的是整个类中最好的函数与基准。二者一般不同。估计方差描述反复重抽训练样本后的函数变动，并非一份数据中标签的方差。恒等式（1.3）也没有要求偏差随容量下降、方差随容量上升；U 形曲线只是一种可能的形状。

训练和验证误差都高时，可能是表示、优化、预处理或噪声限制；训练误差低而验证误差高时，可能是样本敏感、分布变化或选择过程不当。单次划分无法把这些现象唯一归因于统计偏差或方差。

## 5. 岭回归的谱推导

暂不拟合截距。岭回归以 $\lambda>0$ 最小化

$$
J_\lambda(\theta)=\frac1{2n}\|y-X\theta\|^2+\frac\lambda2\|\theta\|^2.
$$

梯度为零给出

$$
(X^\top X+n\lambda I_d)\widehat\theta_\lambda=X^\top y,
\qquad
\widehat\theta_\lambda=(X^\top X+n\lambda I_d)^{-1}X^\top y.
\tag{1.4}
$$

对任意非零 $v$，$v^\top(X^\top X+n\lambda I)v=\|Xv\|^2+n\lambda\|v\|^2>0$，故解唯一。$n\lambda$ 来自经验损失的 $1/n$ 归一化。当 $\lambda=0$，以下均以 $\widehat\theta_0=X^\dagger y$ 解释，不在秩不足时使用普通逆矩阵。

取秩 $r$ 的薄 SVD：

$$
X=U_r\operatorname{diag}(s_1,\ldots,s_r)V_r^\top,\qquad s_j>0.
$$

由于 $X^\top Xv_j=s_j^2v_j$、$X^\top y=\sum_j s_j(u_j^\top y)v_j$，逐方向作用逆矩阵得

$$
\widehat\theta_\lambda=
\sum_{j=1}^r\frac{s_j}{s_j^2+n\lambda}(u_j^\top y)v_j,
\quad
\widehat y_\lambda=X\widehat\theta_\lambda
=\sum_{j=1}^r a_j(u_j^\top y)u_j,
\quad a_j=\frac{s_j^2}{s_j^2+n\lambda}.
\tag{1.5}
$$

相对最小二乘，拟合值在 $u_j$ 方向收缩 $a_j$ 倍；小奇异值受到更强收缩。拟合矩阵 $H_\lambda=U_r\operatorname{diag}(a_j)U_r^\top$ 的迹为 $\sum_j a_j$，称为该线性拟合的有效自由度。$\lambda>0$ 时它通常不是幂等投影。

将 $y$ 分为列空间分量和正交分量，各项正交可得

$$
\|y-\widehat y_\lambda\|^2
=\|(I-U_rU_r^\top)y\|^2
+\sum_{j=1}^r\left(\frac{n\lambda}{s_j^2+n\lambda}\right)^2(u_j^\top y)^2.
\tag{1.6}
$$

每个变量项随 $\lambda$ 非减，因此训练残差倾向于选择 $\lambda=0$，不能用于判断正则化能否改善新数据预测。比较不同 $\lambda$ 的最小惩罚目标也不合适，因为被最小化的目标本身已经改变。

### 固定设计下的系数误差与预测误差

假设真实模型满足

$$
y=X\theta^\star+\varepsilon,\qquad
\mathbb E[\varepsilon\mid X]=0,\quad
\operatorname{Cov}(\varepsilon\mid X)=\sigma^2I_n.
$$

记 $\alpha_j=v_j^\top\theta^\star$、$P=V_rV_r^\top$。由 $u_j^\top X\theta^\star=s_j\alpha_j$，

$$
v_j^\top\widehat\theta_\lambda
=a_j\alpha_j+\frac{s_j}{s_j^2+n\lambda}u_j^\top\varepsilon.
$$

所以第 $j$ 个可识别方向的条件偏差为 $-(1-a_j)\alpha_j$，条件方差为 $\sigma^2s_j^2/(s_j^2+n\lambda)^2$。零空间上的估计恒为零，偏差为 $-(I-P)\theta^\star$。正交求和得到完整系数风险：

$$
\boxed{
\mathbb E[\|\widehat\theta_\lambda-\theta^\star\|^2\mid X]
=\|(I-P)\theta^\star\|^2
+\sum_{j=1}^r\left[
\frac{(n\lambda)^2\alpha_j^2}{(s_j^2+n\lambda)^2}
+\frac{\sigma^2s_j^2}{(s_j^2+n\lambda)^2}
\right].
}
\tag{1.7}
$$

这里只需二阶矩假设，不需要噪声高斯。增大 $\lambda$ 时，各可识别方向的偏差平方非减、方差非增；它们的和未必单调。

若独立测试特征 $x'$ 满足 $\mathbb E[x'x'^\top]=\Sigma$，记 $b_\lambda=\mathbb E[\widehat\theta_\lambda\mid X]-\theta^\star$、$C_\lambda=\operatorname{Cov}(\widehat\theta_\lambda\mid X)$，则

$$
\mathbb E[(x'^\top(\widehat\theta_\lambda-\theta^\star))^2\mid X]
=b_\lambda^\top\Sigma b_\lambda+\operatorname{tr}(\Sigma C_\lambda).
\tag{1.8}
$$

证明是先对 $x'$ 条件平均，再使用二次型的期望公式。此处 $\Sigma$ 是二阶矩矩阵；测试特征均值为零时才等于协方差。只有 $\Sigma=I_d$ 时，（1.8）直接等于（1.7）。加入独立、均值为零、方差为 $\sigma_{\rm test}^2$ 的测试标签噪声后，测试 MSE 还要增加 $\sigma_{\rm test}^2$；半平方风险再整体除以 $2$。

**例。** 仅一个可识别方向、信号系数为 $\alpha$ 时，其风险是

$$
r(t)=\frac{t^2\alpha^2+\sigma^2s^2}{(s^2+t)^2},\qquad t=n\lambda.
$$

直接求导得 $r'(t)=2s^2(t\alpha^2-\sigma^2)/(s^2+t)^3$。若 $\alpha\ne0$，最优值是 $t=\sigma^2/\alpha^2$；若 $\alpha=0$，风险随 $t$ 下降。多方向的最优收缩程度通常不同，一个共同的 $\lambda$ 必须折中。

## 6. 验证与超参数选择

参数在训练样本上拟合；特征处理、正则化强度、模型大小、步长和停止时间等选择也要计入学习过程。训练集拟合各个候选方案，独立验证集选择方案，所有选择冻结之后再用测试集评价。按测试结果反复调参后，该测试集便已参与选择。

以岭回归为例，应在共同的验证损失上比较

$$
\widehat\lambda\in\arg\min_{\lambda\in\Lambda}
\frac1{2n_{\rm val}}\|y_{\rm val}-X_{\rm val}\widehat\theta_\lambda^{\rm tr}\|^2.
$$

所有中心化、标准化及特征筛选参数均应由相应训练部分估计。截距通常不惩罚；如果借助中心化处理截距，也必须沿用训练均值变换验证与测试数据。

$k$ 折交叉验证让每个观测都由未用过它的模型预测，但各次训练集重叠，折间误差通常不独立。若同一轮交叉验证既选模型又报告最佳表现，报告仍有选择偏差；最终评价可使用独立测试集或嵌套交叉验证的外层留出。

## 练习

1. 设 $X$ 等概率取 $-1,0,1$，$Y=X^2+\varepsilon$，$\mathbb E[\varepsilon\mid X]=0$，且 $\mathbb E\varepsilon^2<\infty$。求最佳仿射预测器及其近似误差。提示：对称性使斜率为零，截距为 $2/3$，半平方近似误差为 $1/9$。
2. 对不含极小点的类证明（1.2），并解释为什么“任选一个总体最优函数”不是必要步骤。提示：证明中对任意比较函数成立，最后取下确界。
3. 令 $X=\operatorname{diag}(s_1,s_2)$ 且 $s_1\gg s_2>0$。用（1.7）比较两个方向在 $\lambda=0$ 时的噪声贡献，并说明小奇异值与正则化的关系。
4. 若测试二阶矩矩阵不在 $V_r$ 基下对角，证明系数偏差的不同奇异方向在预测偏差中会出现交叉项。提示：展开 $b_\lambda^\top\Sigma b_\lambda$；不能只把（1.7）每一项乘某个标量。

后续的[双下降](02-double-descent.md)将在一个固定的高斯回归问题中，精确计算训练误差、测试误差和随机设计带来的偏差—方差变化。推导所需的矩阵运算见[线性代数工具](linear-algebra.md)。
