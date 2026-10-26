# 作业 4 · 集中不等式、重尾截断与覆盖数

[中文目录](../../notes/zh/README.md) · [集中不等式](../../notes/zh/07-concentration.md) · [作业 5](homework05.md)

本作业共 5 道大题、15 个小问，讨论从矩母函数推导集中不等式、从尾概率控制期望、估计轻尾随机变量的最大值、截断重尾变量，以及用体积估计欧氏覆盖数。各题的条件只在对应题内使用。

所有 $\log$ 均为自然对数。$\mathbb E$ 表示期望，$\mathbb P$ 表示概率，$\mathbf1_A$ 表示事件 $A$ 的指示函数。第 2 题的参数 $b$ 与第 1 题的有界阈值同名，但各自含义以题面为准。

| 题目 | 需要证明什么 | 核心方法 |
|---|---|---|
| 第 1 题 | Bennett 与双侧 Bernstein 不等式 | 指数级数、独立性、指数 Markov、不等式比较 |
| 第 2 题 | Bernstein 尾界下的期望误差 | 尾积分公式、概率截断为 1、矩条件推出 MGF |
| 第 3 题 | 多个轻尾变量的最大值期望 | 指数化最大值、Jensen、求和上界、选择参数 |
| 第 4 题 | Pareto 样本和的高概率误差 | 对称性、截断、Bernstein、并集界、解二次方程 |
| 第 5 题 | 球与球面的覆盖数 | 覆盖体积下界、分离点集、不相交小球 |

## 常用基础工具

**指数 Markov 不等式。** 对随机变量 $W$ 和 $\lambda>0$，若矩母函数有限，则

$$
\mathbb P(W\ge a)
=\mathbb P(e^{\lambda W}\ge e^{\lambda a})
\le e^{-\lambda a}\mathbb E e^{\lambda W}.
$$

这是先把尾事件变成非负随机变量的事件，再使用普通 Markov 不等式。

**并集界。** 不要求事件独立：

$$
\mathbb P\left(\bigcup_i A_i\right)\le\sum_i\mathbb P(A_i).
$$

**尾积分公式。** 若 $Z\ge0$，则

$$
\mathbb EZ=\int_0^\infty\mathbb P(Z\ge t)\,dt.
$$

因为每个实现都满足 $Z=\int_0^\infty\mathbf1_{\{Z\ge t\}}\,dt$，由非负函数的 Tonelli 定理交换积分和期望。该公式允许先从概率上界证明期望有限。

**凸函数的 Jensen 不等式。** 特别地，

$$
e^{\lambda\mathbb EW}\le\mathbb Ee^{\lambda W}.
$$



## 第1题 Bennett不等式

### 题目条件

设 $X_1,\ldots,X_n$ 相互独立，满足

$$
\mathbb EX_i=0,\qquad |X_i|\le b\ \text{几乎必然},\qquad b>0.
$$

记

$$
\sigma_i^2=\mathbb EX_i^2,\qquad
v=\frac1n\sum_{i=1}^n\sigma_i^2>0,
\qquad
h(u)=(1+u)\log(1+u)-u\quad(u\ge0).
$$

本题先控制单个变量的矩母函数，再利用独立性控制样本和，最后把 Bennett 的函数 $h$ 下界化为更容易使用的有理式。

### 第1题a 单变量的对数矩母函数界

**结论。** 对每个 $\lambda\ge0$，

$$
\log\mathbb Ee^{\lambda X_i}
\le\frac{\sigma_i^2}{b^2}(e^{\lambda b}-1-\lambda b).
$$

 均值为零使指数级数的一阶项消失；有界性使每个高阶矩都能用二阶矩控制。

对任意整数 $k\ge2$，有

$$
\mathbb EX_i^k
\le\mathbb E|X_i|^k
\le b^{k-2}\mathbb EX_i^2
=b^{k-2}\sigma_i^2.
$$

第一步没有假设奇数阶矩非负：若它为负，这个上界仍成立。由于 $X_i$ 有界，指数级数在其取值范围上一致绝对收敛，可以逐项取期望。故

$$
\begin{aligned}
\mathbb Ee^{\lambda X_i}
&=1+\lambda\mathbb EX_i
+\sum_{k=2}^\infty\frac{\lambda^k\mathbb EX_i^k}{k!}\\
&\le1+\sigma_i^2\sum_{k=2}^\infty\frac{\lambda^kb^{k-2}}{k!}\\
&=1+\frac{\sigma_i^2}{b^2}
\sum_{k=2}^\infty\frac{(\lambda b)^k}{k!}\\
&=1+\frac{\sigma_i^2}{b^2}(e^{\lambda b}-1-\lambda b).
\end{aligned}
$$

括号中的量非负。利用 $\log(1+a)\le a$（$a\ge0$），得到

$$
\boxed{
\log\mathbb Ee^{\lambda X_i}
\le\frac{\sigma_i^2}{b^2}(e^{\lambda b}-1-\lambda b).
}
$$

$\lambda=0$ 时两边均为零，结论同样成立。

### 第1题b Bennett上尾与下尾界

**结论。** 对 $t>0$，

$$
\mathbb P\left(\frac1n\sum_{i=1}^nX_i\ge t\right)
\le\exp\left[-\frac{nv}{b^2}h\left(\frac{bt}{v}\right)\right].
$$

令 $S_n=\sum_{i=1}^nX_i$。对任意 $\lambda>0$，指数 Markov 不等式给出

$$
\mathbb P(S_n\ge nt)
\le e^{-\lambda nt}\mathbb Ee^{\lambda S_n}.
$$

由独立性，

$$
\mathbb Ee^{\lambda S_n}=\prod_{i=1}^n\mathbb Ee^{\lambda X_i}.
$$

取对数，应用 (a)：

$$
\begin{aligned}
\log\mathbb Ee^{\lambda S_n}
&=\sum_{i=1}^n\log\mathbb Ee^{\lambda X_i}\\
&\le\frac{\sum_i\sigma_i^2}{b^2}(e^{\lambda b}-1-\lambda b)\\
&=\frac{nv}{b^2}(e^{\lambda b}-1-\lambda b).
\end{aligned}
$$

因此

$$
\mathbb P(S_n\ge nt)
\le\exp\left[-n\lambda t+
\frac{nv}{b^2}(e^{\lambda b}-1-\lambda b)\right].
$$

**选择指数参数。** 为使上界尽可能小，最小化指数

$$
F(\lambda)=-n\lambda t+\frac{nv}{b^2}(e^{\lambda b}-1-\lambda b).
$$

其导数和二阶导数为

$$
F'(\lambda)=-nt+\frac{nv}{b}(e^{\lambda b}-1),\qquad
F''(\lambda)=nv e^{\lambda b}>0.
$$

故唯一最小点满足

$$
e^{\lambda_*b}-1=\frac{bt}{v},\qquad
\lambda_*=\frac{1}{b}\log\left(1+\frac{bt}{v}\right)>0.
$$

记 $u=bt/v$。代入后，$\lambda_*b=\log(1+u)$、$e^{\lambda_*b}-1=u$，并且 $nt/b=(nv/b^2)u$。所以

$$
\begin{aligned}
F(\lambda_*)
&=\frac{nv}{b^2}\{-u\log(1+u)+u-\log(1+u)\}\\
&=-\frac{nv}{b^2}\{(1+u)\log(1+u)-u\}\\
&=-\frac{nv}{b^2}h(u).
\end{aligned}
$$

由此得到

$$
\boxed{
\mathbb P\left(\frac1n\sum_{i=1}^nX_i\ge t\right)
\le\exp\left[-\frac{nv}{b^2}h\left(\frac{bt}{v}\right)\right].
}
$$

**下尾界。** 对变量 $-X_1,\ldots,-X_n$ 使用同一结果。它们仍独立、均值为零、绝对值不超过 $b$，二阶矩不变。因此

$$
\boxed{
\mathbb P\left(\frac1n\sum_{i=1}^nX_i\le-t\right)
\le\exp\left[-\frac{nv}{b^2}h\left(\frac{bt}{v}\right)\right].
}
$$

### 第1题c 函数下界与双侧Bernstein界

**第一步，证明函数下界。** 定义

$$
r(u)=\frac{u^2}{2(1+u/3)}=\frac{3u^2}{2(u+3)},\qquad
D(u)=h(u)-r(u).
$$

要证明 $D(u)\ge0$。直接比较函数不容易，但两者在零点处的值和一阶导数都相同，所以可以比较二阶导数。

对 $h$，

$$
h(0)=0,\qquad h'(u)=\log(1+u),\qquad h''(u)=\frac1{1+u}.
$$

为方便求导，先写

$$
r(u)=\frac32\left(u-3+\frac9{u+3}\right).
$$

因此

$$
r(0)=0,\qquad
r'(u)=\frac32\left(1-\frac9{(u+3)^2}\right),\qquad
r''(u)=\frac{27}{(u+3)^3}.
$$

于是 $D(0)=D'(0)=0$，并且对 $u\ge0$，

$$
\begin{aligned}
D''(u)
&=\frac1{1+u}-\frac{27}{(u+3)^3}\\
&=\frac{(u+3)^3-27(1+u)}{(1+u)(u+3)^3}\\
&=\frac{u^3+9u^2}{(1+u)(u+3)^3}\\
&=\frac{u^2(u+9)}{(1+u)(u+3)^3}\ge0.
\end{aligned}
$$

由微积分基本定理，

$$
D'(u)=D'(0)+\int_0^uD''(s)\,ds\ge0,
$$

$$
D(u)=D(0)+\int_0^uD'(s)\,ds\ge0.
$$

所以

$$
\boxed{h(u)\ge\frac{u^2}{2(1+u/3)},\qquad u\ge0.}
$$

**第二步，代入 Bennett 上界。** 因为指数前有负号，对 $h$ 使用下界会得到概率的较宽上界：

$$
\begin{aligned}
\frac{nv}{b^2}h\left(\frac{bt}{v}\right)
&\ge\frac{nv}{b^2}
\frac{b^2t^2/v^2}{2(1+bt/(3v))}\\
&=\frac{nt^2}{2(v+bt/3)}.
\end{aligned}
$$

因此上尾和下尾分别不超过

$$
\exp\left[-\frac{nt^2}{2(v+bt/3)}\right].
$$

最后用并集界合并两尾：

$$
\boxed{
\mathbb P\left(\left|\frac1n\sum_{i=1}^nX_i\right|\ge t\right)
\le2\exp\left[-\frac{nt^2}{2(v+bt/3)}\right].
}
$$

**本题的作答主线。** 有界性控制高阶矩；均值为零去掉线性项；独立性使样本和的 MGF 分解成乘积；优化指数参数得到 Bennett；比较二阶导数并合并两尾得到 Bernstein。

## 第2题 Bernstein尾界与期望误差

### 第2题a 对非负变量的尾概率积分

**题设与目标。** 设 $Z\ge0$，对所有 $t\ge0$ 有

$$
\mathbb P(Z\ge t)
\le C\exp\left[-\frac{t^2}{2(\nu^2+bt)}\right],\qquad
C\ge1,\quad \nu,b>0.
$$

要求证明

$$
\mathbb EZ
\le2\nu(\sqrt\pi+\sqrt{\log C})+4b(1+\log C).
$$

 小阈值处右侧可能超过 1，需要先与 1 取最小值；随后把 Bernstein 指数拆成高斯型尾与指数型尾，再分别积分。

因为

$$
\nu^2+bt\le2\max\{\nu^2,bt\},
$$

对 $t>0$ 有

$$
\frac{t^2}{2(\nu^2+bt)}
\ge\min\left\{\frac{t^2}{4\nu^2},\frac{t}{4b}\right\}.
$$

故

$$
\begin{aligned}
\exp\left[-\frac{t^2}{2(\nu^2+bt)}\right]
&\le\exp\left[-\min\left\{\frac{t^2}{4\nu^2},\frac{t}{4b}\right\}\right]\\
&=\max\{e^{-t^2/(4\nu^2)},e^{-t/(4b)}\}\\
&\le e^{-t^2/(4\nu^2)}+e^{-t/(4b)}.
\end{aligned}
$$

由于概率至多为 1，且对 $a,c\ge0$ 有 $\min\{1,a+c\}\le\min\{1,a\}+\min\{1,c\}$，得到

$$
\mathbb P(Z\ge t)
\le\min\{1,Ce^{-t^2/(4\nu^2)}\}
+\min\{1,Ce^{-t/(4b)}\}.
$$

使用尾积分公式，令

$$
I_G=\int_0^\infty\min\{1,Ce^{-t^2/(4\nu^2)}\}\,dt,
\qquad
I_E=\int_0^\infty\min\{1,Ce^{-t/(4b)}\}\,dt.
$$

则 $\mathbb EZ\le I_G+I_E$。

**计算高斯型积分。** 令 $a=2\nu\sqrt{\log C}\ge0$，使 $Ce^{-a^2/(4\nu^2)}=1$。从 $a$ 处分段，并令 $t=a+s$：

$$
\begin{aligned}
I_G
&\le a+\int_a^\infty Ce^{-t^2/(4\nu^2)}\,dt\\
&=a+\int_0^\infty
\exp\left[-\frac{2as+s^2}{4\nu^2}\right]ds\\
&\le a+\int_0^\infty e^{-s^2/(4\nu^2)}\,ds\\
&=2\nu\sqrt{\log C}+\nu\sqrt\pi.
\end{aligned}
$$

最后一步令 $s=2\nu z$，并使用 $\int_0^\infty e^{-z^2}\,dz=\sqrt\pi/2$。$C=1$ 时 $a=0$，同样成立。

**计算指数型积分。** 令 $a_E=4b\log C$，则

$$
\begin{aligned}
I_E
&=a_E+\int_{a_E}^\infty Ce^{-t/(4b)}\,dt\\
&=4b\log C+4bC e^{-a_E/(4b)}\\
&=4b(1+\log C).
\end{aligned}
$$

合并可得一个略强的界：

$$
\mathbb EZ
\le2\nu\sqrt{\log C}+\nu\sqrt\pi+4b(1+\log C).
$$

将 $\nu\sqrt\pi$ 放宽为 $2\nu\sqrt\pi$，即得题目要求：

$$
\boxed{
\mathbb EZ
\le2\nu(\sqrt\pi+\sqrt{\log C})+4b(1+\log C).
}
$$

**容易漏掉的步骤。** 不能把 $C$ 当成小于 1 的概率；先取 $\min\{1,\cdot\}$ 才能在积分中获得 $\sqrt{\log C}$ 和 $\log C$，而不是多余的线性因子 $C$。

### 第2题b Bernstein矩条件下的样本均值绝对误差

**题设。** $X_1,\ldots,X_n$ 独立同分布，$\mathbb EX_i=0$，$\mathbb EX_1^2=\sigma^2$，并对整数 $k\ge2$ 满足

$$
|\mathbb EX_1^k|\le\frac{k!}{2}\sigma^2b^{k-2}.
$$

令 $\bar X_n=n^{-1}\sum_iX_i$。要求证明

$$
\mathbb E|\bar X_n|
\le\frac{2\sigma}{\sqrt n}(\sqrt\pi+\sqrt{\log2})
+\frac{4b}{n}(1+\log2).
$$

若 $\sigma^2=0$，则 $X_i=0$ 几乎必然，结论直接成立。以下假设 $\sigma^2>0$。

**第一步，确认指数级数可以取期望。** 本题给的是 $|\mathbb EX^k|$，不是 $\mathbb E|X|^k$，因此先用偶数阶矩说明指数可积。对 $0\le a<1/b$，由非负级数的单调收敛定理，

$$
\begin{aligned}
\mathbb E\cosh(aX_1)
&=1+\sum_{m=1}^\infty\frac{a^{2m}\mathbb EX_1^{2m}}{(2m)!}\\
&\le1+\frac{\sigma^2a^2}{2}
\sum_{m=1}^\infty(ab)^{2m-2}\\
&=1+\frac{\sigma^2a^2}{2(1-a^2b^2)}<\infty.
\end{aligned}
$$

又因为 $e^{a|x|}\le e^{ax}+e^{-ax}=2\cosh(ax)$，故 $\mathbb Ee^{a|X_1|}<\infty$。因此当 $|\lambda|<1/b$ 时，指数级数可与期望交换。

**第二步，由矩条件控制 MGF。** 由均值为零与题设，

$$
\begin{aligned}
\mathbb Ee^{\lambda X_1}
&=1+\sum_{k=2}^\infty\frac{\lambda^k\mathbb EX_1^k}{k!}\\
&\le1+\sum_{k=2}^\infty\frac{|\lambda|^k|\mathbb EX_1^k|}{k!}\\
&\le1+\frac{\sigma^2\lambda^2}{2}
\sum_{k=2}^\infty(b|\lambda|)^{k-2}\\
&=1+\frac{\sigma^2\lambda^2}{2(1-b|\lambda|)}.
\end{aligned}
$$

使用 $\log(1+a)\le a$：

$$
\boxed{
\log\mathbb Ee^{\lambda X_1}
\le\frac{\sigma^2\lambda^2}{2(1-b|\lambda|)},\qquad |\lambda|<1/b.
}
$$

**第三步，推出样本均值的双侧尾界。** 对 $t>0$、$0<\lambda<1/b$，独立性与指数 Markov 给出

$$
\mathbb P(\bar X_n\ge t)
\le\exp\left[-n\lambda t+
\frac{n\sigma^2\lambda^2}{2(1-b\lambda)}\right].
$$

取

$$
\lambda=\frac{t}{\sigma^2+bt}.
$$

它满足 $0<b\lambda<1$，且

$$
1-b\lambda=\frac{\sigma^2}{\sigma^2+bt}.
$$

逐项代入指数：

$$
\begin{aligned}
-n\lambda t+
\frac{n\sigma^2\lambda^2}{2(1-b\lambda)}
&=-\frac{nt^2}{\sigma^2+bt}
+\frac{nt^2}{2(\sigma^2+bt)}\\
&=-\frac{nt^2}{2(\sigma^2+bt)}.
\end{aligned}
$$

对 $-X_i$ 做同样计算，它也满足同一矩条件。由并集界，

$$
\mathbb P(|\bar X_n|\ge t)
\le2\exp\left[-\frac{nt^2}{2(\sigma^2+bt)}\right]
=2\exp\left[-\frac{t^2}{2(\sigma^2/n+(b/n)t)}\right].
$$

这里的 $bt$ 系数来自 Bernstein 矩条件，与第 1 题有界变量结论中的 $bt/3$ 不同，不能直接替换。

**第四步，应用 (a)。** 令 $Z=|\bar X_n|$，其尾界参数是

$$
C=2,\qquad\nu=\frac{\sigma}{\sqrt n},\qquad
b_{\rm tail}=\frac bn.
$$

代入 (a)，得到

$$
\boxed{
\mathbb E|\bar X_n|
\le\frac{2\sigma}{\sqrt n}(\sqrt\pi+\sqrt{\log2})
+\frac{4b}{n}(1+\log2).
}
$$

**本题的作答主线。** 矩条件控制 MGF，独立性与指数 Markov 推出双侧尾界，重新识别均值的尺度参数，最后对尾概率积分。

## 第3题 轻尾随机变量的最大值

### 题目条件与共同方法

$X_1,\ldots,X_n$ 均值为零，**本题不假设它们相互独立**。最大值的控制依靠逐点不等式与 MGF 的求和，而不是随机变量之和的 MGF 分解。

令 $M=\max_{1\le i\le n}X_i$。对 $\lambda>0$，逐点有

$$
e^{\lambda M}=\max_i e^{\lambda X_i}
\le\sum_{i=1}^n e^{\lambda X_i}.
$$

结合指数函数的 Jensen 不等式，

$$
e^{\lambda\mathbb EM}
\le\mathbb Ee^{\lambda M}
\le\sum_{i=1}^n\mathbb Ee^{\lambda X_i}.
$$

所以

$$
\boxed{
\mathbb EM\le\frac1\lambda
\log\left(\sum_{i=1}^n\mathbb Ee^{\lambda X_i}\right).
}
$$

在本题 MGF 条件下各 $X_i$ 可积，且 $|M|\le\sum_i|X_i|$，所以 $M$ 可积，Jensen 的使用合法。

### 第3题a 次高斯变量的最大值

**假设。** 对所有 $\lambda\in\mathbb R$，

$$
\mathbb Ee^{\lambda X_i}\le e^{\sigma^2\lambda^2/2},\qquad \sigma\ge0.
$$

代入共同方法，对任意 $\lambda>0$，

$$
\mathbb E\max_iX_i
\le\frac1\lambda\log\left(ne^{\sigma^2\lambda^2/2}\right)
=\frac{\log n}{\lambda}+\frac{\sigma^2\lambda}{2}.
$$

当 $n\ge2$、$\sigma>0$ 时，取

$$
\lambda_* =\frac{\sqrt{2\log n}}{\sigma}.
$$

它也可由对上界求导得到：$-(\log n)/\lambda^2+\sigma^2/2=0$。此时两项相等，分别为 $\sigma\sqrt{\log n/2}$，所以

$$
\boxed{
\mathbb E\max_{1\le i\le n}X_i
\le\sigma\sqrt{2\log n}.
}
$$

**边界情况。** 若 $n=1$，左边为 $\mathbb EX_1=0$，右边也是零。若 $\sigma=0$，由上面的通用界 $\mathbb E\max_iX_i\le(\log n)/\lambda$，令 $\lambda\to\infty$ 即得左边不超过零，仍满足要求。

**结论含义。** 即使变量相关，最大值的期望也至多随变量数量按 $\sqrt{\log n}$ 增长；证明只使用每个变量自身的 MGF 界。

### 第3题b 次高斯变量绝对值的最大值

由于

$$
\max_{1\le i\le n}|X_i|
=\max\{X_1,\ldots,X_n,-X_1,\ldots,-X_n\},
$$

可把问题变成 $2n$ 个变量的最大值。每个 $-X_i$ 均值仍为零，且

$$
\mathbb Ee^{\lambda(-X_i)}
=\mathbb Ee^{-\lambda X_i}
\le e^{\sigma^2\lambda^2/2}.
$$

这 $2n$ 个变量高度相关也没有关系，因为 (a) 不要求独立。由 (a)，

$$
\mathbb E\max_i|X_i|\le\sigma\sqrt{2\log(2n)}.
$$

当 $n\ge2$ 时，$2n\le n^2$，故

$$
\log(2n)\le2\log n.
$$

于是

$$
\sigma\sqrt{2\log(2n)}
\le\sigma\sqrt{4\log n}
=2\sigma\sqrt{\log n}.
$$

最终

$$
\boxed{
\mathbb E\max_{1\le i\le n}|X_i|
\le\sigma\sqrt{2\log(2n)}
\le2\sigma\sqrt{\log n},\qquad n\ge2.
}
$$

### 第3题c 局部MGF条件下的最大值

**新假设。** 存在共同参数 $\nu,b>0$，使

$$
\mathbb Ee^{\lambda X_i}\le e^{\nu^2\lambda^2/2},
\qquad |\lambda|<1/b.
$$

这次只能在局部区间内使用 MGF 界，因此选择参数时必须满足 $\lambda<1/b$，不能不加检查地沿用 (a) 的最小点。

由共同方法，对 $0<\lambda<1/b$，

$$
\mathbb E\max_iX_i
\le\frac{\log n}{\lambda}+\frac{\nu^2\lambda}{2}.
$$

令 $L=\log n>0$，选择

$$
\lambda=\frac{1}{b+\nu/\sqrt{2L}}.
$$

分母严格大于 $b$，故 $0<\lambda<1/b$。逐项估计：

$$
\frac L\lambda=bL+\nu\sqrt{\frac L2},
$$

$$
\frac{\nu^2\lambda}{2}
=\frac{\nu^2}{2(b+\nu/\sqrt{2L})}
\le\frac{\nu^2}{2(\nu/\sqrt{2L})}
=\nu\sqrt{\frac L2}.
$$

两项相加，得到

$$
\boxed{
\mathbb E\max_{1\le i\le n}X_i
\le\nu\sqrt{2\log n}+b\log n,\qquad n\ge2.
}
$$

对绝对值最大值，仍考虑 $2n$ 个变量 $X_i,-X_i$。局部 MGF 条件关于 $\lambda$ 对称，故这些变量满足相同条件。用 $L=\log(2n)$ 做上述计算，得到

$$
\boxed{
\mathbb E\max_{1\le i\le n}|X_i|
\le\nu\sqrt{2\log(2n)}+b\log(2n).
}
$$

**本题的作答主线。** 指数化把最大值变成最大指数，最大指数不超过指数之和；Jensen 将其转成期望最大值的上界；随后选择一个满足 MGF 定义域的指数参数。全程没有用独立性。

## 第4题 对称Pareto变量与截断

### 题目条件

设 $X_1,\ldots,X_n$ 独立同分布，密度为

$$
f_\alpha(x)=\frac\alpha2|x|^{-\alpha-1}\mathbf1_{\{|x|\ge1\}},\qquad\alpha>2.
$$

本题的尾概率按幂次衰减。先把变量截断为有界变量，使用第 1 题的 Bernstein 界，再单独控制截断发生的概率。

### 第4题a 尾概率与前两阶矩

**尾概率。** 密度关于零对称。对 $u\ge1$，

$$
\begin{aligned}
\mathbb P(|X_i|>u)
&=2\int_u^\infty\frac\alpha2x^{-\alpha-1}\,dx\\
&=\alpha\left[-\frac{x^{-\alpha}}\alpha\right]_u^\infty\\
&=\boxed{u^{-\alpha}}.
\end{aligned}
$$

**均值。** 先确认绝对可积：

$$
\mathbb E|X_i|
=\alpha\int_1^\infty x^{-\alpha}\,dx
=\frac\alpha{\alpha-1}<\infty.
$$

由于 $xf_\alpha(x)$ 是奇函数，正负两侧积分相消，故

$$
\boxed{\mathbb EX_i=0.}
$$

**二阶矩。** 由对称性，

$$
\begin{aligned}
\mathbb EX_i^2
&=2\int_1^\infty x^2\frac\alpha2x^{-\alpha-1}\,dx\\
&=\alpha\int_1^\infty x^{1-\alpha}\,dx\\
&=\alpha\left[\frac{x^{2-\alpha}}{2-\alpha}\right]_1^\infty\\
&=\boxed{\frac\alpha{\alpha-2}=:\sigma_\alpha^2}.
\end{aligned}
$$

$\alpha>2$ 保证无穷远的 $x^{2-\alpha}$ 趋于零，二阶矩有限。

### 第4题b 截断后的均值与二阶矩

给定 $u\ge1$，定义

$$
Y_i=X_i\mathbf1_{\{|X_i|\le u\}}.
$$

**均值为零。** 截断区间关于零对称，因而

$$
\mathbb EY_i=\int_{-u}^u xf_\alpha(x)\,dx=0.
$$

**有界性。** 若 $|X_i|\le u$，则 $Y_i=X_i$；若 $|X_i|>u$，则 $Y_i=0$。所以

$$
\boxed{|Y_i|\le u\quad\text{几乎必然}.}
$$

**二阶矩。** 逐点有 $Y_i^2=X_i^2\mathbf1_{\{|X_i|\le u\}}\le X_i^2$，所以

$$
\boxed{\mathbb EY_i^2\le\mathbb EX_i^2=\sigma_\alpha^2.}
$$

也可直接算出

$$
\mathbb EY_i^2
=\alpha\int_1^ux^{1-\alpha}\,dx
=\sigma_\alpha^2(1-u^{2-\alpha})\le\sigma_\alpha^2.
$$

各 $Y_i$ 分别是独立变量 $X_i$ 的确定函数，因此仍相互独立。这保证下一问能使用独立变量的 Bernstein 界。

### 第4题c 样本和的截断集中界

记

$$
S_X=\sum_{i=1}^nX_i,\qquad S_Y=\sum_{i=1}^nY_i,
\qquad A_u=\{\max_i|X_i|\le u\}.
$$

在事件 $A_u$ 上，每个样本都没有被截断，因此 $S_X=S_Y$。于是事件包含关系为

$$
\{|S_X|\ge t\}
\subseteq\{|S_Y|\ge t\}\cup A_u^c.
$$

由并集界，

$$
\mathbb P(|S_X|\ge t)
\le\mathbb P(|S_Y|\ge t)+\mathbb P(A_u^c).
$$

**控制截断概率。** 利用 (a)，

$$
\begin{aligned}
\mathbb P(A_u^c)
&=\mathbb P\left(\bigcup_{i=1}^n\{|X_i|>u\}\right)\\
&\le\sum_{i=1}^n\mathbb P(|X_i|>u)\\
&=nu^{-\alpha}.
\end{aligned}
$$

**控制截断变量之和。** 第 1 题的 Bernstein 结论改写为和的形式是

$$
\mathbb P(|S_Y|\ge t)
\le2\exp\left[-\frac{t^2}{2(\sum_i\mathbb EY_i^2+ut/3)}\right].
$$

这里使用 $|Y_i|\le u$，即第 1 题中的有界参数换为 $u$；并将均值阈值取成 $t/n$。由 $\sum_i\mathbb EY_i^2\le n\sigma_\alpha^2$，

$$
\mathbb P(|S_Y|\ge t)
\le2\exp\left[-\frac{t^2}{2(n\sigma_\alpha^2+ut/3)}\right].
$$

增大分母会使负指数变大，得到的是较宽的概率上界。$u=1$ 时 $Y_i=0$ 几乎必然，此界也直接成立，无需对零方差使用第 1 题的 $v>0$ 形式。

最终

$$
\boxed{
\mathbb P\left(\left|\sum_{i=1}^nX_i\right|\ge t\right)
\le2\exp\left[-\frac{t^2}{2(n\sigma_\alpha^2+ut/3)}\right]
+nu^{-\alpha},\qquad t>0,\ u\ge1.
}
$$

第一项控制没有极端样本时的集中误差，第二项控制出现极端样本而导致截断的概率。

### 第4题d 显式选择阈值与误差水平

固定 $\delta\in(0,1)$。要求把两个概率上界分别控制到 $\delta/2$，并求出对应的 $u$ 与 $\varepsilon_{n,\delta}$。

**第一步，选择最小截断阈值。** 解

$$
nu^{-\alpha}\le\frac\delta2
\quad\Longleftrightarrow\quad
u^\alpha\ge\frac{2n}\delta
\quad\Longleftrightarrow\quad
u\ge\left(\frac{2n}\delta\right)^{1/\alpha}.
$$

由于 $n\ge1$、$\delta<1$，右侧大于 1。因此满足要求的最小阈值为

$$
\boxed{u_{n,\delta}=\left(\frac{2n}\delta\right)^{1/\alpha}.}
$$

它使 $nu_{n,\delta}^{-\alpha}=\delta/2$。

**第二步，令指数项等于剩下的 $\delta/2$。** 将 $t=n\varepsilon$ 代入 (c)，要求

$$
2\exp\left[-\frac{n^2\varepsilon^2}
{2(n\sigma_\alpha^2+un\varepsilon/3)}\right]
=\frac\delta2.
$$

先除以 2，再取对数。记

$$
L_\delta=\log\frac4\delta>0.
$$

方程等价于

$$
\frac{n\varepsilon^2}{2(\sigma_\alpha^2+u\varepsilon/3)}=L_\delta,
$$

即

$$
n\varepsilon^2-\frac{2L_\delta u}{3}\varepsilon
-2L_\delta\sigma_\alpha^2=0.
$$

由二次方程求根公式，其唯一正根为

$$
\begin{aligned}
\varepsilon_{n,\delta}
&=\frac{2L_\delta u/3+
\sqrt{(2L_\delta u/3)^2+8nL_\delta\sigma_\alpha^2}}{2n}\\
&=\frac{L_\delta u}{3n}
+\sqrt{\frac{2L_\delta\sigma_\alpha^2}{n}
+\frac{L_\delta^2u^2}{9n^2}}.
\end{aligned}
$$

取 $u=u_{n,\delta}$，显式答案为

$$
\boxed{
\varepsilon_{n,\delta}
=\frac{\log(4/\delta)}{3n}
\left(\frac{2n}{\delta}\right)^{1/\alpha}
+\sqrt{
\frac{2\sigma_\alpha^2\log(4/\delta)}{n}
+\frac{\log^2(4/\delta)}{9n^2}
\left(\frac{2n}{\delta}\right)^{2/\alpha}
},\quad
\sigma_\alpha^2=\frac\alpha{\alpha-2}.
}
$$

**第三步，合并概率。** 该选择使 (c) 的两项分别等于 $\delta/2$，故

$$
\boxed{
\mathbb P\left(\left|\sum_{i=1}^nX_i\right|
\ge n\varepsilon_{n,\delta}\right)\le\delta.
}
$$

等价地，样本均值以至少 $1-\delta$ 的概率满足 $|\bar X_n|<\varepsilon_{n,\delta}$。

**第四步，固定置信水平时的渐近主项。** 固定 $\alpha>2$ 与 $\delta$。此时 $L_\delta$ 为常数，且

$$
\frac{u_{n,\delta}}n
=\left(\frac2\delta\right)^{1/\alpha}n^{1/\alpha-1}
=o(n^{-1/2}),
$$

因为 $1/\alpha<1/2$。根号内的第二项也小于第一项的量级：

$$
\frac{u_{n,\delta}^2}{n^2}
=O(n^{2/\alpha-2})=o(n^{-1}).
$$

因此

$$
\boxed{
\varepsilon_{n,\delta}
\sim\sqrt{2\sigma_\alpha^2\log(4/\delta)}\,n^{-1/2}
=\sqrt{\frac{2\alpha}{\alpha-2}\log\frac4\delta}\,n^{-1/2}.
}
$$

其中 $\sim$ 表示两边之比趋于 1，故主导阶为 $\Theta(n^{-1/2})$。这是固定 $\delta$ 的结论；上述比较不能直接当成任意随 $n$ 变化的置信水平下的统一主项。

**本题的作答主线。** 对称截断保留零均值；截断后使用有界变量的 Bernstein 界；并集界补偿截断失败；先分配失败概率，再解阈值不等式与误差的二次方程。

## 第5题 欧氏球的覆盖数

### 定义与几何工具

设整数 $d\ge1$，定义单位闭球与单位球面

$$
B_2^d=\{x\in\mathbb R^d:\|x\|_2\le1\},\qquad
S^{d-1}=\{x\in\mathbb R^d:\|x\|_2=1\}.
$$

对 $K\subseteq\mathbb R^d$、$\varepsilon>0$，若有限集合 $\mathcal N\subseteq K$ 满足：每个 $x\in K$ 都能找到 $z\in\mathcal N$，使 $\|x-z\|_2\le\varepsilon$，则称 $\mathcal N$ 为 $K$ 的 $\varepsilon$-网。$N(K,\varepsilon)$ 是这类网的最小点数。**网点必须位于 $K$ 内部或其边界。**

记 $V_d=\operatorname{vol}_d(B_2^d)>0$。半径为 $r$ 的欧氏球体积为 $V_dr^d$：平移不改变体积，缩放 $x=rz$ 的 Jacobian 为 $r^d$。开球和闭球边界的 $d$ 维体积为零，因此体积相同。

本题的两种思路要区分：覆盖下界使用“球的体积不能超过覆盖小球的体积之和”；覆盖上界先选分离的点，利用不相交小球限制点数，再证明这些点形成网。

### 第5题a 单位球覆盖数的下界

任取一个 $\varepsilon$-网 $\mathcal N=\{x_1,\ldots,x_M\}\subseteq B_2^d$。由网的定义，以各网点为中心、半径 $\varepsilon$ 的闭球覆盖单位球：

$$
B_2^d\subseteq\bigcup_{j=1}^M\{x:\|x-x_j\|_2\le\varepsilon\}.
$$

对两侧取体积，使用有限并集的体积次可加性：

$$
V_d\le\sum_{j=1}^MV_d\varepsilon^d
=MV_d\varepsilon^d.
$$

因为 $V_d>0$，可约去得到 $M\ge\varepsilon^{-d}$。该结论对任何 $\varepsilon$-网都成立，故对最小点数也成立：

$$
\boxed{N(B_2^d,\varepsilon)\ge\varepsilon^{-d}.}
$$

更精确地可写成整数下界 $N(B_2^d,\varepsilon)\ge\lceil\varepsilon^{-d}\rceil$。即使覆盖小球互相重叠，体积次可加性仍成立，所以这里没有要求网点彼此分离。

### 第5题b 单位球覆盖数的上界

**第一步，考察分离点集。** 在 $B_2^d$ 中取点 $x_1,\ldots,x_M$，使任意不同两点满足

$$
\|x_i-x_j\|_2>\varepsilon.
$$

以这些点为中心画半径 $\varepsilon/2$ 的开球

$$
D_j=\{x:\|x-x_j\|_2<\varepsilon/2\}.
$$

这些小球两两不相交。否则若 $x\in D_i\cap D_j$，三角不等式给出

$$
\|x_i-x_j\|_2
\le\|x_i-x\|_2+\|x-x_j\|_2
<\varepsilon,
$$

与分离条件矛盾。

**第二步，用体积限制分离点数。** 对 $x\in D_j$，因 $\|x_j\|_2\le1$，

$$
\|x\|_2\le\|x-x_j\|_2+\|x_j\|_2<\varepsilon/2+1.
$$

所以所有小球都包含在以原点为中心、半径 $1+\varepsilon/2$ 的大球内。由两两不相交，

$$
M V_d(\varepsilon/2)^d
=\operatorname{vol}_d\left(\bigcup_{j=1}^MD_j\right)
\le V_d(1+\varepsilon/2)^d.
$$

约去 $V_d$ 后，

$$
M\le\left(\frac{1+\varepsilon/2}{\varepsilon/2}\right)^d
=\left(1+\frac2\varepsilon\right)^d.
$$

**第三步，构造一个不能继续加点的分离集。** 从单位球内任取一点开始。只要存在一个点与所有已选点距离都大于 $\varepsilon$，就把它加入。

每次加入后仍满足分离条件，而上述体积界限制了所有有限分离集的点数。所以该过程不可能无限继续，必在有限步后停止，得到一个包含关系意义下的极大分离集 $\mathcal N$。这里“极大”只表示不能再加入新点，不要求它在所有分离集中点数最多。

**第四步，证明它是网。** 若某个 $x\in B_2^d$ 与所有 $z\in\mathcal N$ 的距离都大于 $\varepsilon$，就可以把 $x$ 加入分离集，与停止条件矛盾。因此对每个 $x\in B_2^d$，都存在 $z\in\mathcal N$ 满足

$$
\|x-z\|_2\le\varepsilon.
$$

故 $\mathcal N$ 是一个中心在单位球内的 $\varepsilon$-网，并且其点数至多为体积上界。于是

$$
\boxed{
N(B_2^d,\varepsilon)
\le\left(1+\frac2\varepsilon\right)^d,\qquad\varepsilon>0.
}
$$

**为什么小球半径取一半？** 点间距离大于 $\varepsilon$，恰好保证半径 $\varepsilon/2$ 的小球不相交；这个半径使分离条件能够用于体积计算。

### 第5题c 球面覆盖数与指定半径

这次要求网点都在 $S^{d-1}$ 上，因此直接在球面上运行上一问的选点过程：只加入属于球面的新点，使任意两点距离大于 $\varepsilon$。

每个中心的范数恰为 1。以这些中心画半径 $\varepsilon/2$ 的开球，仍两两不相交，并且都位于半径 $1+\varepsilon/2$ 的环境欧氏球中。因此同样的体积比较给出

$$
M\le\left(1+\frac2\varepsilon\right)^d.
$$

这保证选点过程有限终止。终止时，对每个球面点，都存在一个已选球面点与它距离不超过 $\varepsilon$，否则可以继续加入。所以得到一个中心在球面上的 $\varepsilon$-网，故

$$
\boxed{
N(S^{d-1},\varepsilon)
\le\left(1+\frac2\varepsilon\right)^d.
}
$$

这里比较的是球面网点周围的 **$d$ 维小球体积**，没有使用球面本身的 $d$ 维体积；球面本身的该体积为零，不能直接用于同样的下界比较。

当 $\varepsilon=1/4$ 时，

$$
1+\frac2\varepsilon=1+8=9.
$$

因此

$$
\boxed{N(S^{d-1},1/4)\le9^d.}
$$

这也同时给出 $N(B_2^d,1/4)\le9^d$，但本问要求的是中心位于球面的网。

**本题的作答主线。** 下界比较覆盖体积；上界通过分离点集产生不相交小球，再利用“不能继续加点”推出覆盖。球面上界沿用环境空间的体积，并始终在球面选中心。

## 条件与边界核对

| 场景 | 必须明确的步骤 | 常见错误 |
|---|---|---|
| 第 1 题的 MGF 与尾界 | 均值为零、独立性、参数最小化、两尾合并 | 忘记指数参数的 $n$；把负号下的函数界方向用反 |
| 第 2 题的尾积分 | 概率与 1 取最小；确定两段的截断位置 | 把前因子 $C$ 直接积分；把 $b/n$ 写成 $b/\sqrt n$ |
| 第 2 题的矩条件 | 先确保指数可积，再使用级数与矩界 | 把 $\lvert\mathbb EX^k\rvert$ 擅自替换成 $\mathbb E\lvert X\rvert^k$ 的相同上界 |
| 第 3 题的最大值 | 指数化最大值不超过指数之和 | 为最大值错误地使用独立性乘积；忽略局部 MGF 定义域 |
| 第 4 题的截断 | 原样本与截断样本只在无截断事件上相同 | 对无界 Pareto 变量直接使用有界 Bernstein；漏掉截断概率 |
| 第 4 题的置信阈值 | 将两个概率各分配 $\delta/2$，保留指数前的 2 | 将 $\log(4/\delta)$ 错写成 $\log(2/\delta)$；漏掉正根 |
| 第 5 题的上界 | 分离、半径减半、体积界、极大性推出覆盖 | 把分离集本身未经证明就当成网；没有保证球面中心仍在球面 |

## 第2题所用高斯积分的补充证明

令 $J=\int_{-\infty}^{\infty}e^{-x^2}\,dx$。先由 $e^{-x^2}\le e^{-|x|}$（$|x|\ge1$）知 $J<\infty$。由非负积分的乘积与平面极坐标变换，

$$
\begin{aligned}
J^2
&=\int_{\mathbb R^2}e^{-(x^2+y^2)}\,dx\,dy\\
&=\int_0^{2\pi}\int_0^\infty e^{-r^2}r\,dr\,d\varphi\\
&=2\pi\left[-\frac12e^{-r^2}\right]_0^\infty\\
&=\pi.
\end{aligned}
$$

因 $J>0$，有 $J=\sqrt\pi$。被积函数为偶函数，所以

$$
\int_0^\infty e^{-x^2}\,dx=\frac{\sqrt\pi}{2}.
$$

## 题目出处

作业题面注明：第 1 至 3 题分别改编自 M. J. Wainwright 的 *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*，习题 2.7、2.8、2.12；第 5 题改编自 R. Vershynin 的 *High-Dimensional Probability: An Introduction with Applications in Data Science*，推论 4.2.13；第 4 题为补充的截断练习。
