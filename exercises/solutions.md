# Worked solutions

[Problems](problems.md) · [Home](../README.md)

## E01

Write $Y-a=(Y-m(X))+(m(X)-a)$. Conditional on $X$, the cross term has expectation zero because $\mathbb E[Y-m(X)\mid X]=0$. Hence the conditional MSE is $\operatorname{Var}(Y\mid X)+(m(X)-a)^2$. The first term is fixed and the second is minimized by $a=m(X)$. Divide by two if using half-squared loss.

## E02

For any comparator $f$, cross from $L(\widetilde f)$ to its empirical value using one uniform-deviation term. Apply approximate empirical optimality, compare to $\widehat L_n(f)$, and cross back using a second deviation term. This gives $L(\widetilde f)\le L(f)+\epsilon_{\rm opt}+2D_n$. Taking an infimum over comparators does not require that it is attained. Subtracting $L^\star$ proves the bound.

## E03

The gradient is $2X^\top(X\theta-y)+2\lambda\theta$. For $\lambda>0$, the solution is $(X^\top X+\lambda I)^{-1}X^\top y$. Under the normalized objective $\|y-X\theta\|^2/(2n)+\lambda\|\theta\|^2/2$, multiply the first-order equation by $n$ to obtain $n\lambda I$. The numerical values of the two penalty parameters are not directly comparable without rescaling.

## E04

The orthogonal residual component is constant. Each remaining component has squared multiplier $(n\lambda/(s_j^2+n\lambda))^2$, which is nondecreasing for $\lambda\ge0$ and $s_j>0$. Test risk also depends on shrinkage bias and reduced noise amplification; those move in opposite directions. Training monotonicity therefore does not imply test monotonicity.

## E05

All interpolators are $(1+c,1-c)$. Their squared norm is $2+2c^2$, minimized at $c=0$, giving $(1,1)$. Initialization $(3,-1)=(1,1)+(2,-2)$ already interpolates and its gradient is zero, so GD remains there. The null-space component survives. Starting at zero would select $(1,1)$ for a stable positive step on the nonzero eigenvalue.

## E06

The threshold is $\tau_d^2/(n-d-1)=8/8=1$. A next feature with $b_4^2>1$ decreases the finite expected test MSE; equality keeps it unchanged. With $b_4=0$, the old risk is $8\cdot11/8=11$, while the next is $8\cdot11/7=88/7>11$. Both dimensions satisfy the finite-expectation conditions.

## E07

The gradient is $5(x-2)$. Thus $e_{t+1}=(1-5\eta)e_t$, $x_T=2+(1-5\eta)^T(x_0-2)$, and the stable interval from every initialization is $0<\eta<2/5$. The objective gap is $(1-5\eta)^{2T}$ times its initial value. At $\eta=1/5$ one update reaches the optimizer.

## E08

At $\eta=1/8$, the multipliers are $3/4$ and zero, so $q=0.75$. At $\eta=0.2$, they are $0.6$ and $-0.6$, so $q=0.6$. The optimized objective upper bound uses $0.6^{2T}$. Oscillation in one coordinate is compatible with shrinking error magnitude.

## E09

$-x^2$ has gradient Lipschitz constant two but fails convexity. $x^4$ is convex with a unique minimizer at zero, but strong convexity at zero would require $x^4\ge\mu x^2/2$ for every small $x$, impossible for any fixed $\mu>0$. A unique minimizer is weaker than a uniform quadratic lower bound.

## E10

Let $g=\nabla\Phi(\theta)$ and complete the square: $g^\top d+\mu\|d\|^2/2=\mu\|d+g/\mu\|^2/2-\|g\|^2/(2\mu)$. The global quadratic lower bound implies $\Phi(\theta^\star)\ge\Phi(\theta)-\|g\|^2/(2\mu)$. Rearrangement gives the claim. The minimizer of the lower model is not claimed to be the minimizer of the actual objective.

## E11

Distance expansion yields $g_t^\top e_t=(\|e_t\|^2-\|e_{t+1}\|^2)/(2\eta)+\eta\|g_t\|^2/2$. Convexity gives $\Delta_t\le g_t^\top e_t$, and descent gives $\eta\|g_t\|^2/2\le\Phi(\theta_t)-\Phi(\theta_{t+1})$. Canceling $\Phi(\theta_t)$ leaves $\Delta_{t+1}\le(\|e_t\|^2-\|e_{t+1}\|^2)/(2\eta)$. Sum to get $\sum_{t<T}\Delta_{t+1}\le\|e_0\|^2/(2\eta)$. **Monotonicity is used now:** each summand is at least $\Delta_T$, giving the result after division by $T$.

## E12

Expand $\|g_t\|^2=\|\nabla\Phi(\theta_t)\|^2+2\nabla\Phi(\theta_t)^\top\xi_t+\|\xi_t\|^2$. Conditional on $\mathcal H_t$, the current gradient is fixed, so the cross term has mean zero. Therefore the conditional second moment equals $\|\nabla\Phi(\theta_t)\|^2+\mathbb E[\|\xi_t\|^2\mid\mathcal H_t]$. No independence from past noise was invoked.

## E13

Set $r=1-\eta h$. Independence of the fresh noise gives $u_{t+1}=r^2u_t+\eta^2s^2$. If $|r|<1$, its stationary value is $u_\infty=\eta^2s^2/(1-r^2)=\eta s^2/[h(2-\eta h)]$. Multiply by $h/2$ to obtain $\eta s^2/[2(2-\eta h)]$. For finite time, add the transient $r^{2t}(u_0-u_\infty)$.

## E14

At $t=1$, $A/\tau=u_1+C/\tau\ge u_1$. Assume the induction statement. With $s=t+\tau$, the next bound is $A(s-2)/[s(s-1)]+C/s^2$. Its first term is short of $A/s$ by $A/[s(s-1)]$, which is at least $C/s^2$ because $A\ge C$ and $s>1$. This proves the next step.

## E15

Conditional distance expansion and convexity give $2\eta_t\mathbb E\Delta_t\le d_t-d_{t+1}+\eta_t^2G^2$. Sum, discard the nonnegative final distance, and apply convexity pointwise to the weighted average before expectation. The bound is $(R^2+G^2Q_T)/(2S_T)$. For fixed $\eta$ it becomes $R^2/(2\eta T)+\eta G^2/2$. Its derivative vanishes at $\eta=R/(G\sqrt T)$ when $R,G>0$, yielding $RG/\sqrt T$. The output in this claim is the average.

## E16

$V=a^2s^2/(2ah-1)$ requires $ah>1/2$. Its derivative is $2as^2(ah-1)/(2ah-1)^2$. It is negative below $1/h$ and positive above, so the minimizer is $a=1/h$ and $V=s^2/h^2$. The variance diverges as $ah\downarrow1/2$ with $s^2>0$.

## E17

$\eta=4/(3+1)^2=1/4$ and $\rho=((3-1)/(3+1))^2=1/4$. For $\lambda=1$, the polynomial is $r^2-r+1/4=(r-1/2)^2$. For $\lambda=9$, it is $r^2+r+1/4=(r+1/2)^2$. Repeated-root solutions have the form $(c_1+c_2t)r^t$, so a polynomial prefactor can occur without changing the limiting root rate $1/2$.

## E18

Unrolling the moment recursions gives $m_t=(1-\beta_1^t)g$ and $v_t=(1-\beta_2^t)(4,16)$. Correction yields $\widehat m_t=(2,-4)$ and $\widehat v_t=(4,16)$. The first displacement is $-\alpha(2/(2+\epsilon),-4/(4+\epsilon))$. When gradient means vary, corrected moments remain weighted averages of past means, not necessarily the current mean. Initialization correction is not a universal current-gradient unbiasedness theorem.

## E19

Let $U\sim\mathrm{Unif}(0,1)$ and $X_n=\sqrt n\mathbf1\{U\le1/n\}$. For each $U>0$, the indicator is eventually zero, proving almost-sure convergence. But $\mathbb EX_n^2=n\cdot(1/n)=1$. Rare large values prevent mean-square convergence.

## E20

At positive $m$, $g'(m)=1/m$, so the delta-method limit is $N(0,s^2/m^2)$. The log is defined with probability tending to one because $T_n\to m>0$ in probability. At zero for the square function, the derivative vanishes; the first-order scaled limit is degenerate. Instead $nT_n^2=(\sqrt nT_n)^2\Rightarrow s^2\chi_1^2$. State the changed scale and the nonnormal limit.
