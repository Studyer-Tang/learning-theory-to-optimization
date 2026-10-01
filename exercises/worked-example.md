# Worked example · Solve the smallest useful GD problem

[Home](../README.md) · [Try the problem set](problems.md)

## Problem

Minimize $f(x)=3x^2/2$ by gradient descent with constant step $\eta>0$. Determine the stable step interval, classify the behavior at $\eta=1/6,1/3,1/2,2/3$, and find the exact objective gap after $T$ steps. Then extend the argument to $H=\operatorname{diag}(1,9)$.

## Step 1: identify the output and target

The output is the scalar iterate $x_T$. The optimizer is $x^\star=0$. Parameter error is $|x_T|$; objective gap is $3x_T^2/2$. These are not the same metric.

## Step 2: write the update, not a theorem number

Since $f'(x)=3x$,

$$x_{t+1}=x_t-3\eta x_t=(1-3\eta)x_t.$$

Repeated substitution gives $x_T=(1-3\eta)^Tx_0$. The problem is now a geometric sequence.

## Step 3: ask when the magnitude decreases

For every nonzero initialization, convergence requires $|1-3\eta|<1$. Solving both sides of $-1<1-3\eta<1$ gives $0<\eta<2/3$.

| Step | Multiplier | Behavior |
|---|---|---|
| $1/6$ | $1/2$ | Same sign, shrinking |
| $1/3$ | $0$ | Reaches the optimum in one step |
| $1/2$ | $-1/2$ | Alternates sign, shrinking |
| $2/3$ | $-1$ | Alternates sign, constant magnitude unless already zero |

## Step 4: translate parameter error into objective error

$$f(x_T)-f(0)=(1-3\eta)^{2T}[f(x_0)-f(0)].$$

The exponent doubles because the objective is quadratic. At $\eta=1/6$, each step halves the parameter error but quarters the objective gap.

## Step 5: transfer to two dimensions

For $f(\theta)=\frac12\theta^\top H\theta$, the two coordinates multiply by $1-\eta$ and $1-9\eta$. Both magnitudes must be less than one, giving $0<\eta<2/9$. The largest curvature supplies the restrictive upper bound.

At $\eta=1/9$, the high-curvature component disappears immediately but the low-curvature component only shrinks by $8/9$. Balancing the two magnitudes gives $1-\eta=9\eta-1$, hence $\eta=0.2$ and factors $0.8,-0.8$.

## Change one condition

Replace $H$ by $\operatorname{diag}(1,0)$. The second coordinate is unchanged, regardless of the positive step used for the first. The objective can converge to zero without the parameter converging to the designated origin. This is exactly why positive definiteness matters.

Before reading further chapters, reproduce these five steps with $f(x)=5(x-2)^2/2$. The derivative, optimizer, and multiplier change; the proof pattern does not.
