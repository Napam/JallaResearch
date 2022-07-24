### Plotting decision boundry of a 2D neuron
Model: $f(x,y) = w_0 + w_1x + w_2y$

Decision boundary is when $f(x,y) = 0$

Solve for $y$ with respect to $x$ at decision boundary:
$$
\begin{align}
    w_0 + w_1x + w_2y = 0 \\
    w_2y = -w_0 - w_1x \\
    y = \frac{-w_0 - w_1x}{w_2} \\
    y = - \frac{w_1}{w_2}x - \frac{w_0}{w_2}
\end{align}
$$

### Un-normalize model
Let $f(x) = ax + b$. $f$ intercepts the two different arbitrary points $p_1, p_2$ when $x$ is $x_1, x_2$ respectively. Both points gets offset by $m\in\mathbb{R}^2$ then scaled by $s\in\mathbb{R}^2$, how do you adjust $a$ and $b$ such that $f$ still intersects each point by only using information about $m, s, a, b$?

Can basically choose any points as long as they are different and intersects with $f$. Choose
$$
p_1 = 
\begin{bmatrix}
    x_1 \\
    y_1
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    b
\end{bmatrix}
,\quad
p_2 = 
\begin{bmatrix}
    x_2 \\
    y_2
\end{bmatrix}
=
\begin{bmatrix}
    -b/a \\
    0
\end{bmatrix}
$$

Let's inpect the transformation of the points. Let $p_1', p_2'$ denote the transformed $p_1, p_2$ respectively
$$
\begin{aligned}
    p_1' &=
    \begin{bmatrix}
        s_1 (x_1 + m_1) \\
        s_2 (y_1 + m_2)
    \end{bmatrix}
    =
    \begin{bmatrix}
        s_1 m_1 \\
        s_2b + s_2m_2
    \end{bmatrix}
    \\
    p_2' &=
    \begin{bmatrix}
        s_1 (x_2 + m_1) \\
        s_2 (y_2 + m_2)
    \end{bmatrix}
    =
    \begin{bmatrix}
        -s_1b/a + s_1m_1 \\
        s_2m_2
    \end{bmatrix}
\end{aligned}
$$

Let $a'$ and $b'$ denote the adjusted values for $a$ and $b$ respectively. Using the fact that $f$ should intersect both $p_1', p_2'$ we can find slope $a'$ and intersection $b'$:
$$
\begin{aligned}
    a' &= \frac{s_2m_2 - s_2b - s_2m_2}{-s_1b/a+s_1m_1 - s_1m_1} \\
    a' &= \frac{- s_2b}{-s_1b/a} \\
    \Rightarrow a' &= \frac{s_2}{s_1}a
\end{aligned}
$$
Then for intercept $b'$
$$
\begin{aligned}
    y &= a'x + b' \\
    \Rightarrow s_2m_2 &= a'(-s_1b/a + s_1m_1) + b' \\
    \Rightarrow b' &= s_2m_2 - a'(-s_1b/a + s_1m_1)
\end{aligned}
$$
Then for intercept $b'$
$$
\begin{aligned}
    y &= a'x + b' \\
    \Rightarrow s_2b+s_2m_2 &= a's_1m_1 + b' \\
    \Rightarrow b' &= s_2b + s_2m_2 - a's_1m_1 
\end{aligned}
$$
<!-- $$
\begin{aligned}
    a' &= \frac{s_2 (y_2 + m_2) - s_2 (y_1 + m_2)}{s_1 (x_2 + m_1) - s_1 (x_1 + m_1)} \\
    a' &= \frac{s_2y_2 - s_2y_1}{s_1x_2 - s_1x_1} \\
    \Rightarrow a' &= -\frac{s_2b}{s_1b/a} =- \frac{s_2}{s_1}a
\end{aligned}
\qquad
\begin{aligned}
    b' &= s_2 (y_1 + m_2) - as_1 (x_1 + m_1) \\
    b' &= s_2y_1 + s_2m_2 - as_1x_1 + as_1m_1 \\
    \Rightarrow b' &= s_2b + s_2m_2 + as_1m_1
\end{aligned}
\qquad
\begin{aligned}
\end{aligned}
$$ -->

### 1-Line Model
Model parameters: 
- $W\in\mathbb{R}^{1\times2}$
- $b\in\mathbb{R}$

Train algorithm:

1. Given dataset $X\in\mathbb{R}^{N\times2}$, with labels $y\in\mathbb{R}^{1\times N}$, obtain predictions $\hat{y}\in\mathbb{R}^{1\times N}$:
    $$
    \begin{align}
    z &= WX^T + b \\
    \hat{y} = \sigma(z) &= \frac{1}{1 + e^{-z}}
    \end{align}
    $$

1. Compute mean squared error loss: 
    $$
    L = \frac{1}{N}\sum_{i} (\hat{y}_i - y_i)^2
    $$

1. Compute gradients for $W$: 
    $$
    \begin{align}
        \frac{\partial}{\partial W} L(\underbrace{\sigma(\overbrace{WX^T +b}^{z})}_{\hat{y}}) &= \frac{\partial L}{\partial\hat{y}} \frac{\partial\sigma}{\partial z} \frac{\partial z}{\partial W}\\
        \frac{\partial}{\partial W} L(\underbrace{\sigma(\overbrace{WX^T +b}^{z})}_{\hat{y}}) &= \frac{\partial L}{\partial\hat{y}} \frac{\partial\sigma}{\partial z} X^T 
        \\
        \frac{\partial}{\partial W} L(\underbrace{\sigma(\overbrace{WX^T +b}^{z})}_{\hat{y}}) &= \frac{\partial L}{\partial\hat{y}} \frac{e^{-z}}{(1+e^{-z})^2} X^T
        \\
        \frac{\partial}{\partial W} L(\underbrace{\sigma(\overbrace{WX^T +b}^{z})}_{\hat{y}}) &= \frac{2}{N}\sum_{i}(\hat{y}_i - y_i) \frac{e^{-z}}{(1+e^{-z})^2} X^T
    \end{align}
    $$
    
