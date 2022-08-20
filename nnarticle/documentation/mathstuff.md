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

