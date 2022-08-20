# Normalize a model
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
    \Rightarrow s_2b+s_2m_2 &= a's_1m_1 + b' \\
    \Rightarrow b' &= s_2b + s_2m_2 - a's_1m_1
\end{aligned}
$$