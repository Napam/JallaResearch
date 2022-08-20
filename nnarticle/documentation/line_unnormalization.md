# Un-normalize a model of the form $ax + b$
Let $f(x)=ax+b,\hspace{0.5em}a,b\in\mathbb{R}$. Let $p_1, p_2 \in \mathbb{R}^2$ be two arbitrary different points that lies on $f$. Both points gets scaled by $s\in\mathbb{R}^2$ then offset by $m\in\mathbb{R}^2$. Let $p_1', p_2'$ be the transformed versions of $p_1, p_2$ respectively. Find $g(x) = a'x + b'$, where $a',b'$ are determined such that $g$ intercepts both $p_1', p_2'$.

Choose the points that lies on each axes:
$$
p_{1} =
\begin{bmatrix}
    p_{11} \\
    p_{12}
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    b
\end{bmatrix},
\quad
p_{2} =
\begin{bmatrix}
    p_{21} \\
    p_{22}
\end{bmatrix}
=
\begin{bmatrix}
    -b/a \\
    0
\end{bmatrix}
$$

The transformed points would then be:
$$
p_{1}' =
\begin{bmatrix}
    p_{11}' \\
    p_{12}'
\end{bmatrix}
=
\begin{bmatrix}
    m_1 \\
    bs_2+m_2
\end{bmatrix},
\quad
p_{2} =
\begin{bmatrix}
    p_{21}' \\
    p_{22}'
\end{bmatrix}
=
\begin{bmatrix}
    (-b/a)s_1 + m_1 \\
    m_2
\end{bmatrix}
$$

Now we can simply find the slope $a'$ and intersection $b'$ given the points $p_1', p_2'$:
$$
\begin{aligned}
    a' &= \frac{m_2 - (bs_2 + m_2)}{(-b/a)s_1 + m_1 - m1} \\
    a' &= \frac{-bs_2}{-bs_1/a} \\
    \Rightarrow a' &= \frac{s_2}{s_1}a
\end{aligned}
\qquad
\begin{aligned}
    bs_2 + m_2 &= a'm_1 + b' \\
    \Rightarrow b' &= bs_2 + m_2 - a'm_1
\end{aligned}
$$