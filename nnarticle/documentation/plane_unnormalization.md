# Un-normalize a model of the form $w_0 + w_1x_1 + w_2x_2$
Let $f(x, y) = w_0 + w_1x + w_2y$ be a hyperplane embedded in $\mathbb{R}^3$. Let $p_1, p_2, p_3 \in \mathbb{R}^2$ be arbitrary different points that lies on the domain of $f$, and there exists no straight line that intercepts all the points, in other words the points are not "placed in a row". The points gets scaled by $s\in\mathbb{R}^2$ then offset by $m\in\mathbb{R}^2$. Let $p_1', p_2', p_3'$ be the transformed versions of $p_1, p_2, p_3$ respectively. Find $g(x, y) = w_0' + w_1'x + w_2'y$ such that $g(p_1') = f(p_1)$, $g(p_2') = f(p_2)$ and $g(p_3') = f(p_3)$

**Solution**

Choose the following points:
$$
p_1 =
\begin{bmatrix}
    p_{11} \\
    p_{12}
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    0
\end{bmatrix},
\quad
p_2 =
\begin{bmatrix}
    p_{21} \\
    p_{22}
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    1
\end{bmatrix},
\quad
p_3 =
\begin{bmatrix}
    p_{31} \\
    p_{32}
\end{bmatrix}
=
\begin{bmatrix}
    1 \\
    0
\end{bmatrix}
$$

Then
$$
p_1' =
\begin{bmatrix}
    p_{11}' \\
    p_{12}'
\end{bmatrix}
=
\begin{bmatrix}
    m_1 \\
    m_2
\end{bmatrix},
\quad
p_2' =
\begin{bmatrix}
    p_{21}' \\
    p_{22}'
\end{bmatrix}
=
\begin{bmatrix}
    m_1 \\
    s_2 + m_2
\end{bmatrix},
\quad
p_3' =
\begin{bmatrix}
    p_{31}' \\
    p_{32}'
\end{bmatrix}
=
\begin{bmatrix}
    s_1 + m_1 \\
    m_2
\end{bmatrix}
$$

Then these equations must be satisfied:
$$
\begin{aligned}
    g(p_{11}', p_{12}') &= w_0' + w_1'p_{11}' + w_2'p_{12}' = w_0 + w_1p_{11} + w_2p_{12} = f(p_{11}, p_{12}) \\
    g(p_{21}', p_{22}') &= w_0' + w_1'p_{21}' + w_2'p_{22}' = w_0 + w_1p_{21} + w_2p_{22} = f(p_{21}, p_{22}) \\
    g(p_{31}', p_{32}') &= w_0' + w_1'p_{31}' + w_2'p_{32}' = w_0 + w_1p_{31} + w_2p_{32} = f(p_{31}, p_{32})
\end{aligned}
$$

Inserting for the points yields
$$
\begin{aligned}
    w_0' + w_1'm_1 + w_2'm_2 &= w_0 \\
    w_0' + w_1'm_1 + w_2'(s_2+m_2) &= w_0 + w_2 \\
    w_0' + w_1'(s_1 + m_1) + w_2'm_2 &= w_0 + w_1
\end{aligned}
$$
In matrix form (since it will be easier to put into wolfram alpha hehe):
$$
\begin{bmatrix}
    1 & m_1 & m_2 \\
    1 & m_1 & s_2 + m_2 \\
    1 & s_1 + m_1 & m_2
\end{bmatrix}
\begin{bmatrix}
    w_0' \\    
    w_1' \\    
    w_2'
\end{bmatrix}
=
\begin{bmatrix}
    w_0 \\
    w_0 +  w_2 \\
    w_0 + w_1
\end{bmatrix}
$$

Wolfram alpha says:
$$
\begin{aligned}
    w_0' &= w_0 - \frac{m_1 w_1}{s_1} - \frac{m_2 w_2}{s_2} \\
    w_1' &= \frac{w_1}{s_1} \\
    w_2' &= \frac{w_2}{s_2}
\end{aligned}
$$