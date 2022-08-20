# Un-normalize a model of the form $w_0 + w_1x_1 + w_2x_2$
Let $f(x, y) = w_0 + w_1x + w_2y$ be a hyperplane embedded in $\mathbb{R}^3$. Let $p_1, p_2, p_3 \in \mathbb{R}^2$ be arbitrary different points that lies on the domain of $f$, and there exists no straight line that intercepts all the points, in other words the points are not "placed in a row". The points gets scaled by $s\in\mathbb{R}^2$ then offset by $m\in\mathbb{R}^2$. Let $p_1', p_2', p_3'$ be the transformed versions of $p_1, p_2, p_3$ respectively. Find $g(x, y) = w_0' + w_1'x + w_2'y$ such that $g(p_1') = f(p_1)$, $g(p_2') = f(p_2)$ and $g(p_3') = f(p_3)$

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
    0 \\
    0
\end{bmatrix},
\quad
p_2' =
\begin{bmatrix}
    p_{21}' \\
    p_{22}'
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    1
\end{bmatrix},
\quad
p_3' =
\begin{bmatrix}
    p_{31}' \\
    p_{32}'
\end{bmatrix}
=
\begin{bmatrix}
    1 \\
    0
\end{bmatrix}
$$