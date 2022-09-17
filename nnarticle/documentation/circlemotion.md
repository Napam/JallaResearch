# Circular animation
We need to animate a dot going around a circle. To do this we will parametrize that motion using the radius and angle (polar coordinates). We will need to convert to cartesian.

Position as polar:
$$
(r,\theta),\quad r \in \mathbb{R}^{+},\quad \theta\in\mathbb{R}
$$

Position as cartesian:
$$
(x, y), \quad x,y\in\mathbb{R}
$$

To convert from polar to cartesian we have the following relations:
$$
\begin{aligned}
    x &= r\cos(\theta)\\
    y &= r\sin(\theta)
\end{aligned}
$$