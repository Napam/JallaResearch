import numpy as np
from matplotlib import pyplot as plt 

XX, YY = np.meshgrid(*[np.linspace(-1,1,8)]*2)
points = np.c_[XX.ravel(), YY.ravel()]

w = np.array([
    [0.5, 0],
    [  0, 4],
])

plt.scatter(*points.T)
plt.scatter(*(w@points.T))
plt.gca().set_aspect('equal')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()