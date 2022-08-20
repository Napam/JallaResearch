from matplotlib import pyplot as plt
import numpy as np

p = np.array([[0,1],[1,0]])
m = np.array([1,2])
s = np.array([3,1])

plt.scatter(*p.T)
plt.scatter(*(p*s + m).T)
plt.show()