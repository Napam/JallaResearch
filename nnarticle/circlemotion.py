import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from time import perf_counter
import sys

fig, ax = plt.subplots(1, 1)
scatter = ax.scatter([], [])
ax.grid()
ax.set_xlim([-2.1, 2.1])
ax.set_ylim([-2.1, 2.1])
ax.set_aspect('equal')

prevTime = perf_counter()
pos = np.array([[0, 0]], dtype=float)


def animate(i: int):
    global prevTime
    pos[0, 0] = 1.75 * math.cos(i * 0.01)
    pos[0, 1] = 1 * math.sin(i * 0.01)
    scatter.set_offsets(pos)
    currTime = perf_counter()
    not i % 50 and sys.stdout.write(f'\r\x1b[KFPS: {1 / (currTime - prevTime)}')
    prevTime = currTime
    return (scatter,)


anime = animation.FuncAnimation(fig, animate, blit=True, interval=0)
plt.show()
print()
print(ax.get_children())
