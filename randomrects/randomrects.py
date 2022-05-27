import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike
from numpy.random import default_rng
rng = default_rng(42069)


def get_rects(n: int) -> np.ndarray:
    """
    Params
    ------
    n: number of rectangles
    """
    rects = []
    ceilsqrtn = int(np.ceil(np.sqrt(n)))
    n_grids = ceilsqrtn * ceilsqrtn

    # Create rectangles in the "full space", that is the [0, 1] space
    rects = rng.uniform(size = (n, 4))

    # To ensure that the rectangles are in (x1, x2, y1, y2) format where
    # Upper left corner is (x1, y1) and bottom right corner (x2, y2)
    # Result looks fine without this, but it's a nice to have
    rects[:,:2].sort(1)
    rects[:,2:].sort(1)

    # Create a ceilsqrtn x ceilsqrtn even grid space
    flat_grid_indices = rng.choice(n_grids, n, False)
    offsets = np.unravel_index(flat_grid_indices, (ceilsqrtn, ceilsqrtn))

    # Move each rectangle into their own randomly assigned grid
    # This will result with triangles in a space that is ceilsqrtn times larger than the [0, 1] space
    rects[:,:2] += offsets[1][..., None]
    rects[:,2:] += offsets[0][..., None]

    # Scale everything down to the [0, 1] space
    rects /= ceilsqrtn

    return rects
    

def plot_rects(rects: ArrayLike, width: int = 10, height: int = 10):
    fig, ax = plt.subplots(figsize=(width, height))
    for x1, x2, y1, y2 in rects:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='gray', edgecolor='black', fill=True))
    plt.show()

rects = get_rects(150)
plot_rects(rects)