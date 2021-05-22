import numpy as np 

"""
                 1
             __________
            |          |
       _____|_____     | 1
      |     |    |     |
   1  |     |    |     |
      |      ----|-----
      |          |  
       ----------
            1
"""

def calc_iou(dim: int):
    """
    Calculates IoU for two boxes that intersects "querterly" (see ASCII image) in R^dim

    x: dimension
    """
    I = 0.5**dim
    U = 2 - I
    return I/U

print(calc_iou(2))
print(calc_iou(3))