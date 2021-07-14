import numpy as np 
from numpy.lib.stride_tricks import as_strided
from typing import Sequence, Optional

def dummy_broadcast_strides(shape: Sequence[int], to: Sequence[int], dtype: np.dtype, strides: Optional[Sequence[int]]=None):
    # Create dummy array that only allocates one byte of space. A bool only takes a 1 byte of memory.
    # Meaning strides will be scaled by 1, my scale with appropriate dtype later.
    dummy = as_strided(True, shape=shape, strides=strides, writeable=False)
    # Multiply with dtype.itemsize to scale strides to dtype
    return np.array(np.broadcast_to(dummy, to).strides) * dtype.itemsize

if __name__ == '__main__':
    shape = (1,5,1,1,5) # shape of 'a' in your example
    print(dummy_broadcast_strides(shape, (2,5,1,5,5), np.float32())) # Must instantiate the dtype
    # [ 0 20  0  0  4]
    
    # Try really "big" array
    shape = (1024,1024,1024,1024,1024,1024) # shape of 'a' in your example
    print(dummy_broadcast_strides(shape, (2,1024,1024,1024,1024,1024,1024), np.float32())) # Must instantiate the dtype
    # [               0 4503599627370496    4398046511104       4294967296       4194304             4096            4]