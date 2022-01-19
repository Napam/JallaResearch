import numpy as np
import unittest

def baseIndex(array: np.ndarray, index: int) -> int:
    base = array.base
    if base is None:
        return index
    size = array.dtype.itemsize
    stride = array.strides[0] // size
    offset = (array.__array_interface__['data'][0] - base.__array_interface__['data'][0]) // size
    return offset + index * stride

a = np.array([0,1,2,3,4,5,6])
b = a
class Test(unittest.TestCase):

    def test_1_simple(self):
        """b = a"""
        b = a
        i = 1
        j = baseIndex(b, i)
        self.assertEqual(a[j], b[i])
    
    def test_2_offset(self):
        """b = a[3:]"""
        b = a[3:]
        i = 1
        j = baseIndex(b, i)
        self.assertEqual(a[j], b[i])
    
    def test_3_strided(self):
        """b = a[1::2]"""
        b = a[1::2]
        i = 1
        j = baseIndex(b, i)
        self.assertEqual(a[j], b[i])
    
    def test_4_reverse_strided(self):
        """b = a[4::-2]"""
        b = a[4::-2]
        i = 1
        j = baseIndex(b, i)
        self.assertEqual(a[j], b[i])


unittest.main(verbosity=2)