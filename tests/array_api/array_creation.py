import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

from functools import reduce

# requires the server to be built with 2D array support

SHAPES = [(), (0,), (0, 0), 1, (5,), (2, 2)]

class ArrayCreationTests(ArkoudaTest):
    def test_zeros(self):
        for shape in SHAPES:
            for dtype in ak.ScalarDTypes:
                a = Array.zeros(shape, dtype=dtype)
                self.assertEqual(a.size, reduce(lambda x, y: x * y, shape, 1))
                self.assertEqual(a.ndim, len(shape))
                self.assertTupleEqual(a.shape, shape)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(a._array.to_ndarray(), np.zeros(shape, dtype=dtype))

    def test_ones(self):
        for shape in SHAPES:
            for dtype in ak.ScalarDTypes:
                a = Array.ones(shape, dtype=dtype)
                self.assertEqual(a.size, reduce(lambda x, y: x * y, shape, 1))
                self.assertEqual(a.ndim, len(shape))
                self.assertTupleEqual(a.shape, shape)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(a._array.to_ndarray(), np.ones(shape, dtype=dtype))
