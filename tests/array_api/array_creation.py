import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

# requires the server to be built with 2D array support
SHAPES = [(), (0,), (0, 0), (1,), (5,), (2, 2), (5, 10)]
SIZES = [1, 0, 0, 1, 5, 4, 50]
DIMS = [0, 1, 2, 1, 1, 2, 2]

class ArrayCreationTests(ArkoudaTest):
    def test_zeros(self):
        for shape, size, dim in zip(SHAPES, SIZES, DIMS):
            for dtype in ak.ScalarDTypes:
                a = Array.zeros(shape, dtype=dtype)
                self.assertEqual(a.size, size)
                self.assertEqual(a.ndim, dim)
                self.assertEqual(a.shape, shape)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(a.tolist(), np.zeros(shape, dtype=dtype).tolist())

    def test_ones(self):
        for shape, size, dim in zip(SHAPES, SIZES, DIMS):
            for dtype in ak.ScalarDTypes:
                a = Array.ones(shape, dtype=dtype)
                self.assertEqual(a.size, size)
                self.assertEqual(a.ndim, dim)
                self.assertEqual(a.shape, shape)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(a.tolist(), np.ones(shape, dtype=dtype).tolist())
