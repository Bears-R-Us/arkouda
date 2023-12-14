import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

# requires the server to be built with 2D array support
SHAPES = [(1,), (5,), (2, 2), (5, 10), (1000, 100)]
SEED = 12345

class ArrayCreationTests(ArkoudaTest):
    def test_argsort(self):
        for shape in SHAPES:
            for axis in range(len(shape)):
                for dtype in ak.ScalarDTypes:
                    a = Array.asarray(ak.randint(0, 100, shape, dtype=dtype, seed=SEED))
                    b = Array.argsort(a, axis=axis)

                    self.assertEqual(b.size, a.size)
                    self.assertEqual(b.ndim, a.ndim)
                    self.assertEqual(b.shape, a.shape)

                    if len(shape) == 1:
                        aSorted = Array.take(a, b, axis=axis).tolist()
                    else:
                        aSorted = []
                        for i in range(shape[0 if axis == 1 else 1]):
                            aSorted.append([])
                            for j in range(shape[1 if axis == 1 else 0]):
                                aSorted[i].append(a[i][b[i, j]])
