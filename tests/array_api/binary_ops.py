import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

# requires the server to be built with 2D array support
SHAPE_A = [(1,), (5,), (2, 2), (5, 10), (20, 10), (20, 10)]
SHAPE_B = [(1,), (5,), (2, 2), (5, 10), (1, 10), (20, 1)]
SEED = 12345

class ArrayCreationTests(ArkoudaTest):
    def test_add(self):
        for shape_a, shape_b in zip(SHAPE_A, SHAPE_B):
            for dtype in ak.ScalarDTypes:
                if dtype != ak.bool:
                    x = Array.asarray(ak.randint(0, 100, shape_a, dtype=dtype, seed=SEED))
                    y = Array.asarray(ak.randint(0, 100, shape_b, dtype=dtype, seed=SEED))

                    z = x + y

                    self.assertEqual(z.size, x.size)
                    self.assertEqual(z.ndim, x.ndim)
                    self.assertEqual(z.shape, x.shape)

                    ybc = Array.broadcast_to(y, shape_a)

                    if z.ndim == 1:
                        for i in range(shape_a[0]):
                            print(z[i], x[i], ybc[i])
                            self.assertEqual(z[i], x[i] + ybc[i])
                    else:
                        for i in range(shape_a[0]):
                            for j in range(shape_a[1]):
                                self.assertEqual(z[i, j], x[i, j] + ybc[i, j])
