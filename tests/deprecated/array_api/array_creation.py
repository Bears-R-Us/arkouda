import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

import arkouda.array_api as xp

# requires the server to be built with 2D array support
SHAPES = [(), (0,), (0, 0), (1,), (5,), (2, 2), (5, 10)]
SIZES = [1, 0, 0, 1, 5, 4, 50]
DIMS = [0, 1, 2, 1, 1, 2, 2]


class ArrayCreationTests(ArkoudaTest):
    def test_zeros(self):
        for shape, size, dim in zip(SHAPES, SIZES, DIMS):
            for dtype in ak.ScalarDTypes:
                a = xp.zeros(shape, dtype=dtype)
                self.assertEqual(a.size, size)
                self.assertEqual(a.ndim, dim)
                self.assertEqual(a.shape, shape)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(a.tolist(), np.zeros(shape, dtype=dtype).tolist())

    def test_ones(self):
        for shape, size, dim in zip(SHAPES, SIZES, DIMS):
            for dtype in ak.ScalarDTypes:
                a = xp.ones(shape, dtype=dtype)
                self.assertEqual(a.size, size)
                self.assertEqual(a.ndim, dim)
                self.assertEqual(a.shape, shape)
                self.assertEqual(a.dtype, dtype)
                self.assertEqual(a.tolist(), np.ones(shape, dtype=dtype).tolist())

    def test_from_numpy(self):
        # TODO: support 0D (scalar) arrays
        # (need changes to the create0D command from #2967)
        for shape in SHAPES[1:]:
            a = np.random.randint(0, 10, size=shape, dtype=np.int64)
            b = xp.asarray(a)
            self.assertEqual(b.size, a.size)
            self.assertEqual(b.ndim, a.ndim)
            self.assertEqual(b.shape, a.shape)
            self.assertEqual(b.tolist(), a.tolist())
