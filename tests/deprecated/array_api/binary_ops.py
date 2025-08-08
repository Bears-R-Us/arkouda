from base_test import ArkoudaTest
from context import arkouda as ak

import arkouda.array_api as xp

# requires the server to be built with 2D array support
SHAPE_A = [(1,), (5,), (2, 2), (20, 10), (20, 10), (1, 10), (5, 10), (5, 10)]
SHAPE_B = [(1,), (5,), (2, 2), (1, 10), (20, 1), (5, 1), (10,), (1,)]
SEED = 123

ops = ['+', '-', '*', '/']


class ArrayCreationTests(ArkoudaTest):
    def test_binops(self):
        for op in ops:
            for shape_a, shape_b in zip(SHAPE_A, SHAPE_B):
                for dtype in ak.ScalarDTypes:
                    if dtype != ak.bool_:
                        x = xp.asarray(ak.randint(0, 100, shape_a, dtype=dtype, seed=SEED))
                        y = xp.asarray(ak.randint(0, 100, shape_b, dtype=dtype, seed=SEED))

                        z = eval('x ' + op + ' y')
                        ybc = xp.broadcast_to(y, z.shape)

                        if z.ndim == 1:
                            for i in range(shape_a[0]):
                                self.assertEqual(z[i], eval('x[i] ' + op + ' ybc[i]'))
                        else:
                            for i in range(shape_a[0]):
                                for j in range(shape_a[1]):
                                    self.assertEqual(z[i, j], eval('x[i, j]' + op + 'ybc[i, j]'))
