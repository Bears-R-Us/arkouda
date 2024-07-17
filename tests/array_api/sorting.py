from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as xp

# requires the server to be built with 2D array support
SHAPES = [(1,), (25,), (5, 10), (10, 5)]
SEED = 12345


class ArrayCreationTests(ArkoudaTest):
    def test_argsort(self):
        for shape in SHAPES:
            for dtype in ak.ScalarDTypes:
                for axis in range(len(shape)):
                    a = xp.asarray(ak.randint(0, 100, shape, dtype=dtype, seed=SEED))
                    b = xp.argsort(a, axis=axis)

                    self.assertEqual(b.size, a.size)
                    self.assertEqual(b.ndim, a.ndim)
                    self.assertEqual(b.shape, a.shape)

                    if len(shape) == 1:
                        aSorted = xp.take(a, b, axis=axis).tolist()

                        for i in range(1, len(aSorted)):
                            self.assertLessEqual(aSorted[i - 1], aSorted[i])
                    else:
                        if axis == 0:
                            for j in range(shape[1]):
                                # TODO: use take once 'squeeze' is implemented
                                # aSorted = xp.take(a, squeeze(b[:, j]), axis=0).tolist())
                                for i in range(shape[0]-1):
                                    self.assertLessEqual(a[b[i, j], j], a[b[i+1, j], j])

                        else:
                            for i in range(shape[0]):
                                # TODO: use take once 'squeeze' is implemented
                                # aSorted = xp.take(a, squeeze(b[i, :]), axis=1).tolist())
                                for j in range(shape[1]-1):
                                    self.assertLessEqual(a[i, b[i, j]], a[i, b[i, j+1]])

    def test_sort(self):
        for shape in SHAPES:
            for dtype in ak.ScalarDTypes:
                if dtype == ak.bool_:
                    continue
                for axis in range(len(shape)):
                    a = xp.asarray(ak.randint(0, 100, shape, dtype=dtype, seed=SEED))
                    sorted = xp.sort(a, axis=axis)

                    self.assertEqual(sorted.size, a.size)
                    self.assertEqual(sorted.ndim, a.ndim)
                    self.assertEqual(sorted.shape, a.shape)

                    if len(shape) == 1:
                        for i in range(1, sorted.size):
                            self.assertLessEqual(sorted[i - 1], sorted[i])

                    else:
                        if axis == 0:
                            for j in range(shape[1]):
                                for i in range(shape[0]-1):
                                    self.assertLessEqual(sorted[i, j], sorted[i+1, j])

                        else:
                            for i in range(shape[0]):
                                for j in range(shape[1]-1):
                                    self.assertLessEqual(sorted[i, j], sorted[i, j+1])
