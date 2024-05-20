from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

SEED = 314159


class SearchingFunctions(ArkoudaTest):

    def test_argmax(self):
        a = Array.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))
        a[3, 2, 1] = 101

        print(a.tolist())

        self.assertEqual(Array.argmax(a), 1 + 2*6 + 3*6*5)

        aArgmax0 = Array.argmax(a, axis=0)
        self.assertEqual(aArgmax0.shape, (5, 6))
        self.assertEqual(aArgmax0[2, 1], 3)

        aArgmax1Keepdims = Array.argmax(a, axis=1, keepdims=True)
        self.assertEqual(aArgmax1Keepdims.shape, (4, 1, 6))
        self.assertEqual(aArgmax1Keepdims[3, 0, 1], 2)

    def test_argmin(self):
        a = Array.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))
        a[3, 2, 1] = -1

        self.assertEqual(Array.argmin(a), 103)

        aArgmin0 = Array.argmin(a, axis=0)
        self.assertEqual(aArgmin0.shape, (5, 6))
        self.assertEqual(aArgmin0[2, 1], 3)

        aArgmin1Keepdims = Array.argmin(a, axis=1, keepdims=True)
        self.assertEqual(aArgmin1Keepdims.shape, (4, 1, 6))
        self.assertEqual(aArgmin1Keepdims[3, 0, 1], 2)

    def test_nonzero(self):
        a = Array.zeros((4, 5, 6), dtype=ak.int64)
        a[0, 1, 0] = 1
        a[1, 2, 3] = 1
        a[2, 2, 2] = 1
        a[3, 2, 1] = 1

        nz = Array.nonzero(a)

        print(nz)

        self.assertEqual(nz[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(nz[1].tolist(), [1, 2, 2, 2])
        self.assertEqual(nz[2].tolist(), [0, 3, 2, 1])

    def test_where(self):
        a = Array.zeros((4, 5, 6), dtype=ak.int64)
        a[1, 2, 3] = 1
        a[3, 2, 1] = 1
        a[2, 2, 2] = 1

        b = Array.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))
        c = Array.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))

        d = Array.where(a, b, c)

        self.assertEqual(d.shape, (4, 5, 6))
        self.assertEqual(d[1, 2, 3], b[1, 2, 3])
        self.assertEqual(d[3, 2, 1], b[3, 2, 1])
        self.assertEqual(d[2, 2, 2], b[2, 2, 2])
        self.assertEqual(d[0, 0, 0], c[0, 0, 0])
        self.assertEqual(d[3, 3, 3], c[3, 3, 3])

    def test_search_sorted(self):
        a = Array.asarray(ak.randint(0, 100, 1000, dtype=ak.float64))
        b = Array.asarray(ak.randint(0, 100, (10, 10), dtype=ak.float64))

        anp = a.to_ndarray()
        bnp = b.to_ndarray()

        sorter = Array.argsort(a)

        for side in ["left", "right"]:
            indices = Array.searchsorted(a, b, side=side, sorter=sorter)
            indicesnp = np.searchsorted(anp, bnp, side=side, sorter=sorter.to_ndarray())

            self.assertEqual(indices.tolist(), indicesnp.tolist())
