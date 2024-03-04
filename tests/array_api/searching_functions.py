import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array

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
