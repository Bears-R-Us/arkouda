import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import math

SEED = 314159

class StatsFunctionTests(ArkoudaTest):
    def test_max(self):
        a = Array.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = 101

        self.assertEqual(Array.max(a), 101)

        aMax0 = Array.max(a, axis=0)
        self.assertEqual(aMax0.shape, (7, 4))
        self.assertEqual(aMax0[6, 2], 101)

        aMax02 = Array.max(a, axis=(0, 2))
        self.assertEqual(aMax02.shape, (7,))
        self.assertEqual(aMax02[6], 101)

        aMax02Keepdims = Array.max(a, axis=(0, 2), keepdims=True)
        self.assertEqual(aMax02Keepdims.shape, (1, 7, 1))
        self.assertEqual(aMax02Keepdims[0, 6, 0], 101)

    def test_min(self):
        a = Array.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = -1

        self.assertEqual(Array.min(a), -1)

        aMin0 = Array.min(a, axis=0)
        self.assertEqual(aMin0.shape, (7, 4))
        self.assertEqual(aMin0[6, 2], -1)

        aMin02 = Array.min(a, axis=(0, 2))
        self.assertEqual(aMin02.shape, (7,))
        self.assertEqual(aMin02[6], -1)

        aMin02Keepdims = Array.min(a, axis=(0, 2), keepdims=True)
        self.assertEqual(aMin02Keepdims.shape, (1, 7, 1))
        self.assertEqual(aMin02Keepdims[0, 6, 0], -1)

    def test_mean(self):
        a = Array.ones((10, 5, 5))
        a[0, 0, 0] = 251

        self.assertEqual(Array.mean(a), 2)

        a[:, 0, 0] = 26

        print(a.tolist())

        aMean0 = Array.mean(a, axis=0)
        self.assertEqual(aMean0.shape, (5, 5))
        self.assertEqual(aMean0[0, 0], 26)
        self.assertEqual(aMean0[2, 2], 1)

        aMean02 = Array.mean(a, axis=(1, 2))
        self.assertEqual(aMean02.shape, (10,))
        self.assertEqual(aMean02[0], 2)
        self.assertEqual(aMean02[2], 2)

        aMean02Keepdims = Array.mean(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aMean02Keepdims.shape, (10, 1, 1))
        self.assertEqual(aMean02Keepdims[0, 0, 0], 2)
        self.assertEqual(aMean02Keepdims[2, 0, 0], 2)

    def test_std(self):
        a = Array.ones((10, 5, 5), dtype=ak.float64)
        a[0, 0, 0] = 26
        self.assertTrue(math.fabs(float(Array.std(a)) - math.sqrt(2.49)) < 1e-10)

        aStd0 = Array.std(a, axis=0)
        self.assertEqual(aStd0.shape, (5, 5))
        self.assertEqual(aStd0[0, 0], 7.5)

        aStd02 = Array.std(a, axis=(1, 2))
        self.assertEqual(aStd02.shape, (10,))
        self.assertTrue(abs(aStd02[0] - math.sqrt(24)) < 1e-10)
        self.assertTrue(abs(aStd02[2] - math.sqrt(24)) < 1e-10)

        aStd02Keepdims = Array.std(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aStd02Keepdims.shape, (10, 1, 1))
        self.assertTrue(abs(aStd02Keepdims[0, 0, 0] - math.sqrt(24)) < 1e-10)
        self.assertTrue(abs(aStd02Keepdims[2, 0, 0] - math.sqrt(24)) < 1e-10)

    def test_var(self):
        a = Array.ones((10, 5, 5), dtype=ak.float64)
        a[0, 0, 0] = 26
        self.assertTrue(math.fabs(float(Array.var(a)) - 2.49) < 1e-10)

        aStd0 = Array.var(a, axis=0)
        self.assertEqual(aStd0.shape, (5, 5))
        self.assertEqual(aStd0[0, 0], 7.5**2)

        aStd02 = Array.var(a, axis=(1, 2))
        self.assertEqual(aStd02.shape, (10,))
        self.assertTrue(abs(aStd02[0] - 24) < 1e-10)
        self.assertTrue(abs(aStd02[2] - 24) < 1e-10)

        aStd02Keepdims = Array.var(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aStd02Keepdims.shape, (10, 1, 1))
        self.assertTrue(abs(aStd02Keepdims[0, 0, 0] - 24) < 1e-10)
        self.assertTrue(abs(aStd02Keepdims[2, 0, 0] - 24) < 1e-10)

    def test_prod(self):
        a = Array.ones((2, 3, 4))
        a = a + a

        self.assertEqual(Array.prod(a), 2**24)

        aProd0 = Array.prod(a, axis=0)
        self.assertEqual(aProd0.shape, (3, 4))
        self.assertEqual(aProd0[0, 0], 2**2)

        aProd02 = Array.prod(a, axis=(1, 2))
        self.assertEqual(aProd02.shape, (2,))
        self.assertEqual(aProd02[0], 2**12)

        aProd02Keepdims = Array.prod(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aProd02Keepdims.shape, (2, 1, 1))
        self.assertEqual(aProd02Keepdims[0, 0, 0], 2**12)

    def test_sum(self):
        a = Array.ones((2, 3, 4))

        print(a.tolist())

        self.assertEqual(Array.sum(a), 24)

        aSum0 = Array.sum(a, axis=0)
        self.assertEqual(aSum0.shape, (3, 4))
        self.assertEqual(aSum0[0, 0], 2)

        aSum02 = Array.sum(a, axis=(1, 2))
        self.assertEqual(aSum02.shape, (2,))
        self.assertEqual(aSum02[0], 12)

        aSum02Keepdims = Array.sum(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aSum02Keepdims.shape, (2, 1, 1))
        self.assertEqual(aSum02Keepdims[0, 0, 0], 12)
