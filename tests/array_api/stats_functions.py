import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array

SEED = 314159


class StatsFunctionTests(ArkoudaTest):
    def test_max(self):
        a = Array.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = 101

        self.assertEqual(Array.max(a)[0], 101)

        aMax0 = Array.max(a, axis=0)
        self.assertEqual(aMax0.shape, (5, 4))
        self.assertEqual(aMax0[3, 2], 101)

        aMax02 = Array.max(a, axis=(0, 2))
        self.assertEqual(aMax02.shape, (7,))
        self.assertEqual(aMax02[6], 101)

        aMax02Keepdims = Array.max(a, axis=(0, 2), keepdims=True)
        self.assertEqual(aMax02Keepdims.shape, (1, 7, 1))
        self.assertEqual(aMax02Keepdims[0, 6, 0], 101)

    def test_min(self):
        a = Array.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = -1

        self.assertEqual(Array.min(a)[0], -1)

        aMin0 = Array.min(a, axis=0)
        self.assertEqual(aMin0.shape, (5, 4))
        self.assertEqual(aMin0[3, 2], -1)

        aMin02 = Array.min(a, axis=(0, 2))
        self.assertEqual(aMin02.shape, (7,))
        self.assertEqual(aMin02[6], -1)

        aMin02Keepdims = Array.min(a, axis=(0, 2), keepdims=True)
        self.assertEqual(aMin02Keepdims.shape, (1, 10, 1))
        self.assertEqual(aMin02Keepdims[0, 6, 0], -1)

    def test_mean(self):
        a = Array.ones((10, 5, 5))
        a[1, 1, 1] = 251

        self.assertEqual(Array.mean(a)[0], 2)

        a[:, 1, 1] = 26

        aMean0 = Array.mean(a, axis=0)
        self.assertEqual(aMean0.shape, (5, 5))
        self.assertEqual(aMean0[0, 0], 2)

        aMean02 = Array.mean(a, axis=(1, 2))
        self.assertEqual(aMean02.shape, (10,))
        self.assertEqual(aMean02[0], 2)
        self.assertEqual(aMean02[2], 1)

        aMean02Keepdims = Array.mean(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aMean02Keepdims.shape, (10, 1, 1))
        self.assertEqual(aMean02Keepdims[0, 0, 0], 2)
        self.assertEqual(aMean02Keepdims[2, 0, 0], 1)

    def test_std(self):
        a = Array.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        self.approx(Array.std(a)[0], 30.1)

        aStd0 = Array.std(a, axis=0)
        self.assertEqual(aStd0.shape, (5, 5))
        self.assertEqual(aStd0[0, 0], 0)

        aStd02 = Array.std(a, axis=(1, 2))
        self.assertEqual(aStd02.shape, (10,))
        self.assertEqual(aStd02[0], 0)
        self.assertEqual(aStd02[2], 0)

        aStd02Keepdims = Array.std(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aStd02Keepdims.shape, (10, 1, 1))
        self.assertEqual(aStd02Keepdims[0, 0, 0], 0)
        self.assertEqual(aStd02Keepdims[2, 0, 0], 0)
