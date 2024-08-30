from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as xp
import math
import numpy as np

SEED = 314159


class StatsFunctionTests(ArkoudaTest):
    def test_max(self):
        a = xp.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = 101

        self.assertEqual(xp.max(a), 101)

        aMax0 = xp.max(a, axis=0)
        self.assertEqual(aMax0.shape, (7, 4))
        self.assertEqual(aMax0[6, 2], 101)

        aMax02 = xp.max(a, axis=(0, 2))
        self.assertEqual(aMax02.shape, (7,))
        self.assertEqual(aMax02[6], 101)

        aMax02Keepdims = xp.max(a, axis=(0, 2), keepdims=True)
        self.assertEqual(aMax02Keepdims.shape, (1, 7, 1))
        self.assertEqual(aMax02Keepdims[0, 6, 0], 101)

    def test_min(self):
        a = xp.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = -1

        self.assertEqual(xp.min(a), -1)

        aMin0 = xp.min(a, axis=0)
        self.assertEqual(aMin0.shape, (7, 4))
        self.assertEqual(aMin0[6, 2], -1)

        aMin02 = xp.min(a, axis=(0, 2))
        self.assertEqual(aMin02.shape, (7,))
        self.assertEqual(aMin02[6], -1)

        aMin02Keepdims = xp.min(a, axis=(0, 2), keepdims=True)
        self.assertEqual(aMin02Keepdims.shape, (1, 7, 1))
        self.assertEqual(aMin02Keepdims[0, 6, 0], -1)

    def test_mean(self):
        a = xp.ones((10, 5, 5))
        a[0, 0, 0] = 251

        self.assertEqual(int(xp.mean(a)), 2)

        a[:, 0, 0] = 26

        print(a.tolist())

        aMean0 = xp.mean(a, axis=0)
        self.assertEqual(aMean0.shape, (5, 5))
        self.assertEqual(aMean0[0, 0], 26)
        self.assertEqual(aMean0[2, 2], 1)

        aMean02 = xp.mean(a, axis=(1, 2))
        self.assertEqual(aMean02.shape, (10,))
        self.assertEqual(aMean02[0], 2)
        self.assertEqual(aMean02[2], 2)

        aMean02Keepdims = xp.mean(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aMean02Keepdims.shape, (10, 1, 1))
        self.assertEqual(aMean02Keepdims[0, 0, 0], 2)
        self.assertEqual(aMean02Keepdims[2, 0, 0], 2)

    def test_std(self):
        a = xp.ones((10, 5, 5), dtype=ak.float64)
        a[0, 0, 0] = 26
        self.assertTrue(math.fabs(float(xp.std(a)) - math.sqrt(2.49)) < 1e-10)

        aStd0 = xp.std(a, axis=0)
        self.assertEqual(aStd0.shape, (5, 5))
        self.assertEqual(aStd0[0, 0], 7.5)

        aStd02 = xp.std(a, axis=(1, 2))
        self.assertEqual(aStd02.shape, (10,))
        self.assertTrue(abs(aStd02[0] - math.sqrt(24)) < 1e-10)
        self.assertTrue(abs(aStd02[2] - math.sqrt(24)) < 1e-10)

        aStd02Keepdims = xp.std(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aStd02Keepdims.shape, (10, 1, 1))
        self.assertTrue(abs(aStd02Keepdims[0, 0, 0] - math.sqrt(24)) < 1e-10)
        self.assertTrue(abs(aStd02Keepdims[2, 0, 0] - math.sqrt(24)) < 1e-10)

    def test_var(self):
        a = xp.ones((10, 5, 5), dtype=ak.float64)
        a[0, 0, 0] = 26
        self.assertTrue(math.fabs(float(xp.var(a)) - 2.49) < 1e-10)

        aStd0 = xp.var(a, axis=0)
        self.assertEqual(aStd0.shape, (5, 5))
        self.assertEqual(aStd0[0, 0], 7.5**2)

        aStd02 = xp.var(a, axis=(1, 2))
        self.assertEqual(aStd02.shape, (10,))
        self.assertTrue(abs(aStd02[0] - 24) < 1e-10)
        self.assertTrue(abs(aStd02[2] - 24) < 1e-10)

        aStd02Keepdims = xp.var(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aStd02Keepdims.shape, (10, 1, 1))
        self.assertTrue(abs(aStd02Keepdims[0, 0, 0] - 24) < 1e-10)
        self.assertTrue(abs(aStd02Keepdims[2, 0, 0] - 24) < 1e-10)

    def test_prod(self):
        a = xp.ones((2, 3, 4))
        a = a + a

        self.assertEqual(xp.prod(a), 2**24)

        aProd0 = xp.prod(a, axis=0)
        self.assertEqual(aProd0.shape, (3, 4))
        self.assertEqual(aProd0[0, 0], 2**2)

        aProd02 = xp.prod(a, axis=(1, 2))
        self.assertEqual(aProd02.shape, (2,))
        self.assertEqual(aProd02[0], 2**12)

        aProd02Keepdims = xp.prod(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aProd02Keepdims.shape, (2, 1, 1))
        self.assertEqual(aProd02Keepdims[0, 0, 0], 2**12)

    def test_sum(self):
        a = xp.ones((2, 3, 4))

        self.assertEqual(xp.sum(a), 24)

        aSum0 = xp.sum(a, axis=0)
        self.assertEqual(aSum0.shape, (3, 4))
        self.assertEqual(aSum0[0, 0], 2)

        aSum02 = xp.sum(a, axis=(1, 2))
        self.assertEqual(aSum02.shape, (2,))
        self.assertEqual(aSum02[0], 12)

        aSum02Keepdims = xp.sum(a, axis=(1, 2), keepdims=True)
        self.assertEqual(aSum02Keepdims.shape, (2, 1, 1))
        self.assertEqual(aSum02Keepdims[0, 0, 0], 12)

    def test_cumsum(self):
        a = xp.asarray(ak.randint(0, 100, (5, 6, 7), seed=SEED))

        a_sum_0 = xp.cumulative_sum(a, axis=0)
        a_sum_0_np = np.cumsum(a.to_ndarray(), axis=0)
        self.assertEqual(a_sum_0.shape, (5, 6, 7))
        self.assertEqual(a_sum_0.tolist(), a_sum_0_np.tolist())

        a_sum_1 = xp.cumulative_sum(a, axis=1)
        a_sum_1_np = np.cumsum(a.to_ndarray(), axis=1)
        self.assertEqual(a_sum_1.shape, (5, 6, 7))
        self.assertEqual(a_sum_1.tolist(), a_sum_1_np.tolist())

        b = xp.ones((5, 6, 7))

        b_sum_0 = xp.cumulative_sum(b, axis=0, include_initial=True)
        self.assertEqual(b_sum_0.shape, (6, 6, 7))
        self.assertEqual(b_sum_0[0, 0, 0], 0)
        self.assertEqual(b_sum_0[1, 0, 0], 1)
        self.assertEqual(b_sum_0[5, 0, 0], 5)

        b_sum_1 = xp.cumulative_sum(b, axis=1, include_initial=True)
        self.assertEqual(b_sum_1.shape, (5, 7, 7))
        self.assertEqual(b_sum_1[0, 0, 0], 0)
        self.assertEqual(b_sum_1[0, 1, 0], 1)
        self.assertEqual(b_sum_1[0, 6, 0], 6)

        c = xp.asarray(ak.randint(0, 100, 50, dtype=ak.float64, seed=SEED))
        c_sum = xp.cumulative_sum(c)
        c_sum_np = np.cumsum(c.to_ndarray())

        self.assertEqual(c_sum.shape, (50,))
        c_list = c_sum.tolist()
        c_sum_np = c_sum_np.tolist()
        for i in range(50):
            self.assertAlmostEqual(c_list[i], c_sum_np[i])
