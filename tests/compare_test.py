import numpy as np
import arkouda as ak
from base_test import ArkoudaTest

N = 1_000_000

class CheckTest(ArkoudaTest):

    def testCompareArange(self):
        # create np version
        nArange = np.arange(N)
        # create ak version
        aArange = ak.arange(N)
        self.assertEqual(nArange.all(), aArange.all())

    def testCompareLinspace(self):
        # create np version
        a = np.linspace(10, 20, N)
        # create ak version
        b = ak.linspace(10, 20, N)
        # print(a,b)
        self.assertTrue(np.allclose(a, b.to_ndarray()))

    def testCompareOnes(self):
        # create np version
        nArray = np.ones(N)
        # create ak version
        aArray = ak.ones(N)
        self.assertEquals(nArray.all(), aArray.to_ndarray().all())

    def testCompareZeros(self):
        # create np version
        nArray = np.zeros(N)
        # create ak version
        aArray = ak.zeros(N)
        self.assertEquals(nArray.all(), aArray.to_ndarray().all())

    def testCompareArgsort(self):
        # create np version
        a = np.arange(N)
        a = a[::-1]
        iv = np.argsort(a)
        a = a[iv]
        # create ak version
        b = ak.arange(N)
        b = b[::-1]
        iv = ak.argsort(b)
        b = b[iv]

    def testCompareSort(self):
        # create np version
        a = np.arange(N)
        a = a[::-1]
        a = np.sort(a)
        # create ak version
        b = ak.arange(N)
        b = b[::-1]
        b = ak.sort(b)
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareGetSlice(self):
        # create np version
        a = np.ones(N)
        a = a[::2]
        # create ak version
        b = ak.ones(N)
        b = b[::2]
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareSetSliceValue(self):
        # create np version
        a = np.ones(N)
        a[::2] = -1
        # create ak version
        b = ak.ones(N)
        b[::2] = -1
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareSetSlice(self):
        # create np version
        a = np.ones(N)
        a[::2] = a[::2] * -1
        # create ak version
        b = ak.ones(N)
        b[::2] = b[::2] * -1
        self.assertEqual(a.all(), b.to_ndarray().all())

    def check_get_bool_iv(self):
        # create np version
        a = np.arange(N)
        a = a[a < N//2]
        # create ak version
        b = ak.arange(N)
        b = b[b < N//2]
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareSet_bool_iv_value(self):
        # create np version
        a = np.arange(N)
        a[a < N//2] = -1
        # create ak version
        b = ak.arange(N)
        b[b < N//2] = -1
        self.assertEqual(a.all(), b.to_ndarray().all())

    def check_set_bool_iv(self):
        # create np version
        a = np.arange(N)
        a[a < N//2] = a[:N//2] * -1
        # create ak version
        b = ak.arange(N)
        b[b < N//2] = b[:N//2] * -1
        self.assertEqual(a.all(), b.to_ndarray().all())
       
    def check_get_integer_iv(self):
        # create np version
        a = np.arange(N)
        iv = np.arange(N//2)
        a = a[iv]
        # create ak version
        b = ak.arange(N)
        iv = ak.arange(N//2)
        b = b[iv]
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareSetIntegerIvVal(self):
        # create np version
        a = np.arange(N)
        iv = np.arange(N//2)
        a[iv] = -1
        # create ak version
        b = ak.arange(N)
        iv = ak.arange(N//2)
        b[iv] = -1
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareSetIntegerIv(self):
        # create np version
        a = np.arange(N)
        iv = np.arange(N//2)
        a[iv] = iv*10
        # create ak version
        b = ak.arange(N)
        iv = ak.arange(N//2)
        b[iv] = iv*10
        self.assertEqual(a.all(), b.to_ndarray().all())

    def testCompareGetIntegerIdx(self):
        # create np version
        a = np.arange(N)
        v1 = a[N//2]
        # create ak version
        b = ak.arange(N)
        v2 = b[N//2]
        self.assertEqual(v1, v2)
        self.assertEqual(a[-1], b[-1])

    def testCompareSetIntegerIdx(self):
        # create np version
        a = np.arange(N)
        a[N//2] = -1
        a[-1] = -1
        v1 = a[N//2]
        # create ak version
        b = ak.arange(N)
        b[N//2] = -1
        b[-1] = -1
        v2 = b[N//2]
        self.assertEqual(v1, v2)
        self.assertEqual(a[-1], b[-1])
