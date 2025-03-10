import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.numpy.sorting import SortingAlgorithm

"""
Encapsulates test cases that test sort functionality
"""


class SortTest(ArkoudaTest):
    def testSort(self):
        pda = ak.randint(0, 100, 100)
        for algo in SortingAlgorithm:
            spda = ak.sort(pda, algo)
            maxIndex = spda.argmax()
            self.assertTrue(maxIndex > 0)

        pda = ak.randint(0, 100, 100, dtype=ak.uint64)
        for algo in SortingAlgorithm:
            spda = ak.sort(pda, algo)
            self.assertTrue(ak.is_sorted(spda))

        shift_up = pda + 2**200
        for algo in SortingAlgorithm:
            sorted_pda = ak.sort(pda, algo)
            sorted_bi = ak.sort(shift_up, algo)
            self.assertListEqual((sorted_bi - 2**200).to_list(), sorted_pda.to_list())

    def testBitBoundaryHardcode(self):
        # test hardcoded 16/17-bit boundaries with and without negative values
        a = ak.array([1, -1, 32767])  # 16 bit
        b = ak.array([1, 0, 32768])  # 16 bit
        c = ak.array([1, -1, 32768])  # 17 bit
        for algo in SortingAlgorithm:
            assert ak.is_sorted(ak.sort(a, algo))
            assert ak.is_sorted(ak.sort(b, algo))
            assert ak.is_sorted(ak.sort(c, algo))

        # test hardcoded 64-bit boundaries with and without negative values
        d = ak.array([1, -1, 2**63 - 1])
        e = ak.array([1, 0, 2**63 - 1])
        f = ak.array([1, -(2**63), 2**63 - 1])
        for algo in SortingAlgorithm:
            assert ak.is_sorted(ak.sort(d, algo))
            assert ak.is_sorted(ak.sort(e, algo))
            assert ak.is_sorted(ak.sort(f, algo))

    def testBitBoundary(self):
        # test 17-bit sort
        L = -(2**15)
        U = 2**16
        a = ak.randint(L, U, 100)
        for algo in SortingAlgorithm:
            assert ak.is_sorted(ak.sort(a, algo))

    def testErrorHandling(self):
        # Test RuntimeError from bool NotImplementedError
        akbools = ak.randint(0, 1, 1000, dtype=ak.bool_)
        bools = ak.randint(0, 1, 1000, dtype=bool)

        for algo in SortingAlgorithm:
            with self.assertRaises(ValueError):
                ak.sort(akbools, algo)

            with self.assertRaises(ValueError):
                ak.sort(bools, algo)

            # Test TypeError from sort attempt on non-pdarray
            with self.assertRaises(TypeError):
                ak.sort(list(range(0, 10)), algo)

            # Test attempt to sort Strings object, which is unsupported
            with self.assertRaises(TypeError):
                ak.sort(ak.array(["String {}".format(i) for i in range(0, 10)]), algo)

    def test_nan_sort(self):
        # Reproducer from #2703
        neg_arr = np.array([-3.14, np.inf, np.nan, -np.inf, 3.14, 0.0, 3.14, -8])
        pos_arr = np.array([3.14, np.inf, np.nan, np.inf, 7.7, 0.0, 3.14, 8])
        for npa in neg_arr, pos_arr:
            self.assertTrue(
                np.allclose(np.sort(npa), ak.sort(ak.array(npa)).to_ndarray(), equal_nan=True)
            )
