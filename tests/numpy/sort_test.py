import numpy as np
import pytest

import arkouda as ak
from arkouda.client import get_array_ranks
from arkouda.numpy.sorting import SortingAlgorithm
from arkouda.testing import assert_arkouda_array_equivalent

NUMERIC_AND_BIGINT_TYPES = ["int64", "float64", "uint64", "bigint"]


def make_ak_arrays(dtype):
    if dtype in ["int64", "uint64"]:
        return ak.randint(0, 100, 1000, dtype)
    elif dtype == "float64":
        return ak.array(np.random.rand(1000) * 10000)
    elif dtype == "bigint":
        return ak.randint(0, 100, 1000, dtype=ak.uint64)
    return None


def np_is_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])


class TestSort:
    def test_sorting_docstrings(self):
        import doctest

        from arkouda.numpy import sorting

        result = doctest.testmod(sorting, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.float64, ak.uint64, ak.float64])
    def test_compare_argsort(self, size, dtype):
        # create np version
        a = np.arange(size, dtype=dtype)
        a = a[::-1]
        iv = np.argsort(a)
        a = a[iv]
        # create ak version
        b = ak.arange(size, dtype=dtype)
        b = b[::-1]
        iv = ak.argsort(b)
        b = b[iv]
        assert np.array_equal(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.float64, ak.uint64, ak.float64])
    def test_compare_sort(self, size, dtype):
        # create np version
        a = np.arange(size, dtype=dtype)
        a = a[::-1]
        a = np.sort(a)
        # create ak version
        b = ak.arange(size, dtype=dtype)
        b = b[::-1]
        b = ak.sort(b)
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("dtype", NUMERIC_AND_BIGINT_TYPES)
    def test_is_sorted(self, dtype):
        pda = make_ak_arrays(dtype)
        sorted_pda = ak.sort(pda)
        assert ak.is_sorted(sorted_pda)
        assert np_is_sorted(sorted_pda.to_ndarray())

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_BIGINT_TYPES)
    def test_sort_dtype(self, algo, dtype):
        pda = make_ak_arrays(dtype)

        if dtype != "bigint":
            assert ak.is_sorted(ak.sort(pda, algo))
        else:
            pda_bigint = pda + 2**200
            sorted_pda = ak.sort(pda, algo)
            sorted_bi = ak.sort(pda_bigint, algo)
            assert (sorted_bi - 2**200).tolist() == sorted_pda.tolist()

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_bit_boundary_hardcode(self, algo):
        # test hardcoded 16/17-bit boundaries with and without negative values
        a = ak.array([1, -1, 32767])  # 16 bit
        b = ak.array([1, 0, 32768])  # 16 bit
        c = ak.array([1, -1, 32768])  # 17 bit
        for arr in a, b, c:
            assert ak.is_sorted(ak.sort(arr, algo))

        # test hardcoded 64-bit boundaries with and without negative values
        d = ak.array([1, -1, 2**63 - 1])
        e = ak.array([1, 0, 2**63 - 1])
        f = ak.array([1, -(2**63), 2**63 - 1])
        for arr in d, e, f:
            assert ak.is_sorted(ak.sort(arr, algo))

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_bit_boundary(self, algo):
        # test 17-bit sort
        lower = -(2**15)
        upper = 2**16
        a = ak.randint(lower, upper, 1000)
        assert ak.is_sorted(ak.sort(a, algo))

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_error_handling(self, algo):
        # Test RuntimeError from bool NotImplementedError
        ak_bools = ak.randint(0, 1, 1000, dtype=ak.bool_)
        bools = ak.randint(0, 1, 1000, dtype=bool)

        for arr in ak_bools, bools:
            with pytest.raises(ValueError):
                ak.sort(arr, algo)

        # Test TypeError from sort attempt on non-pdarray
        with pytest.raises(TypeError):
            ak.sort(list(range(0, 10)), algo)

        # Test attempt to sort Strings object, which is unsupported
        with pytest.raises(TypeError):
            ak.sort(ak.array([f"string {i}" for i in range(10)]), algo)

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_nan_sort(self, algo):
        # Reproducer from #2703
        neg_arr = np.array([-3.14, np.inf, np.nan, -np.inf, 3.14, 0.0, 3.14, -8])
        pos_arr = np.array([3.14, np.inf, np.nan, np.inf, 7.7, 0.0, 3.14, 8])
        for npa in neg_arr, pos_arr:
            assert np.allclose(np.sort(npa), ak.sort(ak.array(npa), algo).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.float64, ak.int64, ak.bigint, ak.uint64])
    @pytest.mark.parametrize("v_shape", [(), (10,), (4, 5), (2, 2, 3)])
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_searchsorted(self, size, dtype, v_shape, side):
        low = 0
        high = 100
        if dtype == ak.bigint:
            shift = 2**200
            dtype_ = ak.int64
        else:
            shift = 0
            dtype_ = dtype
        if v_shape == ():
            v = ak.randint(low=low, high=high, size=1, dtype=dtype_, seed=pytest.seed)[0]
            if dtype != ak.bigint:
                v = dtype(v)
            else:
                v = int(v) + shift
        else:
            if len(v_shape) not in get_array_ranks():
                pytest.skip(f"Server not compiled for rank {len(v_shape)}")
            else:
                v = ak.randint(low=low, high=high, size=v_shape, dtype=dtype_, seed=pytest.seed)
                v = ak.array(v, dtype) + shift
        a = ak.randint(low=low, high=high, size=size, dtype=dtype_, seed=pytest.seed)
        a = ak.sort(a)
        a = ak.array(a, dtype) + shift
        np_a = a.to_ndarray()
        if isinstance(v, ak.pdarray):
            np_v = v.to_ndarray()
        else:
            np_v = v
        np_output = np.searchsorted(np_a, np_v, side)
        ak_output = ak.searchsorted(a, v, side)
        if v_shape == ():
            assert ak_output == np_output
        else:
            assert_arkouda_array_equivalent(ak_output, np_output)

    @pytest.mark.parametrize("dtype", [ak.float64, ak.int64, ak.uint64, ak.bigint])
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_searchsorted_fast_edge_cases(self, dtype, side):
        # Test edge cases for searchsorted with fast path enabled
        # This is when x2 is also sorted
        # List of (x1, x2) edge case pairs
        edge_cases = [
            # Test case 1: basic case with small arrays
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [5, 8, 10, 12]),

            # Test case 2: x2 element equals boundary values (left/right side differences)
            ([1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11], [2, 4, 8, 10]),

            # Test case 4: All x2 elements smaller than x1[0] (all assigned to locale 0)
            ([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], [1, 2, 3, 4, 5, 6, 7, 8]),

            # Test case 5: All x2 elements larger than x1[last] (all assigned to last locale)
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [20, 25, 30, 35, 40, 45, 50, 55]),

            # Test case 6: x2 elements exactly at locale boundary transitions
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 4, 7, 8, 11, 12]),

            # Test case 7: Duplicate values spanning multiple locales
            ([1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13], [1, 5, 9, 13]),

            # Test case 8: x2 value equal to prevHigh and myLow (left side ownership test)
            ([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], [6, 8, 16, 20]),

            # Test case 9: x2 value equal to myHigh and nextLow (right side ownership test)
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], [7, 9, 11, 15, 19]),

            # Test case 10: Mixed boundary conditions with left/right side differences
            ([0, 2, 2, 4, 4, 4, 6, 6, 8, 8, 8, 10, 12, 14, 16, 18], [2, 4, 6, 8, 10, 12]),

            # Test case 11: Binary search edge case - value not found, insertion point
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]),

            # Test case 12: Large duplicates that may create empty ranges
            ([1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20], [1, 5, 10, 15, 20, 25]),

            # Test case 13: Arithmetic edge cases (potential overflow/underflow in mid calculation)
            ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600], [150, 350, 550, 750, 950, 1150, 1350, 1550]),

            # Test case 14: First locale special case where myLow doesn't matter
            ([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], [1, 2, 3, 12, 18, 22, 28, 35]),

            # Test case 15: Last locale special case where myHigh doesn't matter
            ([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], [38, 42, 47, 50, 55, 60, 70, 80]),

            # Test case 16: Out of bounds binary search results
            ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160], [5, 15, 25, 95, 105, 125, 145, 165]),

            # Test case 17: The specific case mentioned in the prompt
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [8, 12, 16, 16]),

            # Test case 18: Sequential boundary values
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),

            # Test case 19: Empty ranges created by boundary logic
            ([1, 1, 1, 1, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30], [5, 15, 25, 35]),

            # Test case 20: Maximum stress test with many duplicates
            ([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5], [1, 2, 3, 4, 5, 6]),

            # Algorithmic edge cases
            # A1: eqOrGtMyLow == x2.domain.high condition for first locale
            ([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], [45]),
            # A2: eqOrGtMyLow out of bounds condition
            ([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200], [10, 20, 30, 40]),
            # A3: gtMyLow out of bounds when searching for next value
            ([5, 5, 5, 10, 10, 10, 15, 15, 15, 20, 20, 20, 25, 25, 25, 30], [10, 15, 20, 35]),
            # A4: eqOrLeMyHigh out of bounds (negative index)
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0]),
            # A5: ltMyHigh out of bounds when searching for previous value
            ([10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40, 50, 50, 50, 60], [5, 10, 20, 30]),
            # A6: Complex boundary value equality across multiple locales
            ([1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10], [3, 6, 9]),
            # A7: Left vs Right side ownership with identical boundary values
            ([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], [8, 16, 24]),
            # A8: gtElem == eqOrGtElem condition testing
            ([1, 1, 1, 5, 5, 5, 9, 9, 9, 13, 13, 13, 17, 17, 17, 21], [1, 5, 9, 13, 17]),
            # A9: ltElem == eqOrLeElem condition testing
            ([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48], [6, 15, 24, 33, 42]),
            # A10: myFirst remains -1 leading to empty range creation
            ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600], [50, 150, 250]),
            # A11: Last locale with eqOrGtElem > myHigh
            ([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61], [35, 45, 55, 65, 75]),
            # A12: First locale with all elements in x2 < myHigh
            ([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200], [10, 20, 30, 40, 45]),
            # A13: Boundary search with leftCmp vs rightCmp differences
            ([2, 2, 2, 6, 6, 6, 10, 10, 10, 14, 14, 14, 18, 18, 18, 22], [2, 6, 10, 14, 18]),
            # A14: Stress test for boundary index calculations
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]),
            # A15: Complex ownership transfer scenarios
            ([5, 10, 10, 15, 20, 20, 25, 30, 30, 35, 40, 40, 45, 50, 50, 55], [8, 10, 12, 18, 20, 22, 28, 30, 32, 38, 40, 42, 48, 50, 52]),
            # Specialized cases
            # S1: Exact boundary value equality for left side ownership
            ([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], [6, 14, 22]),
            # S2: Exact boundary value equality for right side ownership
            ([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], [8, 16, 24]),
            # S3: Binary search returns domain.high exactly
            ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160], [95, 105, 115, 125]),
            # S4: Empty range handling with myFirst = -1
            ([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], [5, 10, 15]),
            # S5: gtElem <= myHigh && gtElem != eqOrGtElem condition
            ([5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 40], [5, 10, 15, 20, 25]),
            # S6: ltElem >= prevHigh && ltElem != eqOrLeElem condition
            ([3, 6, 6, 9, 9, 12, 12, 15, 15, 18, 18, 21, 21, 24, 24, 27], [6, 9, 12, 15, 18, 21]),
            # S7: eqOrLeMyHigh negative index scenario
            ([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600], [50]),
            # S8: All elements equal (extreme duplicate case)
            ([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5]),
            # S9: Interleaved boundaries with first/last locale special cases
            ([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], [0, 5, 35, 75, 125, 160]),
            # S10: Maximum boundary stress test
            ([1, 1, 5, 5, 5, 10, 10, 15, 15, 15, 20, 20, 25, 25, 25, 30], [1, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 35]),
            # Uneven blockDist distribution: x1 length 17
            ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], [1,8,16,17]),
        ]
        for x1, x2 in edge_cases:
            ak_x1 = ak.array(x1, dtype=dtype)
            ak_x2 = ak.array(x2, dtype=dtype)
            if dtype == ak.bigint:
                ak_x1 = ak_x1 + 2**200
                ak_x2 = ak_x2 + 2**200
            np_x1 = ak_x1.to_ndarray()
            if isinstance(ak_x2, ak.pdarray):
                np_x2 = ak_x2.to_ndarray()
            else:
                np_x2 = x2
            np_output = np.searchsorted(np_x1, np_x2, side)
            ak_output = ak.searchsorted(ak_x1, ak_x2, side, x2_sorted=True) # x2_sorted=True for fast path
            assert_arkouda_array_equivalent(ak_output, np_output)

