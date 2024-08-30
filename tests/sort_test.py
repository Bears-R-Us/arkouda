import numpy as np
import pytest

import arkouda as ak
from arkouda.sorting import SortingAlgorithm

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
            assert (sorted_bi - 2**200).to_list() == sorted_pda.to_list()

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
