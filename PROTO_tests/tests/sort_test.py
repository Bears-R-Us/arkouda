import numpy as np
import pytest

import arkouda as ak
from arkouda.sorting import SortingAlgorithm

NUMERIC_AND_BIGINT_TYPES = ["int64", "float64", "uint64", "bigint"]


def make_ak_arrays(dtype):
    if dtype in ["int64", "uint64"]:
        return ak.randint(0, 100, 1000, dtype)
    elif dtype == "float64":
        return ak.array(np.random.rand(100) * 10000)
    elif dtype == "bigint":
        return ak.randint(0, 100, 1000, dtype=ak.uint64)
    return None


class TestSort:
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
        ak_bools = ak.randint(0, 1, 1000, dtype=ak.bool)
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
