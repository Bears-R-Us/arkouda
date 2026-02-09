import numpy as np
import pytest

import arkouda as ak

from arkouda.testing import assert_arkouda_array_equivalent


NUMERIC_TYPES = ["int64", "uint64", "float64", "bool"]


def make_np_arrays(size, dtype):
    if dtype == "int64":
        return np.random.randint(-(2**32), 2**32, size=size, dtype=dtype)
    elif dtype == "uint64":
        return ak.cast(ak.randint(-(2**32), 2**32, size=size), dtype)
    elif dtype == "float64":
        return np.random.uniform(-(2**32), 2**32, size=size)
    elif dtype == "bool":
        return np.random.randint(0, 1, size=size, dtype=dtype)
    return None


def make_np_edge_cases(dtype):
    if dtype == "int64":
        return np.array([np.iinfo(np.int64).min, -1, 0, 3, np.iinfo(np.int64).max])
    elif dtype == "uint64":
        return np.array([17, 2**64 - 1, 0, 3, 2**63 + 10], dtype=np.uint64)
    elif dtype == "float64":
        return np.array(
            [
                np.nan,
                -np.inf,
                np.finfo(np.float64).min,
                -3.14,
                0.0,
                3.14,
                8,
                np.finfo(np.float64).max,
                np.inf,
                np.nan,
            ]
        )
    return None


@pytest.mark.requires_chapel_module("KExtremeMsg")
class TestExtrema:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", ["int64", "uint64", "float64"])
    def test_extrema(self, prob_size, dtype):
        pda = ak.array(make_np_arrays(prob_size, dtype))
        ak_sorted = ak.sort(pda)
        K = prob_size // 2

        # compare minimums against first K elements from sorted array
        assert (ak.mink(pda, K) == ak_sorted[:K]).all()
        assert (pda[ak.argmink(pda, K)] == ak_sorted[:K]).all()

        # compare maximums against last K elements from sorted array
        assert (ak.maxk(pda, K) == ak_sorted[-K:]).all()
        assert (pda[ak.argmaxk(pda, K)] == ak_sorted[-K:]).all()

    @pytest.mark.parametrize("dtype", ["int64", "uint64", "float64"])
    def test_extrema_edge_cases(self, dtype):
        edge_cases = make_np_edge_cases(dtype)
        size = edge_cases.size // 2
        # Due to #2754, we need to have replacement off to avoid all values = min/max(dtype)
        npa = np.random.choice(edge_cases, edge_cases.size // 2, replace=False)
        pda = ak.array(npa)
        K = size // 2
        np_sorted = np.sort(npa)

        # extremas ignore nans
        non_nan_sorted = np_sorted[~np.isnan(np_sorted)]

        if non_nan_sorted.size >= K:
            # compare minimums against first K elements from sorted array
            assert np.allclose(ak.mink(pda, K).to_ndarray(), non_nan_sorted[:K], equal_nan=True)
            # check for -1s to avoid oob due to #2754
            arg_min_k = ak.argmink(pda, K)
            if (arg_min_k != -1).all():
                assert np.allclose(pda[arg_min_k].to_ndarray(), non_nan_sorted[:K], equal_nan=True)

            # compare maximums against last K elements from sorted array
            assert np.allclose(ak.maxk(pda, K).to_ndarray(), non_nan_sorted[-K:], equal_nan=True)
            # check for -1s to avoid oob due to #2754
            arg_max_k = ak.argmaxk(pda, K)
            if (arg_max_k != -1).all():
                assert np.allclose(pda[arg_max_k].to_ndarray(), non_nan_sorted[-K:], equal_nan=True)

    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_argmin_and_argmax(self, dtype):
        np_arr = make_np_arrays(1000, dtype)
        ak_arr = ak.array(np_arr)

        assert np_arr.argmin() == ak_arr.argmin()
        assert np_arr.argmax() == ak_arr.argmax()

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_argmin_and_argmax_2dim(self, dtype):
        np_arr = make_np_arrays(1000, dtype).reshape(10, 100)
        ak_arr = ak.array(np_arr)

        for axis in range(-2, 2):
            assert_arkouda_array_equivalent(ak_arr.argmin(axis=axis), np_arr.argmin(axis=axis))
            assert_arkouda_array_equivalent(ak_arr.argmax(axis=axis), np_arr.argmax(axis=axis))

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_argmin_and_argmax_3dim(self, dtype):
        np_arr = make_np_arrays(1000, dtype).reshape(10, 10, 10)
        ak_arr = ak.array(np_arr)

        for axis in range(-3, 3):
            assert_arkouda_array_equivalent(ak_arr.argmin(axis=axis), np_arr.argmin(axis=axis))
            assert_arkouda_array_equivalent(ak_arr.argmax(axis=axis), np_arr.argmax(axis=axis))

    def test_error_handling(self):
        test_array = ak.randint(0, 100, 100)
        for op in ak.mink, ak.maxk, ak.argmink, ak.argmaxk:
            with pytest.raises(TypeError):
                op(list(range(10)), 1)

            with pytest.raises(TypeError):
                op(test_array, "1")

            with pytest.raises(ValueError):
                op(test_array, -1)

            with pytest.raises(ValueError):
                op(ak.array([]), 1)
