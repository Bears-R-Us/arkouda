import numpy as np
import pytest

import arkouda as ak

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

    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_argmin_and_argmax(self, dtype):
        np_arr = make_np_arrays(1000, dtype)
        ak_arr = ak.array(np_arr)

        assert np_arr.argmin() == ak_arr.argmin()
        assert np_arr.argmax() == ak_arr.argmax()

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
