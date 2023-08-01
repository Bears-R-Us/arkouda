from itertools import permutations

import pytest

import arkouda as ak
from arkouda.sorting import SortingAlgorithm

NUMERIC_AND_BIGINT_TYPES = ["int64", "float64", "uint64", "bigint"]


def make_ak_arrays(size, dtype, minimum=-(2**32), maximum=2**32):
    if dtype in ["int64", "float64"]:
        # randint for float is equivalent to uniform
        return ak.randint(minimum, maximum, size=size, dtype=dtype)
    elif dtype == "uint64":
        return ak.cast(ak.randint(minimum, maximum, size=size), dtype)
    elif dtype == "bigint":
        return ak.cast(ak.randint(minimum, maximum, size=size), dtype) + 2**200
    return None


class TestCoargsort:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_AND_BIGINT_TYPES)
    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_coargsort_single_type(self, prob_size, dtype, algo):
        # add test for large number of arrays and all permutations of mixed sizes
        arr_lists = [
            [make_ak_arrays(prob_size, dtype, 0, 2**63) for _ in range(10)],
            *permutations(make_ak_arrays(prob_size, dtype, 0, 2**exp) for exp in (2, 4, 8)),
        ]

        for exp in 8 * 2, 8 * 4, 8 * 8:
            a, b, c, d = (make_ak_arrays(prob_size, dtype, 0, 2**exp) for _ in range(4))
            z = ak.zeros(prob_size, dtype=dtype)
            n = make_ak_arrays(prob_size, dtype, -(2 ** (exp - 1)), 2**exp)
            arr_lists.extend(
                [[a], [n], [a, b], [b, a], [a, b, c, d], [z, b, c, d], [z, z, c, d], [z, z, z, d]]
            )

        for arr_list in arr_lists:
            if not isinstance(arr_list, list):
                arr_list = list(arr_list)
            perm = ak.coargsort(arr_list, algo)
            # TODO remove once ak.is_sorted is implemented for bigint
            if dtype == "bigint":
                # shift it down for is_cosorted check
                arr_list[0] = ak.cast(arr_list[0] - 2**200, ak.int64)
            assert ak.is_cosorted([arr[perm] for arr in arr_list])

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_coargsort_mixed_types(self, prob_size, algo):
        for arr_list in permutations(
            make_ak_arrays(prob_size, dt, 0, 2**63) for dt in NUMERIC_AND_BIGINT_TYPES
        ):
            if not isinstance(arr_list, list):
                arr_list = list(arr_list)
            perm = ak.coargsort(arr_list, algo)
            # TODO remove once ak.is_sorted is implemented for bigint
            if arr_list[0].dtype == ak.bigint:
                # shift it down for is_cosorted check
                arr_list[0] = ak.cast(arr_list[0] - 2**200, ak.int64)
            assert ak.is_cosorted([arr[perm] for arr in arr_list])

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_coargsort_categorical_and_strings(self, algo):
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        # coargsort sorts using codes, the order isn't guaranteed, only grouping
        cat_ans, str_ans = ["a", "a", "b", "b", "c"], ["b", "b", "a", "a", "c"]

        for cat_list in [
            [cat],
            [cat_from_codes],
            [cat, cat_from_codes],
            *permutations([cat_from_codes, string, cat]),
        ]:
            ans = cat_ans if isinstance(cat_list[0], ak.Categorical) else str_ans
            assert ans == cat_list[0][ak.coargsort(cat_list, algo)].to_list()

        empty_str = ak.random_strings_uniform(1, 16, 0)
        for empty in empty_str, ak.Categorical(empty_str):
            assert [] == ak.coargsort([empty], algo).to_list()

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_error_handling(self, algo):
        ones, short_ones = ak.ones(100), ak.ones(10)

        with pytest.raises(ValueError):
            ak.coargsort([ones, short_ones], algo)

        with pytest.raises(TypeError):
            ak.coargsort([list(range(0, 10)), [0]], algo)
