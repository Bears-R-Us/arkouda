from itertools import permutations

import numpy as np
import pytest

import arkouda as ak
from arkouda.numpy.sorting import SortingAlgorithm

NUMERIC_TYPES = ["int64", "float64", "uint64", "bigint", "bool"]


def make_ak_arrays(size, dtype, minimum=-(2**32), maximum=2**32, seed=1):
    if dtype in ["int64", "float64"]:
        return ak.randint(minimum, maximum, size=size, dtype=dtype, seed=seed)
    elif dtype == "uint64":
        return ak.cast(ak.randint(minimum, maximum, size=size, seed=seed), dtype)
    elif dtype == "bool":
        return ak.cast(ak.randint(0, 2, size=size, seed=seed), dtype)
    elif dtype == "bigint":
        return ak.cast(ak.randint(minimum, maximum, size=size, seed=seed), dtype) + 2**200
    raise ValueError(f"Unsupported dtype: {dtype}")


def assert_cosorted(arr_list, perm):
    arrays = [arr[perm] for arr in arr_list]
    assert ak.is_cosorted(arrays)


class TestCoargsort:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_coargsort_single_type(self, prob_size, dtype, algo):
        arr_lists = [[make_ak_arrays(prob_size, dtype, 0, 2**63) for _ in range(10)]]

        for exp_set in [[2, 4, 8], [16, 32, 64]]:
            arrs = [make_ak_arrays(prob_size, dtype, 0, 2**e) for e in exp_set[:3]]
            arr_lists.extend(list(permutations(arrs)))

            if exp_set == [16, 32, 64]:
                for exp in exp_set:
                    a, b, c, d = (make_ak_arrays(prob_size, dtype, 0, 2**exp) for _ in range(4))
                    z = ak.zeros(prob_size, dtype=dtype)
                    n = make_ak_arrays(prob_size, dtype, -(2 ** (exp - 1)), 2**exp)
                    arr_lists.extend(
                        [
                            [a],
                            [n],
                            [a, b],
                            [b, a],
                            [a, b, c, d],
                            [z, b, c, d],
                            [z, z, c, d],
                            [z, z, z, d],
                        ]
                    )

        for arr_list in arr_lists:
            perm = ak.coargsort(arr_list, algo)
            assert_cosorted(arr_list, perm)

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("algo", SortingAlgorithm)
    @pytest.mark.parametrize("base_dtype", NUMERIC_TYPES)
    def test_coargsort_mixed_types(self, prob_size, algo, base_dtype):
        np.random.seed(1)
        dtypes = [base_dtype] + np.random.choice(
            list(set(NUMERIC_TYPES) - {base_dtype}),
            size=len(NUMERIC_TYPES) - 1,
            replace=False,
        ).tolist()
        arr_list = [make_ak_arrays(prob_size, d, 0, 2**63) for d in dtypes]
        if arr_list[0].dtype == ak.bigint:
            arr_list[0] = ak.cast(arr_list[0] - 2**200, ak.int64)
        perm = ak.coargsort(arr_list, algo)
        assert_cosorted(arr_list, perm)

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_coargsort_categorical_and_strings(self, algo):
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        cat_ans, str_ans = ["a", "a", "b", "b", "c"], ["b", "b", "a", "a", "c"]

        for cat_list in [
            [cat],
            [cat_from_codes],
            [cat, cat_from_codes],
            *permutations([cat_from_codes, string, cat]),
        ]:
            ans = cat_ans if isinstance(cat_list[0], ak.Categorical) else str_ans
            assert ans == cat_list[0][ak.coargsort(cat_list, algo)].tolist()

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_coargsort_empty_categorical_and_strings(self, algo):
        empty_str = ak.random_strings_uniform(1, 16, 0)
        for empty in [empty_str, ak.Categorical(empty_str)]:
            assert ak.coargsort([empty], algo).tolist() == []

    def test_coargsort_bool(self):
        args = [ak.arange(5) % 2 == 0, ak.arange(5, 0, -1)]
        perm = ak.coargsort(args)
        assert args[0][perm].tolist() == [False, False, True, True, True]
        assert args[1][perm].tolist() == [2, 4, 1, 3, 5]

    @pytest.mark.parametrize("algo", SortingAlgorithm)
    def test_error_handling(self, algo):
        ones, short_ones = ak.ones(100), ak.ones(10)
        with pytest.raises(ValueError):
            ak.coargsort([ones, short_ones], algo)
        with pytest.raises(TypeError):
            ak.coargsort([[0, 1], [1, 2]], algo)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_coargsort_wstrings(self, ascending):
        size = 100
        ak_char_array = ak.random_strings_uniform(1, 2, seed=1, size=size)
        ak_int_array = ak.randint(0, 10 * size, size, dtype=ak.int64)

        perm = ak.coargsort([ak_char_array, ak_int_array], ascending=ascending)
        arrays = [ak_char_array[perm], ak_int_array[perm]]

        #   is_cosargsorted does not work on mixed dtypes
        #   plus, coargsort does not sort strings (yet), it just groups
        boundary = arrays[0][:-1] != arrays[0][1:]
        for array in arrays[1:]:
            left = array[:-1]
            right = array[1:]
            _ = left <= right if ascending else left >= right
            if not ak.all(_ | boundary):
                raise AssertionError("Array not sorted or grouped correctly")
            boundary = boundary | (left != right)

    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize("dtype1", NUMERIC_TYPES)
    @pytest.mark.parametrize("dtype2", NUMERIC_TYPES)
    def test_coargsort_numeric(self, ascending, dtype1, dtype2):
        from arkouda.alignment import is_cosorted
        from arkouda.numpy.manipulation_functions import flip

        size = 100
        array1 = make_ak_arrays(size, dtype1, 0, size)
        array2 = make_ak_arrays(size, dtype2, 0, size)

        perm = ak.coargsort([array1, array2], ascending=ascending)
        arrays = [
            array1[perm] if ascending else flip(array1[perm]),
            array2[perm] if ascending else flip(array2[perm]),
        ]
        assert is_cosorted(arrays)

    def test_coargsort_empty_and_singleton(self):
        empty = ak.array([])
        singleton = ak.array([42])
        assert ak.coargsort([empty]).tolist() == []
        assert ak.coargsort([singleton]).tolist() == [0]
