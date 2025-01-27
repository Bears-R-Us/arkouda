from typing import Optional, Tuple, Union

import numpy as np
import pytest

import arkouda as ak
from arkouda.testing import assert_equal as ak_assert_equal
from arkouda.testing import assert_equivalent as ak_assert_equivalent

SEED = 314159
import numpy

import arkouda.pdarrayclass

REDUCTION_OPS = list(set(ak.pdarrayclass.SUPPORTED_REDUCTION_OPS) - set(["isSorted", "isSortedLocally"]))
INDEX_REDUCTION_OPS = ak.pdarrayclass.SUPPORTED_INDEX_REDUCTION_OPS

DTYPES = ["int64", "float64", "bool", "uint64"]

#   TODO: add unint8 to DTYPES


class TestPdarrayClass:

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reshape(self, dtype):
        a = ak.arange(4, dtype=dtype)
        r = a.reshape((2, 2))
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)
        b = r.reshape(4)
        assert ak.all(a==b)

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_reshape_and_flatten_bug_reproducer(self):
        dtype = "bigint"
        size = 10
        x = ak.arange(size, dtype=dtype).reshape((1, size, 1))
        ak_assert_equal(x.flatten(), ak.arange(size, dtype=dtype))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shape(self, dtype):
        a = ak.arange(4, dtype=dtype)
        np_a = np.arange(4)
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", list(set(DTYPES) - set(["bool"])))
    def test_shape_multidim(self, dtype):
        a = ak.arange(4, dtype=dtype).reshape((2, 2))
        np_a = np.arange(4, dtype=dtype).reshape((2, 2))
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_flatten(self, size, dtype):
        a = ak.arange(size, dtype=dtype)
        ak_assert_equal(a.flatten(), a)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flatten(self, size, dtype):
        size = size - (size % 4)
        a = ak.arange(size, dtype=dtype)
        b = a.reshape((2, 2, size / 4))
        ak_assert_equal(b.flatten(), a)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_is_sorted(self, size, dtype, axis):

        a = ak.arange(size, dtype=dtype)
        assert ak.is_sorted(a, axis=axis)

        b = ak.flip(a)
        assert not ak.is_sorted(b, axis=axis)

        c = ak.randint(0, size // 10, size, seed=SEED)
        assert not ak.is_sorted(c, axis=axis)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("dtype", list(set(DTYPES) - set(["bool"])))
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_is_sorted_multidim(self, dtype, axis):

        a = ak.array(ak.randint(0, 100, (5, 7, 4), dtype=dtype, seed=SEED))
        sorted = ak.is_sorted(a, axis=axis)
        if isinstance(sorted, np.bool_):
            assert not sorted
        else:
            assert ak.all(sorted == False)

        x = ak.arange(40).reshape((2, 10, 2))
        sorted = ak.is_sorted(x, axis=axis)
        if isinstance(sorted, np.bool_):
            assert sorted
        else:
            assert ak.all(sorted)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_is_locally_sorted(self, size, dtype, axis):
        from arkouda.pdarrayclass import is_locally_sorted

        a = ak.arange(size)
        assert is_locally_sorted(a, axis=axis)

        assert not is_locally_sorted(ak.flip(a), axis=axis)

        b = ak.randint(0, size // 10, size)
        assert not is_locally_sorted(b, axis=axis)

    @pytest.mark.skip_if_nl_greater_than(2)
    @pytest.mark.skip_if_nl_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_is_locally_sorted_multi_locale(self, size, dtype):
        from arkouda.pdarrayclass import is_locally_sorted, is_sorted

        size = size // 2
        a = ak.concatenate([ak.arange(size, dtype=dtype), ak.arange(size, dtype=dtype)])
        assert is_locally_sorted(a)
        assert not is_sorted(a)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.skip_if_nl_greater_than(2)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_is_locally_sorted_multidim(self, dtype, axis):
        from arkouda.pdarrayclass import is_locally_sorted

        a = ak.array(ak.randint(0, 100, (20, 20, 20), dtype=dtype, seed=SEED))
        sorted = is_locally_sorted(a, axis=axis)
        if isinstance(sorted, np.bool_):
            assert not sorted
        else:
            assert ak.all(sorted == False)

        x = ak.arange(40).reshape((2, 10, 2))
        sorted = is_locally_sorted(x, axis=axis)
        if isinstance(sorted, np.bool_):
            assert sorted
        else:
            assert ak.all(sorted)

    def assert_reduction_ops_match(
        self, op: str, pda: ak.pdarray, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ):

        ak_op = getattr(arkouda.pdarrayclass, op)
        np_op = getattr(numpy, op)
        nda = pda.to_ndarray()

        ak_result = ak_op(pda, axis=axis)
        ak_assert_equivalent(ak_result, np_op(nda, axis=axis))

    @pytest.mark.parametrize("op", INDEX_REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("arry_gen", [ak.zeros, ak.ones, ak.arange])
    @pytest.mark.parametrize("axis", [0, None])
    def test_index_reduction_1D(self, op, dtype, arry_gen, size, axis):
        pda = arry_gen(size, dtype=dtype)
        ak_op = getattr(arkouda.pdarrayclass, op)
        np_op = getattr(numpy, op)
        nda = pda.to_ndarray()
        ak_result = ak_op(pda, axis=axis)
        ak_assert_equivalent(ak_result, np_op(nda, axis=axis))

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("op", INDEX_REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("arry_gen", [ak.zeros, ak.ones, ak.arange])
    @pytest.mark.parametrize("axis", [0, 1, None])
    def test_index_reduction_multi_dim(self, op, dtype, arry_gen, size, axis):
        size = 10
        pda = arry_gen(size * size * size, dtype=dtype).reshape((size, size, size))
        ak_op = getattr(arkouda.pdarrayclass, op)
        np_op = getattr(numpy, op)
        nda = pda.to_ndarray()
        ak_result = ak_op(pda, axis=axis)
        ak_assert_equivalent(ak_result, np_op(nda, axis=axis))

    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("arry_gen", [ak.zeros, ak.ones, ak.arange])
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_reductions_match_numpy_1D(self, op, size, dtype, arry_gen, axis):
        size = min(size, 100) if op == "prod" else size
        pda = arry_gen(size, dtype=dtype)
        self.assert_reduction_ops_match(op, pda, axis=axis)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("arry_gen", [ak.zeros, ak.ones])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_reductions_match_numpy_3D_zeros(self, op, size, dtype, arry_gen, axis):
        size = 10 if op == "prod" else round(size ** (1.0 / 3))
        pda = arry_gen((size, size, size), dtype=dtype)
        self.assert_reduction_ops_match(op, pda, axis=axis)

    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_reductions_match_numpy_1D_TF(self, op, axis):
        pda = ak.array([True, True, False, True, True, True, True, True])
        self.assert_reduction_ops_match(op, pda, axis=axis)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_reductions_match_numpy_3D_TF(self, op, axis):
        pda = ak.array([True, True, False, True, True, True, True, True]).reshape((2, 2, 2))
        self.assert_reduction_ops_match(op, pda, axis=axis)
