from typing import Optional, Tuple, Union

import numpy as np
import pytest

import arkouda as ak

from arkouda.core.client import get_array_ranks, get_max_array_rank
from arkouda.numpy.dtypes import bigint, uint8
from arkouda.testing import assert_almost_equivalent as ak_assert_almost_equivalent
from arkouda.testing import assert_arkouda_array_equivalent, assert_equivalent
from arkouda.testing import assert_equal as ak_assert_equal
from arkouda.testing import assert_equivalent as ak_assert_equivalent


seed = 314159  # this hardcoded seed is retained because the sorted tests
# require known results


REDUCTION_OPS = list(
    set(ak.numpy.pdarrayclass.SUPPORTED_REDUCTION_OPS) - set(["isSorted", "isSortedLocally"])
)
INDEX_REDUCTION_OPS = ak.numpy.pdarrayclass.SUPPORTED_INDEX_REDUCTION_OPS

SHAPES = [(1,), (25,), (5, 10), (10, 5)]
DTYPES = ["int64", "float64", "bool", "uint64"]
NUMERIC_TYPES = [ak.int64, ak.float64, ak.bool_, ak.uint64]
NUMERIC_TYPES_NO_BOOL = [ak.int64, ak.float64, ak.uint64]


def _singleton_bigint(init=0):
    # Guaranteed length-1 bigint array, then set initial value.
    a = ak.arange(1, dtype=ak.bigint)
    a[0] = init
    return a


def _detect_limb_params():
    """
    Detect limb_bytes (4 or 8) and limb_bits by watching nbytes jumps.
    Creates a length-1 bigint safely (no empty array surprises).
    """
    import math

    a = _singleton_bigint(0)

    prev = int(a.nbytes)
    deltas = []

    # bump value and watch for capacity jumps
    for e in range(1, 1025):
        a[0] = 1 << e
        nb = int(a.nbytes)
        if nb > prev:
            deltas.append(nb - prev)
            prev = nb
            if len(deltas) >= 3:
                break

    if not deltas:
        pytest.skip("Could not detect limb parameters from nbytes deltas")

    limb_bytes = deltas[0]
    for d in deltas[1:]:
        limb_bytes = math.gcd(limb_bytes, d)
    assert limb_bytes in (4, 8), f"Unexpected limb size: {limb_bytes}"

    limb_bits = limb_bytes * 8
    return limb_bytes, limb_bits


def _limbs_needed(val: int, limb_bits: int) -> int:
    """Number of limbs required for the value (used, not capacity)."""
    if val == 0:
        return 1
    return (val.bit_length() + limb_bits - 1) // limb_bits


#   TODO: add uint8 to DTYPES


class TestPdarrayClass:
    @pytest.mark.requires_chapel_module(["StatsMsg", "LinalgMsg", "KExtremeMsg"])
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_pdarrayclass_docstrings(self):
        import doctest

        from arkouda.numpy import pdarrayclass

        result = doctest.testmod(
            pdarrayclass, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reshape(self, dtype):
        a = ak.arange(4, dtype=dtype)
        r = a.reshape(2, 2)  # test sequence
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)
        b = r.reshape(4)  # test integer
        assert ak.all(a == b)
        r = a.reshape((2, 2))  # test tuple
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)
        b = r.reshape(4)
        assert ak.all(a == b)
        r = a.reshape(np.array([2, 2]))  # test ndarray
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)
        b = r.reshape(4)
        assert ak.all(a == b)
        r = a.reshape(ak.array([2, 2]))  # test pdarray
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)
        b = r.reshape(4)
        assert ak.all(a == b)

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_reshape_and_flatten_bug_reproducer(self):
        dtype = "bigint"
        size = 10
        x = ak.arange(size, dtype=dtype).reshape((1, size, 1))
        ak_assert_equal(x.flatten(), ak.arange(size, dtype=dtype))

    def test_reshape_bigint_bug_4165_reproducer(self):
        x = ak.arange(10, dtype=bigint)
        x.max_bits = 64
        assert x.max_bits == 64
        y = x.reshape((10,))
        assert y.max_bits == 64

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
    def test_flatten_multidim(self, size, dtype):
        size = size - (size % 4)
        a = ak.arange(size, dtype=dtype)
        b = a.reshape((2, 2, size // 4))
        ak_assert_equal(b.flatten(), a)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_is_sorted(self, size, dtype, axis):
        a = ak.arange(size, dtype=dtype)
        assert ak.is_sorted(a, axis=axis)

        b = ak.flip(a)
        assert not ak.is_sorted(b, axis=axis)

        if size > 99:
            c = ak.randint(0, size // 10, size, seed=seed)
            assert not ak.is_sorted(c, axis=axis)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("dtype", list(set(DTYPES) - set(["bool"])))
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_is_sorted_multidim(self, dtype, axis):
        a = ak.array(ak.randint(0, 100, (5, 7, 4), dtype=dtype, seed=seed))
        sorted = ak.is_sorted(a, axis=axis)
        if isinstance(sorted, np.bool_):
            assert sorted is np.False_
        else:
            assert ak.all(sorted == False)  # noqa: E712

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
        from arkouda.numpy.pdarrayclass import is_locally_sorted

        a = ak.arange(size)
        assert is_locally_sorted(a, axis=axis)

        assert not is_locally_sorted(ak.flip(a), axis=axis)

        if size > 99:
            b = ak.randint(0, size // 10, size, seed=pytest.seed)
            assert not is_locally_sorted(b, axis=axis)

    @pytest.mark.skip_if_nl_greater_than(2)
    @pytest.mark.skip_if_nl_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_is_locally_sorted_multi_locale(self, size, dtype):
        from arkouda.numpy.pdarrayclass import is_locally_sorted, is_sorted

        size = size // 2
        a = ak.concatenate([ak.arange(size, dtype=dtype), ak.arange(size, dtype=dtype)])
        assert is_locally_sorted(a)
        assert not is_sorted(a)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.skip_if_nl_greater_than(2)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_is_locally_sorted_multidim(self, dtype, axis):
        from arkouda.numpy.pdarrayclass import is_locally_sorted

        high = 100 if dtype != "bool" else 2

        a = ak.array(ak.randint(0, high, (20, 20, 20), dtype=dtype, seed=seed))
        sorted = is_locally_sorted(a, axis=axis)
        if isinstance(sorted, np.bool_):
            assert sorted is np.False_
        else:
            assert ak.all(sorted == False)  # noqa: E712

        x = ak.arange(40).reshape((2, 10, 2))
        sorted = is_locally_sorted(x, axis=axis)
        if isinstance(sorted, np.bool_):
            assert sorted
        else:
            assert ak.all(sorted)

    def assert_reduction_ops_match(
        self,
        op: str,
        pda: ak.pdarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        ak_op = getattr(ak.pdarrayclass, op)
        np_op = getattr(np, op)
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
        ak_op = getattr(ak.pdarrayclass, op)
        np_op = getattr(np, op)
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
        ak_op = getattr(ak.pdarrayclass, op)
        np_op = getattr(np, op)
        nda = pda.to_ndarray()
        ak_result = ak_op(pda, axis=axis)
        ak_assert_equivalent(ak_result, np_op(nda, axis=axis))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("arry_gen", [ak.zeros, ak.ones, ak.arange])
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_reductions_match_numpy_1D(self, op, size, dtype, arry_gen, axis):
        size = min(size, 100) if op == "prod" else size
        pda = arry_gen(size, dtype=dtype)
        self.assert_reduction_ops_match(op, pda, axis=axis)

    @pytest.mark.requires_chapel_module("StatsMsg")
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

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("axis", [0, (0,), None])
    def test_reductions_match_numpy_1D_TF(self, op, axis):
        pda = ak.array([True, True, False, True, True, True, True, True])
        self.assert_reduction_ops_match(op, pda, axis=axis)

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("op", REDUCTION_OPS)
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (0, 1, 2)])
    def test_reductions_match_numpy_3D_TF(self, op, axis):
        pda = ak.array([True, True, False, True, True, True, True, True]).reshape((2, 2, 2))
        self.assert_reduction_ops_match(op, pda, axis=axis)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype1", NUMERIC_TYPES)
    @pytest.mark.parametrize("dtype2", NUMERIC_TYPES)
    def test_dot(self, size, dtype1, dtype2):
        #   two 1D vectors

        nda1 = np.random.randint(0, 2, size).astype(dtype1)
        nda2 = np.random.randint(0, 2, size).astype(dtype2)
        pda1 = ak.array(nda1)
        pda2 = ak.array(nda2)
        assert ak.dot(pda1, pda2) == np.dot(nda1, nda2)  # results are scalar

        #   one 1D vector and one scalar

        factor = 5
        assert_arkouda_array_equivalent(ak.dot(pda1, factor), np.dot(nda1, factor))
        assert_arkouda_array_equivalent(ak.dot(factor, pda2), np.dot(factor, nda2))

    @pytest.mark.requires_chapel_module(["StatsMsg", "LinalgMsg"])
    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype1", NUMERIC_TYPES)
    @pytest.mark.parametrize("dtype2", NUMERIC_TYPES)
    def test_dot_2D(self, size, dtype1, dtype2):
        #   two 2D arrays

        right = size // 2
        nda1 = np.arange(2 * right).astype(dtype1).reshape(2, right)
        nda2 = np.arange(2 * right).astype(dtype2).reshape(right, 2)
        pda1 = ak.array(nda1)
        pda2 = ak.array(nda2)
        assert_arkouda_array_equivalent(ak.dot(pda1, pda2), np.dot(nda1, nda2))

    @pytest.mark.requires_chapel_module(["StatsMsg", "LinalgMsg"])
    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype1", NUMERIC_TYPES)
    @pytest.mark.parametrize("dtype2", NUMERIC_TYPES)
    def test_dot_multi_dim(self, size, dtype1, dtype2):
        mrank = get_max_array_rank()

        #   one max rank array and one 1D array

        if mrank > 1:
            mshape = (mrank - 1) * [2]
            fsize = size // (2 ** (mrank - 1))
            mshape.append(fsize)
            msize = 2 ** (mrank - 1) * fsize
            nda1 = np.arange(msize).astype(dtype1).reshape(tuple(mshape))
            nda2 = np.arange(fsize).astype(dtype2)
            pda1 = ak.array(nda1)
            pda2 = ak.array(nda2)
            assert_arkouda_array_equivalent(ak.dot(pda1, pda2), np.dot(nda1, nda2))

        #   one max rank-1 array, and one 2D array

        if mrank > 2 and 2 in get_array_ranks() and mrank - 1 in get_array_ranks():
            mshape = (mrank - 2) * [2]
            fsize = size // (2 ** (mrank - 2))
            mshape.append(fsize)
            msize = 2 ** (mrank - 2) * fsize
            nda1 = np.arange(msize).astype(dtype1).reshape(tuple(mshape))
            nda2 = np.arange(2 * fsize).astype(dtype2).reshape((fsize, 2))
            pda1 = ak.array(nda1)
            pda2 = ak.array(nda2)
            assert_arkouda_array_equivalent(ak.dot(pda1, pda2), np.dot(nda1, nda2))

    @pytest.mark.requires_chapel_module("UtilMsg")
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_diff_1d(self, dtype, size):
        if dtype == "bool":
            a = ak.randint(0, 2, size, dtype=dtype, seed=seed)
        else:
            a = ak.randint(0, 100, size, dtype=dtype, seed=seed)
        anp = a.to_ndarray()

        a_d = ak.diff(a, n=1)
        anp_d = np.diff(anp, n=1)
        assert_arkouda_array_equivalent(a_d, anp_d)

        a_d = ak.diff(a, n=5)
        anp_d = np.diff(anp, n=5)
        assert_arkouda_array_equivalent(a_d, anp_d)

    @pytest.mark.requires_chapel_module("UtilMsg")
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    @pytest.mark.parametrize("n", [1, 2])
    def test_diff_multidim(self, dtype, axis, n):
        if dtype == "bool":
            a = ak.randint(0, 2, (5, 6, 7), dtype=dtype, seed=seed)
        else:
            a = ak.randint(0, 100, (5, 6, 7), dtype=dtype, seed=seed)
        anp = a.to_ndarray()

        if axis is not None:
            a_d = ak.diff(a, n=n, axis=axis)
            anp_d = np.diff(anp, n=n, axis=axis)
        else:
            a_d = ak.diff(a, n=n)
            anp_d = np.diff(anp, n=n)
        assert_arkouda_array_equivalent(a_d, anp_d)

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_mean_1D(self, size, dtype):
        nda = np.random.randint(0, size, size).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.mean(nda), pda.mean())
        ak.assert_almost_equivalent(np.mean(nda), ak.mean(pda))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_mean_2D(self, size, dtype):
        nda = np.random.randint(0, size, (2, size // 2)).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.mean(nda), pda.mean())
        ak.assert_almost_equivalent(np.mean(nda), ak.mean(pda))
        for axis in range(-2, 2):
            ak_assert_almost_equivalent(np.mean(nda, axis=axis), pda.mean(axis=axis))
            ak.assert_almost_equivalent(np.mean(nda, axis=axis), ak.mean(pda, axis=axis))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_mean_3D(self, size, dtype):
        nda = np.random.randint(0, size, (2, 2, size // 4)).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.mean(nda), pda.mean())
        ak.assert_almost_equivalent(np.mean(nda), ak.mean(pda))
        for axis in range(-3, 3):
            ak_assert_almost_equivalent(np.mean(nda, axis=axis), pda.mean(axis=axis))
            ak.assert_almost_equivalent(np.mean(nda, axis=axis), ak.mean(pda, axis=axis))
        for axis in [(0, 1), (0, 2), (1, 2), (-3, -2), (-3, -1), (-2, -1)]:
            ak_assert_almost_equivalent(np.mean(nda, axis=axis), pda.mean(axis=axis))
            ak.assert_almost_equivalent(np.mean(nda, axis=axis), ak.mean(pda, axis=axis))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_var_1D(self, size, dtype):
        nda = np.random.randint(0, size, size).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.var(nda), pda.var())
        ak.assert_almost_equivalent(np.var(nda), ak.var(pda))
        ak_assert_almost_equivalent(np.var(nda, ddof=1), pda.var(ddof=1))
        ak.assert_almost_equivalent(np.var(nda, ddof=1), ak.var(pda, ddof=1))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_var_2D(self, size, dtype):
        nda = np.random.randint(0, size, (2, size // 2)).astype(dtype)
        pda = ak.array(nda)
        ak.assert_almost_equivalent(np.var(nda), pda.var())
        ak.assert_almost_equivalent(np.var(nda), ak.var(pda))
        ak.assert_almost_equivalent(np.var(nda, ddof=1), pda.var(ddof=1))
        ak.assert_almost_equivalent(np.var(nda, ddof=1), ak.var(pda, ddof=1))
        for axis in range(2):
            ak_assert_almost_equivalent(np.var(nda, axis=axis, ddof=1), pda.var(axis=axis, ddof=1))
            ak.assert_almost_equivalent(np.var(nda, axis=axis, ddof=1), ak.var(pda, axis=axis, ddof=1))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_var_3D(self, size, dtype):
        nda = np.random.randint(0, size, (2, 2, size // 4)).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.var(nda), pda.var())
        ak.assert_almost_equivalent(np.var(nda), ak.var(pda))
        ak_assert_almost_equivalent(np.var(nda, ddof=1), pda.var(ddof=1))
        ak.assert_almost_equivalent(np.var(nda, ddof=1), ak.var(pda, ddof=1))
        for axis in range(2):
            ak_assert_almost_equivalent(np.var(nda, axis=axis, ddof=1), pda.var(axis=axis, ddof=1))
            ak.assert_almost_equivalent(np.var(nda, axis=axis, ddof=1), ak.var(pda, axis=axis, ddof=1))
        for axis in [(0, 1), (0, 2), (1, 2), (-3, -2), (-3, -1), (-2, -1)]:
            ak_assert_almost_equivalent(np.var(nda, ddof=1, axis=axis), pda.var(ddof=1, axis=axis))
            ak.assert_almost_equivalent(np.var(nda, ddof=1, axis=axis), ak.var(pda, ddof=1, axis=axis))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_std_1D(self, size, dtype):
        nda = np.random.randint(0, size, size).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.std(nda), pda.std())
        ak.assert_almost_equivalent(np.std(nda), ak.std(pda))
        ak_assert_almost_equivalent(np.std(nda, ddof=1), pda.std(ddof=1))
        ak.assert_almost_equivalent(np.std(nda, ddof=1), ak.std(pda, ddof=1))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_std_2D(self, size, dtype):
        nda = np.random.randint(0, size, (2, size // 2)).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.std(nda), pda.std())
        ak.assert_almost_equivalent(np.std(nda), ak.std(pda))
        ak_assert_almost_equivalent(np.std(nda, ddof=1), pda.std(ddof=1))
        ak.assert_almost_equivalent(np.std(nda, ddof=1), ak.std(pda, ddof=1))
        for axis in range(2):
            ak_assert_almost_equivalent(np.std(nda, axis=axis, ddof=1), pda.std(axis=axis, ddof=1))
            ak.assert_almost_equivalent(np.std(nda, axis=axis, ddof=1), ak.std(pda, axis=axis, ddof=1))

    @pytest.mark.requires_chapel_module("StatsMsg")
    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES_NO_BOOL)
    def test_std_3D(self, size, dtype):
        nda = np.random.randint(0, size, (2, 2, size // 4)).astype(dtype)
        pda = ak.array(nda)
        ak_assert_almost_equivalent(np.std(nda), pda.std())
        ak.assert_almost_equivalent(np.std(nda), ak.std(pda))
        ak_assert_almost_equivalent(np.std(nda, ddof=1), pda.std(ddof=1))
        ak.assert_almost_equivalent(np.std(nda, ddof=1), ak.std(pda, ddof=1))
        for axis in range(2):
            ak_assert_almost_equivalent(np.std(nda, axis=axis, ddof=1), pda.std(axis=axis, ddof=1))
            ak.assert_almost_equivalent(np.std(nda, axis=axis, ddof=1), ak.std(pda, axis=axis, ddof=1))
        for axis in [(0, 1), (0, 2), (1, 2), (-3, -2), (-3, -1), (-2, -1)]:
            ak_assert_almost_equivalent(np.std(nda, ddof=1, axis=axis), pda.std(ddof=1, axis=axis))
            ak.assert_almost_equivalent(np.std(nda, ddof=1, axis=axis), ak.std(pda, ddof=1, axis=axis))

    @pytest.mark.parametrize(
        "data,expected",
        [
            ([42, 7, 19], [1, 2, 0]),
            ([3, 3, 1], [2, 0, 1]),  # test for duplicates
            ([], []),  # empty array
        ],
    )
    def test_argsort_default(self, data, expected):
        a = ak.array(data)
        result = a.argsort()
        assert result.tolist() == expected
        assert a[result].tolist() == sorted(data)

    def test_argsort_descending(self):
        a = ak.array([42, 7, 19])
        result = a.argsort(ascending=False)
        assert result.tolist() == [0, 2, 1]
        assert a[result].tolist() == sorted([42, 7, 19], reverse=True)

    def test_argsort_bigint(self):
        a = ak.array([2**70, 1, 2**69], dtype=ak.bigint)
        result = a.argsort()
        sorted_vals = a[result]
        expected = sorted([2**70, 1, 2**69])
        assert sorted_vals.tolist() == expected

    def test_argsort_invalid_axis(self):
        a = ak.array([1, 2, 3])
        with pytest.raises(IndexError):
            a.argsort(axis=2)

    def test_argsort_axis_minus1(self):
        a = ak.array([5, 3, 4])
        result = a.argsort(axis=-1)
        assert a[result].tolist() == [3, 4, 5]

    def test_argsort_empty_array(self):
        a = ak.array([], dtype=ak.int64)
        result = a.argsort()
        assert result.size == 0

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_argsort_multidim_ignored_axis(self):
        a = ak.arange(6).reshape((2, 3))
        result = a.argsort()  # axis default = 0
        assert isinstance(result, ak.pdarray)

    def test_argsort_algorithm_enum(self):
        from arkouda.numpy.sorting import SortingAlgorithm

        a = ak.array([4, 1, 3])
        result = a.argsort(algorithm=SortingAlgorithm.RadixSortLSD)
        assert a[result].tolist() == sorted([4, 1, 3])

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize("dtype", [ak.int64, ak.float64, ak.bool_, ak.uint64, ak.bigint])
    def test_argsort_random(self, size, ascending, dtype):
        high = 1_000_000 if dtype != ak.bool_ else 2
        a = ak.randint(0, high, size, dtype=dtype if dtype != ak.bigint else ak.int64, seed=seed)
        if dtype == ak.bigint:
            a = a + 2**70
        perm = a.argsort(ascending=ascending)
        sorted_vals = a[perm]

        if ascending:
            assert ak.all(sorted_vals[1:] >= sorted_vals[:-1])
        else:
            assert ak.all(sorted_vals[1:] <= sorted_vals[:-1])

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ak.ScalarDTypes)
    @pytest.mark.parametrize("ascending", [True, False])
    def test_argsort_numpy_alignment(self, shape, dtype, ascending):
        high = 100 if dtype != "bool_" else 2
        for axis in range(len(shape)):
            a = ak.randint(0, high, shape, dtype=dtype, seed=seed)
            b = a.argsort(axis=axis, ascending=ascending)
            np_b = a.to_ndarray().argsort(axis=axis, stable=True)
            np_b = np.flip(np_b, axis=axis) if not ascending else np_b

            assert b.size == a.size
            assert b.ndim == a.ndim
            assert b.shape == a.shape

            assert_equivalent(b, np_b)

    @pytest.mark.parametrize("dtype", DTYPES + [uint8])
    def test_copy(self, dtype):
        fixed_size = 100
        a = ak.arange(fixed_size, dtype=dtype)
        a_cpy = a.copy()

        assert a_cpy is not a
        ak_assert_equal(a, a_cpy)

    @pytest.mark.skip_if_max_rank_less_than(3)
    @pytest.mark.parametrize("dtype", DTYPES + [uint8])
    def test_copy_multidim(self, dtype):
        a = ak.arange(1000, dtype=dtype).reshape((10, 10, 10))
        a_cpy = a.copy()

        assert a_cpy is not a
        ak_assert_equal(a, a_cpy)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize(
        "dtype",
        [
            ak.numpy.dtypes.uint8,
            ak.numpy.dtypes.uint64,
            ak.numpy.dtypes.int64,
            ak.numpy.dtypes.float64,
            ak.numpy.dtypes.bool_,
        ],
    )
    def test_nbytes(self, size, dtype):
        a = ak.array(ak.arange(size), dtype=dtype)
        assert a.nbytes == size * ak.dtype(dtype).itemsize

    def test_nbytes_str(self):
        a = ak.array(["a", "b", "c"])
        c = ak.Categorical(a)
        assert c.nbytes == 82

    def test_bigint_nbytes_does_not_shrink_on_zero_assignment(self):
        limb_bytes, limb_bits = _detect_limb_params()

        N = 64
        four_limb_val = 1 << (3 * limb_bits)  # needs 4 limbs
        a = ak.array([four_limb_val] * N, dtype=ak.bigint)

        before = int(a.nbytes)
        a[0] = 0
        after = int(a.nbytes)

        # Capacity does not auto-shrink
        assert after >= before

    def test_bigint_nbytes_grows_by_integral_limb_chunks(self):
        limb_bytes, limb_bits = _detect_limb_params()

        two_limb_val = 1 << (1 * limb_bits)  # ~2 limbs
        five_limb_val = 1 << (4 * limb_bits)  # ~5 limbs

        a = ak.array([two_limb_val], dtype=ak.bigint)
        base = int(a.nbytes)

        a[0] = five_limb_val
        grown = int(a.nbytes)
        delta = grown - base

        assert delta % limb_bytes == 0
        assert delta >= 3 * limb_bytes  # grew by at least 3 limbs

    def test_bigint_nbytes_delta_matches_limbs_times_elements(self):
        limb_bytes, limb_bits = _detect_limb_params()

        N = 128
        two_limb_val = 1 << (1 * limb_bits)  # 2 limbs
        four_limb_val = 1 << (3 * limb_bits)  # 4 limbs

        a = ak.array([four_limb_val] * N, dtype=ak.bigint)
        b = ak.array([two_limb_val] * N, dtype=ak.bigint)

        bytes_a = int(a.nbytes)
        bytes_b = int(b.nbytes)

        expected_delta = (4 - 2) * N * limb_bytes
        assert (bytes_a - bytes_b) == expected_delta

    def test_invert_bool(self):
        a = ak.array([True, False, True])
        got = ~a
        exp = ak.array([False, True, False])
        assert (got == exp).all()

    def test_invert_int64_matches_numpy_semantics(self):
        a = ak.array([0, 1, 2, -1], dtype=ak.int64)
        got = ~a
        # NumPy semantics: ~x == -x - 1 for signed integers
        exp = (-a) - 1
        assert (got == exp).all()

    def test_invert_uint64_matches_numpy_semantics(self):
        a = ak.array([0, 1, 2, 2**63], dtype=ak.uint64)
        got = ~a
        # For unsigned ints: ~x == max_uint - x
        max_u64 = (2**64) - 1
        exp = ak.array([max_u64, max_u64 - 1, max_u64 - 2, max_u64 - (2**63)], dtype=ak.uint64)
        assert (got == exp).all()

    def test_invert_float_raises(self):
        a = ak.array([0.0, 1.0], dtype=ak.float64)
        with pytest.raises(TypeError):
            _ = ~a

    def test_invert_strings_raises(self):
        s = ak.array(["a", "b"])
        with pytest.raises(TypeError):
            _ = ~s

    def test_invert_bigint_roundtrip_small_values(self):
        """Invert not yet supported for bigint."""
        a = ak.array([0, 1, 2, 123456789], dtype=ak.bigint)

        with pytest.raises(TypeError):
            _ = ~a

    @pytest.mark.parametrize(
        "vals",
        [
            [0, 1, -2, 3],
            [-1, -2, -3, 0],
            [5, 0, 7, 8],
        ],
    )
    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    def test_abs_int64_float64(self, vals, dtype):
        a = ak.array(vals, dtype=dtype)
        got = abs(a)
        exp = ak.where(a < 0, -a, a)
        assert (got == exp).all()

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize(
        "vals",
        [
            [[0, 1], [-2, 3]],
            [[-1, -2], [-3, 0]],
            [[5, 0], [7, 8]],
        ],
    )
    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    def test_abs_int64_float64_multidim(self, vals, dtype):
        a = ak.array(vals, dtype=dtype)
        got = abs(a)
        exp = ak.where(a < 0, -a, a)
        assert (got == exp).all()

    def test_abs_uint64_is_identity(self):
        a = ak.array([0, 1, 2, 2**63], dtype="uint64")
        got = abs(a)
        assert (got == a).all()

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_abs_uint64_is_identity_multidim(self):
        a = ak.array([[0, 1], [2, 2**63]], dtype="uint64")
        got = abs(a)
        assert (got == a).all()

    def test_abs_bool_is_identity(self):
        a = ak.array([True, False, True])
        got = abs(a)
        assert (got == a).all()

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_abs_bool_is_identity_multidim(self):
        a = ak.array([[True, False], [True, False]])
        got = abs(a)
        assert (got == a).all()

    def test_abs_bigint(self):
        a = ak.array([0, 1, -2, 3, -123456789], dtype=ak.bigint)
        got = abs(a)

        #   where is not implemented for bigint, so cast to int64
        a = a.astype("int64")
        got = got.astype("int64")

        exp = ak.where(a < 0, -a, a)
        assert (got == exp).all()

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_abs_bigint_multidim(self):
        a = ak.array([[0, 1], [3, -123456789]], dtype=ak.bigint)
        got = abs(a)

        #   where is not implemented for bigint, so cast to int64
        a = a.astype("int64")
        got = got.astype("int64")

        exp = ak.where(a < 0, -a, a)
        assert (got == exp).all()
