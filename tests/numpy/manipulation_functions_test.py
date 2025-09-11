from typing import Tuple

import numpy as np
import pytest

import arkouda as ak
from arkouda.pandas.categorical import Categorical
from arkouda.testing import assert_arkouda_array_equivalent, assert_equal


DTYPES = ["uint64", "uint8", "int64", "float64", "bigint", "bool"]


class TestNumpyManipulationFunctions:
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_manipulation_functions_docstrings(self):
        import doctest

        from arkouda import manipulation_functions

        result = doctest.testmod(
            manipulation_functions,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_flip_pdarray(self, size, dtype):
        a = ak.arange(size, dtype=dtype)
        f = ak.flip(a)
        assert_equal(f, a[::-1])

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    def test_flip_multi_dim(self, size, dtype):
        a = ak.arange(size * 4, dtype=dtype).reshape((2, 2, size))
        f = ak.flip(a)
        assert_equal(f, (size * 4 - 1) - a)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flip_multi_dim_bool(self, size):
        shape = (size, 2)

        vals = ak.array([True, False])
        segs = ak.array([0, size])
        perm = ak.concatenate([ak.arange(size) * 2, ak.arange(size) * 2 + 1])
        a = ak.broadcast(segments=segs, values=vals, permutation=perm).reshape(shape)
        f = ak.flip(a)

        vals2 = ak.array([False, True])
        f2 = ak.broadcast(segments=segs, values=vals2, permutation=perm).reshape(shape)
        assert_equal(f, f2)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flip_string(self, size):
        s = ak.random_strings_uniform(1, 2, size, seed=pytest.seed)
        assert_equal(ak.flip(s), s[::-1])

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flip_categorical(self, size):
        s = ak.random_strings_uniform(1, 2, size, seed=pytest.seed)
        c = Categorical(s)
        assert_equal(ak.flip(c), c[::-1])

        # test case when c.permutation = None
        c2 = Categorical(c.to_pandas())
        assert_equal(ak.flip(c2), c2[::-1])

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_repeat_pdarray(self, size, dtype):
        if dtype == ak.bigint:
            a = ak.arange(2**200, 2**200 + size, dtype=dtype)
            np_a = np.arange(2**200, 2**200 + size)
        else:
            a = ak.arange(size, dtype=dtype)
            np_a = np.arange(size, dtype=dtype)
        f = ak.repeat(a, 2)
        np_f = np.repeat(np_a, 2)
        assert_arkouda_array_equivalent(np_f, f)
        f = ak.repeat(a, 2, axis=0)
        np_f = np.repeat(np_a, 2, axis=0)
        assert_arkouda_array_equivalent(np_f, f)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize("shape", [(2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)])
    def test_repeat_dim_3(self, dtype, shape: Tuple[int, ...]):
        from arkouda.pdarraycreation import randint as akrandint

        shape_prod = 1
        for i in shape:
            shape_prod *= i
        if dtype == ak.bigint:
            np_a = np.arange(2**200, 2**200 + shape_prod)
            a = ak.array(np_a, dtype=dtype).reshape(shape)
            np_a = np_a.reshape(shape)
        else:
            a = ak.arange(shape_prod, dtype=dtype).reshape(shape)
            np_a = np.arange(shape_prod, dtype=dtype).reshape(shape)
        reps = akrandint(0, 10, 1, seed=pytest.seed)
        f = ak.repeat(a, reps)
        np_f = np.repeat(np_a, reps.to_ndarray())
        assert_arkouda_array_equivalent(np_f, f)
        for axis in range(-3, 3):
            reps = akrandint(0, 10, 1, seed=pytest.seed)
            f = ak.repeat(a, reps, axis=axis)
            np_f = np.repeat(np_a, reps.to_ndarray(), axis=axis)
            assert_arkouda_array_equivalent(np_f, f)
            reps = akrandint(0, 10, size=shape[axis], seed=pytest.seed)
            f = ak.repeat(a, reps, axis=axis)
            np_f = np.repeat(np_a, reps.to_ndarray(), axis=axis)
            assert_arkouda_array_equivalent(np_f, f)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 2), (2, 1), (1, 2)])
    def test_repeat_dim_2(self, dtype, shape: Tuple[int, ...]):
        from arkouda.pdarraycreation import randint as akrandint

        shape_prod = 1
        for i in shape:
            shape_prod *= i
        if dtype == ak.bigint:
            a = ak.arange(2**200, 2**200 + shape_prod).reshape(shape)
            np_a = np.arange(2**200, 2**200 + shape_prod).reshape(shape)
        else:
            a = ak.arange(shape_prod, dtype=dtype).reshape(shape)
            np_a = np.arange(shape_prod, dtype=dtype).reshape(shape)
        reps = akrandint(0, 10, 1, seed=pytest.seed)
        f = ak.repeat(a, reps)
        np_f = np.repeat(np_a, reps.to_ndarray())
        assert_arkouda_array_equivalent(np_f, f)
        for axis in range(-2, 2):
            reps = akrandint(0, 10, 1, seed=pytest.seed)
            f = ak.repeat(a, reps, axis=axis)
            np_f = np.repeat(np_a, reps.to_ndarray(), axis=axis)
            assert_arkouda_array_equivalent(np_f, f)
            reps = akrandint(0, 10, size=shape[axis], seed=pytest.seed)
            f = ak.repeat(a, reps, axis=axis)
            np_f = np.repeat(np_a, reps.to_ndarray(), axis=axis)
            assert_arkouda_array_equivalent(np_f, f)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_squeeze_1D(self, size, dtype):
        x = ak.arange(size, dtype=dtype)
        assert_equal(ak.squeeze(x), ak.arange(size, dtype=dtype))

        y = 1
        assert_equal(ak.squeeze(y), ak.array([1]))

        z = ak.array([1])
        assert_equal(ak.squeeze(z), ak.array([1]))

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_squeeze(self, size, dtype):
        if dtype == "bigint":
            pytest.skip("Skip until #3870 is resolved.")

        x = ak.arange(size, dtype=dtype).reshape((1, size, 1))
        assert_equal(ak.squeeze(x, axis=None), ak.arange(size, dtype=dtype))
        assert_equal(ak.squeeze(x, axis=-3), ak.arange(size, dtype=dtype).reshape((size, 1)))
        assert_equal(ak.squeeze(x, axis=-1), ak.arange(size, dtype=dtype).reshape((1, size)))
        assert_equal(ak.squeeze(x, axis=0), ak.arange(size, dtype=dtype).reshape((size, 1)))
        assert_equal(ak.squeeze(x, axis=2), ak.arange(size, dtype=dtype).reshape((1, size)))
        assert_equal(ak.squeeze(x, axis=(0, 2)), ak.arange(size, dtype=dtype))

        y = 1
        assert_equal(ak.squeeze(y), ak.array([1]))

        z = ak.array([[[1]]])
        assert_equal(ak.squeeze(z), ak.array([1]))

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    def test_tile_pdarray(self, size, dtype):
        a = ak.arange(size, dtype=dtype)
        np_a = np.arange(size, dtype=dtype)
        f = ak.tile(a, 2)
        np_f = np.tile(np_a, 2)
        assert_arkouda_array_equivalent(np_f, f)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    @pytest.mark.parametrize("shape", [3, (2, 3), (2, 2, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2)])
    def test_tile_dim_2_and_3(self, size, dtype, shape):
        a = ak.arange(size * 4, dtype=dtype).reshape((2, 2, size))
        np_a = np.arange(size * 4, dtype=dtype).reshape((2, 2, size))
        f = ak.tile(a, shape)
        np_f = np.tile(np_a, shape)
        assert_arkouda_array_equivalent(np_f, f)
        a = ak.arange(size, dtype=dtype)
        np_a = np.arange(size, dtype=dtype)
        f = ak.tile(a, shape)
        np_f = np.tile(np_a, shape)
        assert_arkouda_array_equivalent(np_f, f)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    @pytest.mark.parametrize("shape", [3, (2, 3), (2, 2), (2, 1), (1, 2)])
    def test_tile_dim_2(self, size, dtype, shape):
        a = ak.arange(size * 2, dtype=dtype).reshape((2, size))
        np_a = np.arange(size * 2, dtype=dtype).reshape((2, size))
        f = ak.tile(a, shape)
        np_f = np.tile(np_a, shape)
        assert_arkouda_array_equivalent(np_f, f)
        a = ak.arange(size, dtype=dtype)
        np_a = np.arange(size, dtype=dtype)
        f = ak.tile(a, shape)
        np_f = np.tile(np_a, shape)
        assert_arkouda_array_equivalent(np_f, f)
