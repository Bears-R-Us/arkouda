import numpy as np
import pytest

import arkouda as ak
from arkouda.categorical import Categorical
from arkouda.testing import assert_arkouda_array_equivalent, assert_equal

seed = pytest.seed

DTYPES = ["uint64", "uint8", "int64", "float64", "bigint", "bool"]


class TestNumpyManipulationFunctions:

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
        s = ak.random_strings_uniform(1, 2, size, seed=seed)
        assert_equal(ak.flip(s), s[::-1])

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flip_categorical(self, size):
        s = ak.random_strings_uniform(1, 2, size, seed=seed)
        c = Categorical(s)
        assert_equal(ak.flip(c), c[::-1])

        # test case when c.permutation = None
        c2 = Categorical(c.to_pandas())
        assert_equal(ak.flip(c2), c2[::-1])

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
