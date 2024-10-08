import numpy as np
import pytest

import arkouda as ak
from arkouda.categorical import Categorical
from arkouda.testing import assert_equal

seed = pytest.seed


class TestNumpyManipulationFunctions:

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_flip_pdarray(self, size, dtype):
        a = ak.arange(size, dtype=dtype)
        f = ak.flip(a)
        assert_equal(f, a[::-1])

    @pytest.mark.skip_if_max_rank_less_than(3)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    def test_flip_multi_dim(self, size, dtype):
        a = ak.arange(size * 4, dtype=dtype).reshape((2, 2, size))
        f = ak.flip(a)
        assert_equal(f, (size * 4 - 1) - a)

    @pytest.mark.skip_if_max_rank_less_than(3)
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

    #   Currently there is a bug in the reverse indexing of Strings (#3821) that causes this test to fail.
    @pytest.mark.skip("Skip until bug fix Ticket #3821 is completed.")
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
