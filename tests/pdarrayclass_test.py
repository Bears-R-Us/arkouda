import numpy as np
import pytest

import arkouda as ak
from arkouda.testing import assert_equal as ak_assert_equal


class TestPdarrayClass:

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_reshape(self):
        a = ak.arange(4)
        r = a.reshape((2, 2))
        assert r.shape == (2, 2)
        assert isinstance(r, ak.pdarray)

    def test_shape(self):
        a = ak.arange(4)
        np_a = np.arange(4)
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_shape_multidim(self):
        a = ak.arange(4).reshape((2, 2))
        np_a = np.arange(4).reshape((2, 2))
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flatten(self, size):
        a = ak.arange(size)
        ak_assert_equal(a.flatten(), a)

    @pytest.mark.skip_if_max_rank_less_than(3)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flatten(self, size):
        size = size - (size % 4)
        a = ak.arange(size)
        b = a.reshape((2, 2, size / 4))
        ak_assert_equal(b.flatten(), a)
