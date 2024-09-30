import pytest

import arkouda as ak
import numpy as np


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
        a = ak.arange(4).reshape((2,2))
        np_a = np.arange(4).reshape((2,2))
        assert isinstance(a.shape, tuple)
        assert a.shape == np_a.shape
