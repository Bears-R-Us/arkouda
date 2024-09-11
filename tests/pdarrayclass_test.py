import pytest

import arkouda as ak


class TestPdarrayClass:

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_reshape(self):
        a = ak.arange(4)
        r = a.reshape((2, 2))
        assert r.shape == [2, 2]
        assert isinstance(r, ak.pdarray)
