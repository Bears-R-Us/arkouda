import pytest

import arkouda as ak
import arkouda.array_api as xp
from arkouda.testing import assert_equal

# requires the server to be built with 2D array support
SHAPES = [(), (0,), (0, 0), (1,), (5,), (2, 2), (5, 10)]
SIZES = [1, 0, 0, 1, 5, 4, 50]
DIMS = [0, 1, 2, 1, 1, 2, 2]


class TestElemenwiseFunctions:
    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_logical_not(self):
        a = xp.asarray(ak.array([True, False, True, False]))
        not_a = xp.asarray(ak.array([False, True, False, True]))
        assert_equal(xp.logical_not(a)._array, not_a._array)
