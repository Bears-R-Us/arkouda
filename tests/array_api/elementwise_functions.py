import pytest

import arkouda as ak
import arkouda.array_api as xp
from arkouda.testing import assert_equal

# requires the server to be built with 2D array support
SHAPES = [(), (0,), (0, 0), (1,), (5,), (2, 2), (5, 10)]
SIZES = [1, 0, 0, 1, 5, 4, 50]
DIMS = [0, 1, 2, 1, 1, 2, 2]


class TestElemenwiseFunctions:
    def test_elementwise_functions_docstrings(self):
        import doctest

        from arkouda.array_api import elementwise_functions

        result = doctest.testmod(
            elementwise_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_logical_not(self):
        a = xp.asarray(ak.array([True, False, True, False]))
        not_a = xp.asarray(ak.array([False, True, False, True]))
        assert_equal(xp.logical_not(a)._array, not_a._array)
