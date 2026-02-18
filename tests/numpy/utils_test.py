import numpy as np
import pytest

import arkouda as ak


class TestFromNumericFunctions:
    # def test_utils_docstrings(self):
    #     import doctest
    #
    #     result = doctest.testmod(utils, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    #     assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("x", [0, [0], [1, 2, 3], np.ndarray([0, 1, 2]), [[1, 3]], np.eye(3, 2)])
    def test_shape(self, x):
        assert ak.shape(x) == np.shape(x)

    def test_shape_pdarray(self):
        a = ak.arange(5)
        assert ak.shape(a) == np.shape(a.to_ndarray())

    def test_shape_strings(self):
        a = ak.array(["a", "b", "c"])
        assert ak.shape(a) == np.shape(a.to_ndarray())

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_shape_multidim_pdarray(self):
        assert ak.shape(ak.eye(3, 2)) == (3, 2)
