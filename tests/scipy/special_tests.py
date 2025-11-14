import numpy as np
import pytest

from scipy.special import xlogy as scipy_xlogy

import arkouda as ak

from arkouda.scipy.special import xlogy


class TestStats:
    def test_scipy_special_docstrings(self):
        import doctest

        from arkouda.scipy import special

        result = doctest.testmod(special, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("np_x", [3, 5, np.float64(6), np.array([1.0, 2.0, 4.5])])
    @pytest.mark.parametrize(
        "np_y",
        [np.array([1, 2, 3]), np.array([10, 100, 100]), np.array([-1, 0, np.nan])],
    )
    def test_xlogy(self, np_x, np_y):
        x = ak.array(np_x) if isinstance(np_x, np.ndarray) else np_x
        y = ak.array(np_y)

        ak_result = xlogy(x, y)
        scipy_result = scipy_xlogy(np_x, np_y)

        assert np.allclose(ak_result.to_ndarray(), scipy_result, equal_nan=True)
