import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp
from arkouda.testing import assert_equivalent

# requires the server to be built with 2D array support
SHAPES = [(1,), (25,), (5, 10), (10, 5)]
SCALAR_TYPES = list(ak.ScalarDTypes)
SCALAR_TYPES.remove("bool_")


class TestSortingFunctions:
    def test_sorting_docstrings(self):
        import doctest

        from arkouda.array_api import sorting_functions

        result = doctest.testmod(
            sorting_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ak.ScalarDTypes)
    @pytest.mark.parametrize("descending", [True, False])
    def test_argsort(self, shape, dtype, descending):
        high = 100 if dtype != "bool_" else 2
        for axis in range(len(shape)):
            a = xp.asarray(ak.randint(0, high, shape, dtype=dtype, seed=pytest.seed))
            b = xp.argsort(a, axis=axis, descending=descending)
            np_b = a.to_ndarray().argsort(axis=axis, stable=True)
            np_b = np.flip(np_b, axis=axis) if descending else np_b

            assert b.size == a.size
            assert b.ndim == a.ndim
            assert b.shape == a.shape

            assert_equivalent(b._array, np_b)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_sort(self, dtype, shape):
        for axis in range(len(shape)):
            a = xp.asarray(ak.randint(0, 100, shape, dtype=dtype, seed=pytest.seed))
            sorted = xp.sort(a, axis=axis)

            assert sorted.size == a.size
            assert sorted.ndim == a.ndim
            assert sorted.shape == a.shape

            if len(shape) == 1:
                for i in range(1, sorted.size):
                    assert sorted[i - 1] <= sorted[i]

            else:
                if axis == 0:
                    for j in range(shape[1]):
                        for i in range(shape[0] - 1):
                            assert sorted[i, j] <= sorted[i + 1, j]

                else:
                    for i in range(shape[0]):
                        for j in range(shape[1] - 1):
                            assert sorted[i, j] <= sorted[i, j + 1]
