import pytest

import arkouda as ak
import arkouda.array_api as xp

# requires the server to be built with 2D array support
SHAPES = [(1,), (25,), (5, 10), (10, 5)]
SEED = 12345
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
    def test_argsort(self):
        for shape in SHAPES:
            for dtype in ak.ScalarDTypes:
                for axis in range(len(shape)):
                    a = xp.asarray(ak.randint(0, 100, shape, dtype=dtype, seed=SEED))
                    b = xp.argsort(a, axis=axis)

                    assert b.size == a.size
                    assert b.ndim == a.ndim
                    assert b.shape == a.shape

                    if len(shape) == 1:
                        aSorted = xp.take(a, b, axis=axis).tolist()

                        for i in range(1, len(aSorted)):
                            assert aSorted[i - 1] <= aSorted[i]
                    else:
                        if axis == 0:
                            for j in range(shape[1]):
                                # TODO: use take once 'squeeze' is implemented
                                # aSorted = xp.take(a, squeeze(b[:, j]), axis=0).tolist())
                                for i in range(shape[0] - 1):
                                    assert a[b[i, j], j] <= a[b[i + 1, j], j]

                        else:
                            for i in range(shape[0]):
                                # TODO: use take once 'squeeze' is implemented
                                # aSorted = xp.take(a, squeeze(b[i, :]), axis=1).tolist())
                                for j in range(shape[1] - 1):
                                    assert a[i, b[i, j]] <= a[i, b[i, j + 1]]

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", SCALAR_TYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_sort(self, dtype, shape):
        for axis in range(len(shape)):
            a = xp.asarray(ak.randint(0, 100, shape, dtype=dtype, seed=SEED))
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
