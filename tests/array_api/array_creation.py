from math import sqrt

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

from arkouda.testing import assert_almost_equivalent, assert_equivalent


# requires the server to be built with 2D array support
SHAPES = [(0,), (0, 0), (1,), (5,), (2, 2), (5, 10)]
SIZES = [0, 0, 0, 1, 5, 4, 50]
DIMS = [1, 1, 2, 1, 1, 2, 2]


class TestArrayCreation:
    def test_creation_functions_docstrings(self):
        import doctest

        from arkouda.array_api import creation_functions

        result = doctest.testmod(
            creation_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ak.ScalarDTypes)
    def test_zeros(self, shape, dtype):
        a = xp.zeros(shape, dtype=dtype)
        assert_equivalent(a._array, np.zeros(shape, dtype=dtype))

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ak.ScalarDTypes)
    def test_ones(self, shape, dtype):
        a = xp.ones(shape, dtype=dtype)
        assert_equivalent(a._array, np.ones(shape, dtype=dtype))

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_from_numpy(self):
        # TODO: support 0D (scalar) arrays
        # (need changes to the create0D command from #2967)
        for shape in SHAPES:
            a = np.random.randint(0, 10, size=shape, dtype=np.int64)
            b = xp.asarray(a)
            assert b.size == a.size
            assert b.ndim == a.ndim
            assert b.shape == a.shape
            assert b.tolist() == a.tolist()

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("data_type", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_triu(self, data_type, prob_size):
        from arkouda.array_api.creation_functions import triu as array_triu

        size = int(sqrt(prob_size))

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            pda = xp.asarray(ak.randint(1, 10, (rows, cols)))
            nda = pda.to_ndarray()
            sweep = range(-(rows - 1), cols - 1)  # sweeps the diagonal from LL to UR
            for diag in sweep:
                np_triu = np.triu(nda, diag)
                ak_triu = array_triu(pda, k=diag)._array
                assert_almost_equivalent(ak_triu, np_triu)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("data_type", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_tril(self, data_type, prob_size):
        from arkouda.array_api.creation_functions import tril as array_tril

        size = int(sqrt(prob_size))

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            pda = xp.asarray(ak.randint(1, 10, (rows, cols)))
            nda = pda.to_ndarray()
            sweep = range(-(rows - 2), cols)  # sweeps the diagonal from LL to UR
            for diag in sweep:
                np_tril = np.tril(nda, diag)
                ak_tril = array_tril(pda, k=diag)._array
                assert_almost_equivalent(np_tril, ak_tril)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("data_type", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_eye(self, data_type, prob_size):
        from arkouda.array_api.creation_functions import eye as array_eye

        size = int(sqrt(prob_size))

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            sweep = range(-(cols - 1), rows)  # sweeps the diagonal from LL to UR
            for diag in sweep:
                np_eye = np.eye(rows, cols, diag, dtype=data_type)
                ak_eye = array_eye(rows, cols, k=diag, dtype=data_type)._array
                assert_almost_equivalent(np_eye, ak_eye)
