from math import sqrt

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp
from arkouda.testing import assert_almost_equivalent

# requires the server to be built with 2D array support
SHAPES = [(), (0,), (0, 0), (1,), (5,), (2, 2), (5, 10)]
SIZES = [1, 0, 0, 1, 5, 4, 50]
DIMS = [0, 1, 2, 1, 1, 2, 2]


class TestArrayCreation:
    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_zeros(self):
        for shape, size, dim in zip(SHAPES, SIZES, DIMS):
            for dtype in ak.ScalarDTypes:
                a = xp.zeros(shape, dtype=dtype)
                assert a.size == size
                assert a.ndim == dim
                assert a.shape == shape
                assert a.dtype == dtype
                assert a.tolist() == np.zeros(shape, dtype=dtype).tolist()

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_ones(self):
        for shape, size, dim in zip(SHAPES, SIZES, DIMS):
            for dtype in ak.ScalarDTypes:
                a = xp.ones(shape, dtype=dtype)
                assert a.size == size
                assert a.ndim == dim
                assert a.shape == shape
                assert a.dtype == dtype
                assert a.tolist() == np.ones(shape, dtype=dtype).tolist()

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_from_numpy(self):
        # TODO: support 0D (scalar) arrays
        # (need changes to the create0D command from #2967)
        for shape in SHAPES[1:]:
            a = np.random.randint(0, 10, size=shape, dtype=np.int64)
            b = xp.asarray(a)
            assert b.size == a.size
            assert b.ndim == a.ndim
            assert b.shape == a.shape
            assert b.tolist() == a.tolist()

    @pytest.mark.skip_if_max_rank_less_than(2)
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

    @pytest.mark.skip_if_max_rank_less_than(2)
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

    @pytest.mark.skip_if_max_rank_less_than(2)
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
