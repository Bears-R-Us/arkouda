import json

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

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
