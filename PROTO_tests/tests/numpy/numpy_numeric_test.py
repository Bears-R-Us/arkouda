import numpy as np
import pytest

import arkouda as ak

NUMERIC_TYPES = [ak.int64, ak.float64, ak.bool_, ak.uint64]
NO_BOOL = [ak.int64, ak.float64, ak.uint64]
NO_FLOAT = [ak.int64, ak.bool_, ak.uint64]
INT_FLOAT = [ak.int64, ak.float64]


class TestNumeric:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_floor_float(self, prob_size):
        from arkouda import all as akall
        from arkouda.numpy import floor as ak_floor

        a = 0.5 * ak.arange(prob_size, dtype="float64")
        a_floor = ak_floor(a)

        expected_size = np.floor((prob_size + 1) / 2).astype("int64")
        expected = ak.array(np.repeat(ak.arange(expected_size, dtype="float64").to_ndarray(), 2))
        #   To deal with prob_size as an odd number:
        expected = expected[0:prob_size]

        assert akall(a_floor == expected)
