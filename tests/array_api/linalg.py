from math import sqrt

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp
from arkouda.testing import assert_almost_equivalent


class TestLinalg:
    def test_linalg_docstrings(self):
        import doctest

        from arkouda.array_api import linalg

        result = doctest.testmod(linalg, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("data_type1", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("data_type2", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_matmul(self, data_type1, data_type2, prob_size):
        size = int(sqrt(prob_size))

        # test on one square and two non-square products

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            arrayLeft = xp.asarray(ak.randint(0, 10, (rows, size), dtype=data_type1))
            ndaLeft = arrayLeft.to_ndarray()
            arrayRight = xp.asarray(ak.randint(0, 10, (size, cols), dtype=data_type2))
            ndaRight = arrayRight.to_ndarray()
            akProduct = xp.matmul(arrayLeft, arrayRight)
            npProduct = np.matmul(ndaLeft, ndaRight)
            assert_almost_equivalent(akProduct._array, npProduct)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("data_type", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_transpose(self, data_type, prob_size):
        size = int(sqrt(prob_size))

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            array = xp.asarray(ak.randint(1, 10, (rows, cols)))
            nda = array.to_ndarray()
            npa = np.transpose(nda)
            ppa = xp.matrix_transpose(array)._array
            assert np.allclose(ppa.to_ndarray(), npa)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("data_type1", [ak.int64, ak.float64, ak.bool_])
    @pytest.mark.parametrize("data_type2", [ak.int64, ak.float64])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_vecdot(self, data_type1, data_type2, prob_size):
        depth = np.random.randint(2, 10)
        width = prob_size // depth

        pda_a = xp.asarray(ak.randint(0, 10, (depth, width), dtype=data_type1))
        nda_a = pda_a.to_ndarray()
        pda_b = xp.asarray(ak.randint(0, 10, (depth, width), dtype=data_type2))
        nda_b = pda_b.to_ndarray()
        akProduct = xp.vecdot(pda_a, pda_b)
        npProduct = np.vecdot(nda_a, nda_b)
        assert_almost_equivalent(npProduct, akProduct._array)
