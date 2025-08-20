import math

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159


class TestStatsFunction:
    def test_statistical_functions_docstrings(self):
        import doctest

        from arkouda.array_api import statistical_functions

        result = doctest.testmod(
            statistical_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_max(self):
        a = xp.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = 101

        assert xp.max(a) == 101

        aMax0 = xp.max(a, axis=0)
        assert aMax0.shape == (7, 4)
        assert aMax0[6, 2] == 101

        aMax02 = xp.max(a, axis=(0, 2))
        assert aMax02.shape == (7,)
        assert aMax02[6] == 101

        aMax02Keepdims = xp.max(a, axis=(0, 2), keepdims=True)
        assert aMax02Keepdims.shape == (1, 7, 1)
        assert aMax02Keepdims[0, 6, 0] == 101

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_min(self):
        a = xp.asarray(ak.randint(0, 100, (5, 7, 4), dtype=ak.int64, seed=SEED))
        a[3, 6, 2] = -1

        assert xp.min(a) == -1

        aMin0 = xp.min(a, axis=0)
        assert aMin0.shape == (7, 4)
        assert aMin0[6, 2] == -1

        aMin02 = xp.min(a, axis=(0, 2))
        assert aMin02.shape == (7,)
        assert aMin02[6] == -1

        aMin02Keepdims = xp.min(a, axis=(0, 2), keepdims=True)
        assert aMin02Keepdims.shape == (1, 7, 1)
        assert aMin02Keepdims[0, 6, 0] == -1

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_mean(self):
        a = xp.ones((10, 5, 5))
        a[0, 0, 0] = 251

        assert int(xp.mean(a)) == 2

        a[:, 0, 0] = 26

        print(a.tolist())

        aMean0 = xp.mean(a, axis=0)
        assert aMean0.shape == (5, 5)
        assert aMean0[0, 0] == 26
        assert aMean0[2, 2] == 1

        aMean02 = xp.mean(a, axis=(1, 2))
        assert aMean02.shape == (10,)
        assert aMean02[0] == 2
        assert aMean02[2] == 2

        aMean02Keepdims = xp.mean(a, axis=(1, 2), keepdims=True)
        assert aMean02Keepdims.shape == (10, 1, 1)
        assert aMean02Keepdims[0, 0, 0] == 2
        assert aMean02Keepdims[2, 0, 0] == 2

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_std(self):
        a = xp.ones((10, 5, 5), dtype=ak.float64)
        a[0, 0, 0] = 26
        assert math.fabs(float(xp.std(a)) - math.sqrt(2.49)) < 1e-10

        aStd0 = xp.std(a, axis=0)
        assert aStd0.shape == (5, 5)
        assert aStd0[0, 0] == 7.5

        aStd02 = xp.std(a, axis=(1, 2))
        assert aStd02.shape == (10,)
        assert abs(aStd02[0] - math.sqrt(24)) < 1e-10

        aStd02Keepdims = xp.std(a, axis=(1, 2), keepdims=True)
        assert aStd02Keepdims.shape == (10, 1, 1)
        assert abs(aStd02Keepdims[0, 0, 0] - math.sqrt(24)) < 1e-10

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_var(self):
        a = xp.ones((10, 5, 5), dtype=ak.float64)
        a[0, 0, 0] = 26
        assert math.fabs(float(xp.var(a)) - 2.49) < 1e-10

        aStd0 = xp.var(a, axis=0)
        assert aStd0.shape == (5, 5)
        assert aStd0[0, 0] == 7.5**2

        aStd02 = xp.var(a, axis=(1, 2))
        assert aStd02.shape == (10,)
        assert abs(aStd02[0] - 24) < 1e-10

        aStd02Keepdims = xp.var(a, axis=(1, 2), keepdims=True)
        assert aStd02Keepdims.shape == (10, 1, 1)
        assert abs(aStd02Keepdims[0, 0, 0] - 24) < 1e-10

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_prod(self):
        a = xp.ones((2, 3, 4))
        a = a + a

        assert xp.prod(a) == 2**24

        aProd0 = xp.prod(a, axis=0)
        assert aProd0.shape == (3, 4)
        assert aProd0[0, 0] == 2**2

        aProd02 = xp.prod(a, axis=(1, 2))
        assert aProd02.shape == (2,)
        assert aProd02[0] == 2**12

        aProd02Keepdims = xp.prod(a, axis=(1, 2), keepdims=True)
        assert aProd02Keepdims.shape == (2, 1, 1)
        assert aProd02Keepdims[0, 0, 0] == 2**12

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_sum(self):
        a = xp.ones((2, 3, 4))

        assert xp.sum(a) == 24

        aSum0 = xp.sum(a, axis=0)
        assert aSum0.shape == (3, 4)
        assert aSum0[0, 0] == 2

        aSum02 = xp.sum(a, axis=(1, 2))
        assert aSum02.shape == (2,)
        assert aSum02[0] == 12

        aSum02Keepdims = xp.sum(a, axis=(1, 2), keepdims=True)
        assert aSum02Keepdims.shape == (2, 1, 1)
        assert aSum02Keepdims[0, 0, 0] == 12

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_cumulative_sum(self):
        a = xp.asarray(ak.randint(0, 100, (5, 6, 7), seed=SEED))

        a_sum_0 = xp.cumulative_sum(a, axis=0)
        a_sum_0_np = np.cumulative_sum(a.to_ndarray(), axis=0)
        assert a_sum_0.shape == (5, 6, 7)
        assert a_sum_0.tolist() == a_sum_0_np.tolist()

        a_sum_1 = xp.cumulative_sum(a, axis=1)
        a_sum_1_np = np.cumulative_sum(a.to_ndarray(), axis=1)
        assert a_sum_1.shape == (5, 6, 7)
        assert a_sum_1.tolist() == a_sum_1_np.tolist()

        b = xp.ones((5, 6, 7))

        b_sum_0 = xp.cumulative_sum(b, axis=0, include_initial=True)
        assert b_sum_0.shape == (6, 6, 7)
        assert b_sum_0[0, 0, 0] == 0
        assert b_sum_0[1, 0, 0] == 1
        assert b_sum_0[5, 0, 0] == 5

        b_sum_1 = xp.cumulative_sum(b, axis=1, include_initial=True)
        assert b_sum_1.shape == (5, 7, 7)
        assert b_sum_1[0, 0, 0] == 0
        assert b_sum_1[0, 1, 0] == 1
        assert b_sum_1[0, 6, 0] == 6

        c = xp.asarray(ak.randint(0, 100, 50, dtype=ak.float64, seed=SEED))
        c_sum = xp.cumulative_sum(c)
        c_sum_np = np.cumulative_sum(c.to_ndarray())

        assert c_sum.shape == (50,)
        c_list = c_sum.tolist()
        c_sum_np = c_sum_np.tolist()
        # for i in range(50):
        np.allclose(c_list, c_sum_np)

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_cumulative_prod(self):
        a = xp.asarray(ak.randint(1, 100, (5, 6, 7), seed=SEED))

        a_prod_0 = xp.cumulative_prod(a, axis=0)
        a_prod_0_np = np.cumulative_prod(a.to_ndarray(), axis=0)
        assert a_prod_0.shape == (5, 6, 7)
        assert a_prod_0.tolist() == a_prod_0_np.tolist()

        a_prod_1 = xp.cumulative_prod(a, axis=1)
        a_prod_1_np = np.cumulative_prod(a.to_ndarray(), axis=1)
        assert a_prod_1.shape == (5, 6, 7)
        assert a_prod_1.tolist() == a_prod_1_np.tolist()

        b = xp.ones((5, 6, 7))

        b_prod_0 = xp.cumulative_prod(b, axis=0, include_initial=True)
        assert b_prod_0.shape == (6, 6, 7)
        assert b_prod_0[0, 0, 0] == 1
        assert b_prod_0[1, 0, 0] == 1
        assert b_prod_0[5, 0, 0] == 1

        b_prod_1 = xp.cumulative_prod(b, axis=1, include_initial=True)
        assert b_prod_1.shape == (5, 7, 7)
        assert b_prod_1[0, 0, 0] == 1
        assert b_prod_1[0, 1, 0] == 1
        assert b_prod_1[0, 6, 0] == 1

        c = xp.asarray(ak.randint(1, 100, 50, dtype=ak.float64, seed=SEED))
        c_prod = xp.cumulative_prod(c)
        c_prod_np = np.cumulative_prod(c.to_ndarray())

        assert c_prod.shape == (50,)
        c_list = c_prod.tolist()
        c_prod_np = c_prod_np.tolist()
        np.allclose(c_list, c_prod_np)
