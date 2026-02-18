import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159


class TestSearchingFunctions:
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_searching_functions_docstrings(self):
        import doctest

        from arkouda.array_api import searching_functions

        result = doctest.testmod(
            searching_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_argmax(self):
        a = xp.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))
        a[3, 2, 1] = 101

        print(a.tolist())

        assert xp.argmax(a) == 1 + 2 * 6 + 3 * 6 * 5

        aArgmax0 = xp.argmax(a, axis=0)
        assert aArgmax0.shape == (5, 6)
        assert aArgmax0[2, 1] == 3

        aArgmax1Keepdims = xp.argmax(a, axis=1, keepdims=True)
        assert aArgmax1Keepdims.shape == (4, 1, 6)
        assert aArgmax1Keepdims[3, 0, 1] == 2

        aArgmax1Keepdims = xp.argmax(a, axis=-2, keepdims=True)
        assert aArgmax1Keepdims.shape == (4, 1, 6)
        assert aArgmax1Keepdims[3, 0, 1] == 2

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_argmin(self):
        a = xp.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))
        a[3, 2, 1] = -1

        assert xp.argmin(a) == 103

        aArgmin0 = xp.argmin(a, axis=0)
        assert aArgmin0.shape == (5, 6)
        assert aArgmin0[2, 1] == 3

        aArgmin1Keepdims = xp.argmin(a, axis=1, keepdims=True)
        assert aArgmin1Keepdims.shape == (4, 1, 6)
        assert aArgmin1Keepdims[3, 0, 1] == 2

        aArgmin1Keepdims = xp.argmin(a, axis=-2, keepdims=True)
        assert aArgmin1Keepdims.shape == (4, 1, 6)
        assert aArgmin1Keepdims[3, 0, 1] == 2

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_nonzero(self):
        a = xp.zeros((40, 15, 16), dtype=ak.int64)
        a[0, 1, 0] = 1
        a[1, 2, 3] = 1
        a[2, 2, 2] = 1
        a[3, 2, 1] = 1
        a[10, 10, 10] = 1
        a[30, 12, 11] = 1
        a[2, 13, 14] = 1
        a[3, 14, 15] = 1

        nz = xp.nonzero(a)

        a_np = a.to_ndarray()
        nz_np = np.nonzero(a_np)

        assert nz[0].tolist() == nz_np[0].tolist()
        assert nz[1].tolist() == nz_np[1].tolist()
        assert nz[2].tolist() == nz_np[2].tolist()

    def test_nonzero_1d(self):
        b = xp.zeros(500, dtype=ak.int64)
        b[0] = 1
        b[12] = 1
        b[100] = 1
        b[205] = 1
        b[490] = 1

        nz = xp.nonzero(b)

        assert nz[0].tolist() == [0, 12, 100, 205, 490]

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_where(self):
        a = xp.zeros((4, 5, 6), dtype=ak.int64)
        a[1, 2, 3] = 1
        a[3, 2, 1] = 1
        a[2, 2, 2] = 1

        b = xp.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))
        c = xp.asarray(ak.randint(0, 100, (4, 5, 6), dtype=ak.int64, seed=SEED))

        d = xp.where(a, b, c)

        assert d.shape == (4, 5, 6)
        assert d[1, 2, 3] == b[1, 2, 3]
        assert d[3, 2, 1] == b[3, 2, 1]
        assert d[2, 2, 2] == b[2, 2, 2]
        assert d[0, 0, 0] == c[0, 0, 0]
        assert d[3, 3, 3] == c[3, 3, 3]

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_search_sorted(self):
        a = xp.asarray(ak.randint(0, 100, 1000, dtype=ak.float64))
        b = xp.asarray(ak.randint(0, 100, (10, 10), dtype=ak.float64))

        anp = a.to_ndarray()
        bnp = b.to_ndarray()

        sorter = xp.argsort(a)

        for side in ["left", "right"]:
            indices = xp.searchsorted(a, b, side=side, sorter=sorter)
            indicesnp = np.searchsorted(anp, bnp, side=side, sorter=sorter.to_ndarray())

            assert indices.tolist() == indicesnp.tolist()
