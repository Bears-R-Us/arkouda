import json

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159


def get_server_max_array_dims():
    try:
        return json.load(open("serverConfig.json", "r"))["max_array_dims"]
    except (ValueError, FileNotFoundError, TypeError, KeyError):
        return 1


class TestSearchingFunctions:
    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_argmax requires server with 'max_array_dims' >= 3",
    )
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

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_argmin requires server with 'max_array_dims' >= 3",
    )
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

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_nonzero requires server with 'max_array_dims' >= 3",
    )
    def test_nonzero(self):
        a = xp.zeros((4, 5, 6), dtype=ak.int64)
        a[0, 1, 0] = 1
        a[1, 2, 3] = 1
        a[2, 2, 2] = 1
        a[3, 2, 1] = 1

        nz = xp.nonzero(a)

        print(nz)

        assert nz[0].tolist() == [0, 1, 2, 3]
        assert nz[1].tolist() == [1, 2, 2, 2]
        assert nz[2].tolist() == [0, 3, 2, 1]

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_where requires server with 'max_array_dims' >= 3",
    )
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

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_search_sorted requires server with 'max_array_dims' >= 3",
    )
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
