import json

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 12345
s = SEED


def get_server_max_array_dims():
    try:
        return json.load(open("serverConfig.json", "r"))["max_array_dims"]
    except (ValueError, FileNotFoundError, TypeError, KeyError):
        return 1


def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class TestIndexing:
    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_rank_changing_assignment requires server with 'max_array_dims' >= 3",
    )
    def test_rank_changing_assignment(self):
        a = randArr((5, 6, 7))
        b = randArr((5, 6))
        c = randArr((6, 7))
        d = randArr((6,))
        e = randArr((5, 6, 7))

        a[:, :, 0] = b
        assert (a[:, :, 0]).tolist() == b.tolist()

        a[1, :, :] = c
        assert (a[1, :, :]).tolist() == c.tolist()

        a[2, :, 3] = d
        assert (a[2, :, 3]).tolist() == d.tolist()

        a[:, :, :] = e
        assert a.tolist() == e.tolist()

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_nd_assignment requires server with 'max_array_dims' >= 3",
    )
    def test_nd_assignment(self):
        a = randArr((5, 6, 7))
        bnp = randArr((5, 6, 7)).to_ndarray()

        a[1, 2, 3] = 42
        assert a[1, 2, 3] == 42

        a[:] = bnp
        assert a.tolist() == bnp.tolist()

        a[:] = 5
        assert (a == 5).all()

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_pdarray_index requires server with 'max_array_dims' >= 3",
    )
    def test_pdarray_index(self):
        a = randArr((5, 6, 7))
        anp = np.asarray(a.tolist())
        idxnp = np.asarray([1, 2, 3, 4])
        idx = xp.asarray(idxnp)

        x = a[idx, idx, idx]
        xnp = anp[idxnp, idxnp, idxnp]
        assert x.tolist() == xnp.tolist()

        x = a[idx, :, 2]
        xnp = anp[idxnp, :, 2]
        assert x.tolist() == xnp.tolist()

        x = a[:, idx, idx]
        xnp = anp[:, idxnp, idxnp]
        assert x.tolist() == xnp.tolist()

        x = a[0, idx, 3]
        xnp = anp[0, idxnp, 3]
        assert x.tolist() == xnp.tolist()

        x = a[..., idx]
        xnp = anp[..., idxnp]
        assert x.tolist() == xnp.tolist()

        x = a[:]
        xnp = anp[:]
        assert x.tolist() == xnp.tolist()

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_none_index requires server with 'max_array_dims' >= 3",
    )
    def test_none_index(self):
        a = randArr((10, 10))
        anp = np.asarray(a.tolist())
        idxnp = np.asarray([1, 2, 3, 4])
        idx = xp.asarray(idxnp)

        x = a[None, 1, :]
        xnp = anp[None, 1, :]
        assert x.tolist() == xnp.tolist()

        x = a[1, None, 2]
        xnp = anp[1, None, 2]
        assert x.tolist() == xnp.tolist()

        x = a[None, ...]
        xnp = anp[None, ...]
        assert x.tolist() == xnp.tolist()

        x = a[idx, None, :]
        xnp = anp[idxnp, None, :]
        assert x.tolist() == xnp.tolist()

        x = a[3, idx, None]
        xnp = anp[3, idxnp, None]
        assert x.tolist() == xnp.tolist()

        b = randArr((10))
        bnp = np.asarray(b.tolist())

        x = b[None, None, :]
        xnp = bnp[None, None, :]
        assert x.tolist() == xnp.tolist()

        x = b[None, None, 1]
        xnp = bnp[None, None, 1]
        assert x.tolist() == xnp.tolist()

        x = b[None, 1, None]
        xnp = bnp[None, 1, None]
        assert x.tolist() == xnp.tolist()

        x = b[None, :, None]
        xnp = bnp[None, :, None]
        assert x.tolist() == xnp.tolist()

        x = b[1, None, None]
        xnp = bnp[1, None, None]
        assert x.tolist() == xnp.tolist()

        x = b[:, None, None]
        xnp = bnp[:, None, None]
        assert x.tolist() == xnp.tolist()
