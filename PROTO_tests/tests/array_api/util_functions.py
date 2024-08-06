import json

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159
s = SEED


def get_server_max_array_dims():
    try:
        return json.load(open('serverConfig.json', 'r'))['max_array_dims']
    except (ValueError, FileNotFoundError, TypeError, KeyError):
        return 1

def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class TestUtilFunctions:
    @pytest.mark.skipif(
        get_server_max_array_dims() < 2,
        reason="test_all requires server with 'max_array_dims' >= 2",
    )
    def test_all(self):
        a = xp.ones((10, 10), dtype=ak.bool_)
        assert xp.all(a)

        a[3, 4] = False
        assert ~xp.all(a)

    @pytest.mark.skipif(
        get_server_max_array_dims() < 2,
        reason="test_any requires server with 'max_array_dims' >= 2",
    )
    def test_any(self):
        a = xp.zeros((10, 10), dtype=ak.bool_)
        assert ~xp.any(a)

        a[3, 4] = True
        assert xp.any(a)

    @pytest.mark.skipif(get_server_max_array_dims() < 3, reason="test_clip requires server with 'max_array_dims' >= 3")
    def test_clip(self):
        a = randArr((5, 6, 7))
        anp = a.to_ndarray()

        a_c = xp.clip(a, 10, 90)
        anp_c = np.clip(anp, 10, 90)
        assert a_c.tolist() == anp_c.tolist()

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_diff requires server with 'max_array_dims' >= 3",
    )
    def test_diff(self):
        a = randArr((5, 6, 7))
        anp = a.to_ndarray()

        a_d = xp.diff(a, n=1, axis=1)
        anp_d = np.diff(anp, n=1, axis=1)
        assert a_d.tolist() == anp_d.tolist()

        a_d = xp.diff(a, n=2, axis=0)
        anp_d = np.diff(anp, n=2, axis=0)

        assert a_d.tolist() == anp_d.tolist()

    @pytest.mark.skipif(
        get_server_max_array_dims() < 3,
        reason="test_pad requires server with 'max_array_dims' >= 3",
    )
    def test_pad(self):
        a = xp.ones((5, 6, 7))
        anp = np.ones((5, 6, 7))

        a_p = xp.pad(
            a, ((1, 1), (2, 2), (3, 3)), mode="constant", constant_values=((-1, 1), (-2, 2), (-3, 3))
        )
        anp_p = np.pad(
            anp, ((1, 1), (2, 2), (3, 3)), mode="constant", constant_values=((-1, 1), (-2, 2), (-3, 3))
        )
        assert a_p.tolist() == anp_p.tolist()

        a_p = xp.pad(a, (2, 3), constant_values=(55, 44))
        anp_p = np.pad(anp, (2, 3), constant_values=(55, 44))
        assert a_p.tolist() == anp_p.tolist()

        a_p = xp.pad(a, 2, constant_values=3)
        anp_p = np.pad(anp, 2, constant_values=3)
        assert a_p.tolist() == anp_p.tolist()
