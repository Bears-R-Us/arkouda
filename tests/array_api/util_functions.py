import json

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159
s = SEED

DTYPES = [ak.int64, ak.float64, ak.uint64, ak.uint8]


def randArr(shape, dtype):

    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s), dtype=dtype)


class TestUtilFunctions:

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_all(self):
        a = xp.ones((10, 10), dtype=ak.bool_)
        assert xp.all(a)

        a[3, 4] = False
        assert not xp.all(a)

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_any(self):
        a = xp.zeros((10, 10), dtype=ak.bool_)
        assert not xp.any(a)

        a[3, 4] = True
        assert xp.any(a)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_clip(self, dtype):
        a = randArr((5, 6, 7), dtype)

        anp = a.to_ndarray()

        a_c = xp.clip(a, 10, 90)
        anp_c = np.clip(anp, 10, 90)
        assert a_c.tolist() == anp_c.tolist()

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_clip_errors(self):
        # bool
        a = xp.asarray(ak.randint(0, 100, (5, 6, 7), dtype=ak.bool_, seed=s), dtype=ak.bool_)
        with pytest.raises(
            RuntimeError, match="Error executing command: clip does not support dtype bool"
        ):
            xp.clip(a, 10, 90)

        # bigint
        bi_arr = ak.array(
            [0, 1, 2, 3, 4, 2**64 - 5, 2**64 - 4, 2**64 - 3, 2**64 - 2, 2**64 - 1],
            dtype=ak.bigint,
            max_bits=64,
        )
        a = xp.asarray(bi_arr, dtype=ak.bool_)
        with pytest.raises(
            RuntimeError, match="Error executing command: clip does not support dtype bigint"
        ):
            xp.clip(a, 10, 90)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_diff(self, dtype):
        a = randArr((5, 6, 7), dtype)
        anp = a.to_ndarray()

        a_d = xp.diff(a, n=1, axis=1)
        anp_d = np.diff(anp, n=1, axis=1)
        assert a_d.tolist() == anp_d.tolist()

        a_d = xp.diff(a, n=2, axis=0)
        anp_d = np.diff(anp, n=2, axis=0)

        assert a_d.tolist() == anp_d.tolist()

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_diff_error(self):
        # bool
        a = xp.asarray(ak.randint(0, 100, (5, 6, 7), dtype=ak.bool_, seed=s), dtype=ak.bool_)
        with pytest.raises(
            RuntimeError, match="Error executing command: diff does not support dtype bool"
        ):
            xp.diff(a, n=2, axis=0)

        # bigint
        bi_arr = ak.array(
            [0, 1, 2, 3, 4, 2**64 - 5, 2**64 - 4, 2**64 - 3, 2**64 - 2, 2**64 - 1],
            dtype=ak.bigint,
            max_bits=64,
        )
        a = xp.asarray(bi_arr, dtype=ak.bool_)
        with pytest.raises(
            RuntimeError, match="Error executing command: diff does not support dtype bigint"
        ):
            xp.diff(a, n=2, axis=0)

    @pytest.mark.skip_if_rank_not_compiled([3])
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

    def test_pad_error(self):
        # bigint
        bi_arr = ak.array(
            [0, 1, 2, 3, 4, 2**64 - 5, 2**64 - 4, 2**64 - 3, 2**64 - 2, 2**64 - 1],
            dtype=ak.bigint,
            max_bits=64,
        )
        a = xp.asarray(bi_arr, dtype=ak.bool_)
        with pytest.raises(
            RuntimeError, match="Error executing command: pad does not support dtype bigint"
        ):
            xp.pad(a, ((1, 1)), mode="constant", constant_values=((-1, 1)))
            xp.diff(a, n=2, axis=0)
