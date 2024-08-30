import json

import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159
s = SEED


def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class TestSetFunction:

    @pytest.mark.skip_if_max_rank_less_than(3)
    def test_set_functions(self):

        for shape in [(1000), (20, 50), (2, 10, 50)]:
            r = randArr(shape)

            ua = xp.unique_all(r)
            uc = xp.unique_counts(r)
            ui = xp.unique_inverse(r)
            uv = xp.unique_values(r)

            (nuv, nuidx, nuinv, nuc) = np.unique(
                r.to_ndarray(), return_index=True, return_inverse=True, return_counts=True
            )

            assert ua.values.tolist() == nuv.tolist()
            assert ua.indices.tolist() == nuidx.tolist()

            assert ua.inverse_indices.tolist() == np.reshape(nuinv, shape).tolist()
            assert ua.counts.tolist() == nuc.tolist()

            assert uc.values.tolist() == nuv.tolist()
            assert uc.counts.tolist() == nuc.tolist()

            assert ui.values.tolist() == nuv.tolist()
            assert ui.inverse_indices.tolist() == np.reshape(nuinv, shape).tolist()

            assert uv.tolist() == nuv.tolist()
