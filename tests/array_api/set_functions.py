import numpy as np

import arkouda as ak
import arkouda.array_api as xp

SEED = 314159
s = SEED


def ret_shapes():
    shapes = [1000]
    if 2 in ak.client.get_array_ranks():
        shapes.append((20, 50))
    if 3 in ak.client.get_array_ranks():
        shapes.append((2, 10, 50))
    return shapes


def rand_arr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class TestSetFunction:
    def test_set_functions_docstrings(self):
        import doctest

        from arkouda.array_api import set_functions

        result = doctest.testmod(
            set_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_set_functions(self):
        for shape in ret_shapes():
            r = rand_arr(shape)

            ua = xp.unique_all(r)
            uc = xp.unique_counts(r)
            ui = xp.unique_inverse(r)
            uv = xp.unique_values(r)

            (nuv, nuidx, nuinv, nuc) = np.unique(
                r.to_ndarray(),
                return_index=True,
                return_inverse=True,
                return_counts=True,
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
