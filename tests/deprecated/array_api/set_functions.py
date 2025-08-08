import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

import arkouda.array_api as xp

SEED = 314159
s = SEED


def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class SetFunctionTests(ArkoudaTest):
    def test_set_functions(self):

        for shape in [(1000), (20, 50), (2, 10, 50)]:
            r = randArr(shape)

            ua = xp.unique_all(r)
            uc = xp.unique_counts(r)
            ui = xp.unique_inverse(r)
            uv = xp.unique_values(r)

            (nuv, nuidx, nuinv, nuc) = np.unique(
                r.to_ndarray(),
                return_index=True,
                return_inverse=True,
                return_counts=True
            )

            self.assertEqual(ua.values.tolist(), nuv.tolist())
            self.assertEqual(ua.indices.tolist(), nuidx.tolist())

            self.assertEqual(ua.inverse_indices.tolist(), np.reshape(nuinv, shape).tolist())
            self.assertEqual(ua.counts.tolist(), nuc.tolist())

            self.assertEqual(uc.values.tolist(), nuv.tolist())
            self.assertEqual(uc.counts.tolist(), nuc.tolist())

            self.assertEqual(ui.values.tolist(), nuv.tolist())
            self.assertEqual(ui.inverse_indices.tolist(), np.reshape(nuinv, shape).tolist())

            self.assertEqual(uv.tolist(), nuv.tolist())
