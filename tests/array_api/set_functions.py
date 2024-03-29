import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np


SEED = 314159
s = SEED


def randArr(shape):
    global s
    s += 2
    return Array.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class SetFunctionTests(ArkoudaTest):
    def test_set_functions(self):

        for shape in [(1000), (20, 50), (2, 10, 50)]:
            r = randArr(shape)

            ua = Array.unique_all(r)
            uc = Array.unique_counts(r)
            ui = Array.unique_inverse(r)
            uv = Array.unique_values(r)

            (nuv, nuidx, nuinv, nuc) = np.unique(
                r.to_ndarray(),
                return_index=True,
                return_inverse=True,
                return_counts=True
            )

            self.assertEqual(ua.values.tolist(), nuv.tolist())
            self.assertEqual(ua.indices.tolist(), nuidx.tolist())
            # compare flattened inverse_indices to numpy (since numpy returns a 1D array)
            self.assertEqual(Array.reshape(ua.inverse_indices, (-1,)).tolist(), nuinv.tolist())
            self.assertEqual(ua.counts.tolist(), nuc.tolist())

            self.assertEqual(uc.values.tolist(), nuv.tolist())
            self.assertEqual(uc.counts.tolist(), nuc.tolist())

            self.assertEqual(ui.values.tolist(), nuv.tolist())
            self.assertEqual(Array.reshape(ui.inverse_indices, (-1,)).tolist(), nuinv.tolist())

            self.assertEqual(uv.tolist(), nuv.tolist())
