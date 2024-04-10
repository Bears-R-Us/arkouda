import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

SEED = 12345
s = SEED


def randArr(shape):
    global s
    s += 2
    return Array.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class IndexingTests(ArkoudaTest):
    def test_pdarray_index(self):
        a = randArr((5, 6, 7))
        anp = np.asarray(a.tolist())
        idxnp = np.asarray([1, 2, 3, 4])
        idx = Array.asarray(idxnp)

        x = a[idx, idx, idx]
        xnp = anp[idxnp, idxnp, idxnp]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[idx, :, 2]
        xnp = anp[idxnp, :, 2]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[:, idx, idx]
        xnp = anp[:, idxnp, idxnp]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[0, idx, 3]
        xnp = anp[0, idxnp, 3]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[..., idx]
        xnp = anp[..., idxnp]
        self.assertEqual(x.tolist(), xnp.tolist())
