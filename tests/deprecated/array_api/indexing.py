from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as xp
import numpy as np

SEED = 12345
s = SEED


def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class IndexingTests(ArkoudaTest):
    def test_rank_changing_assignment(self):
        a = randArr((5, 6, 7))
        b = randArr((5, 6))
        c = randArr((6, 7))
        d = randArr((6,))
        e = randArr((5, 6, 7))

        a[:, :, 0] = b
        self.assertEqual((a[:, :, 0]).tolist(), b.tolist())

        a[1, :, :] = c
        self.assertEqual((a[1, :, :]).tolist(), c.tolist())

        a[2, :, 3] = d
        self.assertEqual((a[2, :, 3]).tolist(), d.tolist())

        a[:, :, :] = e
        self.assertEqual(a.tolist(), e.tolist())

    def test_nd_assignment(self):
        a = randArr((5, 6, 7))
        bnp = randArr((5, 6, 7)).to_ndarray()

        a[1, 2, 3] = 42
        self.assertEqual(a[1, 2, 3], 42)

        a[:] = bnp
        self.assertEqual(a.tolist(), bnp.tolist())

        a[:] = 5
        self.assertTrue((a == 5).all())

    def test_pdarray_index(self):
        a = randArr((5, 6, 7))
        anp = np.asarray(a.tolist())
        idxnp = np.asarray([1, 2, 3, 4])
        idx = xp.asarray(idxnp)

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

        x = a[:]
        xnp = anp[:]
        self.assertEqual(x.tolist(), xnp.tolist())

    def test_none_index(self):
        a = randArr((10, 10))
        anp = np.asarray(a.tolist())
        idxnp = np.asarray([1, 2, 3, 4])
        idx = xp.asarray(idxnp)

        x = a[None, 1, :]
        xnp = anp[None, 1, :]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[1, None, 2]
        xnp = anp[1, None, 2]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[None, ...]
        xnp = anp[None, ...]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[idx, None, :]
        xnp = anp[idxnp, None, :]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = a[3, idx, None]
        xnp = anp[3, idxnp, None]
        self.assertEqual(x.tolist(), xnp.tolist())

        b = randArr((10))
        bnp = np.asarray(b.tolist())

        x = b[None, None, :]
        xnp = bnp[None, None, :]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = b[None, None, 1]
        xnp = bnp[None, None, 1]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = b[None, 1, None]
        xnp = bnp[None, 1, None]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = b[None, :, None]
        xnp = bnp[None, :, None]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = b[1, None, None]
        xnp = bnp[1, None, None]
        self.assertEqual(x.tolist(), xnp.tolist())

        x = b[:, None, None]
        xnp = bnp[:, None, None]
        self.assertEqual(x.tolist(), xnp.tolist())
