
from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as xp
import numpy as np

SEED = 314159
s = SEED


def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class UtilFunctions(ArkoudaTest):
    def test_all(self):
        a = xp.ones((10, 10), dtype=ak.bool_)
        self.assertTrue(xp.all(a))

        a[3, 4] = False
        self.assertFalse(xp.all(a))

    def test_any(self):
        a = xp.zeros((10, 10), dtype=ak.bool_)
        self.assertFalse(xp.any(a))

        a[3, 4] = True
        self.assertTrue(xp.any(a))

    def test_clip(self):
        a = randArr((5, 6, 7))
        anp = a.to_ndarray()

        a_c = xp.clip(a, 10, 90)
        anp_c = np.clip(anp, 10, 90)
        self.assertEqual(a_c.tolist(), anp_c.tolist())

    def test_diff(self):
        a = randArr((5, 6, 7))
        anp = a.to_ndarray()

        a_d = xp.diff(a, n=1, axis=1)
        anp_d = np.diff(anp, n=1, axis=1)
        self.assertEqual(a_d.tolist(), anp_d.tolist())

        a_d = xp.diff(a, n=2, axis=0)
        anp_d = np.diff(anp, n=2, axis=0)

        self.assertEqual(a_d.tolist(), anp_d.tolist())

    def test_pad(self):
        a = xp.ones((5, 6, 7))
        anp = np.ones((5, 6, 7))

        a_p = xp.pad(a, ((1, 1), (2, 2), (3, 3)), mode='constant', constant_values=((-1, 1), (-2, 2), (-3, 3)))
        anp_p = np.pad(anp, ((1, 1), (2, 2), (3, 3)), mode='constant', constant_values=((-1, 1), (-2, 2), (-3, 3)))
        self.assertEqual(a_p.tolist(), anp_p.tolist())

        a_p = xp.pad(a, (2, 3), constant_values=(55, 44))
        anp_p = np.pad(anp, (2, 3), constant_values=(55, 44))
        self.assertEqual(a_p.tolist(), anp_p.tolist())

        a_p = xp.pad(a, 2, constant_values=3)
        anp_p = np.pad(anp, 2, constant_values=3)
        self.assertEqual(a_p.tolist(), anp_p.tolist())
