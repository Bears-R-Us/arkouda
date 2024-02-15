import unittest

from base_test import ArkoudaTest
from context import arkouda as ak
import arkouda.array_api as Array
import numpy as np

SEED = 12345

# requires the server to be built with 3D array support

class ManipulationTests(ArkoudaTest):
    def test_broadcast(self):
        a = Array.ones((1, 6, 1))
        b = Array.ones((5, 1, 10))
        c = Array.ones((5, 6, 1))

        abc = Array.broadcast_arrays(a, b, c)
        self.assertEqual(len(abc), 3)
        self.assertEqual(abc[0].shape, (5, 6, 10))
        self.assertEqual(abc[1].shape, (5, 6, 10))
        self.assertEqual(abc[2].shape, (5, 6, 10))

    def test_concat(self):
        a = Array.ones((5, 3, 10))
        b = Array.ones((5, 3, 2))
        c = Array.ones((5, 3, 17))

        abcConcat = Array.concat([a, b, c], axis=2)
        self.assertEqual(abcConcat.shape, (5, 3, 29))
        self.assertTrue(Array.all(abcConcat))

        d = Array.ones((10, 8))
        e = Array.ones((11, 8))
        f = Array.ones((12, 8))

        defConcat = Array.concat([d, e, f])
        self.assertEqual(defConcat.shape, (33, 8))
        self.assertTrue(Array.all(defConcat))

        defConcatNeg = Array.concat((d, e, f), axis=-2)
        self.assertEqual(defConcatNeg.shape, (33, 8))
        self.assertTrue(Array.all(defConcatNeg))

        h = Array.ones((1, 2, 3))
        i = Array.ones((1, 2, 3))
        j = Array.ones((1, 2, 3))

        hijConcat = Array.concat((h, i, j), axis=None)
        self.assertEqual(hijConcat.shape, (18,))
        self.assertTrue(Array.all(hijConcat))

    def test_expand_dims(self):
        a = Array.asarray(ak.randint(0, 100, (5, 3), dtype=ak.int64, seed=SEED))
        alist = a.tolist()

        # TODO: once rank reducing slices are implemented,
        # the squeeze operations can be removed below:

        a0 = Array.expand_dims(a, axis=0)
        self.assertEqual(a0.shape, (1, 5, 3))
        self.assertEqual(Array.squeeze(a0[0, :, :], axis=0).tolist(), alist)

        a1 = Array.expand_dims(a, axis=1)
        self.assertEqual(a1.shape, (5, 1, 3))
        self.assertEqual(Array.squeeze(a1[:, 0, :], axis=1).tolist(), alist)

        a2 = Array.expand_dims(a, axis=2)
        self.assertEqual(a2.shape, (5, 3, 1))
        self.assertEqual(Array.squeeze(a2[:, :, 0], axis=2).tolist(), alist)

        aNeg1 = Array.expand_dims(a, axis=-1)
        self.assertEqual(aNeg1.shape, (5, 3, 1))
        self.assertEqual(Array.squeeze(aNeg1[:, :, 0], axis=2).tolist(), alist)

        aNeg2 = Array.expand_dims(a, axis=-2)
        self.assertEqual(aNeg2.shape, (5, 1, 3))
        self.assertEqual(Array.squeeze(aNeg2[:, 0, :], axis=1).tolist(), alist)

        aNeg3 = Array.expand_dims(a, axis=-3)
        self.assertEqual(aNeg3.shape, (1, 5, 3))
        self.assertEqual(Array.squeeze(aNeg3[0, :, :], axis=0).tolist(), alist)

        with self.assertRaises(IndexError):
            Array.expand_dims(a, axis=3)

        with self.assertRaises(IndexError):
            Array.expand_dims(a, axis=-4)

    def test_flip(self):
        # 1D case
        a = Array.arange(10)
        b1 = Array.flip(a)
        b2 = Array.flip(a, axis=0)

        self.assertEqual(b1.tolist(), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(b2.tolist(), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        # ND case
        r = Array.asarray(ak.randint(0, 100, (7, 8, 9), dtype=ak.int64, seed=SEED))
        rn = np.asarray(r.tolist())

        f1 = Array.flip(r) # flip all axes
        f2 = Array.flip(r, axis=0)
        f3 = Array.flip(r, axis=1)
        f4 = Array.flip(r, axis=(0, 2))

        self.assertEqual(f1.shape, (7, 8, 9))
        self.assertEqual(f2.shape, (7, 8, 9))
        self.assertEqual(f3.shape, (7, 8, 9))
        self.assertEqual(f4.shape, (7, 8, 9))

        nf1 = np.flip(rn)
        nf2 = np.flip(rn, axis=0)
        nf3 = np.flip(rn, axis=1)
        nf4 = np.flip(rn, axis=(0, 2))

        self.assertEqual(f1.tolist(), nf1.tolist())
        self.assertEqual(f2.tolist(), nf2.tolist())
        self.assertEqual(f3.tolist(), nf3.tolist())
        self.assertEqual(f4.tolist(), nf4.tolist())

        with self.assertRaises(IndexError):
            Array.flip(r, axis=3)

        with self.assertRaises(IndexError):
            Array.flip(r, axis=-4)

    def test_permute_dims(self):
        r = Array.asarray(ak.randint(0, 100, (7, 8, 9), dtype=ak.int64, seed=SEED))

        p1 = Array.permute_dims(r, (0, 1, 2))
        p2 = Array.permute_dims(r, (2, 1, 0))
        p3 = Array.permute_dims(r, (1, 0, 2))

        self.assertEqual(p1.shape, (7, 8, 9))
        self.assertEqual(p2.shape, (9, 8, 7))
        self.assertEqual(p3.shape, (8, 7, 9))

        npr = np.asarray(r.tolist())
        np1 = np.transpose(npr, (0, 1, 2))
        np2 = np.transpose(npr, (2, 1, 0))
        np3 = np.transpose(npr, (1, 0, 2))

        self.assertEqual(p1.tolist(), np1.tolist())
        self.assertEqual(p2.tolist(), np2.tolist())
        self.assertEqual(p3.tolist(), np3.tolist())

        with self.assertRaises(IndexError):
            Array.permute_dims(r, (0, 1, 3))

        with self.assertRaises(IndexError):
            Array.permute_dims(r, (0, 1, -4))

    def test_reshape(self):
        r = Array.asarray(ak.randint(0, 100, (2, 6, 12), dtype=ak.int64, seed=SEED))
        nr = np.asarray(r.tolist())

        for shape in [(12, 12), (3, 12, 4), (2, 72), (6, 2, 12), (144,)]:
            rs = Array.reshape(r, shape)
            nrs = np.reshape(nr, shape)
            self.assertEqual(rs.shape, shape)
            self.assertEqual(rs.tolist(), nrs.tolist())

        for shape, inferred in zip([(72, -1), (16, 9, -1), (-1,)], [(72, 2), (16, 9, 1), (144,)]):
            rs = Array.reshape(r, shape)
            nrs = np.reshape(nr, inferred)
            self.assertEqual(rs.shape, inferred)
            self.assertEqual(rs.tolist(), nrs.tolist())

        with self.assertRaises(ValueError):
            # total # of elements doesn't match
            Array.reshape(r, (2, 73))

        with self.assertRaises(ValueError):
            # no whole number size for inferred dimension
            Array.reshape(r, (7, -1))

        with self.assertRaises(ValueError):
            # more than one dimension can't be inferred
            Array.reshape(r, (2, -1, -1))

    def test_roll(self):
        # 1D case
        a = Array.arange(10)
        b1 = Array.roll(a, 3)
        b2 = Array.roll(a, -3)

        self.assertEqual(b1.tolist(), [7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(b2.tolist(), [3, 4, 5, 6, 7, 8, 9, 0, 1, 2])

        # ND case
        r = Array.asarray(ak.randint(0, 100, (7, 8, 9), dtype=ak.int64, seed=SEED))

        f1 = Array.roll(r, 3)
        f2 = Array.roll(r, -3)
        f3 = Array.roll(r, (3, -2), axis=(0, 2))
        f4 = Array.roll(r, 5, axis=(1, 2))

        self.assertEqual(f1.shape, (7, 8, 9))
        self.assertEqual(f2.shape, (7, 8, 9))
        self.assertEqual(f3.shape, (7, 8, 9))
        self.assertEqual(f4.shape, (7, 8, 9))

        npr = np.asarray(r.tolist())
        np1 = np.roll(npr, 3)
        np2 = np.roll(npr, -3)
        np3 = np.roll(npr, (3, -2), axis=(0, 2))
        np4 = np.roll(npr, 5, axis=(1, 2))

        self.assertEqual(f1.tolist(), np1.tolist())
        self.assertEqual(f2.tolist(), np2.tolist())
        self.assertEqual(f3.tolist(), np3.tolist())
        self.assertEqual(f4.tolist(), np4.tolist())

        with self.assertRaises(IndexError):
            Array.roll(r, 3, axis=(0, 1, 2, 3))

        with self.assertRaises(IndexError):
            Array.roll(r, 3, axis=3)

        with self.assertRaises(IndexError):
            Array.roll(r, 3, axis=-4)

    def test_squeeze(self):
        r1 = Array.asarray(ak.randint(0, 100, (1, 2, 3), dtype=ak.int64, seed=SEED))
        r2 = Array.asarray(ak.randint(0, 100, (2, 1, 3), dtype=ak.int64, seed=SEED))
        r3 = Array.asarray(ak.randint(0, 100, (2, 3, 1), dtype=ak.int64, seed=SEED))
        r4 = Array.asarray(ak.randint(0, 100, (1, 3, 1), dtype=ak.int64, seed=SEED))

        s1 = Array.squeeze(r1, axis=0)
        s2 = Array.squeeze(r2, axis=1)
        s3 = Array.squeeze(r3, axis=2)
        s4 = Array.squeeze(r4, axis=(0, 2))

        self.assertEqual(s1.shape, (2, 3))
        self.assertEqual(s2.shape, (2, 3))
        self.assertEqual(s3.shape, (2, 3))
        self.assertEqual(s4.shape, (3,))

        nps1 = np.squeeze(np.asarray(r1.tolist()), axis=0)
        nps2 = np.squeeze(np.asarray(r2.tolist()), axis=1)
        nps3 = np.squeeze(np.asarray(r3.tolist()), axis=2)
        nps4 = np.squeeze(np.asarray(r4.tolist()), axis=(0, 2))

        self.assertEqual(s1.tolist(), nps1.tolist())
        self.assertEqual(s2.tolist(), nps2.tolist())
        self.assertEqual(s3.tolist(), nps3.tolist())
        self.assertEqual(s4.tolist(), nps4.tolist())

        with self.assertRaises(ValueError):
            Array.squeeze(r1, axis=1)

        with self.assertRaises(ValueError):
            Array.squeeze(r4, axis=1)
