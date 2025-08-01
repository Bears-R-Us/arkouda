import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

import arkouda.array_api as xp

SEED = 12345
s = SEED


def randArr(shape):
    global s
    s += 2
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=s))


class ManipulationTests(ArkoudaTest):
    def test_broadcast(self):
        a = xp.ones((1, 6, 1))
        b = xp.ones((5, 1, 10))
        c = xp.ones((5, 6, 1))
        d = xp.ones((6, 10))

        abcd = xp.broadcast_arrays(a, b, c, d)
        self.assertEqual(len(abcd), 4)
        self.assertEqual(abcd[0].shape, (5, 6, 10))
        self.assertEqual(abcd[1].shape, (5, 6, 10))
        self.assertEqual(abcd[2].shape, (5, 6, 10))
        self.assertEqual(abcd[3].shape, (5, 6, 10))

        self.assertTrue((abcd[0] == 1).all())
        self.assertTrue((abcd[1] == 1).all())
        self.assertTrue((abcd[2] == 1).all())
        self.assertTrue((abcd[3] == 1).all())

    def test_concat(self):
        a = randArr((5, 3, 10))
        b = randArr((5, 3, 2))
        c = randArr((5, 3, 17))

        abcConcat = xp.concat([a, b, c], axis=2)
        abcNP = np.concatenate([a.to_ndarray(), b.to_ndarray(), c.to_ndarray()], axis=2)
        self.assertEqual(abcConcat.shape, (5, 3, 29))
        self.assertEqual(abcConcat.tolist(), abcNP.tolist())

        d = randArr((10, 8))
        e = randArr((11, 8))
        f = randArr((12, 8))

        defConcat = xp.concat([d, e, f])
        defNP = np.concatenate([d.to_ndarray(), e.to_ndarray(), f.to_ndarray()])
        self.assertEqual(defConcat.shape, (33, 8))
        self.assertEqual(defConcat.tolist(), defNP.tolist())

        defConcatNeg = xp.concat((d, e, f), axis=-2)
        self.assertEqual(defConcatNeg.shape, (33, 8))
        self.assertEqual(defConcatNeg.tolist(), defNP.tolist())

        h = randArr((1, 2, 3))
        i = randArr((1, 2, 3))
        j = randArr((1, 2, 3))

        hijConcat = xp.concat((h, i, j), axis=None)
        hijNP = np.concatenate([h.to_ndarray(), i.to_ndarray(), j.to_ndarray()], axis=None)
        self.assertEqual(hijConcat.shape, (18,))
        self.assertEqual(hijConcat.tolist(), hijNP.tolist())

    def test_expand_dims(self):
        a = randArr((5, 3))
        alist = a.tolist()

        a0 = xp.expand_dims(a, axis=0)
        self.assertEqual(a0.shape, (1, 5, 3))
        self.assertEqual(a0[0, ...].tolist(), alist)

        a1 = xp.expand_dims(a, axis=1)
        self.assertEqual(a1.shape, (5, 1, 3))
        self.assertEqual(a1[:, 0, :].tolist(), alist)

        a2 = xp.expand_dims(a, axis=2)
        self.assertEqual(a2.shape, (5, 3, 1))
        self.assertEqual(a2[..., 0].tolist(), alist)

        aNeg1 = xp.expand_dims(a, axis=-1)
        self.assertEqual(aNeg1.shape, (5, 3, 1))
        self.assertEqual(aNeg1[:, :, 0].tolist(), alist)

        aNeg2 = xp.expand_dims(a, axis=-2)
        self.assertEqual(aNeg2.shape, (5, 1, 3))
        self.assertEqual(aNeg2[:, 0, :].tolist(), alist)

        aNeg3 = xp.expand_dims(a, axis=-3)
        self.assertEqual(aNeg3.shape, (1, 5, 3))
        self.assertEqual(aNeg3[0, :, :].tolist(), alist)

        with self.assertRaises(IndexError):
            xp.expand_dims(a, axis=3)

        with self.assertRaises(IndexError):
            xp.expand_dims(a, axis=-4)

    def test_flip(self):
        # 1D case
        a = xp.arange(10)
        b1 = xp.flip(a)
        b2 = xp.flip(a, axis=0)

        self.assertEqual(b1.tolist(), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(b2.tolist(), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        # ND case
        r = xp.asarray(ak.randint(0, 100, (7, 8, 9), dtype=ak.int64, seed=SEED))
        rn = np.asarray(r.tolist())

        f1 = xp.flip(r)  # flip all axes
        f2 = xp.flip(r, axis=0)
        f3 = xp.flip(r, axis=1)
        f4 = xp.flip(r, axis=(0, 2))

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
            xp.flip(r, axis=3)

        with self.assertRaises(IndexError):
            xp.flip(r, axis=-4)

    def test_permute_dims(self):
        r = randArr((7, 8, 9))

        p1 = xp.permute_dims(r, (0, 1, 2))
        p2 = xp.permute_dims(r, (2, 1, 0))
        p3 = xp.permute_dims(r, (1, 0, 2))

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
            xp.permute_dims(r, (0, 1, 3))

        with self.assertRaises(IndexError):
            xp.permute_dims(r, (0, 1, -4))

    def test_reshape(self):
        r = randArr((2, 6, 12))
        nr = np.asarray(r.tolist())

        for shape in [(12, 12), (3, 12, 4), (2, 72), (6, 2, 12), (144,)]:
            rs = xp.reshape(r, shape)
            nrs = np.reshape(nr, shape)
            self.assertEqual(rs.shape, shape)
            self.assertEqual(rs.tolist(), nrs.tolist())

        for shape, inferred in zip([(72, -1), (16, 9, -1), (-1,)], [(72, 2), (16, 9, 1), (144,)]):
            rs = xp.reshape(r, shape)
            nrs = np.reshape(nr, inferred)
            self.assertEqual(rs.shape, inferred)
            self.assertEqual(rs.tolist(), nrs.tolist())

        with self.assertRaises(ValueError):
            # total # of elements doesn't match
            xp.reshape(r, (2, 73))

        with self.assertRaises(ValueError):
            # no whole number size for inferred dimension
            xp.reshape(r, (7, -1))

        with self.assertRaises(ValueError):
            # more than one dimension can't be inferred
            xp.reshape(r, (2, -1, -1))

    def test_roll(self):
        # 1D case
        a = xp.arange(10)
        b1 = xp.roll(a, 3)
        b2 = xp.roll(a, -3)

        self.assertEqual(b1.tolist(), [7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(b2.tolist(), [3, 4, 5, 6, 7, 8, 9, 0, 1, 2])

        # ND case
        r = randArr((7, 8, 9))

        f1 = xp.roll(r, 3)
        f2 = xp.roll(r, -3)
        f3 = xp.roll(r, (3, -2), axis=(0, 2))
        f4 = xp.roll(r, 5, axis=(1, 2))

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
            xp.roll(r, 3, axis=(0, 1, 2, 3))

        with self.assertRaises(IndexError):
            xp.roll(r, 3, axis=3)

        with self.assertRaises(IndexError):
            xp.roll(r, 3, axis=-4)

    def test_squeeze(self):
        r1 = randArr((1, 2, 3))
        r2 = randArr((2, 1, 3))
        r3 = randArr((2, 3, 1))
        r4 = randArr((1, 3, 1))

        s1 = xp.squeeze(r1, axis=0)
        s2 = xp.squeeze(r2, axis=1)
        s3 = xp.squeeze(r3, axis=2)
        s4 = xp.squeeze(r4, axis=(0, 2))

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
            xp.squeeze(r1, axis=1)

        with self.assertRaises(ValueError):
            xp.squeeze(r4, axis=1)

    def test_stack_unstack(self):
        a = randArr((5, 4))
        b = randArr((5, 4))
        c = randArr((5, 4))

        abcStack0 = xp.stack([a, b, c], axis=0)
        npabcStack0 = np.stack([a.to_ndarray(), b.to_ndarray(), c.to_ndarray()], axis=0)
        self.assertEqual(abcStack0.shape, (3, 5, 4))
        self.assertEqual(abcStack0.tolist(), npabcStack0.tolist())

        (ap, bp, cp) = xp.unstack(abcStack0, axis=0)
        self.assertEqual(ap.tolist(), a.tolist())
        self.assertEqual(bp.tolist(), b.tolist())
        self.assertEqual(cp.tolist(), c.tolist())

        abcStackm1 = xp.stack([a, b, c], axis=-1)
        npabcStackm1 = np.stack([a.to_ndarray(), b.to_ndarray(), c.to_ndarray()], axis=-1)
        self.assertEqual(abcStackm1.shape, (5, 4, 3))
        self.assertEqual(abcStackm1.tolist(), npabcStackm1.tolist())

        (ap, bp, cp) = xp.unstack(abcStackm1, axis=-1)
        self.assertEqual(ap.tolist(), a.tolist())
        self.assertEqual(bp.tolist(), b.tolist())
        self.assertEqual(cp.tolist(), c.tolist())

    def test_tile(self):
        a = randArr((2, 3))

        print(a)

        for reps in [(2, 1), (1, 2), (2, 2), (1, 1, 3), (3,)]:
            at = xp.tile(a, reps)
            npat = np.tile(np.asarray(a), reps)
            self.assertEqual(at.shape, npat.shape)
            self.assertEqual(at.tolist(), npat.tolist())

    def test_repeat(self):
        a = randArr((5, 10))
        r = randArr((50,))

        ar1 = xp.repeat(a, 2)
        nar1 = np.repeat(np.asarray(a), 2)
        self.assertEqual(ar1.tolist(), nar1.tolist())

        ar2 = xp.repeat(a, r)
        nar2 = np.repeat(np.asarray(a), np.asarray(r))
        self.assertEqual(ar2.tolist(), nar2.tolist())
