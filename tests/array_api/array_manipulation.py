import numpy as np
import pytest

import arkouda as ak
import arkouda.array_api as xp


seed = pytest.seed


def randArr(shape):
    global seed
    seed += 2  # ensures that unique values are created each time randArr is invoked
    return xp.asarray(ak.randint(0, 100, shape, dtype=ak.int64, seed=seed))


class TestManipulation:
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_manipulation_functions_docstrings(self):
        import doctest

        from arkouda.array_api import manipulation_functions

        result = doctest.testmod(
            manipulation_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_broadcast(self):
        a = xp.ones((1, 6, 1))
        b = xp.ones((5, 1, 10))
        c = xp.ones((5, 6, 1))
        d = xp.ones((6, 10))

        abcd = xp.broadcast_arrays(a, b, c, d)
        assert len(abcd) == 4
        assert abcd[0].shape == (5, 6, 10)
        assert abcd[1].shape == (5, 6, 10)
        assert abcd[2].shape == (5, 6, 10)
        assert abcd[3].shape == (5, 6, 10)

        assert (abcd[0] == 1).all()
        assert (abcd[1] == 1).all()
        assert (abcd[2] == 1).all()
        assert (abcd[3] == 1).all()

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_concat(self):
        a = randArr((5, 3, 10))
        b = randArr((5, 3, 2))
        c = randArr((5, 3, 17))

        abcConcat = xp.concat([a, b, c], axis=2)
        abcNP = np.concatenate([a.to_ndarray(), b.to_ndarray(), c.to_ndarray()], axis=2)
        assert abcConcat.shape == (5, 3, 29)
        assert abcConcat.tolist() == abcNP.tolist()

        d = randArr((10, 8))
        e = randArr((11, 8))
        f = randArr((12, 8))

        defConcat = xp.concat([d, e, f])
        defNP = np.concatenate([d.to_ndarray(), e.to_ndarray(), f.to_ndarray()])
        assert defConcat.shape, (33, 8)
        assert defConcat.tolist(), defNP.tolist()

        defConcatNeg = xp.concat((d, e, f), axis=-2)
        assert defConcatNeg.shape == (33, 8)
        assert defConcatNeg.tolist() == defNP.tolist()

        h = randArr((1, 2, 3))
        i = randArr((1, 2, 3))
        j = randArr((1, 2, 3))

        hijConcat = xp.concat((h, i, j), axis=None)
        hijNP = np.concatenate([h.to_ndarray(), i.to_ndarray(), j.to_ndarray()], axis=None)
        assert hijConcat.shape == (18,)
        assert hijConcat.tolist() == hijNP.tolist()

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_expand_dims(self):
        a = randArr((5, 3))
        alist = a.tolist()

        a0 = xp.expand_dims(a, axis=0)
        assert a0.shape == (1, 5, 3)
        assert a0[0, ...].tolist() == alist

        a1 = xp.expand_dims(a, axis=1)
        assert a1.shape == (5, 1, 3)
        assert a1[:, 0, :].tolist() == alist

        a2 = xp.expand_dims(a, axis=2)
        assert a2.shape == (5, 3, 1)
        assert a2[..., 0].tolist() == alist

        aNeg1 = xp.expand_dims(a, axis=-1)
        assert aNeg1.shape == (5, 3, 1)
        assert aNeg1[:, :, 0].tolist() == alist

        aNeg2 = xp.expand_dims(a, axis=-2)
        assert aNeg2.shape == (5, 1, 3)
        assert aNeg2[:, 0, :].tolist() == alist

        aNeg3 = xp.expand_dims(a, axis=-3)
        assert aNeg3.shape == (1, 5, 3)
        assert aNeg3[0, :, :].tolist() == alist

        with pytest.raises(IndexError):
            xp.expand_dims(a, axis=3)

        with pytest.raises(IndexError):
            xp.expand_dims(a, axis=-4)

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_flip(self):
        # 1D case
        a = xp.asarray(ak.arange(10))
        b1 = xp.flip(a)
        b2 = xp.flip(a, axis=0)

        assert b1.tolist() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        assert b2.tolist() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        # ND case
        r = xp.asarray(ak.randint(0, 100, (7, 8, 9), dtype=ak.int64, seed=seed))
        rn = np.asarray(r.tolist())

        f1 = xp.flip(r)  # flip all axes
        f2 = xp.flip(r, axis=0)
        f3 = xp.flip(r, axis=1)
        f4 = xp.flip(r, axis=(0, 2))

        assert f1.shape == (7, 8, 9)
        assert f2.shape == (7, 8, 9)
        assert f3.shape == (7, 8, 9)
        assert f4.shape == (7, 8, 9)

        nf1 = np.flip(rn)
        nf2 = np.flip(rn, axis=0)
        nf3 = np.flip(rn, axis=1)
        nf4 = np.flip(rn, axis=(0, 2))

        assert f1.tolist() == nf1.tolist()
        assert f2.tolist() == nf2.tolist()
        assert f3.tolist() == nf3.tolist()
        assert f4.tolist() == nf4.tolist()

        with pytest.raises(IndexError):
            xp.flip(r, axis=3)

        with pytest.raises(IndexError):
            xp.flip(r, axis=-4)

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_permute_dims(self):
        r = randArr((7, 8, 9))

        p1 = xp.permute_dims(r, (0, 1, 2))
        p2 = xp.permute_dims(r, (2, 1, 0))
        p3 = xp.permute_dims(r, (1, 0, 2))

        assert p1.shape == (7, 8, 9)
        assert p2.shape == (9, 8, 7)
        assert p3.shape == (8, 7, 9)

        npr = np.asarray(r.tolist())
        np1 = np.transpose(npr, (0, 1, 2))
        np2 = np.transpose(npr, (2, 1, 0))
        np3 = np.transpose(npr, (1, 0, 2))

        assert p1.tolist() == np1.tolist()
        assert p2.tolist() == np2.tolist()
        assert p3.tolist() == np3.tolist()

        with pytest.raises(IndexError):
            xp.permute_dims(r, (0, 1, 3))

        with pytest.raises(IndexError):
            xp.permute_dims(r, (0, 1, -4))

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_reshape(self):
        r = randArr((2, 6, 12))
        nr = np.asarray(r.tolist())

        for shape in [(12, 12), (3, 12, 4), (2, 72), (6, 2, 12), (144,)]:
            rs = xp.reshape(r, shape)
            nrs = np.reshape(nr, shape)
            assert rs.shape == shape
            assert rs.tolist() == nrs.tolist()

        for shape, inferred in zip([(72, -1), (16, 9, -1), (-1,)], [(72, 2), (16, 9, 1), (144,)]):
            rs = xp.reshape(r, shape)
            nrs = np.reshape(nr, inferred)
            assert rs.shape == inferred
            assert rs.tolist() == nrs.tolist()

        with pytest.raises(ValueError):
            # total # of elements doesn't match
            xp.reshape(r, (2, 73))

        with pytest.raises(ValueError):
            # no whole number size for inferred dimension
            xp.reshape(r, (7, -1))

        with pytest.raises(ValueError):
            # more than one dimension can't be inferred
            xp.reshape(r, (2, -1, -1))

    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_roll(self):
        # 1D case
        a = xp.asarray(ak.arange(10))
        b1 = xp.roll(a, 3)
        b2 = xp.roll(a, -3)

        assert b1.tolist() == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert b2.tolist() == [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]

        # ND case
        r = randArr((7, 8, 9))

        f1 = xp.roll(r, 3)
        f2 = xp.roll(r, -3)
        f3 = xp.roll(r, (3, -2), axis=(0, 2))
        f4 = xp.roll(r, 5, axis=(1, 2))

        assert f1.shape == (7, 8, 9)
        assert f2.shape == (7, 8, 9)
        assert f3.shape == (7, 8, 9)
        assert f4.shape == (7, 8, 9)

        npr = np.asarray(r.tolist())
        np1 = np.roll(npr, 3)
        np2 = np.roll(npr, -3)
        np3 = np.roll(npr, (3, -2), axis=(0, 2))
        np4 = np.roll(npr, 5, axis=(1, 2))

        assert f1.tolist() == np1.tolist()
        assert f2.tolist() == np2.tolist()
        assert f3.tolist() == np3.tolist()
        assert f4.tolist() == np4.tolist()

        with pytest.raises(IndexError):
            xp.roll(r, 3, axis=(0, 1, 2, 3))

        with pytest.raises(IndexError):
            xp.roll(r, 3, axis=3)

        with pytest.raises(IndexError):
            xp.roll(r, 3, axis=-4)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_squeeze(self):
        r1 = randArr((1, 2, 3))
        r2 = randArr((2, 1, 3))
        r3 = randArr((2, 3, 1))
        r4 = randArr((1, 3, 1))

        s1 = xp.squeeze(r1, axis=0)
        s2 = xp.squeeze(r2, axis=1)
        s3 = xp.squeeze(r3, axis=2)
        s4 = xp.squeeze(r4, axis=(0, 2))

        assert s1.shape == (2, 3)
        assert s2.shape == (2, 3)
        assert s3.shape == (2, 3)
        assert s4.shape == (3,)

        nps1 = np.squeeze(np.asarray(r1.tolist()), axis=0)
        nps2 = np.squeeze(np.asarray(r2.tolist()), axis=1)
        nps3 = np.squeeze(np.asarray(r3.tolist()), axis=2)
        nps4 = np.squeeze(np.asarray(r4.tolist()), axis=(0, 2))

        assert s1.tolist() == nps1.tolist()
        assert s2.tolist() == nps2.tolist()
        assert s3.tolist() == nps3.tolist()
        assert s4.tolist() == nps4.tolist()

        with pytest.raises(ValueError):
            xp.squeeze(r1, axis=1)

        with pytest.raises(ValueError):
            xp.squeeze(r4, axis=1)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_stack_unstack(self):
        a = randArr((5, 4))
        b = randArr((5, 4))
        c = randArr((5, 4))

        abcStack0 = xp.stack([a, b, c], axis=0)
        npabcStack0 = np.stack([a.to_ndarray(), b.to_ndarray(), c.to_ndarray()], axis=0)
        assert abcStack0.shape == (3, 5, 4)
        assert abcStack0.tolist() == npabcStack0.tolist()

        (ap, bp, cp) = xp.unstack(abcStack0, axis=0)
        assert ap.tolist() == a.tolist()
        assert bp.tolist() == b.tolist()
        assert cp.tolist() == c.tolist()

        abcStackm1 = xp.stack([a, b, c], axis=-1)
        npabcStackm1 = np.stack([a.to_ndarray(), b.to_ndarray(), c.to_ndarray()], axis=-1)
        assert abcStackm1.shape == (5, 4, 3)
        assert abcStackm1.tolist() == npabcStackm1.tolist()

        (ap, bp, cp) = xp.unstack(abcStackm1, axis=-1)
        assert ap.tolist() == a.tolist()
        assert bp.tolist() == b.tolist()
        assert cp.tolist() == c.tolist()

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    def test_tile(self):
        a = randArr((2, 3))

        for reps in [(2, 1), (1, 2), (2, 2), (1, 1, 3), (3,)]:
            at = xp.tile(a, reps)
            npat = np.tile(np.asarray(a), reps)
            assert at.shape == npat.shape
            assert at.tolist() == npat.tolist()

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_repeat(self):
        a = randArr((5, 10))
        r = randArr((50,))

        ar1 = xp.repeat(a, 2)
        nar1 = np.repeat(np.asarray(a), 2)
        assert ar1.tolist() == nar1.tolist()

        ar2 = xp.repeat(a, r)
        nar2 = np.repeat(np.asarray(a), np.asarray(r))
        assert ar2.tolist() == nar2.tolist()
