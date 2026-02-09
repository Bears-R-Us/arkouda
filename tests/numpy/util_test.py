import numpy as np
import pytest

import arkouda as ak

from arkouda.numpy import util
from arkouda.numpy.util import may_share_memory, shares_memory
from arkouda.testing import assert_arkouda_array_equivalent
from arkouda.util import is_float, is_int, is_numeric, map


class TestUtil:
    @pytest.mark.requires_chapel_module("In1dMsg")
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_util_docstrings(self):
        import doctest

        result = doctest.testmod(util, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.requires_chapel_module("KExtremeMsg")
    def test_sparse_sum_helper(self):
        cfg = ak.get_config()
        N = (10**4) * cfg["numLocales"]
        select_from = ak.arange(N)
        inds1 = select_from[ak.randint(0, 10, N) % 3 == 0]
        inds2 = select_from[ak.randint(0, 10, N) % 3 == 0]
        vals1 = ak.randint(-(2**32), 2**32, N)[inds1]
        vals2 = ak.randint(-(2**32), 2**32, N)[inds2]

        merge_idx, merge_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=True)
        sort_idx, sort_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=False)
        gb_idx, gb_vals = ak.GroupBy(ak.concatenate([inds1, inds2], ordered=False)).sum(
            ak.concatenate((vals1, vals2), ordered=False)
        )

        assert (merge_idx == sort_idx).all()
        assert (merge_idx == gb_idx).all()
        assert (merge_vals == sort_vals).all()

    def test_is_numeric(self):
        strings = ak.array(["a", "b"])
        ints = ak.array([1, 2])
        categoricals = ak.Categorical(strings)
        floats = ak.array([1, np.nan])

        from arkouda.pandas.index import Index
        from arkouda.pandas.series import Series

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            assert not is_numeric(item)

        for item in [
            ints,
            Index(ints),
            Series(ints),
            floats,
            Index(floats),
            Series(floats),
        ]:
            assert is_numeric(item)

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
            floats,
            Index(floats),
            Series(floats),
        ]:
            assert not is_int(item)

        for item in [ints, Index(ints), Series(ints)]:
            assert is_int(item)

        for item in [
            strings,
            Index(strings),
            Series(strings),
            ints,
            Index(ints),
            Series(ints),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            assert not is_float(item)

        for item in [floats, Index(floats), Series(floats)]:
            assert is_float(item)

    @pytest.mark.requires_chapel_module("In1dMsg")
    def test_map(self):
        a = ak.array(["1", "1", "4", "4", "4"])
        b = ak.array([2, 3, 2, 3, 4])
        c = ak.array([1.0, 1.0, 2.2, 2.2, 4.4])
        d = ak.Categorical(a)

        result = map(a, {"4": 25, "5": 30, "1": 7})
        assert result.tolist() == [7, 7, 25, 25, 25]

        result = map(a, {"1": 7})
        assert result.tolist() == ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).tolist()

        result = map(a, {"1": 7.0})
        assert np.allclose(result.tolist(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

        result = map(b, {4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        assert result.tolist() == [30.0, 5.0, 30.0, 5.0, 25.0]

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        assert result.tolist() == ["a", "a", "b", "b", "c"]

        result = map(c, {1.0: "a"})
        assert result.tolist() == ["a", "a", "null", "null", "null"]

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        assert result.tolist() == ["a", "a", "b", "b", "c"]

        result = map(d, {"4": 25, "5": 30, "1": 7})
        assert result.tolist() == [7, 7, 25, 25, 25]

        result = map(d, {"1": 7})
        assert np.allclose(
            result.tolist(),
            ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).tolist(),
            equal_nan=True,
        )

        result = map(d, {"1": 7.0})
        assert np.allclose(result.tolist(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

    @pytest.mark.parametrize("dtype", [ak.int64, ak.float64, ak.bool_, ak.bigint, ak.str_])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_copy(self, dtype, size):
        a = ak.arange(size, dtype=dtype)
        b = ak.numpy.util.copy(a)

        from arkouda import assert_equal as ak_assert_equal

        assert a is not b
        ak_assert_equal(a, b)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_broadcast_shapes(self, size):
        chop = size // 4
        a = ak.arange(chop).reshape(1, 1, chop)
        b = ak.arange(2 * chop).reshape(1, 2, chop)
        c = ak.arange(2 * chop).reshape(2, 1, chop)
        bigshape = ak.broadcast_shapes(a.shape, b.shape, c.shape)
        assert bigshape == (2, 2, chop)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_broadcast_to(self, size):
        pd = ak.broadcast_to(1, size)
        nd = np.broadcast_to(1, size)
        assert_arkouda_array_equivalent(pd, nd)
        chop = size // 4
        pd = ak.broadcast_to(1, (2, chop))
        nd = np.broadcast_to(1, (2, chop))
        assert_arkouda_array_equivalent(pd, nd)
        a = ak.arange(chop).reshape(1, 1, chop)
        b = ak.arange(2 * chop).reshape(1, 2, chop)
        c = ak.arange(2 * chop).reshape(2, 1, chop)
        bigshape = ak.broadcast_shapes(a.shape, b.shape, c.shape)
        pd = ak.broadcast_to(c, bigshape)
        nd = np.broadcast_to(c.to_ndarray(), bigshape)
        assert_arkouda_array_equivalent(pd, nd)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_broadcast_arrays(self, size):
        chop = size // 4
        a = ak.arange(chop).reshape(1, 1, chop)
        b = ak.arange(2 * chop).reshape(1, 2, chop)
        c = ak.arange(2 * chop).reshape(2, 1, chop)
        pe = ak.broadcast_arrays(a, b, c)
        ne = np.broadcast_arrays(a.to_ndarray(), b.to_ndarray(), c.to_ndarray())
        assert_arkouda_array_equivalent(pe[0], ne[0])
        assert_arkouda_array_equivalent(pe[1], ne[1])
        assert_arkouda_array_equivalent(pe[2], ne[2])


@pytest.mark.parametrize("n", [0, 1, 5, 1024])
def test_pdarray_identity_and_copy(n):
    a = ak.arange(n)
    b = a  # alias
    assert shares_memory(a, b)
    assert may_share_memory(a, b)

    c = a + 0  # materialized new buffer
    assert not shares_memory(a, c)
    assert not may_share_memory(a, c)

    # slicing should materialize (no views today)
    d = a[::2]
    assert not shares_memory(a, d)
    assert not may_share_memory(a, d)


def test_pdarray_after_reassignment_breaks_alias():
    a = ak.arange(10)
    b = a
    assert shares_memory(a, b)
    # rebind a to a new server buffer
    a = a + 1
    assert not shares_memory(a, b)


def test_gather_materializes():
    a = ak.arange(10)
    idx = ak.array([0, 2, 4, 6, 8])
    g = a[idx]
    assert not shares_memory(a, g)
    assert not may_share_memory(a, g)


def test_strings_identity_and_slice():
    s = ak.array(["a", "bb", "ccc", "dddd"])
    s_alias = s
    assert shares_memory(s, s_alias)
    assert may_share_memory(s, s_alias)

    s_tail = s[1:]  # should materialize new offsets/bytes
    assert not shares_memory(s, s_tail)
    assert not may_share_memory(s, s_tail)


def test_categorical_components_share_with_self():
    s = ak.array(["x", "y", "x", "z", "y"])
    cat = ak.Categorical(s)

    # Components share with themselves
    assert shares_memory(cat.codes, cat.codes)
    assert shares_memory(cat.categories, cat.categories)

    # The Categorical wrapper should report sharing with its own components
    assert shares_memory(cat, cat.codes)
    assert shares_memory(cat, cat.categories)
    assert shares_memory(cat, cat.permutation)
    assert shares_memory(cat, cat.segments)

    # But not necessarily with the original Strings (don't assert either way).
    # We only guarantee that different freshly materialized objects don't share.
    s_copy = s[::2]
    assert not shares_memory(s_copy, cat.codes)
    # categories may or may not share with s; no assertion here


def test_segarray_components_and_aliasing():
    segs = ak.array([0, 2, 5])
    vals = ak.array([10, 11, 20, 21, 22])
    from arkouda.segarray import SegArray

    sa = SegArray(segs, vals)
    # SegArray reports sharing with its own components
    assert shares_memory(sa, segs)
    assert shares_memory(sa, vals)

    # Aliasing the SegArray should share
    sa2 = sa
    assert shares_memory(sa, sa2)

    # New construction with copied buffers should not share
    segs2 = segs + 0
    vals2 = vals + 0
    sa_new = SegArray(segs2, vals2)
    assert not shares_memory(sa_new, sa)
    assert not shares_memory(sa_new, segs)
    assert not shares_memory(sa_new, vals)


def test_nested_containers_detect_shared_pdarray():
    a = ak.arange(5)
    nested1 = (a, {"k": 1})
    nested2 = {"x": a, "y": [42]}
    assert shares_memory(nested1, nested2)
    assert may_share_memory(nested1, nested2)

    other = ak.arange(5) + 1
    nested3 = (other, {"k": 1})
    assert not shares_memory(nested1, nested3)
    assert not may_share_memory(nested1, nested3)


def test_disparate_types_and_python_scalars():
    a = ak.arange(3)
    assert not shares_memory(a, 123)
    assert not shares_memory("abc", a)
    assert not shares_memory(None, None)
    assert not may_share_memory(a, 123)


def test_empty_and_size_one_arrays():
    empty = ak.arange(0)
    one = ak.arange(1)
    assert not shares_memory(empty, one)
    # alias still shares though
    alias = empty
    assert shares_memory(empty, alias)


def test_idempotence_and_symmetry():
    a = ak.arange(7)
    b = a + 0
    # symmetry
    assert shares_memory(a, a)
    assert shares_memory(a, a) == shares_memory(a, a)  # idempotent
    assert shares_memory(a, b) == shares_memory(b, a)
    assert may_share_memory(a, b) == may_share_memory(b, a)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: ak.arange(5),
        lambda: ak.array(["a", "bb", "c"]),
        lambda: ak.Categorical(ak.array(["t", "t", "f"])),
    ],
)
def test_aliasing_true_for_same_object(builder):
    obj = builder()
    alias = obj
    assert shares_memory(obj, alias)
    assert may_share_memory(obj, alias)
