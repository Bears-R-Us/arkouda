import numpy as np

import arkouda as ak
from arkouda.util import is_float, is_int, is_numeric, map


class TestUtil:
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

        from arkouda.series import Series
        from arkouda.index import Index

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            assert not is_numeric(item)

        for item in [ints, Index(ints), Series(ints), floats, Index(floats), Series(floats)]:
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

    def test_map(self):
        a = ak.array(["1", "1", "4", "4", "4"])
        b = ak.array([2, 3, 2, 3, 4])
        c = ak.array([1.0, 1.0, 2.2, 2.2, 4.4])
        d = ak.Categorical(a)

        result = map(a, {"4": 25, "5": 30, "1": 7})
        assert result.to_list() == [7, 7, 25, 25, 25]

        result = map(a, {"1": 7})
        assert (
            result.to_list() == ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list()
        )

        result = map(a, {"1": 7.0})
        assert np.allclose(result.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

        result = map(b, {4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        assert result.to_list() == [30.0, 5.0, 30.0, 5.0, 25.0]

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        assert result.to_list() == ["a", "a", "b", "b", "c"]

        result = map(c, {1.0: "a"})
        assert result.to_list() == ["a", "a", "null", "null", "null"]

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        assert result.to_list() == ["a", "a", "b", "b", "c"]

        result = map(d, {"4": 25, "5": 30, "1": 7})
        assert result.to_list() == [7, 7, 25, 25, 25]

        result = map(d, {"1": 7})
        assert np.allclose(
            result.to_list(),
            ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list(),
            equal_nan=True,
        )

        result = map(d, {"1": 7.0})
        assert np.allclose(result.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)
