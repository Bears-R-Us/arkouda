import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import arkouda as ak

DTYPES = [ak.int64, ak.uint64, ak.bool, ak.float64, ak.bigint, ak.str_]
NO_STRING = [ak.int64, ak.uint64, ak.bool, ak.float64, ak.bigint]
NUMERICAL_TYPES = [ak.int64, ak.uint64, ak.float64, ak.bigint]
INTEGRAL_TYPES = [ak.int64, ak.uint64, ak.bool, ak.bigint]


class TestSeries:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_series_creation(self, dtype):
        idx = ak.arange(3, dtype=dtype)
        for val in idx, ak.array(["A", "B", "C"]):
            ans = ak.Series(data=val, index=idx).to_list()
            for series in (
                ak.Series(data=val, index=idx),
                ak.Series(data=val),
                ak.Series(val, idx),
                ak.Series(val),
                ak.Series((idx, val)),
            ):
                assert isinstance(series, ak.Series)
                assert isinstance(series.index, ak.Index)
                assert series.to_list() == ans

        with pytest.raises(TypeError):
            ak.Series(index=idx)

        with pytest.raises(TypeError):
            ak.Series((ak.arange(3),))

        with pytest.raises(TypeError):
            ak.Series()

        with pytest.raises(ValueError):
            ak.Series(data=ak.arange(3), index=ak.arange(6))

    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    @pytest.mark.parametrize("dtype_index", [ak.int64, ak.uint64])
    def test_lookup(self, dtype, dtype_index):
        pda = ak.arange(3, dtype=dtype)
        for val in pda, ak.array(["A", "B", "C"]):
            s = ak.Series(data=val, index=ak.arange(3, dtype=dtype_index))

            for key in (
                1,
                ak.Index(ak.array([1], dtype=dtype_index)),
                ak.Index(ak.array([0, 2], dtype=dtype_index)),
            ):
                lk = s.locate(key)
                assert isinstance(lk, ak.Series)
                key = ak.array(key) if not isinstance(key, int) else key
                assert (lk.index == s.index[key]).all()
                assert (lk.values == s.values[key]).all()

            # testing multi-index lookup
            mi = ak.MultiIndex([pda, pda[::-1]])
            s = ak.Series(data=val, index=mi)
            lk = s.locate(mi[0])
            assert isinstance(lk, ak.Series)
            assert lk.index.index == mi[0].index
            assert lk.values[0] == val[0]

            # ensure error with scalar and multi-index
            with pytest.raises(TypeError):
                s.locate(0)

            with pytest.raises(TypeError):
                s.locate([0, 2])

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NO_STRING)
    def test_add(self, prob_size, dtype):
        added = ak.Series(ak.arange(prob_size // 2, dtype=dtype)).add(
            ak.Series(
                data=ak.arange(prob_size // 2, prob_size, dtype=dtype),
                index=ak.arange(prob_size // 2, prob_size),
            )
        )
        assert (added.index == ak.arange(prob_size)).all()
        assert (added.values == ak.arange(prob_size, dtype=dtype)).all()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    def test_topn(self, prob_size, dtype):
        top = ak.Series(ak.arange(100, dtype=dtype)).topn(50)
        assert top.values.to_list() == list(range(99, 49, -1))
        assert top.index.to_list() == list(range(99, 49, -1))

    @pytest.mark.parametrize("dtype", NUMERICAL_TYPES)
    @pytest.mark.parametrize("dtype_index", NUMERICAL_TYPES)
    def test_sort(self, dtype, dtype_index):
        def gen_perm(n):
            idx_left = set(range(n))
            perm = np.arange(n)
            while len(idx_left) >= 2:
                inds = np.random.choice(list(idx_left), 2, replace=False)
                tmp = perm[inds[0]]
                perm[inds[0]] = perm[inds[1]]
                perm[inds[1]] = tmp
                idx_left.remove(inds[0])
                idx_left.remove(inds[1])
            return perm

        ordered = ak.arange(100, dtype=dtype)
        perm = ak.array(gen_perm(100), dtype=dtype_index)

        idx_sort = ak.Series(data=ordered, index=perm).sort_index()
        assert idx_sort.index.to_list() == ordered.to_list()
        assert idx_sort.values.to_list() == perm.to_list()

        val_sort = ak.Series(data=perm, index=ordered).sort_values()
        assert val_sort.index.to_pandas().tolist() == perm.to_list()
        assert val_sort.values.to_list() == ordered.to_list()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_head_tail(self, dtype):
        size = 100
        s = ak.Series(ak.arange(size, dtype=dtype))

        head = s.head(size)
        assert head.index.to_list() == list(range(size))
        assert head.values.to_list() == ak.arange(size, dtype=dtype).to_list()

        tail = s.tail(size)
        assert tail.index.to_list() == ak.arange(size)[-size:size].to_list()
        assert tail.values.to_list() == ak.arange(size, dtype=dtype)[-size:size].to_list()

    def test_value_counts(self):
        s = ak.Series(ak.array([1, 2, 0, 2, 0]))

        c = s.value_counts()
        assert c.index.to_list() == [0, 2, 1]
        assert c.values.to_list() == [2, 2, 1]

        c = s.value_counts(sort=False)
        assert c.index.to_list() == list(range(3))
        assert c.values.to_list() == [2, 1, 2]

    def test_concat(self):
        s = ak.Series(ak.arange(5))
        s2 = ak.Series(ak.arange(5, 11), ak.arange(5, 11))
        s3 = ak.Series(ak.arange(5, 10), ak.arange(5, 10))

        df = ak.Series.concat([s, s2], axis=1)
        assert isinstance(df, ak.DataFrame)

        ref_df = pd.DataFrame(
            {
                "idx": list(range(11)),
                "val_0": [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
                "val_1": [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10],
            }
        )
        assert_frame_equal(ref_df, df.to_pandas())

        def list_helper(arr):
            return arr.to_list() if isinstance(arr, (ak.pdarray, ak.Index)) else arr.tolist()

        for fname in "concat", "pdconcat":
            func = getattr(ak.Series, fname)
            c = func([s, s2])
            assert list_helper(c.index) == list(range(11))
            assert list_helper(c.values) == list(range(11))

            df = func([s, s3], axis=1)
            if fname == "concat":
                ref_df = pd.DataFrame(
                    {"idx": [0, 1, 2, 3, 4], "val_0": [0, 1, 2, 3, 4], "val_1": [5, 6, 7, 8, 9]}
                )
                assert isinstance(df, ak.DataFrame)
                assert_frame_equal(ref_df, df.to_pandas())
            else:
                ref_df = pd.DataFrame({0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})
                assert isinstance(df, pd.DataFrame)
                assert_frame_equal(ref_df, df)

    def test_index_as_index_compat(self):
        # added to validate functionality for issue #1506
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        g = df.groupby(["a", "b"])
        g.broadcast(g.sum("c"))
