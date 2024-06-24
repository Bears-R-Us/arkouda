import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import arkouda as ak
from arkouda.series import Series

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

    @pytest.mark.parametrize("dtype", NO_STRING)
    def test_add(self, dtype):
        size = 100
        added = ak.Series(ak.arange(size // 2, dtype=dtype)).add(
            ak.Series(
                data=ak.arange(size // 2, size, dtype=dtype),
                index=ak.arange(size // 2, size),
            )
        )
        assert (added.index == ak.arange(size)).all()
        if dtype != ak.bool:
            assert all(i in added.values.to_list() for i in range(size))
        else:
            # we have exactly one False
            assert added.values.sum() == 99

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    def test_topn(self, dtype):
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
        n = 10
        s = ak.Series(ak.arange(n, dtype=dtype))
        for i in range(n):
            head = s.head(i)
            assert head.index.to_list() == list(range(i))
            assert head.values.to_list() == ak.arange(i, dtype=dtype).to_list()

            tail = s.tail(i)
            assert tail.index.to_list() == ak.arange(n)[-i:n].to_list()
            assert tail.values.to_list() == ak.arange(n, dtype=dtype)[-i:n].to_list()

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
        # added data= and index= clarifiers to the next two lines.
        # without them, the index was interpreted as the name.
        s2 = ak.Series(data=ak.arange(5, 11), index=ak.arange(5, 11))
        s3 = ak.Series(data=ak.arange(5, 10), index=ak.arange(5, 10))

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
        series = ak.Series(data=g.sum("c")["c"], index=g.sum("c").index)
        g.broadcast(series)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_memory_usage(self, size):
        s = ak.Series(ak.arange(size))
        assert s.memory_usage(unit="GB", index=False) == size * ak.dtypes.int64.itemsize / (
            1024 * 1024 * 1024
        )
        assert s.memory_usage(unit="MB", index=False) == size * ak.dtypes.int64.itemsize / (1024 * 1024)
        assert s.memory_usage(unit="KB", index=False) == size * ak.dtypes.int64.itemsize / 1024
        assert s.memory_usage(unit="B", index=False) == size * ak.dtypes.int64.itemsize

        assert s.memory_usage(unit="GB", index=True) == 2 * size * ak.dtypes.int64.itemsize / (
            1024 * 1024 * 1024
        )
        assert s.memory_usage(unit="MB", index=True) == 2 * size * ak.dtypes.int64.itemsize / (
            1024 * 1024
        )
        assert s.memory_usage(unit="KB", index=True) == 2 * size * ak.dtypes.int64.itemsize / 1024
        assert s.memory_usage(unit="B", index=True) == 2 * size * ak.dtypes.int64.itemsize

    def test_map(self):
        a = ak.Series(ak.array(["1", "1", "4", "4", "4"]))
        b = ak.Series(ak.array([2, 3, 2, 3, 4]))
        c = ak.Series(ak.array([1.0, 1.0, 2.2, 2.2, 4.4]), index=ak.array([5, 4, 2, 3, 1]))

        result = a.map({"4": 25, "5": 30, "1": 7})
        assert result.index.values.to_list() == [0, 1, 2, 3, 4]
        assert result.values.to_list() == [7, 7, 25, 25, 25]

        result = a.map({"1": 7})
        assert result.index.values.to_list() == [0, 1, 2, 3, 4]
        assert (
            result.values.to_list()
            == ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list()
        )

        result = a.map({"1": 7.0})
        assert result.index.values.to_list() == [0, 1, 2, 3, 4]
        assert np.allclose(result.values.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

        result = b.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        assert result.index.values.to_list() == [0, 1, 2, 3, 4]
        assert result.values.to_list() == [30.0, 5.0, 30.0, 5.0, 25.0]

        result = c.map({1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        assert result.index.values.to_list() == [5, 4, 2, 3, 1]
        assert result.values.to_list() == ["a", "a", "b", "b", "c"]

        result = c.map({1.0: "a"})
        assert result.index.values.to_list() == [5, 4, 2, 3, 1]
        assert result.values.to_list() == ["a", "a", "null", "null", "null"]

        result = c.map({1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        assert result.index.values.to_list() == [5, 4, 2, 3, 1]
        assert result.values.to_list() == ["a", "a", "b", "b", "c"]

    def test_to_markdown(self):
        s = ak.Series(["elk", "pig", "dog", "quetzal"], name="animal")
        assert (
            s.to_markdown() == "+----+----------+\n"
            "|    | animal   |\n"
            "+====+==========+\n"
            "|  0 | elk      |\n"
            "+----+----------+\n"
            "|  1 | pig      |\n"
            "+----+----------+\n"
            "|  2 | dog      |\n"
            "+----+----------+\n"
            "|  3 | quetzal  |\n"
            "+----+----------+"
        )

        assert (
            s.to_markdown(index=False) == "+----------+\n"
            "| animal   |\n"
            "+==========+\n"
            "| elk      |\n"
            "+----------+\n"
            "| pig      |\n"
            "+----------+\n"
            "| dog      |\n"
            "+----------+\n"
            "| quetzal  |\n"
            "+----------+"
        )

        assert (
            s.to_markdown(tablefmt="grid") == "+----+----------+\n"
            "|    | animal   |\n"
            "+====+==========+\n"
            "|  0 | elk      |\n"
            "+----+----------+\n"
            "|  1 | pig      |\n"
            "+----+----------+\n"
            "|  2 | dog      |\n"
            "+----+----------+\n"
            "|  3 | quetzal  |\n"
            "+----+----------+"
        )

        assert s.to_markdown(tablefmt="grid") == s.to_pandas().to_markdown(tablefmt="grid")
        assert s.to_markdown(tablefmt="grid", index=False) == s.to_pandas().to_markdown(
            tablefmt="grid", index=False
        )
        assert s.to_markdown(tablefmt="jira") == s.to_pandas().to_markdown(tablefmt="jira")

    def test_isna_int(self):
        # Test case with integer data type
        data_int = Series([1, 2, 3, 4, 5])
        expected_int = Series([False, False, False, False, False])
        assert np.allclose(data_int.isna().values.to_ndarray(), expected_int.values.to_ndarray())
        assert np.allclose(data_int.isnull().values.to_ndarray(), expected_int.values.to_ndarray())
        assert np.allclose(data_int.notna().values.to_ndarray(), ~expected_int.values.to_ndarray())
        assert np.allclose(data_int.notnull().values.to_ndarray(), ~expected_int.values.to_ndarray())
        assert ~data_int.hasnans()

    def test_isna_float(self):
        # Test case with float data type
        data_float = Series([1.0, 2.0, 3.0, np.nan, 5.0])
        expected_float = Series([False, False, False, True, False])
        assert np.allclose(data_float.isna().values.to_ndarray(), expected_float.values.to_ndarray())
        assert np.allclose(data_float.isnull().values.to_ndarray(), expected_float.values.to_ndarray())
        assert np.allclose(data_float.notna().values.to_ndarray(), ~expected_float.values.to_ndarray())
        assert np.allclose(data_float.notnull().values.to_ndarray(), ~expected_float.values.to_ndarray())
        assert data_float.hasnans()

    def test_isna_string(self):
        # Test case with string data type
        data_string = Series(["a", "b", "c", "d", "e"])
        expected_string = Series([False, False, False, False, False])
        assert np.allclose(data_string.isna().values.to_ndarray(), expected_string.values.to_ndarray())
        assert np.allclose(data_string.isnull().values.to_ndarray(), expected_string.values.to_ndarray())
        assert np.allclose(data_string.notna().values.to_ndarray(), ~expected_string.values.to_ndarray())
        assert np.allclose(
            data_string.notnull().values.to_ndarray(), ~expected_string.values.to_ndarray()
        )
        assert ~data_string.hasnans()

    def test_fillna(self):
        data = ak.Series([1, np.nan, 3, np.nan, 5])

        fill_values1 = ak.ones(5)
        assert data.fillna(fill_values1).to_list() == [1.0, 1.0, 3.0, 1.0, 5.0]

        fill_values2 = Series(2 * ak.ones(5))
        assert data.fillna(fill_values2).to_list() == [1.0, 2.0, 3.0, 2.0, 5.0]

        fill_values3 = 100.0
        assert data.fillna(fill_values3).to_list() == [1.0, 100.0, 3.0, 100.0, 5.0]
