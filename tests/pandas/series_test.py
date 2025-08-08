import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from pandas.testing import assert_series_equal as pd_assert_series_equal
import pytest

import arkouda as ak
from arkouda.pandas.series import Series
from arkouda.testing import assert_series_equal as ak_assert_series_equal

DTYPES = [ak.int64, ak.uint64, ak.bool_, ak.float64, ak.bigint, ak.str_]
NO_STRING = [ak.int64, ak.uint64, ak.bool_, ak.float64, ak.bigint]
NUMERICAL_TYPES = [ak.int64, ak.uint64, ak.float64, ak.bigint]
INTEGRAL_TYPES = [ak.int64, ak.uint64, ak.bool_, ak.bigint]


class TestSeries:
    def test_series_docstrings(self):
        import doctest

        from arkouda import series

        result = doctest.testmod(series, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_series_creation(self, dtype):
        idx = ak.arange(3, dtype=dtype)
        for val in idx, ak.array(["A", "B", "C"]):
            ans = ak.Series(data=val, index=idx).tolist()
            for series in (
                ak.Series(data=val, index=idx),
                ak.Series(data=val),
                ak.Series(val, idx),
                ak.Series(val),
                ak.Series((idx, val)),
            ):
                assert isinstance(series, ak.Series)
                assert isinstance(series.index, ak.Index)
                assert series.tolist() == ans

        with pytest.raises(TypeError):
            ak.Series(index=idx)

        with pytest.raises(TypeError):
            ak.Series((ak.arange(3),))

        with pytest.raises(TypeError):
            ak.Series()

        with pytest.raises(ValueError):
            ak.Series(data=ak.arange(3), index=ak.arange(6))

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_series_creation_pandas_series(self, size):
        str_vals = ak.random_strings_uniform(9, 10, size)
        idx = ak.arange(size) * -1

        vals = [str_vals, ak.Categorical(str_vals), ak.arange(size) * -2]
        for val in vals:
            if isinstance(val, ak.Categorical):
                pd_ser = pd.Series(val.to_pandas(), idx.to_ndarray())
            else:
                pd_ser = pd.Series(val.to_ndarray(), idx.to_ndarray())
            ak_ser = Series(pd_ser)
            expected = Series(val, index=idx)
            ak_assert_series_equal(ak_ser, expected)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_pandas(self, size):
        str_vals = ak.random_strings_uniform(9, 10, size)
        idx = ak.arange(size)

        vals = [str_vals, ak.Categorical(str_vals), ak.arange(size) * -2]
        for val in vals:
            ak_ser = Series(val, idx)
            if isinstance(val, ak.Categorical):
                pd_ser = pd.Series(val.to_pandas(), idx.to_ndarray())
            else:
                pd_ser = pd.Series(val.to_ndarray(), idx.to_ndarray())
            pd_assert_series_equal(ak_ser.to_pandas(), pd_ser)

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

    def test_shape(self):
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        (l,) = s.shape
        assert l == 3

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
        if dtype != ak.bool_:
            assert all(i in added.values.tolist() for i in range(size))
        else:
            # we have exactly one False
            assert added.values.sum() == 99

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    def test_topn(self, dtype):
        top = ak.Series(ak.arange(100, dtype=dtype)).topn(50)
        assert top.values.tolist() == list(range(99, 49, -1))
        assert top.index.tolist() == list(range(99, 49, -1))

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
        assert idx_sort.index.tolist() == ordered.tolist()
        assert idx_sort.values.tolist() == perm.tolist()

        val_sort = ak.Series(data=perm, index=ordered).sort_values()
        assert val_sort.index.to_pandas().tolist() == perm.tolist()
        assert val_sort.values.tolist() == ordered.tolist()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_head_tail(self, dtype):
        n = 10
        s = ak.Series(ak.arange(n, dtype=dtype))
        for i in range(n):
            head = s.head(i)
            assert head.index.tolist() == list(range(i))
            assert head.values.tolist() == ak.arange(i, dtype=dtype).tolist()

            tail = s.tail(i)
            assert tail.index.tolist() == ak.arange(n)[-i:n].tolist()
            assert tail.values.tolist() == ak.arange(n, dtype=dtype)[-i:n].tolist()

    def test_value_counts(self):
        s = ak.Series(ak.array([1, 2, 0, 2, 0]))

        c = s.value_counts()
        assert c.index.tolist() == [0, 2, 1]
        assert c.values.tolist() == [2, 2, 1]

        c = s.value_counts(sort=False)
        assert c.index.tolist() == list(range(3))
        assert c.values.tolist() == [2, 1, 2]

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
        pd_assert_frame_equal(ref_df, df.to_pandas())

        def list_helper(arr):
            return arr.tolist() if isinstance(arr, (ak.pdarray, ak.Index)) else arr.tolist()

        for fname in "concat", "pdconcat":
            func = getattr(ak.Series, fname)
            c = func([s, s2])
            assert list_helper(c.index) == list(range(11))
            assert list_helper(c.values) == list(range(11))

            df = func([s, s3], axis=1)
            if fname == "concat":
                ref_df = pd.DataFrame(
                    {
                        "idx": [0, 1, 2, 3, 4],
                        "val_0": [0, 1, 2, 3, 4],
                        "val_1": [5, 6, 7, 8, 9],
                    }
                )
                assert isinstance(df, ak.DataFrame)
                pd_assert_frame_equal(ref_df, df.to_pandas())
            else:
                ref_df = pd.DataFrame({0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})
                assert isinstance(df, pd.DataFrame)
                pd_assert_frame_equal(ref_df, df)

    def test_index_as_index_compat(self):
        # added to validate functionality for issue #1506
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        g = df.groupby(["a", "b"])
        series = ak.Series(data=g.sum("c")["c"], index=g.sum("c").index)
        g.broadcast(series)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_memory_usage(self, size):
        s = ak.Series(ak.arange(size))
        int64_size = ak.dtype(ak.int64).itemsize

        assert s.memory_usage(unit="GB", index=False) == size * int64_size / (1024 * 1024 * 1024)
        assert s.memory_usage(unit="MB", index=False) == size * int64_size / (1024 * 1024)
        assert s.memory_usage(unit="KB", index=False) == size * int64_size / 1024
        assert s.memory_usage(unit="B", index=False) == size * int64_size

        assert s.memory_usage(unit="GB", index=True) == 2 * size * int64_size / (1024 * 1024 * 1024)
        assert s.memory_usage(unit="MB", index=True) == 2 * size * int64_size / (1024 * 1024)
        assert s.memory_usage(unit="KB", index=True) == 2 * size * int64_size / 1024
        assert s.memory_usage(unit="B", index=True) == 2 * size * int64_size

    def test_map(self):
        a = ak.Series(ak.array(["1", "1", "4", "4", "4"]))
        b = ak.Series(ak.array([2, 3, 2, 3, 4]))
        c = ak.Series(ak.array([1.0, 1.0, 2.2, 2.2, 4.4]), index=ak.array([5, 4, 2, 3, 1]))

        result = a.map({"4": 25, "5": 30, "1": 7})
        assert result.index.values.tolist() == [0, 1, 2, 3, 4]
        assert result.values.tolist() == [7, 7, 25, 25, 25]

        result = a.map({"1": 7})
        assert result.index.values.tolist() == [0, 1, 2, 3, 4]
        assert (
            result.values.tolist()
            == ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).tolist()
        )

        result = a.map({"1": 7.0})
        assert result.index.values.tolist() == [0, 1, 2, 3, 4]
        assert np.allclose(result.values.tolist(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

        result = b.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        assert result.index.values.tolist() == [0, 1, 2, 3, 4]
        assert result.values.tolist() == [30.0, 5.0, 30.0, 5.0, 25.0]

        result = c.map({1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        assert result.index.values.tolist() == [5, 4, 2, 3, 1]
        assert result.values.tolist() == ["a", "a", "b", "b", "c"]

        result = c.map({1.0: "a"})
        assert result.index.values.tolist() == [5, 4, 2, 3, 1]
        assert result.values.tolist() == ["a", "a", "null", "null", "null"]

        result = c.map({1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        assert result.index.values.tolist() == [5, 4, 2, 3, 1]
        assert result.values.tolist() == ["a", "a", "b", "b", "c"]

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

    def test_has_repeat_labels(self):
        ints = ak.array([0, 1, 3, 7, 3])
        floats = ak.array([0.0, 1.5, 0.5, 1.5, -1.0])
        strings = ak.array(["A", "C", "C", "DE", "Z"])
        for idxs in [ints, floats, strings]:
            s1 = ak.Series(index=idxs, data=floats)
            assert s1.has_repeat_labels()

        s2 = ak.Series(index=ak.array([0, 1, 3, 4, 5]), data=floats)
        assert not s2.has_repeat_labels()

    def test_isna_int(self):
        # Test case with integer data type
        data_int = Series([1, 2, 3, 4, 5])
        expected_int = Series([False, False, False, False, False])
        assert np.allclose(data_int.isna().values.to_ndarray(), expected_int.values.to_ndarray())
        assert np.allclose(data_int.isnull().values.to_ndarray(), expected_int.values.to_ndarray())
        assert np.allclose(data_int.notna().values.to_ndarray(), ~expected_int.values.to_ndarray())
        assert np.allclose(data_int.notnull().values.to_ndarray(), ~expected_int.values.to_ndarray())
        assert not data_int.hasnans()

    def test_isna_float(self):
        # Test case with float data type
        data_float = Series([1.0, 2.0, 3.0, np.nan, 5.0])
        expected_float = Series([False, False, False, True, False])
        assert np.allclose(data_float.isna().values.to_ndarray(), expected_float.values.to_ndarray())
        assert np.allclose(data_float.isnull().values.to_ndarray(), expected_float.values.to_ndarray())
        assert np.allclose(data_float.notna().values.to_ndarray(), ~expected_float.values.to_ndarray())
        assert np.allclose(
            data_float.notnull().values.to_ndarray(),
            ~expected_float.values.to_ndarray(),
        )
        assert data_float.hasnans()

    def test_isna_string(self):
        # Test case with string data type
        data_string = Series(["a", "b", "c", "d", "e"])
        expected_string = Series([False, False, False, False, False])
        assert np.allclose(data_string.isna().values.to_ndarray(), expected_string.values.to_ndarray())
        assert np.allclose(
            data_string.isnull().values.to_ndarray(),
            expected_string.values.to_ndarray(),
        )
        assert np.allclose(
            data_string.notna().values.to_ndarray(),
            ~expected_string.values.to_ndarray(),
        )
        assert np.allclose(
            data_string.notnull().values.to_ndarray(),
            ~expected_string.values.to_ndarray(),
        )
        assert not data_string.hasnans()

    def test_fillna(self):
        data = ak.Series([1, np.nan, 3, np.nan, 5])

        fill_values1 = ak.ones(5)
        assert data.fillna(fill_values1).tolist() == [1.0, 1.0, 3.0, 1.0, 5.0]

        fill_values2 = Series(2 * ak.ones(5))
        assert data.fillna(fill_values2).tolist() == [1.0, 2.0, 3.0, 2.0, 5.0]

        fill_values3 = 100.0
        assert data.fillna(fill_values3).tolist() == [1.0, 100.0, 3.0, 100.0, 5.0]

    def test_series_segarray_to_pandas(self):
        # reproducer for issue #3222
        sa = ak.SegArray(ak.arange(0, 30, 3), ak.arange(30))
        akdf = ak.DataFrame({"test": sa})
        pddf = pd.DataFrame({"test": sa.tolist()})

        pd_assert_frame_equal(akdf.to_pandas(), pddf)
        pd_assert_series_equal(akdf.to_pandas()["test"], pddf["test"], check_names=False)

    def test_getitem_scalars(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        with pytest.raises(TypeError):
            s1[1.0]
        with pytest.raises(TypeError):
            s1[1]
        _s1_a1 = _s1["A"]
        s1_a1 = s1["A"]

        assert isinstance(_s1_a1, float)
        assert isinstance(s1_a1, float)
        assert s1_a1 == _s1_a1
        _s1_a2 = _s1["C"]
        s1_a2 = s1["C"]
        assert isinstance(_s1_a2, pd.Series)
        assert isinstance(s1_a2, ak.Series)
        assert s1_a2.index.tolist() == _s1_a2.index.tolist()
        assert s1_a2.values.tolist() == _s1_a2.values.tolist()

        _s2 = pd.Series(index=np.array(ints), data=np.array(strings))
        s2 = ak.Series(index=ak.array(ints), data=ak.array(strings))
        with pytest.raises(TypeError):
            s2[1.0]
        with pytest.raises(TypeError):
            s2["A"]
        _s2_a1 = _s2[7]
        s2_a1 = s2[7]
        assert isinstance(_s2_a1, str)
        assert isinstance(s2_a1, str)
        assert _s2_a1 == s2_a1

        _s2_a2 = _s2[3]
        s2_a2 = s2[3]
        assert isinstance(_s2_a2, pd.Series)
        assert isinstance(s2_a2, ak.Series)
        assert s2_a2.index.tolist() == _s2_a2.index.tolist()
        assert s2_a2.values.tolist() == _s2_a2.values.tolist()

        _s3 = pd.Series(index=np.array(floats), data=np.array(ints))
        s3 = ak.Series(index=ak.array(floats), data=ak.array(ints))
        with pytest.raises(TypeError):
            s3[1]
        with pytest.raises(TypeError):
            s3["A"]
        _s3_a1 = _s3[0.0]
        s3_a1 = s3[0.0]
        assert isinstance(_s3_a1, np.int64)
        assert isinstance(s3_a1, np.int64)

        _s3_a2 = _s3[1.5]
        s3_a2 = s3[1.5]
        assert isinstance(_s3_a2, pd.Series)
        assert isinstance(s3_a2, ak.Series)
        assert s3_a2.index.tolist() == _s3_a2.index.tolist()
        assert s3_a2.values.tolist() == _s3_a2.values.tolist()

    def test_getitem_vectors(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))

        # Arkouda requires homogeneous index type
        with pytest.raises(TypeError):
            s1[[1.0, 2.0]]
        with pytest.raises(TypeError):
            s1[[1, 2]]
        with pytest.raises(TypeError):
            s1[ak.array([1.0, 2.0])]
        with pytest.raises(TypeError):
            s1[ak.array([1, 2])]

        _s1_a1 = _s1[np.array(["A", "Z"])]
        s1_a1 = s1[ak.array(["A", "Z"])]
        assert isinstance(s1_a1, ak.Series)
        assert s1_a1.index.tolist() == _s1_a1.index.tolist()
        assert s1_a1.values.tolist() == _s1_a1.values.tolist()

        _s1_a2 = _s1[["C", "DE"]]
        s1_a2 = s1[["C", "DE"]]
        assert isinstance(s1_a2, ak.Series)
        assert s1_a2.index.tolist() == _s1_a2.index.tolist()
        assert s1_a2.values.tolist() == _s1_a2.values.tolist()

        _s1_a3 = _s1[[True, False, True, False, False]]
        s1_a3 = s1[[True, False, True, False, False]]
        assert isinstance(s1_a3, ak.Series)
        assert s1_a3.index.tolist() == _s1_a3.index.tolist()
        assert s1_a3.values.tolist() == _s1_a3.values.tolist()

        with pytest.raises(IndexError):
            _s1[[True, False, True]]
        with pytest.raises(IndexError):
            s1[[True, False, True]]
        with pytest.raises(IndexError):
            _s1[np.array([True, False, True])]
        with pytest.raises(IndexError):
            s1[ak.array([True, False, True])]

        _s2 = pd.Series(index=np.array(floats), data=np.array(ints))
        s2 = ak.Series(index=ak.array(floats), data=ak.array(ints))
        with pytest.raises(TypeError):
            s2[["A"]]
        with pytest.raises(TypeError):
            s2[[1, 2]]
        with pytest.raises(TypeError):
            s2[ak.array(["A", "B"])]
        with pytest.raises(TypeError):
            s2[ak.array([1, 2])]

        _s2_a1 = _s2[[0.5, 0.0]]
        s2_a1 = s2[[0.5, 0.0]]
        assert isinstance(s1_a2, ak.Series)
        assert s2_a1.index.tolist() == _s2_a1.index.tolist()
        assert s2_a1.values.tolist() == _s2_a1.values.tolist()

        _s2_a2 = _s2[np.array([0.5, 0.0])]
        s2_a2 = s2[ak.array([0.5, 0.0])]
        assert isinstance(s1_a2, ak.Series)
        assert s2_a2.index.tolist() == _s2_a2.index.tolist()
        assert s2_a2.values.tolist() == _s2_a2.values.tolist()

        with pytest.raises(KeyError):
            _s2_a3 = _s2[[1.5, 1.2]]
        with pytest.raises(KeyError):
            s2_a3 = s2[[1.5, 1.2]]

        _s2_a3 = _s2[[1.5, 0.0]]
        s2_a3 = s2[[1.5, 0.0]]
        assert isinstance(s2_a2, ak.Series)
        assert s2_a3.index.tolist() == _s2_a3.index.tolist()
        assert s2_a3.values.tolist() == _s2_a3.values.tolist()

    def test_setitem_scalars(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))

        with pytest.raises(TypeError):
            s1[1.0] = 1.0
        with pytest.raises(TypeError):
            s1[1] = 1.0
        with pytest.raises(TypeError):
            s1["A"] = 1
        with pytest.raises(TypeError):
            s1["A"] = "C"

        s1["A"] = 0.2
        _s1["A"] = 0.2
        assert s1.values.tolist() == _s1.values.tolist()
        s1["C"] = 1.2
        _s1["C"] = 1.2
        assert s1.values.tolist() == _s1.values.tolist()
        s1["X"] = 0.0
        _s1["X"] = 0.0
        assert s1.index.tolist() == _s1.index.tolist()
        assert s1.values.tolist() == _s1.values.tolist()
        s1["C"] = [0.3, 0.4]
        _s1["C"] = [0.3, 0.4]
        assert s1.values.tolist() == _s1.values.tolist()

        with pytest.raises(ValueError):
            s1["C"] = [0.4, 0.3, 0.2]

        # cannot assign to Strings
        s2 = ak.Series(index=ak.array(ints), data=ak.array(strings))
        with pytest.raises(TypeError):
            s2[1.0] = "D"
        with pytest.raises(TypeError):
            s2["C"] = "E"
        with pytest.raises(TypeError):
            s2[0] = 1.0
        with pytest.raises(TypeError):
            s2[0] = 1
        with pytest.raises(TypeError):
            s2[7] = "L"
        with pytest.raises(TypeError):
            s2[3] = ["X1", "X2"]

        s3 = ak.Series(index=ak.array(floats), data=ak.array(ints))
        _s3 = pd.Series(index=np.array(floats), data=np.array(ints))
        assert s3.values.tolist() == [0, 1, 3, 7, 3]
        assert s3.index.tolist() == [0.0, 1.5, 0.5, 1.5, -1.0]
        assert s3.values.tolist() == _s3.values.tolist()
        assert s3.index.tolist() == _s3.index.tolist()
        s3[0.0] = 2
        _s3[0.0] = 2
        assert s3.values.tolist() == _s3.values.tolist()
        _s3[1.5] = 8
        s3[1.5] = 8
        assert s3.values.tolist() == _s3.values.tolist()
        _s3[2.0] = 9
        s3[2.0] = 9
        assert s3.index.tolist() == _s3.index.tolist()
        assert s3.values.tolist() == _s3.values.tolist()
        _s3[1.5] = [4, 5]
        s3[1.5] = [4, 5]
        assert s3.values.tolist() == _s3.values.tolist()
        _s3[1.5] = np.array([6, 7])
        s3[1.5] = ak.array([6, 7])
        assert s3.values.tolist() == _s3.values.tolist()
        _s3[1.5] = [8]
        s3[1.5] = [8]
        assert s3.values.tolist() == _s3.values.tolist()
        _s3[1.5] = np.array([2])
        s3[1.5] = ak.array([2])
        assert s3.values.tolist() == _s3.values.tolist()
        with pytest.raises(ValueError):
            s3[1.5] = [9, 10, 11]
        with pytest.raises(ValueError):
            s3[1.5] = ak.array([0, 1, 2])

        # adding new entries
        _s3[-1.0] = 14
        s3[-1.0] = 14
        assert s3.values.tolist() == _s3.values.tolist()
        assert s3.index.tolist() == _s3.index.tolist()

        # pandas makes the entry a list, which is not what we want.
        with pytest.raises(ValueError):
            s3[-11.0] = [13, 14, 15]

    def test_setitem_vectors(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))

        # mismatching types for values and indices
        with pytest.raises(TypeError):
            s1[[0.1, 0.2]] = 1.0
        with pytest.raises(TypeError):
            s1[[0, 3]] = 1.0
        with pytest.raises(TypeError):
            s1[ak.array([0, 3])] = 1.0
        with pytest.raises(TypeError):
            s1[["A", "B"]] = 1
        with pytest.raises(TypeError):
            s1[["A", "B"]] = "C"
        with pytest.raises(TypeError):
            s1[ak.array(["A", "B"])] = 1

        # indexing using list of labels only valid with uniquely labeled Series
        with pytest.raises(pd.errors.InvalidIndexError):
            _s1[["A", "Z"]] = 2.0
        assert s1.has_repeat_labels()
        with pytest.raises(ValueError):
            s1[["A", "Z"]] = 2.0

        s2 = ak.Series(index=ak.array(["A", "C", "DE", "F", "Z"]), data=ak.array(ints))
        _s2 = pd.Series(index=pd.array(["A", "C", "DE", "F", "Z"]), data=pd.array(ints))
        s2[["A", "Z"]] = 2
        _s2[["A", "Z"]] = 2
        assert s2.values.tolist() == _s2.values.tolist()
        s2[ak.array(["A", "Z"])] = 3
        _s2[np.array(["A", "Z"])] = 3
        assert s2.values.tolist() == _s2.values.tolist()
        with pytest.raises(ValueError):
            _s2[np.array(["A", "Z"])] = [3]
        with pytest.raises(ValueError):
            s2[ak.array(["A", "Z"])] = [3]

        with pytest.raises(KeyError):
            _s2[np.array(["B", "D"])] = 0
        with pytest.raises(KeyError):
            s2[ak.array(["B", "D"])] = 0
        with pytest.raises(KeyError):
            _s2[["B", "D"]] = 0
        with pytest.raises(KeyError):
            s2[["B", "D"]] = 0
        with pytest.raises(KeyError):
            _s2[["B"]] = 0
        with pytest.raises(KeyError):
            s2[["B"]] = 0
        assert s2.values.tolist() == _s2.values.tolist()
        assert s2.index.tolist() == _s2.index.tolist()

        _s2[np.array(["A", "C", "F"])] = [10, 11, 12]
        s2[ak.array(["A", "C", "F"])] = [10, 11, 12]
        assert s2.values.tolist() == _s2.values.tolist()

    def test_iloc(self):
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))

        with pytest.raises(TypeError):
            s1.iloc["A"]
        with pytest.raises(TypeError):
            s1.iloc["A"] = 1.0
        with pytest.raises(TypeError):
            s1.iloc[0] = 1

        s1_a1 = s1.iloc[0]
        assert isinstance(s1_a1, ak.Series)
        assert s1_a1.index.tolist() == ["A"]
        assert s1_a1.values.tolist() == [0.0]
        _s1.iloc[0] = 1.0
        s1.iloc[0] = 1.0
        assert s1.values.tolist() == _s1.values.tolist()

        with pytest.raises(pd.errors.IndexingError):
            _s1_a2 = _s1.iloc[1, 3]
        with pytest.raises(TypeError):
            s1_a2 = s1.iloc[1, 3]
        with pytest.raises(IndexError):
            _s1.iloc[1, 3] = 2.0
        with pytest.raises(TypeError):
            s1.iloc[1, 3] = 2.0

        _s1_a2 = _s1.iloc[[1, 2]]
        s1_a2 = s1.iloc[[1, 2]]
        assert s1_a2.index.tolist() == _s1_a2.index.tolist()
        assert s1_a2.values.tolist() == _s1_a2.values.tolist()
        _s1.iloc[[1, 2]] = 0.2
        s1.iloc[[1, 2]] = 0.2
        assert s1.values.tolist() == _s1.values.tolist()

        with pytest.raises(ValueError):
            _s1.iloc[[3, 4]] = [0.3]
        with pytest.raises(ValueError):
            s1.iloc[[3, 4]] = [0.3]

        _s1.iloc[[3, 4]] = [0.4, 0.5]
        s1.iloc[[3, 4]] = [0.4, 0.5]
        assert s1.values.tolist() == _s1.values.tolist()

        with pytest.raises(TypeError):
            # in pandas this hits a NotImplementedError
            s1.iloc[3, 4] = [0.4, 0.5]

        with pytest.raises(ValueError):
            s1.iloc[[3, 4]] = ak.array([0.3])
        with pytest.raises(ValueError):
            s1.iloc[[3, 4]] = [0.1, 0.2, 0.3]

        # iloc does not enlarge its target object
        with pytest.raises(IndexError):
            _s1.iloc[5]
        with pytest.raises(IndexError):
            s1.iloc[5]
        with pytest.raises(IndexError):
            s1.iloc[5] = 2
        with pytest.raises(IndexError):
            s1.iloc[[3, 5]]
        with pytest.raises(IndexError):
            s1.iloc[[3, 5]] = [0.1, 0.2]

        # can also take boolean array
        _b = _s1.iloc[[True, False, True, True, False]]
        b = s1.iloc[[True, False, True, True, False]]
        assert b.values.tolist() == _b.values.tolist()

        _s1.iloc[[True, False, False, True, False]] = [0.5, 0.6]
        s1.iloc[[True, False, False, True, False]] = [0.5, 0.6]
        assert b.values.tolist() == _b.values.tolist()

        with pytest.raises(IndexError):
            _s1.iloc[[True, False, True]]
        with pytest.raises(IndexError):
            s1.iloc[[True, False, True]]
