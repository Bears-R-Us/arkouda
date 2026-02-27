import itertools
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

import arkouda as ak

from arkouda.pandas import io_util
from arkouda.scipy import chisquare as akchisquare
from arkouda.testing import assert_frame_equal as ak_assert_frame_equal
from arkouda.testing import assert_frame_equivalent


def alternating_1_0(n):
    a = np.full(n, 0)
    a[::2] = 1
    return ak.array(a)


@pytest.fixture
def df_test_base_tmp(request):
    df_test_base_tmp = "{}/.df_test".format(os.getcwd())
    io_util.get_directory(df_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(df_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return df_test_base_tmp


@pytest.mark.requires_chapel_module("DataFrameIndexingMsg")
class TestDataFrame:
    def test_dataframe_docstrings(self, df_test_base_tmp):
        import doctest

        from arkouda.pandas import dataframe

        with tempfile.TemporaryDirectory(dir=df_test_base_tmp) as tmp_dirname:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp_dirname)  # Change to temp directory
                result = doctest.testmod(
                    dataframe, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                )
            finally:
                os.chdir(old_cwd)  # Always return to original directory

            assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @staticmethod
    def build_pd_df():
        username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]
        userid = [111, 222, 111, 333, 222, 111]
        item = [0, 0, 1, 1, 2, 0]
        day = [5, 5, 6, 5, 6, 6]
        amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6]
        bi = [2**200, 2**200 + 1, 2**200 + 2, 2**200 + 3, 2**200 + 4, 2**200 + 5]
        ui = (np.arange(6).astype(ak.uint64)) + 2**63
        return pd.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
                "ui": ui,
            }
        )

    @staticmethod
    def build_ak_df():
        return ak.DataFrame(TestDataFrame.build_pd_df())

    @staticmethod
    def build_ak_df_example2():
        data = {
            "key1": ["valuew", "valuex", "valuew", "valuex"],
            "key2": ["valueA", "valueB", "valueA", "valueB"],
            "key3": ["value1", "value2", "value3", "value4"],
            "count": [34, 25, 11, 4],
            "nums": [1, 2, 5, 21],
        }
        ak_df = ak.DataFrame({k: ak.array(v) for k, v in data.items()})
        return ak_df

    @staticmethod
    def build_ak_df_with_nans():
        data = {
            "key1": ["valuew", "valuex", "valuew", "valuex"],
            "key2": ["valueA", "valueB", "valueA", "valueB"],
            "nums1": [1, np.nan, 3, 4],
            "nums2": [1, np.nan, np.nan, 7],
            "nums3": [10, 8, 9, 7],
            "bools": [True, False, True, False],
        }
        ak_df = ak.DataFrame({k: ak.array(v) for k, v in data.items()})
        return ak_df

    @staticmethod
    def build_ak_df_example_numeric_types():
        ak_df = ak.DataFrame(
            {
                "gb_id": ak.randint(0, 5, 20, dtype=ak.int64),
                "float64": ak.randint(0, 1, 20, dtype=ak.float64),
                "int64": ak.randint(0, 10, 20, dtype=ak.int64),
                "uint64": ak.randint(0, 10, 20, dtype=ak.uint64),
                "bigint": ak.randint(2**200, 2**200 + 10, 20, dtype=ak.uint64),
            }
        )
        return ak_df

    @staticmethod
    def build_pd_df_duplicates():
        username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]
        userid = [111, 222, 111, 333, 222, 111]
        item = [0, 1, 0, 2, 1, 0]
        day = [5, 5, 5, 5, 5, 5]
        return pd.DataFrame({"userName": username, "userID": userid, "item": item, "day": day})

    @staticmethod
    def build_ak_df_duplicates():
        return ak.DataFrame(TestDataFrame.build_pd_df_duplicates())

    @staticmethod
    def build_ak_append():
        username = ak.array(["John", "Carol"])
        userid = ak.array([444, 333])
        item = ak.array([0, 2])
        day = ak.array([1, 2])
        amount = ak.array([0.5, 5.1])
        bi = ak.array([2**200 + 6, 2**200 + 7])
        ui = ak.array([6, 7], dtype=ak.uint64) + 2**63
        return ak.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
                "ui": ui,
            }
        )

    @staticmethod
    def build_pd_df_append():
        username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice", "John", "Carol"]
        userid = [111, 222, 111, 333, 222, 111, 444, 333]
        item = [0, 0, 1, 1, 2, 0, 0, 2]
        day = [5, 5, 6, 5, 6, 6, 1, 2]
        amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6, 0.5, 5.1]
        bi = np.arange(2**200, 2**200 + 8).tolist()  # (np.arange(8) + 2**200).tolist()
        ui = (np.arange(8).astype(ak.uint64)) + 2**63
        return pd.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
                "ui": ui,
            }
        )

    @staticmethod
    def build_ak_keyerror():
        userid = ak.array([444, 333])
        item = ak.array([0, 2])
        return ak.DataFrame({"user_id": userid, "item": item})

    @staticmethod
    def build_ak_typeerror():
        username = ak.array([111, 222, 111, 333, 222, 111])
        userid = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        bi = ak.arange(2**200, 2**200 + 6)
        ui = ak.arange(6, dtype=ak.uint64) + 2**63
        return ak.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
                "ui": ui,
            }
        )

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_dataframe_creation(self, size):
        # Validate empty DataFrame
        df = ak.DataFrame()
        assert isinstance(df, ak.DataFrame)
        assert df.empty

        # Validation of Creation from Pandas
        pddf = pd.DataFrame(
            {
                "int": np.arange(size),
                "uint": np.random.randint(0, size / 2, size, dtype=np.uint64),
                "bigint": np.arange(2**200, 2**200 + size),
                "bool": np.random.randint(0, 1, size=size, dtype=bool),
                "segarray": [np.random.randint(0, size / 2, 2) for i in range(size)],
            }
        )
        akdf = ak.DataFrame(pddf)
        assert isinstance(akdf, ak.DataFrame)
        assert len(akdf) == size
        assert_frame_equal(pddf, akdf.to_pandas())

        # validation of creation from dictionary
        akdf = ak.DataFrame(
            {
                "int": ak.arange(size),
                "uint": ak.array(pddf["uint"]),
                "bigint": ak.arange(2**200, 2**200 + size),
                "bool": ak.array(pddf["bool"]),
                "segarray": ak.SegArray.from_multi_array([ak.array(x) for x in pddf["segarray"]]),
            }
        )
        assert isinstance(akdf, ak.DataFrame)
        assert len(akdf) == size

        assert_frame_equal(pddf, akdf.to_pandas())

        # validation of creation from list

        # The line ****'d  below fails in the arkouda INDEX class if size > 1000,
        #  so it is hardcoded to that limit.

        size1000 = 1000
        x = [
            np.arange(size1000),
            np.random.randint(0, 5, size1000),
            np.random.randint(5, 10, size1000),
        ]
        pddf = pd.DataFrame(x)
        pddf.columns = pddf.columns.astype(str)
        akdf = ak.DataFrame([ak.array(val) for val in list(zip(*x))])
        assert isinstance(akdf, ak.DataFrame)
        assert len(akdf) == len(pddf)
        # arkouda does not allow for numeric columns.
        assert akdf.columns.values == [str(x) for x in pddf.columns.values]
        # use the columns from the pandas created for equivalence check
        # these should be equivalent
        ak_to_pd = akdf.to_pandas()
        assert_frame_equal(pddf, ak_to_pd)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", ["float64", "int64"])
    def test_from_pandas_with_index(self, size, dtype):
        np_arry = np.arange(size, dtype=dtype)
        np_arry = np_arry[::2] * -1.0  # Alter so that index is different from default
        idx = pd.Index(np_arry)
        pd_df = pd.DataFrame({"col1": np_arry}, index=idx)
        ak_df = ak.DataFrame(pd_df)
        assert pd_df.index.inferred_type == ak_df.index.inferred_type

        ak_arry = ak.arange(size, dtype=dtype)
        ak_arry = ak_arry[::2] * -1.0
        idx = ak.Index(ak_arry)
        expected_df = ak.DataFrame({"col1": ak_arry}, index=idx)

        ak_assert_frame_equal(ak_df, expected_df)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", ["float64", "int64"])
    def test_round_trip_pandas_conversion(self, size, dtype):
        a = ak.arange(size, dtype=dtype)
        a = a[::2] * -1.0  # Alter so that index is different from default
        idx = ak.Index(a)
        original_df = ak.DataFrame({"col1": a}, index=idx)
        round_trip_df = ak.DataFrame(original_df.to_pandas(retain_index=True))

        ak_assert_frame_equal(original_df, round_trip_df)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_round_trip_dataframe_conversion2(self, size):
        a = ak.arange(size, dtype="float64") + 0.001

        idx = ak.Index(a)
        df = ak.DataFrame({"col1": a}, index=idx)
        pd_df = df.to_pandas(retain_index=True)
        round_trip_df = ak.DataFrame(pd_df)

        ak.assert_frame_equal(df, round_trip_df)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_pandas_categorical_column(self, size):
        str_val = ak.random_strings_uniform(9, 10, size)
        cat_val = ak.Categorical(str_val)
        num_val = ak.arange(size)

        df = ak.DataFrame({"str_val": str_val, "cat_val": cat_val, "num_val": num_val})
        expected_df = pd.DataFrame(
            {
                "str_val": str_val.to_ndarray(),
                "cat_val": cat_val.to_pandas(),
                "num_val": num_val.to_ndarray(),
            }
        )

        pd_assert_frame_equal(df.to_pandas(retain_index=True), expected_df)

    def test_convenience_init(self):
        dict1 = {"0": [1, 2], "1": [True, False], "2": ["foo", "bar"], "3": [2.3, -1.8]}
        dict2 = {"0": (1, 2), "1": (True, False), "2": ("foo", "bar"), "3": (2.3, -1.8)}
        dict3 = {"0": (1, 2), "1": [True, False], "2": ["foo", "bar"], "3": (2.3, -1.8)}
        dict_dfs = [ak.DataFrame(d) for d in [dict1, dict2, dict3]]

        lists1 = [[1, 2], [True, False], ["foo", "bar"], [2.3, -1.8]]
        lists2 = [(1, 2), (True, False), ("foo", "bar"), (2.3, -1.8)]
        lists3 = [(1, 2), [True, False], ["foo", "bar"], (2.3, -1.8)]
        lists_dfs = [ak.DataFrame(lst) for lst in [lists1, lists2, lists3]]

        for df in dict_dfs + lists_dfs:
            assert isinstance(df, ak.DataFrame)
            assert isinstance(df["0"], ak.pdarray)
            assert df["0"].dtype == int
            assert isinstance(df["1"], ak.pdarray)
            assert df["1"].dtype == bool
            assert isinstance(df["2"], ak.Strings)
            assert df["2"].dtype == str
            assert isinstance(df["3"], ak.pdarray)
            assert df["3"].dtype == float

    def test_client_type_creation(self):
        f = ak.Fields(ak.arange(10), ["A", "B", "c"])
        ip = ak.ip_address(ak.arange(10))
        d = ak.Datetime(ak.arange(10))
        bv = ak.BitVector(ak.arange(10), width=4)

        df_dict = {"fields": f, "ip": ip, "date": d, "bitvector": bv}
        df = ak.DataFrame(df_dict)
        pd_d = [pd.to_datetime(x, unit="ns") for x in d.tolist()]
        pddf = pd.DataFrame(
            {
                "fields": f.tolist(),
                "ip": ip.tolist(),
                "date": pd_d,
                "bitvector": bv.tolist(),
            }
        )
        assert_frame_equal(pddf, df.to_pandas())

        # validate that set max_rows adjusts the repr properly
        shape = f"({df._shape_str()})".replace("(", "[").replace(")", "]")
        pd.set_option("display.max_rows", 4)
        s = df.__repr__().replace(f" ({df._shape_str()})", f"\n\n{shape}")
        assert s == pddf.__repr__()

        pddf = pd.DataFrame({"a": list(range(1000)), "b": list(range(1000))})
        pddf["a"] = pddf["a"].apply(lambda x: "AA" + str(x))
        pddf["b"] = pddf["b"].apply(lambda x: "BB" + str(x))

        df = ak.DataFrame(pddf)
        assert_frame_equal(pddf, df.to_pandas())

        pd.set_option("display.max_rows", 10)
        shape = f"({df._shape_str()})".replace("(", "[").replace(")", "]")
        s = df.__repr__().replace(f" ({df._shape_str()})", f"\n\n{shape}")
        assert s == pddf.__repr__()

    def test_boolean_indexing(self):
        df = self.build_ak_df()
        ref_df = self.build_pd_df()
        row = df[df["userName"] == "Carol"]

        assert len(row) == 1
        assert ref_df[ref_df["userName"] == "Carol"].equals(row.to_pandas(retain_index=True))

    def test_column_indexing(self):
        df = self.build_ak_df()
        ref_df = self.build_pd_df()

        # index validation
        assert isinstance(df.index, ak.Index)
        assert df.index.tolist() == ref_df.index.tolist()

        for cname in df.columns.values:
            col, ref_col = getattr(df, cname), getattr(ref_df, cname)
            assert isinstance(col, ak.Series)
            assert col.tolist() == ref_col.tolist()
            assert isinstance(df[cname], (ak.pdarray, ak.Strings, ak.Categorical))
            assert df[cname].tolist() == ref_df[cname].tolist()

        # check mult-column list
        col_list = ["userName", "amount", "bi"]
        assert isinstance(df[col_list], ak.DataFrame)
        assert_frame_equal(df[col_list].to_pandas(), ref_df[col_list])

        # check multi-column tuple
        col_tup = ("userID", "item", "day", "bi")
        assert isinstance(df[col_tup], ak.DataFrame)
        # pandas only supports lists of columns, not tuples
        assert_frame_equal(df[col_tup].to_pandas(), ref_df[list(col_tup)])

    def test_dtype_prop(self):
        str_arr = ak.random_strings_uniform(1, 5, 3)
        df_dict = {
            "i": ak.arange(3),
            "c_1": ak.arange(3, 6, 1),
            "c_2": ak.arange(6, 9, 1),
            "c_3": str_arr,
            "c_4": ak.Categorical(ak.array(["str"] * 3)),
            "c_5": ak.SegArray(ak.array([0, 9, 14]), ak.arange(20)),
            "c_6": ak.arange(2**200, 2**200 + 3),
        }
        akdf = ak.DataFrame(df_dict)

        assert len(akdf.columns.values) == len(akdf.dtypes)
        # dtypes returns objType for categorical, segarray. We should probably fix
        # this and add a df.objTypes property. pdarrays return actual dtype
        for ref_type, c in zip(
            ["int64", "int64", "int64", "str", "Categorical", "SegArray", "bigint"],
            akdf.columns.values,
        ):
            assert ref_type == str(akdf.dtypes[c])

    def test_drop(self):
        # create an arkouda df.
        df = self.build_ak_df()
        # create pandas df to validate functionality against
        pd_df = self.build_pd_df()

        # test out of place drop
        df_drop = df.drop([0, 1, 2])
        pddf_drop = pd_df.drop(labels=[0, 1, 2])
        pddf_drop.reset_index(drop=True, inplace=True)
        assert_frame_equal(pddf_drop, df_drop.to_pandas())

        df_drop = df.drop("userName", axis=1)
        pddf_drop = pd_df.drop(labels=["userName"], axis=1)
        assert_frame_equal(pddf_drop, df_drop.to_pandas())

        # Test dropping columns
        df.drop("userName", axis=1, inplace=True)
        pd_df.drop(labels=["userName"], axis=1, inplace=True)

        assert_frame_equal(pd_df, df.to_pandas())

        # Test dropping rows
        df.drop([0, 2, 5], inplace=True)
        # pandas retains original indexes when dropping rows, need to reset to line up with arkouda
        pd_df.drop(labels=[0, 2, 5], inplace=True)
        pd_df.reset_index(drop=True, inplace=True)

        assert_frame_equal(pd_df, df.to_pandas())

        # verify that index keys must be ints
        with pytest.raises(TypeError):
            df.drop("index")

        # verify axis can only be 0 or 1
        with pytest.raises(ValueError):
            df.drop("amount", 15)

    def test_drop_duplicates(self):
        df = self.build_ak_df_duplicates()
        ref_df = self.build_pd_df_duplicates()

        dedup = df.drop_duplicates()
        dedup_pd = ref_df.drop_duplicates()
        # pandas retains original indexes when dropping dups, need to reset to line up with arkouda
        dedup_pd.reset_index(drop=True, inplace=True)

        dedup_test = dedup.to_pandas().sort_values("userName").reset_index(drop=True)
        dedup_pd_test = dedup_pd.sort_values("userName").reset_index(drop=True)

        assert_frame_equal(dedup_pd_test, dedup_test)

    def test_shape(self):
        df = self.build_ak_df()

        row, col = df.shape
        assert row == 6
        assert col == 7

    def test_reset_index(self):
        df = self.build_ak_df()

        slice_df = df[ak.array([1, 3, 5])]
        assert slice_df.index.tolist() == [1, 3, 5]

        df_reset = slice_df.reset_index()
        assert df_reset.index.tolist() == [0, 1, 2]
        assert slice_df.index.tolist(), [1, 3, 5]

        df_reset2 = slice_df.reset_index(size=3)
        assert df_reset2.index.tolist() == [0, 1, 2]
        assert slice_df.index.tolist() == [1, 3, 5]

        slice_df.reset_index(inplace=True)
        assert slice_df.index.tolist(), [0, 1, 2]

    def test_rename(self):
        df = self.build_ak_df()

        rename = {"userName": "name_col", "userID": "user_id"}

        # Test out of Place - column
        df_rename = df.rename(rename, axis=1)

        assert "user_id" in df_rename.columns.values
        assert "name_col" in df_rename.columns.values
        assert "userName" not in df_rename.columns.values
        assert "userID" not in df_rename.columns.values
        assert "userID" in df.columns.values
        assert "userName" in df.columns.values
        assert "user_id" not in df.columns.values
        assert "name_col" not in df.columns.values

        # Test in place - column
        df.rename(column=rename, inplace=True)
        assert "user_id" in df.columns.values
        assert "name_col" in df.columns.values
        assert "userName" not in df.columns.values
        assert "userID" not in df.columns.values

        # prep for index renaming
        rename_idx = {1: 17, 2: 93}
        conf = list(range(6))
        conf[1] = 17
        conf[2] = 93

        # Test out of Place - index
        df_rename = df.rename(rename_idx)
        assert df_rename.index.values.tolist() == conf
        assert df.index.values.tolist() == list(range(6))

        # Test in place - index
        df.rename(index=rename_idx, inplace=True)
        assert df.index.values.tolist() == conf

    def test_append(self):
        df = self.build_ak_df()

        df.append(self.build_ak_append())

        ref_df = self.build_pd_df_append()

        # dataframe equality returns series with bool result for each row.
        assert_frame_equal(ref_df, df.to_pandas())

        idx = np.arange(8)
        assert idx.tolist() == df.index.index.tolist()

        df_keyerror = self.build_ak_keyerror()
        with pytest.raises(KeyError):
            df.append(df_keyerror)

        df_typeerror = self.build_ak_typeerror()
        with pytest.raises(TypeError):
            df.append(df_typeerror)

    def test_concat(self):
        df = self.build_ak_df()

        glued = ak.DataFrame.concat([df, self.build_ak_append()])

        ref_df = self.build_pd_df_append()

        # dataframe equality returns series with bool result for each row.
        assert_frame_equal(ref_df, glued.to_pandas())

        df_keyerror = self.build_ak_keyerror()
        with pytest.raises(KeyError):
            ak.DataFrame.concat([df, df_keyerror])

        df_typeerror = self.build_ak_typeerror()
        with pytest.raises(TypeError):
            ak.DataFrame.concat([df, df_typeerror])

    def test_head(self):
        df = self.build_ak_df()
        ref_df = self.build_pd_df()

        hdf = df.head(3)
        hdf_ref = ref_df.head(3).reset_index(drop=True)
        assert_frame_equal(hdf_ref, hdf.to_pandas())

    def test_tail(self):
        df = self.build_ak_df()
        ref_df = self.build_pd_df()

        tdf = df.tail(2)
        tdf_ref = ref_df.tail(2).reset_index(drop=True)
        assert_frame_equal(tdf_ref, tdf.to_pandas())

    def test_groupby_standard(self):
        df = self.build_ak_df()
        gb = df._build_groupby("userName")
        keys, count = gb.size()
        assert keys.tolist() == ["Bob", "Alice", "Carol"]
        assert count.tolist() == [2, 3, 1]
        assert gb.permutation.tolist() == [1, 4, 0, 2, 5, 3]

        gb = df._build_groupby(["userName", "userID"])
        keys, count = gb.size()
        assert len(keys) == 2
        assert keys[0].tolist() == ["Bob", "Alice", "Carol"]
        assert keys[1].tolist() == [222, 111, 333]
        assert count.tolist() == [2, 3, 1]

        # testing counts with IPv4 column
        s = ak.DataFrame({"a": ak.IPv4(ak.arange(1, 5))}).groupby("a").size()
        pds = pd.Series(
            data=np.ones(4, dtype=np.int64),
            index=pd.Index(
                data=np.array(["0.0.0.1", "0.0.0.2", "0.0.0.3", "0.0.0.4"], dtype="<U7"),
                name="a",
            ),
        )
        assert_series_equal(pds, s.to_pandas())

        # testing counts with Categorical column
        s = ak.DataFrame({"a": ak.Categorical(ak.array(["a", "a", "a", "b"]))}).groupby("a").size()
        pds = pd.Series(
            data=np.array([3, 1]),
            index=pd.Index(data=pd.Categorical(np.array(["a", "b"])), name="a"),
        )
        assert_series_equal(pds, s.to_pandas(), check_categorical=False)

    def test_gb_series(self):
        df = self.build_ak_df()

        gb = df._build_groupby("userName", use_series=True)

        c = gb.size(as_series=True)
        assert isinstance(c, ak.Series)
        assert c.index.tolist() == ["Alice", "Bob", "Carol"]
        assert c.values.tolist() == [3, 2, 1]

    @pytest.mark.parametrize("agg", ["sum", "first", "count"])
    def test_gb_aggregations(self, agg):
        df = self.build_ak_df()
        pd_df = self.build_pd_df()
        # remove strings col because many aggregations don't support it
        cols_without_str = list(set(df.columns) - {"userName"})
        df = df[cols_without_str]
        pd_df = pd_df[cols_without_str]

        group_on = "userID"
        ak_result = getattr(df.groupby(group_on), agg)()
        pd_result = getattr(pd_df.groupby(group_on), agg)()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    @pytest.mark.parametrize("agg", ["sum", "first", "count"])
    def test_gb_aggregations_example_numeric_types(self, agg):
        df = self.build_ak_df_example_numeric_types()
        pd_df = df.to_pandas()

        group_on = "gb_id"
        ak_result = getattr(df.groupby(group_on), agg)()
        pd_result = getattr(pd_df.groupby(group_on), agg)()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("agg", ["count", "max", "mean", "median", "min", "std", "sum", "var"])
    def test_gb_aggregations_with_nans(self, agg, dropna):
        df = self.build_ak_df_with_nans()
        # @TODO handle bool columns correctly
        df.drop("bools", axis=1, inplace=True)
        pd_df = df.to_pandas()

        group_on = ["key1", "key2"]
        ak_result = getattr(df.groupby(group_on, dropna=dropna), agg)()
        pd_result = getattr(pd_df.groupby(group_on, as_index=False, dropna=dropna), agg)()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

        # TODO aggregations of string columns not currently supported (even for count)
        df.drop("key1", axis=1, inplace=True)
        df.drop("key2", axis=1, inplace=True)
        pd_df = df.to_pandas()

        group_on = ["nums1", "nums2"]
        ak_result = getattr(df.groupby(group_on, dropna=dropna), agg)()
        pd_result = getattr(pd_df.groupby(group_on, as_index=False, dropna=dropna), agg)()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

        # TODO aggregation mishandling NaN see issue #3765
        df.drop("nums2", axis=1, inplace=True)
        pd_df = df.to_pandas()
        group_on = "nums1"
        ak_result = getattr(df.groupby(group_on, dropna=dropna), agg)()
        pd_result = getattr(pd_df.groupby(group_on, dropna=dropna), agg)()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    @pytest.mark.parametrize("dropna", [True, False])
    def test_count_nan_bug(self, dropna):
        # verify reproducer for #3762 is fixed
        df = ak.DataFrame({"A": [1, 2, 2, np.nan], "B": [3, 4, 5, 6], "C": [1, np.nan, 2, 3]})
        ak_result = df.groupby("A", dropna=dropna).count()
        pd_result = df.to_pandas().groupby("A", dropna=dropna).count()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

        ak_result = df.groupby(["A", "C"], as_index=False, dropna=dropna).count()
        pd_result = df.to_pandas().groupby(["A", "C"], as_index=False, dropna=dropna).count()
        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    def test_gb_aggregations_return_dataframe(self):
        ak_df = self.build_ak_df_example2()
        pd_df = ak_df.to_pandas(retain_index=True)

        pd_result1 = pd_df.groupby(["key1", "key2"], as_index=False)[["count"]].sum()
        ak_result1 = ak_df.groupby(["key1", "key2"]).sum("count")
        assert_frame_equal(pd_result1, ak_result1.to_pandas(retain_index=True))
        assert isinstance(ak_result1, ak.pandas.dataframe.DataFrame)

        pd_result2 = pd_df.groupby(["key1", "key2"], as_index=False)[["count"]].sum()
        ak_result2 = ak_df.groupby(["key1", "key2"]).sum(["count"])
        assert_frame_equal(pd_result2, ak_result2.to_pandas(retain_index=True))
        assert isinstance(ak_result2, ak.pandas.dataframe.DataFrame)

        pd_result3 = pd_df.groupby(["key1", "key2"], as_index=False)[["count", "nums"]].sum()
        ak_result3 = ak_df.groupby(["key1", "key2"]).sum(["count", "nums"])
        assert_frame_equal(pd_result3, ak_result3.to_pandas(retain_index=True))
        assert isinstance(ak_result3, ak.pandas.dataframe.DataFrame)

        pd_result4 = pd_df.groupby(["key1", "key2"], as_index=False).sum(numeric_only=True)
        ak_result4 = ak_df.groupby(["key1", "key2"]).sum()
        assert_frame_equal(pd_result4, ak_result4.to_pandas(retain_index=True))
        assert isinstance(ak_result4, ak.pandas.dataframe.DataFrame)

    def test_gb_aggregations_numeric_types(self):
        ak_df = self.build_ak_df_example_numeric_types()
        pd_df = ak_df.to_pandas(retain_index=True)

        assert_frame_equal(
            ak_df.groupby("gb_id").sum().to_pandas(retain_index=True),
            pd_df.groupby("gb_id").sum(),
        )
        assert set(ak_df.groupby("gb_id").sum().columns.values) == set(
            pd_df.groupby("gb_id").sum().columns.values
        )

        assert_frame_equal(
            ak_df.groupby(["gb_id"]).sum().to_pandas(retain_index=True),
            pd_df.groupby(["gb_id"]).sum(),
        )
        assert set(ak_df.groupby(["gb_id"]).sum().columns.values) == set(
            pd_df.groupby(["gb_id"]).sum().columns.values
        )

    def test_gb_size_single(self):
        ak_df = self.build_ak_df_example_numeric_types()
        pd_df = ak_df.to_pandas(retain_index=True)

        assert_frame_equal(
            ak_df.groupby("gb_id", as_index=False).size().to_pandas(retain_index=True),
            pd_df.groupby("gb_id", as_index=False).size(),
        )

        assert_frame_equal(
            ak_df.groupby(["gb_id"], as_index=False).size().to_pandas(retain_index=True),
            pd_df.groupby(["gb_id"], as_index=False).size(),
        )

    def test_gb_size_multiple(self):
        ak_df = self.build_ak_df_example2()
        pd_df = ak_df.to_pandas(retain_index=True)

        pd_result1 = pd_df.groupby(["key1", "key2"], as_index=False).size()
        ak_result1 = ak_df.groupby(["key1", "key2"], as_index=False).size()
        assert_frame_equal(pd_result1, ak_result1.to_pandas(retain_index=True))
        assert isinstance(ak_result1, ak.pandas.dataframe.DataFrame)

        assert_frame_equal(
            ak_df.groupby(["key1", "key2"], as_index=False).size().to_pandas(retain_index=True),
            pd_df.groupby(["key1", "key2"], as_index=False).size(),
        )

        assert_series_equal(
            ak_df.groupby(["key1", "key2"], as_index=True).size().to_pandas(),
            pd_df.groupby(["key1", "key2"], as_index=True).size(),
        )

        assert_frame_equal(
            ak_df.groupby(["key1"], as_index=False).size().to_pandas(retain_index=True),
            pd_df.groupby(["key1"], as_index=False).size(),
        )

        assert_frame_equal(
            ak_df.groupby("key1", as_index=False).size().to_pandas(retain_index=True),
            pd_df.groupby("key1", as_index=False).size(),
        )

        assert_series_equal(
            ak_df.groupby("key1").size(as_series=True).to_pandas(),
            pd_df.groupby("key1").size(),
        )

    def test_gb_size_match_pandas(self):
        ak_df = self.build_ak_df_with_nans()
        pd_df = ak_df.to_pandas(retain_index=True)

        for as_index in [True, False]:
            for dropna in [True, False]:
                for gb_keys in [
                    "nums1",
                    "nums2",
                    ["nums1", "nums2"],
                    ["nums1", "nums3"],
                    ["nums3", "nums1"],
                    ["nums1", "nums2", "nums3"],
                ]:
                    ak_result = ak_df.groupby(gb_keys, as_index=as_index, dropna=dropna).size()
                    pd_result = pd_df.groupby(gb_keys, as_index=as_index, dropna=dropna).size()

                    if isinstance(ak_result, ak.pandas.dataframe.DataFrame):
                        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)
                    else:
                        assert_series_equal(ak_result.to_pandas(), pd_result)

    def test_gb_size_as_index_cases(self):
        ak_df = self.build_ak_df_example2()
        pd_df = ak_df.to_pandas(retain_index=True)

        pd1 = pd_df.groupby(["key1", "key2"], as_index=False).size()
        ak1 = ak_df.groupby(["key1", "key2"], as_index=False).size()
        assert_frame_equal(pd1, ak1.to_pandas())

        pd2 = pd_df.groupby(["key1", "key2"], as_index=True).size()
        ak2 = ak_df.groupby(["key1", "key2"], as_index=True).size()
        assert_series_equal(pd2, ak2.to_pandas(), check_names=False)

        pd3 = pd_df.groupby(["key1"], as_index=False).size()
        ak3 = ak_df.groupby(["key1"], as_index=False).size()
        assert_frame_equal(pd3, ak3.to_pandas())

        pd4 = pd_df.groupby(["key1"], as_index=True).size()
        ak4 = ak_df.groupby(["key1"], as_index=True).size()
        assert_series_equal(pd4, ak4.to_pandas())

        pd5 = pd_df.groupby("key1", as_index=False).size()
        ak5 = ak_df.groupby("key1", as_index=False).size()
        assert_frame_equal(pd5, ak5.to_pandas())

        pd6 = pd_df.groupby("key1", as_index=True).size()
        ak6 = ak_df.groupby("key1", as_index=True).size()
        assert_series_equal(pd6, ak6.to_pandas())

    def test_argsort(self):
        df = self.build_ak_df()

        p = df.argsort(key="userName")
        assert p.tolist() == [0, 2, 5, 1, 4, 3]

        p = df.argsort(key="userName", ascending=False)
        assert p.tolist() == [3, 4, 1, 5, 2, 0]

    def test_coargsort(self):
        df = self.build_ak_df()

        p = df.coargsort(keys=["userID", "amount"])
        assert p.tolist() == [0, 5, 2, 1, 4, 3]

        p = df.coargsort(keys=["userID", "amount"], ascending=False)
        assert p.tolist() == [3, 4, 1, 2, 5, 0]

    def test_sort_values(self):
        userid = [111, 222, 111, 333, 222, 111]
        userid_ak = ak.array(userid)

        # sort userid to build dataframes to reference
        userid.sort()

        df = ak.DataFrame({"userID": userid_ak})
        ord = df.sort_values()
        assert_frame_equal(pd.DataFrame(data=userid, columns=["userID"]), ord.to_pandas())
        ord = df.sort_values(ascending=False)
        userid.reverse()
        assert_frame_equal(pd.DataFrame(data=userid, columns=["userID"]), ord.to_pandas())

        df = self.build_ak_df()
        ord = df.sort_values(by="userID")
        ref_df = self.build_pd_df()
        ref_df = ref_df.sort_values(by="userID").reset_index(drop=True)
        assert_frame_equal(ref_df, ord.to_pandas())

        ord = df.sort_values(by=["userID", "day"])
        ref_df = ref_df.sort_values(by=["userID", "day"]).reset_index(drop=True)
        assert_frame_equal(ref_df, ord.to_pandas())

        with pytest.raises(TypeError):
            df.sort_values(by=1)

    def test_sort_index(self):
        ak_df = self.build_ak_df_example_numeric_types()
        ak_df["string"] = ak.array(
            [
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "a",
                "b",
                "c",
                "d",
                "e",
                "p",
                "q",
                "r",
                "s",
                "t",
            ]
        )
        ak_df["negs"] = -1 * ak_df["int64"]

        group_bys = [
            "gb_id",
            "float64",
            "int64",
            "uint64",
            "bigint",
            "negs",
            "string",
            ["gb_id", "negs"],
        ]
        for group_by in group_bys:
            ak_result = ak_df.groupby(group_by).size()
            pd_result = ak_result.to_pandas()
            if isinstance(ak_result, ak.pandas.dataframe.DataFrame):
                assert_frame_equal(
                    ak_result.sort_index().to_pandas(retain_index=True),
                    pd_result.sort_index(),
                )
            else:
                assert_series_equal(ak_result.sort_index().to_pandas(), pd_result.sort_index())

    def test_intx(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        df_1 = ak.DataFrame({"user_name": username, "user_id": userid})

        username = ak.array(["Bob", "Alice"])
        userid = ak.array([222, 445])
        df_2 = ak.DataFrame({"user_name": username, "user_id": userid})

        rows = ak.intx(df_1, df_2)
        assert rows.tolist() == [False, True, False, False, True, False]

        df_3 = ak.DataFrame({"user_name": username, "user_number": userid})
        with pytest.raises(ValueError):
            rows = ak.intx(df_1, df_3)

    def test_apply_perm(self):
        df = self.build_ak_df()
        ref_df = self.build_pd_df()

        ord = df.sort_values(by="userID")
        perm_list = [0, 3, 1, 5, 4, 2]
        default_perm = ak.array(perm_list)
        ord.apply_permutation(default_perm)

        ord_ref = ref_df.sort_values(by="userID").reset_index(drop=True)
        ord_ref = ord_ref.reindex(perm_list).reset_index(drop=True)
        assert_frame_equal(ord_ref, ord.to_pandas())

    def test_filter_by_range(self):
        userid = ak.array([111, 222, 111, 333, 222, 111])
        amount = ak.array([0, 1, 1, 2, 3, 15])
        df = ak.DataFrame({"userID": userid, "amount": amount})

        filtered = df.filter_by_range(keys=["userID"], low=1, high=2)
        assert filtered.tolist() == [False, True, False, True, True, False]

    def test_copy(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        df = ak.DataFrame({"userName": username, "userID": userid})

        df_copy = df.copy(deep=True)
        assert_frame_equal(df.to_pandas(), df_copy.to_pandas())

        df_copy.__setitem__("userID", ak.array([1, 2, 1, 3, 2, 1]))
        assert df["userID"].tolist() != df_copy["userID"].tolist()

        df_copy = df.copy(deep=False)
        df_copy.__setitem__("userID", ak.array([1, 2, 1, 3, 2, 1]))
        assert_frame_equal(df.to_pandas(), df_copy.to_pandas())

    def test_isin(self):
        df = ak.DataFrame({"col_A": ak.array([7, 3]), "col_B": ak.array([1, 9])})

        # test against pdarray
        test_df = df.isin(ak.array([0, 1]))
        assert test_df["col_A"].tolist() == [False, False]
        assert test_df["col_B"].tolist() == [True, False]

        # Test against dict
        test_df = df.isin({"col_A": ak.array([0, 3])})
        assert test_df["col_A"].tolist() == [False, True]
        assert test_df["col_B"].tolist() == [False, False]

        # test against series
        i = ak.Index(ak.arange(2))
        s = ak.Series(data=ak.array([3, 9]), index=i.index)
        test_df = df.isin(s)
        assert test_df["col_A"].tolist() == [False, False]
        assert test_df["col_B"].tolist() == [False, True]

        # test against another dataframe
        other_df = ak.DataFrame({"col_A": ak.array([7, 3], dtype=ak.bigint), "col_C": ak.array([0, 9])})
        test_df = df.isin(other_df)
        assert test_df["col_A"].tolist() == [True, True]
        assert test_df["col_B"].tolist() == [False, False]

    def test_count(self):
        akdf = self.build_ak_df_with_nans()
        pddf = akdf.to_pandas()

        for truth in [True, False]:
            for axis in [0, 1, "index", "columns"]:
                assert_series_equal(
                    akdf.count(axis=axis, numeric_only=truth).to_pandas(),
                    pddf.count(axis=axis, numeric_only=truth),
                )

    def test_corr(self):
        df = ak.DataFrame({"col1": [1, 2], "col2": [-1, -2]})
        corr = df.corr()
        pd_corr = df.to_pandas().corr()
        assert_frame_equal(corr.to_pandas(retain_index=True), pd_corr)

        for i in range(5):
            df = ak.DataFrame({"col1": ak.randint(0, 10, 10), "col2": ak.randint(0, 10, 10)})
            corr = df.corr()
            pd_corr = df.to_pandas().corr()
            assert_frame_equal(corr.to_pandas(retain_index=True), pd_corr)

    def test_multiindex_compat(self):
        # Added for testing Issue #1505
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        df.groupby(["a", "b"]).sum("c")

    def test_uint_greediness(self):
        # default to uint when all supportedInt and any value > 2**63
        # to avoid loss of precision see (#1983)
        df = pd.DataFrame({"Test": [2**64 - 1, 0]})
        assert df["Test"].dtype == ak.uint64

    def test_head_tail_datetime_display(self):
        # Reproducer for issue #2596
        values = ak.array([1689221916000000] * 100, dtype=ak.int64)
        dt = ak.Datetime(values, unit="u")
        df = ak.DataFrame({"Datetime from Microseconds": dt})
        # verify _get_head_tail and _get_head_tail_server match
        assert df._get_head_tail_server().__repr__() == df._get_head_tail().__repr__()

    def test_head_tail_resetting_index(self):
        # Test that issue #2183 is resolved
        df = ak.DataFrame({"cnt": ak.arange(65)})
        # Note we have to call __repr__ to trigger head_tail_server call

        bool_idx = df[df["cnt"] > 3]
        bool_idx.__repr__()
        assert bool_idx.index.index.tolist() == list(range(4, 65))

        slice_idx = df[:]
        slice_idx.__repr__()
        assert slice_idx.index.index.tolist() == list(range(65))

        # verify it persists non-int Index
        idx = ak.concatenate([ak.zeros(5, bool), ak.ones(60, bool)])
        df = ak.DataFrame({"cnt": ak.arange(65)}, index=idx)

        bool_idx = df[df["cnt"] > 3]
        bool_idx.__repr__()
        # the new index is first False and rest True (because we lose first 4),
        # so equivalent to arange(61, bool)
        assert bool_idx.index.index.tolist() == ak.arange(61, dtype=bool).tolist()

        slice_idx = df[:]
        slice_idx.__repr__()
        assert slice_idx.index.index.tolist() == idx.tolist()

    def test_subset(self):
        df = ak.DataFrame(
            {
                "a": ak.arange(100),
                "b": ak.randint(0, 20, 100),
                "c": ak.random_strings_uniform(0, 16, 100),
                "d": ak.randint(25, 75, 100),
            }
        )
        df2 = df[["a", "b"]]
        assert ["a", "b"] == df2.columns.values
        assert df.index.tolist() == df2.index.tolist()
        assert df["a"].tolist() == df2["a"].tolist()
        assert df["b"].tolist() == df2["b"].tolist()

    @pytest.mark.timeout(9000 if pytest.asan else 1800)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_col_merge(self, size):
        size = min(size, 1000)
        a = ak.randint(-size // 10, size // 10, size, seed=pytest.seed)
        b = ak.randint(-size // 10, size // 10, size, seed=pytest.seed + 1)
        c = ak.randint(-size // 10, size // 10, size, seed=pytest.seed + 2)
        d = ak.randint(-size // 10, size // 10, size, seed=pytest.seed + 3)
        ones = ak.ones(size, int)
        altr = alternating_1_0(size)
        for truth in itertools.product([True, False], repeat=3):
            left_arrs = [pda if t else pda_to_str_helper(pda) for pda, t in zip([a, b, ones], truth)]
            right_arrs = [pda if t else pda_to_str_helper(pda) for pda, t in zip([c, d, altr], truth)]
            left_df = ak.DataFrame({k: v for k, v in zip(["first", "second", "third"], left_arrs)})
            right_df = ak.DataFrame({k: v for k, v in zip(["first", "second", "third"], right_arrs)})
            l_pd, r_pd = left_df.to_pandas(), right_df.to_pandas()

            for how in "inner", "left", "right":
                for on in (
                    ["first", "third"],
                    ["second", "third"],
                    None,
                ):
                    ak_merge = ak.merge(left_df, right_df, on=on, how=how)
                    pd_merge = pd.merge(l_pd, r_pd, on=on, how=how)

                    sorted_column_names = sorted(ak_merge.columns.values)
                    assert sorted_column_names == sorted(pd_merge.columns.values)
                    for col in sorted_column_names:
                        from_ak = ak_merge[col].to_ndarray()
                        from_pd = pd_merge[col].to_numpy()
                        if isinstance(ak_merge[col], ak.pdarray):
                            assert np.allclose(np.sort(from_ak), np.sort(from_pd), equal_nan=True)
                        else:
                            # we have to cast to str because pandas arrays converted to numpy
                            # have dtype object and have float NANs in line with the str values
                            assert (np.sort(from_ak) == np.sort(from_pd.astype(str))).all()
                    # TODO arkouda seems to be sometimes convert columns to floats on a right merge
                    #  when pandas doesnt. Eventually we want to test frame_equal, not just value
                    #  equality
                    # from pandas.testing import assert_frame_equal
                    # assert_frame_equal(sorted_ak.to_pandas()[sorted_column_names],
                    # sorted_pd[sorted_column_names])

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    @pytest.mark.parametrize(
        ("left_on", "right_on"),
        [
            ("a", "x"),
            ("c", "z"),
            (["a", "c"], ["x", "z"]),
        ],
    )
    def test_merge_left_on_right_on(self, how, left_on, right_on):
        size = 1000

        a = ak.randint(-size // 10, size // 10, size, seed=pytest.seed)
        b = ak.randint(-size // 10, size // 10, size, seed=pytest.seed + 1)
        c = ak.randint(-size // 10, size // 10, size, seed=pytest.seed + 2)
        d = ak.randint(-size // 10, size // 10, size, seed=pytest.seed + 3)
        ones = ak.ones(size, int)
        altr = alternating_1_0(size)

        for truth in itertools.product([True, False], repeat=3):
            left_arrs = [pda if t else pda_to_str_helper(pda) for pda, t in zip([a, b, ones], truth)]
            right_arrs = [pda if t else pda_to_str_helper(pda) for pda, t in zip([c, d, altr], truth)]

            left_df = ak.DataFrame({k: v for k, v in zip(["a", "b", "c"], left_arrs)})
            right_df = ak.DataFrame({k: v for k, v in zip(["x", "y", "z"], right_arrs)})

            l_pd, r_pd = left_df.to_pandas(), right_df.to_pandas()

            ak_merge = ak.merge(
                left_df,
                right_df,
                left_on=left_on,
                right_on=right_on,
                how=how,
                sort=True,
            )
            pd_merge = pd.merge(
                l_pd,
                r_pd,
                left_on=left_on,
                right_on=right_on,
                how=how,
                copy=True,
                sort=True,
            )

            # Numeric sorting works okay when floats aren't involved
            if truth == (True, True, True) and not ak_merge.isna().any():
                assert_frame_equivalent(ak_merge, pd_merge)

            # If there are strings involved, do it this way
            else:
                sorted_column_names = sorted(ak_merge.columns.values)
                assert sorted_column_names == sorted(pd_merge.columns.values)

                for col in sorted_column_names:
                    from_ak = ak_merge[col].to_ndarray()
                    from_pd = pd_merge[col].to_numpy()

                    if isinstance(ak_merge[col], ak.pdarray):
                        assert np.allclose(np.sort(from_ak), np.sort(from_pd), equal_nan=True)
                    else:
                        assert (np.sort(from_ak) == np.sort(from_pd.astype(str))).all()

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    @pytest.mark.parametrize(
        ("left_on", "right_on"),
        [
            ("a", "x"),
            ("c", "z"),
            (["a", "c"], ["x", "z"]),
        ],
    )
    def test_merge_categoricals_left_on_right_on(self, how, left_on, right_on):
        size = 20

        a = ak.randint(0, 5, size, seed=pytest.seed)
        b = ak.randint(0, 5, size, seed=pytest.seed + 1)
        c = ak.randint(0, 5, size, seed=pytest.seed + 2)
        d = ak.randint(0, 5, size, seed=pytest.seed + 3)

        a = ak.Categorical(ak.array([str(i) for i in list(a.to_ndarray())]))
        b = ak.Categorical(ak.array([str(i) for i in list(b.to_ndarray())]))
        c = ak.Categorical(ak.array([str(i) for i in list(c.to_ndarray())]))
        d = ak.Categorical(ak.array([str(i) for i in list(d.to_ndarray())]))

        ones = ak.ones(size, int)
        altr = alternating_1_0(size)

        left_arrs = [pda for pda in [a, b, ones]]
        right_arrs = [pda for pda in [c, d, altr]]

        left_df = ak.DataFrame({k: v for k, v in zip(["a", "b", "c"], left_arrs)})
        right_df = ak.DataFrame({k: v for k, v in zip(["x", "y", "z"], right_arrs)})

        l_pd, r_pd = left_df.to_pandas(), right_df.to_pandas()

        ak_merge = ak.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how, sort=True)
        pd_merge = pd.merge(
            l_pd,
            r_pd,
            left_on=left_on,
            right_on=right_on,
            how=how,
            copy=True,
            sort=True,
        )

        # Numeric sorting works okay when floats aren't involved
        if not ak_merge.isna().any():
            assert_frame_equivalent(ak_merge, pd_merge)

        # If there are strings involved, do it this way
        else:
            sorted_column_names = sorted(ak_merge.columns.values)
            assert sorted_column_names == sorted(pd_merge.columns.values)

            for col in sorted_column_names:
                from_ak = ak_merge[col].to_ndarray()
                from_pd = pd_merge[col].to_numpy()

                if isinstance(ak_merge[col], ak.pdarray):
                    assert np.allclose(np.sort(from_ak), np.sort(from_pd), equal_nan=True)
                elif isinstance(ak_merge[col], ak.Categorical):
                    na = ak_merge[col].na_value
                    from_ak = np.where(from_ak == na, "nan", from_ak)
                    from_ak = np.sort(from_ak)
                    from_pd = np.sort(from_pd.astype(str))

                    assert (from_ak == from_pd).all()
                else:
                    assert (np.sort(from_ak) == np.sort(from_pd.astype(str))).all()

    @pytest.mark.parametrize(
        "df_init, merge",
        [
            pytest.param(
                ak.DataFrame,
                lambda df_1, df_2: df_1.merge(df_2, on=["idx"], how="inner"),
                id="Arkouda via `DataFrame.merge`",
            ),
            pytest.param(
                ak.DataFrame,
                lambda df_1, df_2: ak.merge(df_1, df_2, on=["idx"], how="inner"),
                id="Arkouda via `arkouda.merge`",
            ),
        ],
    )
    def test_merge_instance_and_module(self, df_init, merge):
        df_first = df_init(
            {
                "idx": list(range(10)),
                "val": list(range(10)),
            }
        )
        df_second = df_init(
            {
                "idx": list(range(10)),
                "val": list(map(lambda x: x + 3, range(10))),
            }
        )
        df_merged = merge(df_first, df_second)
        assert (df_merged["val_y"] - df_merged["val_x"] == 3).all()

    def test_isna_notna(self):
        df = ak.DataFrame(
            {
                "A": [np.nan, 2, 2, 3],
                "B": [3, np.nan, 5, 0],
                "C": [1, np.nan, 2, np.nan],
                "D": ["a", "b", "c", ""],
            }
        )
        assert_frame_equal(df.isna().to_pandas(), df.to_pandas().isna())
        assert_frame_equal(df.notna().to_pandas(), df.to_pandas().notna())

    def test_any_all(self):
        df1 = ak.DataFrame(
            {
                "A": [True, True, True, True],
                "B": [True, True, True, False],
                "C": [True, False, True, False],
                "D": [False, False, False, False],
                "E": [0, 1, 2, 3],
                "F": ["a", "b", "c", ""],
            }
        )

        df2 = ak.DataFrame(
            {
                "A": [True, True, True, True],
                "B": [True, True, True, True],
            }
        )

        df3 = ak.DataFrame(
            {
                "A": [False, False],
                "B": [False, False],
            }
        )

        df4 = ak.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})

        df5 = ak.DataFrame()

        for df in [df1, df2, df3, df4, df5]:
            for axis in [0, 1, "index", "columns"]:
                # There's a bug in assert_series_equal where two empty series will not register as equal.
                if df.to_pandas().any(axis=axis, bool_only=True).empty:
                    assert df.any(axis=axis).to_pandas().empty is True
                else:
                    assert_series_equal(
                        df.any(axis=axis).to_pandas(),
                        df.to_pandas().any(axis=axis, bool_only=True),
                    )
                if df.to_pandas().all(axis=axis, bool_only=True).empty:
                    assert df.all(axis=axis).to_pandas().empty is True
                else:
                    assert_series_equal(
                        df.all(axis=axis).to_pandas(),
                        df.to_pandas().all(axis=axis, bool_only=True),
                    )
            # Test is axis=None
            assert df.any(axis=None) == df.to_pandas().any(axis=None, bool_only=True)
            assert df.all(axis=None) == df.to_pandas().all(axis=None, bool_only=True)

    def test_dropna(self):
        df1 = ak.DataFrame(
            {
                "A": [True, True, True, True],
                "B": [1, np.nan, 2, np.nan],
                "C": [1, 2, 3, np.nan],
                "D": [False, False, False, False],
                "E": [1, 2, 3, 4],
                "F": ["a", "b", "c", "d"],
                "G": [1, 2, 3, 4],
            }
        )

        df2 = ak.DataFrame(
            {
                "A": [True, True, True, True],
                "B": [True, True, True, True],
            }
        )

        df3 = ak.DataFrame(
            {
                "A": [False, False],
                "B": [False, False],
            }
        )

        df4 = ak.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})

        df5 = ak.DataFrame()

        for df in [df1, df2, df3, df4, df5]:
            for axis in [0, 1, "index", "columns"]:
                for how in ["any", "all"]:
                    for ignore_index in [True, False]:
                        assert_frame_equal(
                            df.dropna(axis=axis, how=how, ignore_index=ignore_index).to_pandas(
                                retain_index=True
                            ),
                            df.to_pandas(retain_index=True).dropna(
                                axis=axis, how=how, ignore_index=ignore_index
                            ),
                        )

                for thresh in [0, 1, 2, 3, 4, 5]:
                    if df.to_pandas(retain_index=True).dropna(axis=axis, thresh=thresh).empty:
                        assert (
                            df.dropna(axis=axis, thresh=thresh).to_pandas(retain_index=True).empty
                            is True
                        )

                    else:
                        assert_frame_equal(
                            df.dropna(axis=axis, thresh=thresh).to_pandas(retain_index=True),
                            df.to_pandas(retain_index=True).dropna(axis=axis, thresh=thresh),
                        )

    def test_memory_usage(self):
        dtypes = [ak.int64, ak.float64, ak.bool_]
        data = dict([(str(ak.dtype(t)), ak.ones(5000, dtype=ak.int64).astype(t)) for t in dtypes])
        df = ak.DataFrame(data)
        ak_memory_usage = df.memory_usage()
        pd_memory_usage = pd.Series(
            [40000, 40000, 40000, 5000], index=["Index", "int64", "float64", "bool"]
        )
        assert_series_equal(ak_memory_usage.to_pandas(), pd_memory_usage)

        assert df.memory_usage_info(unit="B") == "125000.00 B"
        assert df.memory_usage_info(unit="KB") == "122.07 KB"
        assert df.memory_usage_info(unit="MB") == "0.12 MB"
        assert df.memory_usage_info(unit="GB") == "0.00 GB"

        ak_memory_usage = df.memory_usage(index=False)
        pd_memory_usage = pd.Series([40000, 40000, 5000], index=["int64", "float64", "bool"])
        assert_series_equal(ak_memory_usage.to_pandas(), pd_memory_usage)

        ak_memory_usage = df.memory_usage(unit="KB")
        pd_memory_usage = pd.Series(
            [39.0625, 39.0625, 39.0625, 4.88281],
            index=["Index", "int64", "float64", "bool"],
        )
        assert_series_equal(ak_memory_usage.to_pandas(), pd_memory_usage)

    def test_to_markdown(self):
        df = ak.DataFrame({"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]})
        assert (
            df.to_markdown() == "+----+------------+------------+\n"
            "|    | animal_1   | animal_2   |\n"
            "+====+============+============+\n"
            "|  0 | elk        | dog        |\n"
            "+----+------------+------------+\n"
            "|  1 | pig        | quetzal    |\n"
            "+----+------------+------------+"
        )

        assert (
            df.to_markdown(index=False) == "+------------+------------+\n"
            "| animal_1   | animal_2   |\n"
            "+============+============+\n"
            "| elk        | dog        |\n"
            "+------------+------------+\n"
            "| pig        | quetzal    |\n"
            "+------------+------------+"
        )

        assert df.to_markdown(tablefmt="grid") == df.to_pandas().to_markdown(tablefmt="grid")
        assert df.to_markdown(tablefmt="grid", index=False) == df.to_pandas().to_markdown(
            tablefmt="grid", index=False
        )
        assert df.to_markdown(tablefmt="jira") == df.to_pandas().to_markdown(tablefmt="jira")

    def test_sample_hypothesis_testing(self):
        # perform a weighted sample and use chisquare to test
        # if the observed frequency matches the expected frequency

        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05

        rng = ak.random.default_rng(pytest.seed)
        num_samples = 10**4

        prob_arr = ak.array([0.35, 0.10, 0.55])
        weights = ak.concatenate([prob_arr, prob_arr, prob_arr])
        keys = ak.concatenate([ak.zeros(3, int), ak.ones(3, int), ak.full(3, 2, int)])
        values = ak.arange(9)

        akdf = ak.DataFrame({"keys": keys, "vals": values})

        g = akdf.groupby("keys")

        weighted_sample = g.sample(n=num_samples, replace=True, weights=weights, random_state=rng)

        # count how many of each category we saw
        uk, f_obs = ak.GroupBy(weighted_sample["vals"]).size()

        # I think the keys should always be sorted but just in case
        if not ak.is_sorted(uk):
            f_obs = f_obs[ak.argsort(uk)]

        f_exp = weights * num_samples
        _, pval = akchisquare(f_obs=f_obs, f_exp=f_exp)

        # if pval <= 0.05, the difference from the expected distribution is significant
        assert pval > 0.05

    def test_sample_flags(self):
        # use numpy to randomly generate a set seed, but seed the generator from the default
        iseed = np.random.default_rng(pytest.seed).choice(2**63)
        cfg = ak.get_config()

        rng = ak.random.default_rng(iseed)
        weights = rng.uniform(size=12)
        a_vals = [
            rng.integers(0, 2**32, size=12, dtype="uint"),
            rng.uniform(-1.0, 1.0, size=12),
            rng.integers(0, 1, size=12, dtype="bool"),
            rng.integers(-(2**32), 2**32, size=12, dtype="int"),
        ]
        grouping_keys = ak.concatenate([ak.zeros(4, int), ak.ones(4, int), ak.full(4, 2, int)])
        rng.shuffle(grouping_keys)

        choice_arrays = []
        # return_indices and permute_samples are tested by the dataframe version
        rng = ak.random.default_rng(iseed)
        for a in a_vals:
            for size in 2, 4:
                for replace in True, False:
                    for p in [None, weights]:
                        akdf = ak.DataFrame({"keys": grouping_keys, "vals": a})
                        g = akdf.groupby("keys")
                        choice_arrays.append(
                            g.sample(n=size, replace=replace, weights=p, random_state=rng)
                        )
                        choice_arrays.append(
                            g.sample(
                                frac=(size / 4),
                                replace=replace,
                                weights=p,
                                random_state=rng,
                            )
                        )

        # reset generator to ensure we get the same arrays
        rng = ak.random.default_rng(iseed)
        for a in a_vals:
            for size in 2, 4:
                for replace in True, False:
                    for p in [None, weights]:
                        previous1 = choice_arrays.pop(0)
                        previous2 = choice_arrays.pop(0)

                        akdf = ak.DataFrame({"keys": grouping_keys, "vals": a})
                        g = akdf.groupby("keys")
                        current1 = g.sample(n=size, replace=replace, weights=p, random_state=rng)
                        current2 = g.sample(
                            frac=(size / 4),
                            replace=replace,
                            weights=p,
                            random_state=rng,
                        )

                        res = (np.allclose(previous1["vals"].tolist(), current1["vals"].tolist())) and (
                            np.allclose(previous2["vals"].tolist(), current2["vals"].tolist())
                        )
                        if not res:
                            warnings.warn(f"\nnum locales: {cfg['numLocales']}")
                            warnings.warn(f"Failure with seed:\n{iseed}")
                        assert res

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_head_tail(self, size):
        bool_col = ak.full(size, False, dtype=ak.bool_)
        bool_col[::2] = True

        df = ak.DataFrame(
            {
                "a": ak.arange(size) % 3,
                "b": ak.arange(size, dtype="int64"),
                "c": ak.arange(size, dtype="float64"),
                "d": ak.random_strings_uniform(size=size, minlen=1, maxlen=2, seed=pytest.seed),
                "e": bool_col,
            }
        )

        size_range = ak.arange(size)
        zeros_idx = size_range[df["a"] == 0][0:2]
        ones_idx = size_range[df["a"] == 1][0:2]
        twos_idx = size_range[df["a"] == 2][0:2]
        head_expected_idx = ak.concatenate([zeros_idx, ones_idx, twos_idx])

        def get_head_values(col):
            zeros_values = df[col][zeros_idx]
            ones_values = df[col][ones_idx]
            twos_values = df[col][twos_idx]
            expected_values = ak.concatenate([zeros_values, ones_values, twos_values])
            return expected_values

        head_df = df.groupby("a").head(n=2, sort_index=False)
        assert ak.all(head_df.index == head_expected_idx)
        for col in df.columns:
            assert ak.all(head_df[col] == get_head_values(col))

        head_df_sorted = df.groupby("a").head(n=2, sort_index=True)
        from pandas.testing import assert_frame_equal

        assert_frame_equal(
            head_df_sorted.to_pandas(retain_index=True),
            df.to_pandas(retain_index=True).groupby("a").head(n=2),
        )

        #   Now test tail
        tail_zeros_idx = size_range[df["a"] == 0][-2:]
        tail_ones_idx = size_range[df["a"] == 1][-2:]
        tail_twos_idx = size_range[df["a"] == 2][-2:]
        tail_expected_idx = ak.concatenate([tail_zeros_idx, tail_ones_idx, tail_twos_idx])

        def get_tail_values(col):
            tail_zeros_values = df[col][tail_zeros_idx]
            tail_ones_values = df[col][tail_ones_idx]
            tail_twos_values = df[col][tail_twos_idx]
            tail_expected_values = ak.concatenate(
                [tail_zeros_values, tail_ones_values, tail_twos_values]
            )
            return tail_expected_values

        tail_df = df.groupby("a").tail(n=2, sort_index=False)
        assert ak.all(tail_df.index == tail_expected_idx)

        for col in df.columns:
            assert ak.all(tail_df[col] == get_tail_values(col))

        tail_df_sorted = df.groupby("a").tail(n=2, sort_index=True)
        from pandas.testing import assert_frame_equal

        assert_frame_equal(
            tail_df_sorted.to_pandas(retain_index=True),
            df.to_pandas(retain_index=True).groupby("a").tail(n=2),
        )

    def test_assign(self):
        ak_df = ak.DataFrame(
            {"temp_c": ak.array([17.0, 25.0])}, index=ak.array(["Portland", "Berkeley"])
        )
        pd_df = ak_df.to_pandas()

        assert_frame_equal(
            ak_df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32).to_pandas(),
            pd_df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32),
        )

        assert_frame_equal(
            ak_df.assign(temp_f=ak_df["temp_c"] * 9 / 5 + 32).to_pandas(),
            pd_df.assign(temp_f=pd_df["temp_c"] * 9 / 5 + 32),
        )

        assert_frame_equal(
            ak_df.assign(
                temp_f=lambda x: x["temp_c"] * 9 / 5 + 32,
                temp_k=lambda x: (x["temp_f"] + 459.67) * 5 / 9,
            ).to_pandas(),
            pd_df.assign(
                temp_f=lambda x: x["temp_c"] * 9 / 5 + 32,
                temp_k=lambda x: (x["temp_f"] + 459.67) * 5 / 9,
            ),
        )


def pda_to_str_helper(pda):
    return ak.array([f"str {i}" for i in pda.tolist()])
