import itertools

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import arkouda as ak


class TestDataFrame:
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
                "bigint": ak.randint(0, 10, 20, dtype=ak.uint64) + 2**200,
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
        bi = (np.arange(8) + 2**200).tolist()
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
        x = [
            np.arange(size),
            np.random.randint(0, 5, size),
            np.random.randint(5, 10, size),
        ]
        pddf = pd.DataFrame(x)
        pddf.columns = pddf.columns.astype(str)
        akdf = ak.DataFrame([ak.array(val) for val in list(zip(*x))])
        assert isinstance(akdf, ak.DataFrame)
        assert len(akdf) == len(pddf)
        # arkouda does not allow for numeric columns.
        assert akdf.column_names == [str(x) for x in pddf.columns.values]
        # use the columns from the pandas created for equivalence check
        # these should be equivalent
        ak_to_pd = akdf.to_pandas()
        assert_frame_equal(pddf, ak_to_pd)

    def test_client_type_creation(self):
        f = ak.Fields(ak.arange(10), ["A", "B", "c"])
        ip = ak.ip_address(ak.arange(10))
        d = ak.Datetime(ak.arange(10))
        bv = ak.BitVector(ak.arange(10), width=4)

        df_dict = {"fields": f, "ip": ip, "date": d, "bitvector": bv}
        df = ak.DataFrame(df_dict)
        pd_d = [pd.to_datetime(x, unit="ns") for x in d.to_list()]
        pddf = pd.DataFrame(
            {"fields": f.to_list(), "ip": ip.to_list(), "date": pd_d, "bitvector": bv.to_list()}
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
        assert df.index.to_list() == ref_df.index.to_list()

        for cname in df.column_names:
            col, ref_col = getattr(df, cname), getattr(ref_df, cname)
            assert isinstance(col, ak.Series)
            assert col.to_list() == ref_col.to_list()
            assert isinstance(df[cname], (ak.pdarray, ak.Strings, ak.Categorical))
            assert df[cname].to_list() == ref_df[cname].to_list()

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
        assert len(akdf.column_names) == len(akdf.dtypes)
        # dtypes returns objType for categorical, segarray. We should probably fix
        # this and add a df.objTypes property. pdarrays return actual dtype
        for ref_type, c in zip(
            ["int64", "int64", "int64", "str", "Categorical", "SegArray", "bigint"], akdf.column_names
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

        # Test dropping column_names
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
        assert slice_df.index.to_list() == [1, 3, 5]

        df_reset = slice_df.reset_index()
        assert df_reset.index.to_list() == [0, 1, 2]
        assert slice_df.index.to_list(), [1, 3, 5]

        slice_df.reset_index(inplace=True)
        assert slice_df.index.to_list(), [0, 1, 2]

    def test_rename(self):
        df = self.build_ak_df()

        rename = {"userName": "name_col", "userID": "user_id"}

        # Test out of Place - column
        df_rename = df.rename(rename, axis=1)
        assert "user_id" in df_rename.column_names
        assert "name_col" in df_rename.column_names
        assert "userName" not in df_rename.column_names
        assert "userID" not in df_rename.column_names
        assert "userID" in df.column_names
        assert "userName" in df.column_names
        assert "user_id" not in df.column_names
        assert "name_col" not in df.column_names

        # Test in place - column
        df.rename(column=rename, inplace=True)
        assert "user_id" in df.column_names
        assert "name_col" in df.column_names
        assert "userName" not in df.column_names
        assert "userID" not in df.column_names

        # prep for index renaming
        rename_idx = {1: 17, 2: 93}
        conf = list(range(6))
        conf[1] = 17
        conf[2] = 93

        # Test out of Place - index
        df_rename = df.rename(rename_idx)
        assert df_rename.index.values.to_list() == conf
        assert df.index.values.to_list() == list(range(6))

        # Test in place - index
        df.rename(index=rename_idx, inplace=True)
        assert df.index.values.to_list() == conf

    def test_append(self):
        df = self.build_ak_df()

        df.append(self.build_ak_append())

        ref_df = self.build_pd_df_append()

        # dataframe equality returns series with bool result for each row.
        assert_frame_equal(ref_df, df.to_pandas())

        idx = np.arange(8)
        assert idx.tolist() == df.index.index.to_list()

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
        gb = df.GroupBy("userName")
        keys, count = gb.count()
        assert keys.to_list() == ["Bob", "Alice", "Carol"]
        assert count.to_list() == [2, 3, 1]
        assert gb.permutation.to_list() == [1, 4, 0, 2, 5, 3]

        gb = df.GroupBy(["userName", "userID"])
        keys, count = gb.count()
        assert len(keys) == 2
        assert keys[0].to_list() == ["Bob", "Alice", "Carol"]
        assert keys[1].to_list() == [222, 111, 333]
        assert count.to_list() == [2, 3, 1]

        # testing counts with IPv4 column
        s = ak.DataFrame({"a": ak.IPv4(ak.arange(1, 5))}).groupby("a").count(as_series=True)
        pds = pd.Series(
            data=np.ones(4, dtype=np.int64),
            index=pd.Index(data=np.array(["0.0.0.1", "0.0.0.2", "0.0.0.3", "0.0.0.4"], dtype="<U7"),name="a"),
        )
        assert_series_equal(pds, s.to_pandas())

        # testing counts with Categorical column
        s = (
            ak.DataFrame({"a": ak.Categorical(ak.array(["a", "a", "a", "b"]))})
            .groupby("a")
            .count(as_series=True)
        )
        pds = pd.Series(data=np.array([3, 1]), index=pd.Index(data=np.array(["a", "b"], dtype="<U7"), name="a"))
        assert_series_equal(pds, s.to_pandas())

    def test_gb_series(self):
        df = self.build_ak_df()

        gb = df.GroupBy("userName", use_series=True)

        c = gb.count(as_series=True)
        assert isinstance(c, ak.Series)
        assert c.index.to_list() == ['Alice', 'Bob', 'Carol']
        assert c.values.to_list() == [3, 2, 1]

    @pytest.mark.parametrize("agg", ["sum", "first"])
    def test_gb_aggregations(self, agg):
        df = self.build_ak_df()
        pd_df = self.build_pd_df()
        # remove strings col because many aggregations don't support it
        cols_without_str = list(set(df.column_names) - {"userName"})
        df = df[cols_without_str]
        pd_df = pd_df[cols_without_str]

        group_on = "userID"
        for col in df.column_names:
            if col == group_on:
                # pandas groupby doesn't return the column used to group
                continue
            ak_ans = getattr(df.groupby(group_on), agg)()[col]
            pd_ans = getattr(pd_df.groupby(group_on), agg)()[col]
            assert ak_ans.to_list() == pd_ans.to_list()

        # pandas groupby doesn't return the column used to group
        cols_without_group_on = list(set(df.column_names) - {group_on})
        ak_ans = getattr(df.groupby(group_on), agg)()[cols_without_group_on]
        pd_ans = getattr(pd_df.groupby(group_on), agg)()[cols_without_group_on]
        assert_frame_equal(pd_ans, ak_ans.to_pandas(retain_index=True))

    def test_gb_aggregations_return_dataframe(self):
        ak_df = self.build_ak_df_example2()
        pd_df = ak_df.to_pandas(retain_index=True)

        pd_result1 = pd_df.groupby(["key1", "key2"], as_index=False).sum("count").drop(["nums"], axis=1)
        ak_result1 = ak_df.groupby(["key1", "key2"]).sum("count")
        assert_frame_equal(pd_result1, ak_result1.to_pandas(retain_index=True))
        assert isinstance(ak_result1, ak.dataframe.DataFrame)

        pd_result2 = (
            pd_df.groupby(["key1", "key2"], as_index=False).sum(["count"]).drop(["nums"], axis=1)
        )
        ak_result2 = ak_df.groupby(["key1", "key2"]).sum(["count"])
        assert_frame_equal(pd_result2, ak_result2.to_pandas(retain_index=True))
        assert isinstance(ak_result2, ak.dataframe.DataFrame)

        pd_result3 = pd_df.groupby(["key1", "key2"], as_index=False).sum(["count", "nums"])
        ak_result3 = ak_df.groupby(["key1", "key2"]).sum(["count", "nums"])
        assert_frame_equal(pd_result3, ak_result3.to_pandas(retain_index=True))
        assert isinstance(ak_result3, ak.dataframe.DataFrame)

        pd_result4 = pd_df.groupby(["key1", "key2"], as_index=False).sum().drop(["key3"], axis=1)
        ak_result4 = ak_df.groupby(["key1", "key2"]).sum()
        assert_frame_equal(pd_result4, ak_result4.to_pandas(retain_index=True))
        assert isinstance(ak_result4, ak.dataframe.DataFrame)

    def test_gb_aggregations_numeric_types(self):
        ak_df = self.build_ak_df_example_numeric_types()
        pd_df = ak_df.to_pandas(retain_index=True)

        assert_frame_equal(
            ak_df.groupby("gb_id").sum().to_pandas(retain_index=True), pd_df.groupby("gb_id").sum()
        )
        assert set(ak_df.groupby("gb_id").sum().column_names) == set(
            pd_df.groupby("gb_id").sum().columns.values
        )

        assert_frame_equal(
            ak_df.groupby(["gb_id"]).sum().to_pandas(retain_index=True), pd_df.groupby(["gb_id"]).sum()
        )
        assert set(ak_df.groupby(["gb_id"]).sum().column_names) == set(
            pd_df.groupby(["gb_id"]).sum().columns.values
        )

    def test_gb_count_single(self):
        ak_df = self.build_ak_df_example_numeric_types()
        pd_df = ak_df.to_pandas(retain_index=True)

        assert_frame_equal(
            ak_df.groupby("gb_id", as_index=False).count().to_pandas(retain_index=True),
            pd_df.groupby("gb_id", as_index=False)
            .count()
            .drop(["int64", "uint64", "bigint"], axis=1)
            .rename(columns={"float64": "count"}, errors="raise"),
        )

        assert_frame_equal(
            ak_df.groupby(["gb_id"], as_index=False).count().to_pandas(retain_index=True),
            pd_df.groupby(["gb_id"], as_index=False)
            .count()
            .drop(["int64", "uint64", "bigint"], axis=1)
            .rename(columns={"float64": "count"}, errors="raise"),
        )

    def test_gb_count_multiple(self):
        ak_df = self.build_ak_df_example2()
        pd_df = ak_df.to_pandas(retain_index=True)

        pd_result1 = (
            pd_df.groupby(["key1", "key2"], as_index=False).count().drop(["nums", "key3"], axis=1)
        )
        ak_result1 = ak_df.groupby(["key1", "key2"]).count(as_series=False)
        assert_frame_equal(pd_result1, ak_result1.to_pandas(retain_index=True))
        assert isinstance(ak_result1, ak.dataframe.DataFrame)

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
        assert isinstance(ak_result1, ak.dataframe.DataFrame)

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
            ak_df.groupby("key1").size(as_series=True).to_pandas(), pd_df.groupby("key1").size()
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

                    if isinstance(ak_result, ak.dataframe.DataFrame):
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
        assert p.to_list() == [0, 2, 5, 1, 4, 3]

        p = df.argsort(key="userName", ascending=False)
        assert p.to_list() == [3, 4, 1, 5, 2, 0]

    def test_coargsort(self):
        df = self.build_ak_df()

        p = df.coargsort(keys=["userID", "amount"])
        assert p.to_list() == [0, 5, 2, 1, 4, 3]

        p = df.coargsort(keys=["userID", "amount"], ascending=False)
        assert p.to_list() == [3, 4, 1, 2, 5, 0]

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

        pd_df = ak_df.to_pandas()

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
            if isinstance(ak_result, ak.dataframe.DataFrame):
                assert_frame_equal(
                    ak_result.sort_index().to_pandas(retain_index=True), pd_result.sort_index()
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
        assert rows.to_list() == [False, True, False, False, True, False]

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
        assert filtered.to_list() == [False, True, False, True, True, False]

    def test_copy(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        df = ak.DataFrame({"userName": username, "userID": userid})

        df_copy = df.copy(deep=True)
        assert_frame_equal(df.to_pandas(), df_copy.to_pandas())

        df_copy.__setitem__("userID", ak.array([1, 2, 1, 3, 2, 1]))
        assert df["userID"].to_list() != df_copy["userID"].to_list()

        df_copy = df.copy(deep=False)
        df_copy.__setitem__("userID", ak.array([1, 2, 1, 3, 2, 1]))
        assert_frame_equal(df.to_pandas(), df_copy.to_pandas())

    def test_isin(self):
        df = ak.DataFrame({"col_A": ak.array([7, 3]), "col_B": ak.array([1, 9])})

        # test against pdarray
        test_df = df.isin(ak.array([0, 1]))
        assert test_df["col_A"].to_list() == [False, False]
        assert test_df["col_B"].to_list() == [True, False]

        # Test against dict
        test_df = df.isin({"col_A": ak.array([0, 3])})
        assert test_df["col_A"].to_list() == [False, True]
        assert test_df["col_B"].to_list() == [False, False]

        # test against series
        i = ak.Index(ak.arange(2))
        s = ak.Series(data=ak.array([3, 9]), index=i.index)
        test_df = df.isin(s)
        assert test_df["col_A"].to_list() == [False, False]
        assert test_df["col_B"].to_list() == [False, True]

        # test against another dataframe
        other_df = ak.DataFrame({"col_A": ak.array([7, 3], dtype=ak.bigint), "col_C": ak.array([0, 9])})
        test_df = df.isin(other_df)
        assert test_df["col_A"].to_list() == [True, True]
        assert test_df["col_B"].to_list() == [False, False]

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
        assert bool_idx.index.index.to_list() == list(range(4, 65))

        slice_idx = df[:]
        slice_idx.__repr__()
        assert slice_idx.index.index.to_list() == list(range(65))

        # verify it persists non-int Index
        idx = ak.concatenate([ak.zeros(5, bool), ak.ones(60, bool)])
        df = ak.DataFrame({"cnt": ak.arange(65)}, index=idx)

        bool_idx = df[df["cnt"] > 3]
        bool_idx.__repr__()
        # the new index is first False and rest True (because we lose first 4),
        # so equivalent to arange(61, bool)
        assert bool_idx.index.index.to_list() == ak.arange(61, dtype=bool).to_list()

        slice_idx = df[:]
        slice_idx.__repr__()
        assert slice_idx.index.index.to_list() == idx.to_list()

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
        assert ["a", "b"] == df2.column_names
        assert df.index.to_list() == df2.index.to_list()
        assert df["a"].to_list() == df2["a"].to_list()
        assert df["b"].to_list() == df2["b"].to_list()

    def test_multi_col_merge(self):
        size = 1000
        seed = 1
        a = ak.randint(-size // 10, size // 10, size, seed=seed)
        b = ak.randint(-size // 10, size // 10, size, seed=seed + 1)
        c = ak.randint(-size // 10, size // 10, size, seed=seed + 2)
        d = ak.randint(-size // 10, size // 10, size, seed=seed + 3)
        ones = ak.ones(size, int)
        altr = ak.cast(ak.arange(size) % 2 == 0, int)
        for truth in itertools.product([True, False], repeat=3):
            left_arrs = [pda if t else pda_to_str_helper(pda) for pda, t in zip([a, b, ones], truth)]
            right_arrs = [pda if t else pda_to_str_helper(pda) for pda, t in zip([c, d, altr], truth)]
            left_df = ak.DataFrame({k: v for k, v in zip(["first", "second", "third"], left_arrs)})
            right_df = ak.DataFrame({k: v for k, v in zip(["first", "second", "third"], right_arrs)})
            l_pd, r_pd = left_df.to_pandas(), right_df.to_pandas()

            for how in "inner", "left", "right":
                for on in "first", "second", "third", ["first", "third"], ["second", "third"], None:
                    ak_merge = ak.merge(left_df, right_df, on=on, how=how)
                    pd_merge = pd.merge(l_pd, r_pd, on=on, how=how)

                    sorted_column_names = sorted(ak_merge.column_names)
                    assert sorted_column_names == sorted(pd_merge.columns.to_list())
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


def pda_to_str_helper(pda):
    return ak.array([f"str {i}" for i in pda.to_list()])
