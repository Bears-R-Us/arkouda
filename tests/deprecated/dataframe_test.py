import glob
import itertools
import os
import random
import string
import tempfile

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from base_test import ArkoudaTest
from context import arkouda as ak
from pandas.testing import assert_frame_equal, assert_series_equal

from arkouda import io_util
from arkouda.index import Index
from arkouda.scipy import chisquare as akchisquare


def build_ak_df():
    username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
    userid = ak.array([111, 222, 111, 333, 222, 111])
    item = ak.array([0, 0, 1, 1, 2, 0])
    day = ak.array([5, 5, 6, 5, 6, 6])
    amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    bi = ak.arange(2**200, 2**200 + 6)
    return ak.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount, "bi": bi}
    )


def build_ak_df_example2():
    data = {
        "key1": ["valuew", "valuex", "valuew", "valuex"],
        "key2": ["valueA", "valueB", "valueA", "valueB"],
        "key3": ["value1", "value2", "value3", "value4"],
        "count": [34, 25, 11, 4],
        "nums": [1, 2, 5, 21],
    }
    index = Index(ak.array([2, 1, 4, 3]), name="index1")
    ak_df = ak.DataFrame({k: ak.array(v) for k, v in data.items()}, index=index)
    return ak_df


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


def build_ak_df_duplicates():
    username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
    userid = ak.array([111, 222, 111, 333, 222, 111])
    item = ak.array([0, 1, 0, 2, 1, 0])
    day = ak.array([5, 5, 5, 5, 5, 5])
    return ak.DataFrame({"userName": username, "userID": userid, "item": item, "day": day})


def build_ak_append():
    username = ak.array(["John", "Carol"])
    userid = ak.array([444, 333])
    item = ak.array([0, 2])
    day = ak.array([1, 2])
    amount = ak.array([0.5, 5.1])
    bi = ak.array([2**200 + 6, 2**200 + 7])
    return ak.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount, "bi": bi}
    )


def build_ak_keyerror():
    userid = ak.array([444, 333])
    item = ak.array([0, 2])
    return ak.DataFrame({"user_id": userid, "item": item})


def build_ak_typeerror():
    username = ak.array([111, 222, 111, 333, 222, 111])
    userid = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
    item = ak.array([0, 0, 1, 1, 2, 0])
    day = ak.array([5, 5, 6, 5, 6, 6])
    amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    bi = ak.arange(2**200, 2**200 + 6)
    return ak.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount, "bi": bi}
    )


def build_pd_df():
    username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]
    userid = [111, 222, 111, 333, 222, 111]
    item = [0, 0, 1, 1, 2, 0]
    day = [5, 5, 6, 5, 6, 6]
    amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6]
    bi = [2**200, 2**200 + 1, 2**200 + 2, 2**200 + 3, 2**200 + 4, 2**200 + 5]
    return pd.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount, "bi": bi}
    )


def build_pd_df_duplicates():
    username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]
    userid = [111, 222, 111, 333, 222, 111]
    item = [0, 1, 0, 2, 1, 0]
    day = [5, 5, 5, 5, 5, 5]
    return pd.DataFrame({"userName": username, "userID": userid, "item": item, "day": day})


def build_pd_df_append():
    username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice", "John", "Carol"]
    userid = [111, 222, 111, 333, 222, 111, 444, 333]
    item = [0, 0, 1, 1, 2, 0, 0, 2]
    day = [5, 5, 6, 5, 6, 6, 1, 2]
    amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6, 0.5, 5.1]
    bi = [
        2**200,
        2**200 + 1,
        2**200 + 2,
        2**200 + 3,
        2**200 + 4,
        2**200 + 5,
        2**200 + 6,
        2**200 + 7,
    ]
    return pd.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount, "bi": bi}
    )


class DataFrameTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(DataFrameTest, cls).setUpClass()
        DataFrameTest.df_test_base_tmp = "{}/df_test".format(os.getcwd())
        io_util.get_directory(DataFrameTest.df_test_base_tmp)

    def test_dataframe_creation(self):
        # Validate empty DataFrame
        df = ak.DataFrame()
        self.assertIsInstance(df, ak.DataFrame)
        self.assertTrue(df.empty)

        df = build_ak_df()
        ref_df = build_pd_df()
        self.assertIsInstance(df, ak.DataFrame)
        self.assertEqual(len(df), 6)
        self.assertTrue(ref_df.equals(df.to_pandas()))

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
        shape = f"({df._shape_str()})".replace("(", "[").replace(")", "]")
        pd.set_option("display.max_rows", 4)
        s = df.__repr__().replace(f" ({df._shape_str()})", f"\n\n{shape}")
        self.assertEqual(s, pddf.__repr__())

        pd.set_option("display.max_rows", 10)
        pdf = pd.DataFrame({"a": list(range(1000)), "b": list(range(1000))})
        pdf["a"] = pdf["a"].apply(lambda x: "AA" + str(x))
        pdf["b"] = pdf["b"].apply(lambda x: "BB" + str(x))
        df = ak.DataFrame(pdf)
        shape = f"({df._shape_str()})".replace("(", "[").replace(")", "]")
        s = df.__repr__().replace(f" ({df._shape_str()})", f"\n\n{shape}")
        self.assertEqual(s, pdf.__repr__())

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
            self.assertTrue(isinstance(df, ak.DataFrame))
            self.assertTrue(isinstance(df["0"], ak.pdarray))
            self.assertEqual(df["0"].dtype, int)
            self.assertTrue(isinstance(df["1"], ak.pdarray))
            self.assertEqual(df["1"].dtype, bool)
            self.assertTrue(isinstance(df["2"], ak.Strings))
            self.assertEqual(df["2"].dtype, str)
            self.assertTrue(isinstance(df["3"], ak.pdarray))
            self.assertEqual(df["3"].dtype, float)

    def test_column_init(self):
        unlabeled_data = [[1, 2], [True, False], ["foo", "bar"], [2.3, -1.8]]
        good_labels = ["one1", "two2", "three3", "four4"]
        bad_labels1 = ["one", "two"]
        bad_labels2 = good_labels + ["five"]

        df = ak.DataFrame(unlabeled_data, columns=good_labels)
        self.assertListEqual(df.columns.values, good_labels)
        self.assertEqual(df["one1"][0], 1)
        self.assertEqual(df["three3"][0], "foo")
        self.assertEqual(df["four4"][1], -1.8)

        with self.assertRaises(ValueError):
            df = ak.DataFrame(unlabeled_data, columns=bad_labels1)
        with self.assertRaises(ValueError):
            df = ak.DataFrame(unlabeled_data, columns=bad_labels2)
        with self.assertRaises(TypeError):
            df = ak.DataFrame(unlabeled_data, columns=["one", "two", 3, "four"])

    def test_boolean_indexing(self):
        df = build_ak_df()
        ref_df = build_pd_df()
        row = df[df["userName"] == "Carol"]

        self.assertEqual(len(row), 1)
        self.assertTrue(ref_df[ref_df["userName"] == "Carol"].equals(row.to_pandas(retain_index=True)))

    def test_column_indexing(self):
        df = build_ak_df()
        self.assertTrue(isinstance(df.userName, ak.Series))
        self.assertTrue(isinstance(df.userID, ak.Series))
        self.assertTrue(isinstance(df.item, ak.Series))
        self.assertTrue(isinstance(df.day, ak.Series))
        self.assertTrue(isinstance(df.amount, ak.Series))
        self.assertTrue(isinstance(df.bi, ak.Series))
        for col in ("userName", "userID", "item", "day", "amount", "bi"):
            self.assertTrue(isinstance(df[col], (ak.pdarray, ak.Strings, ak.Categorical)))
        self.assertTrue(isinstance(df[["userName", "amount", "bi"]], ak.DataFrame))
        self.assertTrue(isinstance(df[("userID", "item", "day", "bi")], ak.DataFrame))
        self.assertTrue(isinstance(df.index, ak.Index))

    def test_dtype_prop(self):
        str_arr = ak.array(
            ["".join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(3)]
        )
        df_dict = {
            "i": ak.arange(3),
            "c_1": ak.arange(3, 6, 1),
            "c_2": ak.arange(6, 9, 1),
            "c_3": str_arr,
            "c_4": ak.Categorical(str_arr),
            "c_5": ak.SegArray(ak.array([0, 9, 14]), ak.arange(20)),
            "c_6": ak.arange(2**200, 2**200 + 3),
        }
        akdf = ak.DataFrame(df_dict)
        self.assertEqual(len(akdf.columns), len(akdf.dtypes))

    def test_from_pandas(self):
        username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice", "John", "Carol"]
        userid = [111, 222, 111, 333, 222, 111, 444, 333]
        item = [0, 0, 1, 1, 2, 0, 0, 2]
        day = [5, 5, 6, 5, 6, 6, 1, 2]
        amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6, 0.5, 5.1]
        bi = 2**200
        bi_arr = [bi, bi + 1, bi + 2, bi + 3, bi + 4, bi + 5, bi + 6, bi + 7]
        ref_df = pd.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi_arr,
            }
        )

        df = ak.DataFrame(ref_df)

        self.assertTrue(((ref_df == df.to_pandas()).all()).all())

        df = ak.DataFrame.from_pandas(ref_df)
        self.assertTrue(((ref_df == df.to_pandas()).all()).all())

    def test_drop(self):
        # create an arkouda df.
        df = build_ak_df()
        # create pandas df to validate functionality against
        pd_df = build_pd_df()

        # test out of place drop
        df_drop = df.drop([0, 1, 2])
        pddf_drop = pd_df.drop(labels=[0, 1, 2])
        pddf_drop.reset_index(drop=True, inplace=True)
        self.assertTrue(pddf_drop.equals(df_drop.to_pandas()))

        df_drop = df.drop("userName", axis=1)
        pddf_drop = pd_df.drop(labels=["userName"], axis=1)
        self.assertTrue(pddf_drop.equals(df_drop.to_pandas()))

        # Test dropping columns
        df.drop("userName", axis=1, inplace=True)
        pd_df.drop(labels=["userName"], axis=1, inplace=True)

        self.assertTrue(((df.to_pandas() == pd_df).all()).all())

        # Test dropping rows
        df.drop([0, 2, 5], inplace=True)
        # pandas retains original indexes when dropping rows, need to reset to line up with arkouda
        pd_df.drop(labels=[0, 2, 5], inplace=True)
        pd_df.reset_index(drop=True, inplace=True)

        self.assertTrue(pd_df.equals(df.to_pandas()))

        # verify that index keys must be ints
        with self.assertRaises(TypeError):
            df.drop("index")

        # verify axis can only be 0 or 1
        with self.assertRaises(ValueError):
            df.drop("amount", 15)

    def test_drop_duplicates(self):
        df = build_ak_df_duplicates()
        ref_df = build_pd_df_duplicates()

        dedup = df.drop_duplicates()
        dedup_pd = ref_df.drop_duplicates()
        # pandas retains original indexes when dropping dups, need to reset to line up with arkouda
        dedup_pd.reset_index(drop=True, inplace=True)

        dedup_test = dedup.to_pandas().sort_values("userName").reset_index(drop=True)
        dedup_pd_test = dedup_pd.sort_values("userName").reset_index(drop=True)

        self.assertTrue(dedup_test.equals(dedup_pd_test))

    def test_shape(self):
        df = build_ak_df()

        row, col = df.shape
        self.assertEqual(row, 6)
        self.assertEqual(col, 6)

    def test_reset_index(self):
        df = build_ak_df()

        slice_df = df[ak.array([1, 3, 5])]
        self.assertListEqual(slice_df.index.to_list(), [1, 3, 5])

        df_reset = slice_df.reset_index()
        self.assertListEqual(df_reset.index.to_list(), [0, 1, 2])
        self.assertListEqual(slice_df.index.to_list(), [1, 3, 5])

        df_reset2 = slice_df.reset_index(size=3)
        self.assertListEqual(df_reset2.index.to_list(), [0, 1, 2])
        self.assertListEqual(slice_df.index.to_list(), [1, 3, 5])

        slice_df.reset_index(inplace=True)
        self.assertListEqual(slice_df.index.to_list(), [0, 1, 2])

    def test_rename(self):
        df = build_ak_df()

        rename = {"userName": "name_col", "userID": "user_id"}

        # Test out of Place - column
        df_rename = df.rename(rename, axis=1)
        self.assertIn("user_id", df_rename.columns)
        self.assertIn("name_col", df_rename.columns)
        self.assertNotIn("userName", df_rename.columns)
        self.assertNotIn("userID", df_rename.columns)
        self.assertIn("userID", df.columns)
        self.assertIn("userName", df.columns)
        self.assertNotIn("user_id", df.columns)
        self.assertNotIn("name_col", df.columns)

        # Test in place - column
        df.rename(column=rename, inplace=True)
        self.assertIn("user_id", df.columns)
        self.assertIn("name_col", df.columns)
        self.assertNotIn("userName", df.columns)
        self.assertNotIn("userID", df.columns)

        # prep for index renaming
        rename_idx = {1: 17, 2: 93}
        conf = list(range(6))
        conf[1] = 17
        conf[2] = 93

        # Test out of Place - index
        df_rename = df.rename(rename_idx)
        self.assertListEqual(df_rename.index.values.to_list(), conf)
        self.assertListEqual(df.index.values.to_list(), list(range(6)))

        # Test in place - index
        df.rename(index=rename_idx, inplace=True)
        self.assertListEqual(df.index.values.to_list(), conf)

    def test_append(self):
        df = build_ak_df()
        df_toappend = build_ak_append()

        df.append(df_toappend)

        ref_df = build_pd_df_append()

        # dataframe equality returns series with bool result for each row.
        self.assertTrue(ref_df.equals(df.to_pandas()))

        idx = np.arange(8)
        self.assertListEqual(idx.tolist(), df.index.index.to_list())

        df_keyerror = build_ak_keyerror()
        with self.assertRaises(KeyError):
            df.append(df_keyerror)

        df_typeerror = build_ak_typeerror()
        with self.assertRaises(TypeError):
            df.append(df_typeerror)

    def test_concat(self):
        df = build_ak_df()
        df_toappend = build_ak_append()

        glued = ak.DataFrame.concat([df, df_toappend])

        ref_df = build_pd_df_append()

        # dataframe equality returns series with bool result for each row.
        self.assertTrue(ref_df.equals(glued.to_pandas()))

        df_keyerror = build_ak_keyerror()
        with self.assertRaises(KeyError):
            ak.DataFrame.concat([df, df_keyerror])

        df_typeerror = build_ak_typeerror()
        with self.assertRaises(TypeError):
            ak.DataFrame.concat([df, df_typeerror])

    def test_head(self):
        df = build_ak_df()
        ref_df = build_pd_df()

        hdf = df.head(3)
        hdf_ref = ref_df.head(3).reset_index(drop=True)
        self.assertTrue(hdf_ref.equals(hdf.to_pandas()))

    def test_tail(self):
        df = build_ak_df()
        ref_df = build_pd_df()

        hdf = df.tail(2)
        hdf_ref = ref_df.tail(2).reset_index(drop=True)
        self.assertTrue(hdf_ref.equals(hdf.to_pandas()))

    def test_gb_series(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        bi = ak.arange(2**200, 2**200 + 6)
        df = ak.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
            }
        )

        gb = df.GroupBy("userName", use_series=True)

        c = gb.size(as_series=True)
        self.assertIsInstance(c, ak.Series)
        self.assertListEqual(c.index.to_list(), ["Alice", "Bob", "Carol"])
        self.assertListEqual(c.values.to_list(), [3, 2, 1])

    def test_gb_aggregations(self):
        df = build_ak_df()
        pd_df = build_pd_df()
        # remove strings col because many aggregations don't support it
        cols_without_str = list(set(df.columns) - {"userName"})
        df = df[cols_without_str]
        pd_df = pd_df[cols_without_str]

        group_on = "userID"
        for agg in ["sum", "first", "count"]:
            ak_result = getattr(df.groupby(group_on), agg)()
            pd_result = getattr(pd_df.groupby(group_on), agg)()
            assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    def test_gb_aggregations_example_numeric_types(self):
        df = build_ak_df_example_numeric_types()
        pd_df = df.to_pandas()

        aggs_to_test = [
            "count",
            "first",
            "sum",
        ]

        group_on = "gb_id"
        for agg in aggs_to_test:
            ak_result = getattr(df.groupby(group_on), agg)()
            pd_result = getattr(pd_df.groupby(group_on), agg)()
            assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    def test_gb_aggregations_with_nans(self):
        df = build_ak_df_with_nans()
        # @TODO handle bool columns correctly
        df.drop("bools", axis=1, inplace=True)
        pd_df = df.to_pandas()

        aggs_to_test = [
            "count",
            "max",
            "mean",
            "median",
            "min",
            "std",
            "sum",
            "var",
        ]

        group_on = ["key1", "key2"]
        for agg in aggs_to_test:
            ak_result = getattr(df.groupby(group_on), agg)()
            pd_result = getattr(pd_df.groupby(group_on, as_index=False), agg)()
            assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)

    def test_gb_aggregations_return_dataframe(self):
        ak_df = build_ak_df_example2()
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
        ak_df = build_ak_df_example_numeric_types()
        pd_df = ak_df.to_pandas(retain_index=True)

        assert_frame_equal(
            ak_df.groupby("gb_id").sum().to_pandas(retain_index=True), pd_df.groupby("gb_id").sum()
        )
        assert set(ak_df.groupby("gb_id").sum().columns) == set(pd_df.groupby("gb_id").sum().columns)

        assert_frame_equal(
            ak_df.groupby(["gb_id"]).sum().to_pandas(retain_index=True), pd_df.groupby(["gb_id"]).sum()
        )
        assert set(ak_df.groupby(["gb_id"]).sum().columns) == set(pd_df.groupby(["gb_id"]).sum().columns)

    def test_gb_size_single(self):
        ak_df = build_ak_df_example_numeric_types()
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
        ak_df = build_ak_df_example2()
        pd_df = ak_df.to_pandas(retain_index=True)

        pd_result1 = pd_df.groupby(["key1", "key2"], as_index=False).size()
        ak_result1 = ak_df.groupby(["key1", "key2"], as_index=False).size()
        assert_frame_equal(pd_result1, ak_result1.to_pandas(retain_index=True))
        assert isinstance(ak_result1, ak.dataframe.DataFrame)

        for as_index in [True, False]:
            for dropna in [True, False]:
                for gb_keys in ["key1", "key2", ["key1", "key2"], ["count", "key1", "key2"]]:
                    ak_result = ak_df.groupby(gb_keys, as_index=as_index, dropna=dropna).size()
                    pd_result = pd_df.groupby(gb_keys, as_index=as_index, dropna=dropna).size()

                    if isinstance(ak_result, ak.dataframe.DataFrame):
                        assert_frame_equal(ak_result.to_pandas(retain_index=True), pd_result)
                    else:
                        assert_series_equal(ak_result.to_pandas(), pd_result)

    def test_gb_size_match_pandas(self):
        ak_df = build_ak_df_with_nans()
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
        ak_df = build_ak_df_example2()
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

    def test_memory_usage(self):
        dtypes = [ak.int64, ak.float64, ak.bool_]
        data = dict([(str(t), ak.ones(5000, dtype=ak.int64).astype(t)) for t in dtypes])
        df = ak.DataFrame(data)
        ak_memory_usage = df.memory_usage()
        pd_memory_usage = pd.Series(
            [40000, 40000, 40000, 5000], index=["Index", "int64", "float64", "bool"]
        )
        assert_series_equal(ak_memory_usage.to_pandas(), pd_memory_usage)

        self.assertEqual(df.memory_usage_info(unit="B"), "125000.00 B")
        self.assertEqual(df.memory_usage_info(unit="KB"), "122.07 KB")
        self.assertEqual(df.memory_usage_info(unit="MB"), "0.12 MB")
        self.assertEqual(df.memory_usage_info(unit="GB"), "0.00 GB")

        ak_memory_usage = df.memory_usage(index=False)
        pd_memory_usage = pd.Series([40000, 40000, 5000], index=["int64", "float64", "bool"])
        assert_series_equal(ak_memory_usage.to_pandas(), pd_memory_usage)

        ak_memory_usage = df.memory_usage(unit="KB")
        pd_memory_usage = pd.Series(
            [39.0625, 39.0625, 39.0625, 4.88281], index=["Index", "int64", "float64", "bool"]
        )
        assert_series_equal(ak_memory_usage.to_pandas(), pd_memory_usage)

    def test_to_pandas(self):
        df = build_ak_df()
        pd_df = build_pd_df()

        self.assertTrue(pd_df.equals(df.to_pandas()))

        slice_df = df[ak.array([1, 3, 5])]
        pd_df = slice_df.to_pandas(retain_index=True)
        self.assertEqual(pd_df.index.tolist(), [1, 3, 5])

        pd_df = slice_df.to_pandas()
        self.assertEqual(pd_df.index.tolist(), [0, 1, 2])

    def test_argsort(self):
        df = build_ak_df()

        p = df.argsort(key="userName")
        self.assertListEqual(p.to_list(), [0, 2, 5, 1, 4, 3])

        p = df.argsort(key="userName", ascending=False)
        self.assertListEqual(p.to_list(), [3, 4, 1, 5, 2, 0])

    def test_coargsort(self):
        df = build_ak_df()

        p = df.coargsort(keys=["userID", "amount"])
        self.assertListEqual(p.to_list(), [0, 5, 2, 1, 4, 3])

        p = df.coargsort(keys=["userID", "amount"], ascending=False)
        self.assertListEqual(p.to_list(), [3, 4, 1, 2, 5, 0])

    def test_sort_values(self):
        userid = [111, 222, 111, 333, 222, 111]
        userid_ak = ak.array(userid)

        # sort userid to build dataframes to reference
        userid.sort()

        df = ak.DataFrame({"userID": userid_ak})
        ord = df.sort_values()
        self.assertTrue(ord.to_pandas().equals(pd.DataFrame(data=userid, columns=["userID"])))
        ord = df.sort_values(ascending=False)
        userid.reverse()
        self.assertTrue(ord.to_pandas().equals(pd.DataFrame(data=userid, columns=["userID"])))

        df = build_ak_df()
        ord = df.sort_values(by="userID")
        ref_df = build_pd_df()
        ref_df = ref_df.sort_values(by="userID").reset_index(drop=True)
        self.assertTrue(ref_df.equals(ord.to_pandas()))

        ord = df.sort_values(by=["userID", "day"])
        ref_df = ref_df.sort_values(by=["userID", "day"]).reset_index(drop=True)
        self.assertTrue(ref_df.equals(ord.to_pandas()))

        with self.assertRaises(TypeError):
            df.sort_values(by=1)

    def test_sort_index(self):
        ak_df = build_ak_df_example_numeric_types()
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
        self.assertListEqual(rows.to_list(), [False, True, False, False, True, False])

        df_3 = ak.DataFrame({"user_name": username, "user_number": userid})
        with self.assertRaises(ValueError):
            rows = ak.intx(df_1, df_3)

    def test_apply_perm(self):
        df = build_ak_df()
        ref_df = build_pd_df()

        ord = df.sort_values(by="userID")
        perm_list = [0, 3, 1, 5, 4, 2]
        default_perm = ak.array(perm_list)
        ord.apply_permutation(default_perm)

        ord_ref = ref_df.sort_values(by="userID").reset_index(drop=True)
        ord_ref = ord_ref.reindex(perm_list).reset_index(drop=True)
        self.assertTrue(ord_ref.equals(ord.to_pandas()))

    def test_filter_by_range(self):
        userid = ak.array([111, 222, 111, 333, 222, 111])
        amount = ak.array([0, 1, 1, 2, 3, 15])
        df = ak.DataFrame({"userID": userid, "amount": amount})

        filtered = df.filter_by_range(keys=["userID"], low=1, high=2)
        self.assertFalse(filtered[0])
        self.assertTrue(filtered[1])
        self.assertFalse(filtered[2])
        self.assertTrue(filtered[3])
        self.assertTrue(filtered[4])
        self.assertFalse(filtered[5])

    def test_copy(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        df = ak.DataFrame({"userName": username, "userID": userid})

        df_copy = df.copy(deep=True)
        self.assertEqual(df.__repr__(), df_copy.__repr__())

        df_copy.__setitem__("userID", ak.array([1, 2, 1, 3, 2, 1]))
        self.assertNotEqual(df.__repr__(), df_copy.__repr__())

        df_copy = df.copy(deep=False)
        df_copy.__setitem__("userID", ak.array([1, 2, 1, 3, 2, 1]))
        self.assertEqual(df.__repr__(), df_copy.__repr__())

    def test_save(self):
        i = list(range(3))
        c1 = [9, 7, 17]
        c2 = [2, 4, 6]
        df_dict = {"i": ak.array(i), "c_1": ak.array(c1), "c_2": ak.array(c2)}

        akdf = ak.DataFrame(df_dict)

        validation_df = pd.DataFrame(
            {
                "i": i,
                "c_1": c1,
                "c_2": c2,
            }
        )
        with tempfile.TemporaryDirectory(dir=DataFrameTest.df_test_base_tmp) as tmp_dirname:
            akdf.to_parquet(f"{tmp_dirname}/testName")

            ak_loaded = ak.DataFrame.load(f"{tmp_dirname}/testName")
            self.assertTrue(validation_df.equals(ak_loaded[akdf.columns.values].to_pandas()))

            # test save with index true
            akdf.to_parquet(f"{tmp_dirname}/testName_with_index.pq", index=True)
            self.assertEqual(
                len(glob.glob(f"{tmp_dirname}/testName_with_index*.pq")), ak.get_config()["numLocales"]
            )

            # Test for df having seg array col
            df = ak.DataFrame({"a": ak.arange(10), "b": ak.SegArray(ak.arange(10), ak.arange(10))})
            df.to_hdf(f"{tmp_dirname}/seg_test.h5")
            self.assertEqual(
                len(glob.glob(f"{tmp_dirname}/seg_test*.h5")), ak.get_config()["numLocales"]
            )
            ak_loaded = ak.DataFrame.load(f"{tmp_dirname}/seg_test.h5")
            self.assertTrue(df.to_pandas().equals(ak_loaded.to_pandas()))

            # test with segarray with _ in column name
            df_dict = {
                "c_1": ak.arange(3, 6),
                "c_2": ak.arange(6, 9),
                "c_3": ak.SegArray(ak.array([0, 9, 14]), ak.arange(20)),
            }
            akdf = ak.DataFrame(df_dict)
            akdf.to_hdf(f"{tmp_dirname}/seg_test.h5")
            self.assertEqual(
                len(glob.glob(f"{tmp_dirname}/seg_test*.h5")), ak.get_config()["numLocales"]
            )
            ak_loaded = ak.DataFrame.load(f"{tmp_dirname}/seg_test.h5")
            self.assertTrue(akdf.to_pandas().equals(ak_loaded.to_pandas()))

            # test load_all and read workflows
            ak_load_all = ak.DataFrame(ak.load_all(f"{tmp_dirname}/seg_test.h5"))
            self.assertTrue(akdf.to_pandas().equals(ak_load_all.to_pandas()))

            ak_read = ak.DataFrame(ak.read(f"{tmp_dirname}/seg_test*"))
            self.assertTrue(akdf.to_pandas().equals(ak_read.to_pandas()))

    def test_isin(self):
        df = ak.DataFrame({"col_A": ak.array([7, 3]), "col_B": ak.array([1, 9])})

        # test against pdarray
        test_df = df.isin(ak.array([0, 1]))
        self.assertListEqual(test_df["col_A"].to_list(), [False, False])
        self.assertListEqual(test_df["col_B"].to_list(), [True, False])

        # Test against dict
        test_df = df.isin({"col_A": ak.array([0, 3])})
        self.assertListEqual(test_df["col_A"].to_list(), [False, True])
        self.assertListEqual(test_df["col_B"].to_list(), [False, False])

        # test against series
        i = ak.Index(ak.arange(2))
        s = ak.Series(data=ak.array([3, 9]), index=i.index)
        test_df = df.isin(s)
        self.assertListEqual(test_df["col_A"].to_list(), [False, False])
        self.assertListEqual(test_df["col_B"].to_list(), [False, True])

        # test against another dataframe
        other_df = ak.DataFrame({"col_A": ak.array([7, 3], dtype=ak.bigint), "col_C": ak.array([0, 9])})
        test_df = df.isin(other_df)
        self.assertListEqual(test_df["col_A"].to_list(), [True, True])
        self.assertListEqual(test_df["col_B"].to_list(), [False, False])

    def test_count(self):
        akdf = build_ak_df_with_nans()
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
        self.assertEqual(df["Test"].dtype, ak.uint64)

    def test_head_tail_datetime_display(self):
        # Reproducer for issue #2596
        values = ak.array([1689221916000000] * 100, dtype=ak.int64)
        dt = ak.Datetime(values, unit="u")
        df = ak.DataFrame({"Datetime from Microseconds": dt})
        # verify _get_head_tail and _get_head_tail_server match
        self.assertEqual(df._get_head_tail_server().__repr__(), df._get_head_tail().__repr__())

    def test_head_tail_resetting_index(self):
        # Test that issue #2183 is resolved
        df = ak.DataFrame({"cnt": ak.arange(65)})
        # Note we have to call __repr__ to trigger head_tail_server call

        bool_idx = df[df["cnt"] > 3]
        bool_idx.__repr__()
        self.assertListEqual(bool_idx.index.index.to_list(), list(range(4, 65)))

        slice_idx = df[:]
        slice_idx.__repr__()
        self.assertListEqual(slice_idx.index.index.to_list(), list(range(65)))

        # verify it persists non-int Index
        idx = ak.concatenate([ak.zeros(5, bool), ak.ones(60, bool)])
        df = ak.DataFrame({"cnt": ak.arange(65)}, index=idx)

        bool_idx = df[df["cnt"] > 3]
        bool_idx.__repr__()
        # the new index is first False and rest True (because we lose first 4),
        # so equivalent to arange(61, bool)
        self.assertListEqual(bool_idx.index.index.to_list(), ak.arange(61, dtype=bool).to_list())

        slice_idx = df[:]
        slice_idx.__repr__()
        self.assertListEqual(slice_idx.index.index.to_list(), idx.to_list())

    def test_ipv4_columns(self):
        # test with single IPv4 column
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=DataFrameTest.df_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname + "/ipv4_df"
            df.to_parquet(fname)

            data = ak.read(fname + "*")
            rddf = ak.DataFrame({"a": data["a"], "b": ak.IPv4(data["b"])})

            self.assertListEqual(df["a"].to_list(), rddf["a"].to_list())
            self.assertListEqual(df["b"].to_list(), rddf["b"].to_list())

        # test with multiple
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10)), "b": ak.IPv4(ak.arange(10))})
        with tempfile.TemporaryDirectory(dir=DataFrameTest.df_test_base_tmp) as tmp_dirname:
            fname = tmp_dirname + "/ipv4_df"
            df.to_parquet(fname)

            data = ak.read(fname + "*")
            rddf = ak.DataFrame({"a": ak.IPv4(data["a"]), "b": ak.IPv4(data["b"])})

            self.assertListEqual(df["a"].to_list(), rddf["a"].to_list())
            self.assertListEqual(df["b"].to_list(), rddf["b"].to_list())

        # test replacement of IPv4 with uint representation
        df = ak.DataFrame({"a": ak.IPv4(ak.arange(10))})
        df["a"] = df["a"].export_uint()
        self.assertListEqual(ak.arange(10).to_list(), df["a"].to_list())

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
        self.assertListEqual(["a", "b"], df2.columns.values)
        self.assertListEqual(df.index.to_list(), df2.index.to_list())
        self.assertListEqual(df["a"].to_list(), df2["a"].to_list())
        self.assertListEqual(df["b"].to_list(), df2["b"].to_list())

    def test_merge(self):
        df1 = ak.DataFrame(
            {
                "key": ak.arange(4),
                "value1": ak.array(["A", "B", "C", "D"]),
                "value3": ak.arange(4, dtype=ak.int64),
            }
        )

        df2 = ak.DataFrame(
            {
                "key": ak.arange(2, 6, 1),
                "value1": ak.array(["A", "B", "D", "F"]),
                "value2": ak.array(["apple", "banana", "cherry", "date"]),
                "value3": ak.ones(4, dtype=ak.int64),
            }
        )

        ij_expected_df = ak.DataFrame(
            {
                "key": ak.array([2, 3]),
                "value1_x": ak.array(["C", "D"]),
                "value3_x": ak.array([2, 3]),
                "value1_y": ak.array(["A", "B"]),
                "value2": ak.array(["apple", "banana"]),
                "value3_y": ak.array([1, 1]),
            }
        )

        ij_merged_df = ak.merge(df1, df2, how="inner", on="key")

        self.assertListEqual(ij_expected_df.columns.values, ij_merged_df.columns.values)
        self.assertListEqual(ij_expected_df["key"].to_list(), ij_merged_df["key"].to_list())
        self.assertListEqual(ij_expected_df["value1_x"].to_list(), ij_merged_df["value1_x"].to_list())
        self.assertListEqual(ij_expected_df["value1_y"].to_list(), ij_merged_df["value1_y"].to_list())
        self.assertListEqual(ij_expected_df["value2"].to_list(), ij_merged_df["value2"].to_list())
        self.assertTrue(
            np.allclose(
                ij_expected_df["value3_x"].to_ndarray(),
                ij_merged_df["value3_x"].to_ndarray(),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                ij_expected_df["value3_y"].to_ndarray(),
                ij_merged_df["value3_y"].to_ndarray(),
                equal_nan=True,
            )
        )

        rj_expected_df = ak.DataFrame(
            {
                "key": ak.array([2, 3, 4, 5]),
                "value1_x": ak.array(["C", "D", "nan", "nan"]),
                "value3_x": ak.array([2.0, 3.0, np.nan, np.nan]),
                "value1_y": ak.array(["A", "B", "D", "F"]),
                "value2": ak.array(["apple", "banana", "cherry", "date"]),
                "value3_y": ak.array([1, 1, 1, 1]),
            }
        )

        rj_merged_df = ak.merge(df1, df2, how="right", on="key")

        self.assertTrue(
            rj_merged_df.dtypes
            == {
                "key": "int64",
                "value1_x": "str",
                "value3_x": "float64",
                "value1_y": "str",
                "value2": "str",
                "value3_y": "int64",
            }
        )

        self.assertListEqual(rj_expected_df.columns.values, rj_merged_df.columns.values)
        self.assertListEqual(rj_expected_df["key"].to_list(), rj_merged_df["key"].to_list())
        self.assertListEqual(rj_expected_df["value1_x"].to_list(), rj_merged_df["value1_x"].to_list())
        self.assertListEqual(rj_expected_df["value1_y"].to_list(), rj_merged_df["value1_y"].to_list())
        self.assertListEqual(rj_expected_df["value2"].to_list(), rj_merged_df["value2"].to_list())
        self.assertTrue(
            np.allclose(
                rj_expected_df["value3_x"].to_ndarray(),
                rj_merged_df["value3_x"].to_ndarray(),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                rj_expected_df["value3_y"].to_ndarray(),
                rj_merged_df["value3_y"].to_ndarray(),
                equal_nan=True,
            )
        )

        rj_merged_df2 = ak.merge(df1, df2, how="right", on="key", convert_ints=False)

        self.assertTrue(
            rj_merged_df2.dtypes
            == {
                "key": "int64",
                "value1_x": "str",
                "value3_x": "int64",
                "value1_y": "str",
                "value2": "str",
                "value3_y": "int64",
            }
        )

        lj_expected_df = ak.DataFrame(
            {
                "key": ak.array(
                    [
                        0,
                        1,
                        2,
                        3,
                    ]
                ),
                "value1_y": ak.array(
                    [
                        "nan",
                        "nan",
                        "A",
                        "B",
                    ]
                ),
                "value2": ak.array(
                    [
                        "nan",
                        "nan",
                        "apple",
                        "banana",
                    ]
                ),
                "value3_y": ak.array(
                    [
                        np.nan,
                        np.nan,
                        1.0,
                        1.0,
                    ]
                ),
                "value1_x": ak.array(
                    [
                        "A",
                        "B",
                        "C",
                        "D",
                    ]
                ),
                "value3_x": ak.array(
                    [
                        0,
                        1,
                        2,
                        3,
                    ]
                ),
            }
        )

        lj_merged_df = ak.merge(df1, df2, how="left", on="key")

        self.assertTrue(
            lj_merged_df.dtypes
            == {
                "key": "int64",
                "value1_y": "str",
                "value2": "str",
                "value3_y": "float64",
                "value1_x": "str",
                "value3_x": "int64",
            }
        )

        self.assertListEqual(lj_expected_df.columns.values, lj_merged_df.columns.values)
        self.assertListEqual(lj_expected_df["key"].to_list(), lj_merged_df["key"].to_list())
        self.assertListEqual(lj_expected_df["value1_x"].to_list(), lj_merged_df["value1_x"].to_list())
        self.assertListEqual(lj_expected_df["value1_y"].to_list(), lj_merged_df["value1_y"].to_list())
        self.assertListEqual(lj_expected_df["value2"].to_list(), lj_merged_df["value2"].to_list())
        self.assertTrue(
            np.allclose(
                lj_expected_df["value3_x"].to_ndarray(),
                lj_merged_df["value3_x"].to_ndarray(),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                lj_expected_df["value3_y"].to_ndarray(),
                lj_merged_df["value3_y"].to_ndarray(),
                equal_nan=True,
            )
        )

        lj_merged_df2 = ak.merge(df1, df2, how="left", on="key", convert_ints=False)

        self.assertTrue(
            lj_merged_df2.dtypes
            == {
                "key": "int64",
                "value1_y": "str",
                "value2": "str",
                "value3_y": "int64",
                "value1_x": "str",
                "value3_x": "int64",
            }
        )

        oj_expected_df = ak.DataFrame(
            {
                "key": ak.array([0, 1, 2, 3, 4, 5]),
                "value1_y": ak.array(["nan", "nan", "A", "B", "D", "F"]),
                "value2": ak.array(["nan", "nan", "apple", "banana", "cherry", "date"]),
                "value3_y": ak.array([np.nan, np.nan, 1.0, 1.0, 1.0, 1.0]),
                "value1_x": ak.array(
                    [
                        "A",
                        "B",
                        "C",
                        "D",
                        "nan",
                        "nan",
                    ]
                ),
                "value3_x": ak.array([0.0, 1.0, 2.0, 3.0, np.nan, np.nan]),
            }
        )

        oj_merged_df = ak.merge(df1, df2, how="outer", on="key")

        self.assertTrue(
            oj_merged_df.dtypes
            == {
                "key": "int64",
                "value1_y": "str",
                "value2": "str",
                "value3_y": "float64",
                "value1_x": "str",
                "value3_x": "float64",
            }
        )

        self.assertListEqual(oj_expected_df.columns.values, oj_merged_df.columns.values)
        self.assertListEqual(oj_expected_df["key"].to_list(), oj_merged_df["key"].to_list())
        self.assertListEqual(oj_expected_df["value1_x"].to_list(), oj_merged_df["value1_x"].to_list())
        self.assertListEqual(oj_expected_df["value1_y"].to_list(), oj_merged_df["value1_y"].to_list())
        self.assertListEqual(oj_expected_df["value2"].to_list(), oj_merged_df["value2"].to_list())
        self.assertTrue(
            np.allclose(
                oj_expected_df["value3_x"].to_ndarray(),
                oj_merged_df["value3_x"].to_ndarray(),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                oj_expected_df["value3_y"].to_ndarray(),
                oj_merged_df["value3_y"].to_ndarray(),
                equal_nan=True,
            )
        )

        oj_merged_df2 = ak.merge(df1, df2, how="outer", on="key", convert_ints=False)

        self.assertTrue(
            oj_merged_df2.dtypes
            == {
                "key": "int64",
                "value1_y": "str",
                "value2": "str",
                "value3_y": "int64",
                "value1_x": "str",
                "value3_x": "int64",
            }
        )

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
                    self.assertTrue(df.any(axis=axis).to_pandas().empty)
                else:
                    assert_series_equal(
                        df.any(axis=axis).to_pandas(), df.to_pandas().any(axis=axis, bool_only=True)
                    )
                if df.to_pandas().all(axis=axis, bool_only=True).empty:
                    self.assertTrue(df.all(axis=axis).to_pandas().empty)
                else:
                    assert_series_equal(
                        df.all(axis=axis).to_pandas(), df.to_pandas().all(axis=axis, bool_only=True)
                    )
            # Test is axis=None
            self.assertEqual(df.any(axis=None), df.to_pandas().any(axis=None, bool_only=True))
            self.assertEqual(df.all(axis=None), df.to_pandas().all(axis=None, bool_only=True))

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
                        self.assertTrue(
                            df.dropna(axis=axis, thresh=thresh).to_pandas(retain_index=True).empty
                        )
                    else:
                        assert_frame_equal(
                            df.dropna(axis=axis, thresh=thresh).to_pandas(retain_index=True),
                            df.to_pandas(retain_index=True).dropna(axis=axis, thresh=thresh),
                        )

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

            for how in "inner", "left", "right", "outer":
                for on in ["first", "third"], ["second", "third"], None:
                    ak_merge = ak.merge(left_df, right_df, on=on, how=how)
                    pd_merge = pd.merge(l_pd, r_pd, on=on, how=how)

                    sorted_columns = sorted(ak_merge.columns)
                    self.assertListEqual(sorted_columns, sorted(pd_merge.columns.to_list()))
                    for col in sorted_columns:
                        from_ak = ak_merge[col].to_ndarray()
                        from_pd = pd_merge[col].to_numpy()
                        if isinstance(ak_merge[col], ak.pdarray):
                            self.assertTrue(
                                np.allclose(np.sort(from_ak), np.sort(from_pd), equal_nan=True)
                            )
                        else:
                            # we have to cast to str because pandas arrays converted to numpy
                            # have dtype object and have float NANs in line with the str values
                            self.assertTrue((np.sort(from_ak) == np.sort(from_pd.astype(str))).all())

                    # TODO arkouda seems to be sometimes convert columns to floats on a right merge
                    #  when pandas doesnt. Eventually we want to test frame_equal,
                    #  not just value equality
                    # from pandas.testing import assert_frame_equal
                    # sorted_ak = ak_merge.sort_values(sorted_columns).reset_index()
                    # sorted_pd = pd_merge.sort_values(sorted_columns).reset_index(drop=True)
                    # assert_frame_equal(sorted_ak.to_pandas()[sorted_columns],
                    # sorted_pd[sorted_columns])

    def test_to_markdown(self):
        df = ak.DataFrame({"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]})
        self.assertEqual(
            df.to_markdown(),
            "+----+------------+------------+\n"
            "|    | animal_1   | animal_2   |\n"
            "+====+============+============+\n"
            "|  0 | elk        | dog        |\n"
            "+----+------------+------------+\n"
            "|  1 | pig        | quetzal    |\n"
            "+----+------------+------------+",
        )

        self.assertEqual(
            df.to_markdown(index=False),
            "+------------+------------+\n"
            "| animal_1   | animal_2   |\n"
            "+============+============+\n"
            "| elk        | dog        |\n"
            "+------------+------------+\n"
            "| pig        | quetzal    |\n"
            "+------------+------------+",
        )

        self.assertEqual(df.to_markdown(tablefmt="grid"), df.to_pandas().to_markdown(tablefmt="grid"))
        self.assertEqual(
            df.to_markdown(tablefmt="grid", index=False),
            df.to_pandas().to_markdown(tablefmt="grid", index=False),
        )
        self.assertEqual(df.to_markdown(tablefmt="jira"), df.to_pandas().to_markdown(tablefmt="jira"))

    def test_sample_hypothesis_testing(self):
        # perform a weighted sample and use chisquare to test
        # if the observed frequency matches the expected frequency

        # I tested this many times without a set seed, but with no seed
        # it's expected to fail one out of every ~20 runs given a pval limit of 0.05
        rng = ak.random.default_rng(43)
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
        self.assertTrue(pval > 0.05)

    def test_sample_flags(self):
        # use numpy to randomly generate a set seed
        seed = np.random.default_rng().choice(2**63)
        cfg = ak.get_config()

        rng = ak.random.default_rng(seed)
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
        rng = ak.random.default_rng(seed)
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
                            g.sample(frac=(size / 4), replace=replace, weights=p, random_state=rng)
                        )

        # reset generator to ensure we get the same arrays
        rng = ak.random.default_rng(seed)
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
                            frac=(size / 4), replace=replace, weights=p, random_state=rng
                        )

                        res = (
                            np.allclose(previous1["vals"].to_list(), current1["vals"].to_list())
                        ) and (np.allclose(previous2["vals"].to_list(), current2["vals"].to_list()))
                        if not res:
                            print(f"\nnum locales: {cfg['numLocales']}")
                            print(f"Failure with seed:\n{seed}")
                        self.assertTrue(res)


def pda_to_str_helper(pda):
    return ak.array([f"str {i}" for i in pda.to_list()])
