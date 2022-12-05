import glob
import os
import random
import string
import tempfile
from shutil import rmtree
from arkouda import io_util

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from base_test import ArkoudaTest
from context import arkouda as ak


def build_ak_df():
    username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
    userid = ak.array([111, 222, 111, 333, 222, 111])
    item = ak.array([0, 0, 1, 1, 2, 0])
    day = ak.array([5, 5, 6, 5, 6, 6])
    amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    return ak.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
    )


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
    return ak.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
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
    return ak.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
    )


def build_pd_df():
    username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"]
    userid = [111, 222, 111, 333, 222, 111]
    item = [0, 0, 1, 1, 2, 0]
    day = [5, 5, 6, 5, 6, 6]
    amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6]
    return pd.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
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
    return pd.DataFrame(
        {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
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
        for col in ("userName", "userID", "item", "day", "amount"):
            self.assertTrue(isinstance(df[col], (ak.pdarray, ak.Strings, ak.Categorical)))
        self.assertTrue(isinstance(df[["userName", "amount"]], ak.DataFrame))
        self.assertTrue(isinstance(df[("userID", "item", "day")], ak.DataFrame))
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
            "c_5": ak.segarray(ak.array([0, 9, 14]), ak.arange(20)),
        }
        akdf = ak.DataFrame(df_dict)
        self.assertEqual(len(akdf.columns), len(akdf.dtypes))

    def test_from_pandas(self):
        username = ["Alice", "Bob", "Alice", "Carol", "Bob", "Alice", "John", "Carol"]
        userid = [111, 222, 111, 333, 222, 111, 444, 333]
        item = [0, 0, 1, 1, 2, 0, 0, 2]
        day = [5, 5, 6, 5, 6, 6, 1, 2]
        amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6, 0.5, 5.1]
        ref_df = pd.DataFrame(
            {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
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
        self.assertEqual(col, 5)

    def test_reset_index(self):
        df = build_ak_df()

        slice_df = df[ak.array([1, 3, 5])]
        self.assertListEqual(slice_df.index.to_list(), [1, 3, 5])

        df_reset = slice_df.reset_index()
        self.assertListEqual(df_reset.index.to_list(), [0, 1, 2])
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

    def test_groupby_standard(self):
        df = build_ak_df()
        gb = df.GroupBy("userName")
        keys, count = gb.count()
        self.assertListEqual(keys.to_list(), ["Bob", "Alice", "Carol"])
        self.assertListEqual(count.to_list(), [2, 3, 1])
        self.assertListEqual(gb.permutation.to_list(), [1, 4, 0, 2, 5, 3])

        gb = df.GroupBy(["userName", "userID"])
        keys, count = gb.count()
        self.assertEqual(len(keys), 2)
        self.assertListEqual(keys[0].to_list(), ["Carol", "Bob", "Alice"])
        self.assertListEqual(keys[1].to_list(), [333, 222, 111])
        self.assertListEqual(count.to_list(), [1, 2, 3])

    def test_gb_series(self):
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame(
            {"userName": username, "userID": userid, "item": item, "day": day, "amount": amount}
        )

        gb = df.GroupBy("userName", use_series=True)

        c = gb.count()
        self.assertIsInstance(c, ak.Series)
        self.assertListEqual(c.index.to_list(), ["Bob", "Alice", "Carol"])
        self.assertListEqual(c.values.to_list(), [2, 3, 1])

        # testing counts with IPv4 column
        s = ak.DataFrame({'a': ak.IPv4(ak.arange(1, 5))}).groupby('a').count()
        pds = pd.Series(data=np.ones(4, dtype=np.int64), index=pd.Index(data=np.array(['0.0.0.1', '0.0.0.2', '0.0.0.3', '0.0.0.4'], dtype="<U7")))
        print(pds)
        print(s.to_pandas())
        self.assertTrue(s.to_pandas().equals(other=pds))

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
            self.assertTrue(validation_df.equals(ak_loaded.to_pandas()))

            # test save with index true
            akdf.to_parquet(f"{tmp_dirname}/testName_with_index.pq", index=True)
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/testName_with_index*.pq")), ak.get_config()["numLocales"])

            # Test for df having seg array col
            df = ak.DataFrame({"a": ak.arange(10), "b": ak.segarray(ak.arange(10), ak.arange(10))})
            df.to_hdf(f"{tmp_dirname}/seg_test.h5")
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/seg_test*.h5")), ak.get_config()["numLocales"])
            ak_loaded = ak.DataFrame.load(f"{tmp_dirname}/seg_test.h5")
            self.assertTrue(df.to_pandas().equals(ak_loaded.to_pandas()))

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
        other_df = ak.DataFrame({"col_A": ak.array([7, 3]), "col_C": ak.array([0, 9])})
        test_df = df.isin(other_df)
        self.assertListEqual(test_df["col_A"].to_list(), [True, True])
        self.assertListEqual(test_df["col_B"].to_list(), [False, False])

    def test_multiindex_compat(self):
        # Added for testing Issue #1505
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        df.groupby(["a", "b"]).sum("c")
