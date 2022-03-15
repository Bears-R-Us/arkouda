import pandas as pd  # type: ignore

from base_test import ArkoudaTest
from context import arkouda as ak


def build_ak_df():
    username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    userid = ak.array([111, 222, 111, 333, 222, 111])
    item = ak.array([0, 0, 1, 1, 2, 0])
    day = ak.array([5, 5, 6, 5, 6, 6])
    amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    return ak.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day, 'amount': amount})


def build_ak_df_duplicates():
    username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    userid = ak.array([111, 222, 111, 333, 222, 111])
    item = ak.array([0, 1, 0, 2, 1, 0])
    day = ak.array([5, 5, 5, 5, 5, 5])
    return ak.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day})


def build_ak_append():
    username = ak.array(['John', 'Carol'])
    userid = ak.array([444, 333])
    item = ak.array([0, 2])
    day = ak.array([1, 2])
    amount = ak.array([0.5, 5.1])
    return ak.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day, 'amount': amount})


def build_ak_keyerror():
    userid = ak.array([444, 333])
    item = ak.array([0, 2])
    return ak.DataFrame({'user_id': userid, 'item': item})


def build_ak_typeerror():
    username = ak.array([111, 222, 111, 333, 222, 111])
    userid = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    item = ak.array([0, 0, 1, 1, 2, 0])
    day = ak.array([5, 5, 6, 5, 6, 6])
    amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    return ak.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day, 'amount': amount})


def build_pd_df():
    username = ['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice']
    userid = [111, 222, 111, 333, 222, 111]
    item = [0, 0, 1, 1, 2, 0]
    day = [5, 5, 6, 5, 6, 6]
    amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6]
    return pd.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day, 'amount': amount})


def build_pd_df_duplicates():
    username = ['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice']
    userid = [111, 222, 111, 333, 222, 111]
    item = [0, 1, 0, 2, 1, 0]
    day = [5, 5, 5, 5, 5, 5]
    return pd.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day})


def build_pd_df_append():
    username = ['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice', 'John', 'Carol']
    userid = [111, 222, 111, 333, 222, 111, 444, 333]
    item = [0, 0, 1, 1, 2, 0, 0, 2]
    day = [5, 5, 6, 5, 6, 6, 1, 2]
    amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6, 0.5, 5.1]
    return pd.DataFrame({'userName': username, 'userID': userid,
                         'item': item, 'day': day, 'amount': amount})

class DataFrameTest(ArkoudaTest):
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

    def test_to_pandas(self):
        df = build_ak_df()
        pd_df = build_pd_df()

        self.assertTrue(pd_df.equals(df.to_pandas()))

        slice_df = df[[1, 3, 5]]
        pd_df = slice_df.to_pandas(retain_index=True)
        self.assertEqual(pd_df.index.tolist(), [1, 3, 5])

        pd_df = slice_df.to_pandas()
        self.assertEqual(pd_df.index.tolist(), [0, 1, 2])

    def test_from_pandas(self):
        username = ['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice', 'John', 'Carol']
        userid = [111, 222, 111, 333, 222, 111, 444, 333]
        item = [0, 0, 1, 1, 2, 0, 0, 2]
        day = [5, 5, 6, 5, 6, 6, 1, 2]
        amount = [0.5, 0.6, 1.1, 1.2, 4.3, 0.6, 0.5, 5.1]
        ref_df = pd.DataFrame({'userName': username, 'userID': userid,
                               'item': item, 'day': day, 'amount': amount})

        df = ak.DataFrame(ref_df)

        self.assertTrue(((ref_df == df.to_pandas()).all()).all())

        df = ak.DataFrame.from_pandas(ref_df)
        self.assertTrue(((ref_df == df.to_pandas()).all()).all())


    def test_drop(self):
        # create an arkouda df.
        df = build_ak_df()
        # create pandas df to validate functionality against
        pd_df = build_pd_df()

        # Test dropping columns
        df.drop('userName', axis=1)
        pd_df.drop(labels=['userName'], axis=1, inplace=True)

        self.assertTrue(((df.to_pandas() == pd_df).all()).all())

        # verify that the index cannot be dropped from ak.DataFrame
        with self.assertRaises(KeyError):
            df.drop('index', axis=1)

        # Test dropping rows
        df.drop([0, 2, 5])
        # pandas retains original indexes when dropping rows, need to reset to line up with arkouda
        pd_df.drop(labels=[0, 2, 5], inplace=True)
        pd_df.reset_index(drop=True, inplace=True)

        self.assertTrue(pd_df.equals(df.to_pandas()))

        # verify that index keys must be ints
        with self.assertRaises(TypeError):
            df.drop('index')

        # verify axis can only be 0 or 1
        with self.assertRaises(ValueError):
            df.drop('amount', 15)

    def test_drop_duplicates(self):
        df = build_ak_df_duplicates()
        ref_df = build_pd_df_duplicates()

        dedup = df.drop_duplicates()
        dedup_pd = ref_df.drop_duplicates()
        # pandas retains original indexes when dropping dups, need to reset to line up with arkouda
        dedup_pd.reset_index(drop=True, inplace=True)

        dedup_test = dedup.to_pandas().sort_values('userName').reset_index(drop=True)
        dedup_pd_test = dedup_pd.sort_values('userName').reset_index(drop=True)

        self.assertTrue(dedup_test.equals(dedup_pd_test))

    def test_shape(self):
        df = build_ak_df()

        row, col = df.shape
        self.assertEqual(row, 6)
        self.assertEqual(col, 5)

    def test_reset_index(self):
        df = build_ak_df()

        slice_df = df[[1, 3, 5]]
        self.assertTrue((slice_df.index == ak.array([1, 3, 5])).all())
        slice_df.reset_index()
        self.assertTrue((slice_df.index == ak.array([0, 1, 2])).all())

    def test_rename(self):
        df = build_ak_df()

        rename = {'userName': 'name_col', 'userID': 'user_id'}
        df.rename(rename)
        self.assertIn("user_id", df.columns)
        self.assertIn("name_col", df.columns)
        self.assertNotIn('userName', df.columns)
        self.assertNotIn('userID', df.columns)

    def test_append(self):
        df = build_ak_df()
        df_toappend = build_ak_append()

        df.append(df_toappend)

        ref_df = build_pd_df_append()

        # dataframe equality returns series with bool result for each row.
        self.assertTrue(ref_df.equals(df.to_pandas()))

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

        gb = df.GroupBy('userName')
        keys, count = gb.count()
        self.assertTrue(keys.to_ndarray().tolist(), ['Alice', 'Carol', 'Bob'])
        self.assertListEqual(count.to_ndarray().tolist(), [3, 1, 2])
        self.assertListEqual(gb.permutation.to_ndarray().tolist(), [0, 2, 5, 3, 1, 4])

        gb = df.GroupBy(['userName', 'userID'])
        keys, count = gb.count()
        self.assertEqual(len(keys), 2)
        self.assertListEqual(keys[0].to_ndarray().tolist(), ['Alice', 'Carol', 'Bob'])
        self.assertTrue(keys[1].to_ndarray().tolist(), [111, 333, 222])
        self.assertTrue(count.to_ndarray().tolist(), [3, 1, 2])

    def test_gb_series(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})

        gb = df.GroupBy('userName', use_series=True)

        c = gb.count()
        self.assertIsInstance(c, ak.Series)
        self.assertListEqual(c.index.to_pandas().tolist(), ['Alice', 'Carol', 'Bob'])
        self.assertListEqual(c.values.to_ndarray().tolist(), [3, 1, 2])

    def test_to_pandas(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})

        pddf = df.to_pandas()
        data = [
            ['Alice', 111, 0, 5, 0.5],
            ['Bob', 222, 0, 5, 0.6],
            ['Alice', 111, 1, 6, 1.1],
            ['Carol', 333, 1, 5, 1.2],
            ['Bob', 222, 2, 6, 4.3],
            ['Alice', 111, 0, 6, 0.6]
        ]
        test_df = pd.DataFrame(data, columns=['userName', 'userID', 'item', 'day', 'amount'])
        self.assertTrue(pddf.equals(test_df))

        slice_df = df[[1, 3, 5]]
        pddf = slice_df.to_pandas(retain_index=True)
        self.assertEqual(pddf.index.tolist(), [1, 3, 5])

        pddf = slice_df.to_pandas()
        self.assertEqual(pddf.index.tolist(), [0, 1, 2])

    def test_argsort(self):
        df = build_ak_df()

        p = df.argsort(key='userName')
        self.assertListEqual(p.to_ndarray().tolist(), [0, 2, 5, 1, 4, 3])

        p = df.argsort(key='userName', ascending=False)
        self.assertListEqual(p.to_ndarray().tolist(), [3, 4, 1, 5, 2, 0])

    def test_coargsort(self):
        df = build_ak_df()

        p = df.coargsort(keys=['userID', 'amount'])
        self.assertListEqual(p.to_ndarray().tolist(), [0, 5, 2, 1, 4, 3])

        p = df.coargsort(keys=['userID', 'amount'], ascending=False)
        self.assertListEqual(p.to_ndarray().tolist(), [3, 4, 1, 2, 5, 0])

    def test_sort_values(self):
        userid = [111, 222, 111, 333, 222, 111]
        userid_ak = ak.array(userid)

        # sort userid to build dataframes to reference
        userid.sort()

        df = ak.DataFrame({'userID': userid_ak})
        ord = df.sort_values()
        self.assertTrue(ord.to_pandas().equals(pd.DataFrame(data=userid, columns=['userID'])))
        ord = df.sort_values(ascending=False)
        userid.reverse()
        self.assertTrue(ord.to_pandas().equals(pd.DataFrame(data=userid, columns=['userID'])))

        df = build_ak_df()
        ord = df.sort_values(by='userID')
        ref_df = build_pd_df()
        ref_df = ref_df.sort_values(by='userID').reset_index(drop=True)
        self.assertTrue(ref_df.equals(ord.to_pandas()))

        ord = df.sort_values(by=['userID', 'day'])
        ref_df = ref_df.sort_values(by=['userID', 'day']).reset_index(drop=True)
        self.assertTrue(ref_df.equals(ord.to_pandas()))

        with self.assertRaises(TypeError):
            df.sort_values(by=1)

    def test_intx(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        df_1 = ak.DataFrame({'user_name': username, 'user_id': userid})

        username = ak.array(['Bob', 'Alice'])
        userid = ak.array([222, 445])
        df_2 = ak.DataFrame({'user_name': username, 'user_id': userid})

        rows = ak.intx(df_1, df_2)
        self.assertListEqual(rows.to_ndarray().tolist(), [False, True, False, False, True, False])

        df_3 = ak.DataFrame({'user_name': username, 'user_number': userid})
        with self.assertRaises(ValueError):
            rows = ak.intx(df_1, df_3)

    def test_apply_perm(self):
        df = build_ak_df()
        ref_df = build_pd_df()

        ord = df.sort_values(by='userID')
        perm_list = [0, 3, 1, 5, 4, 2]
        default_perm = ak.array(perm_list)
        ord.apply_permutation(default_perm)

        ord_ref = ref_df.sort_values(by='userID').reset_index(drop=True)
        ord_ref = ord_ref.reindex(perm_list).reset_index(drop=True)
        self.assertTrue(ord_ref.equals(ord.to_pandas()))

    def test_filter_by_range(self):
        userid = ak.array([111, 222, 111, 333, 222, 111])
        amount = ak.array([0, 1, 1, 2, 3, 15])
        df = ak.DataFrame({'userID': userid, 'amount': amount})

        filtered = df.filter_by_range(keys=['userID'], low=1, high=2)
        self.assertFalse(filtered[0])
        self.assertTrue(filtered[1])
        self.assertFalse(filtered[2])
        self.assertTrue(filtered[3])
        self.assertTrue(filtered[4])
        self.assertFalse(filtered[5])

    def test_copy(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        df = ak.DataFrame({'userName': username, 'userID': userid})

        df_copy = df.copy(deep=True)
        self.assertEqual(df.__repr__(), df_copy.__repr__())

        df_copy.__setitem__('userID', ak.array([1, 2, 1, 3, 2, 1]))
        self.assertNotEqual(df.__repr__(), df_copy.__repr__())

        df_copy = df.copy(deep=False)
        df_copy.__setitem__('userID', ak.array([1, 2, 1, 3, 2, 1]))
        self.assertEqual(df.__repr__(), df_copy.__repr__())
