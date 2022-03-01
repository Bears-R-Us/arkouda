import pandas

from base_test import ArkoudaTest
from context import arkouda as ak

class DataFrameTest(ArkoudaTest):
    def test_dataframe_creation(self):
        df = ak.DataFrame()
        self.assertIsInstance(df, ak.DataFrame)

        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})
        self.assertIsInstance(df, ak.DataFrame)
        self.assertEqual(df[0], {'index': 0, 'userName': 'Alice', 'userID': 111, 'item': 0, 'day': 5, 'amount': 0.5})
        self.assertTrue((df['userName'] == ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])).all())
        self.assertEqual(len(df), 6)

    def test_drop(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})
        df.drop('userName')
        self.assertEquals(df.__str__(), ak.DataFrame({'userID': userid,
                                           'item': item, 'day': day, 'amount': amount}).__str__())

        with self.assertRaises(KeyError):
            df.drop('index')

    def test_drop_duplicates(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 1, 0, 2, 1, 0])
        day = ak.array([5, 5, 5, 5, 5, 5])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day})

        dedup = df.drop_duplicates()

        username2 = ak.array(['Alice', 'Bob', 'Carol'])
        userid2 = ak.array([111, 222, 333])
        item2 = ak.array([0, 1, 2])
        day2 = ak.array([5, 5, 5])
        self.assertEquals(dedup.__str__(), ak.DataFrame({'userName': username2, 'userID': userid2,
                           'item': item2, 'day': day2}).__str__())

    def test_shape(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 1, 0, 2, 1, 0])
        day = ak.array([5, 5, 5, 5, 5, 5])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day})

        row, col = df.shape
        self.assertEqual(row, 6)
        self.assertEqual(col, 4)

    def test_reset_index(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 1, 0, 2, 1, 0])
        day = ak.array([5, 5, 5, 5, 5, 5])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day})

        slice_df = df[[1, 3, 5]]
        self.assertTrue((slice_df.index == ak.array([1, 3, 5])).all())
        slice_df.reset_index()
        self.assertTrue((slice_df.index == ak.array([0, 1, 2])).all())

    def test_rename(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 1, 0, 2, 1, 0])
        day = ak.array([5, 5, 5, 5, 5, 5])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day})

        rename = {'userName': 'name_col', 'userID': 'user_id'}
        df.rename(rename)
        self.assertIn("user_id", df.__str__())
        self.assertIn("name_col", df.__str__())
        self.assertNotIn('userName', df.__str__())
        self.assertNotIn('userID', df.__str__())
        print(df.__str__())

    def test_head(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 1, 0, 2, 1, 0])
        day = ak.array([5, 5, 5, 5, 5, 5])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day})

        hdf = df.head(3)
        self.assertEqual(len(hdf), 3)
        self.assertTrue((hdf.index == ak.array([0, 1, 2])).all())

    def test_tail(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 1, 0, 2, 1, 0])
        day = ak.array([5, 5, 5, 5, 5, 5])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day})

        hdf = df.tail(2)
        self.assertEqual(len(hdf), 2)
        self.assertTrue((hdf.index == ak.array([4, 5])).all())

    def test_groupby_standard(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})

        gb = df.GroupBy('userName')
        keys, count = gb.count()
        self.assertTrue((keys == ak.array(['Alice', 'Carol', 'Bob'])).all())
        self.assertTrue((count == ak.array([3, 1, 2])).all())

        gb = df.GroupBy(['userName', 'userID'])
        keys, count = gb.count()
        self.assertEqual(len(keys), 2)
        self.assertTrue((keys[0] == ak.array(['Alice', 'Carol', 'Bob'])).all())
        self.assertTrue((keys[1] == ak.array([111, 333, 222])).all())
        self.assertTrue((count == ak.array([3, 1, 2])).all())

        with self.assertRaises(NotImplementedError):
            gb = df.GroupBy('userName', use_series=True)

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
        test_df = pandas.DataFrame(data, columns=['userName', 'userID', 'item', 'day', 'amount'])
        self.assertTrue(pddf.equals(test_df))


        slice_df = df[[1, 3, 5]]
        pddf = slice_df.to_pandas(retain_index=True)
        self.assertEqual(pddf.index.tolist(), [1, 3, 5])

        pddf = slice_df.to_pandas()
        self.assertEqual(pddf.index.tolist(), [0, 1, 2])

    def test_argsort(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})

        p = df.argsort(key='userName')
        self.assertTrue((p == ak.array([0, 2, 5, 1, 4, 3])).all())

        p = df.argsort(key='userName', ascending=False)
        self.assertTrue((p == ak.array([3, 4, 1, 5, 2, 0])).all())

    def test_coargsort(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})

        p = df.coargsort(keys=['userID', 'amount'])
        self.assertTrue((p == ak.array([0, 5, 2, 1, 4, 3])).all())

        p = df.coargsort(keys=['userID', 'amount'], ascending=False)
        self.assertTrue((p == ak.array([3, 4, 1, 2, 5, 0])).all())

    def test_sort_values(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])

        df = ak.DataFrame({'userID':userid})
        ord = df.sort_values()
        self.assertEqual(ord.__repr__(), ak.DataFrame({'userID': ak.array([111, 111, 111, 222, 222, 333])}).__repr__())
        ord = df.sort_values(ascending=False)
        self.assertEqual(ord.__repr__(), ak.DataFrame({'userID': ak.array([333, 222, 222, 111, 111, 111])}).__repr__())

        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})
        ord = df.sort_values(by='userID')
        test_un = ak.array(['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Carol'])
        test_uid = ak.array([111, 111, 111, 222, 222, 333])
        test_i = ak.array([0, 1, 0, 0, 2, 1])
        test_d = ak.array([5, 6, 6, 5, 6, 5])
        test_a = ak.array([0.5, 1.1, 0.6, 0.6, 4.3, 1.2])
        test_df = ak.DataFrame({'userName': test_un, 'userID': test_uid,
                                'item': test_i, 'day': test_d, 'amount': test_a})
        self.assertEqual(ord.__repr__(), test_df.__repr__())

        ord = df.sort_values(by=['userID', 'day'])
        self.assertEqual(ord.__repr__(), test_df.__repr__())

        with self.assertRaises(TypeError):
            df.sort_values(by=1)

    def test_apply_perm(self):
        username = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        df = ak.DataFrame({'userName': username, 'userID': userid,
                           'item': item, 'day': day, 'amount': amount})

        ord = df.sort_values(by='userID')
        default_perm = ak.array([0, 3, 1, 5, 4, 2])
        ord.apply_permutation(default_perm)
        self.assertEqual(ord.__repr__(), df.__repr__())

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
