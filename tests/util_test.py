import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import list_registry
from arkouda.util import attach, unregister_by_name


class utilTest(ArkoudaTest):
    def test_simple_attach(self):
        a = ak.array(["abc", "123", "def"])
        b = ak.arange(10)

        # Attach the Strings array and the pdarray to new objects
        a_attached = attach(a.name)
        a_typed_attach = attach(a.name, "strings")
        b_attached = attach(b.name)
        b_typed_attach = attach(b.name, "pdarray")

        self.assertListEqual(a.to_list(), a_attached.to_list())
        self.assertIsInstance(a_attached, ak.Strings)

        self.assertListEqual(a.to_list(), a_typed_attach.to_list())
        self.assertIsInstance(a_typed_attach, ak.Strings)

        self.assertListEqual(b.to_list(), b_attached.to_list())
        self.assertIsInstance(b_attached, ak.pdarray)

        self.assertListEqual(b.to_list(), b_typed_attach.to_list())
        self.assertIsInstance(b_typed_attach, ak.pdarray)

    def test_categorical_attach(self):
        strings = ak.array(
            ["hurrah", ",", "hurrah", ",", "one", "by", "one", "marching", "go", "ants", "the"]
        )
        cat = ak.Categorical(strings)
        cat.register("catTest")

        attached = attach("catTest")
        self.assertListEqual(cat.to_list(), attached.to_list())
        self.assertIsInstance(attached, ak.Categorical)

        attached_typed = attach("catTest", "Categorical")
        self.assertListEqual(cat.to_list(), attached_typed.to_list())
        self.assertIsInstance(attached_typed, ak.Categorical)

    def test_segArray_attach(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.segarray(segments, akflat)
        segarr.register("segTest")

        attached = attach("segTest")
        self.assertListEqual(segarr.to_list(), attached.to_list())
        self.assertIsInstance(attached, ak.SegArray)

        attached_typed = attach("segTest", "SegArray")
        self.assertListEqual(segarr.to_list(), attached_typed.to_list())
        self.assertIsInstance(attached_typed, ak.SegArray)

    def test_series_attach(self):
        index_tuple = (ak.arange(5), ak.arange(5, 10))
        s = ak.Series(data=ak.arange(0, 10, 2), index=index_tuple)  # MultiIndex Series
        s2 = ak.Series(data=ak.arange(5), index=ak.arange(5))  # Single Index Series

        s.register("series_test")
        s2.register("series_2_test")

        s_attach = ak.util.attach("series_test")
        s2_attach = ak.util.attach("series_2_test")

        self.assertListEqual(s_attach.values.to_list(), s.values.to_list())
        sEq = s_attach.index == s.index
        self.assertTrue(all(sEq.to_ndarray()))

        self.assertListEqual(s2_attach.values.to_list(), s2.values.to_list())
        s2Eq = s2_attach.index == s2.index
        self.assertTrue(all(s2Eq.to_ndarray()))

    def test_dataframe_attach(self):
        ind = ak.Index.factory(ak.array(["a", "b", "c", "1", "2", "3"]))
        item = ak.array([0, 0, 1, 1, 2, 0])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        userid = ak.array([111, 222, 111, 333, 222, 111])
        username = ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])
        cat = ak.Categorical(username)

        a = [2, 1, 2, 3]
        b = [2, 2, 3]
        c = [2, 1, 2]
        d = [2, 1, 3]
        e = [1, 2, 3]
        f = [2, 3, 1]

        flat = a + b + c + d + e + f
        akflat = ak.array(flat)
        segments = ak.array([0, 4, 7, 10, 13, 16])
        segarr = ak.segarray(segments, akflat)

        df = ak.DataFrame(
            {"username": cat, "user_ID": userid, "item": item, "amount": amount, "visits": segarr}, ind
        )
        df.register("df_test")

        dfAtt = ak.util.attach("df_test")

        self.assertTrue(df.to_pandas().equals(dfAtt.to_pandas()))

    def test_unregister_by_name(self):
        # Register the four supported object types
        # pdarray
        pda = ak.arange(10)
        pda.register("pdaUnregisterTest")
        self.assertTrue(pda.is_registered())

        # Strings
        s1 = ak.array(["123", "abc", "def"])
        s1.register("stringsUnregisterTest")
        self.assertTrue(s1.is_registered())

        # Categorical
        s2 = ak.array(["abc", "123", "abc"])
        cat = ak.Categorical(s2)
        cat.register("catUnregisterTest")
        self.assertTrue(cat.is_registered())

        # Series
        # Single Index
        s3 = ak.Series(
            index=ak.array(np.random.randint(0, 20, 10)), data=ak.array(np.random.randint(0, 20, 10))
        )
        s3.register("seriesSingleTest")
        self.assertTrue(s3.is_registered())

        multiInd = []
        for x in range(3):
            multiInd.append(ak.array(np.random.randint(0, 20, 10)))
        s4 = ak.Series(index=multiInd, data=ak.array(np.random.randint(0, 20, 10)))
        s4.register("seriesMultiTest")
        registry = list_registry()
        # Series.is_registered() does not support multiIndex
        self.assertTrue("seriesMultiTest_value" in registry)
        self.assertTrue("seriesMultiTest_key_0" in registry)
        self.assertTrue("seriesMultiTest_key_1" in registry)
        self.assertTrue("seriesMultiTest_key_2" in registry)

        unregister_by_name("pdaUnregisterTest")
        self.assertFalse(pda.is_registered())

        unregister_by_name("stringsUnregisterTest")
        self.assertFalse(s1.is_registered())

        unregister_by_name("catUnregisterTest")
        self.assertFalse(cat.is_registered())

        unregister_by_name("seriesSingleTest")
        self.assertFalse(s3.is_registered())

        unregister_by_name("seriesMultiTest")
        registry = list_registry()
        self.assertFalse("seriesMultiTest_value" in registry)
        self.assertFalse("seriesMultiTest_key_0" in registry)
        self.assertFalse("seriesMultiTest_key_1" in registry)
        self.assertFalse("seriesMultiTest_key_2" in registry)
