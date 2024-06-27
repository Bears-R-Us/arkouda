import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak
from pandas.testing import assert_frame_equal, assert_series_equal

from arkouda.series import Series


class SeriesTest(ArkoudaTest):
    def test_series_creation(self):
        # Use positional arguments
        ar_tuple = ak.arange(3), ak.arange(3)
        s = ak.Series(ar_tuple)
        self.assertIsInstance(s, ak.Series)

        ar_tuple = ak.array(["A", "B", "C"]), ak.arange(3)
        s = ak.Series(ar_tuple)
        self.assertIsInstance(s, ak.Series)

        # Both data and index are supplied
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        self.assertIsInstance(s, ak.Series)
        self.assertIsInstance(s.index, ak.Index)

        # Just data is supplied
        s = ak.Series(data=v)
        self.assertIsInstance(s, ak.Series)
        self.assertIsInstance(s.index, ak.Index)

        # Just index is supplied (keyword argument)
        with self.assertRaises(TypeError):
            s = ak.Series(index=i)

        # Just data is supplied (positional argument)
        s = ak.Series(ak.array(["A", "B", "C"]))
        self.assertIsInstance(s, ak.Series)

        # Just index is supplied (ar_tuple argument)
        ar_tuple = (ak.arange(3),)
        with self.assertRaises(TypeError):
            s = ak.Series(ar_tuple)

        # No arguments are supplied
        with self.assertRaises(TypeError):
            s = ak.Series()

        with self.assertRaises(ValueError):
            s = ak.Series(data=ak.arange(3), index=ak.arange(6))

    def test_lookup(self):
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        lk = s.locate(1)
        self.assertIsInstance(lk, ak.Series)
        self.assertEqual(lk.index[0], 1)
        self.assertEqual(lk.values[0], "B")

        lk = s.locate([0, 2])
        self.assertIsInstance(lk, ak.Series)
        self.assertEqual(lk.index[0], 0)
        self.assertEqual(lk.values[0], "A")
        self.assertEqual(lk.index[1], 2)
        self.assertEqual(lk.values[1], "C")

        # testing index lookup
        i = ak.Index([1])
        lk = s.locate(i)
        self.assertIsInstance(lk, ak.Series)
        self.assertListEqual(lk.index.to_list(), i.index.to_list())
        self.assertEqual(lk.values[0], v[1])

        i = ak.Index([0, 2])
        lk = s.locate(i)
        self.assertIsInstance(lk, ak.Series)
        self.assertListEqual(lk.index.to_list(), i.index.to_list())
        self.assertEqual(lk.values.to_list(), v[ak.array([0, 2])].to_list())

        # testing multi-index lookup
        mi = ak.MultiIndex([ak.arange(3), ak.array([2, 1, 0])])
        s = ak.Series(data=v, index=mi)
        lk = s.locate(mi[0])
        self.assertIsInstance(lk, ak.Series)
        self.assertListEqual(lk.index.index, mi[0].index)
        self.assertEqual(lk.values[0], v[0])

        # ensure error with scalar and multi-index
        with self.assertRaises(TypeError):
            lk = s.locate(0)

        with self.assertRaises(TypeError):
            lk = s.locate([0, 2])

    def test_shape(self):
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        (l,) = s.shape
        self.assertEqual(l, 3)

    def test_to_markdown(self):
        s = ak.Series(["elk", "pig", "dog", "quetzal"], name="animal")
        self.assertEqual(
            s.to_markdown(),
            "+----+----------+\n"
            "|    | animal   |\n"
            "+====+==========+\n"
            "|  0 | elk      |\n"
            "+----+----------+\n"
            "|  1 | pig      |\n"
            "+----+----------+\n"
            "|  2 | dog      |\n"
            "+----+----------+\n"
            "|  3 | quetzal  |\n"
            "+----+----------+",
        )
        self.assertEqual(
            s.to_markdown(index=False),
            "+----------+\n"
            "| animal   |\n"
            "+==========+\n"
            "| elk      |\n"
            "+----------+\n"
            "| pig      |\n"
            "+----------+\n"
            "| dog      |\n"
            "+----------+\n"
            "| quetzal  |\n"
            "+----------+",
        )
        self.assertEqual(
            s.to_markdown(tablefmt="grid"),
            "+----+----------+\n"
            "|    | animal   |\n"
            "+====+==========+\n"
            "|  0 | elk      |\n"
            "+----+----------+\n"
            "|  1 | pig      |\n"
            "+----+----------+\n"
            "|  2 | dog      |\n"
            "+----+----------+\n"
            "|  3 | quetzal  |\n"
            "+----+----------+",
        )

        self.assertEqual(s.to_markdown(tablefmt="grid"), s.to_pandas().to_markdown(tablefmt="grid"))
        self.assertEqual(
            s.to_markdown(tablefmt="grid", index=False),
            s.to_pandas().to_markdown(tablefmt="grid", index=False),
        )
        self.assertEqual(s.to_markdown(tablefmt="jira"), s.to_pandas().to_markdown(tablefmt="jira"))

    def test_add(self):
        i = ak.arange(3)
        v = ak.arange(3, 6, 1)
        s = ak.Series(data=i, index=i)

        s_add = ak.Series(data=v, index=v)

        added = s.add(s_add)

        idx_list = added.index.to_pandas().tolist()
        val_list = added.values.to_list()
        for i in range(6):
            self.assertIn(i, idx_list)
            self.assertIn(i, val_list)

    def test_topn(self):
        v = ak.arange(3)
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        top = s.topn(2)
        self.assertListEqual(top.index.to_pandas().tolist(), [2, 1])
        self.assertListEqual(top.values.to_list(), [2, 1])

    def test_sort_idx(self):
        v = ak.arange(5)
        i = ak.array([3, 1, 4, 0, 2])
        s = ak.Series(data=v, index=i)

        sorted = s.sort_index()
        self.assertListEqual(sorted.index.to_pandas().tolist(), [i for i in range(5)])
        self.assertListEqual(sorted.values.to_list(), [3, 1, 4, 0, 2])

    def test_sort_value(self):
        v = ak.array([3, 1, 4, 0, 2])
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        sorted = s.sort_values()
        self.assertListEqual(sorted.index.to_pandas().tolist(), [3, 1, 4, 0, 2])
        self.assertListEqual(sorted.values.to_list(), [i for i in range(5)])

    def test_head_tail(self):
        v = ak.arange(5)
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        head = s.head(2)
        self.assertListEqual(head.index.to_pandas().tolist(), [0, 1])
        self.assertListEqual(head.values.to_list(), [0, 1])

        tail = s.tail(3)
        self.assertListEqual(tail.index.to_pandas().tolist(), [2, 3, 4])
        self.assertListEqual(tail.values.to_list(), [2, 3, 4])

    def test_value_counts(self):
        v = ak.array([0, 0, 1, 2, 2])
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        c = s.value_counts()
        self.assertListEqual(c.index.to_pandas().tolist(), [0, 2, 1])
        self.assertListEqual(c.values.to_list(), [2, 2, 1])

        c = s.value_counts(sort=True)
        self.assertListEqual(c.index.to_pandas().tolist(), [0, 2, 1])
        self.assertListEqual(c.values.to_list(), [2, 2, 1])

    def test_concat(self):
        v = ak.arange(5)
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        v = ak.arange(5, 11, 1)
        i = ak.arange(5, 11, 1)
        s2 = ak.Series(data=v, index=i)

        c = ak.Series.concat([s, s2])
        self.assertListEqual(c.index.to_pandas().tolist(), [i for i in range(11)])
        self.assertListEqual(c.values.to_list(), [i for i in range(11)])

        df = ak.Series.concat([s, s2], axis=1)
        self.assertIsInstance(df, ak.DataFrame)

        ref_df = pd.DataFrame(
            {
                "idx": [i for i in range(11)],
                "val_0": [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
                "val_1": [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10],
            }
        )
        self.assertTrue(((ref_df == df.to_pandas()).all()).all())

    def test_pdconcat(self):
        v = ak.arange(5)
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        v = ak.arange(5, 11, 1)
        i = ak.arange(5, 11, 1)
        s2 = ak.Series(data=v, index=i)

        c = ak.Series.pdconcat([s, s2])
        self.assertIsInstance(c, pd.Series)
        self.assertListEqual(c.index.tolist(), [i for i in range(11)])
        self.assertListEqual(c.values.tolist(), [i for i in range(11)])

        v = ak.arange(5, 10, 1)
        i = ak.arange(5, 10, 1)
        s2 = ak.Series(data=v, index=i)

        df = ak.Series.pdconcat([s, s2], axis=1)
        self.assertIsInstance(df, pd.DataFrame)

        ref_df = pd.DataFrame({0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})
        self.assertTrue((ref_df == df).all().all())

    def test_index_as_index_compat(self):
        # added to validate functionality for issue #1506
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        g = df.groupby(["a", "b"])
        series = ak.Series(data=g.sum("c")["c"], index=g.sum("c").index)
        g.broadcast(series)

    def test_has_repeat_labels(self):
        ints = ak.array([0, 1, 3, 7, 3])
        floats = ak.array([0.0, 1.5, 0.5, 1.5, -1.0])
        strings = ak.array(["A", "C", "C", "DE", "Z"])
        for idxs in [ints, floats, strings]:
            s1 = ak.Series(index=idxs, data=floats)
            self.assertTrue(s1.has_repeat_labels())

        s2 = ak.Series(index=ak.array([0, 1, 3, 4, 5]), data=floats)
        self.assertFalse(s2.has_repeat_labels())

    def test_getitem_scalars(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        with self.assertRaises(TypeError):
            s1[1.0]
        with self.assertRaises(TypeError):
            s1[1]
        _s1_a1 = _s1["A"]
        s1_a1 = s1["A"]

        self.assertTrue(isinstance(_s1_a1, float))
        self.assertTrue(isinstance(s1_a1, float))
        self.assertEqual(s1_a1, _s1_a1)
        _s1_a2 = _s1["C"]
        s1_a2 = s1["C"]
        self.assertTrue(isinstance(_s1_a2, pd.Series))
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s1_a2.index.to_list(), _s1_a2.index.tolist())
        self.assertListEqual(s1_a2.values.to_list(), _s1_a2.values.tolist())

        _s2 = pd.Series(index=np.array(ints), data=np.array(strings))
        s2 = ak.Series(index=ak.array(ints), data=ak.array(strings))
        with self.assertRaises(TypeError):
            s2[1.0]
        with self.assertRaises(TypeError):
            s2["A"]
        _s2_a1 = _s2[7]
        s2_a1 = s2[7]
        self.assertTrue(isinstance(_s2_a1, str))
        self.assertTrue(isinstance(s2_a1, str))
        self.assertEqual(_s2_a1, s2_a1)

        _s2_a2 = _s2[3]
        s2_a2 = s2[3]
        self.assertTrue(isinstance(_s2_a2, pd.Series))
        self.assertTrue(isinstance(s2_a2, ak.Series))
        self.assertListEqual(s2_a2.index.to_list(), _s2_a2.index.tolist())
        self.assertListEqual(s2_a2.values.to_list(), _s2_a2.values.tolist())

        _s3 = pd.Series(index=np.array(floats), data=np.array(ints))
        s3 = ak.Series(index=ak.array(floats), data=ak.array(ints))
        with self.assertRaises(TypeError):
            s3[1]
        with self.assertRaises(TypeError):
            s3["A"]
        _s3_a1 = _s3[0.0]
        s3_a1 = s3[0.0]
        self.assertIsInstance(_s3_a1, np.int64)
        self.assertIsInstance(s3_a1, np.int64)

        _s3_a2 = _s3[1.5]
        s3_a2 = s3[1.5]
        self.assertTrue(isinstance(_s3_a2, pd.Series))
        self.assertTrue(isinstance(s3_a2, ak.Series))
        self.assertListEqual(s3_a2.index.to_list(), _s3_a2.index.tolist())
        self.assertListEqual(s3_a2.values.to_list(), _s3_a2.values.tolist())

    def test_getitem_vectors(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))

        # Arkouda requires homogeneous index type
        with self.assertRaises(TypeError):
            s1[[1.0, 2.0]]
        with self.assertRaises(TypeError):
            s1[[1, 2]]
        with self.assertRaises(TypeError):
            s1[ak.array([1.0, 2.0])]
        with self.assertRaises(TypeError):
            s1[ak.array([1, 2])]

        _s1_a1 = _s1[np.array(["A", "Z"])]
        s1_a1 = s1[ak.array(["A", "Z"])]
        self.assertTrue(isinstance(s1_a1, ak.Series))
        self.assertListEqual(s1_a1.index.to_list(), _s1_a1.index.tolist())
        self.assertListEqual(s1_a1.values.to_list(), _s1_a1.values.tolist())

        _s1_a2 = _s1[["C", "DE"]]
        s1_a2 = s1[["C", "DE"]]
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s1_a2.index.to_list(), _s1_a2.index.tolist())
        self.assertListEqual(s1_a2.values.to_list(), _s1_a2.values.tolist())

        _s1_a3 = _s1[[True, False, True, False, False]]
        s1_a3 = s1[[True, False, True, False, False]]
        self.assertTrue(isinstance(s1_a3, ak.Series))
        self.assertListEqual(s1_a3.index.to_list(), _s1_a3.index.tolist())
        self.assertListEqual(s1_a3.values.to_list(), _s1_a3.values.tolist())

        with self.assertRaises(IndexError):
            _s1[[True, False, True]]
        with self.assertRaises(IndexError):
            s1[[True, False, True]]
        with self.assertRaises(IndexError):
            _s1[np.array([True, False, True])]
        with self.assertRaises(IndexError):
            s1[ak.array([True, False, True])]

        _s2 = pd.Series(index=np.array(floats), data=np.array(ints))
        s2 = ak.Series(index=ak.array(floats), data=ak.array(ints))
        with self.assertRaises(TypeError):
            s2[["A"]]
        with self.assertRaises(TypeError):
            s2[[1, 2]]
        with self.assertRaises(TypeError):
            s2[ak.array(["A", "B"])]
        with self.assertRaises(TypeError):
            s2[ak.array([1, 2])]

        _s2_a1 = _s2[[0.5, 0.0]]
        s2_a1 = s2[[0.5, 0.0]]
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s2_a1.index.to_list(), _s2_a1.index.tolist())
        self.assertListEqual(s2_a1.values.to_list(), _s2_a1.values.tolist())

        _s2_a2 = _s2[np.array([0.5, 0.0])]
        s2_a2 = s2[ak.array([0.5, 0.0])]
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s2_a2.index.to_list(), _s2_a2.index.tolist())
        self.assertListEqual(s2_a2.values.to_list(), _s2_a2.values.tolist())

        with self.assertRaises(KeyError):
            _s2_a3 = _s2[[1.5, 1.2]]
        with self.assertRaises(KeyError):
            s2_a3 = s2[[1.5, 1.2]]

        _s2_a3 = _s2[[1.5, 0.0]]
        s2_a3 = s2[[1.5, 0.0]]
        self.assertTrue(isinstance(s2_a2, ak.Series))
        self.assertListEqual(s2_a3.index.to_list(), _s2_a3.index.tolist())
        self.assertListEqual(s2_a3.values.to_list(), _s2_a3.values.tolist())

    def test_setitem_scalars(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))

        with self.assertRaises(TypeError):
            s1[1.0] = 1.0
        with self.assertRaises(TypeError):
            s1[1] = 1.0
        with self.assertRaises(TypeError):
            s1["A"] = 1
        with self.assertRaises(TypeError):
            s1["A"] = "C"

        s1["A"] = 0.2
        _s1["A"] = 0.2
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())
        s1["C"] = 1.2
        _s1["C"] = 1.2
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())
        s1["X"] = 0.0
        _s1["X"] = 0.0
        self.assertListEqual(s1.index.to_list(), _s1.index.tolist())
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())
        s1["C"] = [0.3, 0.4]
        _s1["C"] = [0.3, 0.4]
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())

        with self.assertRaises(ValueError):
            s1["C"] = [0.4, 0.3, 0.2]

        # cannot assign to Strings
        s2 = ak.Series(index=ak.array(ints), data=ak.array(strings))
        with self.assertRaises(TypeError):
            s2[1.0] = "D"
        with self.assertRaises(TypeError):
            s2["C"] = "E"
        with self.assertRaises(TypeError):
            s2[0] = 1.0
        with self.assertRaises(TypeError):
            s2[0] = 1
        with self.assertRaises(TypeError):
            s2[7] = "L"
        with self.assertRaises(TypeError):
            s2[3] = ["X1", "X2"]

        s3 = ak.Series(index=ak.array(floats), data=ak.array(ints))
        _s3 = pd.Series(index=np.array(floats), data=np.array(ints))
        self.assertListEqual(s3.values.to_list(), [0, 1, 3, 7, 3])
        self.assertListEqual(s3.index.to_list(), [0.0, 1.5, 0.5, 1.5, -1.0])
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        self.assertListEqual(s3.index.to_list(), _s3.index.tolist())
        s3[0.0] = 2
        _s3[0.0] = 2
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        _s3[1.5] = 8
        s3[1.5] = 8
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        _s3[2.0] = 9
        s3[2.0] = 9
        self.assertListEqual(s3.index.to_list(), _s3.index.tolist())
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        _s3[1.5] = [4, 5]
        s3[1.5] = [4, 5]
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        _s3[1.5] = np.array([6, 7])
        s3[1.5] = ak.array([6, 7])
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        _s3[1.5] = [8]
        s3[1.5] = [8]
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        _s3[1.5] = np.array([2])
        s3[1.5] = ak.array([2])
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        with self.assertRaises(ValueError):
            s3[1.5] = [9, 10, 11]
        with self.assertRaises(ValueError):
            s3[1.5] = ak.array([0, 1, 2])

        # adding new entries
        _s3[-1.0] = 14
        s3[-1.0] = 14
        self.assertListEqual(s3.values.to_list(), _s3.values.tolist())
        self.assertListEqual(s3.index.to_list(), _s3.index.tolist())

        # pandas makes the entry a list, which is not what we want.
        with self.assertRaises(ValueError):
            s3[-11.0] = [13, 14, 15]

    def test_setitem_vectors(self):
        ints = [0, 1, 3, 7, 3]
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]

        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))

        # mismatching types for values and indices
        with self.assertRaises(TypeError):
            s1[[0.1, 0.2]] = 1.0
        with self.assertRaises(TypeError):
            s1[[0, 3]] = 1.0
        with self.assertRaises(TypeError):
            s1[ak.array([0, 3])] = 1.0
        with self.assertRaises(TypeError):
            s1[["A", "B"]] = 1
        with self.assertRaises(TypeError):
            s1[["A", "B"]] = "C"
        with self.assertRaises(TypeError):
            s1[ak.array(["A", "B"])] = 1

        # indexing using list of labels only valid with uniquely labeled Series
        with self.assertRaises(pd.errors.InvalidIndexError):
            _s1[["A", "Z"]] = 2.0
        self.assertTrue(s1.has_repeat_labels())
        with self.assertRaises(ValueError):
            s1[["A", "Z"]] = 2.0

        s2 = ak.Series(index=ak.array(["A", "C", "DE", "F", "Z"]), data=ak.array(ints))
        _s2 = pd.Series(index=pd.array(["A", "C", "DE", "F", "Z"]), data=pd.array(ints))
        s2[["A", "Z"]] = 2
        _s2[["A", "Z"]] = 2
        self.assertListEqual(s2.values.to_list(), _s2.values.tolist())
        s2[ak.array(["A", "Z"])] = 3
        _s2[np.array(["A", "Z"])] = 3
        self.assertListEqual(s2.values.to_list(), _s2.values.tolist())
        with self.assertRaises(ValueError):
            _s2[np.array(["A", "Z"])] = [3]
        with self.assertRaises(ValueError):
            s2[ak.array(["A", "Z"])] = [3]

        with self.assertRaises(KeyError):
            _s2[np.array(["B", "D"])] = 0
        with self.assertRaises(KeyError):
            s2[ak.array(["B", "D"])] = 0
        with self.assertRaises(KeyError):
            _s2[["B", "D"]] = 0
        with self.assertRaises(KeyError):
            s2[["B", "D"]] = 0
        with self.assertRaises(KeyError):
            _s2[["B"]] = 0
        with self.assertRaises(KeyError):
            s2[["B"]] = 0
        self.assertListEqual(s2.values.to_list(), _s2.values.tolist())
        self.assertListEqual(s2.index.to_list(), _s2.index.tolist())

        _s2[np.array(["A", "C", "F"])] = [10, 11, 12]
        s2[ak.array(["A", "C", "F"])] = [10, 11, 12]
        self.assertListEqual(s2.values.to_list(), _s2.values.tolist())

    def test_iloc(self):
        floats = [0.0, 1.5, 0.5, 1.5, -1.0]
        strings = ["A", "C", "C", "DE", "Z"]
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        _s1 = pd.Series(index=np.array(strings), data=np.array(floats))

        with self.assertRaises(TypeError):
            s1.iloc["A"]
        with self.assertRaises(TypeError):
            s1.iloc["A"] = 1.0
        with self.assertRaises(TypeError):
            s1.iloc[0] = 1

        s1_a1 = s1.iloc[0]
        self.assertTrue(isinstance(s1_a1, ak.Series))
        self.assertListEqual(s1_a1.index.to_list(), ["A"])
        self.assertListEqual(s1_a1.values.to_list(), [0.0])
        _s1.iloc[0] = 1.0
        s1.iloc[0] = 1.0
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())

        with self.assertRaises(pd.errors.IndexingError):
            _s1_a2 = _s1.iloc[1, 3]
        with self.assertRaises(TypeError):
            s1_a2 = s1.iloc[1, 3]
        with self.assertRaises(IndexError):
            _s1.iloc[1, 3] = 2.0
        with self.assertRaises(TypeError):
            s1.iloc[1, 3] = 2.0

        _s1_a2 = _s1.iloc[[1, 2]]
        s1_a2 = s1.iloc[[1, 2]]
        self.assertListEqual(s1_a2.index.to_list(), _s1_a2.index.tolist())
        self.assertListEqual(s1_a2.values.to_list(), _s1_a2.values.tolist())
        _s1.iloc[[1, 2]] = 0.2
        s1.iloc[[1, 2]] = 0.2
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())

        with self.assertRaises(ValueError):
            _s1.iloc[[3, 4]] = [0.3]
        with self.assertRaises(ValueError):
            s1.iloc[[3, 4]] = [0.3]

        _s1.iloc[[3, 4]] = [0.4, 0.5]
        s1.iloc[[3, 4]] = [0.4, 0.5]
        self.assertListEqual(s1.values.to_list(), _s1.values.tolist())

        with self.assertRaises(TypeError):
            # in pandas this hits a NotImplementedError
            s1.iloc[3, 4] = [0.4, 0.5]

        with self.assertRaises(ValueError):
            s1.iloc[[3, 4]] = ak.array([0.3])
        with self.assertRaises(ValueError):
            s1.iloc[[3, 4]] = [0.1, 0.2, 0.3]

        # iloc does not enlarge its target object
        with self.assertRaises(IndexError):
            _s1.iloc[5]
        with self.assertRaises(IndexError):
            s1.iloc[5]
        with self.assertRaises(IndexError):
            s1.iloc[5] = 2
        with self.assertRaises(IndexError):
            s1.iloc[[3, 5]]
        with self.assertRaises(IndexError):
            s1.iloc[[3, 5]] = [0.1, 0.2]

        # can also take boolean array
        _b = _s1.iloc[[True, False, True, True, False]]
        b = s1.iloc[[True, False, True, True, False]]
        self.assertListEqual(b.values.to_list(), _b.values.tolist())

        _s1.iloc[[True, False, False, True, False]] = [0.5, 0.6]
        s1.iloc[[True, False, False, True, False]] = [0.5, 0.6]
        self.assertListEqual(b.values.to_list(), _b.values.tolist())

        with self.assertRaises(IndexError):
            _s1.iloc[[True, False, True]]
        with self.assertRaises(IndexError):
            s1.iloc[[True, False, True]]

    def test_memory_usage(self):
        n = 2000
        s = ak.Series(ak.arange(n))
        self.assertEqual(
            s.memory_usage(unit="GB", index=False), n * ak.dtypes.int64.itemsize / (1024 * 1024 * 1024)
        )
        self.assertEqual(
            s.memory_usage(unit="MB", index=False), n * ak.dtypes.int64.itemsize / (1024 * 1024)
        )
        self.assertEqual(s.memory_usage(unit="KB", index=False), n * ak.dtypes.int64.itemsize / 1024)
        self.assertEqual(s.memory_usage(unit="B", index=False), n * ak.dtypes.int64.itemsize)

        self.assertEqual(
            s.memory_usage(unit="GB", index=True),
            2 * n * ak.dtypes.int64.itemsize / (1024 * 1024 * 1024),
        )
        self.assertEqual(
            s.memory_usage(unit="MB", index=True), 2 * n * ak.dtypes.int64.itemsize / (1024 * 1024)
        )
        self.assertEqual(s.memory_usage(unit="KB", index=True), 2 * n * ak.dtypes.int64.itemsize / 1024)
        self.assertEqual(s.memory_usage(unit="B", index=True), 2 * n * ak.dtypes.int64.itemsize)

    def test_map(self):
        a = ak.Series(ak.array(["1", "1", "4", "4", "4"]))
        b = ak.Series(ak.array([2, 3, 2, 3, 4]))
        c = ak.Series(ak.array([1.0, 1.0, 2.2, 2.2, 4.4]), index=ak.array([5, 4, 2, 3, 1]))
        d = ak.Series(ak.Categorical(a.values))

        result = a.map({"4": 25, "5": 30, "1": 7})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertListEqual(result.values.to_list(), [7, 7, 25, 25, 25])

        result = a.map({"1": 7})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertListEqual(
            result.values.to_list(),
            ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list(),
        )

        result = a.map({"1": 7.0})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertTrue(
            np.allclose(result.values.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)
        )

        result = b.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertListEqual(result.values.to_list(), [30.0, 5.0, 30.0, 5.0, 25.0])

        result = c.map({1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        self.assertListEqual(result.index.values.to_list(), [5, 4, 2, 3, 1])
        self.assertListEqual(result.values.to_list(), ["a", "a", "b", "b", "c"])

        result = c.map({1.0: "a"})
        self.assertListEqual(result.index.values.to_list(), [5, 4, 2, 3, 1])
        self.assertListEqual(result.values.to_list(), ["a", "a", "null", "null", "null"])

        result = c.map({1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        self.assertListEqual(result.index.values.to_list(), [5, 4, 2, 3, 1])
        self.assertListEqual(result.values.to_list(), ["a", "a", "b", "b", "c"])

        result = d.map({"4": 25, "5": 30, "1": 7})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertListEqual(result.values.to_list(), [7, 7, 25, 25, 25])

        result = d.map({"1": 7})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertListEqual(
            result.values.to_list(),
            ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list(),
        )

        result = d.map({"1": 7.0})
        self.assertListEqual(result.index.values.to_list(), [0, 1, 2, 3, 4])
        self.assertTrue(
            np.allclose(result.values.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)
        )

    def test_isna_int(self):
        # Test case with integer data type
        data_int = Series([1, 2, 3, 4, 5])
        expected_int = Series([False, False, False, False, False])
        self.assertTrue(
            np.allclose(data_int.isna().values.to_ndarray(), expected_int.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_int.isnull().values.to_ndarray(), expected_int.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_int.notna().values.to_ndarray(), ~expected_int.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_int.notnull().values.to_ndarray(), ~expected_int.values.to_ndarray())
        )
        self.assertFalse(data_int.hasnans())

    def test_isna_float(self):
        # Test case with float data type
        data_float = Series([1.0, 2.0, 3.0, np.nan, 5.0])
        expected_float = Series([False, False, False, True, False])
        self.assertTrue(
            np.allclose(data_float.isna().values.to_ndarray(), expected_float.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_float.isnull().values.to_ndarray(), expected_float.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_float.notna().values.to_ndarray(), ~expected_float.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_float.notnull().values.to_ndarray(), ~expected_float.values.to_ndarray())
        )
        self.assertTrue(data_float.hasnans())

    def test_isna_string(self):
        # Test case with string data type
        data_string = Series(["a", "b", "c", "d", "e"])
        expected_string = Series([False, False, False, False, False])
        self.assertTrue(
            np.allclose(data_string.isna().values.to_ndarray(), expected_string.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_string.isnull().values.to_ndarray(), expected_string.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_string.notna().values.to_ndarray(), ~expected_string.values.to_ndarray())
        )
        self.assertTrue(
            np.allclose(data_string.notnull().values.to_ndarray(), ~expected_string.values.to_ndarray())
        )
        self.assertFalse(data_string.hasnans())

    def test_fillna(self):
        data = ak.Series([1, np.nan, 3, np.nan, 5])

        fill_values1 = ak.ones(5)
        self.assertListEqual(data.fillna(fill_values1).to_list(), [1.0, 1.0, 3.0, 1.0, 5.0])

        fill_values2 = Series(2 * ak.ones(5))
        self.assertListEqual(data.fillna(fill_values2).to_list(), [1.0, 2.0, 3.0, 2.0, 5.0])

        fill_values3 = 100.0
        self.assertListEqual(data.fillna(fill_values3).to_list(), [1.0, 100.0, 3.0, 100.0, 5.0])
