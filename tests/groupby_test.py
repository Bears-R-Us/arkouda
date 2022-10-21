import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.dtypes import float64, int64
from arkouda.groupbyclass import GroupByReductionType

SIZE = 100
GROUPS = 8
verbose = True


def groupby_to_arrays(df: pd.DataFrame, kname, vname, op, levels):
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op.replace("arg", "idx"))
    if op == "prod":
        # There appears to be a bug in pandas where it sometimes
        # reports the product of a segment as NaN when it should be 0
        agg[agg.isna()] = 0
    if levels == 1:
        keys = agg.index.values
    else:
        keys = tuple(zip(*(agg.index.values)))
    return keys, agg.values


def make_arrays():
    keys = np.random.randint(0, GROUPS, SIZE, dtype=np.uint64)
    keys2 = np.random.randint(0, GROUPS, SIZE)
    i = np.random.randint(0, SIZE // GROUPS, SIZE)
    u = np.random.randint(0, SIZE // GROUPS, SIZE, dtype=np.uint64)
    f = np.random.randn(SIZE)  # normally dist random numbers
    b = (i % 2) == 0
    d = {"keys": keys, "keys2": keys2, "int64": i, "uint64": u, "float64": f, "bool": b}

    return d


def compare_keys(pdkeys, akkeys, levels, pdvals, akvals) -> int:
    """
    Compares the numpy and arkouda arrays via the numpy.allclose method with the
    default relative and absolute tolerances, returning 0 if the arrays are similar
    element-wise within the tolerances, 1 if they are dissimilar.element

    :return: 0 (identical) or 1 (dissimilar)
    :rtype: int
    """
    if levels == 1:
        akkeys = akkeys.to_ndarray()
        if not np.allclose(pdkeys, akkeys):
            print("Different keys")
            return 1
    else:
        for lvl in range(levels):
            if not np.allclose(pdkeys[lvl], akkeys[lvl].to_ndarray()):
                print("Different keys")
                return 1

    if not np.allclose(pdvals, akvals, equal_nan=True):
        print(f"Different values (abs diff = {np.abs(pdvals - akvals).sum()})")
        return 1
    return 0


def run_test(levels, verbose=False):
    """
    The run_test method enables execution of ak.GroupBy and ak.GroupBy.Reductions
    on a randomized set of arrays on the specified number of levels.

    Note: the current set of valid levels is {1,2}
    :return:
    """
    d = make_arrays()
    df = pd.DataFrame(d)
    akdf = {k: ak.array(v) for k, v in d.items()}

    if levels == 1:
        akg = ak.GroupBy(akdf["keys"])
        keyname = "keys"
    elif levels == 2:
        akg = ak.GroupBy([akdf["keys"], akdf["keys2"]])
        keyname = ["keys", "keys2"]
    tests = 0
    failures = 0
    not_impl = 0
    if verbose:
        print("Doing .count()")
    tests += 1
    pdkeys, pdvals = groupby_to_arrays(df, keyname, "int64", "count", levels)
    akkeys, akvals = akg.count()
    akvals = akvals.to_ndarray()
    failures += compare_keys(pdkeys, akkeys, levels, pdvals, akvals)
    for vname in ("int64", "uint64", "float64", "bool"):
        for op in ak.GroupBy.Reductions:
            if verbose:
                print(f"\nDoing aggregate({vname}, {op})")
            tests += 1
            do_check = True
            try:
                pdkeys, pdvals = groupby_to_arrays(df, keyname, vname, op, levels)
            except Exception:
                if verbose:
                    print("Pandas does not implement")
                do_check = False
            try:
                akkeys, akvals = akg.aggregate(akdf[vname], op)
                akvals = akvals.to_ndarray()
            except Exception as E:
                if verbose:
                    print("Arkouda error: ", E)
                not_impl += 1
                do_check = False
                continue
            if not do_check:
                continue
            if op.startswith("arg"):
                pdextrema = df[vname][pdvals]
                akextrema = akdf[vname][ak.array(akvals)].to_ndarray()
                if not np.allclose(pdextrema, akextrema):
                    print("Different argmin/argmax: Arkouda failed to find an extremum")
                    print("pd: ", pdextrema)
                    print("ak: ", akextrema)
                    failures += 1
            else:
                if op != "unique":
                    failures += compare_keys(pdkeys, akkeys, levels, pdvals, akvals)
    print(
        f"{tests - failures - not_impl} / {tests - not_impl} passed, "
        f"{failures} errors, {not_impl} not implemented"
    )
    return failures


"""
The GroupByTest class encapsulates specific calls to the run_test
method within a Python unittest.TestCase object,
which enables integration into a pytest test harness.
"""


class GroupByTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)

        self.bvalues = ak.randint(0, 1, 10, dtype=bool)
        self.fvalues = ak.randint(0, 1, 10, dtype=float)
        self.ivalues = ak.array([4, 1, 3, 2, 2, 2, 5, 5, 2, 3])
        self.uvalues = ak.cast(self.ivalues, ak.uint64)
        self.igb = ak.GroupBy(self.ivalues)
        self.ugb = ak.GroupBy(self.uvalues)

    def test_groupby_on_one_level(self):
        """
        Executes run_test with levels=1 and asserts whether there are any errors

        :return: None
        :raise: AssertionError if there are any errors encountered in run_test with levels = 1
        """
        self.assertEqual(0, run_test(1, verbose))

    def test_groupby_on_two_levels(self):
        """
        Executes run_test with levels=1 and asserts whether there are any errors

        :return: None
        :raise: AssertionError if there are any errors encountered in run_test with levels = 2
        """
        self.assertEqual(0, run_test(2, verbose))

    def test_argmax_argmin(self):
        b = ak.array([True, False, True, True, False, True])
        x = ak.array([True, True, False, True, False, False])
        g = ak.GroupBy(x)
        keys, locs = g.argmin(b)
        self.assertListEqual(keys.to_list(), [False, True])
        self.assertListEqual(locs.to_list(), [4, 1])

        keys, locs = g.argmax(b)
        self.assertListEqual(keys.to_list(), [False, True])
        self.assertListEqual(locs.to_list(), [2, 0])

    def test_boolean_arrays(self):
        a = ak.array([True, False, True, True, False])
        true_ct = a.sum()
        g = ak.GroupBy(a)
        k, ct = g.count()

        self.assertEqual(ct[1], true_ct)
        self.assertListEqual(k.to_list(), [False, True])

        # This test was added since we added the size method for issue #1353
        k, ct = g.size()

        self.assertEqual(ct[1], true_ct)
        self.assertListEqual(k.to_list(), [False, True])

        b = ak.array([False, False, True, False, False])
        g = ak.GroupBy([a, b])
        k, ct = g.count()
        self.assertListEqual(ct.to_list(), [2, 2, 1])
        self.assertListEqual(k[0].to_list(), [False, True, True])
        self.assertListEqual(k[1].to_list(), [False, False, True])

    def test_bitwise_aggregations(self):
        revs = ak.arange(self.igb.length) % 2
        self.assertListEqual(self.igb.OR(revs)[1].to_list(), self.igb.max(revs)[1].to_list())
        self.assertListEqual(self.igb.AND(revs)[1].to_list(), self.igb.min(revs)[1].to_list())
        self.assertListEqual(
            self.igb.XOR(revs)[1].to_list(),
            (self.igb.sum(revs)[1] % 2).to_list(),
        )

    def test_standalone_broadcast(self):
        segs = ak.arange(10) ** 2
        vals = ak.arange(10)
        size = 100
        check = ((2 * vals + 1) * vals).sum()
        self.assertEqual(ak.broadcast(segs, vals, size=size).sum(), check)
        perm = ak.arange(99, -1, -1)
        bcast = ak.broadcast(segs, vals, permutation=perm)
        self.assertEqual(bcast.sum(), check)
        self.assertTrue((bcast[:-1] >= bcast[1:]).all())

    def test_broadcast_ints(self):
        keys, counts = self.igb.count()

        self.assertListEqual([1, 4, 2, 1, 2], counts.to_list())
        self.assertListEqual([1, 2, 3, 4, 5], keys.to_list())

        results = self.igb.broadcast(1 * (counts > 2), permute=False)
        self.assertListEqual([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], results.to_list())

        results = self.igb.broadcast(1 * (counts == 2), permute=False)
        self.assertListEqual([0, 0, 0, 0, 0, 1, 1, 0, 1, 1], results.to_list())

        results = self.igb.broadcast(1 * (counts < 4), permute=False)
        self.assertListEqual([1, 0, 0, 0, 0, 1, 1, 1, 1, 1], results.to_list())

        results = self.igb.broadcast(1 * (counts > 2))
        self.assertListEqual([0, 0, 0, 1, 1, 1, 0, 0, 1, 0], results.to_list())

        results = self.igb.broadcast(1 * (counts == 2))
        self.assertListEqual([0, 0, 1, 0, 0, 0, 1, 1, 0, 1], results.to_list())

        results = self.igb.broadcast(1 * (counts < 4))
        self.assertListEqual([1, 1, 1, 0, 0, 0, 1, 1, 0, 1], results.to_list())

    def test_broadcast_uints(self):
        keys, counts = self.ugb.count()
        self.assertListEqual([1, 4, 2, 1, 2], counts.to_list())
        self.assertListEqual([1, 2, 3, 4, 5], keys.to_list())

        u_results = self.ugb.broadcast(1 * (counts > 2))
        i_results = self.igb.broadcast(1 * (counts > 2))
        self.assertListEqual(i_results.to_list(), u_results.to_list())

        u_results = self.ugb.broadcast(1 * (counts == 2))
        i_results = self.igb.broadcast(1 * (counts == 2))
        self.assertListEqual(i_results.to_list(), u_results.to_list())

        u_results = self.ugb.broadcast(1 * (counts < 4))
        i_results = self.igb.broadcast(1 * (counts < 4))
        self.assertListEqual(i_results.to_list(), u_results.to_list())

        # test uint Groupby.broadcast with and without permute
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64), permute=False)
        i_results = self.igb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64), permute=False)
        self.assertListEqual(i_results.to_list(), u_results.to_list())
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64))
        i_results = self.igb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64))
        self.assertListEqual(i_results.to_list(), u_results.to_list())

        # test uint broadcast
        u_results = ak.broadcast(ak.array([0]), ak.array([1], dtype=ak.uint64), 1)
        i_results = ak.broadcast(ak.array([0]), ak.array([1]), 1)
        self.assertListEqual(i_results.to_list(), u_results.to_list())

    def test_broadcast_booleans(self):
        keys, counts = self.igb.count()

        self.assertListEqual([1, 4, 2, 1, 2], counts.to_list())
        self.assertListEqual([1, 2, 3, 4, 5], keys.to_list())

        results = self.igb.broadcast(counts > 2, permute=False)
        self.assertListEqual([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], results.to_list())

        results = self.igb.broadcast(counts == 2, permute=False)
        self.assertListEqual([0, 0, 0, 0, 0, 1, 1, 0, 1, 1], results.to_list())

        results = self.igb.broadcast(counts < 4, permute=False)
        self.assertListEqual([1, 0, 0, 0, 0, 1, 1, 1, 1, 1], results.to_list())

        results = self.igb.broadcast(counts > 2)
        self.assertListEqual([0, 0, 0, 1, 1, 1, 0, 0, 1, 0], results.to_list())

        results = self.igb.broadcast(counts == 2)
        self.assertListEqual([0, 0, 1, 0, 0, 0, 1, 1, 0, 1], results.to_list())

        results = self.igb.broadcast(counts < 4)
        self.assertListEqual([1, 1, 1, 0, 0, 0, 1, 1, 0, 1], results.to_list())

    def test_count(self):
        keys, counts = self.igb.count()

        self.assertListEqual([1, 2, 3, 4, 5], keys.to_list())
        self.assertListEqual([1, 4, 2, 1, 2], counts.to_list())

    def test_groupby_reduction_type(self):
        self.assertEqual("any", str(GroupByReductionType.ANY))
        self.assertEqual("all", str(GroupByReductionType.ALL))
        self.assertEqual(GroupByReductionType.ANY, GroupByReductionType("any"))

        with self.assertRaises(ValueError):
            GroupByReductionType("an")

        self.assertIsInstance(ak.GROUPBY_REDUCTION_TYPES, frozenset)
        self.assertTrue("any" in ak.GROUPBY_REDUCTION_TYPES)

    def test_error_handling(self):
        d = make_arrays()
        akdf = {k: ak.array(v) for k, v in d.items()}
        gb = ak.GroupBy([akdf["keys"], akdf["keys2"]])

        with self.assertRaises(TypeError):
            ak.GroupBy(ak.arange(4), ak.arange(4))

        with self.assertRaises(TypeError):
            ak.GroupBy(self.fvalues)

        with self.assertRaises(TypeError):
            gb.broadcast([])

        with self.assertRaises(TypeError):
            self.igb.nunique(ak.randint(0, 1, 10, dtype=float64))

        with self.assertRaises(TypeError):
            self.igb.any(ak.randint(0, 1, 10, dtype=float64))

        with self.assertRaises(TypeError):
            self.igb.any(ak.randint(0, 1, 10, dtype=int64))

        with self.assertRaises(TypeError):
            self.igb.all(ak.randint(0, 1, 10, dtype=float64))

        with self.assertRaises(TypeError):
            self.igb.all(ak.randint(0, 1, 10, dtype=int64))

        with self.assertRaises(TypeError):
            self.igb.min(ak.randint(0, 1, 10, dtype=bool))

        with self.assertRaises(TypeError):
            self.igb.max(ak.randint(0, 1, 10, dtype=bool))

    def test_aggregate_strings(self):
        s = ak.array(["a", "b", "a", "b", "c"])
        i = ak.arange(s.size)
        grouping = ak.GroupBy(s)
        labels, values = grouping.nunique(i)

        actual = {label: value for (label, value) in zip(labels.to_ndarray(), values.to_ndarray())}
        self.assertDictEqual({"a": 2, "b": 2, "c": 1}, actual)

    def test_multi_level_categorical(self):
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        i = ak.arange(string.size)
        expected = {("a", "a"): 2, ("b", "b"): 2, ("c", "c"): 1}

        # list of 2 strings
        str_grouping = ak.GroupBy([string, string])
        str_labels, str_values = str_grouping.nunique(i)
        str_dict = to_tuple_dict(str_labels, str_values)
        self.assertDictEqual(expected, str_dict)

        # list of 2 cats (one from_codes)
        cat_grouping = ak.GroupBy([cat, cat_from_codes])
        cat_labels, cat_values = cat_grouping.nunique(i)
        cat_dict = to_tuple_dict(cat_labels, cat_values)
        self.assertDictEqual(expected, cat_dict)

        # One cat (from_codes) and one string
        mixed_grouping = ak.GroupBy([cat_from_codes, string])
        mixed_labels, mixed_values = mixed_grouping.nunique(i)
        mixed_dict = to_tuple_dict(mixed_labels, mixed_values)
        self.assertDictEqual(expected, mixed_dict)

    def test_nunique_types(self):
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        i = ak.array([5, 3, 5, 3, 1])
        # Try GroupBy.nunique with every combination of types, including mixed
        keys = (string, cat, i, (string, cat, i))
        for key in keys:
            g = ak.GroupBy(key)
            for val in keys:
                k, n = g.nunique(val)
                self.assertListEqual(n.to_list(), [1, 1, 1])

    def test_type_failure_multilevel_groupby_aggregate(self):
        # just checking no error occurs with hotfix for Issue 858
        keys = [ak.randint(0, 10, 100), ak.randint(0, 10, 100)]
        g = ak.GroupBy(keys)
        g.min(ak.randint(0, 10, 100))

    def test_uint64_aggregate(self):
        # reproducer for Issue #1129
        u = ak.cast(ak.arange(100), ak.uint64)
        i = ak.arange(100)
        gu = ak.GroupBy(u)
        gi = ak.GroupBy(i)
        u_keys, u_group_sums = gu.sum(u)
        i_keys, i_group_sums = gi.sum(i)

        self.assertListEqual(u_keys.to_list(), i_keys.to_list())
        self.assertListEqual(u_group_sums.to_list(), i_group_sums.to_list())

        # verify the multidim unsigned version doesnt break
        multi_gu = ak.GroupBy([u, u])

        u_data = ak.array(np.array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4], dtype=np.uint64))
        i_data = ak.array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4])
        labels = ak.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        g = ak.GroupBy(labels)
        u_unique_keys, u_group_nunique = g.nunique(u_data)
        i_unique_keys, i_group_nunique = g.nunique(i_data)
        self.assertListEqual(u_unique_keys.to_list(), i_unique_keys.to_list())
        self.assertListEqual(u_group_nunique.to_list(), i_group_nunique.to_list())

    def test_zero_length_groupby(self):
        """
        This tests groupby boundary condition on a zero length pdarray, see Issue #900 for details
        """
        g = ak.GroupBy(ak.zeros(0, dtype=ak.int64))
        str(g.segments)  # passing condition, if this was deleted it will cause the test to fail

    def test_first_aggregation(self):
        keys = ak.array([0, 1, 0, 1, 0, 1])
        vals = ak.array([9, 8, 7, 6, 5, 4])
        ans = [9, 8]
        g = ak.GroupBy(keys)
        _, res = g.first(vals)
        self.assertListEqual(ans, res.to_list())

    def test_mode_aggregation(self):
        keys = ak.array([0, 1, 0, 1, 0, 1, 0, 1])
        vals = ak.array([4, 3, 5, 3, 5, 2, 6, 2])
        ans = [5, 3]
        g = ak.GroupBy(keys)
        _, res = g.mode(vals)
        self.assertListEqual(ans, res.to_list())
        # Test with multi-array values
        _, res2 = g.mode([vals, vals])
        self.assertListEqual(ans, res2[0].to_list())
        self.assertListEqual(ans, res2[1].to_list())

    def test_unique_aggregation(self):
        keys = ak.array([0, 1, 0, 1, 0, 1, 0, 1])
        vals = ak.array([4, 3, 5, 3, 5, 2, 6, 2])
        ans = [[4, 5, 6], [2, 3]]
        g = ak.GroupBy(keys)
        _, res = g.unique(vals)
        for a, r in zip(ans, res.to_list()):
            self.assertListEqual(a, r)
        # Test with multi-array values
        _, res2 = g.unique([vals, vals])
        for a, r in zip(ans, res2[0].to_list()):
            self.assertListEqual(a, r)
        for a, r in zip(ans, res2[1].to_list()):
            self.assertListEqual(a, r)

    def test_nunique_ordering_bug(self):
        keys = ak.array(["1" for _ in range(8)] + ["2" for _ in range(3)])
        vals = ak.array([str(i) for i in range(8)] + [str(i) for i in range(3)])
        g = ak.GroupBy(keys)
        unique_keys, nuniq = g.nunique(vals)
        expected_unique_keys = ["1", "2"]
        expected_nuniq = [8, 3]
        self.assertListEqual(expected_unique_keys, unique_keys.to_list())
        self.assertListEqual(expected_nuniq, nuniq.to_list())


def to_tuple_dict(labels, values):
    # transforms labels from list of arrays into a list of tuples by index and builds a dictionary
    # labels: [array(['b', 'a', 'c']), array(['b', 'a', 'c'])] -> [('b', 'b'), ('a', 'a'), ('c', 'c')]
    return {
        label: value
        for (label, value) in zip(
            [index_tuple for index_tuple in zip(*[pda.to_ndarray() for pda in labels])],
            values.to_ndarray(),
        )
    }
