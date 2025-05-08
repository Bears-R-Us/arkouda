import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.numpy.dtypes import float64, int64
from arkouda.pandas.groupbyclass import GroupByReductionType
from arkouda.scipy import chisquare as akchisquare

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
        print("Doing .size()")
    tests += 1
    pdkeys, pdvals = groupby_to_arrays(df, keyname, "int64", "count", levels)
    akkeys, akvals = akg.size()
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
        self.svalues = ak.cast(self.ivalues, str)
        self.bivalues = ak.cast(self.ivalues, ak.bigint)
        self.igb = ak.GroupBy(self.ivalues)
        self.ugb = ak.GroupBy(self.uvalues)
        self.sgb = ak.GroupBy(self.svalues)
        self.bigb = ak.GroupBy(self.bivalues)

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
        k, ct = g.size()

        self.assertEqual(ct[1], true_ct)
        self.assertListEqual(k.to_list(), [False, True])

        # This test was added since we added the size method for issue #1353
        k, ct = g.size()

        self.assertEqual(ct[1], true_ct)
        self.assertListEqual(k.to_list(), [False, True])

        b = ak.array([False, False, True, False, False])
        g = ak.GroupBy([a, b])
        k, ct = g.size()
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

    def test_empty_segs_broadcast(self):
        # verify the reproducer from issue #3035 gives correct answer
        # note: this was due to a race condition, so it will only appear with multiple locales

        # test with int and bool vals
        for vals in ak.arange(7), (ak.arange(7) % 2 == 0):
            segs = ak.array([3, 3, 5, 6, 6, 7, 7])
            size = 10
            perm = ak.array([9, 1, 0, 5, 7, 2, 8, 4, 3, 6])

            # filter out empty segs
            non_empty_segs = ak.array([False, True, True, False, True, False, True])
            compressed_segs = segs[non_empty_segs]
            compressed_vals = vals[non_empty_segs]

            self.assertEqual(
                ak.broadcast(segs, vals, size).to_list(),
                ak.broadcast(compressed_segs, compressed_vals, size).to_list(),
            )
            self.assertEqual(
                ak.broadcast(segs, vals, size, perm).to_list(),
                ak.broadcast(compressed_segs, compressed_vals, size, perm).to_list(),
            )

    def test_nan_broadcast(self):
        # verify the reproducer from issue #3001 gives correct answer
        # test with int and bool vals
        res = ak.broadcast(
            ak.array([0, 2, 4]), ak.array([np.nan, 5.0, 25.0]), permutation=ak.array([0, 1, 2, 3, 4])
        )
        self.assertTrue(
            np.allclose(res.to_ndarray(), np.array([np.nan, np.nan, 5.0, 5.0, 25.0]), equal_nan=True)
        )

    def test_broadcast_ints(self):
        keys, counts = self.igb.size()

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
        keys, counts = self.ugb.size()
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

    def test_broadcast_strings(self):
        keys, counts = self.sgb.size()
        self.assertListEqual([1, 4, 2, 1, 2], counts.to_list())
        self.assertListEqual(["1", "2", "3", "4", "5"], keys.to_list())

        s_results = self.sgb.broadcast(1 * (counts > 2))
        i_results = self.igb.broadcast(1 * (counts > 2))
        self.assertListEqual(i_results.to_list(), s_results.to_list())

        s_results = self.sgb.broadcast(1 * (counts == 2))
        i_results = self.igb.broadcast(1 * (counts == 2))
        self.assertListEqual(i_results.to_list(), s_results.to_list())

        s_results = self.sgb.broadcast(1 * (counts < 4))
        i_results = self.igb.broadcast(1 * (counts < 4))
        self.assertListEqual(i_results.to_list(), s_results.to_list())

        # test str Groupby.broadcast with and without permute
        s_results = self.sgb.broadcast(ak.array(["1", "2", "6", "8", "9"]), permute=False)
        i_results = self.igb.broadcast(ak.array(["1", "2", "6", "8", "9"]), permute=False)
        self.assertListEqual(i_results.to_list(), s_results.to_list())
        s_results = self.sgb.broadcast(ak.array(["1", "2", "6", "8", "9"]))
        i_results = self.igb.broadcast(ak.array(["1", "2", "6", "8", "9"]))
        self.assertListEqual(i_results.to_list(), s_results.to_list())

    def test_broadcast_bigints(self):
        # use reproducer to verify >64 bits work
        a = ak.arange(3, dtype=ak.bigint)
        a += 2**200
        segs = ak.array([0, 2, 5])
        bi_broad = ak.groupbyclass.broadcast(segs, a, 8)
        indices = ak.broadcast(segs, ak.arange(3), 8)
        self.assertListEqual(bi_broad.to_list(), a[indices].to_list())
        self.assertEqual(bi_broad.max_bits, a.max_bits)

        # verify max_bits is preserved by broadcast
        a.max_bits = 201
        bi_broad = ak.broadcast(segs, a, 8)
        self.assertListEqual(bi_broad.to_list(), a[indices].to_list())
        self.assertEqual(bi_broad.max_bits, a.max_bits)

        # do the same tests as uint and compare the results
        keys, counts = self.bigb.size()
        self.assertListEqual([1, 4, 2, 1, 2], counts.to_list())
        self.assertListEqual([1, 2, 3, 4, 5], keys.to_list())

        u_results = self.ugb.broadcast(1 * (counts > 2))
        bi_results = self.bigb.broadcast(1 * (counts > 2))
        self.assertListEqual(bi_results.to_list(), u_results.to_list())

        u_results = self.ugb.broadcast(1 * (counts == 2))
        bi_results = self.bigb.broadcast(1 * (counts == 2))
        self.assertListEqual(bi_results.to_list(), u_results.to_list())

        u_results = self.ugb.broadcast(1 * (counts < 4))
        bi_results = self.bigb.broadcast(1 * (counts < 4))
        self.assertListEqual(bi_results.to_list(), u_results.to_list())

        # test bigint Groupby.broadcast with and without permute with > 64 bit values
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64), permute=False)
        bi_results = self.bigb.broadcast(
            ak.array([1, 2, 6, 8, 9], dtype=ak.bigint) + 2**200, permute=False
        )
        self.assertListEqual((bi_results - 2**200).to_list(), u_results.to_list())
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64))
        bi_results = self.bigb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.bigint) + 2**200)
        self.assertListEqual((bi_results - 2**200).to_list(), u_results.to_list())

        # test bigint broadcast
        u_results = ak.broadcast(ak.array([0]), ak.array([1], dtype=ak.uint64), 1)
        bi_results = ak.broadcast(ak.array([0]), ak.array([1], dtype=ak.bigint), 1)
        self.assertListEqual(bi_results.to_list(), u_results.to_list())

    def test_broadcast_booleans(self):
        keys, counts = self.igb.size()

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
        keys, counts = self.igb.size()

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

    def test_groupby_count(self):
        a = ak.array([1, 0, -1, 1, -1, -1])
        b0 = ak.array([1, np.nan, 1, 1, np.nan, np.nan])

        dtypes = ["float64", "bool", "int64"]

        gb = ak.GroupBy(a)

        for dt in dtypes:
            b = ak.cast(b0, dt=dt)
            keys, counts = gb.count(b)
            self.assertTrue(np.allclose(keys.to_ndarray(), np.array([-1, 0, 1]), equal_nan=True))

            if dt == "float64":
                self.assertTrue(np.allclose(counts.to_ndarray(), np.array([1, 0, 2]), equal_nan=True))
            else:
                self.assertTrue(np.allclose(counts.to_ndarray(), np.array([3, 1, 2]), equal_nan=True))

        #   Test BigInt separately
        b = ak.array(np.array([1, 0, 1, 1, 0, 0]), dtype="bigint")
        self.assertTrue(np.allclose(keys.to_ndarray(), np.array([-1, 0, 1]), equal_nan=True))
        self.assertTrue(np.allclose(counts.to_ndarray(), np.array([3, 1, 2]), equal_nan=True))

    def test_bigint_groupby(self):
        bi = 2**200
        # these bigint arrays are the int arrays shifted up by 2**200
        a = ak.array([1, 0, -1, 1, 0, -1])
        bi_a = a + bi
        b = ak.full(6, 10, dtype=ak.uint64)
        bi_b = b + bi

        # single level groupby
        int_arrays = [a, b]
        bigint_arrays = [bi_a, bi_b]
        for i_arr, bi_arr in zip(int_arrays, bigint_arrays):
            i_unique, i_counts = ak.GroupBy(i_arr).size()
            bi_unique, bi_counts = ak.GroupBy(bi_arr).size()
            shift_down = ak.cast(bi_unique - bi, ak.int64)
            # order isn't guaranteed so argsort and permute
            i_perm = ak.argsort(i_unique)
            bi_perm = ak.argsort(shift_down)
            self.assertListEqual(i_counts[i_perm].to_list(), bi_counts[bi_perm].to_list())
            self.assertListEqual(i_unique[i_perm].to_list(), shift_down[bi_perm].to_list())

        # multilevel groupby
        (i1_unique, i2_unique), i_counts = ak.GroupBy(int_arrays).size()
        (bi1_unique, bi2_unique), bi_counts = ak.GroupBy(bigint_arrays).size()
        shift_down1 = ak.cast(bi1_unique - bi, ak.int64)
        shift_down2 = ak.cast(bi2_unique - bi, ak.int64)
        # order isn't guaranteed so argsort and permute
        i_perm = ak.coargsort((i1_unique, i2_unique))
        bi_perm = ak.coargsort((shift_down1, shift_down2))
        self.assertListEqual(i_counts[i_perm].to_list(), bi_counts[bi_perm].to_list())
        self.assertListEqual(i1_unique[i_perm].to_list(), shift_down1[bi_perm].to_list())
        self.assertListEqual(i2_unique[i_perm].to_list(), shift_down2[bi_perm].to_list())

        # verify we can groupby bigint with other typed arrays
        mixted_types_arrays = [[bi_a, b], [a, bi_b], [bi_b, a], [b, bi_a]]
        for arrs in mixted_types_arrays:
            ak.GroupBy(arrs).size()

    def test_bigint_groupby_aggregations(self):
        # test equivalent to uint when max_bits=64
        u = ak.cast(ak.arange(10) % 2 + 2**63, ak.uint64)
        bi = ak.cast(u, ak.bigint)
        bi.max_bits = 64
        vals = ak.cast(ak.arange(2**63 - 11, 2**63 - 1), ak.bigint)
        vals.max_bits = 64

        u_gb = ak.GroupBy(u)
        bi_gb = ak.GroupBy(bi)
        aggregations = ["or", "sum", "and", "min", "max", "nunique", "first", "mode", "unique"]
        for agg in aggregations:
            u_res = u_gb.aggregate(vals, agg)
            bi_res = bi_gb.aggregate(vals, agg)
            self.assertListEqual(u_res[0].to_list(), bi_res[0].to_list())
            self.assertListEqual(u_res[1].to_list(), bi_res[1].to_list())

            u_res = u_gb.aggregate(bi, agg)
            bi_res = bi_gb.aggregate(bi, agg)
            self.assertListEqual(u_res[0].to_list(), bi_res[0].to_list())
            self.assertListEqual(u_res[1].to_list(), bi_res[1].to_list())

        # test aggregations with > 64 bits and scale back down
        i = ak.arange(10)
        bi = ak.arange(2**200, 2**200 + 10, max_bits=201)
        revs = ak.arange(10) % 2 == 0
        gb = ak.GroupBy(revs)
        other_aggs = ["or", "and", "min", "max"]
        for agg in other_aggs:
            i_res = gb.aggregate(i, agg)
            bi_res = gb.aggregate(bi, agg)
            self.assertListEqual(i_res[0].to_list(), bi_res[0].to_list())
            self.assertListEqual(i_res[1].to_list(), (bi_res[1] - 2**200).to_list())

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

    def test_large_mean_aggregation(self):
        # reproducer for integer overflow in groupby.mean
        a = ak.full(10, 2**63 - 1, dtype=ak.int64)

        # since all values of a are the same, all means should be 2**63 - 1
        _, means = ak.GroupBy(ak.arange(10) % 3).mean(a)
        for m in means.to_list():
            self.assertTrue(np.isclose(float(a[0]), m))

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

        g = ak.GroupBy(keys)

        weighted_sample = g.sample(
            values, n=num_samples, replace=True, weights=weights, random_state=rng
        )

        # count how many of each category we saw
        uk, f_obs = ak.GroupBy(weighted_sample).size()

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
                        g = ak.GroupBy(grouping_keys)
                        choice_arrays.append(
                            g.sample(a, n=size, replace=replace, weights=p, random_state=rng)
                        )
                        choice_arrays.append(
                            g.sample(a, frac=(size / 4), replace=replace, weights=p, random_state=rng)
                        )

        # reset generator to ensure we get the same arrays
        rng = ak.random.default_rng(seed)
        for a in a_vals:
            for size in 2, 4:
                for replace in True, False:
                    for p in [None, weights]:
                        previous1 = choice_arrays.pop(0)
                        previous2 = choice_arrays.pop(0)
                        g = ak.GroupBy(grouping_keys)
                        current1 = g.sample(a, n=size, replace=replace, weights=p, random_state=rng)
                        current2 = g.sample(
                            a, frac=(size / 4), replace=replace, weights=p, random_state=rng
                        )

                        res = np.allclose(previous1.to_list(), current1.to_list()) and np.allclose(
                            previous2.to_list(), current2.to_list()
                        )
                        if not res:
                            print(f"\nnum locales: {cfg['numLocales']}")
                            print(f"Failure with seed:\n{seed}")
                        self.assertTrue(res)

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
