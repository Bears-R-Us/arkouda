import numpy as np
import pandas as pd
import pytest

import arkouda as ak
from arkouda.groupbyclass import GroupByReductionType


def to_tuple_dict(labels, values):
    # transforms labels from list of arrays into a list of tuples by index and builds a dictionary
    # labels: [array(['b', 'a', 'c']), array(['b', 'a', 'c'])] -> [('b', 'b'), ('a', 'a'), ('c', 'c')]
    return dict(zip(list(zip(*[pda.to_ndarray() for pda in labels])), values.to_ndarray()))


class TestGroupBy:
    GROUPS = 8
    LEVELS = [1, 2]
    OPS = list(ak.GroupBy.Reductions)
    OPS.append("count")
    NAN_OPS = frozenset(["mean", "min", "max", "sum", "prod"])

    @classmethod
    def setup_class(cls):
        cls.bvalues = ak.randint(0, 1, 10, dtype=bool)
        cls.fvalues = ak.randint(0, 1, 10, dtype=float)
        cls.ivalues = ak.array([4, 1, 3, 2, 2, 2, 5, 5, 2, 3])
        cls.uvalues = ak.cast(cls.ivalues, ak.uint64)
        cls.svalues = ak.cast(cls.ivalues, str)
        cls.bivalues = ak.cast(cls.ivalues, ak.bigint)
        cls.igb = ak.GroupBy(cls.ivalues)
        cls.ugb = ak.GroupBy(cls.uvalues)
        cls.sgb = ak.GroupBy(cls.svalues)
        cls.bigb = ak.GroupBy(cls.bivalues)

    def make_arrays(self, size):
        keys = np.random.randint(0, self.GROUPS, size, dtype=np.uint64)
        keys2 = np.random.randint(0, self.GROUPS, size)
        i = np.random.randint(0, size // self.GROUPS, size)
        u = np.random.randint(0, size // self.GROUPS, size, dtype=np.uint64)
        f = np.random.randn(size)  # normally dist random numbers
        b = (i % 2) == 0
        d = {"keys": keys, "keys2": keys2, "int64": i, "uint64": u, "float64": f, "bool": b}

        return d

    def make_arrays_nan(self, size):
        keys = np.random.randint(0, self.GROUPS, size)
        f = np.random.randn(size)

        for i in range(size):
            if np.random.rand() < 0.2:
                f[i] = np.nan
        d = {"keys": keys, "float64": f}

        return d

    def groupby_to_arrays(self, df: pd.DataFrame, kname, vname, op, levels):
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

    def compare_keys(self, pdkeys, akkeys, levels, pdvals, akvals):
        """
        Compares the numpy and arkouda arrays via the numpy.allclose method with the
        default relative and absolute tolerances, returning 0 if the arrays are similar
        element-wise within the tolerances, 1 if they are dissimilar.element

        :return: 0 (identical) or 1 (dissimilar)
        :rtype: int
        """
        if levels == 1:
            assert np.allclose(pdkeys, akkeys.to_ndarray())  # key validation
        else:
            for lvl in range(levels):
                assert np.allclose(pdkeys[lvl], akkeys[lvl].to_ndarray())

        assert np.allclose(pdvals, akvals.to_ndarray(), equal_nan=True)  # value validation

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("levels", LEVELS)
    @pytest.mark.parametrize("op", OPS)
    def test_pandas_equivalency(self, size, levels, op):
        data = self.make_arrays(size)
        df = pd.DataFrame(data)
        akdf = {k: ak.array(v) for k, v in data.items()}
        if levels == 1:
            akg = ak.GroupBy(akdf["keys"])
            keyname = "keys"
        elif levels == 2:
            akg = ak.GroupBy([akdf["keys"], akdf["keys2"]])
            keyname = ["keys", "keys2"]

        for vname in ("int64", "uint64", "float64", "bool"):
            if op == "count":
                print(f"Doing .count() - {vname}")
            else:
                print(f"\nDoing aggregate({vname}, {op})")

            do_check = True
            try:
                pdkeys, pdvals = self.groupby_to_arrays(df, keyname, vname, op, levels)
            except Exception:
                print("Pandas does not implement")
                do_check = False
            try:
                akkeys, akvals = akg.count() if op == "count" else akg.aggregate(akdf[vname], op)
            except Exception as E:
                print("Arkouda error: ", E)
                continue # skip check
            if do_check:
                if op.startswith("arg"):
                    pdextrema = df[vname][pdvals]
                    akextrema = akdf[vname][akvals].to_ndarray()
                    # check so we can get meaningful output if needed
                    if not np.allclose(pdextrema, akextrema):
                        print("Different argmin/argmax: Arkouda failed to find an extremum")
                        print("pd: ", pdextrema)
                        print("ak: ", akextrema)
                    assert np.allclose(pdextrema, akextrema)
                else:
                    if op != "unique":
                        self.compare_keys(pdkeys, akkeys, levels, pdvals, akvals)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("op", NAN_OPS)
    def test_pandas_equivalency_nan(self, size, op):
        d = self.make_arrays_nan(size)
        df = pd.DataFrame(d)
        akdf = {k: ak.array(v) for k, v in d.items()}

        akg = ak.GroupBy(akdf["keys"])
        keyname = "keys"

        print(f"\nDoing aggregate(float64, {op})")

        do_check = True
        try:
            pdkeys, pdvals = self.groupby_to_arrays(df, keyname, "float64", op, 1)
        except Exception:
            print("Pandas does not implement")
            do_check = False
        try:
            akkeys, akvals = akg.aggregate(akdf["float64"], op, True)
        except RuntimeError as E:
            print("Arkouda error: ", E)
            do_check = False
        if do_check:
            for i in range(pdvals.size):
                if np.isnan(pdvals[i]):
                    pdvals[i] = 0.0  # clear out any nans to match ak implementation
            self.compare_keys(pdkeys, akkeys, 1, pdvals, akvals)

    def test_argmax_argmin(self):
        b = ak.array([True, False, True, True, False, True])
        x = ak.array([True, True, False, True, False, False])
        g = ak.GroupBy(x)
        keys, locs = g.argmin(b)
        assert keys.to_list() == [False, True]
        assert locs.to_list() == [4, 1]

        keys, locs = g.argmax(b)
        assert keys.to_list() == [False, True]
        assert locs.to_list() == [2, 0]

    def test_boolean_arrays(self):
        a = ak.array([True, False, True, True, False])
        true_ct = a.sum()
        g = ak.GroupBy(a)
        k, ct = g.count()

        assert ct[1] == true_ct
        assert k.to_list() == [False, True]

        # This test was added since we added the size method for issue #1353
        k, ct = g.size()

        assert ct[1] == true_ct
        assert k.to_list() == [False, True]

        b = ak.array([False, False, True, False, False])
        g = ak.GroupBy([a, b])
        k, ct = g.count()
        assert ct.to_list() == [2, 2, 1]
        assert k[0].to_list() == [False, True, True]
        assert k[1].to_list() == [False, False, True]

    def test_bitwise_aggregations(self):
        revs = ak.arange(self.igb.length) % 2
        assert self.igb.OR(revs)[1].to_list() == self.igb.max(revs)[1].to_list()
        assert self.igb.AND(revs)[1].to_list() == self.igb.min(revs)[1].to_list()
        assert self.igb.XOR(revs)[1].to_list() == (self.igb.sum(revs)[1] % 2).to_list()

    def test_standalone_broadcast(self):
        segs = ak.arange(10) ** 2
        vals = ak.arange(10)
        size = 100
        check = ((2 * vals + 1) * vals).sum()
        assert ak.broadcast(segs, vals, size=size).sum() == check
        perm = ak.arange(99, -1, -1)
        bcast = ak.broadcast(segs, vals, permutation=perm)
        assert bcast.sum() == check
        assert (bcast[:-1] >= bcast[1:]).all()

    def test_count(self):
        keys, counts = self.igb.count()

        assert [1, 2, 3, 4, 5] == keys.to_list()
        assert [1, 4, 2, 1, 2] == counts.to_list()

    def test_broadcast_ints(self):
        keys, counts = self.igb.count()

        results = self.igb.broadcast(1 * (counts > 2), permute=False)
        assert [0, 1, 1, 1, 1, 0, 0, 0, 0, 0] == results.to_list()

        results = self.igb.broadcast(1 * (counts == 2), permute=False)
        assert [0, 0, 0, 0, 0, 1, 1, 0, 1, 1] == results.to_list()

        results = self.igb.broadcast(1 * (counts < 4), permute=False)
        assert [1, 0, 0, 0, 0, 1, 1, 1, 1, 1] == results.to_list()

        results = self.igb.broadcast(1 * (counts > 2))
        assert [0, 0, 0, 1, 1, 1, 0, 0, 1, 0] == results.to_list()

        results = self.igb.broadcast(1 * (counts == 2))
        assert [0, 0, 1, 0, 0, 0, 1, 1, 0, 1] == results.to_list()

        results = self.igb.broadcast(1 * (counts < 4))
        assert [1, 1, 1, 0, 0, 0, 1, 1, 0, 1] == results.to_list()

    def test_broadcast_uints(self):
        keys, counts = self.ugb.count()
        assert [1, 4, 2, 1, 2] == counts.to_list()
        assert [1, 2, 3, 4, 5] == keys.to_list()

        u_results = self.ugb.broadcast(1 * (counts > 2))
        i_results = self.igb.broadcast(1 * (counts > 2))
        assert i_results.to_list() == u_results.to_list()

        u_results = self.ugb.broadcast(1 * (counts == 2))
        i_results = self.igb.broadcast(1 * (counts == 2))
        assert i_results.to_list() == u_results.to_list()

        u_results = self.ugb.broadcast(1 * (counts < 4))
        i_results = self.igb.broadcast(1 * (counts < 4))
        assert i_results.to_list() == u_results.to_list()

        # test uint Groupby.broadcast with and without permute
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64), permute=False)
        i_results = self.igb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64), permute=False)
        assert i_results.to_list() == u_results.to_list()
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64))
        i_results = self.igb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64))
        assert i_results.to_list() == u_results.to_list()

        # test uint broadcast
        u_results = ak.broadcast(ak.array([0]), ak.array([1], dtype=ak.uint64), 1)
        i_results = ak.broadcast(ak.array([0]), ak.array([1]), 1)
        assert i_results.to_list() == u_results.to_list()

    def test_broadcast_strings(self):
        keys, counts = self.sgb.count()
        assert [1, 4, 2, 1, 2] == counts.to_list()
        assert ["1", "2", "3", "4", "5"] == keys.to_list()

        s_results = self.sgb.broadcast(1 * (counts > 2))
        i_results = self.igb.broadcast(1 * (counts > 2))
        assert i_results.to_list() == s_results.to_list()

        s_results = self.sgb.broadcast(1 * (counts == 2))
        i_results = self.igb.broadcast(1 * (counts == 2))
        assert i_results.to_list() == s_results.to_list()

        s_results = self.sgb.broadcast(1 * (counts < 4))
        i_results = self.igb.broadcast(1 * (counts < 4))
        assert i_results.to_list() == s_results.to_list()

        # test str Groupby.broadcast with and without permute
        s_results = self.sgb.broadcast(ak.array(["1", "2", "6", "8", "9"]), permute=False)
        i_results = self.igb.broadcast(ak.array(["1", "2", "6", "8", "9"]), permute=False)
        assert i_results.to_list() == s_results.to_list()
        s_results = self.sgb.broadcast(ak.array(["1", "2", "6", "8", "9"]))
        i_results = self.igb.broadcast(ak.array(["1", "2", "6", "8", "9"]))
        assert i_results.to_list() == s_results.to_list()

    def test_broadcast_bigints(self):
        # use reproducer to verify >64 bits work
        a = ak.arange(3, dtype=ak.bigint)
        a += 2**200
        segs = ak.array([0, 2, 5])
        bi_broad = ak.groupbyclass.broadcast(segs, a, 8)
        indices = ak.broadcast(segs, ak.arange(3), 8)
        assert bi_broad.to_list() == a[indices].to_list()
        assert bi_broad.max_bits == a.max_bits

        # verify max_bits is preserved by broadcast
        a.max_bits = 201
        bi_broad = ak.broadcast(segs, a, 8)
        assert bi_broad.to_list() == a[indices].to_list()
        assert bi_broad.max_bits == a.max_bits

        # do the same tests as uint and compare the results
        keys, counts = self.bigb.count()
        assert [1, 4, 2, 1, 2] == counts.to_list()
        assert [1, 2, 3, 4, 5] == keys.to_list()

        u_results = self.ugb.broadcast(1 * (counts > 2))
        bi_results = self.bigb.broadcast(1 * (counts > 2))
        assert bi_results.to_list() == u_results.to_list()

        u_results = self.ugb.broadcast(1 * (counts == 2))
        bi_results = self.bigb.broadcast(1 * (counts == 2))
        assert bi_results.to_list() == u_results.to_list()

        u_results = self.ugb.broadcast(1 * (counts < 4))
        bi_results = self.bigb.broadcast(1 * (counts < 4))
        assert bi_results.to_list() == u_results.to_list()

        # test bigint Groupby.broadcast with and without permute with > 64 bit values
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64), permute=False)
        bi_results = self.bigb.broadcast(
            ak.array([1, 2, 6, 8, 9], dtype=ak.bigint) + 2**200, permute=False
        )
        assert (bi_results - 2**200).to_list() == u_results.to_list()
        u_results = self.ugb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.uint64))
        bi_results = self.bigb.broadcast(ak.array([1, 2, 6, 8, 9], dtype=ak.bigint) + 2**200)
        assert (bi_results - 2**200).to_list() == u_results.to_list()

        # test bigint broadcast
        u_results = ak.broadcast(ak.array([0]), ak.array([1], dtype=ak.uint64), 1)
        bi_results = ak.broadcast(ak.array([0]), ak.array([1], dtype=ak.bigint), 1)
        assert bi_results.to_list() == u_results.to_list()

    def test_broadcast_booleans(self):
        keys, counts = self.igb.count()

        results = self.igb.broadcast(counts > 2, permute=False)
        assert [0, 1, 1, 1, 1, 0, 0, 0, 0, 0] == results.to_list()

        results = self.igb.broadcast(counts == 2, permute=False)
        assert [0, 0, 0, 0, 0, 1, 1, 0, 1, 1] == results.to_list()

        results = self.igb.broadcast(counts < 4, permute=False)
        assert [1, 0, 0, 0, 0, 1, 1, 1, 1, 1] == results.to_list()

        results = self.igb.broadcast(counts > 2)
        assert [0, 0, 0, 1, 1, 1, 0, 0, 1, 0] == results.to_list()

        results = self.igb.broadcast(counts == 2)
        assert [0, 0, 1, 0, 0, 0, 1, 1, 0, 1] == results.to_list()

        results = self.igb.broadcast(counts < 4)
        assert [1, 1, 1, 0, 0, 0, 1, 1, 0, 1] == results.to_list()

    def test_groupby_reduction_type(self):
        assert "any" == str(GroupByReductionType.ANY)
        assert "all" == str(GroupByReductionType.ALL)
        assert GroupByReductionType.ANY == GroupByReductionType("any")

        with pytest.raises(ValueError):
            GroupByReductionType("an")

        assert isinstance(ak.GROUPBY_REDUCTION_TYPES, frozenset)
        assert "any" in ak.GROUPBY_REDUCTION_TYPES

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_error_handling(self, size):
        d = self.make_arrays(size)
        akdf = {k: ak.array(v) for k, v in d.items()}
        gb = ak.GroupBy([akdf["keys"], akdf["keys2"]])

        with pytest.raises(TypeError):
            ak.GroupBy(ak.arange(4), ak.arange(4))

        with pytest.raises(TypeError):
            ak.GroupBy(self.fvalues)

        with pytest.raises(TypeError):
            gb.broadcast([])

        with pytest.raises(TypeError):
            self.igb.nunique(ak.randint(0, 1, 10, dtype=ak.float64))

        with pytest.raises(TypeError):
            self.igb.any(ak.randint(0, 1, 10, dtype=ak.float64))

        with pytest.raises(TypeError):
            self.igb.any(ak.randint(0, 1, 10, dtype=ak.int64))

        with pytest.raises(TypeError):
            self.igb.all(ak.randint(0, 1, 10, dtype=ak.float64))

        with pytest.raises(TypeError):
            self.igb.all(ak.randint(0, 1, 10, dtype=ak.int64))

        with pytest.raises(TypeError):
            self.igb.min(ak.randint(0, 1, 10, dtype=bool))

        with pytest.raises(TypeError):
            self.igb.max(ak.randint(0, 1, 10, dtype=bool))

    def test_aggregate_strings(self):
        s = ak.array(["a", "b", "a", "b", "c"])
        i = ak.arange(s.size)
        grouping = ak.GroupBy(s)
        labels, values = grouping.nunique(i)

        actual = {label: value for (label, value) in zip(labels.to_ndarray(), values.to_ndarray())}
        assert {"a": 2, "b": 2, "c": 1} == actual

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
        assert expected == str_dict

        # list of 2 cats (one from_codes)
        cat_grouping = ak.GroupBy([cat, cat_from_codes])
        cat_labels, cat_values = cat_grouping.nunique(i)
        cat_dict = to_tuple_dict(cat_labels, cat_values)
        assert expected == cat_dict

        # One cat (from_codes) and one string
        mixed_grouping = ak.GroupBy([cat_from_codes, string])
        mixed_labels, mixed_values = mixed_grouping.nunique(i)
        mixed_dict = to_tuple_dict(mixed_labels, mixed_values)
        assert expected == mixed_dict

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
                assert n.to_list() == [1, 1, 1]

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

        assert u_keys.to_list() == i_keys.to_list()
        assert u_group_sums.to_list() == i_group_sums.to_list()

        # verify the multidim unsigned version doesnt break
        multi_gu = ak.GroupBy([u, u])

        u_data = ak.array(np.array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4], dtype=np.uint64))
        i_data = ak.array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4])
        labels = ak.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        g = ak.GroupBy(labels)
        u_unique_keys, u_group_nunique = g.nunique(u_data)
        i_unique_keys, i_group_nunique = g.nunique(i_data)
        assert u_unique_keys.to_list() == i_unique_keys.to_list()
        assert u_group_nunique.to_list() == i_group_nunique.to_list()

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
            i_unique, i_counts = ak.GroupBy(i_arr).count()
            bi_unique, bi_counts = ak.GroupBy(bi_arr).count()
            shift_down = ak.cast(bi_unique - bi, ak.int64)
            # order isn't guaranteed so argsort and permute
            i_perm = ak.argsort(i_unique)
            bi_perm = ak.argsort(shift_down)
            assert i_counts[i_perm].to_list() == bi_counts[bi_perm].to_list()
            assert i_unique[i_perm].to_list() == shift_down[bi_perm].to_list()

        # multilevel groupby
        (i1_unique, i2_unique), i_counts = ak.GroupBy(int_arrays).count()
        (bi1_unique, bi2_unique), bi_counts = ak.GroupBy(bigint_arrays).count()
        shift_down1 = ak.cast(bi1_unique - bi, ak.int64)
        shift_down2 = ak.cast(bi2_unique - bi, ak.int64)
        # order isn't guaranteed so argsort and permute
        i_perm = ak.coargsort((i1_unique, i2_unique))
        bi_perm = ak.coargsort((shift_down1, shift_down2))
        assert i_counts[i_perm].to_list() == bi_counts[bi_perm].to_list()
        assert i1_unique[i_perm].to_list() == shift_down1[bi_perm].to_list()
        assert i2_unique[i_perm].to_list() == shift_down2[bi_perm].to_list()

        # verify we can groupby bigint with other typed arrays
        mixted_types_arrays = [[bi_a, b], [a, bi_b], [bi_b, a], [b, bi_a]]
        for arrs in mixted_types_arrays:
            ak.GroupBy(arrs).count()

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
            assert u_res[0].to_list() == bi_res[0].to_list()
            assert u_res[1].to_list() == bi_res[1].to_list()

            u_res = u_gb.aggregate(bi, agg)
            bi_res = bi_gb.aggregate(bi, agg)
            assert u_res[0].to_list() == bi_res[0].to_list()
            assert u_res[1].to_list() == bi_res[1].to_list()

        # test aggregations with > 64 bits and scale back down
        i = ak.arange(10)
        bi = ak.arange(2**200, 2**200 + 10, max_bits=201)
        revs = ak.arange(10) % 2 == 0
        gb = ak.GroupBy(revs)
        other_aggs = ["or", "and", "min", "max"]
        for agg in other_aggs:
            i_res = gb.aggregate(i, agg)
            bi_res = gb.aggregate(bi, agg)
            assert i_res[0].to_list() == bi_res[0].to_list()
            assert i_res[1].to_list() == (bi_res[1] - 2**200).to_list()

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
        assert ans == res.to_list()

    def test_mode_aggregation(self):
        keys = ak.array([0, 1, 0, 1, 0, 1, 0, 1])
        vals = ak.array([4, 3, 5, 3, 5, 2, 6, 2])
        ans = [5, 3]
        g = ak.GroupBy(keys)
        _, res = g.mode(vals)
        assert ans == res.to_list()
        # Test with multi-array values
        _, res2 = g.mode([vals, vals])
        assert ans == res2[0].to_list()
        assert ans == res2[1].to_list()

    def test_large_mean_aggregation(self):
        # reproducer for integer overflow in groupby.mean
        a = ak.full(10, 2**63 - 1, dtype=ak.int64)

        # since all values of a are the same, all means should be 2**63 - 1
        _, means = ak.GroupBy(ak.arange(10) % 3).mean(a)
        for m in means.to_list():
            assert np.isclose(float(a[0]), m)

    def test_unique_aggregation(self):
        keys = ak.array([0, 1, 0, 1, 0, 1, 0, 1])
        vals = ak.array([4, 3, 5, 3, 5, 2, 6, 2])
        ans = [[4, 5, 6], [2, 3]]
        g = ak.GroupBy(keys)
        _, res = g.unique(vals)
        for a, r in zip(ans, res.to_list()):
            assert a == r
        # Test with multi-array values
        _, res2 = g.unique([vals, vals])
        for a, r in zip(ans, res2[0].to_list()):
            assert a == r
        for a, r in zip(ans, res2[1].to_list()):
            assert a == r

    def test_nunique_ordering_bug(self):
        keys = ak.array(["1" for _ in range(8)] + ["2" for _ in range(3)])
        vals = ak.array([str(i) for i in range(8)] + [str(i) for i in range(3)])
        g = ak.GroupBy(keys)
        unique_keys, nuniq = g.nunique(vals)
        expected_unique_keys = ["1", "2"]
        expected_nuniq = [8, 3]
        assert expected_unique_keys == unique_keys.to_list()
        assert expected_nuniq == nuniq.to_list()
