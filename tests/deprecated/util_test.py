import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.numpy.util import is_float, is_int, is_numeric, map


class UtilTest(ArkoudaTest):
    def test_sparse_sum_helper(self):
        rng = np.random.default_rng()
        seeds = [rng.choice(2**63), rng.choice(2**63), rng.choice(2**63), rng.choice(2**63)]
        set_seeds = [1000509587142185552, 5931535381009490148, 5631286082363685405, 3867516237354681488]
        set_seeds2 = [8086790153783974714, 2380734683647922470, 2906987507681887800, 4967208342496478642]
        # run three times: with random seeds and with the seeds that previously failed
        for seed1, seed2, seed3, seed4 in [seeds, set_seeds, set_seeds2]:
            cfg = ak.get_config()
            N = (10**3) * cfg["numLocales"]
            select_from = ak.arange(N)
            inds1 = select_from[ak.randint(0, 10, N, seed=seed1) % 3 == 0]
            inds2 = select_from[ak.randint(0, 10, N, seed=seed2) % 3 == 0]
            vals1 = ak.randint(-(2**32), 2**32, inds1.size, seed=seed3)
            vals2 = ak.randint(-(2**32), 2**32, inds2.size, seed=seed4)

            merge_idx, merge_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=True)
            sort_idx, sort_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=False)
            gb_idx, gb_vals = ak.GroupBy(ak.concatenate([inds1, inds2], ordered=False)).sum(
                ak.concatenate((vals1, vals2), ordered=False)
            )

            def are_pdarrays_equal(pda1, pda2):
                # we first check the sizes so that we won't hit shape mismatch
                # before we can print the seed (due to short-circuiting)
                return (pda1.size == pda2.size) and ((pda1 == pda2).all())

            cond = (
                are_pdarrays_equal(merge_idx, sort_idx)
                and are_pdarrays_equal(merge_idx, gb_idx)
                and are_pdarrays_equal(merge_vals, sort_vals)
                and are_pdarrays_equal(merge_vals, gb_vals)
            )
            if not cond:
                print(f"\nnum locales: {cfg['numLocales']}")
                print(f"Failure with seeds:\n{seed1},\n{seed2},\n{seed3},\n{seed4}")
            self.assertTrue(cond)

    def test_is_numeric(self):
        strings = ak.array(["a", "b"])
        ints = ak.array([1, 2])
        categoricals = ak.Categorical(strings)
        floats = ak.array([1, np.nan])

        from arkouda.pandas.series import Series
        from arkouda.pandas.index import Index

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            self.assertFalse(is_numeric(item))

        for item in [ints, Index(ints), Series(ints), floats, Index(floats), Series(floats)]:
            self.assertTrue(is_numeric(item))

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
            floats,
            Index(floats),
            Series(floats),
        ]:
            self.assertFalse(is_int(item))

        for item in [ints, Index(ints), Series(ints)]:
            self.assertTrue(is_int(item))

        for item in [
            strings,
            Index(strings),
            Series(strings),
            ints,
            Index(ints),
            Series(ints),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            self.assertFalse(is_float(item))

        for item in [floats, Index(floats), Series(floats)]:
            self.assertTrue(is_float(item))

    def test_map(self):
        a = ak.array(["1", "1", "4", "4", "4"])
        b = ak.array([2, 3, 2, 3, 4])
        c = ak.array([1.0, 1.0, 2.2, 2.2, 4.4])
        d = ak.Categorical(a)

        result = map(a, {"4": 25, "5": 30, "1": 7})
        self.assertListEqual(result.to_list(), [7, 7, 25, 25, 25])

        result = map(a, {"1": 7})
        self.assertListEqual(
            result.to_list(), ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list()
        )

        result = map(a, {"1": 7.0})
        self.assertTrue(
            np.allclose(result.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)
        )

        result = map(b, {4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        self.assertListEqual(result.to_list(), [30.0, 5.0, 30.0, 5.0, 25.0])

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        self.assertListEqual(result.to_list(), ["a", "a", "b", "b", "c"])

        result = map(c, {1.0: "a"})
        self.assertListEqual(result.to_list(), ["a", "a", "null", "null", "null"])

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        self.assertListEqual(result.to_list(), ["a", "a", "b", "b", "c"])

        result = map(d, {"4": 25, "5": 30, "1": 7})
        self.assertListEqual(result.to_list(), [7, 7, 25, 25, 25])

        result = map(d, {"1": 7})
        self.assertTrue(
            np.allclose(
                result.to_list(),
                ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).to_list(),
            )
        )

        result = map(d, {"1": 7.0})
        self.assertTrue(
            np.allclose(result.to_list(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)
        )
