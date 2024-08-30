import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

SIZE = 10
OPS = frozenset(["intersect1d", "union1d", "setxor1d", "setdiff1d"])

TYPES = ("int64", "uint64")


def make_arrays(dtype):
    if dtype == "int64":
        a = ak.randint(0, SIZE, SIZE)
        b = ak.randint(SIZE / 2, 2 * SIZE, SIZE)
        return a, b
    elif dtype == "uint64":
        a = ak.randint(0, SIZE, SIZE, dtype=ak.uint64)
        b = ak.randint(SIZE / 2, 2 * SIZE, SIZE, dtype=ak.uint64)
        return a, b


def compare_results(akvals, npvals) -> int:
    """
    Compares the numpy and arkouda arrays via the numpy.allclose method with the
    default relative and absolute tolerances, returning 0 if the arrays are similar
    element-wise within the tolerances, 1 if they are dissimilar.element

    :return: 0 (identical) or 1 (dissimilar)
    :rtype: int
    """
    akvals = akvals.to_ndarray()

    if not np.array_equal(akvals, npvals):
        akvals = ak.array(akvals)
        npvals = ak.array(npvals)
        innp = npvals[
            ak.in1d(ak.array(npvals), ak.array(akvals), True)
        ]  # values in np array, but not ak array
        inak = akvals[
            ak.in1d(ak.array(akvals), ak.array(npvals), True)
        ]  # values in ak array, not not np array
        print(f"(values in np but not ak: {innp}) (values in ak but not np: {inak})")
        return 1
    return 0


def run_test(verbose=True):
    """
    The run_test method enables execution of the set operations
    intersect1d, union1d, setxor1d, and setdiff1d
    on a randomized set of arrays.
    :return:
    """
    tests = 0
    failures = 0
    not_impl = 0

    for dtype in TYPES:
        aka, akb = make_arrays(dtype)
        for op in OPS:
            tests += 1
            do_check = True
            try:
                fxn = getattr(ak, op)
                akres = fxn(aka, akb)
                fxn = getattr(np, op)
                npres = fxn(aka.to_ndarray(), akb.to_ndarray())
            except RuntimeError as E:
                if verbose:
                    print("Arkouda error: ", E)
                not_impl += 1
                do_check = False
                continue
            if not do_check:
                continue
            failures += compare_results(akres, npres)

    return failures


class SetOpsTest(ArkoudaTest):
    def test_setops(self):
        """
        Executes run_test and asserts whether there are any errors

        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        """
        self.assertEqual(0, run_test())

    def testSetxor1d(self):
        pdaOne = ak.array([1, 2, 3, 2, 4])
        pdaTwo = ak.array([2, 3, 5, 7, 5])

        self.assertListEqual([1, 4, 5, 7], ak.setxor1d(pdaOne, pdaTwo).to_list())

    def testSetxor1d_Multi(self):
        # Test Numeric pdarray
        a = [1, 2, 3, 4, 5]
        b = [1, 5, 2, 3, 4]
        c = [1, 3, 2, 5, 4]
        a1 = ak.array(a)
        a2 = ak.array(a)
        b1 = ak.array(b)
        b2 = ak.array(c)

        la = set([(x, y) for x, y in zip(a, a)])
        lb = set([(x, y) for x, y in zip(b, c)])
        lr = list(sorted(la.symmetric_difference(lb)))
        npr0, npr1 = map(list, zip(*lr))

        # Testing
        t = ak.setxor1d([a1, a2], [b1, b2])
        self.assertListEqual(t[0].to_list(), npr0)
        self.assertListEqual(t[1].to_list(), npr1)

        # Testing tuple input
        t = ak.setxor1d((a1, a2), (b1, b2))
        self.assertListEqual(t[0].to_list(), npr0)
        self.assertListEqual(t[1].to_list(), npr1)

        # Test for strings
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["abc", "def"]
        d = ["000", "456"]
        a1 = ak.array(a)
        a2 = ak.array(b)
        b1 = ak.array(c)
        b2 = ak.array(d)
        t = ak.setxor1d([a1, a2], [b1, b2])
        self.assertListEqual(["abc", "abc"], t[0].to_list())
        self.assertListEqual(["000", "123"], t[1].to_list())

        # Test for Categorical
        cat_a1 = ak.Categorical(a1)
        cat_a2 = ak.Categorical(a2)
        cat_b1 = ak.Categorical(b1)
        cat_b2 = ak.Categorical(b2)
        t = ak.setxor1d([cat_a1, cat_a2], [cat_b1, cat_b2])
        self.assertListEqual(["abc", "abc"], t[0].to_list())
        self.assertListEqual(["000", "123"], t[1].to_list())

    def testSetdiff1d(self):
        pdaOne = ak.array([1, 2, 3, 2, 4, 1])
        pdaTwo = ak.array([3, 4, 5, 6])

        self.assertListEqual([1, 2], ak.setdiff1d(pdaOne, pdaTwo).to_list())

        with self.assertRaises(RuntimeError):
            ak.setdiff1d(ak.array([True, False, True]), ak.array([True, True]))

    def testSetDiff1d_Multi(self):
        # Test for numeric pdarray
        a = [1, 2, 3, 4, 5]
        b = [1, 5, 2, 3, 4]
        c = [1, 3, 2, 5, 4]
        a1 = ak.array(a)
        a2 = ak.array(a)
        b1 = ak.array(b)
        b2 = ak.array(c)

        la = set([(x, y) for x, y in zip(a, a)])
        lb = set([(x, y) for x, y in zip(b, c)])
        lr = list(sorted(la.difference(lb)))
        npr0, npr1 = map(list, zip(*lr))

        t = ak.setdiff1d([a1, a2], [b1, b2])
        self.assertListEqual(t[0].to_list(), npr0)
        self.assertListEqual(t[1].to_list(), npr1)

        # Test for strings
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["abc", "def"]
        d = ["000", "456"]
        a1 = ak.array(a)
        a2 = ak.array(b)
        b1 = ak.array(c)
        b2 = ak.array(d)
        t = ak.setdiff1d([a1, a2], [b1, b2])
        self.assertListEqual(["abc"], t[0].to_list())
        self.assertListEqual(["123"], t[1].to_list())

        # Test for Categorical
        cat_a1 = ak.Categorical(a1)
        cat_a2 = ak.Categorical(a2)
        cat_b1 = ak.Categorical(b1)
        cat_b2 = ak.Categorical(b2)
        t = ak.setdiff1d([cat_a1, cat_a2], [cat_b1, cat_b2])
        self.assertListEqual(["abc"], t[0].to_list())
        self.assertListEqual(["123"], t[1].to_list())

    def testIntersect1d(self):
        pdaOne = ak.array([1, 3, 4, 3])
        pdaTwo = ak.array([3, 1, 2, 1])
        self.assertListEqual([1, 3], ak.intersect1d(pdaOne, pdaTwo).to_list())

    def testIntersect1d_Multi(self):
        # Test for numeric
        a = [1, 2, 3, 4, 5]
        b = [1, 5, 2, 3, 4]
        c = [1, 3, 2, 5, 4]
        a1 = ak.array(a)
        a2 = ak.array(a)
        b1 = ak.array(b)
        b2 = ak.array(c)

        la = set([(x, y) for x, y in zip(a, a)])
        lb = set([(x, y) for x, y in zip(b, c)])
        lr = list(sorted(la.intersection(lb)))
        npr0, npr1 = map(list, zip(*lr))

        t = ak.intersect1d([a1, a2], [b1, b2])
        self.assertListEqual(t[0].to_list(), npr0)
        self.assertListEqual(t[1].to_list(), npr1)

        # Test for strings
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["abc", "def"]
        d = ["000", "456"]
        a1 = ak.array(a)
        a2 = ak.array(b)
        b1 = ak.array(c)
        b2 = ak.array(d)
        t = ak.intersect1d([a1, a2], [b1, b2])
        self.assertListEqual(["def"], t[0].to_list())
        self.assertListEqual(["456"], t[1].to_list())

        # Test for Categorical
        cat_a1 = ak.Categorical(a1)
        cat_a2 = ak.Categorical(a2)
        cat_b1 = ak.Categorical(b1)
        cat_b2 = ak.Categorical(b2)
        t = ak.intersect1d([cat_a1, cat_a2], [cat_b1, cat_b2])
        self.assertListEqual(["def"], t[0].to_list())
        self.assertListEqual(["456"], t[1].to_list())

    def testUnion1d(self):
        pdaOne = ak.array([-1, 0, 1])
        pdaTwo = ak.array([-2, 0, 2])
        self.assertListEqual([-2, -1, 0, 1, 2], ak.union1d(pdaOne, pdaTwo).to_list())

    def testUnion1d_Multi(self):
        # test for numeric
        a = [1, 2, 3, 4, 5]
        b = [1, 5, 2, 3, 4]
        c = [1, 3, 2, 5, 4]
        a1 = ak.array(a)
        a2 = ak.array(a)
        b1 = ak.array(b)
        b2 = ak.array(c)

        la = set([(x, y) for x, y in zip(a, a)])
        lb = set([(x, y) for x, y in zip(b, c)])
        lr = list(sorted(la.union(lb)))
        npr0, npr1 = map(list, zip(*lr))
        t = ak.union1d([a1, a2], [b1, b2])
        self.assertListEqual(t[0].to_list(), npr0)
        self.assertListEqual(t[1].to_list(), npr1)

        # Test for Strings
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["xyz"]
        d = ["0"]
        a1 = ak.array(a)
        a2 = ak.array(b)
        b1 = ak.array(c)
        b2 = ak.array(d)
        t = ak.union1d([a1, a2], [b1, b2])
        self.assertListEqual(["abc", "def", "xyz"], t[0].to_list())
        self.assertListEqual(["123", "456", "0"], t[1].to_list())

        # Test for Categorical
        cat_a1 = ak.Categorical(a1)
        cat_a2 = ak.Categorical(a2)
        cat_b1 = ak.Categorical(b1)
        cat_b2 = ak.Categorical(b2)
        t = ak.union1d([cat_a1, cat_a2], [cat_b1, cat_b2])
        self.assertListEqual(["abc", "def", "xyz"], t[0].to_list())
        self.assertListEqual(["123", "456", "0"], t[1].to_list())

    def testIn1d(self):
        pdaOne = ak.array([-1, 0, 1, 3])
        pdaTwo = ak.array([-1, 2, 2, 3])
        bi_one = pdaOne + 2**200
        bi_two = pdaTwo + 2**200
        ans = [True, False, False, True]
        self.assertListEqual(ak.in1d(pdaOne, pdaTwo).to_list(), ans)
        # test bigint
        self.assertListEqual(ak.in1d(bi_one, bi_two).to_list(), ans)
        # test multilevel mixed types (int and bigint)
        self.assertListEqual(ak.in1d([pdaOne, bi_one], [pdaTwo, bi_two]).to_list(), ans)
        self.assertListEqual(ak.in1d([bi_one, pdaOne], [bi_two, pdaTwo]).to_list(), ans)

        stringsOne = ak.array(["String {}".format(i % 3) for i in range(10)])
        stringsTwo = ak.array(["String {}".format(i % 2) for i in range(10)])
        self.assertListEqual([(x % 3) < 2 for x in range(10)], ak.in1d(stringsOne, stringsTwo).to_list())

        # adding tests for unique dtypes
        a = ak.arange(10)
        b = ak.arange(5, 15)

        ip1 = ak.ip_address(a)
        ip2 = ak.ip_address(b)
        self.assertListEqual([x >= 5 for x in range(10)], ak.in1d(ip1, ip2).to_list())

        dt1 = ak.Datetime(a)
        dt2 = ak.Datetime(b)
        self.assertListEqual([x >= 5 for x in range(10)], ak.in1d(dt1, dt2).to_list())

        f1 = ak.Fields(a, names="ABCD")
        f2 = ak.Fields(b, names="ABCD")
        self.assertListEqual([x >= 5 for x in range(10)], ak.in1d(f1, f2).to_list())

    def test_multiarray_validation(self):
        x = [ak.arange(3), ak.arange(3), ak.arange(3)]
        y = [ak.arange(2), ak.arange(2)]
        with self.assertRaises(ValueError):
            ak.pdarraysetops.multiarray_setop_validation(x, y)

        x = [ak.arange(3), ak.arange(5)]
        with self.assertRaises(ValueError):
            ak.pdarraysetops.multiarray_setop_validation(x, y)

        with self.assertRaises(ValueError):
            ak.pdarraysetops.multiarray_setop_validation(y, x)

        x = [ak.arange(3, dtype=ak.uint64), ak.arange(3)]
        with self.assertRaises(TypeError):
            ak.pdarraysetops.multiarray_setop_validation(x, y)

    def test_index_of(self):
        # index of nan (reproducer from #3009)
        s = ak.Series(ak.array([1, 2, 3]), index=ak.array([1, 2, np.nan]))
        self.assertTrue(ak.indexof1d(ak.array([np.nan]), s.index.values).to_list() == [2])
        rng = np.random.default_rng()
        seeds = [rng.choice(2**63), rng.choice(2**63), rng.choice(2**63), rng.choice(2**63)]
        print("seeds: \n", seeds)

        def are_pdarrays_equal(pda1, pda2):
            # we first check the sizes so that we won't hit shape mismatch
            # before we can print the seed (due to short-circuiting)
            return (pda1.size == pda2.size) and ((pda1 == pda2).all())

        count = 0
        select_from_list = [
            ak.randint(-(2**32), 2**32, 10, seed=seeds[0]),
            ak.linspace(-(2**32), 2**32, 10),
            ak.random_strings_uniform(1, 16, 10, seed=seeds[1]),
        ]
        for select_from in select_from_list:
            count += 1
            arr1 = select_from[ak.randint(0, select_from.size, 20, seed=seeds[2]+count)]

            # test unique search space, this should be identical to find
            # be sure to test when all items are present and when there are items missing
            for arr2 in select_from, select_from[:5], select_from[5:]:
                found_in_second = ak.in1d(arr1, arr2)
                idx_of_first_in_second = ak.indexof1d(arr1, arr2)

                # search space not guaranteed to be unique since select_from could have duplicates
                # we will only match find with remove_missing when there's only one occurrence in the search space
                all_unique = ak.unique(arr2).size == arr2.size
                if all_unique:
                    # ensure we match find
                    if not are_pdarrays_equal(idx_of_first_in_second, ak.find(arr1, arr2, remove_missing=True)):
                        print("failed to match find")
                        print("second array all unique: ", all_unique)
                        print(seeds)
                    self.assertTrue((idx_of_first_in_second == ak.find(arr1, arr2, remove_missing=True)).all())

                    # if an element of arr1 is found in arr2, return the index of that item in arr2
                    if not are_pdarrays_equal(arr2[idx_of_first_in_second], arr1[found_in_second]):
                        print("arr1 at indices found_in_second doesn't match arr2[indexof1d]")
                        print("second array all unique: ", all_unique)
                        print(seeds)
                    self.assertTrue(
                        (arr2[idx_of_first_in_second] == arr1[found_in_second]).all()
                    )

            # test duplicate items in search space, the easiest way I can think
            # of to do this is to compare against pandas series getitem
            arr2 = select_from[ak.randint(0, select_from.size, 20, seed=seeds[3]+count)]
            pd_s = pd.Series(index=arr1.to_ndarray(), data=arr2.to_ndarray())
            ak_s = ak.Series(index=arr1, data=arr2)

            arr1_keys = ak.GroupBy(arr1).unique_keys
            arr2_keys = ak.GroupBy(arr2).unique_keys
            in_both = ak.intersect1d(arr1_keys, arr2_keys)

            for i in in_both.to_list():
                pd_i = pd_s[i]
                ak_i = ak_s[i]
                if isinstance(pd_i, pd.Series):
                    self.assertIsInstance(ak_i, ak.Series)
                    self.assertEqual(pd_i.values.tolist(), ak_i.values.to_list())
                else:
                    self.assertEqual(pd_i, ak_i)
