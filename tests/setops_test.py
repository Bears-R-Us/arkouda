import numpy as np
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

        with self.assertRaises(TypeError):
            ak.setxor1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))

        with self.assertRaises(RuntimeError):
            ak.setxor1d(ak.array([True, False, True]), ak.array([True, True]))

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

        with self.assertRaises(TypeError):
            ak.setdiff1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))

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

        with self.assertRaises(TypeError):
            ak.intersect1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))

        with self.assertRaises(RuntimeError):
            ak.intersect1d(ak.array([True, False, True]), ak.array([True, True]))

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

        with self.assertRaises(TypeError):
            ak.union1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))

        # with self.assertRaises(RuntimeError):
        #     ak.union1d(ak.array([True, True, True]), ak.array([True,False,True]))

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
        self.assertListEqual(["xyz", "def", "abc"], t[0].to_list())
        self.assertListEqual(["0", "456", "123"], t[1].to_list())

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
        self.assertListEqual(
            ak.in1d(pdaOne, pdaTwo).to_list(),
            [True, False, False, True],
        )

        vals = [i % 3 for i in range(10)]
        valsTwo = [i % 2 for i in range(10)]
        stringsOne = ak.array(["String {}".format(i) for i in vals])
        stringsTwo = ak.array(["String {}".format(i) for i in valsTwo])

        self.assertListEqual([x < 2 for x in vals], ak.in1d(stringsOne, stringsTwo).to_list())

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
