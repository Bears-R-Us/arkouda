import numpy as np
import pytest

import arkouda as ak

OPS = ["in1d", "intersect1d", "union1d", "setxor1d", "setdiff1d"]
INTEGRAL_TYPES = [ak.int64, ak.uint64, ak.bigint]
NUMERIC_TYPES = [ak.int64, ak.uint64, ak.bigint, ak.bool]


class TestSetOps:
    @staticmethod
    def make_np_arrays(size, dtype):
        if dtype == ak.int64 or dtype == ak.uint64:
            a = np.random.randint(0, size, size=size, dtype=dtype)
            b = np.random.randint(size / 2, 2 * size, size=size, dtype=dtype)
        elif dtype == ak.bigint:
            a = np.array([2**200 + i for i in range(size)])
            b = np.array([2**200 + i for i in range(int(size / 2), size * 2)])
        elif dtype == ak.float64:
            # only used for error handling tests
            a = np.random.random(size)
            b = np.random.random(size)
        elif dtype == bool:
            a = np.random.randint(0, 1, size=size, dtype=dtype)
            b = np.random.randint(0, 1, size=size, dtype=dtype)
        else:
            a = b = None

        return a, b

    @staticmethod
    def make_np_arrays_small(dtype):
        if dtype == ak.int64 or dtype == ak.uint64:
            a = np.array([-1, 0, 1, 3]).astype(dtype)
            b = np.array([-1, 2, 2, 3]).astype(dtype)
        elif dtype == ak.bigint:
            a = np.array([-1, 0, 1, 3]).astype(ak.uint64) + 2**200
            b = np.array([-1, 2, 2, 3]).astype(ak.uint64) + 2**200
        elif dtype == ak.bool:
            a = np.array([True, False, False, True]).astype(dtype)
            b = np.array([True, True, False, False]).astype(dtype)
        else:
            a = b = None
        return a, b

    @staticmethod
    def make_np_arrays_cross_type(dtype1, dtype2):
        if dtype1 == ak.int64 or dtype1 == ak.uint64:
            a = np.array([-1, -3, 0, 1, 2, 3]).astype(dtype1)
            c = np.array([-1, 0, 0, 7, 8, 3]).astype(dtype1)
        elif dtype1 == ak.bigint:
            a = np.array([-1, -3, 0, 1, 2, 3]).astype(ak.uint64) + 2 ** 200
            c = np.array([-1, 0, 0, 7, 8, 3]).astype(ak.uint64) + 2 ** 200
        elif dtype1 == ak.bool:
            a = np.array([True, False, False, True, True])
            c = np.array([True, True, False, False, True])
        else:
            a = c = None

        if dtype2 == ak.int64 or dtype2 == ak.uint64:
            b = np.array([-1, -11, 0, 4, 5, 3]).astype(dtype2)
            d = np.array([-1, -4, 0, 7, 8, 3]).astype(dtype2)
        elif dtype2 == ak.bigint:
            b = np.array([-1, -11, 0, 4, 5, 3]).astype(ak.uint64) + 2 ** 200
            d = np.array([-1, -4, 0, 7, 8, 3]).astype(ak.uint64) + 2 ** 200
        elif dtype2 == ak.bool:
            b = np.array([True, True, False, False, True])
            d = np.array([True, True, False, False, True])
        else:
            b = d = None

        return a, b, c, d

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    @pytest.mark.parametrize("op", OPS)
    def test_setops_integral_type(self, size, dtype, op):
        a, b = self.make_np_arrays(size, dtype)

        func = getattr(ak, op)
        ak_result = func(ak.array(a, dtype=dtype), ak.array(b, dtype=dtype))
        np_func = getattr(np, op)
        np_result = np_func(a, b)
        assert np.array_equal(ak_result.to_ndarray(), np_result)

        a, b = self.make_np_arrays_small(dtype)
        func = getattr(ak, op)
        ak_result = func(ak.array(a, dtype=dtype), ak.array(b, dtype=dtype))
        np_func = getattr(np, op)
        np_result = np_func(a, b)
        assert np.array_equal(ak_result.to_ndarray(), np_result)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("op", OPS)
    def test_setop_error_handling(self, size, op):
        a, b = self.make_np_arrays(size, ak.float64)
        func = getattr(ak, op)

        # float64 not implemented for in1dmsg
        with pytest.raises(RuntimeError if op == "in1d" else TypeError):
            func(ak.array(a, dtype=ak.float64), ak.array(b, dtype=ak.float64))

        # # bool is not supported by argsortMsg (only impacts single array case)
        a, b = self.make_np_arrays(size, ak.bool)
        if op in ["in1d", "setdiff1d"]:
            with pytest.raises(RuntimeError):
                func(ak.array(a, dtype=ak.bool), ak.array(b, dtype=ak.bool))

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("op", OPS)
    def test_setops_str(self, size, op):
        a = ak.random_strings_uniform(1, 5, size)
        b = ak.random_strings_uniform(1, 5, size)
        func = getattr(ak, op)
        ak_result = func(a, b)
        np_func = getattr(np, op)
        np_result = np_func(a.to_ndarray(), b.to_ndarray())
        assert np.array_equal(ak_result.to_ndarray(), np_result)

        a = ak.array(["a", "b", "c", "abc", "1"])
        b = ak.array(["x", "a", "y", "z", "abc", "123"])
        func = getattr(ak, op)
        ak_result = func(a, b)
        np_func = getattr(np, op)
        np_result = np_func(a.to_ndarray(), b.to_ndarray())
        assert np.array_equal(ak_result.to_ndarray(), np_result)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("op", OPS)
    def test_setops_categorical(self, size, op):
        a = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        func = getattr(ak, op)
        ak_result = func(a, b)
        np_func = getattr(np, op)
        np_result = np_func(a.to_ndarray(), b.to_ndarray())
        assert np.array_equal(ak_result.to_ndarray(), np_result)

        a = ak.Categorical(ak.array(["a", "b", "c", "abc", "1"]))
        b = ak.Categorical(ak.array(["x", "a", "y", "z", "abc", "123"]))
        func = getattr(ak, op)
        ak_result = func(a, b)
        np_func = getattr(np, op)
        np_result = np_func(a.to_ndarray(), b.to_ndarray())
        assert np.array_equal(ak_result.to_ndarray(), np_result)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    def test_in1d_multiarray_numeric_types(self, size, dtype):
        a, b = self.make_np_arrays(size, dtype)
        c, d = self.make_np_arrays(size, dtype)

        l1 = [ak.array(a), ak.array(c)]
        l2 = [ak.array(b), ak.array(d)]
        ak_result = ak.in1d(l1, l2)

        la = list(zip(a, c))
        lb = list(zip(b, d))
        lr = [x in lb for x in la]

        assert ak_result.to_list() == lr

    @pytest.mark.parametrize("dtype1", INTEGRAL_TYPES)
    @pytest.mark.parametrize("dtype2", INTEGRAL_TYPES)
    def test_in1d_multiarray_cross_type(self, dtype1, dtype2):
        a, b, c, d = self.make_np_arrays_cross_type(dtype1, dtype2)

        l1 = [ak.array(a), ak.array(b)]
        l2 = [ak.array(c), ak.array(d)]
        ak_result = ak.in1d(l1, l2)

        la = list(zip(a, b))
        lb = list(zip(c, d))
        lr = [x in lb for x in la]

        assert ak_result.to_list() == lr

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_in1d_multiarray_str(self, size):
        if size < 1000:
            size = 1000
        # large general test
        a = ak.random_strings_uniform(1, 2, size)
        b = ak.random_strings_uniform(1, 2, size)

        c = ak.random_strings_uniform(1, 2, size)
        d = ak.random_strings_uniform(1, 2, size)

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.in1d(l1, l2)

        la = list(zip(a.to_list(), b.to_list()))
        lb = list(zip(c.to_list(), d.to_list()))
        lr = [x in lb for x in la]
        assert ak_result.to_list() == lr

        stringsOne = ak.array(["String {}".format(i % 3) for i in range(10)])
        stringsTwo = ak.array(["String {}".format(i % 2) for i in range(10)])
        assert [(x % 3) < 2 for x in range(10)] == ak.in1d(stringsOne, stringsTwo).to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_in1d_multiarray_categorical(self, size):
        if size < 1000:
            size = 1000
        # large general test
        a = ak.Categorical(ak.random_strings_uniform(1, 2, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 2, size))

        c = ak.Categorical(ak.random_strings_uniform(1, 2, size))
        d = ak.Categorical(ak.random_strings_uniform(1, 2, size))

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.in1d(l1, l2)

        la = list(zip(a.to_list(), b.to_list()))
        lb = list(zip(c.to_list(), d.to_list()))
        lr = [x in lb for x in la]
        assert ak_result.to_list() == lr

        stringsOne = ak.Categorical(ak.array(["String {}".format(i % 3) for i in range(10)]))
        stringsTwo = ak.Categorical(ak.array(["String {}".format(i % 2) for i in range(10)]))
        assert [(x % 3) < 2 for x in range(10)] == ak.in1d(stringsOne, stringsTwo).to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    def test_intersect1d_multiarray_numeric_types(self, size, dtype):
        a, b = self.make_np_arrays(size, dtype)
        c, d = self.make_np_arrays(size, dtype)

        l1 = [ak.array(a), ak.array(c)]
        l2 = [ak.array(b), ak.array(d)]

        ak_result = ak.intersect1d(l1, l2)

        la = set(zip(a, c))
        lb = set(zip(b, d))
        lr = sorted(la.intersection(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        assert ak_result == lr

    @pytest.mark.parametrize("dtype1", INTEGRAL_TYPES)
    @pytest.mark.parametrize("dtype2", INTEGRAL_TYPES)
    def test_intersect1d_multiarray_cross_type(self, dtype1, dtype2):
        a, b, c, d = self.make_np_arrays_cross_type(dtype1, dtype2)

        l1 = [ak.array(a), ak.array(b)]
        l2 = [ak.array(c), ak.array(d)]
        ak_result = ak.intersect1d(l1, l2)

        la = set(zip(a, b))
        lb = set(zip(c, d))
        lr = sorted(la.intersection(lb))

        ak_result = [x.to_list() for x in ak_result]
        # sorting applied for bigint case. Numbers are right, but not ordering properly
        ak_result = sorted(list(zip(*ak_result)))
        assert ak_result == lr

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_intersect1d_multiarray_str(self, size):
        if size < 1000:
            size = 1000
        # large general test
        a = ak.random_strings_uniform(1, 2, size)
        b = ak.random_strings_uniform(1, 2, size)

        c = ak.random_strings_uniform(1, 2, size)
        d = ak.random_strings_uniform(1, 2, size)

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.intersect1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.intersection(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = sorted(list(zip(*ak_result)))
        assert ak_result == lr

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
        assert ["def"] == t[0].to_list()
        assert ["456"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_intersect1d_multiarray_categorical(self, size):
        if size < 1000:
            size = 1000
        # large general test
        a = ak.Categorical(ak.random_strings_uniform(1, 2, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 2, size))

        c = ak.Categorical(ak.random_strings_uniform(1, 2, size))
        d = ak.Categorical(ak.random_strings_uniform(1, 2, size))

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.intersect1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.intersection(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        assert ak_result == lr

        # Test for cat
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["abc", "def"]
        d = ["000", "456"]
        a1 = ak.Categorical(ak.array(a))
        a2 = ak.Categorical(ak.array(b))
        b1 = ak.Categorical(ak.array(c))
        b2 = ak.Categorical(ak.array(d))
        t = ak.intersect1d([a1, a2], [b1, b2])
        assert ["def"] == t[0].to_list()
        assert ["456"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_union1d_multiarray_numeric_types(self, size, dtype):
        a, b = self.make_np_arrays(size, dtype)
        c, d = self.make_np_arrays(size, dtype)

        l1 = [ak.array(a), ak.array(c)]
        l2 = [ak.array(b), ak.array(d)]

        ak_result = ak.union1d(l1, l2)

        la = set(zip(a, c))
        lb = set(zip(b, d))
        lr = sorted(la.union(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        assert ak_result == lr

    @pytest.mark.parametrize("dtype1", INTEGRAL_TYPES)
    @pytest.mark.parametrize("dtype2", INTEGRAL_TYPES)
    def test_union1d_multiarray_cross_type(self, dtype1, dtype2):
        a, b, c, d = self.make_np_arrays_cross_type(dtype1, dtype2)

        l1 = [ak.array(a), ak.array(b)]
        l2 = [ak.array(c), ak.array(d)]
        ak_result = ak.union1d(l1, l2)

        la = set(zip(a, b))
        lb = set(zip(c, d))
        lr = sorted(la.union(lb))

        ak_result = [x.to_list() for x in ak_result]
        # sorting applied for bigint case. Numbers are right, but not ordering properly
        ak_result = sorted(list(zip(*ak_result)))
        assert ak_result == lr

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_union1d_multiarray_str(self, size):
        # large general test
        a = ak.random_strings_uniform(1, 5, size)
        b = ak.random_strings_uniform(1, 5, size)

        c = ak.random_strings_uniform(1, 5, size)
        d = ak.random_strings_uniform(1, 5, size)

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.union1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.union(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        # because strings are grouped not sorted we are verifying the tuple exists
        for x in ak_result:
            assert x in lr

        # small scale known test
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["xyz"]
        d = ["0"]
        a1 = ak.array(a)
        a2 = ak.array(b)
        b1 = ak.array(c)
        b2 = ak.array(d)
        t = ak.union1d([a1, a2], [b1, b2])
        assert ["xyz", "def", "abc"] == t[0].to_list()
        assert ["0", "456", "123"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_union1d_multiarray_categorical(self, size):
        a = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        c = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        d = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.union1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.union(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        # because strings are grouped not sorted we are verifying the tuple exists
        for x in ak_result:
            assert x in lr

        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["xyz"]
        d = ["0"]
        a1 = ak.array(a)
        a2 = ak.array(b)
        b1 = ak.array(c)
        b2 = ak.array(d)
        cat_a1 = ak.Categorical(a1)
        cat_a2 = ak.Categorical(a2)
        cat_b1 = ak.Categorical(b1)
        cat_b2 = ak.Categorical(b2)
        t = ak.union1d([cat_a1, cat_a2], [cat_b1, cat_b2])
        assert ["abc", "def", "xyz"] == t[0].to_list()
        assert ["123", "456", "0"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    def test_setxor1d_multiarray_numeric_types(self, size, dtype):
        a, b = self.make_np_arrays(size, dtype)
        c, d = self.make_np_arrays(size, dtype)

        l1 = [ak.array(a), ak.array(c)]
        l2 = [ak.array(b), ak.array(d)]

        ak_result = ak.setxor1d(l1, l2)

        la = set(zip(a, c))
        lb = set(zip(b, d))
        lr = sorted(la.symmetric_difference(lb))

        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        assert ak_result == lr

    @pytest.mark.parametrize("dtype1", INTEGRAL_TYPES)
    @pytest.mark.parametrize("dtype2", INTEGRAL_TYPES)
    def test_setxor1d_multiarray_cross_type(self, dtype1, dtype2):
        a, b, c, d = self.make_np_arrays_cross_type(dtype1, dtype2)

        l1 = [ak.array(a), ak.array(b)]
        l2 = [ak.array(c), ak.array(d)]
        ak_result = ak.setxor1d(l1, l2)

        la = set(zip(a, b))
        lb = set(zip(c, d))
        lr = sorted(la.symmetric_difference(lb))

        ak_result = [x.to_list() for x in ak_result]
        # sorting applied for bigint case. Numbers are right, but not ordering properly
        ak_result = sorted(list(zip(*ak_result)))
        assert ak_result == lr

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_setxor1d_multiarray_str(self, size):
        # large general test
        a = ak.random_strings_uniform(1, 5, size)
        b = ak.random_strings_uniform(1, 5, size)

        c = ak.random_strings_uniform(1, 5, size)
        d = ak.random_strings_uniform(1, 5, size)

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.setxor1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.symmetric_difference(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        # because strings are grouped not sorted we are verifying the tuple exists
        for x in ak_result:
            assert x in lr

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
        assert ["abc", "abc"] == t[0].to_list()
        assert ["000", "123"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_setxor1d_multiarray_categorical(self, size):
        # large general test
        a = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        c = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        d = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.setxor1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.symmetric_difference(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        # because strings are grouped not sorted we are verifying the tuple exists
        for x in ak_result:
            assert x in lr

        # Test for strings
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["abc", "def"]
        d = ["000", "456"]
        a1 = ak.Categorical(ak.array(a))
        a2 = ak.Categorical(ak.array(b))
        b1 = ak.Categorical(ak.array(c))
        b2 = ak.Categorical(ak.array(d))
        t = ak.setxor1d([a1, a2], [b1, b2])
        assert ["abc", "abc"] == t[0].to_list()
        assert ["000", "123"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    def test_setdiff1d_multiarray_numeric_types(self, size, dtype):
        a, b = self.make_np_arrays(size, dtype)
        c, d = self.make_np_arrays(size, dtype)

        l1 = [ak.array(a), ak.array(c)]
        l2 = [ak.array(b), ak.array(d)]

        ak_result = ak.setdiff1d(l1, l2)

        la = set(zip(a, c))
        lb = set(zip(b, d))
        lr = sorted(la.difference(lb))

        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        assert ak_result == lr

    @pytest.mark.parametrize("dtype1", INTEGRAL_TYPES)
    @pytest.mark.parametrize("dtype2", INTEGRAL_TYPES)
    def test_setdiff1d_multiarray_cross_type(self, dtype1, dtype2):
        a, b, c, d = self.make_np_arrays_cross_type(dtype1, dtype2)

        l1 = [ak.array(a), ak.array(b)]
        l2 = [ak.array(c), ak.array(d)]
        ak_result = ak.setdiff1d(l1, l2)

        la = set(zip(a, b))
        lb = set(zip(c, d))
        lr = sorted(la.difference(lb))

        ak_result = [x.to_list() for x in ak_result]
        # sorting applied for bigint case. Numbers are right, but not ordering properly
        ak_result = sorted(list(zip(*ak_result)))
        assert ak_result == lr

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_setdiff1d_multiarray_str(self, size):
        # large general test
        a = ak.random_strings_uniform(1, 5, size)
        b = ak.random_strings_uniform(1, 5, size)

        c = ak.random_strings_uniform(1, 5, size)
        d = ak.random_strings_uniform(1, 5, size)

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.setdiff1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.difference(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        # because strings are grouped not sorted we are verifying the tuple exists
        for x in ak_result:
            assert x in lr

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
        assert ["abc"] == t[0].to_list()
        assert ["123"] == t[1].to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_setdiff1d_multiarray_categorical(self, size):
        # large general test
        a = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        c = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        d = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.setdiff1d(l1, l2)

        la = set(zip(a.to_list(), b.to_list()))
        lb = set(zip(c.to_list(), d.to_list()))
        lr = sorted(la.difference(lb))
        ak_result = [x.to_list() for x in ak_result]
        ak_result = list(zip(*ak_result))
        # because strings are grouped not sorted we are verifying the tuple exists
        for x in ak_result:
            assert x in lr

        # Test for strings
        a = ["abc", "def"]
        b = ["123", "456"]
        c = ["abc", "def"]
        d = ["000", "456"]
        a1 = ak.Categorical(ak.array(a))
        a2 = ak.Categorical(ak.array(b))
        b1 = ak.Categorical(ak.array(c))
        b2 = ak.Categorical(ak.array(d))
        t = ak.setdiff1d([a1, a2], [b1, b2])
        assert ["abc"] == t[0].to_list()
        assert ["123"] == t[1].to_list()

    def test_multiarray_validation(self):
        x = [ak.arange(3), ak.arange(3), ak.arange(3)]
        y = [ak.arange(2), ak.arange(2)]
        with pytest.raises(ValueError):
            ak.pdarraysetops.multiarray_setop_validation(x, y)

        x = [ak.arange(3), ak.arange(5)]
        with pytest.raises(ValueError):
            ak.pdarraysetops.multiarray_setop_validation(x, y)

        with pytest.raises(ValueError):
            ak.pdarraysetops.multiarray_setop_validation(y, x)

        x = [ak.arange(3, dtype=ak.uint64), ak.arange(3)]
        with pytest.raises(TypeError):
            ak.pdarraysetops.multiarray_setop_validation(x, y)
