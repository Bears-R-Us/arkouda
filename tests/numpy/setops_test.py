import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.testing import assert_arkouda_array_equivalent


OPS = ["in1d", "intersect1d", "union1d", "setxor1d", "setdiff1d"]
INTEGRAL_TYPES = [ak.int64, ak.uint64, ak.bigint]
NUMERIC_TYPES = [ak.int64, ak.uint64, ak.bigint, ak.bool_]


class TestSetOps:
    @pytest.mark.requires_chapel_module("In1dMsg")
    def test_pdarraysetops_docstrings(self):
        import doctest

        from arkouda.numpy import pdarraysetops

        result = doctest.testmod(
            pdarraysetops, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

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
        elif dtype == ak.bool_:
            a = np.random.randint(0, 2, size=size, dtype=dtype)
            b = np.random.randint(0, 2, size=size, dtype=dtype)
        else:
            a = b = None

        return a, b

    @staticmethod
    def make_np_arrays_small(dtype):
        if dtype == ak.int64 or dtype == ak.uint64:
            a = np.array([-1, 0, 1, 3]).astype(dtype)
            b = np.array([-1, 2, 2, 3]).astype(dtype)
        elif dtype == ak.bigint:
            a = np.array([i + 2**200 for i in [-1, 0, 1, 3]])
            b = np.array([i + 2**200 for i in [-1, 2, 2, 3]])
        elif dtype == ak.bool_:
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
            a = np.array([i + 2**200 for i in [-1, -3, 0, 1, 2, 3]])
            c = np.array([i + 2**200 for i in [-1, 0, 0, 7, 8, 3]])
        elif dtype1 == ak.bool_:
            a = np.array([True, False, False, True, True])
            c = np.array([True, True, False, False, True])
        else:
            a = c = None

        if dtype2 == ak.int64 or dtype2 == ak.uint64:
            b = np.array([-1, -11, 0, 4, 5, 3]).astype(dtype2)
            d = np.array([-1, -4, 0, 7, 8, 3]).astype(dtype2)
        elif dtype2 == ak.bigint:
            b = np.array([i + 2**200 for i in [-1, -11, 0, 4, 5, 3]])
            d = np.array([i + 2**200 for i in [-1, -4, 0, 7, 8, 3]])
        elif dtype2 == ak.bool_:
            b = np.array([True, True, False, False, True])
            d = np.array([True, True, False, False, True])
        else:
            b = d = None

        return a, b, c, d

    @pytest.mark.requires_chapel_module("In1dMsg")
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INTEGRAL_TYPES)
    @pytest.mark.parametrize("op", OPS)
    def test_setops_integral_type(self, size, dtype, op):
        a, b = self.make_np_arrays(size, dtype)

        func = getattr(ak, op)
        ak_result = func(ak.array(a, dtype=dtype), ak.array(b, dtype=dtype))

        # NumPy 2.x compatibility: in1d -> isin
        if op == "in1d":
            np_result = np.isin(a, b)
        else:
            np_func = getattr(np, op)
            np_result = np_func(a, b)

        assert np.array_equal(ak_result.to_ndarray(), np_result)

        a, b = self.make_np_arrays_small(dtype)
        func = getattr(ak, op)
        ak_result = func(ak.array(a, dtype=dtype), ak.array(b, dtype=dtype))

        if op == "in1d":
            np_result = np.isin(a, b)
        else:
            np_func = getattr(np, op)
            np_result = np_func(a, b)

        assert np.array_equal(ak_result.to_ndarray(), np_result)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("op", OPS)
    def test_setop_error_handling(self, size, op):
        a, b = self.make_np_arrays(size, ak.float64)
        func = getattr(ak, op)

        # # bool is not supported by argsortMsg (only impacts single array case)
        a, b = self.make_np_arrays(size, ak.bool_)
        if op in ["in1d", "setdiff1d"]:
            with pytest.raises(RuntimeError):
                func(ak.array(a, dtype=ak.bool_), ak.array(b, dtype=ak.bool_))

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

        assert ak_result.tolist() == lr

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

        assert ak_result.tolist() == lr

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

        la = list(zip(a.tolist(), b.tolist()))
        lb = list(zip(c.tolist(), d.tolist()))
        lr = [x in lb for x in la]
        assert ak_result.tolist() == lr

        stringsOne = ak.array(["String {}".format(i % 3) for i in range(10)])
        stringsTwo = ak.array(["String {}".format(i % 2) for i in range(10)])
        assert [(x % 3) < 2 for x in range(10)] == ak.in1d(stringsOne, stringsTwo).tolist()

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

        la = list(zip(a.tolist(), b.tolist()))
        lb = list(zip(c.tolist(), d.tolist()))
        lr = [x in lb for x in la]
        assert ak_result.tolist() == lr

        stringsOne = ak.Categorical(ak.array(["String {}".format(i % 3) for i in range(10)]))
        stringsTwo = ak.Categorical(ak.array(["String {}".format(i % 2) for i in range(10)]))
        assert [(x % 3) < 2 for x in range(10)] == ak.in1d(stringsOne, stringsTwo).tolist()

    @pytest.mark.requires_chapel_module("In1dMsg")
    def test_in1d_symmetric(self):
        # Duplicates to exercise assume_unique=False (GroupBy/broadcast path)
        a = ak.array([1, 2, 2, 3, 4])
        b = ak.array([2, 4, 4, 5])

        def exp_in(x, y):
            yset = set(y)
            return [xi in yset for xi in x]

        a_list = a.to_ndarray().tolist()
        b_list = b.to_ndarray().tolist()

        # assume_unique=False path (duplicates allowed; should match membership semantics)
        am2, bm2 = ak.in1d(a, b, assume_unique=False, symmetric=True, invert=False)
        assert am2.tolist() == exp_in(a_list, b_list)
        assert bm2.tolist() == exp_in(b_list, a_list)

        am2_i, bm2_i = ak.in1d(a, b, assume_unique=False, symmetric=True, invert=True)
        assert am2_i.tolist() == [not v for v in exp_in(a_list, b_list)]
        assert bm2_i.tolist() == [not v for v in exp_in(b_list, a_list)]

        # assume_unique=True path (inputs must be unique for this branch to be valid)
        au = ak.array([1, 2, 3, 4])
        bu = ak.array([2, 4, 5])

        au_list = au.to_ndarray().tolist()
        bu_list = bu.to_ndarray().tolist()

        am, bm = ak.in1d(au, bu, assume_unique=True, symmetric=True, invert=False)
        assert am.tolist() == exp_in(au_list, bu_list)
        assert bm.tolist() == exp_in(bu_list, au_list)

        am_i, bm_i = ak.in1d(au, bu, assume_unique=True, symmetric=True, invert=True)
        assert am_i.tolist() == [not v for v in exp_in(au_list, bu_list)]
        assert bm_i.tolist() == [not v for v in exp_in(bu_list, au_list)]

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
        ak_result = [x.tolist() for x in ak_result]
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

        ak_result = [x.tolist() for x in ak_result]
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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.intersection(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["def"] == t[0].tolist()
        assert ["456"] == t[1].tolist()

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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.intersection(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["def"] == t[0].tolist()
        assert ["456"] == t[1].tolist()

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
        ak_result = [x.tolist() for x in ak_result]
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

        ak_result = [x.tolist() for x in ak_result]
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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.union(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert len({"xyz", "def", "abc"}.symmetric_difference(t[0].tolist())) == 0
        assert len({"0", "456", "123"}.symmetric_difference(t[1].tolist())) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_union1d_many_strings(self, size):
        import random

        base_array = [str(i) for i in range(size)]
        list_1 = []
        for i in range(size // 5):
            val = random.choice(base_array)
            list_1.append(val)
            base_array.remove(val)

        base_array = [str(i) for i in range(size)]
        list_2 = []
        for i in range(size // 5):
            val = random.choice(base_array)
            list_2.append(val)
            base_array.remove(val)

        ak_arr_1 = ak.array(list_1)
        ak_arr_2 = ak.array(list_2)

        np_arr_1 = np.array(list_1)
        np_arr_2 = np.array(list_2)

        ak_result = ak.union1d(ak_arr_1, ak_arr_2)
        np_result = np.union1d(np_arr_1, np_arr_2)

        assert_arkouda_array_equivalent(ak_result, np_result)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_union1d_multiarray_categorical(self, size):
        a = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        b = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        c = ak.Categorical(ak.random_strings_uniform(1, 5, size))
        d = ak.Categorical(ak.random_strings_uniform(1, 5, size))

        l1 = [a, b]
        l2 = [c, d]

        ak_result = ak.union1d(l1, l2)

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.union(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["abc", "def", "xyz"] == t[0].tolist()
        assert ["123", "456", "0"] == t[1].tolist()

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

        ak_result = [x.tolist() for x in ak_result]
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

        ak_result = [x.tolist() for x in ak_result]
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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.symmetric_difference(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["abc", "abc"] == t[0].tolist()
        assert ["000", "123"] == t[1].tolist()

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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.symmetric_difference(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["abc", "abc"] == t[0].tolist()
        assert ["000", "123"] == t[1].tolist()

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

        ak_result = [x.tolist() for x in ak_result]
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

        ak_result = [x.tolist() for x in ak_result]
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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.difference(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["abc"] == t[0].tolist()
        assert ["123"] == t[1].tolist()

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

        la = set(zip(a.tolist(), b.tolist()))
        lb = set(zip(c.tolist(), d.tolist()))
        lr = sorted(la.difference(lb))
        ak_result = [x.tolist() for x in ak_result]
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
        assert ["abc"] == t[0].tolist()
        assert ["123"] == t[1].tolist()

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

    @pytest.mark.requires_chapel_module("In1dMsg")
    def test_index_of(self):
        # index of nan (reproducer from #3009)
        s = ak.Series(ak.array([1, 2, 3]), index=ak.array([1, 2, np.nan]))
        assert ak.indexof1d(ak.array([np.nan]), s.index.values).tolist() == [2]
        rng = np.random.default_rng(pytest.seed)
        seeds = [
            rng.choice(2**63),
            rng.choice(2**63),
            rng.choice(2**63),
            rng.choice(2**63),
        ]
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
            arr1 = select_from[ak.randint(0, select_from.size, 20, seed=seeds[2] + count)]

            # test unique search space, this should be identical to find
            # be sure to test when all items are present and when there are items missing
            for arr2 in select_from, select_from[:5], select_from[5:]:
                found_in_second = ak.in1d(arr1, arr2)
                idx_of_first_in_second = ak.indexof1d(arr1, arr2)

                # search space not guaranteed to be unique since select_from could have duplicates
                # we will only match find with remove_missing
                # when there's only one occurrence in the search space
                all_unique = ak.unique(arr2).size == arr2.size
                if all_unique:
                    # ensure we match find
                    if not are_pdarrays_equal(
                        idx_of_first_in_second, ak.numpy.alignment.find(arr1, arr2, remove_missing=True)
                    ):
                        print("failed to match find")
                        print("second array all unique: ", all_unique)
                        print(seeds)
                    assert (
                        idx_of_first_in_second
                        == ak.numpy.alignment.find(arr1, arr2, remove_missing=True)
                    ).all()

                    # if an element of arr1 is found in arr2, return the index of that item in arr2
                    if not are_pdarrays_equal(arr2[idx_of_first_in_second], arr1[found_in_second]):
                        print("arr1 at indices found_in_second doesn't match arr2[indexof1d]")
                        print("second array all unique: ", all_unique)
                        print(seeds)
                    assert (arr2[idx_of_first_in_second] == arr1[found_in_second]).all()

            # test duplicate items in search space, the easiest way I can think
            # of to do this is to compare against pandas series getitem
            arr2 = select_from[ak.randint(0, select_from.size, 20, seed=seeds[3] + count)]
            pd_s = pd.Series(index=arr1.to_ndarray(), data=arr2.to_ndarray())
            ak_s = ak.Series(index=arr1, data=arr2)

            arr1_keys = ak.GroupBy(arr1).unique_keys
            arr2_keys = ak.GroupBy(arr2).unique_keys
            in_both = ak.intersect1d(arr1_keys, arr2_keys)

            for i in in_both.tolist():
                pd_i = pd_s[i]
                ak_i = ak_s[i]
                if isinstance(pd_i, pd.Series):
                    assert isinstance(ak_i, ak.Series)
                    assert pd_i.values.tolist() == ak_i.values.tolist()
                else:
                    assert pd_i == ak_i
