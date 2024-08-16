import numpy as np
import pytest

import arkouda as ak

N = 1_000_000

"""
The CompareTest class encapsulates unit tests that compare the results of analogous
method invocations against Numpy ndarrays and Arkouda pdarrays to ensure equivalent
results are generated.
"""


class TestCompare:
    def test_compare_arange(self):
        # create np version
        nArange = np.arange(N)
        # create ak version
        aArange = ak.arange(N)
        assert nArange.all() == aArange.all()

    def test_compare_linspace(self):
        # create np version
        a = np.linspace(10, 20, N)
        # create ak version
        b = ak.linspace(10, 20, N)
        # print(a,b)
        assert np.allclose(a, b.to_ndarray())

    def test_compare_ones(self):
        # create np version
        nArray = np.ones(N)
        # create ak version
        aArray = ak.ones(N)
        assert nArray.all() == aArray.to_ndarray().all()

    def test_compare_zeros(self):
        # create np version
        nArray = np.zeros(N)
        # create ak version
        aArray = ak.zeros(N)
        assert nArray.all() == aArray.to_ndarray().all()

    def test_compare_argsort(self):
        # create np version
        a = np.arange(N)
        a = a[::-1]
        iv = np.argsort(a)
        a = a[iv]
        # create ak version
        b = ak.arange(N)
        b = b[::-1]
        iv = ak.argsort(b)
        b = b[iv]
        assert np.array_equal(a, b.to_ndarray())


    def test_compare_sort(self):
        # create np version
        a = np.arange(N)
        a = a[::-1]
        a = np.sort(a)
        # create ak version
        b = ak.arange(N)
        b = b[::-1]
        b = ak.sort(b)
        assert a.all() == b.to_ndarray().all()

    def test_compare_get_slice(self):
        # create np version
        a = np.ones(N)
        a = a[::2]
        # create ak version
        b = ak.ones(N)
        b = b[::2]
        assert a.all() == b.to_ndarray().all()

    def test_compare_set_slice_value(self):
        # create np version
        a = np.ones(N)
        a[::2] = -1
        # create ak version
        b = ak.ones(N)
        b[::2] = -1
        assert a.all() == b.to_ndarray().all()

    def test_compare_set_slice(self):
        # create np version
        a = np.ones(N)
        a[::2] = a[::2] * -1
        # create ak version
        b = ak.ones(N)
        b[::2] = b[::2] * -1
        assert a.all() == b.to_ndarray().all()

    def test_compare_get_bool_iv(self):
        # create np version
        a = np.arange(N)
        a = a[a < N // 2]
        # create ak version
        b = ak.arange(N)
        b = b[b < N // 2]
        assert a.all() == b.to_ndarray().all()

    def test_compare_set_bool_iv_value(self):
        # create np version
        a = np.arange(N)
        a[a < N // 2] = -1
        # create ak version
        b = ak.arange(N)
        b[b < N // 2] = -1
        assert a.all() == b.to_ndarray().all()

    def check_set_bool_iv(self):
        # create np version
        a = np.arange(N)
        a[a < N // 2] = a[: N // 2] * -1
        # create ak version
        b = ak.arange(N)
        b[b < N // 2] = b[: N // 2] * -1
        assert a.all() == b.to_ndarray().all()

    def check_get_integer_iv(self):
        # create np version
        a = np.arange(N)
        iv = np.arange(N // 2)
        a = a[iv]
        # create ak version
        b = ak.arange(N)
        iv = ak.arange(N // 2)
        b = b[iv]
        assert a.all() == b.to_ndarray().all()

    def test_compare_set_integer_iv_val(self):
        # create np version
        a = np.arange(N)
        iv = np.arange(N // 2)
        a[iv] = -1
        # create ak version
        b = ak.arange(N)
        iv = ak.arange(N // 2)
        b[iv] = -1
        assert a.all() == b.to_ndarray().all()

    def test_compare_set_integer_iv(self):
        # create np version
        a = np.arange(N)
        iv = np.arange(N // 2)
        a[iv] = iv * 10
        # create ak version
        b = ak.arange(N)
        iv = ak.arange(N // 2)
        b[iv] = iv * 10
        assert a.all() == b.to_ndarray().all()

    def test_compare_get_integer_idx(self):
        # create np version
        a = np.arange(N)
        v1 = a[N // 2]
        # create ak version
        b = ak.arange(N)
        v2 = b[N // 2]
        assert v1 == v2
        assert a[-1] == b[-1]

    def test_compare_set_integer_idx(self):
        # create np version
        a = np.arange(N)
        a[N // 2] = -1
        a[-1] = -1
        v1 = a[N // 2]
        # create ak version
        b = ak.arange(N)
        b[N // 2] = -1
        b[-1] = -1
        v2 = b[N // 2]
        assert v1 == v2
        assert a[-1]== b[-1]
