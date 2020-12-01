#!/usr/bin/env python3                                                         

import importlib
import numpy as np
import math
import gc
import sys

from context import arkouda as ak
from base_test import ArkoudaTest

N = 1_000_000
errors = False

def pass_fail(f):
    global errors
    errors = errors or not f
    return ("Passed" if f else "Failed")

def check_bool(N):
    a = ak.arange(N)
    b = ak.ones(N)
    try:
        c = a and b
    except ValueError:
        correct = True
    except:
        correct = False
    d = ak.array([1])
    correct = correct and (d and 5)
    return pass_fail(correct)

def check_arange(N):
    # create np version
    a = np.arange(N)
    # create ak version
    b = ak.arange(N)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_linspace(N):
    # create np version
    a = np.linspace(10, 20, N)
    # create ak version
    b = ak.linspace(10, 20, N)
    # print(a,b)
    f = np.allclose(a, b.to_ndarray())
    return pass_fail(f)

def check_ones(N):
    # create np version
    a = np.ones(N)
    # create ak version
    b = ak.ones(N)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_zeros(N):
    # create np version
    a = np.zeros(N)
    # create ak version
    b = ak.zeros(N)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_argsort(N):
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
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_coargsort(N):
    # create np version
    a = np.arange(N)
    a = a[::-1]
    iv = np.lexsort([a, a])
    a = a[iv]
    # create ak version
    b = ak.arange(N)
    b = b[::-1]
    iv = ak.coargsort([b, b])
    b = b[iv]
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_sort(N):
    # create np version
    a = np.arange(N)
    a = a[::-1]
    a = np.sort(a)
    # create ak version
    b = ak.arange(N)
    b = b[::-1]
    b = ak.sort(b)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_get_slice(N):
    # create np version
    a = np.ones(N)
    a = a[::2]
    # create ak version
    b = ak.ones(N)
    b = b[::2]
    # print(a,b)
    c = a == b.to_ndarray()
    return pass_fail(c.all())

def check_set_slice_value(N):
    # create np version
    a = np.ones(N)
    a[::2] = -1
    # create ak version
    b = ak.ones(N)
    b[::2] = -1
    # print(a,b)
    c = a == b.to_ndarray()
    return pass_fail(c.all())

def check_set_slice(N):
    # create np version
    a = np.ones(N)
    a[::2] = a[::2] * -1
    # create ak version
    b = ak.ones(N)
    b[::2] = b[::2] * -1
    # print(a,b)
    c = a == b.to_ndarray()
    return pass_fail(c.all())

def check_get_bool_iv(N):
    # create np version
    a = np.arange(N)
    a = a[a < N//2]
    # create ak version
    b = ak.arange(N)
    b = b[b < N//2]
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_set_bool_iv_value(N):
    # create np version
    a = np.arange(N)
    a[a < N//2] = -1
    # create ak version
    b = ak.arange(N)
    b[b < N//2] = -1
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_set_bool_iv(N):
    # create np version
    a = np.arange(N)
    a[a < N//2] = a[:N//2] * -1
    # create ak version
    b = ak.arange(N)
    b[b < N//2] = b[:N//2] * -1
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_get_integer_iv(N):
    # create np version
    a = np.arange(N)
    iv = np.arange(N//2)
    a = a[iv]
    # create ak version
    b = ak.arange(N)
    iv = ak.arange(N//2)
    b = b[iv]
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_set_integer_iv_value(N):
    # create np version
    a = np.arange(N)
    iv = np.arange(N//2)
    a[iv] = -1
    # create ak version
    b = ak.arange(N)
    iv = ak.arange(N//2)
    b[iv] = -1
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_set_integer_iv(N):
    # create np version
    a = np.arange(N)
    iv = np.arange(N//2)
    a[iv] = iv*10
    # create ak version
    b = ak.arange(N)
    iv = ak.arange(N//2)
    b[iv] = iv*10
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

def check_get_integer_idx(N):
    # create np version
    a = np.arange(N)
    v1 = a[N//2]
    # create ak version
    b = ak.arange(N)
    v2 = b[N//2]
    return pass_fail(v1 == v2) and pass_fail(a[-1] == b[-1])

def check_set_integer_idx(N):
    # create np version
    a = np.arange(N)
    a[N//2] = -1
    a[-1] = -1
    v1 = a[N//2]
    # create ak version
    b = ak.arange(N)
    b[N//2] = -1
    b[-1] = -1
    v2 = b[N//2]
    return pass_fail(v1 == v2) and pass_fail(a[-1] == b[-1])

'''
Encapsulates test cases that invoke the run_tests method.
'''
class CheckTest(ArkoudaTest):
    
    def testBool(self):
        self.assertTrue(check_bool(N))
    
    def testArange(self):
        self.assertTrue(check_arange(N))
        
    def testLinspace(self):
        self.assertTrue(check_linspace(N))
        
    def testOnes(self):
        self.assertTrue(check_ones(N))
        
    def testZeros(self):
        self.assertTrue(check_zeros(N))
        
    def testArgsort(self):
        self.assertTrue(check_argsort(N))
        
    def testCoargsort(self):
        self.assertTrue(check_coargsort(N))
        
    def testSort(self):
        self.assertTrue(check_sort(N))
        
    def testGetSlice(self):
        self.assertTrue(check_get_slice(N))
        
    def testSetSliceValue(self):
        self.assertTrue(check_set_slice_value(N))
        
    def testSetSlice(self):
        self.assertTrue(check_set_slice(N))
        
    def testGetBoolIv(self):
        self.assertTrue(check_get_bool_iv(N))
        
    def testGetBoolIvValue(self):
        self.assertTrue(check_set_bool_iv_value(N))
        
    def testSetBoolIv(self):
        self.assertTrue(check_set_bool_iv(N))
        
    def testGetIntegerIv(self):
        self.assertTrue(check_get_integer_iv(N))
        
    def testSetIntegerIvValue(self):
        self.assertTrue(check_set_integer_iv_value(N))
        
    def testSetIntegerIv(self):
        self.assertTrue(check_set_integer_iv(N))
        
    def testGetIntegerIdx(self):
        self.assertTrue(check_get_integer_idx(N))
        
    def testSetIntegerIdx(self):
        self.assertTrue(check_set_integer_idx(N))

if __name__ == '__main__':
    N = 1_000_000
    print(">>> Sanity checks on the arkouda_server")

    ak.verbose = False
    if len(sys.argv) > 1:
        ak.connect(server=sys.argv[1], port=sys.argv[2])
    else:
        ak.connect()

    # Run Tests
    print("check boolean :", check_bool(N))
    print("check arange :", check_arange(N))
    print("check linspace :", check_linspace(N))
    print("check ones :", check_ones(N))
    print("check zeros :", check_zeros(N))
    print("check argsort :", check_argsort(N))
    print("check coargsort :", check_coargsort(N))
    print("check sort :", check_sort(N))
    print("check get slice [::2] :", check_get_slice(N))
    print("check set slice [::2] = value:", check_set_slice_value(N))
    print("check set slice [::2] = pda:", check_set_slice(N))
    print("check (compressing) get bool iv :", check_get_bool_iv(N))
    print("check (expanding) set bool iv = value:", check_set_bool_iv_value(N))
    print("check (expanding) set bool iv = pda:", check_set_bool_iv(N))
    print("check (gather) get integer iv:", check_get_integer_iv(N))
    print("check (scatter) set integer iv = value:", check_set_integer_iv_value(N))
    print("check (scatter) set integer iv = pda:", check_set_integer_iv(N))
    print("check get integer idx :", check_get_integer_idx(N))
    print("check set integer idx = value:", check_set_integer_idx(N))

    ak.disconnect()
    sys.exit(errors)
    