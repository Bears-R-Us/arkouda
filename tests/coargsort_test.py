#!/usr/bin/env python3

import time, argparse
import numpy as np
from context import arkouda as ak 
from base_test import ArkoudaTest

def check_int(N, algo):
    z = ak.zeros(N, dtype=ak.int64)

    a2 = ak.randint(0, 2**16, N)
    b2 = ak.randint(0, 2**16, N)
    c2 = ak.randint(0, 2**16, N)
    d2 = ak.randint(0, 2**16, N)
    n2 = ak.randint(-(2**15), 2**15, N)

    perm = ak.coargsort([a2], algo)
    assert ak.is_sorted(a2[perm])

    perm = ak.coargsort([n2], algo)
    assert ak.is_sorted(n2[perm])

    perm = ak.coargsort([a2, b2, c2, d2], algo)
    assert ak.is_sorted(a2[perm])

    perm = ak.coargsort([z, b2, c2, d2], algo)
    assert ak.is_sorted(b2[perm])

    perm = ak.coargsort([z, z, c2, d2], algo)
    assert ak.is_sorted(c2[perm])

    perm = ak.coargsort([z, z, z, d2], algo)
    assert ak.is_sorted(d2[perm])


    a4 = ak.randint(0, 2**32, N)
    b4 = ak.randint(0, 2**32, N)
    n4 = ak.randint(-(2**31), 2**31, N)

    perm = ak.coargsort([a4], algo)
    assert ak.is_sorted(a4[perm])

    perm = ak.coargsort([n4], algo)
    assert ak.is_sorted(n4[perm])

    perm = ak.coargsort([a4, b4], algo)
    assert ak.is_sorted(a4[perm])

    perm = ak.coargsort([b4, a4], algo)
    assert ak.is_sorted(b4[perm])


    a8 = ak.randint(0, 2**64, N)
    b8 = ak.randint(0, 2**64, N)
    n8 = ak.randint(-(2**63), 2**64, N)
 

    perm = ak.coargsort([a8], algo)
    assert ak.is_sorted(a8[perm])

    perm = ak.coargsort([n8], algo)
    assert ak.is_sorted(n8[perm])

    perm = ak.coargsort([b8, a8], algo)
    assert ak.is_sorted(b8[perm])

    from itertools import permutations

    all_perm = permutations([a2, a4, a8])
    for p in all_perm:
        perm = ak.coargsort(p, algo)
        assert ak.is_sorted(p[0][perm])

def check_float(N, algo):
    a = ak.randint(0, 1, N, dtype=ak.float64)
    n = ak.randint(-1, 1, N, dtype=ak.float64)
    z = ak.zeros(N, dtype=ak.float64)

    perm = ak.coargsort([a], algo)
    assert ak.is_sorted(a[perm])

    perm = ak.coargsort([a, n], algo)
    assert ak.is_sorted(a[perm])

    perm = ak.coargsort([n, a], algo)
    assert ak.is_sorted(n[perm])

    perm = ak.coargsort([z, a], algo)
    assert ak.is_sorted(a[perm])

    perm = ak.coargsort([z, n], algo)
    assert ak.is_sorted(n[perm])

def check_int_float(N, algo):
    f = ak.randint(0, 2**63, N, dtype=ak.float64)
    i = ak.randint(0, 2**63, N, dtype=ak.int64)

    perm = ak.coargsort([f, i], algo)
    assert ak.is_sorted(f[perm])

    perm = ak.coargsort([i, f], algo)
    assert ak.is_sorted(i[perm])

def check_large(N, algo):
    l = [ak.randint(0, 2**63, N) for _ in range(10)]
    perm = ak.coargsort(l, algo)
    assert ak.is_sorted(l[0][perm])

def check_coargsort(N_per_locale):
    print(">>> arkouda coargsort")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    for algo in ak.SortingAlgorithm:
        check_int(N, algo)
        check_float(N, algo)
        check_int_float(N, algo)
        check_large(N, algo)

class CoargsortTest(ArkoudaTest):

    def test_int(self):
        for algo in ak.SortingAlgorithm:
            check_int(10**3, algo)

    def test_float(self):
        for algo in ak.SortingAlgorithm:
            check_float(10**3, algo)

    def test_int_float(self):
        for algo in ak.SortingAlgorithm:
            check_int_float(10**3, algo)

    def test_large(self):
        for algo in ak.SortingAlgorithm:
            check_large(10**3, algo)
        
    def test_error_handling(self):
        ones = ak.ones(100)
        short_ones = ak.ones(10)

        for algo in ak.SortingAlgorithm:
            with self.assertRaises(ValueError):
                ak.coargsort([ones, short_ones], algo)

        for algo in ak.SortingAlgorithm:
            with self.assertRaises(TypeError):
                ak.coargsort([list(range(0,10)), [0]], algo)

    def test_coargsort_categorical(self):
        string = ak.array(['a', 'b', 'a', 'b', 'c'])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(codes=ak.array([0, 1, 0, 1, 2]),
                                                   categories=ak.array(['a', 'b', 'c']))
        for algo in ak.SortingAlgorithm:
            str_perm = ak.coargsort([string], algo)
            str_sorted = string[str_perm].to_ndarray()

            # coargsort on categorical
            cat_perm = ak.coargsort([cat], algo)
            cat_sorted = cat[cat_perm].to_ndarray()
            self.assertTrue((str_sorted == cat_sorted).all())

            # coargsort on categorical.from_codes
            # coargsort sorts using codes, the order isn't guaranteed, only grouping
            from_codes_perm = ak.coargsort([cat_from_codes], algo)
            from_codes_sorted = cat_from_codes[from_codes_perm].to_ndarray()
            self.assertTrue((['a', 'a', 'b', 'b', 'c'] == from_codes_sorted).all())

            # coargsort on 2 categoricals (one from_codes)
            cat_perm = ak.coargsort([cat, cat_from_codes], algo)
            cat_sorted = cat[cat_perm].to_ndarray()
            self.assertTrue((str_sorted == cat_sorted).all())

            # coargsort on mixed strings and categoricals
            mixed_perm = ak.coargsort([cat, string, cat_from_codes], algo)
            mixed_sorted = cat_from_codes[mixed_perm].to_ndarray()
            self.assertTrue((str_sorted == mixed_sorted).all())


def create_parser():
    parser = argparse.ArgumentParser(description="Check coargsort correctness.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**3, help='Problem size: length of array to argsort')
    return parser

if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    ak.verbose = False
    ak.connect(server=args.hostname, port=args.port)

    print("array size = {:,}".format(args.size))
    check_coargsort(args.size)
    sys.exit(0)
