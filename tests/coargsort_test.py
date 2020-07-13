#!/usr/bin/env python3

import time, argparse
import numpy as np
from context import arkouda as ak 
from base_test import ArkoudaTest

def check_int(N):
    z = ak.zeros(N, dtype=ak.int64)

    a2 = ak.randint(0, 2**16, N)
    b2 = ak.randint(0, 2**16, N)
    c2 = ak.randint(0, 2**16, N)
    d2 = ak.randint(0, 2**16, N)
    n2 = ak.randint(-(2**15), 2**15, N)

    perm = ak.coargsort([a2])
    assert ak.is_sorted(a2[perm])

    perm = ak.coargsort([n2])
    assert ak.is_sorted(n2[perm])

    perm = ak.coargsort([a2, b2, c2, d2])
    assert ak.is_sorted(a2[perm])

    perm = ak.coargsort([z, b2, c2, d2])
    assert ak.is_sorted(b2[perm])

    perm = ak.coargsort([z, z, c2, d2])
    assert ak.is_sorted(c2[perm])

    perm = ak.coargsort([z, z, z, d2])
    assert ak.is_sorted(d2[perm])


    a4 = ak.randint(0, 2**32, N)
    b4 = ak.randint(0, 2**32, N)
    n4 = ak.randint(-(2**31), 2**31, N)

    perm = ak.coargsort([a4])
    assert ak.is_sorted(a4[perm])

    perm = ak.coargsort([n4])
    assert ak.is_sorted(n4[perm])

    perm = ak.coargsort([a4, b4])
    assert ak.is_sorted(a4[perm])

    perm = ak.coargsort([b4, a4])
    assert ak.is_sorted(b4[perm])


    a8 = ak.randint(0, 2**64, N)
    b8 = ak.randint(0, 2**64, N)
    n8 = ak.randint(-(2**63), 2**64, N)
 

    perm = ak.coargsort([a8])
    assert ak.is_sorted(a8[perm])

    perm = ak.coargsort([n8])
    assert ak.is_sorted(n8[perm])

    perm = ak.coargsort([b8, a8])
    assert ak.is_sorted(b8[perm])

    from itertools import permutations

    all_perm = permutations([a2, a4, a8])
    for p in all_perm:
        perm = ak.coargsort(p)
        assert ak.is_sorted(p[0][perm])

def check_float(N):
    a = ak.randint(0, 1, N, dtype=ak.float64)
    n = ak.randint(-1, 1, N, dtype=ak.float64)
    z = ak.zeros(N, dtype=ak.float64)

    perm = ak.coargsort([a])
    assert ak.is_sorted(a[perm])

    perm = ak.coargsort([a, n])
    assert ak.is_sorted(a[perm])

    perm = ak.coargsort([n, a])
    assert ak.is_sorted(n[perm])

    perm = ak.coargsort([z, a])
    assert ak.is_sorted(a[perm])

    perm = ak.coargsort([z, n])
    assert ak.is_sorted(n[perm])

def check_int_float(N):
    f = ak.randint(0, 2**63, N, dtype=ak.float64)
    i = ak.randint(0, 2**63, N, dtype=ak.int64)

    perm = ak.coargsort([f, i])
    assert ak.is_sorted(f[perm])

    perm = ak.coargsort([i, f])
    assert ak.is_sorted(i[perm])

def check_large(N):
    l = [ak.randint(0, 2**63, N) for _ in range(10)]
    perm = ak.coargsort(l)
    assert ak.is_sorted(l[0][perm])

def check_coargsort(N_per_locale):
    print(">>> arkouda coargsort")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    check_int(N)
    check_float(N)
    check_int_float(N)
    check_large(N)

class CoargsortTest(ArkoudaTest):

    def test_int(self):
        check_int(10**3)

    def test_float(self):
        check_float(10**3)

    def test_int_float(self):
        check_int_float(10**3)

    def test_large(self):
        check_large(10**3)

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
