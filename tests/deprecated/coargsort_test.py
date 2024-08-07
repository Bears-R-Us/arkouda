#!/usr/bin/env python3

import argparse

from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda.sorting import SortingAlgorithm


def check_integral(N, algo, dtype):
    z = ak.zeros(N, dtype=dtype)

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


def check_int_uint_float_bigint(N, algo):
    f = ak.randint(0, 2**63, N, dtype=ak.float64)
    u = ak.randint(0, 2**63, N, dtype=ak.uint64)
    i = ak.randint(0, 2**63, N, dtype=ak.int64)
    bi = u + 2**200

    perm = ak.coargsort([f, u, i, bi], algo)
    assert ak.is_sorted(f[perm])

    perm = ak.coargsort([u, i, bi, f], algo)
    assert ak.is_sorted(u[perm])

    perm = ak.coargsort([i, bi, f, u], algo)
    assert ak.is_sorted(i[perm])

    perm = ak.coargsort([bi, f, u, i], algo)
    # TODO remove once ak.is_sorted is avail for bigint
    shifted_down = ak.cast(bi[perm] - 2**200, ak.uint64)
    assert ak.is_sorted(shifted_down)


def check_large(N, algo):
    lg = [ak.randint(0, 2**63, N) for _ in range(10)]
    perm = ak.coargsort(lg, algo)
    assert ak.is_sorted(lg[0][perm])


def check_coargsort(N_per_locale):
    print(">>> arkouda coargsort")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    for algo in SortingAlgorithm:
        check_integral(N, algo, ak.int64)
        check_integral(N, algo, ak.uint64)
        check_float(N, algo)
        check_int_uint_float_bigint(N, algo)
        check_large(N, algo)


class CoargsortTest(ArkoudaTest):
    def test_int(self):
        for algo in SortingAlgorithm:
            check_integral(10**3, algo, ak.int64)

    def test_uint(self):
        for algo in SortingAlgorithm:
            check_integral(10**3, algo, ak.uint64)

    def test_float(self):
        for algo in SortingAlgorithm:
            check_float(10**3, algo)

    def test_int_uint_float_bigint(self):
        for algo in SortingAlgorithm:
            check_int_uint_float_bigint(10**3, algo)

    def test_large(self):
        for algo in SortingAlgorithm:
            check_large(10**3, algo)

    def test_error_handling(self):
        ones = ak.ones(100)
        short_ones = ak.ones(10)

        for algo in SortingAlgorithm:
            with self.assertRaises(ValueError):
                ak.coargsort([ones, short_ones], algo)

        for algo in SortingAlgorithm:
            with self.assertRaises(TypeError):
                ak.coargsort([list(range(0, 10)), [0]], algo)

    def test_coargsort_categorical(self):
        string = ak.array(["a", "b", "a", "b", "c"])
        cat = ak.Categorical(string)
        cat_from_codes = ak.Categorical.from_codes(
            codes=ak.array([0, 1, 0, 1, 2]), categories=ak.array(["a", "b", "c"])
        )
        for algo in SortingAlgorithm:
            # coargsort on categorical
            # coargsort sorts using codes, the order isn't guaranteed, only grouping
            cat_perm = ak.coargsort([cat], algo)
            self.assertListEqual(["a", "a", "b", "b", "c"], cat[cat_perm].to_list())

            # coargsort on categorical.from_codes
            # coargsort sorts using codes, the order isn't guaranteed, only grouping
            from_codes_perm = ak.coargsort([cat_from_codes], algo)
            self.assertListEqual(["a", "a", "b", "b", "c"], cat_from_codes[from_codes_perm].to_list())

            # coargsort on 2 categoricals (one from_codes)
            cat_perm = ak.coargsort([cat, cat_from_codes], algo)
            self.assertListEqual(["a", "a", "b", "b", "c"], cat[cat_perm].to_list())

            # coargsort on mixed strings and categoricals
            mixed_perm = ak.coargsort([cat, string, cat_from_codes], algo)
            self.assertListEqual(["a", "a", "b", "b", "c"], cat_from_codes[mixed_perm].to_list())

    def test_coargsort_empty(self):
        empty_str = ak.random_strings_uniform(1, 16, 0)
        empty_cat = ak.Categorical(empty_str)
        self.assertEqual(0, len(ak.coargsort([empty_str])))
        self.assertEqual(0, len(ak.coargsort([empty_cat])))

    def test_coargsort_bool(self):
        # Reproducer for issue #2675
        args = [ak.arange(5) % 2 == 0, ak.arange(5, 0, -1)]
        perm = ak.coargsort(args)
        self.assertListEqual(args[0][perm].to_list(), [False, False, True, True, True])
        self.assertListEqual(args[1][perm].to_list(), [2, 4, 1, 3, 5])


def create_parser():
    parser = argparse.ArgumentParser(description="Check coargsort correctness.")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**3, help="Problem size: length of array to argsort"
    )
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
