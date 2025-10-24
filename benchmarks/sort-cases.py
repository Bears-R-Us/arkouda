#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak
from arkouda.numpy.sorting import SortingAlgorithm


def is_cosorted(data):
    # (b[0] > a[0]) | ((b[0] == a[0]) & recurse(a[1], b[1]))
    def helper(x, right):
        return (x[1:] > x[:-1]) | ((x[1:] == x[:-1]) & right)

    right = ak.ones(data[0].size - 1, dtype=ak.bool_)
    for x in reversed(data):
        right = helper(x, right)
    return right.all()


def get_nbytes(data):
    if isinstance(data, ak.pdarray):
        return data.size * data.itemsize
    elif isinstance(data, ak.Strings):
        return data.size * 8 + data.nbytes
    else:
        return sum(get_nbytes(x) for x in data)


def apply_perm(data, perm):
    if isinstance(data, (ak.pdarray, ak.Strings)):
        return data[perm]
    else:
        return [x[perm] for x in data]


def check_sorted(s):
    if isinstance(s, (ak.pdarray, ak.Strings)):
        return ak.is_sorted(s)
    else:
        return is_cosorted(s)


def do_argsort(data, algo):
    if isinstance(data, (ak.pdarray, ak.Strings)):
        return ak.argsort(data, algo)
    else:
        return ak.coargsort(data, algo)


def check_correctness(data):
    """Only check accuracy of sorting, do not measure performance."""
    for algo in SortingAlgorithm:
        perm = do_argsort(data, algo)
        s = apply_perm(data, perm)
        assert check_sorted(s)


def time_sort(name, data, trials):
    """Measure both performance and correctness of sorting."""
    for algo in SortingAlgorithm:
        timings = []
        for i in range(trials):
            start = time.time()
            perm = do_argsort(data, algo)
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials
        nbytes = get_nbytes(data)
        print("{} {} average time = {:.4f} sec".format(name, algo.name, tavg))
        bytes_per_sec = nbytes / tavg
        print("{} {} average rate = {:.4f} GiB/sec".format(name, algo.name, bytes_per_sec / 2**30))
        s = apply_perm(data, perm)
        assert check_sorted(s)


def random_uniform(N):
    """
    Uniformly distributed integers of 1, 2, and 4 digits.
    Uniformly distributed reals in (0, 1)
    """
    for lbound, ubound, bstr in (
        (0, 2**16, "16-bit"),
        (0, 2**32, "32-bit"),
        (-(2**63), 2**63, "64-bit"),
    ):
        name = "uniform int64 {}".format(bstr)
        data = ak.randint(lbound, ubound, N)
        yield name, data
    name = "uniform float64"
    data = ak.uniform(N)
    yield name, data


def power_law(N):
    """Power law distributed (alpha = 2.5) reals and integers in (1, 2**32)."""
    y = ak.uniform(N)
    a = -2.5  # power law exponent, between -2 and -3
    ub = 2**32  # upper bound
    data = ((ub ** (a + 1) - 1) * y + 1) ** (1 / (a + 1))
    yield "power-law float64", data

    datai = ak.cast(data, ak.int64)
    yield "power-law int64", datai


def rmat(size):
    """RMAT-generated edges (coargsort of two vertex arrays)."""
    # N = number of edges = number of elements / 2
    N = size // 2
    avgdegree = 10
    lgNv = int(np.log2(N / avgdegree))
    # probabilities
    a = 0.01
    b = (1.0 - a) / 3.0
    c = b
    d = b
    # quantites to use in edge generation loop
    ab = a + b
    c_norm = c / (c + d)
    a_norm = a / (a + b)
    # init edge arrays
    ii = ak.ones(N, dtype=ak.int64)
    jj = ak.ones(N, dtype=ak.int64)
    # generate edges
    for ib in range(1, lgNv):
        ii_bit = ak.uniform(N) > ab
        jj_bit = ak.uniform(N) > (c_norm * ii_bit + a_norm * (~ii_bit))
        ii = ii + ((2 ** (ib - 1)) * ii_bit)
        jj = jj + ((2 ** (ib - 1)) * jj_bit)

    yield "RMAT int64", (ii, jj)


def block_sorted(N):
    """
    The concatenation of two sorted arrays of unequal length
    The interleaving of two sorted arrays of unequal length

    Most often occurs in array setops, where two arrays are
    uniqued (via sorting), then concatenated and sorted
    """
    splitpoint = 0.4
    Na = int(splitpoint * N)
    Nb = N - Na
    # Construct a and b such that:
    #   1) Values overlap
    #   2) a and b are sorted
    a = ak.arange(Na)
    b = ak.arange(Nb)
    c = ak.concatenate((a, b), ordered=True)
    yield "block-sorted concat int64", c

    ci = ak.concatenate((a, b), ordered=False)
    yield "block-sorted interleaved int64", ci


def refinement(N):
    """
    Coargsort of two arrays, where the first is already sorted
    but has many repeated values
    """
    groupsize = 100
    a = ak.arange(N // 2) // groupsize
    factor = 2**32 // a.max()
    a *= factor
    b = ak.randint(0, 2**32, N // 2)
    yield "refinement int64", (a, b)


def time_like(N):
    """
    Data like a datetime64[ns]:
    - spanning 1 year
    - with second granularity
    - but stored with nanosecond precision
    """
    # seconds in a year
    year_sec = 365 * 24 * 60 * 60
    # offset to almost 2020 (yeah yeah, leap days)
    twentytwenty = 50 * year_sec
    # second-resolution timestamps spanning approx 2020-2021
    a = ak.randint(0, year_sec, N) + twentytwenty
    # stored as datetime64[ns]
    a *= 10**9
    yield "datetime64[ns]", a


def IP_like(N):
    """Data like a 90/10 mix of IPv4 and IPv6 addresses."""
    multiplicity = 10
    nunique = N // (2 * multiplicity)
    # First generate unique addresses, then sample with replacement
    u1 = ak.zeros(nunique, dtype=ak.int64)
    u2 = ak.zeros(nunique, dtype=ak.int64)
    v4 = ak.uniform(nunique) < 0.9
    n4 = v4.sum()
    v6 = ~v4
    n6 = v4.size - n4
    u1[v4] = ak.randint(0, 2**32, n4)
    u1[v6] = ak.randint(-(2**63), 2**63, n6)
    u2[v6] = ak.randint(-(2**63), 2**63, n6)
    sample = ak.randint(0, nunique, N // 2)
    IP1 = u1[sample]
    IP2 = u2[sample]
    yield "IP-like 2*int64", (IP1, IP2)


GENERATORS = (random_uniform, power_law, rmat, block_sorted, refinement, time_like, IP_like)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure performance of sorting an array of random values."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**8, help="Problem size: length of array to argsort"
    )
    parser.add_argument(
        "-t", "--trials", type=int, default=1, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    # parser.add_argument('-s', '--seed', default=None, type=int, help='Value to initialize
    # random number generator')
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        args.size = 1000
    else:
        print("array size = {:,}".format(args.size))
        print("number of trials = ", args.trials)
    for gen in GENERATORS:
        for name, data in gen(args.size):
            if args.correctness_only:
                check_correctness(data)
            else:
                time_sort(name, data, args.trials)
    sys.exit(0)
