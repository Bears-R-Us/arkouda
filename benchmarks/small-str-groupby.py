#!/usr/bin/env python3

import argparse
import time

import arkouda as ak


SIZES = {"small": 6, "medium": 12, "big": 24}


def time_ak_groupby(N_per_locale, trials, seed):
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    for k, v in SIZES.items():
        a = ak.random_strings_uniform(1, v, N, seed=seed)
        totalbytes = a.nbytes * a.entry.itemsize
        timings = []
        for i in range(trials):
            start = time.time()
            g = ak.GroupBy(a)
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials
        print("{} str array Average time = {:.4f} sec".format(k, tavg))
        bytes_per_sec = totalbytes / tavg
        print("{} str array Average rate = {:.4f} GiB/sec".format(k, bytes_per_sec / 2**30))


def check_correctness(s, seed):
    a = ak.random_strings_uniform(1, SIZES[s], 1000, seed=seed)
    g = ak.GroupBy(a)
    perm = ak.argsort(ak.randint(0, 2**32, a.size))
    g2 = ak.GroupBy(a[perm])
    assert (g.unique_keys == g2.unique_keys).all()
    assert (g.segments == g2.segments).all()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure performance of grouping arrays of random values."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=10**8,
        help="Problem size: total length of all arrays to group",
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
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="Value to initialize random number generator"
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for s in SIZES:
            check_correctness(s, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_groupby(args.size, args.trials, args.seed)
    sys.exit(0)
