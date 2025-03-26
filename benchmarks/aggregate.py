#!/usr/bin/env python3

import argparse
import time

import arkouda as ak


BOOLOPS = ("any", "all")


def generate_arrays(N, seed):
    # Sort keys so that aggregations will not have to permute values
    # We just want to measure aggregation time, not gather
    keys = ak.sort(ak.randint(0, 2**32, N, seed=seed))
    if seed is not None:
        seed += 1
    intvals = ak.randint(0, 2**16, N, seed=seed)
    boolvals = (intvals % 2) == 0
    return keys, intvals, boolvals


def time_ak_aggregate(N_per_locale, trials, seed):
    print(">>> arkouda aggregate")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    keys, intvals, boolvals = generate_arrays(N, seed)
    g = ak.GroupBy(keys, assume_sorted=True)
    for op in ak.GroupBy.Reductions:
        if op in BOOLOPS:
            v = boolvals
        else:
            v = intvals
        totalbytes = v.size * v.itemsize
        timings = []
        for i in range(trials):
            start = time.time()
            res = g.aggregate(v, op)[1]
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials
        print("Aggregate {} Average time = {:.4f} sec".format(op, tavg))
        bytes_per_sec = totalbytes / tavg
        print("Aggregate {} Average rate = {:.4f} GiB/sec".format(op, bytes_per_sec / 2**30))


def check_correctness():
    keys = ak.arange(1000) % 10
    ones = ak.ones_like(keys)
    g = ak.GroupBy(keys)
    # Make sure keys are correct
    assert (g.unique_keys == ak.arange(10)).all()
    # Check value of sums
    assert (g.sum(ones)[1] == 100).all()
    # For other ops, just run them and make sure they return the right size vector
    for op in ak.GroupBy.Reductions:
        if op in BOOLOPS:
            res = g.aggregate((ones == 1), op)[1]
        else:
            res = g.aggregate(ones, op)[1]
        assert res.size == g.unique_keys.size


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure performance of aggregations on grouped arrays."
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
        check_correctness()
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_aggregate(args.size, args.trials, args.seed)
    sys.exit(0)
