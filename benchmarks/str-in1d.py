#!/usr/bin/env python3

import argparse
import time

import arkouda as ak


# Tied to src/In1d.chpl:threshold, which defaults to 2**23
THRESHOLD = 2**23

MEDIUM = THRESHOLD - 1
LARGE = THRESHOLD + 1

# Controls how many unique values are possible
MAXSTRLEN = 5


def time_ak_in1d(size, trials):
    print(">>> arkouda string in1d")
    cfg = ak.get_config()
    N = size * cfg["numNodes"]
    a = ak.random_strings_uniform(1, MAXSTRLEN, N)

    for regime, bsize in zip(("Medium", "Large"), (MEDIUM, LARGE)):
        print(
            "{} regime: numNodes = {}  a.size = {:,}  b.size = {:,}".format(
                regime, cfg["numNodes"], N, bsize
            )
        )
        b = ak.random_strings_uniform(1, MAXSTRLEN, bsize)
        timings = []
        for _ in range(trials):
            start = time.time()
            c = ak.in1d(a, b)
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials
        print("{} average time = {:.4f} sec".format(regime, tavg))
        bytes_per_sec = (a.size * 8 + a.nbytes + b.size * 8 + b.nbytes) / tavg
        print("{} average rate = {:.2f} GiB/sec".format(regime, bytes_per_sec / 2**30))


def check_correctness():
    asize = 10**4
    bsize = 10**3
    a = ak.array([str(i) for i in range(asize)])
    b = ak.array([str(i) for i in range(bsize)])
    c = ak.in1d(a, b)
    assert c.sum() == bsize, "Incorrect result"


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of in1d with strings: c = ak.in1d(a, b)"
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**7, help="Problem size: length of array a")
    parser.add_argument(
        "-t", "--trials", type=int, default=3, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
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

    print("problem size per node = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_in1d(args.size, args.trials)
    sys.exit(0)
