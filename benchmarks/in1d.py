#!/usr/bin/env python3

import argparse
import time

import arkouda as ak

TYPES = ("int64", "uint64")

# Tied to src/In1d.chpl:threshold, which defaults to 2**23
THRESHOLD = 2**23
MEDIUM = THRESHOLD - 1
LARGE = THRESHOLD + 1


def time_ak_in1d(N_per_locale, trials, dtype):
    print(f">>> arkouda {dtype} in1d")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    a = ak.arange(N) % LARGE
    if dtype == "uint64":
        a = ak.cast(a, ak.uint64)

    for regime, bsize in zip(("Medium", "Large"), (MEDIUM, LARGE)):
        print(
            "{} regime: numLocales = {}  a.size = {:,}  b.size = {:,}".format(
                regime, cfg["numLocales"], N, bsize
            )
        )
        b = ak.arange(bsize)
        if dtype == "uint64":
            b = ak.cast(b, ak.uint64)
        expected_misses = (LARGE - bsize) * (a.size // LARGE) + max((0, (a.size % LARGE) - bsize))
        timings = []
        for _ in range(trials):
            start = time.time()
            c = ak.in1d(a, b)
            end = time.time()
            timings.append(end - start)
            assert (c.size - c.sum()) == expected_misses, "Incorrect result"
        tavg = sum(timings) / trials
        print("{} average time = {:.4f} sec".format(regime, tavg))
        bytes_per_sec = (a.size * a.itemsize + b.size * b.itemsize) / tavg
        print("{} average rate = {:.2f} GiB/sec".format(regime, bytes_per_sec / 2**30))


def check_correctness(dtype):
    asize = 10**4
    bsize = 10**3
    a = ak.arange(asize)
    if dtype == "uint64":
        a = ak.cast(a, ak.uint64)
    b = ak.arange(bsize)
    if dtype == "uint64":
        b = ak.cast(b, ak.uint64)
    c = ak.in1d(a, b)
    assert c.sum() == bsize, "Incorrect result"


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of in1d: c = ak.in1d(a, b)")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**8, help="Problem size: length of array a")
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
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
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format("/".join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype)
        sys.exit(0)

    print("problem size per node = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_in1d(args.size, args.trials, args.dtype)
    sys.exit(0)
