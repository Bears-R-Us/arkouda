#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak


TYPES = ("int64", "float64", "bool")


def create_ak_array(size, dtype):
    if dtype == "bool":
        return ak.arange(size) % 2 == 0
    return ak.arange(size).astype(dtype)


def create_np_array(size, dtype):
    if dtype == "bool":
        return np.arange(size) % 2 == 0
    return np.arange(size, dtype=dtype)


def time_ak_diff(size_per_locale, trials, dtype):
    print(">>> arkouda {} diff".format(dtype))
    cfg = ak.get_config()
    size = size_per_locale * cfg["numNodes"]
    print("numLocales = {}, numNodes {}, N = {:,}".format(cfg["numLocales"], cfg["numNodes"], size))
    a = create_ak_array(size, dtype)

    timings = []
    for _ in range(trials):
        start = time.time()
        result = ak.diff(a)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = ((a.size + result.size) * a.itemsize) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec / 2**30))


def time_np_diff(size, trials, dtype):
    print(">>> numpy {} diff".format(dtype))
    print("N = {:,}".format(size))
    a = create_np_array(size, dtype)

    timings = []
    for _ in range(trials):
        start = time.time()
        result = np.diff(a)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = ((a.size + result.size) * a.itemsize) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec / 2**30))


def check_correctness(dtype):
    size = 10**4
    npa = create_np_array(size, dtype)
    aka = create_ak_array(size, dtype)
    assert np.array_equal(np.diff(npa), ak.diff(aka).to_ndarray())


def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of first differences.")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**8, help="Input array size per locale")
    parser.add_argument(
        "-t", "--trials", type=int, default=6, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
    parser.add_argument(
        "--numpy",
        default=False,
        action="store_true",
        help="Run the same operation in NumPy to compare performance.",
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

    args = create_parser().parse_args()
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format("/".join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype)
        sys.exit(0)

    print("array size per locale = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_diff(args.size, args.trials, args.dtype)
    if args.numpy:
        time_np_diff(args.size, args.trials, args.dtype)
        print("Verifying agreement between arkouda and NumPy on small problem... ", end="")
        check_correctness(args.dtype)
        print("CORRECT")

    sys.exit(0)
