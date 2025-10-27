#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak


OPS = ("zeros", "ones", "randint")
TYPES = ("int64", "float64", "uint64")


def create_ak_array(N, op, dtype, seed):
    if op == "zeros":
        a = ak.zeros(N, dtype=dtype)
    elif op == "ones":
        a = ak.ones(N, dtype=dtype)
    elif op == "randint":
        a = ak.randint(0, 2**32, N, dtype=dtype, seed=seed)
    return a


def create_np_array(N, op, dtype, seed):
    if op == "zeros":
        a = np.zeros(N, dtype=dtype)
    elif op == "ones":
        a = np.ones(N, dtype=dtype)
    elif op == "randint":
        if seed is not None:
            np.random.seed(seed)
        if dtype == "int64":
            a = np.random.randint(1, N, N)
        elif dtype == "float64":
            a = np.random.random(N) + 0.5
        elif dtype == "uint64":
            a = np.random.randint(1, N, N, "uint64")
    return a


def time_ak_array_create(N_per_locale, trials, dtype, random, seed):
    print(">>> arkouda {} array creation".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numNodes"]
    print("numNodes = {}, N = {:,}".format(cfg["numNodes"], N))

    timings = {op: [] for op in OPS}
    for i in range(trials):
        for op in timings.keys():
            start = time.time()
            a = create_ak_array(N, op, dtype, seed)
            end = time.time()
            timings[op].append(end - start)
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = (a.size * a.itemsize) / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2**30))


def time_np_array_create(N, trials, dtype, random, seed):
    print(">>> numpy {} array creation".format(dtype))
    print("N = {:,}".format(N))

    timings = {op: [] for op in OPS}
    for i in range(trials):
        for op in timings.keys():
            start = time.time()
            a = create_np_array(N, op, dtype, seed)
            end = time.time()
            timings[op].append(end - start)
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = (a.size * a.itemsize) / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2**30))


def check_correctness(dtype, random, seed):
    N = 10**4

    for op in OPS:
        npa = create_np_array(N, op, dtype, seed)
        aka = create_ak_array(N, op, dtype, seed)
        if op != "randint":
            assert np.allclose(npa, aka.to_ndarray())


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of array creation.")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**8, help="Problem size: length of array")
    parser.add_argument(
        "-t", "--trials", type=int, default=6, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
    parser.add_argument(
        "-r",
        "--randomize",
        default=False,
        action="store_true",
        help="Fill array with random values instead of range (unused)",
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
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="Value to initialize random number generator"
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
            check_correctness(dtype, args.randomize, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_array_create(args.size, args.trials, args.dtype, args.randomize, args.seed)
    if args.numpy:
        time_np_array_create(args.size, args.trials, args.dtype, args.randomize, args.seed)
    sys.exit(0)
