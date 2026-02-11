#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak


OPS = ("cumsum", "cumprod")
TYPES = ("int64", "float64")


def time_ak_scan(N_per_locale, trials, dtype, random, seed):
    print(">>> arkouda {} scan".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numNodes"]
    print("numLocales = {}, numNodes {}, N = {:,}".format(cfg["numLocales"], cfg["numNodes"], N))
    if random or args.seed is not None:
        if dtype == "int64":
            a = ak.randint(1, N, N, seed=seed)
        elif dtype == "float64":
            a = ak.uniform(N, seed=seed) + 0.5
    else:
        a = ak.arange(1, N, 1)
        if dtype == "float64":
            a = 1.0 * a

    timings = {op: [] for op in OPS}
    final_values = {}
    for i in range(trials):
        for op in timings.keys():
            fxn = getattr(ak, op)
            start = time.time()
            r = fxn(a)
            end = time.time()
            timings[op].append(end - start)
            final_values[op] = r[r.size - 1]
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("{}, final value = {}".format(op, final_values[op]))
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = (a.size * a.itemsize * 2) / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2**30))


def time_np_scan(N, trials, dtype, random, seed):
    print(">>> numpy {} scan".format(dtype))
    print("N = {:,}".format(N))
    if seed is not None:
        np.random.seed(seed)
    if random or seed is not None:
        if dtype == "int64":
            a = np.random.randint(1, N, N)
        elif dtype == "float64":
            a = np.random.random(N) + 0.5
    else:
        a = np.arange(1, N, 1, dtype=dtype)

    timings = {op: [] for op in OPS}
    final_values = {}
    for i in range(trials):
        for op in timings.keys():
            fxn = getattr(np, op)
            start = time.time()
            r = fxn(a)
            end = time.time()
            timings[op].append(end - start)
            final_values[op] = r[r.size - 1]
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("{}, final value = {}".format(op, final_values[op]))
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = (a.size * a.itemsize * 2) / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2**30))


def check_correctness(dtype, random, seed):
    N = 10**4
    if seed is not None:
        np.random.seed(seed)
    if random or seed is not None:
        if dtype == "int64":
            a = np.random.randint(1, N, N)
        elif dtype == "float64":
            a = np.random.random(N) + 0.5
    else:
        if dtype == "int64":
            a = np.arange(1, N, 1, dtype=dtype)
        elif dtype == "float64":
            a = np.arange(1, 1 + 1 / N, (1 / N) / N, dtype=dtype)

    for op in OPS:
        npa = a
        aka = ak.array(a)
        fxn = getattr(np, op)
        npr = fxn(npa)
        fxn = getattr(ak, op)
        akr = fxn(aka).to_ndarray()
        # Because np.prod() returns an integer type with no infinity, it returns
        # zero on overflow.
        # By contrast, ak.prod() returns float64, so it returns inf on overflow
        if dtype == "int64" and op == "prod":
            akr[akr == np.inf] = 0
        assert np.allclose(npr, akr)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of scans (cumulative reductions) over arrays."
    )
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
        help="Fill array with random values instead of range",
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
    time_ak_scan(args.size, args.trials, args.dtype, args.randomize, args.seed)
    if args.numpy:
        time_np_scan(args.size, args.trials, args.dtype, args.randomize, args.seed)
    sys.exit(0)
