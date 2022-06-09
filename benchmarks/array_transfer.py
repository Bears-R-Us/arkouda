#!/usr/bin/env python3

import argparse
import gc
import time

import numpy as np

import arkouda as ak

TYPES = ("int64", "float64")


def time_ak_array_transfer(N, trials, dtype, seed):
    print(">>> arkouda {} array creation".format(dtype))
    cfg = ak.get_config()
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    a = ak.randint(0, 2**32, N, dtype=dtype, seed=seed)
    nb = a.size * a.itemsize
    ak.client.maxTransferBytes = nb

    to_ndarray_times = []
    to_pdarray_times = []
    for i in range(trials):
        start = time.time()
        npa = a.to_ndarray()
        end = time.time()
        to_ndarray_times.append(end - start)
        start = time.time()
        aka = ak.array(npa)
        end = time.time()
        to_pdarray_times.append(end - start)
        gc.collect()
    avgnd = sum(to_ndarray_times) / trials
    avgpd = sum(to_pdarray_times) / trials

    print("to_ndarray Average time = {:.4f} sec".format(avgnd))
    print("ak.array Average time = {:.4f} sec".format(avgpd))

    print("to_ndarray Average rate = {:.4f} GiB/sec".format(nb / 2**30 / avgnd))
    print("ak.array Average rate = {:.4f} GiB/sec".format(nb / 2**30 / avgpd))


def check_correctness(dtype, seed):
    N = 10**4

    if seed is not None:
        np.random.seed(seed)
    if dtype == "int64":
        a = np.random.randint(1, N, N)
    elif dtype == "float64":
        a = np.random.random(N) + 0.5

    aka = ak.array(a)
    npa = aka.to_ndarray()
    assert np.allclose(a, npa)


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of transferring arrays.")
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
            check_correctness(dtype, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_array_transfer(args.size, args.trials, args.dtype, args.seed)
    sys.exit(0)
