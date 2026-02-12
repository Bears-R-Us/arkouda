#!/usr/bin/env python3

import argparse
import gc
import time

import numpy as np

import arkouda as ak


TYPES = ("int64", "float64")


def time_ak_array_transfer(N, trials, dtype, seed, max_bits=-1):
    print(">>> arkouda {} array transfer".format(dtype))
    cfg = ak.get_config()
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    if dtype == ak.bigint.name:
        u1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
        u2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
        a = ak.bigint_from_uint_arrays([u1, u2], max_bits=max_bits)
        # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
        # if max_bits in [0, 64] then they're essentially 1 uint64 array
        nb = a.size * 8 if max_bits != -1 and max_bits <= 64 else a.size * 8 * 2
        ak.core.client.maxTransferBytes = nb
    else:
        a = ak.randint(0, 2**32, N, dtype=dtype, seed=seed)
        nb = a.size * a.itemsize
        ak.core.client.maxTransferBytes = nb

    to_ndarray_times = []
    to_pdarray_times = []
    for i in range(trials):
        start = time.time()
        npa = a.to_ndarray()
        end = time.time()
        to_ndarray_times.append(end - start)
        if dtype == ak.bigint.name:
            start = time.time()
            aka = ak.array(npa, max_bits=max_bits, dtype=dtype, unsafe=True, num_bits=128, any_neg=False)
            end = time.time()
        else:
            start = time.time()
            aka = ak.array(npa, max_bits=max_bits, dtype=dtype)
            end = time.time()
        to_pdarray_times.append(end - start)
        gc.collect()
    avgnd = sum(to_ndarray_times) / trials
    avgpd = sum(to_pdarray_times) / trials

    print("to_ndarray Average time = {:.4f} sec".format(avgnd))
    print("ak.array Average time = {:.4f} sec".format(avgpd))

    print("to_ndarray Average rate = {:.4f} GiB/sec".format(nb / 2**30 / avgnd))
    print("ak.array Average rate = {:.4f} GiB/sec".format(nb / 2**30 / avgpd))


def check_correctness(dtype, seed, max_bits=-1):
    N = 10**4
    if seed is not None:
        np.random.seed(seed)
    if dtype == ak.bigint.name:
        u1 = np.random.randint(1, N, N)
        u2 = np.random.randint(1, N, N)
        a = (u1.astype("O") << 64) + u2.astype("O")
        aka = ak.array(a, max_bits=max_bits)
        npa = aka.to_ndarray()
        if max_bits == -1 or max_bits >= 128:
            assert np.all(a == npa)
        elif max_bits <= 64:
            npa2 = (npa % 2**64).astype(np.uint)
            assert np.all(u2 % (2**max_bits) == npa2)
        else:
            max_bits -= 64
            npa1, npa2 = (npa >> 64).astype(np.uint), (npa % 2**64).astype(np.uint)
            assert np.all(u1 % (2**max_bits) == npa1)
            assert np.all(u2 == npa2)
    else:
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
