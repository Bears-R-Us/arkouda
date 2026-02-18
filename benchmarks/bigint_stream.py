#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak


def time_ak_stream(N_per_locale, trials, alpha, max_bits, random, seed):
    print(">>> arkouda bigint stream")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    # default tot_bytes to ones case
    tot_bytes = N * 8 * 3
    if random or seed is not None:
        a1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
        a2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
        a = ak.bigint_from_uint_arrays([a1, a2], max_bits=max_bits)
        b1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
        b2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
        b = ak.bigint_from_uint_arrays([b1, b2], max_bits=max_bits)
        # update tot_bytes to account for using 2 uint64
        tot_bytes *= 2
    else:
        a = ak.bigint_from_uint_arrays([ak.ones(N, dtype=ak.uint64)], max_bits=max_bits)
        b = ak.bigint_from_uint_arrays([ak.ones(N, dtype=ak.uint64)], max_bits=max_bits)

    timings = []
    for i in range(trials):
        start = time.time()
        c = a + b * alpha  # noqa: F841
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average bigint stream time = {:.4f} sec".format(tavg))
    bytes_per_sec = tot_bytes / tavg
    print("Average bigint stream rate = {:.2f} GiB/sec".format(bytes_per_sec / 2**30))


def check_correctness(alpha, max_bits, random, seed):
    N = 10**4
    if seed is not None:
        np.random.seed(seed)
    if random or seed is not None:
        a = np.random.randint(0, 2**32, N)
        b = np.random.randint(0, 2**32, N)
    else:
        a = np.ones(N, dtype=np.uint)
        b = np.ones(N, dtype=np.uint)
    npc = a + b * alpha
    akc = (
        ak.array(a, dtype=ak.bigint, max_bits=max_bits)
        + ak.array(b, dtype=ak.bigint, max_bits=max_bits) * alpha
    )
    np_ans = (npc % (2**max_bits)).astype(np.uint) if max_bits != -1 else npc
    ak_ans = akc.to_ndarray().astype(np.uint)
    assert np.all(np_ans == ak_ans)


def create_parser():
    parser = argparse.ArgumentParser(description="Run the bigint stream benchmark: C = A + alpha*B")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=10**8,
        help="Problem size: length of arrays A and B",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=6,
        help="Number of times to run the benchmark",
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=-1,
        help="Maximum number of bits, so values > 2**max_bits will wraparound. "
        "-1 is interpreted as no maximum",
    )
    parser.add_argument(
        "-r",
        "--randomize",
        default=False,
        action="store_true",
        help="Fill arrays with random values instead of ones",
    )
    parser.add_argument("-a", "--alpha", default=1, help="Scalar multiple")
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help="Value to initialize random number generator",
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    args.alpha = int(args.alpha)
    ak.verbose = False
    ak.connect(server=args.hostname, port=args.port)

    if args.correctness_only:
        check_correctness(args.alpha, args.max_bits, args.randomize, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_stream(args.size, args.trials, args.alpha, args.max_bits, args.randomize, args.seed)

    sys.exit(0)
