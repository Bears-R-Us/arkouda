#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak


def time_ak_bitwise_binops(N_per_locale, trials, max_bits, seed):
    print(">>> arkouda bigint bitwise binops")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numNodes"]
    print("numNodes = {}, N = {:,}".format(cfg["numNodes"], N))
    a1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    a2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    a = ak.bigint_from_uint_arrays([a1, a2], max_bits=max_bits)
    b1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    b2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    b = ak.bigint_from_uint_arrays([b1, b2], max_bits=max_bits)

    # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
    # if max_bits in [0, 64] then they're essentially 1 uint64 array
    tot_bytes = N * 8 if max_bits != -1 and max_bits <= 64 else N * 8 * 2

    and_timings = []
    or_timings = []
    shift_timings = []
    for i in range(trials):
        start = time.time()
        c = a & b
        end = time.time()
        and_timings.append(end - start)

        start = time.time()
        c = a | b
        end = time.time()
        or_timings.append(end - start)

        start = time.time()
        c = a >> 10  # noqa: F841
        end = time.time()
        shift_timings.append(end - start)

    andtavg = sum(and_timings) / trials
    ortavg = sum(or_timings) / trials
    shifttavg = sum(shift_timings) / trials

    print("Average bigint AND time = {:.4f} sec".format(andtavg))
    bytes_per_sec = (tot_bytes * 2) / andtavg
    print("Average bigint AND rate = {:.2f} GiB/sec".format(bytes_per_sec / 2**30))
    print()

    print("Average bigint OR time = {:.4f} sec".format(ortavg))
    bytes_per_sec = (tot_bytes * 2) / ortavg
    print("Average bigint OR rate = {:.2f} GiB/sec".format(bytes_per_sec / 2**30))
    print()

    print("Average bigint SHIFT time = {:.4f} sec".format(shifttavg))
    bytes_per_sec = tot_bytes / shifttavg
    print("Average bigint SHIFT rate = {:.2f} GiB/sec".format(bytes_per_sec / 2**30))


def check_correctness(max_bits, seed):
    N = 10**4
    if seed is not None:
        np.random.seed(seed)
    np_a, np_b = np.random.randint(0, 2**32, N), np.random.randint(0, 2**32, N)
    ak_a = ak.array(np_a, dtype=ak.bigint, max_bits=max_bits)
    ak_b = ak.array(np_b, dtype=ak.bigint, max_bits=max_bits)
    np_arrays = [np_a & np_b, np_a | np_b, np_a >> 10]
    ak_arrays = [ak_a & ak_b, ak_a | ak_b, ak_a >> 10]

    for npc, akc in zip(np_arrays, ak_arrays):
        np_ans = (npc % (2**max_bits)).astype(np.uint) if max_bits != -1 else npc
        ak_ans = akc.to_ndarray().astype(np.uint)
        assert np.all(np_ans == ak_ans)


def create_parser():
    parser = argparse.ArgumentParser(description="Run the bigint bitwise binops benchmarks")
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
    ak.verbose = False
    ak.connect(server=args.hostname, port=args.port)

    if args.correctness_only:
        check_correctness(args.max_bits, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_bitwise_binops(args.size, args.trials, args.max_bits, args.seed)

    sys.exit(0)
