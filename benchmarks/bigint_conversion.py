import argparse
import time

import arkouda as ak


def time_bigint_conversion(N_per_locale, trials, seed, max_bits):
    print(">>> arkouda uint arrays from bigint array")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numNodes"]
    print("numLocales = {}, numNodes {}, N = {:,}".format(cfg["numLocales"], cfg["numNodes"], N))

    a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    b = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)

    # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
    # if max_bits in [0, 64] then they're essentially 1 uint64 array
    tot_bytes = N * 8 if max_bits != -1 and max_bits <= 64 else N * 8 * 2

    convert_to_bigint_times = []
    for i in range(trials):
        start = time.time()
        ba = ak.bigint_from_uint_arrays([a, b], max_bits=max_bits)
        end = time.time()
        convert_to_bigint_times.append(end - start)
    avg_conversion = sum(convert_to_bigint_times) / trials

    print("bigint_from_uint_arrays Average time = {:.4f} sec".format(avg_conversion))
    print(
        "bigint_from_uint_arrays Average rate = {:.4f} GiB/sec".format(
            tot_bytes / 2**30 / avg_conversion
        )
    )
    print()
    print(">>> arkouda bigint array to uint arrays")

    convert_from_bigint_times = []
    for i in range(trials):
        start = time.time()
        u_arrays = ba.bigint_to_uint_arrays()
        end = time.time()
        convert_from_bigint_times.append(end - start)
    avg_conversion = sum(convert_from_bigint_times) / trials

    print("bigint_to_uint_arrays Average time = {:.4f} sec".format(avg_conversion))
    print(
        "bigint_to_uint_arrays Average rate = {:.4f} GiB/sec".format(tot_bytes / 2**30 / avg_conversion)
    )
    if max_bits == -1 or max_bits > 128:
        assert ak.all(a == u_arrays[0])
        assert ak.all(b == u_arrays[1])
    elif max_bits <= 64:
        assert ak.all(b % (2**max_bits - 1) == u_arrays[0])
    else:
        max_bits -= 64
        assert ak.all(a & (2**max_bits - 1) == u_arrays[0])
        assert ak.all(b == u_arrays[1])


def check_correctness(seed, max_bits):
    N = 10**4

    a = ak.randint(0, N, N, dtype=ak.uint64, seed=seed)
    b = ak.randint(0, N, N, dtype=ak.uint64, seed=seed)
    u_arrays = ak.bigint_from_uint_arrays([a, b], max_bits=max_bits).bigint_to_uint_arrays()

    if max_bits == -1 or max_bits >= 128:
        assert ak.all(a == u_arrays[0])
        assert ak.all(b == u_arrays[1])
    elif max_bits <= 64:
        assert ak.all(b % (2**max_bits) == u_arrays[0])
    else:
        max_bits -= 64
        assert ak.all(a % (2**max_bits) == u_arrays[0])
        assert ak.all(b == u_arrays[1])


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of converting between bigint and uint arrays."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**8, help="Problem size: length of array")
    parser.add_argument(
        "--max-bits",
        type=int,
        default=-1,
        help="Maximum number of bits, so values > 2**max_bits will wraparound. "
        "-1 is interpreted as no maximum",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=6,
        help="Number of times to run the benchmark",
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
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        check_correctness(args.seed, args.max_bits)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_bigint_conversion(args.size, args.trials, args.seed, args.max_bits)
    sys.exit(0)
