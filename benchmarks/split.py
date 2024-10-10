#!/usr/bin/env python3

import argparse
import time

import arkouda as ak


def time_split(N_per_locale, trials):
    print(">>> arkouda split")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    thirds = [ak.cast(ak.arange(i, N * 3, 3), "str") for i in range(3)]
    thickrange = thirds[0].stick(thirds[1], delimiter="_").stick(thirds[2], delimiter="_")
    nbytes = thickrange.nbytes * thickrange.entry.itemsize

    non_regex_times = []
    regex_literal_times = []
    regex_pattern_times = []
    for i in range(trials):
        start = time.time()
        non_regex = thickrange.split("_")
        end = time.time()
        non_regex_times.append(end - start)

        start = time.time()
        regex_literal = thickrange.split("_", regex=True)
        end = time.time()
        regex_literal_times.append(end - start)

        start = time.time()
        regex_pattern = thickrange.split("_+", regex=True)
        end = time.time()
        regex_pattern_times.append(end - start)

    avg_non_regex = sum(non_regex_times) / trials
    avg_regex_literal = sum(regex_literal_times) / trials
    avg_regex_pattern = sum(regex_pattern_times) / trials

    answer = ak.cast(ak.arange(N * 3), "str")
    assert (non_regex == answer).all()
    assert (regex_literal == answer).all()
    assert (regex_pattern == answer).all()

    print("non-regex split with literal delimiter Average time = {:.4f} sec".format(avg_non_regex))
    print("regex split with literal delimiter Average time = {:.4f} sec".format(avg_regex_literal))
    print("regex split with pattern delimiter Average time = {:.4f} sec".format(avg_regex_pattern))

    print(
        "non-regex split with literal delimiter Average rate = {:.4f} GiB/sec".format(
            nbytes / 2**30 / avg_non_regex
        )
    )
    print(
        "regex split with literal delimiter Average rate = {:.4f} GiB/sec".format(
            nbytes / 2**30 / avg_regex_literal
        )
    )
    print(
        "regex split with pattern delimiter Average rate = {:.4f} GiB/sec".format(
            nbytes / 2**30 / avg_regex_pattern
        )
    )


def check_correctness():
    N = 10**4

    thirds = [ak.cast(ak.arange(i, N * 3, 3), "str") for i in range(3)]
    thickrange = thirds[0].stick(thirds[1], delimiter="_").stick(thirds[2], delimiter="_")

    answer = ak.cast(ak.arange(N * 3), "str")
    assert (thickrange.split("_") == answer).all()
    assert (thickrange.split("_", regex=True) == answer).all()
    assert (thickrange.split("_+", regex=True) == answer).all()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of regex and non-regex split on Strings."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**5, help="Problem size: Number of Strings to split"
    )
    parser.add_argument(
        "-t", "--trials", type=int, default=1, help="Number of times to run the benchmark"
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

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_split(args.size, args.trials)
    sys.exit(0)
