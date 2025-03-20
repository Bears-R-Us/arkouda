#!/usr/bin/env python3

import argparse
import time

import arkouda as ak


def time_substring_search(N_per_locale, trials, seed):
    print(">>> arkouda substring search")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    start = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)
    end = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)

    # each string in test_substring contains '1 string 1' with random strings before and after
    test_substring = start.stick(end, delimiter="1 string 1")
    nbytes = test_substring.nbytes * test_substring.entry.itemsize

    non_regex_times = []
    regex_literal_times = []
    regex_pattern_times = []
    for i in range(trials):
        start = time.time()
        non_regex = test_substring.contains("1 string 1")
        end = time.time()
        non_regex_times.append(end - start)

        start = time.time()
        regex_literal = test_substring.contains("1 string 1", regex=True)
        end = time.time()
        regex_literal_times.append(end - start)

        start = time.time()
        regex_pattern = test_substring.contains("\\d string \\d", regex=True)
        end = time.time()
        regex_pattern_times.append(end - start)

    avg_non_regex = sum(non_regex_times) / trials
    avg_regex_literal = sum(regex_literal_times) / trials
    avg_regex_pattern = sum(regex_pattern_times) / trials

    assert non_regex.all()
    assert regex_literal.all()
    assert regex_pattern.all()

    print("non-regex with literal substring Average time = {:.4f} sec".format(avg_non_regex))
    print("regex with literal substring Average time = {:.4f} sec".format(avg_regex_literal))
    print("regex with pattern Average time = {:.4f} sec".format(avg_regex_pattern))

    print(
        "non-regex with literal substring Average rate = {:.4f} GiB/sec".format(
            nbytes / 2**30 / avg_non_regex
        )
    )
    print(
        "regex with literal substring Average rate = {:.4f} GiB/sec".format(
            nbytes / 2**30 / avg_regex_literal
        )
    )
    print("regex with pattern Average rate = {:.4f} GiB/sec".format(nbytes / 2**30 / avg_regex_pattern))


def check_correctness(seed):
    N = 10**4

    start = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)
    end = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)

    # each string in test_substring contains '1 string 1' with random strings before and after
    test_substring = start.stick(end, delimiter="1 string 1")

    assert test_substring.contains("1 string 1").all()
    assert test_substring.contains("1 string 1", regex=True).all()
    assert test_substring.contains("\\d string \\d", regex=True).all()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of regex and non-regex substring searches."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**8, help="Problem size: Number of Strings to search"
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
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="Value to initialize random number generator"
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        check_correctness(args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_substring_search(args.size, args.trials, args.seed)
    sys.exit(0)
