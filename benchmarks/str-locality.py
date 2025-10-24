#!/usr/bin/env python3

import argparse
import time

import arkouda as ak


def generate_data(N, seed):
    prefix = ak.random_strings_uniform(minlen=1, maxlen=16, size=N, seed=seed, characters="numeric")
    if seed is not None:
        seed += 1
    suffix = ak.random_strings_uniform(minlen=1, maxlen=16, size=N, seed=seed, characters="numeric")
    random_strings = prefix.stick(suffix, delimiter=".")
    perm = ak.argsort(random_strings.get_lengths())
    sorted_strings = random_strings[perm]
    return random_strings, sorted_strings, perm


def time_all_ops(N_per_locale, trials, seed, correctnessOnly):
    if correctnessOnly:
        N = 10**4
    else:
        print(">>> arkouda string locality tests")
        nl = ak.get_config()["numNodes"]
        N = nl * N_per_locale
        print("numNodes = {}, N = {:,}".format(nl, N))
    random_strings, sorted_strings, perm = generate_data(N, seed)
    nbytes = random_strings.nbytes

    def time_op(op, name):
        random_times = []
        sorted_times = []
        for i in range(trials):
            start = time.time()
            random_ans = op(random_strings)
            end = time.time()
            random_times.append(end - start)
            start = time.time()
            sorted_ans = op(sorted_strings)
            end = time.time()
            sorted_times.append(end - start)
        if not correctnessOnly:
            avg = sum(random_times) / trials
            rate = nbytes / 2**30 / avg
            print("{} good locality Average time = {:.4f} sec".format(name, avg))
            print("{} good locality Average rate = {:.4f} GiB/sec".format(name, rate))
            avg = sum(sorted_times) / trials
            rate = nbytes / 2**30 / avg
            print("{} poor locality Average time = {:.4f} sec".format(name, avg))
            print("{} poor locality Average rate = {:.4f} GiB/sec".format(name, rate))
        return random_ans, sorted_ans

    # Hash
    op = lambda x: x.hash()
    ans1, ans2 = time_op(op, "Hashing")
    for i in range(2):
        assert (ans1[i][perm] == ans2[i]).all()

    # Substring search
    op = lambda x: x.contains(r"\d{3,5}\.\d{5,8}", regex=True)
    ans1, ans2 = time_op(op, "Regex searching")
    assert (ans1[perm] == ans2).all()

    # Cast
    op = lambda x: ak.cast(x, ak.float64)
    ans1, ans2 = time_op(op, "Casting")
    assert (ans1[perm] == ans2).all()

    # Scalar compare
    op = lambda x: (x == "5.5")
    ans1, ans2 = time_op(op, "Comparing to scalar")
    assert (ans1[perm] == ans2).all()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure the performance of various string operations on "
        "strings with good locality (random) and poor locality (sorted)."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**7, help="Problem size: Number of Strings to search"
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

    if not args.correctness_only:
        print("array size = {:,}".format(args.size))
        print("number of trials = ", args.trials)
    time_all_ops(args.size, args.trials, args.seed, args.correctness_only)
    sys.exit(0)
