#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak
import arkouda.core.client


def time_ak_noop(trial_time):
    print(">>> arkouda noop")
    start = time.time()
    trials = 0
    while time.time() - start < trial_time:
        trials += 1
        arkouda.core.client._no_op()
    end = time.time()

    timing = end - start
    tavg = timing / trials

    print("Average time = {:.6f} sec".format(tavg))
    print("Average rate = {:.2f} ops/sec".format(trials / timing))


def time_np_noop(trial_time):
    print(">>> numpy noop")
    start = time.time()
    trials = 0
    while time.time() - start < trial_time:
        trials += 1
        np.get_include()  # closet I could find to a noop
    end = time.time()

    timing = end - start
    tavg = timing / trials

    print("Average time = {:.6f} sec".format(tavg))
    print("Average rate = {:.2f} ops/sec".format(trials / timing))


def check_correctness():
    assert arkouda.core.client._no_op() == "noop"


def create_parser():
    parser = argparse.ArgumentParser(description="Run a noop benchmark")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=1, help="Problem size (unused)")
    parser.add_argument(
        "-t",
        "--trials",
        "--trials-time",
        type=int,
        default=1,
        help="Amount of time to run the benchmark",
    )
    parser.add_argument("-d", "--dtype", default="int64", help="Dtype of arrays (unused)")
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

    print("number of trials = ", args.trials)
    time_ak_noop(args.trials)
    if args.numpy:
        time_np_noop(args.size)
    sys.exit(0)
