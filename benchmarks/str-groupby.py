#!/usr/bin/env python3

import argparse

from groupby import check_correctness, time_ak_groupby

import arkouda as ak

TYPES = ("str", "mixed")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure performance of grouping arrays of random values."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=10**8,
        help="Problem size: total length of all arrays to group",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=1,
        help="Number of times to run the benchmark",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="str",
        help="Dtype of array ({})".format(", ".join(TYPES)),
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
    time_ak_groupby(args.size, args.trials, args.dtype, args.seed)
    sys.exit(0)
